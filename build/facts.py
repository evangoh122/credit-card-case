"""Fact table generation.

A sequential, per-account statement-cycle simulator:
  - transactions accrue inside each statement cycle and are capped by available credit
  - a statement is cut every month on the account's statement day
  - a payment is scheduled against each statement according to payment behaviour
  - interest, annual fee, foreign transaction fee and late fee are charged to the statement
  - delinquency is measured on unpaid statements at each statement date

This keeps every fact table internally consistent (transactions -> statements ->
payments -> balances), which a pure independent-sampling approach cannot do.
"""
from __future__ import annotations

import datetime as dt
import uuid

import numpy as np
import pandas as pd

from . import config as C
from . import txgen
from .utils import (Buf, daterange, hash_id, json_dumps, rng_for, synthetic_ip)

# --------------------------------------------------------------------------
# Static lookups
# --------------------------------------------------------------------------
MCC_CODE = [m[0] for m in C.MCC_CATALOG]
MCC_DESC = {m[0]: m[1] for m in C.MCC_CATALOG}
MCC_CAT = {m[0]: m[2] for m in C.MCC_CATALOG}
MCC_MEDIAN = {m[0]: m[3] for m in C.MCC_CATALOG}
MCC_SIGMA = {m[0]: m[4] for m in C.MCC_CATALOG}
MCC_ONLINE = {m[0]: m[5] for m in C.MCC_CATALOG}
MCC_WKND = {m[0]: m[6] for m in C.MCC_CATALOG}
MCC_FOREIGN = {m[0]: m[8] for m in C.MCC_CATALOG}
_MCC_W = np.array([m[7] for m in C.MCC_CATALOG], float)
_MCC_W = _MCC_W / _MCC_W.sum()


# --------------------------------------------------------------------------
# Amount calibration: 1:1 PHP -> USD (no FX division)
# --------------------------------------------------------------------------
def _calibrate_amounts():
    """Find a monotone transform a * x^b so the MCC-weighted mixture of ticket
    amounts matches a US card-spend median and p95.

    Higher-value categories (travel, electronics) stay above lower-value ones
    (fast food, digital), but the book as a whole lands on US norms.
    """
    rs = np.random.default_rng(90210)
    n = 120_000
    idx = rs.choice(len(MCC_CODE), size=n, p=_MCC_W)
    med = np.array([MCC_MEDIAN[MCC_CODE[i]] for i in idx], float)
    sig = np.array([MCC_SIGMA[MCC_CODE[i]] for i in idx], float) * 0.75
    vals = np.clip(rs.lognormal(np.log(med), sig), 1.0, 20_000.0)
    m0 = float(np.median(vals))
    p95 = float(np.quantile(vals, 0.95))
    b = float(np.log(C.US_TICKET_P95 / C.US_TICKET_MEDIAN) / np.log(p95 / m0))
    a = float(C.US_TICKET_MEDIAN / (m0 ** b))
    return a, b


AMOUNT_A, AMOUNT_B = C.AMOUNT_A, C.AMOUNT_B
AMOUNT_FLOOR = 1.0

SEGMENT_PARAMS = {s["segment"]: s for s in C.CUSTOMER_SEGMENTS}

DOW_MULT = np.array([0.94, 0.94, 0.96, 0.99, 1.12, 1.20, 1.05])  # Mon..Sun
MONTH_MULT = {1: 0.91, 2: 0.95, 3: 1.00, 4: 1.00, 5: 1.02, 6: 1.05,
              7: 1.04, 8: 1.02, 9: 0.99, 10: 1.03, 11: 1.16, 12: 1.27}

HOUR_PROFILE = {}


def _hour_profile(group: str) -> np.ndarray:
    if group in HOUR_PROFILE:
        return HOUR_PROFILE[group]
    w = np.ones(24, float)
    if group == "Dining":
        w[:7] = 0.02; w[11:14] = 1.7; w[17:21] = 1.9; w[21:] = 1.1
    elif group == "Grocery":
        w[:7] = 0.05; w[9:20] = 1.4; w[20:] = 0.5
    elif group == "Fuel":
        w[:5] = 0.05; w[6:20] = 1.3; w[20:] = 0.4
    elif group in ("Travel", "Transport"):
        w[:5] = 0.25; w[5:23] = 1.1
    elif group in ("Digital", "E-Commerce", "Telecom"):
        w[:7] = 0.15; w[18:24] = 1.7; w[12:18] = 0.8
    elif group in ("Entertainment",):
        w[:9] = 0.05; w[17:24] = 1.8
    elif group == "Utilities":
        w[:6] = 0.05; w[8:21] = 1.0
    else:
        w[:7] = 0.05; w[8:21] = 1.2; w[21:] = 0.6
    w = w / w.sum()
    HOUR_PROFILE[group] = w
    return w


def _merchant_index(merch: pd.DataFrame):
    lookup = {}
    for code, grp in merch.groupby("mcc"):
        idx = grp.index.to_numpy()
        p = grp["popularity_weight"].to_numpy(float)
        p = p / p.sum()
        lookup[code] = (idx, p)
    return lookup


def _channel_index(ch: pd.DataFrame):
    online = ch[ch.channel_group.isin(["ONLINE", "MOBILE_APP", "PHONE_ORDER"])].index.to_numpy()
    instore = ch[ch.channel_group == "IN_STORE"].index.to_numpy()
    pos_w = ch.loc[instore, "channel_name"].map(
        {"In Store - Chip": 0.52, "In Store - Contactless": 0.42, "In Store - Swipe": 0.06}).to_numpy(float)
    pos_w = pos_w / pos_w.sum()
    on_w = np.array([0.76, 0.20, 0.04])
    return online, on_w, instore, pos_w


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------
def simulate(dims: dict) -> dict:
    rng = rng_for("facts")
    accounts = dims["dim_account"].copy()
    customers = dims["dim_customer"].set_index("customer_id")
    merch = dims["dim_merchant"]
    channels = dims["dim_channel"]
    ch_lookup = channels.set_index("channel_id")

    merch_lookup = _merchant_index(merch)
    online_idx, on_w, instore_idx, pos_w = _channel_index(channels)
    ctx = txgen.TxContext(merch, channels)
    merch_ids = merch["merchant_id"].to_numpy()
    merch_geo = merch["geo_id"].to_numpy()
    merch_intl = merch["is_international"].to_numpy()
    merch_cat = merch["category"].to_numpy()

    tx_parts: list[pd.DataFrame] = []
    tx_chunks: list[pd.DataFrame] = []

    def flush_tx(force: bool = False):
        if len(tx_parts) >= 4000 or (force and tx_parts):
            tx_chunks.append(pd.concat(tx_parts, ignore_index=True))
            tx_parts.clear()
            if len(tx_chunks) > 4:
                tx_chunks[:] = [pd.concat(tx_chunks, ignore_index=True)]

    stmt = Buf()
    pay = Buf()
    alloc = Buf()
    plan = Buf()
    sched = Buf()
    lim = Buf()
    delinq = Buf()

    txn_no = 1
    stmt_no = 1
    pay_no = 1
    alloc_no = 1
    plan_no = 1
    sched_no = 1
    lim_no = 1
    snap_no = 1

    acct_state = {}

    for _, a in accounts.iterrows():
        acc_id = a.account_id
        cust = customers.loc[a.customer_id]
        open_date = a.open_date if isinstance(a.open_date, dt.date) else a.open_date.date()
        planned_close = a.planned_close_date
        if planned_close is not None and not isinstance(planned_close, dt.date):
            planned_close = pd.Timestamp(planned_close).date()
        eff_end = min(planned_close or C.PERIOD_END, C.PERIOD_END)
        if eff_end < C.PERIOD_START:
            continue

        limit = float(a.credit_limit)
        apr = float(a.apr_purchase)
        statement_day = int(a.statement_day)

        # behaviour segment drives transaction frequency
        seg_cfg = SEGMENT_PARAMS.get(str(cust.customer_segment), SEGMENT_PARAMS["CORE_MAINSTREAM"])
        base_lambda = seg_cfg["median_txn"] * float(
            np.clip(rng.lognormal(0.0, seg_cfg["sigma"]), 0.15, 4.5))
        base_lambda = float(np.clip(base_lambda, 0.4, 260.0))

        # opening balance for accounts that pre-date the fact window
        if open_date < C.PERIOD_START:
            util = float(np.clip(rng.beta(2.1, 3.4), 0.02, 0.72))
            if rng.random() < 0.55:          # transactors carry little
                util *= 0.35
            opening = round(limit * util, 2)
        else:
            opening = 0.0

        # card pool
        cards = dims["dim_card"]
        card_ids = cards.loc[cards.account_id == acc_id, "card_id"].tolist() or [None]

        # persistent payment behaviour
        p_pif = float(np.clip(rng.normal(0.60, 0.10), 0.30, 0.82))
        p_min = float(np.clip(rng.normal(0.16, 0.04), 0.06, 0.28))
        p_part = float(np.clip(rng.normal(0.12, 0.04), 0.04, 0.22))
        p_late = float(np.clip(rng.normal(0.07, 0.03), 0.01, 0.15))
        # remainder = missed

        # statement dates inside the window
        stmt_dates = []
        for y, m in _months(open_date, eff_end):
            d = dt.date(y, m, min(statement_day, _last_day(y, m)))
            if open_date <= d <= eff_end and d >= C.PERIOD_START:
                stmt_dates.append(d)
        if not stmt_dates:
            continue

        prev_cut = None
        balance = opening
        open_stmts: list[dict] = []
        pending_pay: list[dict] = []
        acct_txn_total = 0.0
        acct_gross_purchases = 0.0
        acct_pay_total = 0.0
        acct_interest = 0.0
        acct_fees = 0.0
        charged_off = False
        snapshots = []
        acct_arrays: list[dict] = []

        for cut in stmt_dates:
            cycle_start = max(open_date, C.PERIOD_START,
                              (prev_cut + dt.timedelta(days=1)) if prev_cut else open_date)
            cycle_end = min(cut, eff_end)
            if cycle_end < cycle_start:
                prev_cut = cut
                continue

            # ---------- payments that land in this cycle ----------
            cycle_pays = [p for p in pending_pay if cycle_start <= p["date"] <= cycle_end]
            pending_pay = [p for p in pending_pay if p["date"] > cycle_end]
            paid_this_cycle = 0.0
            for p in cycle_pays:
                paid_this_cycle += p["amount"]
                acct_pay_total += p["amount"]
                # a payment reduces every statement already issued
                for s in open_stmts:
                    s["paid_since"] += p["amount"]
                pay.add(
                    payment_id=f"PAY{pay_no:09d}",
                    account_id=acc_id,
                    customer_id=a.customer_id,
                    payment_date=p["date"],
                    date_key=int(p["date"].strftime("%Y%m%d")),
                    amount_usd=round(p["amount"], 2),
                    payment_method=p["method"],
                    payment_channel=p["channel"],
                    payment_status="POSTED",
                    is_autopay=p["autopay"],
                    reference_number=p["ref"],
                    applied_statement_id=p["stmt_id"],
                    posted_ts=pd.Timestamp(p["date"]) + pd.Timedelta(hours=int(rng.integers(6, 21))),
                )
                pay_no += 1

            # ---------- transactions in the cycle ----------
            running = max(0.0, balance - paid_this_cycle)
            # month-to-month overdispersion + occasional spike month
            month_mult = float(rng.lognormal(0.0, C.TXN_MONTH_SIGMA))
            if rng.random() < C.SPIKE_MONTH_PROB:
                month_mult *= float(rng.uniform(*C.SPIKE_MULT_RANGE))
            days = list(daterange(cycle_start, cycle_end))
            out = None
            if days:
                w = np.array([
                    MONTH_MULT[d.month] * DOW_MULT[d.weekday()] * (1.10 if d in _HOLIDAYS else 1.0)
                    for d in days
                ], float)
                w = w / w.sum()
                lam = base_lambda / 30.44
                daily = np.array([lam * MONTH_MULT[d.month] * DOW_MULT[d.weekday()] *
                                  (1.10 if d in _HOLIDAYS else 1.0) for d in days])
                n_draw = min(int(rng.poisson(daily.sum() * month_mult)), C.MAX_TXN_PER_MONTH)
                out, cash_avail = txgen.generate_cycle(
                    ctx, rng, days, w, n_draw, limit, running,
                    (cycle_start - open_date).days)

            if out is not None:
                m = len(out["amount_usd"])
                stmt_id = f"STM{stmt_no:09d}"
                ids = np.array([f"TXN{txn_no + i:09d}" for i in range(m)], dtype=object)
                txn_no += m

                ttype = out["transaction_type"]
                status = out["status"]
                amt = out["amount_usd"]

                appr = status == "APPROVED"
                cyc_sum = float(amt[appr].sum())
                cyc_purch = float(amt[appr & (ttype == "PURCHASE")].sum())
                cyc_refund = float(amt[ttype == "REFUND"].sum())
                cyc_cash = 0.0
                acct_txn_total += cyc_sum
                acct_gross_purchases += cyc_purch

                plan_ids = np.full(m, None, dtype=object)
                cand = np.where(appr & (ttype == "PURCHASE") & (amt >= 150))[0]
                if len(cand):
                    take = cand[rng.random(len(cand)) < 0.05]
                    for j in take:
                        pid = f"PLN{plan_no:08d}"
                        added = _emit_plan(
                            rng, plan, sched, pid, a, cust,
                            dict(transaction_id=ids[j],
                                 date=pd.Timestamp(out["ts"][j]).date(),
                                 amount_usd=float(amt[j])),
                            cycle_end, sched_no)
                        sched_no += added
                        plan_no += 1
                        plan_ids[j] = pid

                ci = out["channel_idx"]
                tsidx = pd.DatetimeIndex(out["ts"])
                card_list = sorted(cards, key=lambda c: c["issue_date"])
                card_for = np.full(m, card_list[0]["card_id"], dtype=object)
                for _c in card_list[1:]:
                    card_for[tsidx >= pd.Timestamp(_c["issue_date"])] = _c["card_id"]
                acct_arrays.append(dict(
                    transaction_id=ids,
                    account_id=np.full(m, acc_id, dtype=object),
                    card_id=np.array(card_ids, dtype=object)[rng.integers(0, len(card_ids), m)],
                    customer_id=np.full(m, a.customer_id, dtype=object),
                    merchant_id=ctx.merch_ids[out["merchant_idx"]],
                    merchant_name=ctx.merch_names[out["merchant_idx"]],
                    mcc=out["mcc"],
                    category=out["category"],
                    channel_id=ctx.ch_id[ci],
                    channel_group=ctx.ch_group[ci],
                    entry_mode=ctx.ch_entry[ci],
                    is_card_present=ctx.ch_present[ci],
                    merchant_geo_id=ctx.merch_geo[out["merchant_idx"]],
                    customer_geo_id=np.full(m, cust.home_geo_id, dtype=object),
                    transaction_date=tsidx.normalize().to_numpy(),
                    date_key=(tsidx.year * 10000 + tsidx.month * 100 + tsidx.day).to_numpy(),
                    transaction_ts=tsidx.to_numpy(),
                    transaction_type=ttype,
                    amount_usd=amt,
                    currency_code=out["currency_code"],
                    original_amount=out["original_amount"],
                    fx_rate=out["fx_rate"],
                    status=status,
                    decline_reason_code=np.where(out["decline_code"] == "", None, out["decline_code"]),
                    decline_reason=np.where(out["decline_code"] == "", None, out["decline_code"]),
                    auth_code=_auth_codes(rng, appr, m),
                    is_international=out["is_international"],
                    is_recurring=ctx.merch_recurring[out["merchant_idx"]],
                    rewards_points=np.zeros(m, dtype=np.int64),
                    installment_plan_id=plan_ids,
                    fraud_score=out["fraud_score"],
                    is_fraud=np.zeros(m, bool),
                    statement_id=np.full(m, stmt_id, dtype=object),
                    source_system=np.full(m, "CARD_PROCESSOR", dtype=object),
                ))

            # ---------- interest & fees ----------
            revolving = balance > paid_this_cycle + 25
            interest = 0.0
            if revolving and running > 25:
                interest = round(max(0.0, running - paid_this_cycle * 0.5) * apr / 12.0, 2)
            fees = 0.0
            if a.initial_credit_limit and open_date.month == cut.month and (cut - open_date).days >= 300:
                fees += float(next((p["annual_fee"] for p in C.PRODUCTS if p["product_id"] == a.product_id), 0.0))
            ftf = float(a.foreign_txn_fee or 0)
            if ftf > 0 and out is not None:
                m_intl = (out["is_international"] & (out["status"] == "APPROVED")
                          & (out["transaction_type"] == "PURCHASE"))
                intl_spend = float(out["amount_usd"][m_intl].sum())
                if intl_spend > 0:
                    fees += round(intl_spend * ftf, 2)
            overdue = [s for s in open_stmts
                       if s["due_date"] < cut and (s["min_due"] - s["paid_since"]) > 1]
            if overdue:
                fees += 35.0 if running > 400 else 29.0

            closing = round(balance + cyc_sum + interest + fees - paid_this_cycle, 2)
            if closing < 0:
                closing = round(closing, 2)   # credit balance is legitimate

            min_due = 0.0
            if closing > 0:
                min_due = round(max(35.0, closing * 0.01 + interest + fees), 2)
                min_due = min(min_due, closing)

            due_date = cut + dt.timedelta(days=25)
            sid = f"STM{stmt_no:09d}"
            stmt.add(
                statement_id=sid,
                account_id=acc_id,
                customer_id=a.customer_id,
                product_id=a.product_id,
                statement_date=cut,
                date_key=int(cut.strftime("%Y%m%d")),
                month_key=cut.strftime("%Y-%m"),
                cycle_start_date=cycle_start,
                cycle_end_date=cycle_end,
                payment_due_date=due_date,
                credit_limit=round(limit, 2),
                opening_balance=round(balance, 2),
                purchases_amount=round(cyc_purch, 2),
                refunds_amount=round(cyc_refund, 2),
                cash_advances_amount=round(cyc_cash, 2),
                payments_amount=round(paid_this_cycle, 2),
                interest_charged=interest,
                fees_charged=round(fees, 2),
                closing_balance=closing,
                minimum_due=min_due,
                available_credit=round(max(0.0, limit - closing), 2),
                utilization_pct=round(min(1.5, closing / limit), 4) if limit else 0.0,
                transaction_count=int(len(out["amount_usd"])) if out is not None else 0,
                paid_amount=0.0,
                payment_status="PENDING",
                is_late=False,
            )
            stmt_no += 1
            acct_gross_purchases += cyc_purch

            open_stmts.append(dict(stmt_id=sid, due_date=due_date, statement_date=cut,
                                   closing=max(0.0, closing), min_due=min_due,
                                   paid_since=0.0))
            acct_interest += interest
            acct_fees += fees

            # ---------- schedule the payment for this statement ----------
            if closing > 0 and due_date <= C.PAYMENT_HORIZON:
                r = rng.random()
                amt = 0.0
                pay_date = due_date - dt.timedelta(days=int(rng.integers(2, 18)))
                if r < p_pif:
                    amt = closing
                elif r < p_pif + p_min:
                    amt = min_due
                elif r < p_pif + p_min + p_part:
                    amt = round(closing * float(rng.uniform(0.20, 0.85)), 2)
                elif r < p_pif + p_min + p_part + p_late:
                    amt = closing
                    pay_date = due_date + dt.timedelta(days=int(rng.integers(3, 21)))
                else:
                    amt = 0.0
                if amt > 0 and pay_date <= C.PAYMENT_HORIZON:
                    method = str(rng.choice(["ACH", "DEBIT_CARD", "CHECK", "WIRE", "INTERNAL_TRANSFER"],
                                            p=[0.56, 0.24, 0.08, 0.02, 0.10]))
                    pending_pay.append(dict(
                        date=pay_date, amount=min(amt, closing), stmt_id=sid,
                        method=method,
                        channel="MOBILE_APP" if method in ("ACH", "INTERNAL_TRANSFER") else "WEB",
                        autopay=bool(method == "ACH" and rng.random() < 0.45),
                        ref=f"PMT{hash_id(str(uuid.UUID(int=int(rng.integers(0, 2**63)))))[:16].upper()}",
                    ))

            # ---------- credit limit management ----------
            if rng.random() < 0.006 and (cut - open_date).days > 240 and not overdue:
                inc = round(limit * float(rng.uniform(0.10, 0.35)) / 100) * 100
                new_limit = min(limit + inc, 50000)
                if new_limit > limit:
                    lim.add(limit_event_id=f"LIM{lim_no:07d}", account_id=acc_id,
                            customer_id=a.customer_id, event_date=cut,
                            date_key=int(cut.strftime("%Y%m%d")), event_type="CREDIT_LIMIT_INCREASE",
                            old_limit=round(limit, 2), new_limit=round(new_limit, 2),
                            change_amount=round(new_limit - limit, 2),
                            change_pct=round((new_limit - limit) / limit, 4),
                            reason=str(rng.choice(["INCOME_VERIFIED", "PAYMENT_HISTORY", "CUSTOMER_REQUEST",
                                                    "PORTFOLIO_REVIEW"])),
                            channel="PERIODIC_REVIEW")
                    lim_no += 1
                    limit = new_limit
            elif rng.random() < 0.004 and overdue and limit > 4500:
                new_limit = max(4500, round(limit * 0.70 / 100) * 100)
                if new_limit < limit:
                    lim.add(limit_event_id=f"LIM{lim_no:07d}", account_id=acc_id,
                            customer_id=a.customer_id, event_date=cut,
                            date_key=int(cut.strftime("%Y%m%d")), event_type="CREDIT_LIMIT_DECREASE",
                            old_limit=round(limit, 2), new_limit=round(new_limit, 2),
                            change_amount=round(new_limit - limit, 2),
                            change_pct=round((new_limit - limit) / limit, 4),
                            reason="RISK_ACTION", channel="RISK_SYSTEM")
                    lim_no += 1
                    limit = new_limit
            elif rng.random() < 0.0018 and overdue:
                new_limit = round(max(300.0, limit * 0.7) / 100) * 100
                lim.add(limit_event_id=f"LIM{lim_no:07d}", account_id=acc_id,
                        customer_id=a.customer_id, event_date=cut,
                        date_key=int(cut.strftime("%Y%m%d")), event_type="CREDIT_LIMIT_DECREASE",
                        old_limit=round(limit, 2), new_limit=round(new_limit, 2),
                        change_amount=round(new_limit - limit, 2),
                        change_pct=round((new_limit - limit) / limit, 4),
                        reason="RISK_ACTION", channel="RISK_SYSTEM")
                lim_no += 1
                limit = new_limit

            # ---------- delinquency snapshot at statement date ----------
            overdue_now = [s for s in open_stmts[:-1]
                           if s["due_date"] < cut and (s["min_due"] - s["paid_since"]) > 1]
            if overdue_now:
                oldest = min(overdue_now, key=lambda s: s["due_date"])
                dpd = (cut - oldest["due_date"]).days
                amt_past = sum(s["min_due"] - s["paid_since"] for s in overdue_now)
                n_unpaid = len(overdue_now)
            else:
                dpd = 0
                amt_past = 0.0
                n_unpaid = 0
            delinq.add(
                snapshot_id=f"SNP{snap_no:09d}",
                account_id=acc_id,
                customer_id=a.customer_id,
                snapshot_date=cut,
                date_key=int(cut.strftime("%Y%m%d")),
                month_key=cut.strftime("%Y-%m"),
                days_past_due=dpd,
                delinquency_bucket=_bucket(dpd),
                balance=closing,
                amount_past_due=round(amt_past, 2),
                minimum_due=min_due,
                unpaid_statement_count=n_unpaid,
                credit_limit=round(limit, 2),
            )
            snap_no += 1
            snapshots.append(dpd)

            if dpd > 180:
                charged_off = True

            balance = closing
            prev_cut = cut
            if charged_off:
                break

        acct_state[acc_id] = dict(
            current_balance=round(balance, 2),
            credit_limit=round(limit, 2),
            lifetime_purchases=round(acct_gross_purchases, 2),
            lifetime_payments=round(acct_pay_total, 2),
            lifetime_interest=round(acct_interest, 2),
            lifetime_fees=round(acct_fees, 2),
            days_past_due=int(snapshots[-1]) if snapshots else 0,
            charged_off=charged_off,
            last_statement_date=prev_cut,
            open_statement_remaining=round(sum(max(0.0, s["closing"] - s["paid_since"])
                                               for s in open_stmts), 2),
        )

        if acct_arrays:
            merged = {k: np.concatenate([x[k] for x in acct_arrays]) for k in acct_arrays[0]}
            tx_parts.append(pd.DataFrame(merged))
            flush_tx()

    flush_tx(force=True)
    if not tx_chunks:
        tx_df = pd.DataFrame()
    elif len(tx_chunks) == 1:
        tx_df = tx_chunks[0]
    else:
        tx_df = pd.concat(tx_chunks, ignore_index=True)

    return dict(tx=tx_df, stmt=stmt, pay=pay, alloc=alloc, plan=plan, sched=sched,
                lim=lim, delinq=delinq, acct_state=acct_state)


# --------------------------------------------------------------------------
# One cycle of card transactions
# --------------------------------------------------------------------------
def _cycle_transactions(*, rng, cycle_start, cycle_end, base_lambda, limit, running,
                        merch_lookup, merch, merch_ids, merch_geo, merch_intl, merch_cat,
                        online_idx, on_w, instore_idx, pos_w, card_ids, channels,
                        customer_id, account_id, cust_geo, account_age_days):
    days = list(daterange(cycle_start, cycle_end))
    if not days:
        return [], 0.0

    w = np.array([
        MONTH_MULT[d.month] * DOW_MULT[d.weekday()] * (1.10 if d in _HOLIDAYS else 1.0)
        for d in days
    ], float)
    w = w / w.sum()
    lam = base_lambda / 30.44
    daily = np.array([lam * MONTH_MULT[d.month] * DOW_MULT[d.weekday()] *
                      (1.10 if d in _HOLIDAYS else 1.0) for d in days])
    n = int(rng.poisson(daily.sum()))
    n = min(n, 260)

    out = []
    balance = max(0.0, running)
    for _ in range(n):
        di = int(rng.choice(len(days), p=w))
        d = days[di]

        mcc = MCC_CODE[int(rng.choice(len(MCC_CODE), p=_MCC_W))]
        cat = MCC_CAT[mcc]
        idx, mp = merch_lookup[mcc]
        mi = int(rng.choice(idx, p=mp))
        online = rng.random() < MCC_ONLINE[mcc]

        if online:
            ci = int(rng.choice(online_idx, p=on_w))
        else:
            ci = int(rng.choice(instore_idx, p=pos_w))
        ch = channels.iloc[ci]

        hour = int(rng.choice(24, p=_hour_profile(cat)))
        minute = int(rng.integers(0, 60))
        second = int(rng.integers(0, 60))
        ts = pd.Timestamp(d) + pd.Timedelta(hours=hour, minutes=minute, seconds=second)

        raw_amt = float(rng.lognormal(np.log(MCC_MEDIAN[mcc]), MCC_SIGMA[mcc] * 0.75))
        amount = float(np.clip(round(AMOUNT_A * (raw_amt ** AMOUNT_B), 2),
                               1.0, C.US_TICKET_MAX))

        is_intl = bool(rng.random() < MCC_FOREIGN[mcc] * 0.35)
        if is_intl:
            cc = str(rng.choice(["CAD", "GBP", "EUR", "MXN", "JPY", "AUD"]))
            fx = {"CAD": 1.36, "GBP": 0.79, "EUR": 0.92, "MXN": 17.4, "JPY": 148.0, "AUD": 1.52}[cc]
            orig = round(amount * fx, 2)
            fx_rate = round(1 / fx, 6)
        else:
            cc, orig, fx_rate = "USD", amount, 1.0

        # ---- authorisation decision
        status = "APPROVED"
        code = None
        reason = None
        if amount > max(0.0, limit - balance) * 0.995:
            status = "DECLINED"
            code = "CREDIT_LIMIT_EXCEEDED"
            reason = C.DECLINE_REASONS[code]
        elif rng.random() < 0.011:
            status = "DECLINED"
            code = str(rng.choice(["DO_NOT_HONOR", "INVALID_CARD", "INCORRECT_PIN", "MERCHANT_BLOCKED"],
                                  p=[0.55, 0.20, 0.10, 0.15]))
            reason = C.DECLINE_REASONS[code]
        elif rng.random() < 0.004:
            status = "DECLINED"
            code = "SUSPECTED_FRAUD"
            reason = C.DECLINE_REASONS[code]
        elif rng.random() < 0.012:
            status = "REVERSED"

        if status == "APPROVED" and not is_intl and rng.random() < 0.022:
            # refund of an earlier purchase in the same cycle
            status = "REFUND_PLACEHOLDER"

        if status == "REFUND_PLACEHOLDER":
            status = "APPROVED"
            out.append(_txn_row(rng, "REFUND", -round(amount * float(rng.uniform(0.25, 1.0)), 2),
                                d, ts, mcc, cat, mi, merch, merch_ids, merch_geo, merch_intl,
                                ch, cc, orig, fx_rate, "APPROVED", None, None,
                                card_ids, customer_id, account_id, cust_geo, account_age_days))
            continue

        out.append(_txn_row(rng, "PURCHASE", amount, d, ts, mcc, cat, mi, merch, merch_ids,
                            merch_geo, merch_intl, ch, cc, orig, fx_rate, status, code, reason,
                            card_ids, customer_id, account_id, cust_geo, account_age_days))
        if status == "APPROVED":
            balance += amount

    # occasional cash advance
    if rng.random() < 0.004 and limit - balance > 500:
        amount = round(float(rng.uniform(500, min(8000, limit - balance))), 2)
        d = days[int(rng.integers(0, len(days)))]
        ts = pd.Timestamp(d) + pd.Timedelta(hours=int(rng.integers(9, 21)))
        ci = int(rng.choice(instore_idx, p=pos_w))
        out.append(_txn_row(rng, "CASH_ADVANCE", amount, d, ts, "6011", "Cash Advance",
                            int(rng.integers(0, len(merch))), merch, merch_ids, merch_geo,
                            merch_intl, channels.iloc[ci], "USD", amount, 1.0, "APPROVED",
                            None, None, card_ids, customer_id, account_id, cust_geo,
                            account_age_days))
        balance += amount

    return out, max(0.0, limit - balance)


def _txn_row(rng, ttype, amount, d, ts, mcc, cat, mi, merch, merch_ids, merch_geo,
             merch_intl, ch, cc, orig, fx_rate, status, code, reason, card_ids,
             customer_id, account_id, cust_geo, account_age_days):
    intl = bool(merch_intl[mi])
    risk = (
        1.55 * (abs(amount) > 500)
        + 2.30 * (not ch.is_card_present)
        + 2.60 * intl
        + 1.10 * (ts.hour <= 5)
        + 1.40 * (cat in ("Entertainment", "Electronics", "Digital", "E-Commerce"))
        + 0.55 * (account_age_days < 60)
        + 0.35 * (status == "DECLINED")
        + float(rng.normal(0, 0.8))
    )
    score = round(float(1 / (1 + np.exp(-(risk - 2.6)))), 4)
    return dict(
        transaction_id=None,
        date=d, ts=ts, mcc=mcc, category=cat,
        merchant_id=merch_ids[mi], merchant_name=merch.iloc[mi]["merchant_name"],
        merchant_geo_id=merch_geo[mi],
        channel_id=ch.channel_id, channel_group=ch.channel_group,
        entry_mode=ch.entry_mode, is_card_present=bool(ch.is_card_present),
        transaction_type=ttype, amount_usd=float(amount),
        currency_code=cc, original_amount=float(orig), fx_rate=float(fx_rate),
        status=status, decline_reason_code=code, decline_reason=reason,
        auth_code=f"{int(rng.integers(100000, 999999))}",
        is_international=intl,
        is_recurring=bool(rng.random() < 0.06),
        rewards_points=int(abs(amount) * 1.0) if status == "APPROVED" else 0,
        fraud_score=score,
        card_id=card_ids[int(rng.integers(0, len(card_ids)))],
    )


def _last_day(y, m):
    nxt = dt.date(y + (m // 12), m % 12 + 1, 1)
    return (nxt - dt.timedelta(days=1)).day


def _months(a: dt.date, b: dt.date):
    y, m = a.year, a.month
    while (y, m) <= (b.year, b.month):
        yield y, m
        m += 1
        if m == 13:
            m = 1
            y += 1


def _stage(dpd: int) -> str:
    """Credit-policy staging: delinquent above 60 DPD, charge-off above 180."""
    if dpd <= 0:
        return "CURRENT"
    if dpd <= 60:
        return "PAST_DUE"
    if dpd <= 180:
        return "DELINQUENT"
    return "CHARGE_OFF"


def _bucket(dpd: int) -> str:
    if dpd <= 0:
        return "CURRENT"
    if dpd < 30:
        return "1-29"
    if dpd < 60:
        return "30-59"
    if dpd < 90:
        return "60-89"
    if dpd < 120:
        return "90-119"
    return "120+"


_HOLIDAYS = set()
for _y in range(2025, 2028):
    from .utils import us_holidays
    _HOLIDAYS |= us_holidays(_y)


# --------------------------------------------------------------------------
# Installment plans
# --------------------------------------------------------------------------
def _emit_plan(rng, plan, sched, plan_id, acct, cust, txn, cut, sched_no):
    term = int(rng.choice([3, 6, 12, 18], p=[0.36, 0.34, 0.22, 0.08]))
    promo = 0.0 if rng.random() < 0.45 else float(round(rng.uniform(0.099, 0.249), 3))
    principal = float(txn["amount_usd"])
    interest_fee = round(principal * promo * term / 12.0, 2)
    monthly = round((principal + interest_fee) / term, 2)
    start = txn["date"]
    end = start + dt.timedelta(days=int(term * 30.44))
    plan.add(
        plan_id=plan_id,
        account_id=acct.account_id,
        customer_id=cust.name,
        transaction_id=txn["transaction_id"],
        plan_start_date=start,
        date_key=int(start.strftime("%Y%m%d")),
        principal_amount=round(principal, 2),
        term_months=term,
        promo_apr=promo,
        total_interest_fee=interest_fee,
        monthly_payment=monthly,
        total_repayment=round(monthly * term, 2),
        remaining_balance=round(monthly * term, 2),
        plan_status="ACTIVE",
        plan_end_date=end,
        offer_channel=str(rng.choice(["APP", "WEB", "CHECKOUT", "SMS"], p=[0.5, 0.25, 0.2, 0.05])),
    )
    remaining = monthly * term
    for k in range(1, term + 1):
        due = start + dt.timedelta(days=int(k * 30.44))
        paid = 0.0
        status = "PENDING"
        paid_date = None
        if due <= cut + dt.timedelta(days=10):
            r = rng.random()
            if r < 0.90:
                paid, status, paid_date = monthly, "PAID", due - dt.timedelta(days=int(rng.integers(0, 6)))
            elif r < 0.96:
                paid, status = round(monthly * 0.5, 2), "PARTIAL"
            else:
                status = "LATE"
        remaining = max(0.0, remaining - paid)
        sched.add(
            schedule_id=f"ISC{sched_no + k - 1:09d}",
            plan_id=plan_id,
            account_id=acct.account_id,
            customer_id=cust.name,
            installment_number=k,
            due_date=due,
            date_key=int(due.strftime("%Y%m%d")),
            amount_due=monthly,
            principal_component=round(principal / term, 2),
            interest_fee_component=round(interest_fee / term, 2),
            paid_amount=paid,
            paid_date=paid_date,
            status=status,
        )
    return term


# --------------------------------------------------------------------------
# Fraud alerts
# --------------------------------------------------------------------------
def build_fraud_alerts(tx: pd.DataFrame, rng) -> pd.DataFrame:
    """Build alerts from the fraud event already attached to each transaction.

    A blocked attempt is never settled and carries no loss; losses exist only for
    confirmed fraud that actually approved.
    """
    flagged = tx[tx.fraud_flag].copy()
    if flagged.empty:
        return pd.DataFrame()

    flagged["alert_type"] = rng.choice(
        list(C.FRAUD_TYPES.keys()), size=len(flagged),
        p=[0.46, 0.13, 0.14, 0.12, 0.09, 0.06])
    flagged["case_status"] = flagged.fraud_case_status
    flagged["decision"] = flagged.fraud_decision
    settled = flagged.status == "APPROVED"
    confirmed = flagged.case_status == "CONFIRMED_FRAUD"
    flagged["loss_amount"] = np.where(settled & confirmed, flagged.amount_usd, 0.0)
    flagged["recovered_amount"] = np.where(
        settled & confirmed,
        np.round(flagged.amount_usd * rng.uniform(0, 0.55, len(flagged)), 2), 0.0)
    flagged["resolution_days"] = rng.integers(1, 75, len(flagged))

    rows = []
    for i, (_, r) in enumerate(flagged.iterrows()):
        alert_date = pd.Timestamp(r.transaction_date).date() + dt.timedelta(days=int(rng.integers(0, 3)))
        res = None
        if r.case_status != "OPEN":
            res = alert_date + dt.timedelta(days=int(r.resolution_days))
            if res > C.PERIOD_END + dt.timedelta(days=45):
                res = None
        rows.append(dict(
            alert_id=f"FRD{i + 1:07d}", transaction_id=r.transaction_id,
            account_id=r.account_id, customer_id=r.customer_id, card_id=r.card_id,
            merchant_id=r.merchant_id, alert_date=alert_date,
            date_key=int(alert_date.strftime("%Y%m%d")), alert_type=r.alert_type,
            risk_score=r.fraud_score, decision=r.decision, case_status=r.case_status,
            loss_amount=round(float(r.loss_amount), 2),
            recovered_amount=round(float(r.recovered_amount), 2),
            net_loss=round(float(r.loss_amount) - float(r.recovered_amount), 2),
            resolution_date=res,
            ip_address=synthetic_ip(rng, foreign=bool(r.is_international)),
            device_id=hash_id(f"dev-{r.customer_id}")[:16],
            analyst_queue=str(rng.choice(["TRIAGE", "INVESTIGATION", "RECOVERY"], p=[0.55, 0.3, 0.15])),
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# App / digital events
# --------------------------------------------------------------------------
def _build_app_events_legacy(customers: pd.DataFrame, accounts: pd.DataFrame, rng) -> pd.DataFrame:
    events = [e for e, _ in C.APP_EVENTS]
    probs = np.array([p for _, p in C.APP_EVENTS], float)
    probs = probs / probs.sum()

    acct_by_cust = accounts.groupby("customer_id").agg(
        first_open=("open_date", "min"), last_open=("open_date", "max"),
        primary=("account_id", "first"), primary_card=("account_id", "first")).to_dict("index")

    rows = []
    eid = 1
    screens = {
        "app_launch": "/home",
        "card_home_viewed": "/card/home",
        "card_activation_intro_viewed": "/card/activation/intro",
        "card_activation_button_clicked": "/card/activation/intro",
        "kyc_document_viewed": "/onboarding/kyc/document",
        "kyc_permission_viewed": "/onboarding/kyc/permissions",
        "application_form_viewed": "/onboarding/application/form",
        "application_submitted": "/onboarding/application/review",
        "credit_decision_viewed": "/onboarding/decision",
        "card_activated_dashboard_viewed": "/card/activated",
        "card_details_viewed": "/card/details",
        "statement_list_viewed": "/statements",
        "statement_detail_viewed": "/statements/detail",
        "payment_page_viewed": "/payments/new",
        "payment_made": "/payments/confirmation",
        "payment_failed": "/payments/error",
        "installment_plan_viewed": "/installments/offer",
        "installment_plan_selected": "/installments/confirmation",
        "rewards_page_viewed": "/rewards",
        "card_lock_toggled": "/card/settings/lock",
        "fraud_alert_viewed": "/alerts/detail",
        "dispute_started": "/disputes/new",
        "support_chat_opened": "/support/chat",
        "notification_opened": "/notifications",
        "settings_viewed": "/settings",
    }

    for _, c in customers.iterrows():
        if c.customer_status == "CHURNED" and c.churn_date is not None:
            end = min(pd.Timestamp(c.churn_date).date(), C.PERIOD_END)
        else:
            end = C.PERIOD_END
        start = max(pd.Timestamp(c.onboarding_date).date(), C.PERIOD_START)
        if end <= start:
            continue
        span = (end - start).days
        n = int(max(4, rng.poisson(88 * (0.6 + 0.4 * min(1.0, span / 365)))))
        n = min(n, 400)
        day_offsets = rng.integers(0, max(1, span), n)
        platform = rng.choice(["ANDROID", "IOS", "WEB"], n, p=[0.53, 0.36, 0.11])

        # sessions
        n_sessions = max(1, n // 4)
        sess_of_event = rng.integers(0, n_sessions, n)

        for k in range(n):
            d = start + dt.timedelta(days=int(day_offsets[k]))
            hour = int(rng.choice(24, p=_hour_profile("Digital")))
            ts = pd.Timestamp(d) + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 60)),
                                                seconds=int(rng.integers(0, 60)))
            ev = events[int(rng.choice(len(events), p=probs))]
            plat = platform[k]
            if plat == "ANDROID":
                device = str(rng.choice(C.ANDROID_DEVICES)); os_v = "Android 14"
                app_v = str(rng.choice(["1.44.0", "1.45.1", "1.46.0", "1.47.2"]))
            elif plat == "IOS":
                device = str(rng.choice(C.IOS_DEVICES)); os_v = "iOS 18.3"
                app_v = str(rng.choice(["1.49.0", "1.50.0", "1.51.0", "1.52.1"]))
            else:
                device = str(rng.choice(C.WEB_BROWSERS)); os_v = None
                app_v = None
            rows.append(dict(
                event_id=f"EVT{eid:09d}",
                customer_id=c.customer_id,
                account_id=acct_by_cust.get(c.customer_id, {}).get("primary"),
                session_id=f"SES{sess_of_event[k]:06d}-{c.customer_id[-6:]}",
                event_timestamp=ts,
                event_date=ts.date(),
                date_key=int(ts.strftime("%Y%m%d")),
                event_name=ev,
                screen_path=screens.get(ev, "/home"),
                platform=plat,
                device=device,
                os_version=os_v,
                app_version=app_v,
                is_conversion_event=bool(ev in {"application_submitted", "card_activated_dashboard_viewed",
                                                "payment_made", "installment_plan_selected"}),
                properties_json=json_dumps({
                    "product_area": "cards",
                    "product_value_stream": "credit-card",
                    "flow_conversion": "yes-positive" if ev.endswith(("_made", "_submitted", "_selected")) else "no",
                    "spec_track": "2.1.0",
                }),
            ))
            eid += 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Account finalisation
# --------------------------------------------------------------------------
def build_app_events(customers: pd.DataFrame, accounts: pd.DataFrame,
                     rng: np.random.Generator) -> pd.DataFrame:
    """Vectorised digital event stream (~88 events per customer per year)."""
    if customers.empty:
        return pd.DataFrame()

    n_cust = len(customers)
    onboard = pd.to_datetime(customers.onboarding_date).to_numpy()
    end = np.where(
        customers.customer_status.eq("CHURNED") & customers.churn_date.notna(),
        pd.to_datetime(customers.churn_date).fillna(pd.Timestamp(C.PERIOD_END)).to_numpy(),
        np.datetime64(C.PERIOD_END))
    start = np.maximum(onboard, np.datetime64(C.PERIOD_START))
    span = ((end - start) / np.timedelta64(1, "D")).astype(int)
    span = np.maximum(span, 1)

    rate = 88 * (0.6 + 0.4 * np.minimum(1.0, span / 365))
    n_ev = np.clip(rng.poisson(rate), 4, 400).astype(int)

    ci = np.repeat(np.arange(n_cust), n_ev)
    total = len(ci)

    day_off = (rng.random(total) * span[ci]).astype(int)
    ts = start[ci] + day_off.astype("timedelta64[D]")

    hours = np.array([p for p in _hour_profile("Digital")])
    hour = rng.choice(24, size=total, p=hours)
    minute = rng.integers(0, 60, total)
    second = rng.integers(0, 60, total)
    ts = ts + hour.astype("timedelta64[h]") + minute.astype("timedelta64[m]") + second.astype("timedelta64[s]")

    events = [e for e, _ in C.APP_EVENTS]
    probs = np.array([p for _, p in C.APP_EVENTS], float)
    probs = probs / probs.sum()
    ev_code = rng.choice(len(events), size=total, p=probs)

    plat_code = rng.choice(3, size=total, p=[0.53, 0.36, 0.11])
    plat = np.array(["ANDROID", "IOS", "WEB"], dtype=object)[plat_code]

    device = np.empty(total, dtype=object)
    osv = np.empty(total, dtype=object)
    appv = np.empty(total, dtype=object)
    for code, (devs, osname, versions) in enumerate([
        (C.ANDROID_DEVICES, "Android 14", ["1.44.0", "1.45.1", "1.46.0", "1.47.2"]),
        (C.IOS_DEVICES, "iOS 18.3", ["1.49.0", "1.50.0", "1.51.0", "1.52.1"]),
        (C.WEB_BROWSERS, None, None),
    ]):
        m = plat_code == code
        if not m.any():
            continue
        k = int(m.sum())
        device[m] = np.array(devs, dtype=object)[rng.integers(0, len(devs), k)]
        osv[m] = osname
        appv[m] = (np.array(versions, dtype=object)[rng.integers(0, len(versions), k)]
                   if versions else None)

    screens = {
        "app_launch": "/home", "card_home_viewed": "/card/home",
        "card_activation_intro_viewed": "/card/activation/intro",
        "card_activation_button_clicked": "/card/activation/intro",
        "kyc_document_viewed": "/onboarding/kyc/document",
        "kyc_permission_viewed": "/onboarding/kyc/permissions",
        "application_form_viewed": "/onboarding/application/form",
        "application_submitted": "/onboarding/application/review",
        "credit_decision_viewed": "/onboarding/decision",
        "card_activated_dashboard_viewed": "/card/activated",
        "card_details_viewed": "/card/details", "statement_list_viewed": "/statements",
        "statement_detail_viewed": "/statements/detail", "payment_page_viewed": "/payments/new",
        "payment_made": "/payments/confirmation", "payment_failed": "/payments/error",
        "installment_plan_viewed": "/installments/offer",
        "installment_plan_selected": "/installments/confirmation",
        "rewards_page_viewed": "/rewards", "card_lock_toggled": "/card/settings/lock",
        "fraud_alert_viewed": "/alerts/detail", "dispute_started": "/disputes/new",
        "support_chat_opened": "/support/chat", "notification_opened": "/notifications",
        "settings_viewed": "/settings",
    }
    ev_name = np.array([events[i] for i in ev_code], dtype=object)
    ev_screen = np.array([screens.get(e, "/home") for e in ev_name], dtype=object)
    conversion = np.isin(ev_name, ["application_submitted", "card_activated_dashboard_viewed",
                                   "payment_made", "installment_plan_selected"])

    acct_by_cust = accounts.groupby("customer_id")["account_id"].first()
    acct_ids = acct_by_cust.reindex(customers.customer_id).to_numpy()

    tsidx = pd.DatetimeIndex(ts)
    n_sessions = np.maximum(1, n_ev // 4)
    sess = rng.integers(0, n_sessions.max(), total) % np.repeat(n_sessions, n_ev)

    df = pd.DataFrame({
        "event_id": [f"EVT{i:09d}" for i in range(1, total + 1)],
        "customer_id": customers.customer_id.to_numpy()[ci],
        "account_id": acct_ids[ci],
        "session_id": np.char.add(np.char.add("SES", sess.astype(str)),
                                  np.char.add("-", customers.customer_id.to_numpy()[ci].astype(str))),
        "event_timestamp": tsidx.to_numpy(),
        "event_date": tsidx.normalize().to_numpy(),
        "date_key": (tsidx.year * 10000 + tsidx.month * 100 + tsidx.day).to_numpy(),
        "event_name": ev_name,
        "screen_path": ev_screen,
        "platform": plat,
        "device": device,
        "os_version": osv,
        "app_version": appv,
        "is_conversion_event": conversion,
        "properties_json": np.full(total,
                                   json_dumps({"product_area": "cards",
                                               "product_value_stream": "credit-card",
                                               "spec_track": "2.1.0"}), dtype=object),
    })
    return df


# --------------------------------------------------------------------------
# Account finalisation
# --------------------------------------------------------------------------
def finalize_accounts(accounts: pd.DataFrame, state: dict, stmt: pd.DataFrame,
                      customers: pd.DataFrame) -> pd.DataFrame:
    acc = accounts.copy()
    mapped_limit = acc.account_id.map(lambda x: state.get(x, {}).get("credit_limit"))
    acc["credit_limit"] = mapped_limit.fillna(acc["credit_limit"])
    acc["current_balance"] = acc.account_id.map(lambda x: state.get(x, {}).get("current_balance", 0.0))
    acc["available_credit"] = (acc.credit_limit - acc.current_balance).clip(lower=0).round(2)
    acc["credit_balance"] = (-acc.current_balance).clip(lower=0).round(2)
    acc["utilization_pct"] = (acc.current_balance / acc.credit_limit.replace(0, np.nan)).fillna(0).round(4)
    acc["lifetime_purchases"] = acc.account_id.map(lambda x: state.get(x, {}).get("lifetime_purchases", 0.0))
    acc["lifetime_payments"] = acc.account_id.map(lambda x: state.get(x, {}).get("lifetime_payments", 0.0))
    acc["lifetime_interest"] = acc.account_id.map(lambda x: state.get(x, {}).get("lifetime_interest", 0.0))
    acc["lifetime_fees"] = acc.account_id.map(lambda x: state.get(x, {}).get("lifetime_fees", 0.0))

    last = stmt.sort_values("statement_date").groupby("account_id").tail(1).set_index("account_id")
    acc["last_statement_date"] = acc.account_id.map(last.statement_date)
    acc["last_statement_balance"] = acc.account_id.map(last.closing_balance).fillna(0.0)

    pay_agg = state.get("payment_agg", {})
    acc["last_payment_date"] = acc.account_id.map(
        lambda x: pay_agg.get(x, {}).get("last_payment_date"))
    acc["last_payment_amount"] = acc.account_id.map(
        lambda x: pay_agg.get(x, {}).get("last_payment_amount", 0.0)).fillna(0.0)

    acc["days_past_due"] = acc.account_id.map(
        lambda x: state.get(x, {}).get("days_past_due", 0)).fillna(0).astype(int)
    acc["delinquency_bucket"] = acc.days_past_due.map(_bucket)
    acc["delinquency_status"] = acc.days_past_due.map(_stage)

    # status
    status = []
    for _, r in acc.iterrows():
        st = state.get(r.account_id, {})
        if st.get("charged_off"):
            status.append("CHARGED_OFF")
        elif pd.notna(r.planned_close_date):
            status.append("CLOSED")
        else:
            status.append("ACTIVE")
    acc["account_status"] = status
    def _close_date(r):
        if r.account_status == "CLOSED":
            return r.planned_close_date
        if r.account_status == "CHARGED_OFF":
            return r.last_statement_date
        return None

    acc["closed_date"] = acc.apply(_close_date, axis=1)
    return acc.drop(columns=["planned_close_date", "target_utilization"])


# --------------------------------------------------------------------------
# Assemble
# --------------------------------------------------------------------------
def finalize_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    tx["is_international"] = tx.is_international.astype(bool)
    tx["is_recurring"] = tx.is_recurring.astype(bool)
    tx["is_card_present"] = tx.is_card_present.astype(bool)
    tx["transaction_date"] = pd.to_datetime(tx.transaction_date)
    return tx


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def _installment_cash(sched: pd.DataFrame, plan: pd.DataFrame):
    """Materialise pay-over-time repayments as cash + an allocation bridge."""
    if sched.empty:
        return pd.DataFrame(), pd.DataFrame()
    paid = sched[sched.paid_amount > 0].copy()
    if paid.empty:
        return pd.DataFrame(), pd.DataFrame()
    pay_date = pd.to_datetime(paid["paid_date"].fillna(paid["due_date"]))
    n = len(paid)
    ids = [f"IPY{i:09d}" for i in range(1, n + 1)]
    ipay = pd.DataFrame({
        "payment_id": ids,
        "account_id": paid["account_id"].to_numpy(),
        "customer_id": paid["customer_id"].to_numpy(),
        "payment_date": pay_date.dt.date.to_numpy(),
        "date_key": (pay_date.dt.year * 10000 + pay_date.dt.month * 100 + pay_date.dt.day).to_numpy(),
        "amount_usd": paid["paid_amount"].round(2).to_numpy(),
        "payment_method": "INSTALLMENT",
        "payment_channel": "PLAN",
        "payment_status": "POSTED",
        "is_autopay": False,
        "reference_number": [f"INST-{x}" for x in paid["schedule_id"]],
        "applied_statement_id": None,
        "posted_ts": pay_date.to_numpy(),
        "unapplied_amount": 0.0,
        "payment_type": "INSTALLMENT",
    })
    ialloc = pd.DataFrame({
        "allocation_id": [f"IAL{i:09d}" for i in range(1, n + 1)],
        "payment_id": ids,
        "schedule_id": paid["schedule_id"].to_numpy(),
        "plan_id": paid["plan_id"].to_numpy(),
        "account_id": paid["account_id"].to_numpy(),
        "allocation_date": pay_date.dt.date.to_numpy(),
        "amount_usd": paid["paid_amount"].round(2).to_numpy(),
    })
    return ipay, ialloc



def _apply_chargeoff_cancellations(sched: pd.DataFrame, plan: pd.DataFrame,
                                   state: dict):
    """Cancel instalments falling after a charge-off and restate the plan."""
    cutoff = {aid: st["last_statement_date"] for aid, st in state.items()
              if st.get("charged_off") and st.get("last_statement_date") is not None}
    if not cutoff or sched.empty:
        return sched, plan
    cut = sched["account_id"].map(cutoff)
    mask = cut.notna() & (sched["due_date"] > cut)
    if not mask.any():
        return sched, plan
    sched = sched.copy()
    sched.loc[mask, "paid_amount"] = 0.0
    sched.loc[mask, "paid_date"] = pd.NaT
    sched.loc[mask, "status"] = "CANCELLED"
    paid = sched.groupby("plan_id")["paid_amount"].sum()
    plan = plan.copy()
    plan["remaining_balance"] = (plan["total_repayment"] -
                                 plan["plan_id"].map(paid).fillna(0.0)).clip(lower=0).round(2)
    affected = set(sched.loc[mask, "plan_id"])
    plan.loc[plan.plan_id.isin(affected) & (plan.remaining_balance > 1), "plan_status"] = "CANCELLED"
    plan.loc[plan.remaining_balance <= 1, "plan_status"] = "COMPLETED"
    return sched, plan


def build_all(dims: dict) -> dict:
    from .utils import rng_for as _rng
    rng = _rng("facts_orchestrator")
    res = simulate_v2(dims)
    state = res["acct_state"]

    tx = res["tx"]
    stmt = res["stmt"].frame()
    pay = res["pay"].frame()
    alloc = res["alloc"].frame()
    plan = res["plan"].frame()
    sched = res["sched"].frame()
    plan_df = res["plan"].frame()
    sched, plan_df = _apply_chargeoff_cancellations(sched, plan_df, state)
    ipay, ialloc = _installment_cash(sched, plan_df)
    if not ipay.empty:
        pay = pd.concat([pay, ipay], ignore_index=True)
    plan = plan_df
    lim = res["lim"].frame()
    delinq = res["delinq"].frame()

    tx = finalize_transactions(tx)

    # rewards: points earned = spend x programme rate, only on approved purchases
    rate = dims["dim_product"].set_index("product_id")["rewards_rate"].to_dict()
    acct_rate = dims["dim_account"].set_index("account_id")["product_id"].map(rate).to_dict()
    pts = (tx.amount_usd.clip(lower=0) * tx.account_id.map(acct_rate).fillna(0) * 100).round().astype(int)
    tx["rewards_points"] = np.where(
        (tx.status == "APPROVED") & (tx.transaction_type == "PURCHASE"), pts, 0)

    # payment aggregates for the account dimension
    pay_agg = {}
    if not pay.empty:
        p = pay.copy()
        p["payment_date"] = pd.to_datetime(p.payment_date)
        last = p.sort_values("payment_date").groupby("account_id").tail(1).set_index("account_id")
        for aid, r in last.iterrows():
            pay_agg[aid] = dict(last_payment_date=r.payment_date.date(),
                                last_payment_amount=float(r.amount_usd))
    state["payment_agg"] = pay_agg

    # fraud
    fraud = build_fraud_alerts(tx, _rng("fraud"))
    if not fraud.empty:
        confirmed = set(fraud.loc[fraud.case_status == "CONFIRMED_FRAUD", "transaction_id"])
        tx["is_fraud"] = tx.transaction_id.isin(confirmed)
    else:
        tx["is_fraud"] = False

    accounts = finalize_accounts(dims["dim_account"], state, stmt, dims["dim_customer"])

    events = build_app_events(dims["dim_customer"], accounts, _rng("app_events"))

    # tidy date dtypes
    for df, cols in [
        (tx, ["transaction_date"]),
        (stmt, ["statement_date", "cycle_start_date", "cycle_end_date", "payment_due_date"]),
        (pay, ["payment_date"]),
        (alloc, ["allocation_date"]),
        (ialloc, ["allocation_date"]),
        (plan, ["plan_start_date", "plan_end_date"]),
        (sched, ["due_date", "paid_date"]),
        (lim, ["event_date"]),
        (delinq, ["snapshot_date"]),
        (fraud, ["alert_date", "resolution_date"]) if not fraud.empty else (pd.DataFrame(), []),
        (accounts, ["open_date", "closed_date", "last_statement_date", "last_payment_date"]),
        (events, ["event_timestamp", "event_date"]),
    ]:
        if df is None or df.empty:
            continue
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])

    return dict(
        fact_transaction=tx,
        fact_statement=stmt,
        fact_payment=pay,
        fact_payment_allocation=alloc,
        fact_installment_allocation=ialloc,
        fact_installment_plan=plan,
        fact_installment_schedule=sched,
        fact_credit_limit_event=lim,
        fact_delinquency_snapshot=delinq,
        fact_fraud_alert=fraud,
        fact_app_event=events,
        dim_account=accounts,
    )


# --------------------------------------------------------------------------
# Statement payment annotation (post-processing)
# --------------------------------------------------------------------------
def annotate_statements(stmt: pd.DataFrame, pay: pd.DataFrame) -> pd.DataFrame:
    """Derive paid_amount, payment_status and is_late per statement from payments.

    A statement is settled by payments received between its cut date and the
    next cut date for the same account.
    """
    if stmt.empty:
        return stmt
    s = stmt.sort_values(["account_id", "statement_date"]).copy()
    s["statement_date"] = pd.to_datetime(s["statement_date"])
    s["payment_due_date"] = pd.to_datetime(s["payment_due_date"])
    s["next_stmt"] = s.groupby("account_id")["statement_date"].shift(-1)
    s["next_stmt"] = s["next_stmt"].fillna(pd.Timestamp(C.PERIOD_END) + pd.Timedelta(days=60))

    paid_amounts, statuses, lates, last_pay_dates = [], [], [], []
    if pay.empty:
        p = pd.DataFrame(columns=["account_id", "payment_date", "amount_usd"])
    else:
        p = pay[["account_id", "payment_date", "amount_usd"]].copy()
        p["payment_date"] = pd.to_datetime(p.payment_date)

    by_acct = {k: v.sort_values("payment_date") for k, v in p.groupby("account_id")}
    empty = p.iloc[0:0]

    for _, r in s.iterrows():
        pp = by_acct.get(r.account_id, empty)
        if len(pp):
            win = pp[(pp.payment_date > r.statement_date) & (pp.payment_date <= r.next_stmt)]
        else:
            win = empty
        paid = float(win.amount_usd.sum()) if len(win) else 0.0
        paid = round(min(paid, max(0.0, r.closing_balance)), 2)
        last_pay = win.payment_date.max() if len(win) else pd.NaT
        late = bool(len(win) and last_pay.normalize() > r.payment_due_date)

        if r.closing_balance <= 0:
            status = "NO_BALANCE"
        elif paid >= r.closing_balance - 0.01:
            status = "PAID_FULL"
        elif paid >= r.minimum_due - 0.01 and paid > 0:
            status = "PAID_PARTIAL"
        elif paid > 0:
            status = "PARTIAL"
        elif r.payment_due_date > pd.Timestamp(C.PERIOD_END):
            status = "PENDING"
        else:
            status = "UNPAID"

        paid_amounts.append(paid)
        statuses.append(status)
        lates.append(late)
        last_pay_dates.append(last_pay)

    s["paid_amount"] = paid_amounts
    s["payment_status"] = statuses
    s["is_late"] = lates
    s["last_payment_date"] = last_pay_dates
    return s.drop(columns=["next_stmt"]).sort_values(["account_id", "statement_date"]).reset_index(drop=True)


# ==========================================================================
# Corrected simulator (v4)
# ==========================================================================
# Fixes the defects found in review:
#   * cycle aggregates reset every cycle (no stale reuse)
#   * one payment waterfall (oldest-due first) drives both DPD and statement
#     paid_amount / status / lateness
#   * payments scheduled by the final statement are flushed after the loop
#   * pay-over-time state is evaluated as of the period end
#   * cards are selected only when valid (issued, not replaced, not cancelled)
#   * charge-offs receive a closed date; churn closes every account
# ==========================================================================

def _auth_codes(rng, approved_mask, n):
    codes = rng.integers(0, 1_000_000, n)
    txt = np.char.zfill(codes.astype(str), 6).astype(object)
    return np.where(approved_mask, txt, None)


def _waterfall(stmts: list[dict], amount: float, pay_date):
    """Allocate a payment oldest-due first: minimums, then billed balances.

    Each statement's `billed` is its incremental debt (including pre-period
    opening debt on the first statement), so allocations never double-count a
    carried balance and never exceed what the statement actually billed.

    Returns (unallocated_remainder, [(statement_id, amount), ...]).
    """
    rem = float(amount)
    allocs: list[tuple[str, float]] = []
    for s in stmts:
        if rem <= 0.005:
            break
        need = s["min_eff"] - s["alloc_min"]
        if need > 0.005:
            take = min(rem, need)
            s["alloc_min"] += take
            s["alloc_bal"] += take
            s["last_pay_date"] = pay_date
            if pay_date <= s["due_date"]:
                s["paid_by_due"] = s["alloc_min"] >= s["min_eff"] - 0.01
            allocs.append((s["stmt_id"], take))
            s.setdefault("allocs", []).append((pay_date, take))
            rem -= take
    for s in stmts:
        if rem <= 0.005:
            break
        need = s["billed"] - s["alloc_bal"]
        if need > 0.005:
            take = min(rem, need)
            s["alloc_bal"] += take
            s["last_pay_date"] = pay_date
            allocs.append((s["stmt_id"], take))
            s.setdefault("allocs", []).append((pay_date, take))
            rem -= take
    return rem, allocs


def _settled_closing_amounts(stmts: list[dict], card_payments: list[tuple]) -> list[float]:
    """Settle each positive closing balance with all later card cash."""
    settled = []
    for s in stmts:
        later_cash = sum(amount for pay_date, amount in card_payments
                         if pay_date > s["statement_date"])
        settled.append(round(min(max(s["closing_balance"], 0.0), later_cash), 2))
    return settled


def _final_dpd(stmts: list[dict], as_of) -> int:
    """Recompute account DPD after every eligible payment at the account as-of date."""
    overdue = [s for s in stmts
               if s["due_date_obj"] <= as_of and s["alloc_min"] < s["min_eff"] - 0.01]
    if not overdue:
        return 0
    oldest = min(s["due_date_obj"] for s in overdue)
    return max(0, (as_of - oldest).days)


def _stmt_status(s: dict, period_end) -> tuple[str, bool]:
    if s["closing_balance"] < -0.01:
        return "CREDIT_BALANCE", False
    if s["billed"] <= 0.01:
        # no new debt was billed this cycle
        return ("NO_BALANCE" if s["closing_balance"] <= 0.0001 else "CARRY_FORWARD"), False
    if s["closing_balance"] <= 0.0001:
        return "NO_BALANCE", False
    if s.get("settled", 0.0) >= s["closing_balance"] - 0.01:
        status = "PAID_FULL"
    elif s["min_eff"] > 0.01 and s["alloc_min"] >= s["min_eff"] - 0.01:
        status = "PAID_PARTIAL"
    elif s["alloc_bal"] > 0.01:
        status = "PARTIAL"
    elif s["due_date"] > period_end:
        status = "PENDING"
    else:
        status = "UNPAID"
    late = bool(s["min_eff"] > 0.01 and not s["paid_by_due"] and s["due_date"] <= period_end)
    return status, late


def _emit_plan_v2(rng, plan, sched, plan_id, acct, customer_id, txn, sched_no, close_date=None):
    term = int(rng.choice([3, 6, 12, 18], p=[0.36, 0.34, 0.22, 0.08]))
    promo = 0.0 if rng.random() < 0.45 else float(round(rng.uniform(0.099, 0.249), 3))
    principal = float(txn["amount_usd"])
    interest_fee = round(principal * promo * term / 12.0, 2)
    monthly = round((principal + interest_fee) / term, 2)
    start = txn["date"]
    end = start + dt.timedelta(days=int(term * 30.44))

    paid_total = 0.0
    delinq = False
    any_cancel = False
    remaining = monthly * term
    rows = []
    for k in range(1, term + 1):
        due = start + dt.timedelta(days=int(k * 30.44))
        paid, status, paid_date = 0.0, "PENDING", None
        cancelled = bool(close_date is not None and due > close_date)
        if cancelled:
            status = "CANCELLED"
            any_cancel = True
        elif due <= min(C.PERIOD_END, close_date or C.PERIOD_END):
            r = rng.random()
            if r < 0.90:
                paid, status = monthly, "PAID"
                paid_date = due - dt.timedelta(days=int(rng.integers(0, 6)))
            elif r < 0.96:
                paid, status = round(monthly * 0.5, 2), "PARTIAL"
                paid_date = due
                delinq = True
            else:
                status = "LATE"
                delinq = True
        paid_total += paid
        remaining = max(0.0, remaining - paid)
        rows.append((k, due, paid, status, paid_date))

    if remaining <= 1.0:
        plan_status = "COMPLETED"
    elif any_cancel:
        plan_status = "CANCELLED"
    elif delinq:
        plan_status = "DELINQUENT"
    else:
        plan_status = "ACTIVE"
    plan.add(
        plan_id=plan_id, account_id=acct.account_id, customer_id=customer_id,
        transaction_id=txn["transaction_id"], plan_start_date=start,
        date_key=int(start.strftime("%Y%m%d")), principal_amount=round(principal, 2),
        term_months=term, promo_apr=promo, total_interest_fee=interest_fee,
        monthly_payment=monthly, total_repayment=round(monthly * term, 2),
        remaining_balance=round(remaining, 2), plan_status=plan_status,
        plan_end_date=end,
        offer_channel=str(rng.choice(["APP", "WEB", "CHECKOUT", "SMS"], p=[0.5, 0.25, 0.2, 0.05])),
    )
    for (k, due, paid, status, paid_date) in rows:
        sched.add(
            schedule_id=f"ISC{sched_no + k - 1:09d}", plan_id=plan_id,
            account_id=acct.account_id, customer_id=customer_id,
            installment_number=k, due_date=due, date_key=int(due.strftime("%Y%m%d")),
            amount_due=monthly, principal_component=round(principal / term, 2),
            interest_fee_component=round(interest_fee / term, 2),
            paid_amount=paid, paid_date=paid_date, status=status,
        )
    return term


def simulate_v2(dims: dict) -> dict:
    rng = rng_for("facts_v4")
    accounts = dims["dim_account"].copy()
    customers = dims["dim_customer"].set_index("customer_id")
    merch = dims["dim_merchant"]
    channels = dims["dim_channel"]
    ctx = txgen.TxContext(merch, channels)

    cards_df = dims["dim_card"]
    cards_by_acct = {
        aid: g[["card_id", "issue_date", "replaced_date"]].to_dict("records")
        for aid, g in cards_df.groupby("account_id")
    }

    tx_parts: list[pd.DataFrame] = []
    tx_chunks: list[pd.DataFrame] = []

    def flush_tx(force=False):
        if len(tx_parts) >= 4000 or (force and tx_parts):
            tx_chunks.append(pd.concat(tx_parts, ignore_index=True))
            tx_parts.clear()
            if len(tx_chunks) > 4:
                tx_chunks[:] = [pd.concat(tx_chunks, ignore_index=True)]

    stmt = Buf()
    pay = Buf()
    alloc = Buf()
    plan = Buf()
    sched = Buf()
    lim = Buf()
    delinq = Buf()

    txn_no = 1
    stmt_no = 1
    pay_no = 1
    alloc_no = 1
    plan_no = 1
    sched_no = 1
    lim_no = 1
    snap_no = 1
    acct_state: dict[str, dict] = {}

    for _, a in accounts.iterrows():
        acc_id = a.account_id
        cust = customers.loc[a.customer_id]
        open_date = a.open_date if isinstance(a.open_date, dt.date) else pd.Timestamp(a.open_date).date()
        planned_close = a.planned_close_date
        if planned_close is not None and not isinstance(planned_close, dt.date):
            planned_close = pd.Timestamp(planned_close).date()
        eff_end = min(planned_close or C.PERIOD_END, C.PERIOD_END)
        if eff_end < C.PERIOD_START:
            continue

        limit = float(a.credit_limit)
        apr = float(a.apr_purchase)
        statement_day = int(a.statement_day)
        cards = cards_by_acct.get(acc_id) or [{"card_id": None, "issue_date": open_date,
                                               "replaced_date": None}]

        seg_cfg = SEGMENT_PARAMS.get(str(cust.customer_segment), SEGMENT_PARAMS["CORE_MAINSTREAM"])
        base_lambda = seg_cfg["median_txn"] * float(
            np.clip(rng.lognormal(0.0, seg_cfg["sigma"]), 0.15, 4.5))
        base_lambda = float(np.clip(base_lambda, 0.4, 260.0))

        if open_date < C.PERIOD_START:
            util = float(np.clip(rng.beta(2.1, 3.4), 0.02, 0.72))
            if rng.random() < 0.55:
                util *= 0.35
            opening = round(limit * util, 2)
        else:
            opening = 0.0

        p_pif = float(np.clip(rng.normal(0.60, 0.10), 0.30, 0.82))
        p_min = float(np.clip(rng.normal(0.16, 0.04), 0.06, 0.28))
        p_part = float(np.clip(rng.normal(0.12, 0.04), 0.04, 0.22))
        p_late = float(np.clip(rng.normal(0.07, 0.03), 0.01, 0.15))

        stmt_dates = []
        for y, m in _months(open_date, eff_end):
            d = dt.date(y, m, min(statement_day, _last_day(y, m)))
            if open_date <= d <= eff_end and d >= C.PERIOD_START:
                stmt_dates.append(d)
        if not stmt_dates:
            continue

        balance = opening
        stmts: list[dict] = []
        pending: list[dict] = []
        card_payments: list[tuple] = []
        acct_arrays: list[dict] = []
        prev_cut = None
        gross_purch = 0.0
        pay_total = 0.0
        interest_total = 0.0
        fees_total = 0.0
        charged_off = False
        last_dpd = 0
        stmt_rows: list[dict] = []

        for cut in stmt_dates:
            cycle_start = max(open_date, C.PERIOD_START,
                              (prev_cut + dt.timedelta(days=1)) if prev_cut else open_date)
            cycle_end = min(cut, eff_end)
            if cycle_end < cycle_start:
                prev_cut = cut
                continue

            opening_balance = balance

            # -------- payments due in this cycle --------
            due_pays = [p for p in pending if p["date"] <= cycle_end]
            pending = [p for p in pending if p["date"] > cycle_end]
            paid_this_cycle = 0.0
            for p in sorted(due_pays, key=lambda x: x["date"]):
                pid = f"PAY{pay_no:09d}"
                rem, allocs = _waterfall(stmts, p["amount"], p["date"])
                balance -= p["amount"]
                paid_this_cycle += p["amount"]
                pay_total += p["amount"]
                card_payments.append((p["date"], p["amount"]))
                pay.add(
                    payment_id=pid, account_id=acc_id,
                    customer_id=a.customer_id, payment_date=p["date"],
                    date_key=int(p["date"].strftime("%Y%m%d")), amount_usd=round(p["amount"], 2),
                    payment_method=p["method"], payment_channel=p["channel"],
                    payment_status="POSTED", is_autopay=p["autopay"], payment_type="CARD",
                    unapplied_amount=round(rem, 2),
                    reference_number=p["ref"],
                    applied_statement_id=(allocs[0][0] if allocs else p["stmt_id"]),
                    posted_ts=pd.Timestamp(p["date"]) + pd.Timedelta(hours=int(rng.integers(6, 21))),
                )
                for sid_a, amt_a in allocs:
                    alloc.add(allocation_id=f"ALC{alloc_no:09d}", payment_id=pid,
                              statement_id=sid_a, account_id=acc_id,
                              allocation_date=p["date"], date_key=int(p["date"].strftime("%Y%m%d")),
                              amount_usd=round(amt_a, 2))
                    alloc_no += 1
                pay_no += 1

            # -------- transactions --------
            running = max(0.0, balance)
            month_mult = float(rng.lognormal(0.0, C.TXN_MONTH_SIGMA))
            if rng.random() < C.SPIKE_MONTH_PROB:
                month_mult *= float(rng.uniform(*C.SPIKE_MULT_RANGE))
            days = list(daterange(cycle_start, cycle_end))
            out = None
            if days:
                w = np.array([MONTH_MULT[d.month] * DOW_MULT[d.weekday()] *
                              (1.10 if d in _HOLIDAYS else 1.0) for d in days], float)
                w = w / w.sum()
                lam = base_lambda / 30.44
                daily = np.array([lam * MONTH_MULT[d.month] * DOW_MULT[d.weekday()] *
                                  (1.10 if d in _HOLIDAYS else 1.0) for d in days])
                n_draw = min(int(rng.poisson(daily.sum() * month_mult)), C.MAX_TXN_PER_MONTH)
                out, _ = txgen.generate_cycle(ctx, rng, days, w, n_draw, limit, running,
                                              (cycle_start - open_date).days)

            cyc_sum = cyc_purch = cyc_refund = 0.0
            if out is not None:
                m = len(out["amount_usd"])
                sid = f"STM{stmt_no:09d}"
                ids = np.array([f"TXN{txn_no + i:09d}" for i in range(m)], dtype=object)
                txn_no += m
                ttype, status, amt = out["transaction_type"], out["status"], out["amount_usd"]
                appr = status == "APPROVED"

                # ---- pay-over-time: financed principal leaves the revolving ledger
                plan_ids = np.full(m, None, dtype=object)
                financed = np.zeros(m, bool)
                cand = np.where(appr & (ttype == "PURCHASE") & (amt >= 150))[0]
                if len(cand):
                    take = cand[rng.random(len(cand)) < 0.05]
                    for j in take:
                        pid = f"PLN{plan_no:08d}"
                        added = _emit_plan_v2(rng, plan, sched, pid, a, a.customer_id,
                                              dict(transaction_id=ids[j],
                                                   date=pd.Timestamp(out["ts"][j]).date(),
                                                   amount_usd=float(amt[j])), sched_no, eff_end)
                        sched_no += added
                        plan_no += 1
                        plan_ids[j] = pid
                        financed[j] = True

                billed = ~financed
                cyc_sum = float(amt[appr & billed].sum())
                cyc_purch = float(amt[appr & (ttype == "PURCHASE") & billed].sum())
                cyc_refund = float(amt[ttype == "REFUND"].sum())
                gross_purch += cyc_purch

                ci = out["channel_idx"]
                tsidx = pd.DatetimeIndex(out["ts"])
                card_list = sorted(cards, key=lambda c: c["issue_date"])
                card_for = np.full(m, card_list[0]["card_id"], dtype=object)
                for _c in card_list[1:]:
                    card_for[tsidx >= pd.Timestamp(_c["issue_date"])] = _c["card_id"]

                pp = out.get("parent_pos")
                if pp is None:
                    orig_ids = np.full(m, None, dtype=object)
                else:
                    orig_ids = np.array([ids[p] if p >= 0 else None for p in pp], dtype=object)

                is_fraud = (out["fraud_case_status"] == "CONFIRMED_FRAUD") & appr
                acct_arrays.append(dict(
                    transaction_id=ids, account_id=np.full(m, acc_id, dtype=object),
                    card_id=card_for,
                    customer_id=np.full(m, a.customer_id, dtype=object),
                    merchant_id=ctx.merch_ids[out["merchant_idx"]],
                    merchant_name=ctx.merch_names[out["merchant_idx"]],
                    mcc=out["mcc"], category=out["category"],
                    channel_id=ctx.ch_id[ci], channel_group=ctx.ch_group[ci],
                    entry_mode=ctx.ch_entry[ci], is_card_present=ctx.ch_present[ci],
                    merchant_geo_id=ctx.merch_geo[out["merchant_idx"]],
                    customer_geo_id=np.full(m, cust.home_geo_id, dtype=object),
                    transaction_date=tsidx.normalize().to_numpy(),
                    date_key=(tsidx.year * 10000 + tsidx.month * 100 + tsidx.day).to_numpy(),
                    transaction_ts=tsidx.to_numpy(), transaction_type=ttype, amount_usd=amt,
                    currency_code=out["currency_code"], original_amount=out["original_amount"],
                    fx_rate=out["fx_rate"], status=status,
                    decline_reason_code=np.where(out["decline_code"] == "", None, out["decline_code"]),
                    decline_reason=np.where(out["decline_code"] == "", None, out["decline_code"]),
                    auth_code=_auth_codes(rng, appr, m),
                    is_international=out["is_international"],
                    is_recurring=ctx.merch_recurring[out["merchant_idx"]],
                    rewards_points=np.zeros(m, dtype=np.int64), installment_plan_id=plan_ids,
                    is_installment_financed=financed,
                    original_transaction_id=orig_ids,
                    fraud_flag=out["fraud_flag"], fraud_decision=out["fraud_decision"],
                    fraud_case_status=out["fraud_case_status"],
                    fraud_score=out["fraud_score"], is_fraud=is_fraud,
                    statement_id=np.full(m, sid, dtype=object),
                    source_system=np.full(m, "CARD_PROCESSOR", dtype=object),
                ))
            else:
                sid = f"STM{stmt_no:09d}"

            # -------- interest & fees --------
            interest = 0.0
            if running > 25 and (not stmts or any(s["alloc_bal"] < s["billed"] - 1 for s in stmts)):
                interest = round(running * apr / 12.0, 2)
            fees = 0.0
            if open_date.month == cut.month and (cut - open_date).days >= 300:
                fees += float(next((p["annual_fee"] for p in C.PRODUCTS
                                    if p["product_id"] == a.product_id), 0.0))
            ftf = float(a.foreign_txn_fee or 0)
            if ftf > 0 and out is not None:
                midx = (out["is_international"] & (out["status"] == "APPROVED")
                        & (out["transaction_type"] == "PURCHASE"))
                intl = float(out["amount_usd"][midx].sum())
                if intl > 0:
                    fees += round(intl * ftf, 2)
            overdue = [s for s in stmts
                       if s["due_date"] < cut and s["alloc_min"] < s["min_eff"] - 0.01]
            if overdue:
                fees += 35.0 if running > 400 else 29.0

            charges = cyc_sum + interest + fees
            closing = round(opening_balance + charges - paid_this_cycle, 2)
            balance = closing
            min_due = 0.0
            if closing > 0:
                min_due = round(max(35.0, closing * 0.01 + interest + fees), 2)
                min_due = min(min_due, closing)
            due_date = cut + dt.timedelta(days=25)

            billed = round(max(charges + (opening_balance if not stmts else 0.0), 0.0), 2)
            min_eff = round(min(min_due, billed), 2)
            rec = dict(stmt_id=sid, account_id=acc_id, customer_id=a.customer_id,
                       product_id=a.product_id, statement_date=cut,
                       date_key=int(cut.strftime("%Y%m%d")), month_key=cut.strftime("%Y-%m"),
                       cycle_start_date=cycle_start, cycle_end_date=cycle_end,
                       payment_due_date=due_date, due_date=due_date,
                       credit_limit=round(limit, 2),
                       opening_balance=round(opening_balance, 2),
                       purchases_amount=round(cyc_purch, 2), refunds_amount=round(cyc_refund, 2),
                       cash_advances_amount=0.0, payments_amount=round(paid_this_cycle, 2),
                       interest_charged=interest, fees_charged=round(fees, 2),
                       closing_balance=closing, minimum_due=min_due,
                       available_credit=round(max(0.0, limit - closing), 2),
                       utilization_pct=round(min(1.5, closing / limit), 4) if limit else 0.0,
                       transaction_count=int(len(out["amount_usd"])) if out is not None else 0,
                       billed=billed,
                       min_due=min_due, min_eff=min_eff,
                       alloc_min=0.0, alloc_bal=0.0, paid_by_due=False, allocs=[],
                       last_pay_date=None, due_date_obj=due_date)
            stmts.append(rec)
            stmt_no += 1
            interest_total += interest
            fees_total += fees

            # -------- schedule the payment --------
            if closing > 0 and due_date <= C.PAYMENT_HORIZON:
                r = rng.random()
                amt = 0.0
                pay_date = due_date - dt.timedelta(days=int(rng.integers(2, 18)))
                if r < p_pif:
                    amt = closing
                elif r < p_pif + p_min:
                    amt = min_due
                elif r < p_pif + p_min + p_part:
                    amt = round(closing * float(rng.uniform(0.20, 0.85)), 2)
                elif r < p_pif + p_min + p_part + p_late:
                    amt = closing
                    pay_date = due_date + dt.timedelta(days=int(rng.integers(3, 21)))
                if amt > 0 and pay_date <= C.PAYMENT_HORIZON:
                    method = str(rng.choice(["ACH", "DEBIT_CARD", "CHECK", "WIRE", "INTERNAL_TRANSFER"],
                                            p=[0.56, 0.24, 0.08, 0.02, 0.10]))
                    pending.append(dict(date=pay_date, amount=min(amt, closing), stmt_id=sid,
                                        method=method,
                                        channel="MOBILE_APP" if method in ("ACH", "INTERNAL_TRANSFER") else "WEB",
                                        autopay=bool(method == "ACH" and rng.random() < 0.45),
                                        ref=f"PMT{hash_id(str(acc_id) + sid + str(pay_date))[:16].upper()}"))

            # -------- snapshot --------
            overdue_now = [s for s in stmts
                           if s["due_date_obj"] < cut and s["alloc_min"] < s["min_eff"] - 0.01]
            if overdue_now:
                oldest = min(overdue_now, key=lambda s: s["due_date_obj"])
                dpd = (cut - oldest["due_date_obj"]).days
                amt_past = sum(s["min_eff"] - s["alloc_min"] for s in overdue_now)
                n_unpaid = len(overdue_now)
            else:
                dpd, amt_past, n_unpaid = 0, 0.0, 0
            last_dpd = dpd
            delinq.add(snapshot_id=f"SNP{snap_no:09d}", account_id=acc_id,
                       customer_id=a.customer_id, snapshot_date=cut,
                       date_key=int(cut.strftime("%Y%m%d")), month_key=cut.strftime("%Y-%m"),
                       days_past_due=dpd, delinquency_bucket=_bucket(dpd),
                       delinquency_status=_stage(dpd),
                       balance=closing, amount_past_due=round(amt_past, 2),
                       minimum_due=min_due, unpaid_statement_count=n_unpaid,
                       credit_limit=round(limit, 2))
            snap_no += 1

            # -------- limit management (applies from the next cycle) --------
            if rng.random() < 0.006 and (cut - open_date).days > 240 and not overdue:
                inc = round(limit * float(rng.uniform(0.10, 0.35)) / 100) * 100
                new_limit = min(limit + inc, 50000)
                if new_limit > limit:
                    lim.add(limit_event_id=f"LIM{lim_no:07d}", account_id=acc_id,
                            customer_id=a.customer_id, event_date=cut,
                            date_key=int(cut.strftime("%Y%m%d")), event_type="CREDIT_LIMIT_INCREASE",
                            old_limit=round(limit, 2), new_limit=round(new_limit, 2),
                            change_amount=round(new_limit - limit, 2),
                            change_pct=round((new_limit - limit) / limit, 4),
                            reason=str(rng.choice(["INCOME_VERIFIED", "PAYMENT_HISTORY",
                                                   "CUSTOMER_REQUEST", "PORTFOLIO_REVIEW"])),
                            channel="PERIODIC_REVIEW")
                    lim_no += 1
                    limit = new_limit
            elif rng.random() < 0.004 and overdue and limit > 4500:
                new_limit = max(4500, round(limit * 0.70 / 100) * 100)
                if new_limit < limit:
                    lim.add(limit_event_id=f"LIM{lim_no:07d}", account_id=acc_id,
                            customer_id=a.customer_id, event_date=cut,
                            date_key=int(cut.strftime("%Y%m%d")), event_type="CREDIT_LIMIT_DECREASE",
                            old_limit=round(limit, 2), new_limit=round(new_limit, 2),
                            change_amount=round(new_limit - limit, 2),
                            change_pct=round((new_limit - limit) / limit, 4),
                            reason="RISK_ACTION", channel="RISK_SYSTEM")
                    lim_no += 1
                    limit = new_limit

            if dpd > 180:
                charged_off = True
            prev_cut = cut
            if charged_off:
                break

        # -------- flush payments scheduled after the final statement --------
        # charged-off accounts receive no further payments (no recovery modelled)
        # no payment may land after the account closes (or after charge-off)
        if charged_off:
            pending_flush = []
        else:
            pending_flush = [x for x in pending
                             if x["date"] <= min(eff_end, C.PAYMENT_HORIZON)]
        for p in sorted(pending_flush, key=lambda x: x["date"]):
            pid = f"PAY{pay_no:09d}"
            rem, allocs = _waterfall(stmts, p["amount"], p["date"])
            balance -= p["amount"]
            pay_total += p["amount"]
            card_payments.append((p["date"], p["amount"]))
            pay.add(payment_id=pid, account_id=acc_id,
                    customer_id=a.customer_id, payment_date=p["date"],
                    date_key=int(p["date"].strftime("%Y%m%d")), amount_usd=round(p["amount"], 2),
                    payment_method=p["method"], payment_channel=p["channel"],
                    payment_status="POSTED", is_autopay=p["autopay"], payment_type="CARD",
                    unapplied_amount=round(rem, 2),
                    reference_number=p["ref"],
                    applied_statement_id=(allocs[0][0] if allocs else p["stmt_id"]),
                    posted_ts=pd.Timestamp(p["date"]) + pd.Timedelta(hours=int(rng.integers(6, 21))))
            for sid_a, amt_a in allocs:
                alloc.add(allocation_id=f"ALC{alloc_no:09d}", payment_id=pid,
                          statement_id=sid_a, account_id=acc_id,
                          allocation_date=p["date"], date_key=int(p["date"].strftime("%Y%m%d")),
                          amount_usd=round(amt_a, 2))
                alloc_no += 1
            pay_no += 1

        # -------- settlement: all later card cash, capped at each closing balance
        for s_i, settled in zip(stmts, _settled_closing_amounts(stmts, card_payments)):
            s_i["settled"] = settled

        # -------- final delinquency state after every eligible payment --------
        as_of = prev_cut if charged_off else eff_end
        last_dpd = _final_dpd(stmts, as_of)

        # -------- materialise statements with final allocation --------
        for s in stmts:
            status, late = _stmt_status(s, C.PERIOD_END)
            stmt.add(
                statement_id=s["stmt_id"], account_id=s["account_id"],
                customer_id=s["customer_id"], product_id=s["product_id"],
                statement_date=s["statement_date"], date_key=s["date_key"],
                month_key=s["month_key"], cycle_start_date=s["cycle_start_date"],
                cycle_end_date=s["cycle_end_date"], payment_due_date=s["payment_due_date"],
                credit_limit=s["credit_limit"], opening_balance=s["opening_balance"],
                purchases_amount=s["purchases_amount"], refunds_amount=s["refunds_amount"],
                cash_advances_amount=s["cash_advances_amount"],
                payments_amount=s["payments_amount"], interest_charged=s["interest_charged"],
                fees_charged=s["fees_charged"], closing_balance=s["closing_balance"],
                minimum_due=s["minimum_due"], available_credit=s["available_credit"],
                utilization_pct=s["utilization_pct"], transaction_count=s["transaction_count"],
                billed_amount=s["billed"],
                paid_amount=round(s["alloc_bal"], 2),
                paid_closing_amount=round(s.get("settled", 0.0), 2),
                payment_status=status, is_late=late,
                last_payment_date=s["last_pay_date"],
            )

        if acct_arrays:
            merged = {k: np.concatenate([x[k] for x in acct_arrays]) for k in acct_arrays[0]}
            tx_parts.append(pd.DataFrame(merged))
            flush_tx()

        acct_state[acc_id] = dict(
            current_balance=round(balance, 2), credit_limit=round(limit, 2),
            lifetime_purchases=round(gross_purch, 2), lifetime_payments=round(pay_total, 2),
            lifetime_interest=round(interest_total, 2), lifetime_fees=round(fees_total, 2),
            days_past_due=int(last_dpd), charged_off=charged_off,
            last_statement_date=prev_cut,
        )

    flush_tx(force=True)
    if not tx_chunks:
        tx_df = pd.DataFrame()
    elif len(tx_chunks) == 1:
        tx_df = tx_chunks[0]
    else:
        tx_df = pd.concat(tx_chunks, ignore_index=True)

    return dict(tx=tx_df, stmt=stmt, pay=pay, alloc=alloc, plan=plan, sched=sched,
                lim=lim, delinq=delinq, acct_state=acct_state)
