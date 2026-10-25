"""Validation: referential integrity, distributions and reconciliation.

Produces docs/validation_report.md with a PASS/FAIL/WARN line per check plus
distribution tables that compare the synthetic book with the PHP source.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config as C
from .utils import log


LAST_FAIL_COUNT = 0


class Report:
    def __init__(self):
        self.rows = []
        self.fail_count = 0

    def check(self, name, ok, detail="", level="PASS"):
        if ok is True:
            level = "PASS"
        elif ok is False:
            level = "FAIL"
        else:
            level = level
        if level == "FAIL":
            self.fail_count += 1
        self.rows.append(dict(check=name, result=level, detail=detail))
        log(f"  [{level}] {name} :: {detail}")

    def md(self) -> str:
        d = pd.DataFrame(self.rows)
        lines = ["# Validation report", "",
                 f"Generated for period **{C.PERIOD_START} to {C.PERIOD_END}**.",
                 "", "| Check | Result | Detail |", "|---|---|---|"]
        for _, r in d.iterrows():
            lines.append(f"| {r['check']} | {r['result']} | {r['detail']} |")
        counts = d["result"].value_counts().to_dict()
        lines += ["", "**Summary:** " + ", ".join(f"{k}: {v}" for k, v in counts.items()), ""]
        return "\n".join(lines)


def _pct(series, q):
    s = pd.to_numeric(series, errors="coerce").dropna()
    s = s[s > 0]
    if s.empty:
        return None
    return round(float(s.quantile(q)), 2)


def _expected_paid_closing(stmt: pd.DataFrame, pay: pd.DataFrame) -> pd.Series:
    """Reconstruct closing settlement from all later CARD cash, not allocations."""
    expected = np.zeros(len(stmt), dtype=float)
    if stmt.empty:
        return pd.Series(expected, index=stmt.index)

    statement_dates = pd.to_datetime(stmt.statement_date).to_numpy()
    closing = stmt.closing_balance.clip(lower=0).to_numpy(dtype=float)
    card = pay.loc[pay.payment_type == "CARD",
                   ["account_id", "payment_date", "amount_usd"]].copy()
    card["payment_date"] = pd.to_datetime(card.payment_date)
    card = card[card.payment_date <= pd.Timestamp(C.PERIOD_END)].sort_values(
        ["account_id", "payment_date"])
    payment_groups = card.groupby("account_id", sort=False).indices
    payment_dates = card.payment_date.to_numpy()
    payment_amounts = card.amount_usd.to_numpy(dtype=float)

    # searchsorted gives the first payment strictly after each statement cut;
    # suffix cash is deliberately reused by every earlier statement.
    for account_id, statement_pos in stmt.groupby("account_id", sort=False).indices.items():
        payment_pos = payment_groups.get(account_id)
        if payment_pos is None or len(payment_pos) == 0:
            continue
        dates = payment_dates[payment_pos]
        amounts = payment_amounts[payment_pos]
        cumulative = np.concatenate(([0.0], np.cumsum(amounts)))
        first_later = np.searchsorted(dates, statement_dates[statement_pos], side="right")
        later_cash = cumulative[-1] - cumulative[first_later]
        expected[statement_pos] = np.minimum(closing[statement_pos], later_cash)
    return pd.Series(expected, index=stmt.index)


def _expected_final_dpd(acct: pd.DataFrame, stmt: pd.DataFrame,
                        pay: pd.DataFrame, alloc: pd.DataFrame) -> pd.Series:
    """Rebuild final account DPD from eligible allocation coverage at cut-off."""
    accounts = acct[["account_id", "account_status", "closed_date"]].copy()
    accounts["cutoff"] = pd.to_datetime(accounts.closed_date).where(
        accounts.account_status.isin(["CLOSED", "CHARGED_OFF"]),
        pd.Timestamp(C.PERIOD_END))
    if accounts.cutoff.isna().any():
        raise ValueError("closed/charged-off account has no closed_date")

    card = pay.loc[pay.payment_type == "CARD", ["payment_id", "payment_date"]].copy()
    card["payment_date"] = pd.to_datetime(card.payment_date)
    statements = stmt[["statement_id", "account_id", "payment_due_date",
                       "minimum_due", "billed_amount"]].merge(
        accounts[["account_id", "cutoff"]], on="account_id", how="left")
    statements["payment_due_date"] = pd.to_datetime(statements.payment_due_date)
    statements["min_eff"] = np.minimum(statements.minimum_due, statements.billed_amount)

    eligible = alloc[["payment_id", "statement_id", "allocation_date", "amount_usd"]].merge(
        card, on="payment_id", how="inner")
    eligible["allocation_date"] = pd.to_datetime(eligible.allocation_date)
    eligible = eligible.merge(statements[["statement_id", "cutoff"]],
                              on="statement_id", how="left")
    eligible = eligible[(eligible.payment_date <= eligible.cutoff)
                        & (eligible.allocation_date <= eligible.cutoff)]
    covered = eligible.groupby("statement_id").amount_usd.sum()
    statements["covered"] = statements.statement_id.map(covered).fillna(0.0)

    overdue = statements[(statements.payment_due_date <= statements.cutoff)
                         & (statements.min_eff > 0.01)
                         & (statements.covered < statements.min_eff - 0.01)]
    oldest = overdue.groupby("account_id").payment_due_date.min()
    expected = pd.Series(0, index=acct.account_id, dtype="int64")
    if not oldest.empty:
        days = (accounts.set_index("account_id").cutoff.loc[oldest.index]
                - oldest).dt.days.clip(lower=0).astype("int64")
        expected.loc[days.index] = days
    return expected


def source_percentiles() -> pd.DataFrame:
    """Pull the PHP source amount percentiles for the 1:1 reconciliation table."""
    try:
        files = sorted(C.SOURCE_DIR.glob("bnpl_purchase_[0-9]*.parquet"))
        if not files:
            return pd.DataFrame()
        df = pd.concat([pq.read_table(f, columns=["AMOUNT"]).to_pandas() for f in files])
        s = pd.to_numeric(df.AMOUNT, errors="coerce").dropna()
        qs = [0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        return pd.DataFrame({
            "percentile": [f"p{int(q*100)}" for q in qs],
            "source": [round(float(s.quantile(q)), 2) for q in qs],
        })
    except Exception as e:
        log(f"  source percentiles unavailable: {e}")
        return pd.DataFrame()


def run(dims: dict, facts: dict, raw: dict) -> str:
    rep = Report()
    tx = facts["fact_transaction"]
    stmt = facts["fact_statement"]
    pay = facts["fact_payment"]
    alloc = facts["fact_payment_allocation"]
    ialloc = facts["fact_installment_allocation"]
    plan = facts["fact_installment_plan"]
    sched = facts["fact_installment_schedule"]
    lim = facts["fact_credit_limit_event"]
    delinq = facts["fact_delinquency_snapshot"]
    fraud = facts["fact_fraud_alert"]
    ev = facts["fact_app_event"]
    acct = dims["dim_account"]
    cust = dims["dim_customer"]
    merch = dims["dim_merchant"]
    geo = dims["dim_location"]
    date = dims["dim_date"]

    # ---------------- volumes ----------------
    for name, df in [("dim_customer", cust), ("dim_account", acct), ("dim_card", dims["dim_card"]),
                     ("dim_merchant", merch), ("dim_location", geo), ("dim_date", date),
                     ("fact_transaction", tx), ("fact_statement", stmt), ("fact_payment", pay), ("fact_payment_allocation", alloc),
                     ("fact_installment_allocation", ialloc),
                     ("fact_installment_plan", plan), ("fact_installment_schedule", sched),
                     ("fact_credit_limit_event", lim), ("fact_delinquency_snapshot", delinq),
                     ("fact_fraud_alert", fraud), ("fact_app_event", ev)]:
        rep.check(f"Row count: {name}", None, f"{len(df):,} rows", level="INFO")

    # ---------------- primary keys ----------------
    for name, df, key in [
        ("dim_customer", cust, "customer_id"), ("dim_account", acct, "account_id"),
        ("dim_card", dims["dim_card"], "card_id"), ("dim_merchant", merch, "merchant_id"),
        ("dim_location", geo, "geo_id"), ("dim_date", date, "date_key"),
        ("fact_transaction", tx, "transaction_id"), ("fact_statement", stmt, "statement_id"),
        ("fact_payment", pay, "payment_id"),
        ("fact_payment_allocation", alloc, "allocation_id"),
        ("fact_installment_allocation", ialloc, "allocation_id"), ("fact_installment_plan", plan, "plan_id"),
        ("fact_installment_schedule", sched, "schedule_id"),
        ("fact_fraud_alert", fraud, "alert_id"), ("fact_app_event", ev, "event_id"),
    ]:
        if df.empty:
            rep.check(f"PK unique: {name}", None, "empty table", "WARN")
            continue
        dup = int(df[key].duplicated().sum())
        nul = int(df[key].isna().sum())
        rep.check(f"PK unique: {name}", dup == 0 and nul == 0, f"duplicates={dup}, nulls={nul}")

    # ---------------- referential integrity ----------------
    def fk(child, ck, parent, pk, label):
        if child.empty:
            rep.check(f"FK: {label}", None, "child empty", "WARN")
            return
        orphans = int((~child[ck].isin(set(parent[pk]))).sum())
        rep.check(f"FK: {label}", orphans == 0, f"orphans={orphans}")

    fk(tx, "account_id", acct, "account_id", "transaction.account -> account")
    fk(tx, "customer_id", cust, "customer_id", "transaction.customer -> customer")
    fk(tx, "card_id", dims["dim_card"], "card_id", "transaction.card -> card")
    if tx.merchant_id.notna().any():
        fk(tx, "merchant_id", merch, "merchant_id", "transaction.merchant -> merchant")
    fk(tx, "statement_id", stmt, "statement_id", "transaction.statement -> statement")
    fk(stmt, "account_id", acct, "account_id", "statement.account -> account")
    fk(pay, "account_id", acct, "account_id", "payment.account -> account")
    fk(alloc, "payment_id", pay, "payment_id", "allocation.payment -> payment")
    fk(alloc, "statement_id", stmt, "statement_id", "allocation.statement -> statement")
    fk(ialloc, "payment_id", pay, "payment_id", "instalment_allocation.payment -> payment")
    fk(ialloc, "schedule_id", sched, "schedule_id", "instalment_allocation.schedule -> schedule")
    fk(plan, "account_id", acct, "account_id", "plan.account -> account")
    fk(plan, "transaction_id", tx, "transaction_id", "plan.transaction -> transaction")
    fk(sched, "plan_id", plan, "plan_id", "schedule.plan -> plan")
    fk(lim, "account_id", acct, "account_id", "limit_event.account -> account")
    fk(delinq, "account_id", acct, "account_id", "delinquency.account -> account")
    fk(fraud, "transaction_id", tx, "transaction_id", "fraud.transaction -> transaction")
    fk(fraud, "customer_id", cust, "customer_id", "fraud.customer -> customer")
    fk(ev, "customer_id", cust, "customer_id", "app_event.customer -> customer")
    fk(acct, "customer_id", cust, "customer_id", "account.customer -> customer")
    fk(cust, "home_geo_id", geo, "geo_id", "customer.home_geo -> location")

    # ---------------- date coverage ----------------
    dmin = pd.to_datetime(tx.transaction_date).min().date()
    dmax = pd.to_datetime(tx.transaction_date).max().date()
    rep.check("Transaction date coverage", dmin >= C.PERIOD_START and dmax <= C.PERIOD_END,
              f"{dmin} .. {dmax}")
    months = sorted(pd.to_datetime(stmt.statement_date).dt.strftime("%Y-%m").unique())
    rep.check("Statements cover every month", len(months) == 13, f"{len(months)} months: {months[0]}..{months[-1]}")

    # ---------------- currency & geography ----------------
    fx_bad = int((tx.currency_code == "PHP").sum())
    rep.check("No PHP transactions remain", fx_bad == 0, f"PHP rows={fx_bad}")
    usd_share = float((tx.currency_code == "USD").mean())
    rep.check("USD share of transactions", usd_share > 0.90, f"{usd_share:.1%} USD, {1-usd_share:.1%} FX")
    us_states = set(geo.state_code.dropna().unique())
    cust_in_us = float(cust.state_code.isin(us_states).mean()) if len(us_states) else 0
    rep.check("All customers in US locations", cust_in_us == 1.0, f"{cust_in_us:.1%} in {len(us_states)} states")
    ph = int((geo.country_code == "PH").sum())
    intl_geo_n = int((geo.country_code != "US").sum())
    rep.check("dim_location has no PH rows", ph == 0,
              f"PH rows={ph}, international rows={intl_geo_n}")
    ph_merch = int((merch.country_code == "PH").sum())
    rep.check("dim_merchant has no PH merchants", ph_merch == 0, f"PH merchants={ph_merch}")

    # ---------------- amount distribution (US card norms) ----------------
    p = {q: _pct(tx.amount_usd, q) for q in (0.25, 0.5, 0.75, 0.95, 0.99)}
    med = p[0.5]
    rep.check("Median ticket at US norm",
              0.72 * C.US_TICKET_MEDIAN <= med <= 1.3 * C.US_TICKET_MEDIAN,
              f"synthetic=${med} vs target=${C.US_TICKET_MEDIAN}")
    rep.check("p95 ticket at US norm",
              0.7 * C.US_TICKET_P95 <= p[0.95] <= 1.4 * C.US_TICKET_P95,
              f"synthetic=${p[0.95]} vs target=${C.US_TICKET_P95}")
    rep.check("Amount tail present", float(tx.amount_usd.max()) > 2000,
              f"max=${tx.amount_usd.max():,.0f}")
    rep.check("No negative purchases", int((tx.loc[tx.transaction_type == "PURCHASE", "amount_usd"] < 0).sum()) == 0,
              "purchase sign convention")

    # ---------------- authorisation mix ----------------
    status_mix = tx.status.value_counts(normalize=True).to_dict()
    appr = status_mix.get("APPROVED", 0)
    rep.check("Approval rate 80-98%", 0.80 <= appr <= 0.98,
              f"{appr:.1%} approved (heavy-user segments push limit declines up)")
    repay = float((tx.transaction_type == "REFUND").mean())
    rep.check("Refund share 0.5-4%", 0.005 <= repay <= 0.04, f"{repay:.2%}")

    pur_tx = tx[tx.transaction_type == "PURCHASE"]
    per_stmt = pur_tx.groupby("statement_id").size().rename("n").reset_index()
    per_stmt = per_stmt.merge(stmt[["statement_id", "account_id", "month_key"]],
                              on="statement_id", how="left")
    monthly = per_stmt.n

    rep.check("Median transactions per account-month ~15",
              12.0 <= float(monthly.median()) <= 18.0,
              f"median={float(monthly.median()):.0f} per account-month")
    mean_freq = float(len(pur_tx) / max(1, len(stmt)))
    rep.check("Mean transactions per account per month plausible",
              10.0 <= mean_freq <= 26.0, f"mean={mean_freq:.2f}/month")

    per_acct = (pur_tx.groupby("account_id").size()
                / stmt.groupby("account_id").size()).dropna()
    seg = acct[["account_id", "customer_id"]].merge(
        cust[["customer_id", "customer_segment"]], on="customer_id", how="left")
    f = per_acct.rename("freq").reset_index().merge(seg, on="account_id", how="left")
    by_seg = f.groupby("customer_segment")["freq"].median().sort_values()
    spread = (float(by_seg.max()) / float(by_seg.min())) if len(by_seg) > 1 else 0
    rep.check("Behavioural segments have distinct frequencies",
              len(by_seg) >= 4 and spread >= 3.0,
              "; ".join(f"{k}={v:.1f}" for k, v in by_seg.items()))

    rep.check("Monthly counts overdispersed (p95 >= 25)",
              float(monthly.quantile(0.95)) >= 25,
              f"p50={monthly.median():.0f}, p95={monthly.quantile(0.95):.0f}, "
              f"p99={monthly.quantile(0.99):.0f}")
    rep.check("High-volume outlier months exist (max >= 80)",
              int(monthly.max()) >= 80, f"max={int(monthly.max())} transactions in one month")
    rep.check("Low-activity months exist (>=5% of months <= 5 txns)",
              float((monthly <= 5).mean()) >= 0.05, f"{(monthly <= 5).mean():.1%} of account-months")

    # ---------------- balance reconciliation ----------------
    last = stmt.sort_values("statement_date").groupby("account_id").tail(1).set_index("account_id")
    # vectorised: payments made after each account's last statement date
    p = pay.loc[pay.payment_type == "CARD", ["account_id", "payment_date", "amount_usd"]].copy()
    p["payment_date"] = pd.to_datetime(p.payment_date)
    p = p.merge(last[["statement_date"]], left_on="account_id", right_index=True, how="inner")
    p = p[p.payment_date > pd.to_datetime(p.statement_date)]
    paid_after = p.groupby("account_id")["amount_usd"].sum()
    est = last.closing_balance - paid_after.reindex(last.index).fillna(0.0)
    act = acct.set_index("account_id").current_balance.reindex(last.index).fillna(0.0)
    diffs = (est - act).abs().to_numpy()
    within = float((diffs < 1.0).mean()) if len(diffs) else 1.0
    rep.check("Balance reconciliation (current = last statement - payments)", within > 0.98,
              f"{within:.1%} of accounts within $1, max diff ${diffs.max():.2f}")

    # ---------------- statement arithmetic ----------------
    calc = (stmt.opening_balance + stmt.purchases_amount + stmt.refunds_amount
            + stmt.cash_advances_amount + stmt.interest_charged + stmt.fees_charged
            - stmt.payments_amount)
    delta = (calc - stmt.closing_balance).abs()
    rep.check("Statement arithmetic reconciles", float(delta.max()) < 0.01,
              f"max delta=${delta.max():.2f}")
    rep.check("Statement minimum due <= closing",
              bool((stmt.minimum_due <= stmt.closing_balance.clip(lower=0) + 0.01).all()),
              "min due capped at balance")

    # ---------------- independent reconciliation: transactions <-> statements
    pur_ok = tx[(tx.transaction_type == "PURCHASE") & (tx.status == "APPROVED")
                & (~tx.is_installment_financed)]
    agg_p = pur_ok.groupby("statement_id").amount_usd.sum().rename("tx_purch")
    cmp_p = stmt.set_index("statement_id")[["purchases_amount"]].join(agg_p, how="left").fillna(0.0)
    dp = (cmp_p.purchases_amount - cmp_p.tx_purch).abs()
    rep.check("Statements reconcile to underlying purchases", float(dp.max()) < 0.01,
              f"max delta=${dp.max():.2f}")

    agg_r = tx[tx.transaction_type == "REFUND"].groupby("statement_id").amount_usd.sum().rename("tx_ref")
    cmp_r = stmt.set_index("statement_id")[["refunds_amount"]].join(agg_r, how="left").fillna(0.0)
    drr = (cmp_r.refunds_amount - cmp_r.tx_ref).abs()
    rep.check("Statements reconcile to underlying refunds", float(drr.max()) < 0.01,
              f"max delta=${drr.max():.2f}")

    zero = stmt[(stmt.transaction_count == 0) &
                ((stmt.purchases_amount.abs() > 0.01) | (stmt.refunds_amount.abs() > 0.01))]
    rep.check("No zero-activity statement carries activity", len(zero) == 0, f"{len(zero)} rows")

    agg_alloc = alloc.groupby("statement_id").amount_usd.sum().rename("alloc_sum")
    cmp_pay = stmt.set_index("statement_id")[["paid_amount"]].join(agg_alloc, how="left").fillna(0.0)
    dpaid = (cmp_pay.paid_amount - cmp_pay.alloc_sum).abs()
    rep.check("Statement paid_amount equals allocated payments", float(dpaid.max()) < 0.01,
              f"max delta=${dpaid.max():.2f}")
    over = int((stmt.paid_amount > stmt.billed_amount + 0.01).sum())
    rep.check("No statement allocated beyond its billed amount", over == 0, f"{over} rows")

    pf = stmt[stmt.payment_status == "PAID_FULL"]
    short = int((pf.paid_closing_amount < pf.closing_balance - 0.01).sum())
    rep.check("PAID_FULL statements have their closing balance settled", short == 0,
              f"{short} of {len(pf):,} PAID_FULL statements short")
    excess = int((stmt.paid_closing_amount > stmt.closing_balance.clip(lower=0) + 0.01).sum())
    rep.check("Settled amount never exceeds closing balance", excess == 0, f"{excess} rows")

    card_pay = pay[pay.payment_type == "CARD"]
    orphan_pay = int((~card_pay.applied_statement_id.isin(stmt.statement_id)).sum())
    rep.check("Every payment targets an existing statement", orphan_pay == 0, f"orphans={orphan_pay}")

    st_min = stmt[["statement_id", "account_id", "minimum_due", "billed_amount", "payment_due_date"]].copy()
    st_min["min_eff"] = np.minimum(st_min.minimum_due, st_min.billed_amount)
    al = alloc[["statement_id", "allocation_date", "amount_usd"]].sort_values(
        ["statement_id", "allocation_date"])
    al["cum"] = al.groupby("statement_id")["amount_usd"].cumsum()
    al = al.merge(st_min[["statement_id", "min_eff"]], on="statement_id", how="left")
    covered = (al[al.cum >= al.min_eff - 0.01]
               .groupby("statement_id")["allocation_date"].min())
    st_min["min_paid_date"] = st_min.statement_id.map(covered)
    st_min.loc[st_min.min_eff <= 0.01, "min_paid_date"] = pd.Timestamp("1970-01-01")
    st_min["min_paid_date"] = st_min.min_paid_date.fillna(pd.Timestamp("2099-01-01"))
    snap = delinq[["snapshot_id", "account_id", "snapshot_date", "days_past_due"]]
    mm = snap.merge(st_min, on="account_id", how="inner")
    mm = mm[(mm.payment_due_date < mm.snapshot_date) & (mm.min_eff > 0.01)]
    unpaid = mm[mm.min_paid_date > mm.snapshot_date]
    if len(unpaid):
        oldest = unpaid.groupby("snapshot_id").payment_due_date.min().rename("oldest_due")
        chk = snap.set_index("snapshot_id").join(oldest)
        chk["recalc"] = (pd.to_datetime(chk.snapshot_date) - pd.to_datetime(chk.oldest_due)).dt.days
        chk["recalc"] = chk["recalc"].fillna(0).clip(lower=0)
        diff = (chk.days_past_due - chk.recalc).abs()
        rep.check("Delinquency snapshot matches payment waterfall", float(diff.max()) <= 1,
                  f"max DPD mismatch {diff.max():.0f} days")
    else:
        rep.check("Delinquency snapshot matches payment waterfall", True, "no delinquency")


    # ---------------- cash application completeness ----------------
    ps = alloc.groupby("payment_id").amount_usd.sum().rename("alloc_sum")
    cpay = card_pay.set_index("payment_id")[["amount_usd", "unapplied_amount"]].join(ps, how="left").fillna(0.0)
    dcash = (cpay.amount_usd - cpay.alloc_sum - cpay.unapplied_amount).abs()
    rep.check("Payment = allocations + unapplied cash", float(dcash.max()) < 0.01,
              f"max delta=${dcash.max():.2f}")
    no_alloc = card_pay[~card_pay.payment_id.isin(set(alloc.payment_id))]
    bad_no_alloc = float((no_alloc.amount_usd - no_alloc.unapplied_amount).abs().max()) if len(no_alloc) else 0.0
    rep.check("Payments without allocations are fully unapplied", bad_no_alloc < 0.01,
              f"{len(no_alloc)} fully-unapplied payments, max delta ${bad_no_alloc:.2f}")

    # ---------------- lifecycle edges ----------------
    closed_acc = acct[acct.account_status.isin(["CLOSED", "CHARGED_OFF"])][["account_id", "closed_date"]]
    if len(closed_acc):
        mp = pay.merge(closed_acc, on="account_id", how="inner")
        after_close_pay = int((mp.payment_date > mp.closed_date).sum())
    else:
        after_close_pay = 0
    rep.check("No payment after account closure", after_close_pay == 0, f"{after_close_pay} rows")

    cmerge = dims["dim_card"][["card_id", "account_id", "issue_date"]].merge(
        acct[["account_id", "closed_date"]], on="account_id", how="left")
    cards_after = int((cmerge.closed_date.notna() & (cmerge.issue_date > cmerge.closed_date)).sum())
    rep.check("No card issued after account closure", cards_after == 0, f"{cards_after} cards")

    # ---------------- dimension coverage ----------------
    bad_sched_dates = int((~sched.date_key.isin(set(dims["dim_date"].date_key))).sum())
    rep.check("Every instalment date exists in dim_date", bad_sched_dates == 0,
              f"{bad_sched_dates} rows outside dim_date")

    null_geo_merch = int(dims["dim_merchant"].geo_id.isna().sum())
    rep.check("Every merchant has a location", null_geo_merch == 0, f"{null_geo_merch} merchants")
    null_geo_tx = int(tx.merchant_geo_id.isna().sum())
    rep.check("Every transaction has a merchant location", null_geo_tx == 0, f"{null_geo_tx} rows")

    # ---------------- raw layer referential integrity ----------------
    if "card_purchase_schedule" in raw and "card_purchase" in raw:
        rp = raw["card_purchase"]["ID"]
        rs = raw["card_purchase_schedule"]["PURCHASE_ID"]
        raw_orphans = int((~rs.isin(set(rp))).sum())
        rep.check("Raw schedule references a sampled raw purchase", raw_orphans == 0,
                  f"{raw_orphans} orphan rows")

    neg_no_bal = int(((stmt.payment_status == "NO_BALANCE") & (stmt.closing_balance < -0.01)).sum())
    rep.check("Credit balances are not labelled NO_BALANCE", neg_no_bal == 0, f"{neg_no_bal} rows")

    exp = dims["dim_card"][["card_id", "expiry_year", "expiry_month"]].copy()
    exp["expiry_end"] = pd.to_datetime(dict(year=exp.expiry_year, month=exp.expiry_month, day=1)) + pd.offsets.MonthEnd(0)
    em = tx[["transaction_date", "card_id"]].merge(exp[["card_id", "expiry_end"]], on="card_id", how="left")
    after_exp = int((em.transaction_date > em.expiry_end).sum())
    rep.check("No transaction after card expiry", after_exp == 0, f"{after_exp} rows")

    overdue_sched = sched[(sched.due_date <= pd.Timestamp(C.PERIOD_END)) &
                          (sched.paid_amount < sched.amount_due - 0.01)]
    bad_active = int(((plan.plan_status == "ACTIVE") &
                      (plan.plan_id.isin(set(overdue_sched.plan_id)))).sum())
    rep.check("No ACTIVE plan has an overdue unpaid instalment", bad_active == 0,
              f"{bad_active} plans")


    # ---------------- new v6 integrity checks ----------------
    # financed purchases reconcile to plan principal and leave the revolving ledger
    fin = tx[tx.is_installment_financed]
    fin_delta = abs(float(fin.amount_usd.sum()) - float(plan.principal_amount.sum()))
    rep.check("Financed purchases equal plan principal", fin_delta < 1.0,
              "difference " + format(fin_delta, ",.2f"))

    # instalment cash reconciles to the schedule
    ia = ialloc.groupby("schedule_id").amount_usd.sum().rename("cash")
    cmp_i = sched.set_index("schedule_id")[["paid_amount"]].join(ia, how="left").fillna(0.0)
    di = (cmp_i.paid_amount - cmp_i.cash).abs()
    rep.check("Instalment cash equals schedule paid amount", float(di.max()) < 0.01,
              "max delta " + format(float(di.max()), ",.2f"))
    missing_cash = int(((sched.status == "PAID") & (~sched.schedule_id.isin(ialloc.schedule_id))).sum())
    rep.check("Every PAID instalment has a cash payment", missing_cash == 0, str(missing_cash) + " rows")

    # merchant denormalisation and international classification
    mm = tx[["merchant_id", "mcc", "category", "merchant_name", "merchant_geo_id",
             "is_international"]].merge(
        merch[["merchant_id", "mcc", "category", "merchant_name", "geo_id", "country_code"]],
        on="merchant_id", how="left", suffixes=("_tx", "_m"))
    bad_mcc = int((mm.mcc_tx != mm.mcc_m).sum())
    bad_cat = int((mm.category_tx != mm.category_m).sum())
    bad_geo = int((mm.merchant_geo_id != mm.geo_id).sum())
    bad_intl = int((mm.is_international != (mm.country_code != "US")).sum())
    rep.check("Transaction MCC matches merchant", bad_mcc == 0, str(bad_mcc) + " rows")
    rep.check("Transaction category matches merchant", bad_cat == 0, str(bad_cat) + " rows")
    rep.check("Transaction merchant geo matches merchant", bad_geo == 0, str(bad_geo) + " rows")
    rep.check("Transaction international flag matches merchant country", bad_intl == 0,
              str(bad_intl) + " rows")

    # churn lifecycle
    ch = cust[cust.customer_status == "CHURNED"]
    rep.check("Every churned customer has a churn date", int(ch.churn_date.isna().sum()) == 0,
              str(len(ch)) + " churned customers")
    if len(ch):
        ca = acct.merge(ch[["customer_id", "churn_date"]], on="customer_id", how="inner")
        still_open = int((ca.account_status != "CLOSED").sum())
        late_close = int((ca.closed_date.notna() & (ca.closed_date > ca.churn_date)).sum())
        rep.check("All accounts of churned customers are closed", still_open == 0,
                  str(still_open) + " accounts")
        rep.check("Churned accounts close on or before churn", late_close == 0,
                  str(late_close) + " accounts")

    # refund lineage
    ref = tx[tx.transaction_type == "REFUND"]
    ref_missing = int(ref.original_transaction_id.isna().sum())
    ref_bad = int((~ref.original_transaction_id.dropna().isin(set(tx.transaction_id))).sum())
    rep.check("Every refund has a parent transaction", ref_missing == 0, str(ref_missing) + " rows")
    rep.check("Every refund parent exists", ref_bad == 0, str(ref_bad) + " rows")

    # allocation account consistency
    aj = alloc.merge(pay[["payment_id", "account_id"]], on="payment_id",
                     how="left", suffixes=("", "_pay"))
    aj2 = aj.merge(stmt[["statement_id", "account_id"]], on="statement_id",
                   how="left", suffixes=("", "_stmt"))
    bad_acc = int((aj2.account_id != aj2.account_id_pay).sum() + (aj2.account_id != aj2.account_id_stmt).sum())
    rep.check("Allocation account matches payment and statement", bad_acc == 0, str(bad_acc) + " rows")

    # fraud lifecycle
    if not fraud.empty:
        blk = fraud[fraud.decision == "BLOCK"]
        blk_loss = int((blk.loss_amount > 0.01).sum())
        rep.check("Blocked fraud attempts carry no loss", blk_loss == 0, str(blk_loss) + " rows")
        loss_on_approved = int(((fraud.loss_amount > 0) &
                                (~fraud.transaction_id.isin(tx.loc[tx.status == "APPROVED", "transaction_id"]))).sum())
        rep.check("Fraud loss only on settled transactions", loss_on_approved == 0,
                  str(loss_on_approved) + " rows")

    # date coverage for every fact with a date key
    dd_keys = set(dims["dim_date"].date_key)
    for nm, dfx in [("transaction", tx), ("statement", stmt), ("payment", pay),
                    ("instalment_schedule", sched)]:
        miss = int((~dfx.date_key.isin(dd_keys)).sum())
        rep.check("Date keys covered: " + nm, miss == 0, str(miss) + " rows")

    # payment typing
    inst_pay = pay[pay.payment_type == "INSTALLMENT"]
    card_pay = pay[pay.payment_type == "CARD"]
    rep.check("Instalment payments are not applied to statements",
              int(inst_pay.applied_statement_id.notna().sum()) == 0,
              str(len(inst_pay)) + " instalment payments")
    rep.check("Card payments are typed correctly", int(card_pay.applied_statement_id.isna().sum()) == 0,
              str(len(card_pay)) + " card payments")


    # ---------------- credit balances and identity hygiene ----------------
    neg_close = stmt[stmt.closing_balance < -0.01]
    bad_neg = int((neg_close.payment_status != "CREDIT_BALANCE").sum())
    rep.check("Negative-closing statements are CREDIT_BALANCE", bad_neg == 0,
              str(bad_neg) + " of " + str(len(neg_close)))
    pos_cb = stmt[stmt.payment_status == "CREDIT_BALANCE"]
    bad_pos = int((pos_cb.closing_balance >= -0.01).sum())
    rep.check("CREDIT_BALANCE only on negative closings", bad_pos == 0, str(bad_pos) + " rows")
    cb_delta = float((acct.credit_balance - (-acct.current_balance).clip(lower=0)).abs().max())
    rep.check("dim_account.credit_balance reconciles", cb_delta < 0.01,
              "max delta " + format(cb_delta, ",.2f"))
    bad_min = int((stmt.loc[stmt.closing_balance <= 0, "minimum_due"].abs() > 0.01).sum())
    rep.check("No minimum due on non-positive balances", bad_min == 0, str(bad_min) + " rows")

    appr_tx = tx[tx.status == "APPROVED"]
    bad_auth = int((appr_tx.auth_code.astype(str).str.len() != 6).sum())
    rep.check("Auth codes are six digits on approvals", bad_auth == 0, str(bad_auth) + " rows")

    dup_email = int(cust.email.duplicated().sum())
    dup_phone = int(cust.phone.duplicated().sum())
    rep.check("Customer emails are unique", dup_email == 0, str(dup_email) + " duplicates")
    rep.check("Customer phones are unique", dup_phone == 0, str(dup_phone) + " duplicates")

    # applied statement must be among the payment's allocations (or fully unapplied)
    alloc_map = alloc.groupby("payment_id")["statement_id"].agg(set)
    mism = 0
    for pid, sid in card_pay.set_index("payment_id")["applied_statement_id"].items():
        if pid in alloc_map.index and sid not in alloc_map.loc[pid]:
            mism += 1
    rep.check("Applied statement agrees with allocation", mism == 0, str(mism) + " payments")


    # ---------------- credit limit event history ----------------
    if not lim.empty:
        dir_bad = int(((lim.event_type == "CREDIT_LIMIT_INCREASE") & (lim.new_limit <= lim.old_limit)).sum()
                      + ((lim.event_type == "CREDIT_LIMIT_DECREASE") & (lim.new_limit >= lim.old_limit)).sum())
        rep.check("Limit event direction matches type", dir_bad == 0, str(dir_bad) + " rows")
        last_ev = lim.sort_values("event_date").groupby("account_id").tail(1)
        le = last_ev.merge(acct[["account_id", "credit_limit"]], on="account_id", how="left")
        rec_bad = int((le.credit_limit != le.new_limit).sum())
        rep.check("Account limit equals last limit event", rec_bad == 0, str(rec_bad) + " accounts")
        sd = stmt.merge(lim[["account_id", "event_date", "old_limit"]],
                        left_on=["account_id", "statement_date"],
                        right_on=["account_id", "event_date"], how="inner")
        sd_bad = int((sd.credit_limit != sd.old_limit).sum())
        rep.check("Statement limit equals pre-event limit on event dates", sd_bad == 0, str(sd_bad) + " rows")

    # ---------------- temporal / lifecycle integrity ----------------
    card_tbl = dims["dim_card"][["card_id", "issue_date", "replaced_date"]]
    cm = tx[["transaction_date", "card_id"]].merge(card_tbl, on="card_id", how="left")
    before_issue = int((cm.transaction_date < cm.issue_date).sum())
    after_repl = int(((cm.replaced_date.notna()) & (cm.transaction_date > cm.replaced_date)).sum())
    rep.check("No transaction before card issue", before_issue == 0, f"{before_issue} rows")
    rep.check("No transaction after card replacement", after_repl == 0, f"{after_repl} rows")

    ac = acct[["account_id", "closed_date"]]
    am = tx[["transaction_date", "account_id"]].merge(ac, on="account_id", how="left")
    after_close = int((am.closed_date.notna() & (am.transaction_date > am.closed_date)).sum())
    rep.check("No transaction after account closure", after_close == 0, f"{after_close} rows")

    churned = cust[cust.churn_date.notna()][["customer_id", "churn_date"]]
    cm2 = tx[["transaction_date", "customer_id"]].merge(churned, on="customer_id", how="inner")
    after_churn = int((cm2.transaction_date > cm2.churn_date).sum())
    rep.check("No transaction after customer churn", after_churn == 0, f"{after_churn} rows")

    co = acct[acct.account_status == "CHARGED_OFF"]
    rep.check("Charge-offs carry a closed date", bool(co.closed_date.notna().all()) if len(co) else True,
              f"{len(co)} charge-offs")

    # ---------------- installments ----------------
    bad_paid = int(((sched.paid_amount > 0) & (sched.paid_date.isna())).sum())
    rep.check("Installments with a payment have a paid date", bad_paid == 0, f"{bad_paid} rows")
    pending_due = int(((sched.status == "PENDING") & (sched.due_date <= pd.Timestamp(C.PERIOD_END))).sum())
    rep.check("No instalment past due is still PENDING", pending_due == 0, f"{pending_due} rows")
    plan_bad = int(((plan.plan_status == "COMPLETED") & (plan.remaining_balance > 1)).sum())
    rep.check("Completed plans have zero remaining balance", plan_bad == 0, f"{plan_bad} plans")
    active_complete = int(((plan.plan_status == "ACTIVE") & (plan.remaining_balance <= 1)).sum())
    rep.check("No fully repaid plan is left ACTIVE", active_complete == 0, f"{active_complete} plans")


    # ---------------- payment behaviour ----------------
    smix = stmt.payment_status.value_counts(normalize=True).to_dict()
    rep.check("Statement payment status mix populated", len(smix) >= 3, str(smix))

    # ---------------- delinquency ----------------
    bk = delinq.delinquency_bucket.value_counts(normalize=True).to_dict()
    d30 = sum(v for k, v in bk.items() if k not in ("CURRENT", "1-29"))
    rep.check("30+ DPD share < 8%", d30 < 0.08, f"{d30:.2%} of snapshots")
    rep.check("Snapshot count matches statements", len(delinq) == len(stmt),
              f"{len(delinq)} snapshots vs {len(stmt)} statements")

    # ---------------- fraud ----------------
    f_rate = len(fraud) / max(1, int((tx.status == "APPROVED").sum()))
    rep.check("Fraud alert rate 0.1-0.8%", 0.001 <= f_rate <= 0.008, f"{f_rate:.2%} of approved txns")
    if not fraud.empty:
        conf = float((fraud.case_status == "CONFIRMED_FRAUD").mean())
        rep.check("Confirmed fraud share 20-55%", 0.20 <= conf <= 0.55, f"{conf:.1%}")
        loss = float(fraud.net_loss.sum())
        rep.check("Net fraud loss reported", loss >= 0, f"${loss:,.0f}")


    # ---------------- credit policy staging ----------------
    def _stage_ref(x):
        if x <= 0:
            return "CURRENT"
        if x <= 60:
            return "PAST_DUE"
        if x <= 180:
            return "DELINQUENT"
        return "CHARGE_OFF"

    if "delinquency_status" in delinq.columns:
        mism = int((delinq.delinquency_status != delinq.days_past_due.map(_stage_ref)).sum())
        rep.check("Delinquency stage matches DPD policy (60/180)", mism == 0, f"{mism} rows")
        d60 = float((delinq.delinquency_status.isin(["DELINQUENT", "CHARGE_OFF"])).mean())
        rep.check("60+ DPD delinquent share < 5%", d60 < 0.05, f"{d60:.2%} of snapshots")
    co_ok = int(((acct.account_status == "CHARGED_OFF") & (acct.days_past_due <= 180)).sum())
    rep.check("No charge-off at or below 180 DPD", co_ok == 0, f"{co_ok} accounts")

    # ---------------- installment plans ----------------
    plan_share = len(plan) / max(1, int((tx.transaction_type == "PURCHASE").sum()))
    rep.check("Pay-over-time penetration 0.5-5%", 0.005 <= plan_share <= 0.05, f"{plan_share:.2%} of purchases")
    rep.check("Every plan has schedule rows", set(plan.plan_id) == set(sched.plan_id),
              f"{sched.plan_id.nunique()} plans with schedules")

    # ---------------- dimension sanity ----------------
    rep.check("Unique customers", cust.customer_id.nunique() == len(cust) == C.N_CUSTOMERS,
              f"{len(cust):,} customers")
    seg_share = cust.customer_segment.value_counts(normalize=True)
    ok_seg = all(abs(seg_share.get(s["segment"], 0) - s["share"]) < 0.02
                 for s in C.CUSTOMER_SEGMENTS)
    rep.check("Segment shares match configuration", ok_seg,
              ", ".join(f"{k} {v:.1%}" for k, v in seg_share.items()))
    nets = set(acct.network.dropna().unique())
    rep.check("All three card networks present",
              {"Visa", "Mastercard", "American Express"}.issubset(nets),
              ", ".join(sorted(nets)))
    rep.check("Accounts per customer", 1.0 <= len(acct) / len(cust) <= 1.5,
              f"{len(acct)/len(cust):.2f}")
    rep.check("Cards per account", 1.0 <= len(dims["dim_card"]) / len(acct) <= 2.5,
              f"{len(dims['dim_card'])/len(acct):.2f}")
    rep.check("Credit limit range", float(acct.credit_limit.min()) >= 2500 and float(acct.credit_limit.max()) <= 160000,
              f"${acct.credit_limit.min():,.0f} - ${acct.credit_limit.max():,.0f} "
              f"(floor reflects risk-driven limit decreases)")
    rep.check("Credit limit median consistent with scaled facility",
              0.65 * C.SOURCE_LIMIT_MEDIAN * C.CREDIT_LIMIT_SCALE
              <= float(acct.credit_limit.median())
              <= 1.45 * C.SOURCE_LIMIT_MEDIAN * C.CREDIT_LIMIT_SCALE,
              f"synthetic=${acct.credit_limit.median():,.0f} vs "
              f"{C.CREDIT_LIMIT_SCALE}x source {C.SOURCE_LIMIT_MEDIAN:,.0f}")
    rep.check("Credit score range", int(cust.credit_score.min()) >= 300 and int(cust.credit_score.max()) <= 850,
              f"{cust.credit_score.min()}-{cust.credit_score.max()}")
    rep.check("Income range plausible", float(cust.income_annual.min()) >= 10000,
              f"${cust.income_annual.min():,.0f} - ${cust.income_annual.max():,.0f}")

    # ---------------- raw mirror reconciliation ----------------
    def near(a, b, tol=0.05):
        return abs(a - b) / max(1, b) <= tol

    rep.check("Raw card_account rows = dim_account",
              near(len(raw["card_account"]), len(acct), 0.001), f"{len(raw['card_account'])} vs {len(acct)}")
    exp_tx = min(len(tx), C.RAW_TXN_CAP)
    rep.check("Raw card_purchase is the documented sample",
              len(raw["card_purchase"]) == exp_tx,
              f"{len(raw['card_purchase']):,} of {len(tx):,} transactions")
    exp_pay = min(len(pay), C.RAW_PAYMENT_CAP) * 3
    rep.check("Raw card_payment_workflow is the documented sample",
              len(raw["card_payment_workflow"]) == exp_pay,
              f"{len(raw['card_payment_workflow']):,} rows ({exp_pay:,} expected)")
    rep.check("Raw card_whitelist rows = dim_account",
              near(len(raw["card_whitelist"]), len(acct), 0.001), f"{len(raw['card_whitelist'])}")

    # ---------------- distribution appendix ----------------
    src = source_percentiles()
    lines = [rep.md()]

    lines += ["", "## Appendix A - transaction amount distribution", "",
              "| Statistic | Synthetic USD |", "|---|---|"]
    s = pd.to_numeric(tx.amount_usd, errors="coerce")
    s = s[s > 0]
    for label, q in [("p25", .25), ("p50", .5), ("p75", .75), ("p90", .9), ("p95", .95),
                     ("p99", .99), ("max", 1.0)]:
        lines.append(f"| {label} | ${s.quantile(q):,.2f} |")
    lines.append(f"| mean | ${s.mean():,.2f} |")
    lines.append(f"| total spend | ${s.sum():,.0f} |")

    if not src.empty:
        syn = {q: _pct(tx.amount_usd, q) for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)}
        lines += ["", "### Comparison with the source (intentionally NOT 1:1)", "",
                  "The source is a Philippine BNPL book priced in PHP. A 1:1 PHP->USD "
                  "carry-over was evaluated and rejected: at the target frequency of 15 "
                  "transactions per account per month it implies roughly $37,000 of monthly "
                  "spend per card, which exceeds a plausible consumer limit and produced "
                  "~50% declines. Amounts are therefore calibrated to US card norms. The "
                  "deltas below are expected and are not a defect.", "",
                  "| Percentile | Source (PHP, unadjusted) | Synthetic USD | Delta |",
                  "|---|---|---|---|"]
        for _, r in src.iterrows():
            q = int(r.percentile[1:]) / 100
            s_val = syn.get(q)
            delta = "" if s_val is None else f"{((s_val - r.source) / r.source * 100):+.1f}%"
            lines.append(f"| {r.percentile} | {r.source:,.2f} | "
                         f"{'' if s_val is None else f'{s_val:,.2f}'} | {delta} |")

    lines += ["", "## Appendix B - volume by month", "",
              "| Month | Transactions | Spend (USD) | Statements | Payments |",
              "|---|---|---|---|---|"]
    txm = tx.assign(m=pd.to_datetime(tx.transaction_date).dt.strftime("%Y-%m"))
    stm = stmt.assign(m=pd.to_datetime(stmt.statement_date).dt.strftime("%Y-%m"))
    pm = pay.assign(m=pd.to_datetime(pay.payment_date).dt.strftime("%Y-%m"))
    for m in sorted(txm.m.unique()):
        t = txm[txm.m == m]
        lines.append(f"| {m} | {len(t):,} | ${t.amount_usd.clip(lower=0).sum():,.0f} | "
                     f"{len(stm[stm.m == m]):,} | {len(pm[pm.m == m]):,} |")

    lines += ["", "## Appendix C - portfolio mix", "",
              "| Dimension | Distribution |", "|---|---|"]
    for label, series in [
        ("Product", acct.product_name.value_counts(normalize=True)),
        ("Risk tier", cust.risk_tier.value_counts(normalize=True)),
        ("Account status", acct.account_status.value_counts(normalize=True)),
        ("Delinquency bucket", delinq.delinquency_bucket.value_counts(normalize=True)),
        ("Channel group", tx.channel_group.value_counts(normalize=True)),
        ("Top MCC", tx.mcc.value_counts(normalize=True).head(8)),
    ]:
        txt = ", ".join(f"{k} {v:.1%}" for k, v in series.items())
        lines.append(f"| {label} | {txt} |")

    # ---- Appendix D: card network mix -------------------------------------
    lines += ["", "## Appendix D - card network mix", "",
              "| Network | Accounts | Credit limit | Outstanding | Approved txns | Spend |",
              "|---|---:|---:|---:|---:|---:|"]
    net_acct = acct.groupby("network").agg(
        accounts=("account_id", "size"),
        limit=("credit_limit", "sum"),
        bal=("current_balance", "sum"))
    if "network" in tx.columns:
        net_tx = (tx[tx.status == "APPROVED"]
                  .groupby("network")
                  .agg(n=("transaction_id", "size"), spend=("amount_usd", "sum")))
    else:
        tmp = tx[tx.status == "APPROVED"][["transaction_id", "amount_usd", "account_id"]].merge(
            acct[["account_id", "network"]], on="account_id", how="left")
        net_tx = tmp.groupby("network").agg(n=("transaction_id", "size"), spend=("amount_usd", "sum"))
    for net in sorted(net_acct.index, key=lambda x: -net_acct.loc[x, "accounts"]):
        a = net_acct.loc[net]
        t = net_tx.loc[net] if net in net_tx.index else None
        lines.append(
            f"| {net} | {int(a.accounts):,} | ${a.limit/1e6:,.1f}M | ${a.bal/1e6:,.2f}M | "
            f"{(int(t.n) if t is not None else 0):,} | ${(t.spend if t is not None else 0)/1e6:,.1f}M |")

    text = "\n".join(lines) + "\n"
    global LAST_FAIL_COUNT
    LAST_FAIL_COUNT = rep.fail_count
    return text
