"""Validation: referential integrity, distributions and reconciliation.

Produces docs/validation_report.md with a PASS/FAIL/WARN line per check plus
distribution tables that compare the synthetic book with the PHP source.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import polars as pl

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


def check_settlement_cash_conservation(stmt: pd.DataFrame, pay: pd.DataFrame):
    """Per-account settlement must not exceed card cash (each dollar once).

    Seeded-defect contract: if a regression restores the v6 suffix-cash reuse,
    ``sum(paid_closing_amount)`` exceeds card cash and this returns ok=False.
    """
    if stmt.empty:
        return True, "no statements"
    card = pay.loc[pay.payment_type == "CARD", ["account_id", "amount_usd"]]
    cash = card.groupby("account_id").amount_usd.sum()
    settled = stmt.groupby("account_id").paid_closing_amount.sum()
    cmp = pd.concat([cash.rename("cash"), settled.rename("settled")], axis=1).fillna(0.0)
    excess = cmp.settled - cmp.cash
    n_over = int((excess > 0.01).sum())
    max_excess = float(excess.max()) if len(excess) else 0.0
    total_cash = float(cmp.cash.sum())
    total_settled = float(cmp.settled.sum())
    ok = n_over == 0 and total_settled <= total_cash + 1.0
    return ok, (f"{n_over} accounts over cash, total settled ${total_settled:,.2f} "
                f"vs cash ${total_cash:,.2f}, max excess ${max_excess:,.2f}")


def check_statement_continuity(stmt: pd.DataFrame):
    """opening(i+1) = closing(i) for every account, max residual < $0.01."""
    if stmt.empty:
        return True, "no statements"
    s = stmt.sort_values(["account_id", "statement_date"]).copy()
    prev_close = s.groupby("account_id")["closing_balance"].shift(1)
    residual = (s.opening_balance - prev_close).abs().dropna()
    max_res = float(residual.max()) if len(residual) else 0.0
    return max_res < 0.01, f"max residual ${max_res:.2f} across {len(residual):,} rolls"


def check_fraud_loss_band(fraud: pd.DataFrame, tx: pd.DataFrame):
    """Net fraud loss must sit in the 3-12 bps band of approved purchase spend."""
    spend = float(tx.loc[(tx.status == "APPROVED")
                         & (tx.transaction_type == "PURCHASE"), "amount_usd"].sum())
    return check_fraud_loss_band_spend(fraud, spend)


def check_fraud_loss_band_spend(fraud: pd.DataFrame, spend: float):
    loss = float(fraud.net_loss.sum()) if not fraud.empty else 0.0
    bps = (loss / spend) if spend > 0 else 0.0
    ok = 0.0003 <= bps <= 0.0012
    return ok, f"net loss ${loss:,.2f} / approved purchase spend ${spend:,.2f} = {bps*10000:.2f} bps"


def check_chargeoff_incidence(acct: pd.DataFrame, minimum: int | None = None):
    """At least ``minimum`` charge-offs, so recovery claims are not measured on 4 rows."""
    minimum = C.CHARGE_OFF_MIN_COUNT if minimum is None else minimum
    n = int((acct.account_status == "CHARGED_OFF").sum())
    return n >= minimum, f"{n} charge-offs (gate >= {minimum})"


def check_accepted_limitations(doc_dir):
    """The auth_code reuse decision must be recorded as an accepted limitation."""
    path = doc_dir / "accepted_limitations.md"
    if not path.exists():
        return False, "docs/accepted_limitations.md missing"
    text = path.read_text(encoding="utf-8")
    ok = "auth_code" in text and "accepted limitation" in text.lower()
    return ok, ("accepted-limitation entry present" if ok else
                "accepted_limitations.md does not record auth_code")


def _rq(M: dict, q: float):
    """Rounded positive-amount quantile from the Polars metrics (as ``_pct``)."""
    v = M.get(f"q{q}")
    return None if v is None else round(float(v), 2)


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
        df = pl.concat([pl.read_parquet(f, columns=["AMOUNT"]) for f in files],
                       how="vertical_relaxed")
        s = pd.to_numeric(df.get_column("AMOUNT").to_pandas(), errors="coerce").dropna()
        qs = [0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        return pd.DataFrame({
            "percentile": [f"p{int(q*100)}" for q in qs],
            "source": [round(float(s.quantile(q)), 2) for q in qs],
        })
    except Exception as e:
        log(f"  source percentiles unavailable: {e}")
        return pd.DataFrame()


# --------------------------------------------------------------------------
# Transaction-fact metrics (Polars, lazy): fact_transaction is never
# materialised in pandas. Each value mirrors the former pandas expression.
# --------------------------------------------------------------------------
def _small(df: pd.DataFrame, cols: list[str], dates: tuple[str, ...] = ()) -> pl.LazyFrame:
    out = pl.from_pandas(df[cols].reset_index(drop=True)).lazy()
    return out.with_columns([pl.col(c).cast(pl.Datetime("ns")) for c in dates])


def _not_in(col: str, parent: pd.Series) -> pl.Expr:
    """1 where ``col`` is not in ``parent`` (null counts as not in, like ``~isin``)."""
    vals = pl.from_pandas(parent.dropna().drop_duplicates().reset_index(drop=True))
    return (~pl.col(col).is_in(vals.implode()).fill_null(False)).cast(pl.Int64)


def _anti(child: pd.DataFrame, key: str, ids: pl.LazyFrame) -> int:
    """Rows of the (small) child whose ``key`` has no match in ``ids`` (column ``id``)."""
    c = _small(child, [key])
    return int(c.join(ids, left_on=key, right_on="id", how="anti")
               .select(pl.len()).collect().item())


def tx_metrics(txl: pl.LazyFrame, *, dims: dict, facts: dict, dd_keys: set) -> dict:
    """Every fact_transaction-derived figure the validation report needs."""
    acct, cust, merch = dims["dim_account"], dims["dim_customer"], dims["dim_merchant"]
    card, stmt = dims["dim_card"], facts["fact_statement"]
    txl = txl.with_columns(pl.col("transaction_date").cast(pl.Datetime("ns")))
    amt = pl.col("amount_usd").cast(pl.Float64, strict=False)
    pos = amt.filter(amt > 0)
    purchase = pl.col("transaction_type") == "PURCHASE"
    approved = pl.col("status") == "APPROVED"
    qs = (0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
    exprs = [
        pl.len().alias("n"),
        pl.col("transaction_id").n_unique().alias("tid_nunique"),
        pl.col("transaction_id").null_count().alias("tid_nulls"),
        _not_in("account_id", acct.account_id).sum().alias("fk_account"),
        _not_in("customer_id", cust.customer_id).sum().alias("fk_customer"),
        _not_in("card_id", card.card_id).sum().alias("fk_card"),
        _not_in("merchant_id", merch.merchant_id).sum().alias("fk_merchant"),
        _not_in("statement_id", stmt.statement_id).sum().alias("fk_statement"),
        pl.col("merchant_id").is_not_null().any().alias("merchant_any"),
        pl.col("transaction_date").min().alias("dmin"),
        pl.col("transaction_date").max().alias("dmax"),
        (pl.col("currency_code") == "PHP").sum().alias("php"),
        (pl.col("currency_code") == "USD").sum().alias("usd"),
        amt.max().alias("amt_max"),
        ((amt < 0) & purchase).sum().alias("neg_purchases"),
        approved.sum().alias("approved"),
        pl.col("status").is_not_null().sum().alias("status_nonnull"),
        (pl.col("transaction_type") == "REFUND").sum().alias("refunds"),
        purchase.sum().alias("purchases"),
        pl.col("merchant_geo_id").null_count().alias("null_geo"),
        amt.filter(pl.col("is_installment_financed")).sum().alias("financed_sum"),
        amt.filter(approved & purchase).sum().alias("approved_purchase_spend"),
        (pl.col("auth_code").cast(pl.String).str.len_chars() != 6).fill_null(True)
        .filter(approved).sum().alias("bad_auth"),
        _not_in("date_key", pd.Series(sorted(dd_keys))).sum().alias("date_key_miss"),
        pos.mean().alias("pos_mean"), pos.sum().alias("pos_sum"), pos.max().alias("pos_max"),
    ] + [pos.quantile(q, interpolation="linear").alias(f"q{q}") for q in qs]
    m = txl.select(exprs).collect(engine="streaming").row(0, named=True)
    m["dmin"], m["dmax"] = m["dmin"].date(), m["dmax"].date()
    m["approved_share"] = m["approved"] / max(m["status_nonnull"], 1)

    def grp(lf, key, agg, name):
        out = lf.group_by(key).agg(agg.alias(name)).filter(pl.col(key).is_not_null())
        return out.collect(engine="streaming").to_pandas()

    pur = txl.filter(purchase)
    m["per_stmt"] = grp(pur, "statement_id", pl.len(), "n")
    m["per_acct"] = grp(pur, "account_id", pl.len(), "n").set_index("account_id")["n"]
    pur_ok = txl.filter(purchase & approved & ~pl.col("is_installment_financed"))
    m["agg_p"] = grp(pur_ok, "statement_id", amt.sum(), "tx_purch").set_index("statement_id")["tx_purch"]
    ref = txl.filter(pl.col("transaction_type") == "REFUND")
    m["agg_r"] = grp(ref, "statement_id", amt.sum(), "tx_ref").set_index("statement_id")["tx_ref"]

    # orphans against transaction_id: anti-joins keep memory at the child's size
    ids = txl.select(pl.col("transaction_id").alias("id"))
    m["plan_orphans"] = _anti(facts["fact_installment_plan"], "transaction_id", ids)
    m["fraud_orphans"] = _anti(facts["fact_fraud_alert"], "transaction_id", ids)
    appr_ids = txl.filter(approved).select(pl.col("transaction_id").alias("id"))
    fr = facts["fact_fraud_alert"]
    m["fraud_loss_unsettled"] = 0 if fr.empty else _anti(
        fr[fr.loss_amount > 0], "transaction_id", appr_ids)
    m["refund_missing"] = int(ref.select(pl.col("original_transaction_id").null_count())
                              .collect().item())
    m["refund_bad"] = int(
        ref.filter(pl.col("original_transaction_id").is_not_null())
        .join(ids, left_on="original_transaction_id", right_on="id", how="anti")
        .select(pl.len()).collect().item())

    exp = card[["card_id", "expiry_year", "expiry_month"]].copy()
    exp["expiry_end"] = (pd.to_datetime(dict(year=exp.expiry_year, month=exp.expiry_month, day=1))
                         + pd.offsets.MonthEnd(0))
    em = txl.join(_small(exp, ["card_id", "expiry_end"], ("expiry_end",)), on="card_id", how="left")
    m["after_exp"] = int(em.select((pl.col("transaction_date") > pl.col("expiry_end")).sum())
                         .collect().item())

    mc = _small(merch, ["merchant_id", "mcc", "category", "merchant_name", "geo_id", "country_code"])
    mm = txl.select("merchant_id", "mcc", "category", "merchant_name", "merchant_geo_id",
                    "is_international").join(mc, on="merchant_id", how="left", suffix="_m")
    bad = mm.select(
        pl.col("mcc").ne_missing(pl.col("mcc_m")).sum().alias("bad_mcc"),
        pl.col("category").ne_missing(pl.col("category_m")).sum().alias("bad_cat"),
        pl.col("merchant_geo_id").ne_missing(pl.col("geo_id")).sum().alias("bad_geo"),
        pl.col("is_international").ne_missing(
            pl.col("country_code").ne_missing(pl.lit("US"))).sum().alias("bad_intl"),
    ).collect(engine="streaming").row(0, named=True)
    m.update(bad)

    ct = _small(card, ["card_id", "issue_date", "replaced_date"], ("issue_date", "replaced_date"))
    cm = txl.join(ct, on="card_id", how="left").select(
        (pl.col("transaction_date") < pl.col("issue_date")).sum().alias("before_issue"),
        (pl.col("replaced_date").is_not_null()
         & (pl.col("transaction_date") > pl.col("replaced_date"))).sum().alias("after_repl"),
    ).collect(engine="streaming").row(0, named=True)
    m.update(cm)
    ac = _small(acct, ["account_id", "closed_date"], ("closed_date",))
    m["after_close"] = int(
        txl.join(ac, on="account_id", how="left")
        .select((pl.col("closed_date").is_not_null()
                 & (pl.col("transaction_date") > pl.col("closed_date"))).sum())
        .collect(engine="streaming").item())
    ch = cust[cust.churn_date.notna()]
    m["after_churn"] = int(
        txl.join(_small(ch, ["customer_id", "churn_date"], ("churn_date",)),
                 on="customer_id", how="inner")
        .select((pl.col("transaction_date") > pl.col("churn_date")).sum())
        .collect(engine="streaming").item())

    m["monthly"] = (
        txl.group_by(pl.col("transaction_date").dt.strftime("%Y-%m").alias("m"))
        .agg(pl.len().alias("n"), amt.clip(lower_bound=0).sum().alias("spend"))
        .filter(pl.col("m").is_not_null()).sort("m")
        .collect(engine="streaming").to_pandas().set_index("m"))

    def mix(col, head=None):
        s = (txl.filter(pl.col(col).is_not_null()).group_by(col).len()
             .sort(["len", col], descending=[True, False]).collect(engine="streaming"))
        tot = s["len"].sum()
        out = pd.Series(s["len"].to_numpy() / tot, index=s[col].to_list())
        return out.head(head) if head else out

    m["channel_mix"], m["mcc_mix"] = mix("channel_group"), mix("mcc", 8)

    cols = txl.collect_schema().names()
    base = txl.filter(approved)
    if "network" not in cols:
        base = base.join(_small(acct, ["account_id", "network"]), on="account_id", how="left")
    m["net_tx"] = (base.filter(pl.col("network").is_not_null()).group_by("network")
                   .agg(pl.len().alias("n"), amt.sum().alias("spend"))
                   .collect(engine="streaming").to_pandas().set_index("network"))
    return m


def run(dims: dict, facts: dict, raw: dict) -> str:
    rep = Report()
    tx_in = facts["fact_transaction"]
    txl = tx_in if isinstance(tx_in, pl.LazyFrame) else pl.from_pandas(tx_in).lazy()
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
    dd_keys = set(date.date_key)
    M = tx_metrics(txl, dims=dims, facts=facts, dd_keys=dd_keys)
    n_tx = M["n"]

    # ---------------- volumes ----------------
    for name, df in [("dim_customer", cust), ("dim_account", acct), ("dim_card", dims["dim_card"]),
                     ("dim_merchant", merch), ("dim_location", geo), ("dim_date", date),
                     ("fact_transaction", range(n_tx)), ("fact_statement", stmt), ("fact_payment", pay), ("fact_payment_allocation", alloc),
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
        ("fact_transaction", None, "transaction_id"), ("fact_statement", stmt, "statement_id"),
        ("fact_payment", pay, "payment_id"),
        ("fact_payment_allocation", alloc, "allocation_id"),
        ("fact_installment_allocation", ialloc, "allocation_id"), ("fact_installment_plan", plan, "plan_id"),
        ("fact_installment_schedule", sched, "schedule_id"),
        ("fact_fraud_alert", fraud, "alert_id"), ("fact_app_event", ev, "event_id"),
    ]:
        if df is None:  # fact_transaction: counted lazily
            if n_tx == 0:
                rep.check(f"PK unique: {name}", None, "empty table", "WARN")
                continue
            dup, nul = n_tx - M["tid_nunique"], M["tid_nulls"]
        elif df.empty:
            rep.check(f"PK unique: {name}", None, "empty table", "WARN")
            continue
        else:
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

    def fk_tx(orphans, label):
        if n_tx == 0:
            rep.check(f"FK: {label}", None, "child empty", "WARN")
        else:
            rep.check(f"FK: {label}", orphans == 0, f"orphans={orphans}")

    fk_tx(M["fk_account"], "transaction.account -> account")
    fk_tx(M["fk_customer"], "transaction.customer -> customer")
    fk_tx(M["fk_card"], "transaction.card -> card")
    if M["merchant_any"]:
        fk_tx(M["fk_merchant"], "transaction.merchant -> merchant")
    fk_tx(M["fk_statement"], "transaction.statement -> statement")
    fk(stmt, "account_id", acct, "account_id", "statement.account -> account")
    fk(pay, "account_id", acct, "account_id", "payment.account -> account")
    fk(alloc, "payment_id", pay, "payment_id", "allocation.payment -> payment")
    fk(alloc, "statement_id", stmt, "statement_id", "allocation.statement -> statement")
    fk(ialloc, "payment_id", pay, "payment_id", "instalment_allocation.payment -> payment")
    fk(ialloc, "schedule_id", sched, "schedule_id", "instalment_allocation.schedule -> schedule")
    fk(plan, "account_id", acct, "account_id", "plan.account -> account")
    if plan.empty:
        rep.check("FK: plan.transaction -> transaction", None, "child empty", "WARN")
    else:
        rep.check("FK: plan.transaction -> transaction", M["plan_orphans"] == 0,
                  f"orphans={M['plan_orphans']}")
    fk(sched, "plan_id", plan, "plan_id", "schedule.plan -> plan")
    fk(lim, "account_id", acct, "account_id", "limit_event.account -> account")
    fk(delinq, "account_id", acct, "account_id", "delinquency.account -> account")
    if fraud.empty:
        rep.check("FK: fraud.transaction -> transaction", None, "child empty", "WARN")
    else:
        rep.check("FK: fraud.transaction -> transaction", M["fraud_orphans"] == 0,
                  f"orphans={M['fraud_orphans']}")
    fk(fraud, "customer_id", cust, "customer_id", "fraud.customer -> customer")
    fk(ev, "customer_id", cust, "customer_id", "app_event.customer -> customer")
    fk(acct, "customer_id", cust, "customer_id", "account.customer -> customer")
    fk(cust, "home_geo_id", geo, "geo_id", "customer.home_geo -> location")

    # ---------------- date coverage ----------------
    dmin, dmax = M["dmin"], M["dmax"]
    rep.check("Transaction date coverage", dmin >= C.PERIOD_START and dmax <= C.PERIOD_END,
              f"{dmin} .. {dmax}")
    months = sorted(pd.to_datetime(stmt.statement_date).dt.strftime("%Y-%m").unique())
    rep.check("Statements cover every month", len(months) == 13, f"{len(months)} months: {months[0]}..{months[-1]}")

    # ---------------- currency & geography ----------------
    fx_bad = int(M["php"])
    rep.check("No PHP transactions remain", fx_bad == 0, f"PHP rows={fx_bad}")
    usd_share = M["usd"] / n_tx
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
    p = {q: _rq(M, q) for q in (0.25, 0.5, 0.75, 0.95, 0.99)}
    med = p[0.5]
    rep.check("Median ticket at US norm",
              0.72 * C.US_TICKET_MEDIAN <= med <= 1.3 * C.US_TICKET_MEDIAN,
              f"synthetic=${med} vs target=${C.US_TICKET_MEDIAN}")
    rep.check("p95 ticket at US norm",
              0.7 * C.US_TICKET_P95 <= p[0.95] <= 1.4 * C.US_TICKET_P95,
              f"synthetic=${p[0.95]} vs target=${C.US_TICKET_P95}")
    rep.check("Amount tail present", float(M["amt_max"]) > 2000,
              f"max=${M['amt_max']:,.0f}")
    rep.check("No negative purchases", int(M["neg_purchases"]) == 0,
              "purchase sign convention")

    # ---------------- authorisation mix ----------------
    appr = M["approved_share"]
    rep.check("Approval rate 80-98%", 0.80 <= appr <= 0.98,
              f"{appr:.1%} approved (heavy-user segments push limit declines up)")
    repay = M["refunds"] / n_tx
    rep.check("Refund share 0.5-4%", 0.005 <= repay <= 0.04, f"{repay:.2%}")

    per_stmt = M["per_stmt"].merge(stmt[["statement_id", "account_id", "month_key"]],
                              on="statement_id", how="left")
    monthly = per_stmt.n

    rep.check("Median transactions per account-month ~15",
              12.0 <= float(monthly.median()) <= 18.0,
              f"median={float(monthly.median()):.0f} per account-month")
    mean_freq = float(M["purchases"] / max(1, len(stmt)))
    rep.check("Mean transactions per account per month plausible",
              10.0 <= mean_freq <= 26.0, f"mean={mean_freq:.2f}/month")

    per_acct = (M["per_acct"] / stmt.groupby("account_id").size()).dropna()
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
    agg_p = M["agg_p"]
    cmp_p = stmt.set_index("statement_id")[["purchases_amount"]].join(agg_p, how="left").fillna(0.0)
    dp = (cmp_p.purchases_amount - cmp_p.tx_purch).abs()
    rep.check("Statements reconcile to underlying purchases", float(dp.max()) < 0.01,
              f"max delta=${dp.max():.2f}")

    agg_r = M["agg_r"]
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

    # v7 A1: settlement is cash-consuming, not suffix-cash reuse
    ok, detail = check_settlement_cash_conservation(stmt, pay)
    rep.check("Settlement consumes card cash at most once", ok, detail)

    # v7 A2: statement continuity
    ok, detail = check_statement_continuity(stmt)
    rep.check("Statement opening equals prior closing", ok, detail)

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
    null_geo_tx = int(M["null_geo"])
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

    after_exp = int(M["after_exp"])
    rep.check("No transaction after card expiry", after_exp == 0, f"{after_exp} rows")

    overdue_sched = sched[(sched.due_date <= pd.Timestamp(C.PERIOD_END)) &
                          (sched.paid_amount < sched.amount_due - 0.01)]
    bad_active = int(((plan.plan_status == "ACTIVE") &
                      (plan.plan_id.isin(set(overdue_sched.plan_id)))).sum())
    rep.check("No ACTIVE plan has an overdue unpaid instalment", bad_active == 0,
              f"{bad_active} plans")


    # ---------------- new v6 integrity checks ----------------
    # financed purchases reconcile to plan principal and leave the revolving ledger
    fin_delta = abs(float(M["financed_sum"]) - float(plan.principal_amount.sum()))
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
    bad_mcc, bad_cat = int(M["bad_mcc"]), int(M["bad_cat"])
    bad_geo, bad_intl = int(M["bad_geo"]), int(M["bad_intl"])
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
    ref_missing, ref_bad = int(M["refund_missing"]), int(M["refund_bad"])
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
        loss_on_approved = int(M["fraud_loss_unsettled"])
        rep.check("Fraud loss only on settled transactions", loss_on_approved == 0,
                  str(loss_on_approved) + " rows")

    # date coverage for every fact with a date key
    for nm, dfx in [("transaction", None), ("statement", stmt), ("payment", pay),
                    ("payment_allocation", alloc), ("instalment_schedule", sched),
                    ("instalment_allocation", ialloc), ("credit_limit_event", lim),
                    ("delinquency_snapshot", delinq), ("fraud_alert", fraud),
                    ("app_event", ev)]:
        miss = (int(M["date_key_miss"]) if dfx is None
                else int((~dfx.date_key.isin(dd_keys)).sum()))
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

    bad_auth = int(M["bad_auth"])
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
    before_issue, after_repl = int(M["before_issue"]), int(M["after_repl"])
    rep.check("No transaction before card issue", before_issue == 0, f"{before_issue} rows")
    rep.check("No transaction after card replacement", after_repl == 0, f"{after_repl} rows")

    after_close = int(M["after_close"])
    rep.check("No transaction after account closure", after_close == 0, f"{after_close} rows")

    after_churn = int(M["after_churn"])
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
    f_rate = len(fraud) / max(1, int(M["approved"]))
    rep.check("Fraud alert rate 0.1-0.8%", 0.001 <= f_rate <= 0.008, f"{f_rate:.2%} of approved txns")
    if not fraud.empty:
        conf = float((fraud.case_status == "CONFIRMED_FRAUD").mean())
        rep.check("Confirmed fraud share 20-55%", 0.20 <= conf <= 0.55, f"{conf:.1%}")
        loss = float(fraud.net_loss.sum())
        rep.check("Net fraud loss reported", loss >= 0, f"${loss:,.0f}")
        ok, detail = check_fraud_loss_band_spend(fraud, float(M["approved_purchase_spend"]))
        rep.check("Fraud net loss within 3-12 bps", ok, detail)


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
    ok, detail = check_chargeoff_incidence(acct)
    rep.check("Charge-off incidence supports structural claims", ok, detail)

    # ---------------- installment plans ----------------
    plan_share = len(plan) / max(1, int(M["purchases"]))
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
    ok, detail = check_accepted_limitations(C.DOC_DIR)
    rep.check("Accepted limitations recorded", ok, detail)

    # ---------------- raw mirror reconciliation ----------------
    def near(a, b, tol=0.05):
        return abs(a - b) / max(1, b) <= tol

    rep.check("Raw card_account rows = dim_account",
              near(len(raw["card_account"]), len(acct), 0.001), f"{len(raw['card_account'])} vs {len(acct)}")
    exp_tx = min(n_tx, C.RAW_TXN_CAP)
    rep.check("Raw card_purchase is the documented sample",
              len(raw["card_purchase"]) == exp_tx,
              f"{len(raw['card_purchase']):,} of {n_tx:,} transactions")
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
    for label, key in [("p25", "q0.25"), ("p50", "q0.5"), ("p75", "q0.75"), ("p90", "q0.9"),
                       ("p95", "q0.95"), ("p99", "q0.99"), ("max", "pos_max")]:
        lines.append(f"| {label} | ${M[key]:,.2f} |")
    lines.append(f"| mean | ${M['pos_mean']:,.2f} |")
    lines.append(f"| total spend | ${M['pos_sum']:,.0f} |")

    if not src.empty:
        syn = {q: _rq(M, q) for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)}
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
    txm = M["monthly"]
    stm = stmt.assign(m=pd.to_datetime(stmt.statement_date).dt.strftime("%Y-%m"))
    pm = pay.assign(m=pd.to_datetime(pay.payment_date).dt.strftime("%Y-%m"))
    for m in txm.index:
        t = txm.loc[m]
        lines.append(f"| {m} | {int(t.n):,} | ${t.spend:,.0f} | "
                     f"{len(stm[stm.m == m]):,} | {len(pm[pm.m == m]):,} |")

    lines += ["", "## Appendix C - portfolio mix", "",
              "| Dimension | Distribution |", "|---|---|"]
    for label, series in [
        ("Product", acct.product_name.value_counts(normalize=True)),
        ("Risk tier", cust.risk_tier.value_counts(normalize=True)),
        ("Account status", acct.account_status.value_counts(normalize=True)),
        ("Delinquency bucket", delinq.delinquency_bucket.value_counts(normalize=True)),
        ("Channel group", M["channel_mix"]),
        ("Top MCC", M["mcc_mix"]),
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
    net_tx = M["net_tx"]
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
