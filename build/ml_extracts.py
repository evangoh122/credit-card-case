"""Build ML-grade extracts under ``<output>/ml`` from the operational parquet.

The operational star schema is never modified. Every table here is a derived
extract designed for specific ML/MLOps use cases:

* ``fraud_*``            - noisy-labelled fraud classification, out-of-time and
                           out-of-entity splits, and labelled test-purchase ->
                           large-purchase sequences
* ``credit_default_*``   - point-in-time account features and a forward default
                           label with an explicit observation/outcome window
* ``churn_*``            - point-in-time customer features and a forward churn
                           label over the same window
* ``segmentation_*``     - RFM and MCC/category share for segmentation
* ``forecast_*``         - merchant-month volume with seasonal features

Run as a blocking subprocess from ``build.run_all`` so the ML pass has its own
memory budget:

    CC_OUT_DIR=credit_card_data_v7 .venv/bin/python -m build.ml_extracts
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from . import config as C
from .utils import log, write_parquet

ML = C.OUT / C.ML_DIR_NAME
RNG = np.random.default_rng(C.SEED + 7007)


# --------------------------------------------------------------------------
# Checks shared with the test suite
# --------------------------------------------------------------------------
def check_no_leakage(df: pd.DataFrame,
                     observation_col: str = "observation_date",
                     max_source_col: str = "max_source_date") -> tuple[bool, str]:
    """Every row's largest source date must not pass its observation date.

    Seeded-defect contract: a feature that reads data after the observation
    date pushes ``max_source_date`` past ``observation_date`` and fails here.
    """
    if df.empty:
        return True, "empty extract"
    for col in (observation_col, max_source_col):
        if col not in df.columns:
            return False, f"missing {col}"
    obs = pd.to_datetime(df[observation_col])
    src = pd.to_datetime(df[max_source_col])
    bad = int((src > obs).sum())
    max_gap = (src - obs).dt.days.max()
    return bad == 0, f"{bad} rows with post-observation features (max +{max_gap} days)"


# Columns that define (or are only known after) the label: never allowed in a
# feature set. They may exist in the source tables and in the label audit only.
LABEL_DEFINING_COLUMNS = {
    "credit_default": ("account_status", "closed_date"),
    "churn": ("customer_status", "churn_date"),
}


def check_no_label_columns(df: pd.DataFrame, task: str) -> tuple[bool, str]:
    """Fail if any label-defining / post-observation column is in the feature set."""
    present = [c for c in LABEL_DEFINING_COLUMNS[task] if c in df.columns]
    return not present, (f"label-defining columns in features: {present}" if present
                         else "no label-defining columns in features")


def check_pr_auc(pr_auc: float, maximum: float | None = None) -> tuple[bool, str]:
    maximum = C.ML_PR_AUC_MAX if maximum is None else maximum
    return pr_auc < maximum, f"PR-AUC {pr_auc:.4f} (gate < {maximum})"


def check_drift(psi: float, minimum: float | None = None) -> tuple[bool, str]:
    minimum = C.ML_DRIFT_PSI_MIN if minimum is None else minimum
    return psi >= minimum, f"PSI {psi:.4f} (gate >= {minimum})"


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------
def _read(table: str, columns: list[str]) -> pd.DataFrame:
    return pl.read_parquet(C.OUT / table, columns=columns).to_pandas()


def _hash_bucket(value: str, mod: int = 5) -> int:
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:8], 16) % mod


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve)."""
    y = np.asarray(y, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    total_pos = max(float(y.sum()), 1.0)
    recall = tp / total_pos
    ap = 0.0
    prev_recall = 0.0
    prev_precision = 1.0
    for r, p in zip(recall, precision):
        if r == prev_recall:
            continue
        ap += (r - prev_recall) * (p + prev_precision) / 2.0
        prev_recall, prev_precision = r, p
    return float(ap)


def population_stability_index(expected: np.ndarray, actual: np.ndarray,
                               bins=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) -> float:
    e = np.histogram(np.clip(expected, 0, 1), bins=bins)[0].astype(float)
    a = np.histogram(np.clip(actual, 0, 1), bins=bins)[0].astype(float)
    e = np.maximum(e / max(e.sum(), 1.0), 1e-6)
    a = np.maximum(a / max(a.sum(), 1.0), 1e-6)
    return float(np.sum((a - e) * np.log(a / e)))


# --------------------------------------------------------------------------
# Transactions
# --------------------------------------------------------------------------
def _load_purchases() -> pd.DataFrame:
    cols = [
        "transaction_id", "account_id", "customer_id", "merchant_id", "mcc",
        "category", "channel_group", "entry_mode", "is_card_present",
        "is_international", "transaction_date", "transaction_ts", "amount_usd",
        "status", "transaction_type", "is_fraud",
    ]
    # filter inside the scan so only approved purchases ever reach pandas
    df = (
        pl.scan_parquet(C.OUT / "fact/fact_transaction.parquet")
        .filter((pl.col("status") == "APPROVED") & (pl.col("transaction_type") == "PURCHASE"))
        .select(cols)
        .collect(engine="streaming")
        .to_pandas()
    )
    df["transaction_date"] = pd.to_datetime(df.transaction_date)
    df["transaction_ts"] = pd.to_datetime(df.transaction_ts)
    df["hour"] = df.transaction_ts.dt.hour.astype("int8")
    df["amount_usd"] = df.amount_usd.astype(float)
    df["is_fraud"] = df.is_fraud.fillna(False).astype(bool)
    return df


def _attach_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Customer-day count and customer-week spend, both point-in-time proxies."""
    day = df.transaction_date.dt.normalize()
    df = df.assign(_day=day, _week=df.transaction_date.dt.to_period("W").astype(str))
    cd = (df.groupby(["customer_id", "_day"], sort=False).size()
            .rename("customer_day_txn_count").reset_index())
    cw = (df.groupby(["customer_id", "_week"], sort=False).amount_usd.sum()
            .rename("customer_week_spend").reset_index())
    first = df.groupby("customer_id", sort=False).transaction_date.min().rename("first_txn_date")
    df = df.merge(cd, on=["customer_id", "_day"], how="left")
    df = df.merge(cw, on=["customer_id", "_week"], how="left")
    df = df.join(first, on="customer_id")
    df["days_since_first_txn"] = (df.transaction_date - df.first_txn_date).dt.days
    return df.drop(columns=["_day", "_week", "first_txn_date"])


def _assign_fraud_type(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Fraud typology with a deliberate regime shift in the final quarter."""
    types = np.full(len(df), "NONE", dtype=object)
    fraud = df.is_fraud.to_numpy()
    n = int(fraud.sum())
    if n == 0:
        return pd.Series(types, index=df.index)
    late = (df.transaction_date >= pd.Timestamp(C.ML_FRAUD_OOT_START)).to_numpy()
    early_idx = np.where(fraud & ~late)[0]
    late_idx = np.where(fraud & late)[0]
    if len(early_idx):
        types[early_idx] = rng.choice(
            ["CARD_NOT_PRESENT", "COUNTERFEIT", "ACCOUNT_TAKEOVER", "LOST_STOLEN",
             "FRIENDLY_FRAUD"],
            size=len(early_idx), p=[0.34, 0.22, 0.18, 0.16, 0.10])
    if len(late_idx):
        # regime shift: final-quarter fraud is dominated by card-not-present
        # and account-takeover attacks
        types[late_idx] = rng.choice(
            ["CARD_NOT_PRESENT", "ACCOUNT_TAKEOVER", "COUNTERFEIT"],
            size=len(late_idx), p=[0.58, 0.30, 0.12])
    return pd.Series(types, index=df.index)


def build_fraud(df: pd.DataFrame) -> dict:
    log("  ml: fraud extracts")
    fraud_true = df[df.is_fraud].copy()
    nonfraud = df[~df.is_fraud]
    take = min(len(nonfraud), C.ML_FRAUD_MAX_ROWS)
    sample = pd.concat(
        [fraud_true, nonfraud.sample(n=take, random_state=C.SEED + 11)],
        ignore_index=True,
    ).sort_values("transaction_date").reset_index(drop=True)

    sample["fraud_type"] = _assign_fraud_type(sample, RNG)
    sample["label_true"] = sample.is_fraud.astype("int8")

    # label noise exists only in the ML copy
    labels = sample.label_true.to_numpy().copy()
    fraud_pos = np.where(labels == 1)[0]
    clean_pos = np.where(labels == 0)[0]
    drop = RNG.random(len(fraud_pos)) < C.ML_FRAUD_LABEL_NOISE_INDISTINGUISHABLE
    add = RNG.random(len(clean_pos)) < C.ML_FRAUD_LABEL_NOISE_FALSE_FLAG
    labels[fraud_pos[drop]] = 0
    labels[clean_pos[add]] = 1
    sample["label_noisy"] = labels.astype("int8")
    sample["label_noise"] = np.where(
        (sample.label_true == 1) & (sample.label_noisy == 0), "INDISTINGUISHABLE",
        np.where((sample.label_true == 0) & (sample.label_noisy == 1), "FALSE_FLAG", "NONE"))

    late = (sample.transaction_date >= pd.Timestamp(C.ML_FRAUD_OOT_START)).to_numpy()
    sample["split_time"] = np.where(late, "test", "train")
    sample["split_entity"] = np.where(
        sample.customer_id.map(lambda x: _hash_bucket(x) == 0).to_numpy(),
        "test_entity", "train_entity")

    # ML-only covariate shift: device risk is elevated in the final quarter, so
    # the drift check has a real signal to detect.
    base_risk = (0.12 + 0.30 * (~sample.is_card_present.astype(bool))
                 + 0.15 * sample.is_international.astype(bool)
                 + 0.10 * (sample.hour <= 5)
                 + 0.10 * sample.category.isin(
                     ["Entertainment", "Electronics", "Digital", "E-Commerce"]))
    shift = 0.42 * late
    sample["device_risk_score"] = np.clip(
        base_risk.to_numpy(float) + shift + RNG.normal(0, 0.05, len(sample)), 0, 1)
    sample["amount_log"] = np.log1p(sample.amount_usd)
    sample["customer_week_spend_log"] = np.log1p(sample.customer_week_spend.clip(lower=0))
    sample["row_origin"] = "operational"
    sample["is_test_purchase_sequence"] = False
    sample["test_transaction_id"] = None
    sample["test_purchase_gap_days"] = np.nan
    sample["paired_fraud_transaction_id"] = None

    # --- test-purchase -> large-purchase sequences -------------------------
    small = df.loc[df.amount_usd <= 10.0,
                   ["transaction_id", "customer_id", "transaction_date", "amount_usd",
                    "merchant_id"]].copy()
    small["transaction_date"] = pd.to_datetime(small.transaction_date)
    small["test_date"] = small["transaction_date"]
    left = fraud_true[["transaction_id", "customer_id", "transaction_date",
                       "amount_usd", "merchant_id"]].copy()
    left["transaction_date"] = pd.to_datetime(left.transaction_date)
    left = left.sort_values("transaction_date")
    small = small.sort_values("transaction_date")
    matched = pd.merge_asof(
        left, small.rename(columns={"transaction_id": "test_transaction_id",
                                    "amount_usd": "test_amount",
                                    "merchant_id": "test_merchant_id"}),
        on="transaction_date", by="customer_id", direction="backward",
        tolerance=pd.Timedelta(days=3),
        suffixes=("", "_test"))
    matched = matched[
        matched.test_date.notna()
        & (matched.test_date < matched.transaction_date)
        & (matched.test_amount <= np.minimum(10.0, 0.10 * matched.amount_usd))
    ].copy()
    matched["test_purchase_gap_days"] = (
        matched.transaction_date - matched.test_date).dt.days

    pairs = []
    seq_map = dict(zip(matched.transaction_id, matched.test_transaction_id))
    gap_map = dict(zip(matched.transaction_id, matched.test_purchase_gap_days))
    amt_map = dict(zip(matched.transaction_id, matched.test_amount))
    sample["is_test_purchase_sequence"] = sample.transaction_id.isin(seq_map)
    sample["test_transaction_id"] = sample.transaction_id.map(seq_map)
    sample["test_purchase_gap_days"] = sample.transaction_id.map(gap_map)
    sub = sample[sample.transaction_id.isin(seq_map)].copy()
    sub["test_transaction_id_v"] = sub.transaction_id.map(seq_map)
    sub["gap_days_v"] = sub.transaction_id.map(gap_map)
    sub["test_amount_v"] = sub.transaction_id.map(amt_map)
    pairs.extend(
        dict(test_transaction_id=t, fraud_transaction_id=f, customer_id=c,
             gap_days=float(g), test_amount=float(a), fraud_amount=float(amt),
             origin="operational")
        for t, f, c, g, a, amt in zip(
            sub.test_transaction_id_v, sub.transaction_id, sub.customer_id,
            sub.gap_days_v, sub.test_amount_v, sub.amount_usd))

    # ML-only synthetic test purchases guarantee the pattern is present and
    # labelled even where the operational history has no small precursor.
    unmatched = sample[(sample.label_true == 1) & (~sample.transaction_id.isin(seq_map))]
    synth_rows = []
    for n, (idx, r) in enumerate(unmatched.iterrows(), start=1):
        syn_id = f"SYNTEST{n:08d}"
        fraud_date = pd.Timestamp(r.transaction_date)
        syn = dict(r)
        syn.update(
            transaction_id=syn_id,
            transaction_date=fraud_date - pd.Timedelta(days=1),
            transaction_ts=fraud_date - pd.Timedelta(days=1, hours=2),
            hour=int((fraud_date - pd.Timedelta(days=1, hours=2)).hour),
            amount_usd=round(float(min(4.0, max(0.50, r.amount_usd * 0.01))), 2),
            fraud_type="NONE", label_true=0, label_noisy=0, label_noise="NONE",
            is_fraud=False, is_test_purchase_sequence=False,
            test_transaction_id=None, test_purchase_gap_days=np.nan,
            row_origin="synthetic_test_purchase",
            paired_fraud_transaction_id=r.transaction_id,
        )
        syn["amount_log"] = np.log1p(syn["amount_usd"])
        syn["customer_week_spend_log"] = np.log1p(max(0.0, r.customer_week_spend))
        synth_rows.append(syn)
        sample.at[idx, "is_test_purchase_sequence"] = True
        sample.at[idx, "test_transaction_id"] = syn_id
        sample.at[idx, "test_purchase_gap_days"] = 1.0
        pairs.append(dict(test_transaction_id=syn_id,
                          fraud_transaction_id=r.transaction_id,
                          customer_id=r.customer_id, gap_days=1.0,
                          test_amount=syn["amount_usd"],
                          fraud_amount=float(r.amount_usd), origin="synthetic"))
    if synth_rows:
        sample = pd.concat([sample, pd.DataFrame(synth_rows)], ignore_index=True)

    sequences = pd.DataFrame(pairs)
    sample = sample.sort_values(["transaction_date", "transaction_id"]).reset_index(drop=True)

    out_cols = [
        "transaction_id", "customer_id", "account_id", "merchant_id", "mcc",
        "category", "channel_group", "entry_mode", "is_card_present",
        "is_international", "transaction_date", "hour", "amount_usd",
        "amount_log", "device_risk_score", "customer_day_txn_count",
        "customer_week_spend_log", "days_since_first_txn", "fraud_type",
        "label_true", "label_noisy", "label_noise", "split_time", "split_entity",
        "row_origin", "is_test_purchase_sequence", "test_transaction_id",
        "test_purchase_gap_days", "paired_fraud_transaction_id",
    ]
    sample = sample[out_cols]
    write_parquet(sample, ML / "fraud_transactions.parquet", "fraud_transactions")
    train = sample[(sample.split_time == "train") & (sample.split_entity == "train_entity")]
    oot = sample[sample.split_time == "test"]
    entity = sample[sample.split_entity == "test_entity"]
    write_parquet(train, ML / "fraud_train.parquet", "fraud_train")
    write_parquet(oot, ML / "fraud_oot.parquet", "fraud_oot")
    write_parquet(entity, ML / "fraud_test_entity.parquet", "fraud_test_entity")
    write_parquet(sequences, ML / "fraud_sequences.parquet", "fraud_sequences")

    # --- baseline model + drift -------------------------------------------
    features = ["amount_log", "hour", "is_card_present", "is_international",
                "device_risk_score", "customer_day_txn_count",
                "customer_week_spend_log", "days_since_first_txn"]
    Xtr = train[features].astype(float).to_numpy()
    ytr = train.label_noisy.to_numpy(dtype=float)
    if len(Xtr) > 500_000:
        idx = np.random.default_rng(C.SEED + 13).choice(len(Xtr), 500_000, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-9] = 1.0
    Xtr = (Xtr - mu) / sd
    Xtr = np.column_stack([np.ones(len(Xtr)), Xtr])
    w = np.zeros(Xtr.shape[1])
    lr = 0.35
    for _ in range(220):
        p = _sigmoid(Xtr @ w)
        grad = Xtr.T @ (p - ytr) / len(ytr) + 1e-4 * np.r_[0.0, w[1:]]
        w -= lr * grad
    Xoot = (oot[features].astype(float).to_numpy() - mu) / sd
    Xoot = np.column_stack([np.ones(len(Xoot)), Xoot])
    score = _sigmoid(Xoot @ w)
    pr_auc = average_precision(oot.label_noisy.to_numpy(), score)
    psi = population_stability_index(
        train.device_risk_score.to_numpy(), oot.device_risk_score.to_numpy())
    ok_pr, pr_detail = check_pr_auc(pr_auc)
    ok_drift, drift_detail = check_drift(psi)
    if not (ok_pr and ok_drift):
        raise RuntimeError(f"fraud acceptance failed: {pr_detail}; {drift_detail}")
    if len(sequences) == 0:
        raise RuntimeError("no test-purchase sequences were produced")

    report = {
        "rows": int(len(sample)),
        "true_fraud_rows": int(sample.label_true.sum()),
        "noisy_fraud_rows": int(sample.label_noisy.sum()),
        "label_noise": {
            "indistinguishable_rate": C.ML_FRAUD_LABEL_NOISE_INDISTINGUISHABLE,
            "false_flag_rate": C.ML_FRAUD_LABEL_NOISE_FALSE_FLAG,
            "indistinguishable_rows": int((sample.label_noise == "INDISTINGUISHABLE").sum()),
            "false_flag_rows": int((sample.label_noise == "FALSE_FLAG").sum()),
        },
        "splits": {"train_rows": int(len(train)), "oot_rows": int(len(oot)),
                   "entity_test_rows": int(len(entity))},
        "baseline_model": "numpy logistic regression (standardised features)",
        "pr_auc_oot": round(pr_auc, 6),
        "pr_auc_gate": C.ML_PR_AUC_MAX,
        "pr_auc_pass": ok_pr,
        "drift_psi_device_risk": round(psi, 6),
        "drift_gate": C.ML_DRIFT_PSI_MIN,
        "drift_pass": ok_drift,
        "sequence_pairs": int(len(sequences)),
        "sequence_pairs_synthetic": int((sequences.origin == "synthetic").sum())
        if len(sequences) else 0,
        "final_quarter_fraud_type_mix": (
            sample[(sample.split_time == "test") & (sample.label_true == 1)]
            .fraud_type.value_counts(normalize=True).round(4).to_dict()),
    }
    (ML / "model_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# --------------------------------------------------------------------------
# As-of payment status (round 5)
# --------------------------------------------------------------------------
# fact_statement.payment_status / paid_amount / last_payment_date are settled
# with ALL later card cash up to PAYMENT_HORIZON (facts._settled_closing_amounts,
# facts._stmt_status), i.e. with payments dated after the observation date. They
# must never feed a point-in-time feature. The status below is recomputed from
# fact_payment_allocation rows dated on or before the observation date only.
# statement.payments_amount / closing_balance / minimum_due / utilization_pct and
# delinquency snapshots are as-of the statement cut (cash dated <= cycle end), so
# they are safe.
def check_payment_sources_asof(alloc_used: pd.DataFrame, obs: pd.Timestamp,
                               date_col: str = "allocation_date") -> tuple[bool, str]:
    """Fail if any payment event feeding a feature is dated after ``obs``."""
    if alloc_used.empty:
        return True, "no payment events used"
    latest = pd.to_datetime(alloc_used[date_col]).max()
    bad = int((pd.to_datetime(alloc_used[date_col]) > obs).sum())
    return bad == 0, f"{bad} payment events after observation (latest {latest.date()})"


def asof_payment_status(hist: pd.DataFrame, allocations: pd.DataFrame,
                        obs: pd.Timestamp) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Payment status of each statement using only payments dated <= ``obs``.

    Returns (status indexed like ``hist``, latest payment date used per
    statement, the allocation rows used). Mirrors ``facts._stmt_status`` with
    allocation-so-far in place of the final settlement.
    """
    a = allocations.copy()
    a["allocation_date"] = pd.to_datetime(a.allocation_date)
    used = a[a.allocation_date <= obs]
    paid = used.groupby("statement_id").amount_usd.sum()
    last_pay = used.groupby("statement_id").allocation_date.max()
    allocated = hist.statement_id.map(paid).fillna(0.0).astype(float)
    close = hist.closing_balance.astype(float)
    billed = hist.billed_amount.astype(float)
    due = pd.to_datetime(hist.payment_due_date)
    min_due = hist.minimum_due.astype(float)
    conds = [
        close < -0.01,
        billed <= 0.01,
        close <= 0.0001,
        allocated >= billed - 0.01,
        (min_due > 0.01) & (allocated >= min_due - 0.01),
        allocated > 0.01,
        due > obs,
    ]
    status = np.select(
        conds,
        ["CREDIT_BALANCE", np.where(close <= 0.0001, "NO_BALANCE", "CARRY_FORWARD"),
         "NO_BALANCE", "PAID_FULL", "PAID_PARTIAL", "PARTIAL", "PENDING"],
        default="UNPAID")
    return (pd.Series(status, index=hist.index),
            hist.statement_id.map(last_pay), used)


# --------------------------------------------------------------------------
# Point-in-time credit default / churn
# --------------------------------------------------------------------------
def build_credit_default(df: pd.DataFrame, accounts: pd.DataFrame,
                         customers: pd.DataFrame, statements: pd.DataFrame,
                         delinq: pd.DataFrame, allocations: pd.DataFrame) -> None:
    log("  ml: credit default")
    obs = pd.Timestamp(C.ML_OBSERVATION_DATE)
    end = pd.Timestamp(C.ML_OUTCOME_END)
    st = statements.copy()
    st["statement_date"] = pd.to_datetime(st.statement_date)
    hist = st[st.statement_date <= obs].sort_values(["account_id", "statement_date"])
    if hist.empty:
        raise RuntimeError("no statements before the observation date")

    # payment_status as known on the observation date (not the final status)
    hist = hist.copy()
    hist["payment_status"], hist["_last_pay"], alloc_used = asof_payment_status(
        hist, allocations, obs)
    ok, detail = check_payment_sources_asof(alloc_used, obs)
    if not ok:
        raise RuntimeError(f"credit_default leakage: {detail}")

    last = hist.groupby("account_id", sort=False).tail(1)
    last3 = hist.groupby("account_id", sort=False).tail(3)
    agg3 = last3.groupby("account_id").agg(
        payments_3m=("payments_amount", "sum"),
        purchases_3m=("purchases_amount", "sum"),
        fees_3m=("fees_charged", "sum"),
        interest_3m=("interest_charged", "sum"),
        statements_3m=("statement_id", "size"),
        paid_full_share_3m=("payment_status", lambda s: float((s == "PAID_FULL").mean())),
    )
    feats = last[["account_id", "customer_id", "product_id", "statement_date",
                  "opening_balance", "closing_balance", "utilization_pct",
                  "minimum_due", "credit_limit", "payment_status"]].copy()
    feats = feats.rename(columns={
        "statement_date": "feature_statement_date",
        "closing_balance": "balance_at_obs",
        "utilization_pct": "utilization_at_obs",
        "minimum_due": "minimum_due_at_obs",
        "credit_limit": "credit_limit_at_obs",
        "payment_status": "status_at_obs",
    })
    feats = feats.join(agg3, on="account_id")
    feats["payment_feature_date"] = feats.account_id.map(
        last3.groupby("account_id")["_last_pay"].max())

    ds = delinq.copy()
    ds["snapshot_date"] = pd.to_datetime(ds.snapshot_date)
    dhist = ds[ds.snapshot_date <= obs].sort_values(["account_id", "snapshot_date"])
    dlast = dhist.groupby("account_id", sort=False).tail(1)[
        ["account_id", "snapshot_date", "days_past_due", "delinquency_status"]].rename(
        columns={"snapshot_date": "delinq_feature_date",
                 "days_past_due": "dpd_at_obs",
                 "delinquency_status": "delinq_status_at_obs"})
    d90 = dhist[dhist.snapshot_date > obs - pd.Timedelta(days=90)]
    any30 = (d90[d90.days_past_due >= 30].groupby("account_id").size()
             .rename("delinq_30d_count_90d"))
    feats = feats.merge(dlast, on="account_id", how="left").merge(
        any30, on="account_id", how="left")
    feats["delinq_30d_count_90d"] = feats.delinq_30d_count_90d.fillna(0)

    tx90 = df[(df.transaction_date > obs - pd.Timedelta(days=90))
              & (df.transaction_date <= obs)]
    spend90 = tx90.groupby("account_id").agg(
        spend_90d=("amount_usd", "sum"), txn_90d=("transaction_id", "size"),
        intl_share_90d=("is_international", "mean"))
    feats = feats.merge(spend90, on="account_id", how="left")
    for col in ("spend_90d", "txn_90d", "intl_share_90d"):
        feats[col] = feats[col].fillna(0.0)

    acct = accounts[["account_id", "open_date", "account_status", "closed_date"]].copy()
    acct["open_date"] = pd.to_datetime(acct.open_date)
    acct["closed_date"] = pd.to_datetime(acct.closed_date)
    # account_status / closed_date define the label: used below, never a feature
    feats = feats.merge(acct[["account_id", "open_date"]], on="account_id", how="left")
    feats["months_on_book_obs"] = (
        (obs - pd.to_datetime(feats.open_date)).dt.days / 30.44).round(2)

    cust = customers[["customer_id", "credit_score", "income_annual",
                      "customer_segment", "risk_tier"]]
    feats = feats.merge(cust, on="customer_id", how="left")

    future = ds[(ds.snapshot_date > obs) & (ds.snapshot_date <= end)]
    default_ids = set(future.loc[future.days_past_due >= 60, "account_id"])
    default_ids |= set(acct.loc[(acct.account_status == "CHARGED_OFF")
                                & (acct.closed_date > obs)
                                & (acct.closed_date <= end), "account_id"])
    feats["label_default"] = feats.account_id.isin(default_ids).astype("int8")
    feats["observation_date"] = obs
    feats["outcome_start"] = obs + pd.Timedelta(days=1)
    feats["outcome_end"] = end
    feats["max_source_date"] = pd.to_datetime(
        feats[["feature_statement_date", "delinq_feature_date",
               "payment_feature_date"]].max(axis=1))
    feats["max_source_date"] = feats.max_source_date.fillna(obs)
    feats["split_entity"] = np.where(
        feats.account_id.map(lambda x: _hash_bucket(x) == 0), "test", "train")

    for gate in (check_no_leakage(feats), check_no_label_columns(feats, "credit_default")):
        if not gate[0]:
            raise RuntimeError(f"credit_default leakage: {gate[1]}")
    write_parquet(feats, ML / "credit_default_features.parquet", "credit_default_features")
    write_parquet(feats[feats.split_entity == "train"],
                  ML / "credit_default_train.parquet", "credit_default_train")
    write_parquet(feats[feats.split_entity == "test"],
                  ML / "credit_default_test.parquet", "credit_default_test")


def build_churn(df: pd.DataFrame, customers: pd.DataFrame,
                statements: pd.DataFrame) -> None:
    log("  ml: churn")
    obs = pd.Timestamp(C.ML_OBSERVATION_DATE)
    end = pd.Timestamp(C.ML_OUTCOME_END)
    cust = customers.copy()
    cust["onboarding_date"] = pd.to_datetime(cust.onboarding_date)
    cust["churn_date"] = pd.to_datetime(cust.churn_date)
    active = cust[(cust.onboarding_date <= obs)
                  & (cust.churn_date.isna() | (cust.churn_date > obs))].copy()

    tx90 = df[(df.transaction_date > obs - pd.Timedelta(days=90))
              & (df.transaction_date <= obs)]
    agg = tx90.groupby("customer_id").agg(
        txn_90d=("transaction_id", "size"),
        spend_90d=("amount_usd", "sum"),
        avg_ticket_90d=("amount_usd", "mean"),
        distinct_merchants_90d=("merchant_id", "nunique"))
    last_tx = df[df.transaction_date <= obs].groupby("customer_id").transaction_date.max()
    # customer_status / churn_date define the label: kept out of the features
    feats = active[["customer_id", "onboarding_date", "credit_score", "income_annual",
                    "customer_segment", "risk_tier"]].copy()
    churn_date = active.set_index("customer_id")["churn_date"]
    feats = feats.merge(agg, on="customer_id", how="left")
    feats["last_txn_date"] = feats.customer_id.map(last_tx)
    for col in ("txn_90d", "spend_90d", "avg_ticket_90d", "distinct_merchants_90d"):
        feats[col] = feats[col].fillna(0.0)
    feats["recency_days"] = (obs - feats.last_txn_date).dt.days
    feats["tenure_days"] = (obs - feats.onboarding_date).dt.days

    st = statements.copy()
    st["statement_date"] = pd.to_datetime(st.statement_date)
    shist = st[st.statement_date <= obs].sort_values(["customer_id", "statement_date"])
    slast = shist.groupby("customer_id", sort=False).tail(1)[
        ["customer_id", "statement_date", "closing_balance", "utilization_pct"]].rename(
        columns={"statement_date": "feature_statement_date",
                 "closing_balance": "balance_at_obs",
                 "utilization_pct": "utilization_at_obs"})
    feats = feats.merge(slast, on="customer_id", how="left")
    cd = feats.customer_id.map(churn_date)
    feats["label_churn"] = ((cd > obs) & (cd <= end)).astype("int8")
    feats["observation_date"] = obs
    feats["outcome_start"] = obs + pd.Timedelta(days=1)
    feats["outcome_end"] = end
    feats["max_source_date"] = pd.to_datetime(
        feats[["feature_statement_date", "last_txn_date"]].max(axis=1)).fillna(obs)
    feats["split_entity"] = np.where(
        feats.customer_id.map(lambda x: _hash_bucket(x) == 0), "test", "train")

    for gate in (check_no_leakage(feats), check_no_label_columns(feats, "churn")):
        if not gate[0]:
            raise RuntimeError(f"churn leakage: {gate[1]}")
    write_parquet(feats, ML / "churn_features.parquet", "churn_features")
    write_parquet(feats[feats.split_entity == "train"],
                  ML / "churn_train.parquet", "churn_train")
    write_parquet(feats[feats.split_entity == "test"],
                  ML / "churn_test.parquet", "churn_test")


# --------------------------------------------------------------------------
# Segmentation / forecasting
# --------------------------------------------------------------------------
def build_segmentation(df: pd.DataFrame) -> None:
    log("  ml: segmentation")
    end = pd.Timestamp(C.PERIOD_END)
    agg = df.groupby("customer_id").agg(
        frequency=("transaction_id", "size"),
        monetary=("amount_usd", "sum"),
        avg_ticket=("amount_usd", "mean"),
        last_purchase=("transaction_date", "max"),
        distinct_mcc=("mcc", "nunique"),
        distinct_merchants=("merchant_id", "nunique"),
    )
    agg["recency_days"] = (end - agg.last_purchase).dt.days
    agg = agg.drop(columns=["last_purchase"]).reset_index()
    write_parquet(agg, ML / "segmentation_rfm.parquet", "segmentation_rfm")

    shares = (df.groupby(["customer_id", "category"]).size()
              .unstack(fill_value=0).astype(float))
    shares = shares.div(shares.sum(axis=1), axis=0).reset_index()
    shares.columns = ["customer_id"] + [f"share_{str(c).lower().replace(' ', '_').replace('-', '_')}"
                                         for c in shares.columns[1:]]
    write_parquet(shares, ML / "segmentation_mcc_share.parquet", "segmentation_mcc_share")


def build_forecast(df: pd.DataFrame) -> None:
    log("  ml: forecast")
    work = df.assign(month=df.transaction_date.dt.strftime("%Y-%m"))
    monthly = work.groupby(["merchant_id", "month"]).agg(
        txn_count=("transaction_id", "size"),
        spend=("amount_usd", "sum"),
        avg_ticket=("amount_usd", "mean"),
    ).reset_index()
    top = df.groupby("merchant_id").amount_usd.sum().nlargest(
        C.ML_FORECAST_MERCHANTS).index
    monthly = monthly[monthly.merchant_id.isin(set(top))].copy()
    monthly = monthly.sort_values(["merchant_id", "month"])
    monthly["prev_month_spend"] = monthly.groupby("merchant_id").spend.shift(1)
    monthly["prev_month_txn"] = monthly.groupby("merchant_id").txn_count.shift(1)
    monthly["month_of_year"] = pd.to_datetime(monthly.month + "-01").dt.month
    monthly["quarter"] = pd.to_datetime(monthly.month + "-01").dt.quarter
    monthly["is_q4"] = (monthly.quarter == 4).astype("int8")
    totals = monthly.groupby("month").spend.sum()
    seasonal = (totals / totals.mean()).round(6)
    monthly["seasonal_index"] = monthly.month.map(seasonal)
    write_parquet(monthly, ML / "forecast_merchant_month.parquet",
                  "forecast_merchant_month")


# --------------------------------------------------------------------------
# README
# --------------------------------------------------------------------------
README = """# ML extracts (v7)

Generated by `build.ml_extracts` during `build.run_all`. The operational star
schema (`dim/`, `fact/`, `raw/`) is unchanged; every file here is a derived
extract. All files are synthetic and are **not calibrated to real-world rates**.

## fraud_*

| File | Contents |
|---|---|
| `fraud_transactions.parquet` | sampled approved purchases with noisy labels, splits, fraud typology, velocity features and a final-quarter device-risk regime shift |
| `fraud_train.parquet` | `split_time=train` and `split_entity=train_entity` |
| `fraud_oot.parquet` | out-of-time test: transactions from 2026-07-01 onward |
| `fraud_test_entity.parquet` | out-of-entity test: 20% of customers by SHA-256 bucket |
| `fraud_sequences.parquet` | labelled test-purchase -> large-purchase pairs |
| `model_report.json` | baseline PR-AUC, drift PSI, label-noise and split counts |

**Labels.** `label_true` is the operational confirmed-fraud flag. `label_noisy`
is the ML-only label: 15% of true fraud is relabelled 0 (indistinguishable) and
2% of non-fraud is relabelled 1 (false flag). Train and evaluate on
`label_noisy`; use `label_true` only for diagnostics.

**Sequences.** A pair is labelled when a small approved purchase (<= $10 and
<= 10% of the later amount) lands on the same customer within the prior 3 days
of a confirmed fraud transaction. If the operational history has no match, an
ML-only synthetic test purchase row is added (`row_origin` =
`synthetic_test_purchase`) so the pattern is always represented.

**Regime shift.** Final-quarter fraud types skew to card-not-present and
account-takeover, and `device_risk_score` is elevated. The drift gate is PSI
>= 0.20 on `device_risk_score` between train and the out-of-time split.

**Baseline.** A numpy logistic regression (sklearn/lightgbm are not runtime
dependencies) must score PR-AUC < 0.95 on the out-of-time split.

## credit_default_*

`credit_default_features.parquet` is account-level, one row per account with an
observation date of 2026-03-31. Features use only statements, transactions and
delinquency snapshots on or before that date; `max_source_date` records the
latest fact date used (statement, delinquency snapshot and payment event) and
must not exceed `observation_date`. `status_at_obs` and `paid_full_share_3m`
are recomputed from payment allocations dated on or before the observation date;
the final `fact_statement.payment_status` is settled with later cash and is not
used. The label
(`label_default`) is 1 when the account reaches 60+ DPD or is charged off in
the outcome window 2026-04-01..2026-09-30. `split_entity` is an 80/20 SHA-256
account split; `credit_default_train.parquet` / `credit_default_test.parquet`
are the two sides.

## churn_*

`churn_features.parquet` is customer-level with the same observation/outcome
window as credit default. It includes RFM-style 90-day activity, tenure and the
last statement balance/utilization at the observation date. `label_churn` is 1
when `churn_date` falls in the outcome window. Customers already churned at the
observation date are excluded.

## segmentation_*

`segmentation_rfm.parquet` has full-period recency, frequency, monetary value,
average ticket, distinct MCCs and distinct merchants. `segmentation_mcc_share.parquet`
has each customer's share of spend by merchant category.

## forecast_*

`forecast_merchant_month.parquet` is merchant-month volume/spend for the 5,000
largest merchants, with `month_of_year`, `quarter`, `is_q4`, a portfolio
`seasonal_index`, and one-month lags. Rows before a merchant's second observed
month have null lags by construction.

## Known limitations

- Synthetic labels and synthetic covariates; do not claim real-world calibration.
- `device_risk_score`, the final-quarter regime shift and synthetic sequence
  test purchases exist only in the ML copy and are deliberate fixtures.
- The fraud extract is a sample of non-fraud rows (all confirmed fraud is kept).
- No causal DGP or treatment assignment is produced (deferred, see the hand-off).
"""


def build_all() -> dict:
    ML.mkdir(parents=True, exist_ok=True)
    log("ml: loading operational tables")
    purchases = _attach_velocity(_load_purchases())
    accounts = _read("dim/dim_account.parquet",
                     ["account_id", "customer_id", "product_id", "open_date",
                      "account_status", "closed_date"])
    customers = _read("dim/dim_customer.parquet",
                      ["customer_id", "onboarding_date", "credit_score",
                       "income_annual", "customer_segment", "risk_tier",
                       "customer_status", "churn_date"])
    statements = _read("fact/fact_statement.parquet",
                       ["statement_id", "account_id", "customer_id", "product_id",
                        "statement_date", "opening_balance", "closing_balance",
                        "purchases_amount", "payments_amount", "interest_charged",
                        "fees_charged", "minimum_due", "credit_limit",
                        "utilization_pct", "payment_status", "billed_amount",
                        "payment_due_date"])
    allocations = _read("fact/fact_payment_allocation.parquet",
                        ["statement_id", "allocation_date", "amount_usd"])
    delinq = _read("fact/fact_delinquency_snapshot.parquet",
                   ["account_id", "snapshot_date", "days_past_due",
                    "delinquency_status"])

    report = build_fraud(purchases)
    build_credit_default(purchases, accounts, customers, statements, delinq, allocations)
    build_churn(purchases, customers, statements)
    build_segmentation(purchases)
    build_forecast(purchases)
    (ML / "README.md").write_text(README, encoding="utf-8")
    log(f"ml: wrote extracts (PR-AUC {report['pr_auc_oot']:.4f}, "
        f"PSI {report['drift_psi_device_risk']:.3f}, "
        f"sequences {report['sequence_pairs']})")
    return report


if __name__ == "__main__":
    build_all()
