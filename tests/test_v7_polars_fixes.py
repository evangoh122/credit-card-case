"""Round-2 fixes: label leakage gates, and no pandas full-load of fact_transaction."""
from __future__ import annotations


import numpy as np
import pandas as pd
import polars as pl
import pytest

from build import config as C
from build import dictionary, ml_extracts, refresh_docs, validation
from build.utils import write_parquet

OBS = pd.Timestamp(C.ML_OBSERVATION_DATE)


# --------------------------------------------------------------------------
# churn / credit-default label leakage
# --------------------------------------------------------------------------
def _customers():
    d = lambda days: OBS + pd.Timedelta(days=days)  # noqa: E731
    return pd.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C4"],
        "onboarding_date": [d(-400)] * 4,
        "credit_score": [700, 650, 600, 720],
        "income_annual": [50e3, 60e3, 70e3, 80e3],
        "customer_segment": ["A", "B", "A", "B"],
        "risk_tier": ["L", "M", "H", "L"],
        "customer_status": ["ACTIVE", "CHURNED", "CHURNED", "ACTIVE"],
        # C2 churns in the outcome window, C3 before obs (excluded), C1/C4 never
        "churn_date": [pd.NaT, d(30), d(-10), pd.NaT],
    })


def _stmts(ids, key="customer_id"):
    return pd.DataFrame({
        "statement_id": [f"S{i}" for i in range(len(ids))],
        key: ids, "account_id": [f"A{i}" for i in ids] if key == "customer_id" else ids,
        "customer_id": ids if key == "customer_id" else [f"C{i}" for i in range(len(ids))],
        "product_id": "P", "statement_date": OBS - pd.Timedelta(days=5),
        "opening_balance": 1.0, "closing_balance": 2.0, "purchases_amount": 3.0,
        "payments_amount": 1.0, "interest_charged": 0.0, "fees_charged": 0.0,
        "minimum_due": 1.0, "credit_limit": 10.0, "utilization_pct": 20.0,
        "payment_status": "PAID_FULL",
    })


def _tx(customer_ids):
    return pd.DataFrame({
        "transaction_id": [f"T{i}" for i in range(len(customer_ids))],
        "customer_id": customer_ids, "account_id": [f"A{c}" for c in customer_ids],
        "merchant_id": "M1", "amount_usd": 10.0,
        "transaction_date": OBS - pd.Timedelta(days=3), "is_international": False,
    })


@pytest.fixture
def captured(monkeypatch, tmp_path):
    out = {}
    monkeypatch.setattr(ml_extracts, "write_parquet",
                        lambda df, path, name=None: out.__setitem__(name, df.copy()))
    monkeypatch.setattr(ml_extracts, "ML", tmp_path)
    return out


def test_churn_features_exclude_label_columns(captured):
    ml_extracts.build_churn(_tx(["C1", "C2", "C4"]), _customers(),
                            _stmts(["C1", "C2", "C4"]))
    for name in ("churn_features", "churn_train", "churn_test"):
        if name in captured:
            assert not set(ml_extracts.LABEL_DEFINING_COLUMNS["churn"]) & set(captured[name])
    feats = captured["churn_features"].set_index("customer_id")
    assert set(feats.index) == {"C1", "C2", "C4"}  # C3 churned before obs
    assert feats.label_churn.to_dict() == {"C1": 0, "C2": 1, "C4": 0}  # label intact


def test_credit_default_features_exclude_label_columns(captured):
    d = lambda days: OBS + pd.Timedelta(days=days)  # noqa: E731
    accounts = pd.DataFrame({
        "account_id": ["AC1", "AC2"], "customer_id": ["C1", "C2"], "product_id": "P",
        "open_date": d(-300), "account_status": ["OPEN", "CHARGED_OFF"],
        "closed_date": [pd.NaT, d(20)],
    })
    stmts = _stmts(["C1", "C2"])
    stmts["account_id"] = ["AC1", "AC2"]
    delinq = pd.DataFrame({"account_id": ["AC1", "AC2"], "snapshot_date": OBS - pd.Timedelta(days=2),
                           "days_past_due": [0, 0], "delinquency_status": "CURRENT"})
    tx = _tx(["C1", "C2"])
    tx["account_id"] = ["AC1", "AC2"]
    stmts["billed_amount"], stmts["payment_due_date"] = 2.0, OBS + pd.Timedelta(days=20)
    alloc = pd.DataFrame({"statement_id": [], "allocation_date": [], "amount_usd": []})
    ml_extracts.build_credit_default(tx, accounts, _customers(), stmts, delinq, alloc)
    for name in ("credit_default_features", "credit_default_train", "credit_default_test"):
        if name in captured:
            assert not set(ml_extracts.LABEL_DEFINING_COLUMNS["credit_default"]) & set(captured[name])
    feats = captured["credit_default_features"].set_index("account_id")
    assert feats.label_default.to_dict() == {"AC1": 0, "AC2": 1}  # label intact


@pytest.mark.parametrize("task", ["churn", "credit_default"])
def test_label_column_gate_fails_on_every_defining_column(task):
    ok, _ = ml_extracts.check_no_label_columns(pd.DataFrame({"x": [1]}), task)
    assert ok
    for col in ml_extracts.LABEL_DEFINING_COLUMNS[task]:
        ok, detail = ml_extracts.check_no_label_columns(pd.DataFrame({"x": [1], col: [1]}), task)
        assert not ok and col in detail


# --------------------------------------------------------------------------
# tx_metrics (Polars) == former pandas expressions
# --------------------------------------------------------------------------
def _tx_frame(n=400, seed=1):
    r = np.random.default_rng(seed)
    d0 = pd.Timestamp("2026-01-01")
    df = pd.DataFrame({
        "transaction_id": [f"T{i}" for i in range(n)],
        "account_id": r.choice(["A1", "A2", "A3", "AX"], n),
        "customer_id": r.choice(["C1", "C2", "C3"], n),
        "card_id": r.choice(["K1", "K2", "KX"], n),
        "merchant_id": r.choice(["M1", "M2", None], n),
        "statement_id": r.choice(["S1", "S2", "S3", None], n),
        "transaction_date": d0 + pd.to_timedelta(r.integers(0, 90, n), unit="D"),
        "currency_code": r.choice(["USD", "USD", "USD", "EUR"], n),
        "amount_usd": r.gamma(2.0, 40.0, n).round(2) * r.choice([1, 1, 1, -1], n),
        "transaction_type": r.choice(["PURCHASE", "PURCHASE", "REFUND"], n),
        "status": r.choice(["APPROVED", "APPROVED", "DECLINED", None], n),
        "auth_code": r.choice(["123456", "12", None], n),
        "is_installment_financed": r.choice([True, False], n),
        "merchant_geo_id": r.choice(["G1", None], n),
        "mcc": r.choice(["5411", "5812"], n), "category": r.choice(["grocery", "dining"], n),
        "merchant_name": "n", "is_international": r.choice([True, False], n),
        "channel_group": r.choice(["POS", "ECOM"], n),
        "original_transaction_id": r.choice(["T1", "TZ", None], n),
        "date_key": r.choice([1, 2, 99], n), "network": r.choice(["VISA", None], n),
    })
    return df


def _small_dims(df):
    d = pd.Timestamp
    return {
        "dim_account": pd.DataFrame({"account_id": ["A1", "A2", "A3"], "network": ["V", "M", "V"],
                                     "closed_date": [pd.NaT, d("2026-02-01"), pd.NaT]}),
        "dim_customer": pd.DataFrame({"customer_id": ["C1", "C2", "C3"],
                                      "churn_date": [pd.NaT, d("2026-02-15"), pd.NaT]}),
        "dim_merchant": pd.DataFrame({"merchant_id": ["M1", "M2"], "mcc": ["5411", "5999"],
                                      "category": ["grocery", "dining"], "merchant_name": "n",
                                      "geo_id": ["G1", "G1"], "country_code": ["US", "GB"]}),
        "dim_card": pd.DataFrame({"card_id": ["K1", "K2"], "expiry_year": [2026, 2027],
                                  "expiry_month": [2, 1], "issue_date": [d("2026-01-10")] * 2,
                                  "replaced_date": [pd.NaT, d("2026-03-01")]}),
    }


def _null_case():
    """Original fixture plus appended null-bearing rows / merchant M3 (baseline untouched)."""
    tx, dims = _tx_frame(), _small_dims(None)
    extra = tx.iloc[:6].copy()
    extra["transaction_id"] = [f"TN{i}" for i in range(6)]
    extra["merchant_id"] = ["M3", "M3", "M3", "M1", "M2", None]
    extra["merchant_geo_id"] = [None, "G1", None, None, None, None]   # null/null with M3
    extra["mcc"] = [None, None, "5411", None, "5812", None]
    extra["category"] = [None, "grocery", None, None, "dining", None]
    extra["is_international"] = [True, False, True, False, True, False]
    dims["dim_merchant"] = pd.concat([dims["dim_merchant"], pd.DataFrame({
        "merchant_id": ["M3"], "mcc": [None], "category": [None], "merchant_name": ["n"],
        "geo_id": [None], "country_code": [None]})], ignore_index=True)
    return pd.concat([tx, extra], ignore_index=True), dims


def test_tx_metrics_match_pandas_reference_original_fixture():
    ref_bad = _assert_metrics_match(_tx_frame(), _small_dims(None))
    assert ref_bad["bad_geo"] == 270  # committed baseline: fixture must not drift


def test_tx_metrics_match_pandas_reference_with_null_rows():
    tx, dims = _null_case()
    ref_bad = _assert_metrics_match(tx, dims)
    assert all(v > 0 for v in ref_bad.values())
    # null/null geo pairs are mismatches under pandas `!=`; they must be counted
    mm = tx.merge(dims["dim_merchant"][["merchant_id", "geo_id"]], on="merchant_id", how="left")
    assert ref_bad["bad_geo"] > int(((mm.merchant_geo_id != mm.geo_id)
                                     & mm.merchant_geo_id.notna() & mm.geo_id.notna()).sum())
    assert int((mm.merchant_geo_id.isna() & mm.geo_id.isna()).sum()) > 0


def _assert_metrics_match(tx, dims):
    stmt = pd.DataFrame({"statement_id": ["S1", "S2", "S3"]})
    facts = {"fact_statement": stmt,
             "fact_installment_plan": pd.DataFrame({"transaction_id": ["T1", "T_missing"]}),
             "fact_fraud_alert": pd.DataFrame({"transaction_id": ["T2", "T5"],
                                               "loss_amount": [5.0, 7.0]})}
    M = validation.tx_metrics(pl.from_pandas(tx).lazy(), dims=dims, facts=facts, dd_keys={1, 2})
    assert M["n"] == len(tx)
    assert M["tid_nunique"] == tx.transaction_id.nunique()
    assert M["fk_account"] == int((~tx.account_id.isin({"A1", "A2", "A3"})).sum())
    assert M["fk_card"] == int((~tx.card_id.isin({"K1", "K2"})).sum())
    assert M["fk_statement"] == int((~tx.statement_id.isin({"S1", "S2", "S3"})).sum())
    assert M["dmin"] == tx.transaction_date.min().date()
    assert M["php"] == int((tx.currency_code == "PHP").sum()) and M["usd"] == int((tx.currency_code == "USD").sum())
    for q in (0.25, 0.5, 0.95):
        pos = tx.amount_usd[tx.amount_usd > 0]
        assert M[f"q{q}"] == pytest.approx(float(pos.quantile(q)))
    assert M["neg_purchases"] == int(((tx.transaction_type == "PURCHASE") & (tx.amount_usd < 0)).sum())
    assert M["approved_share"] == pytest.approx(tx.status.value_counts(normalize=True)["APPROVED"])
    pur = tx[tx.transaction_type == "PURCHASE"]
    pd.testing.assert_series_equal(
        M["per_stmt"].set_index("statement_id")["n"].sort_index(),
        pur.groupby("statement_id").size().rename("n").sort_index(), check_dtype=False)
    ok = tx[(tx.transaction_type == "PURCHASE") & (tx.status == "APPROVED") & ~tx.is_installment_financed]
    pd.testing.assert_series_equal(
        M["agg_p"].sort_index(), ok.groupby("statement_id").amount_usd.sum().rename("tx_purch").sort_index())
    app = tx[tx.status == "APPROVED"]
    assert M["bad_auth"] == int((app.auth_code.astype(str).str.len() != 6).sum())
    assert M["approved_purchase_spend"] == pytest.approx(
        float(tx.loc[(tx.status == "APPROVED") & (tx.transaction_type == "PURCHASE"), "amount_usd"].sum()))
    assert M["date_key_miss"] == int((~tx.date_key.isin({1, 2})).sum())
    ref = tx[tx.transaction_type == "REFUND"]
    assert M["refund_missing"] == int(ref.original_transaction_id.isna().sum())
    assert M["refund_bad"] == int((~ref.original_transaction_id.dropna().isin(set(tx.transaction_id))).sum())
    assert M["plan_orphans"] == 1 and M["fraud_orphans"] == 0
    # joined checks
    exp = dims["dim_card"].copy()
    exp["expiry_end"] = pd.to_datetime(dict(year=exp.expiry_year, month=exp.expiry_month, day=1)) + pd.offsets.MonthEnd(0)
    em = tx[["transaction_date", "card_id"]].merge(exp[["card_id", "expiry_end"]], on="card_id", how="left")
    assert M["after_exp"] == int((em.transaction_date > em.expiry_end).sum())
    cm = tx[["transaction_date", "card_id"]].merge(dims["dim_card"][["card_id", "issue_date", "replaced_date"]], on="card_id", how="left")
    assert M["before_issue"] == int((cm.transaction_date < cm.issue_date).sum())
    assert M["after_repl"] == int((cm.replaced_date.notna() & (cm.transaction_date > cm.replaced_date)).sum())
    am = tx[["transaction_date", "account_id"]].merge(dims["dim_account"][["account_id", "closed_date"]], on="account_id", how="left")
    assert M["after_close"] == int((am.closed_date.notna() & (am.transaction_date > am.closed_date)).sum())
    assert M["fk_customer"] == int((~tx.customer_id.isin({"C1", "C2", "C3"})).sum())
    assert M["fk_merchant"] == int((~tx.merchant_id.isin(set(dims["dim_merchant"].merchant_id))).sum())
    assert M["null_geo"] == int(tx.merchant_geo_id.isna().sum())
    assert M["approved"] == int((tx.status == "APPROVED").sum())
    assert M["purchases"] == int((tx.transaction_type == "PURCHASE").sum())
    assert M["refunds"] == int((tx.transaction_type == "REFUND").sum())
    assert M["financed_sum"] == pytest.approx(float(tx[tx.is_installment_financed].amount_usd.sum()))
    ch = dims["dim_customer"].dropna(subset=["churn_date"])
    cm2 = tx[["transaction_date", "customer_id"]].merge(ch[["customer_id", "churn_date"]], on="customer_id")
    assert M["after_churn"] == int((cm2.transaction_date > cm2.churn_date).sum())
    # exact pandas expressions from the former validation.run (null rows NOT filtered)
    mm = tx[["merchant_id", "mcc", "category", "merchant_name", "merchant_geo_id",
             "is_international"]].merge(
        dims["dim_merchant"][["merchant_id", "mcc", "category", "merchant_name", "geo_id",
                              "country_code"]],
        on="merchant_id", how="left", suffixes=("_tx", "_m"))
    ref_bad = {
        "bad_mcc": int((mm.mcc_tx != mm.mcc_m).sum()),
        "bad_cat": int((mm.category_tx != mm.category_m).sum()),
        "bad_geo": int((mm.merchant_geo_id != mm.geo_id).sum()),
        "bad_intl": int((mm.is_international != (mm.country_code != "US")).sum()),
    }
    assert {k: M[k] for k in ref_bad} == ref_bad
    monthly = tx.assign(m=tx.transaction_date.dt.strftime("%Y-%m")).groupby("m").agg(
        n=("transaction_id", "size"), spend=("amount_usd", lambda s: s.clip(lower=0).sum()))
    pd.testing.assert_frame_equal(M["monthly"], monthly, check_dtype=False)
    net = app.groupby("network").agg(n=("transaction_id", "size"), spend=("amount_usd", "sum"))
    pd.testing.assert_frame_equal(M["net_tx"].sort_index(), net.sort_index(), check_dtype=False)
    return ref_bad


# --------------------------------------------------------------------------
# refresh path never materialises fact_transaction in pandas
# --------------------------------------------------------------------------
@pytest.fixture
def tiny_out(monkeypatch, tmp_path):
    for attr, sub in (("DIM_DIR", "dim"), ("FACT_DIR", "fact"), ("RAW_DIR", "raw")):
        monkeypatch.setattr(C, attr, tmp_path / sub)
    monkeypatch.setattr(C, "OUT", tmp_path)
    write_parquet(_tx_frame(50), C.FACT_DIR / "fact_transaction.parquet", "fact_transaction")
    write_parquet(pd.DataFrame({"statement_id": ["S1"]}), C.FACT_DIR / "fact_statement.parquet")
    write_parquet(pd.DataFrame({"customer_id": ["C1"]}), C.DIM_DIR / "dim_customer.parquet")
    return tmp_path


def test_refresh_never_pandas_loads_fact_transaction(tiny_out, monkeypatch):
    seen = []
    real = refresh_docs.read_parquet

    def spy(path):
        seen.append(path.name)
        return real(path)

    monkeypatch.setattr(refresh_docs, "read_parquet", spy)
    dims, facts, raw = refresh_docs.load_all()
    assert "fact_transaction.parquet" not in seen
    assert isinstance(facts["fact_transaction"], pl.LazyFrame)
    assert "fact_statement.parquet" in seen  # other tables still load as before


def test_dictionary_lazy_equals_pandas(tiny_out):
    pdf = pd.read_parquet(C.FACT_DIR / "fact_transaction.parquet")
    lazy = pl.scan_parquet(C.FACT_DIR / "fact_transaction.parquet")
    dd_p, md_p = dictionary.build({"fact_transaction": pdf})
    dd_l, md_l = dictionary.build({"fact_transaction": lazy})
    pd.testing.assert_frame_equal(dd_l, dd_p)
    assert md_l == md_p
