"""Round 5: credit-default payment features must be as-of the observation date.

CONFIRMED finding: ``fact_statement.payment_status`` is settled with ALL later
card cash (facts._settled_closing_amounts / _stmt_status), so a statement dated
before the observation date can be PAID_FULL because of a payment made after it.
``status_at_obs`` and ``paid_full_share_3m`` are therefore recomputed from
payment allocations dated <= observation date.
"""
from __future__ import annotations

import pandas as pd
import pytest

from build import config as C
from build import facts, ml_extracts

OBS = pd.Timestamp(C.ML_OBSERVATION_DATE)


def d(days):
    return OBS + pd.Timedelta(days=days)


def _statements(final_status="PAID_FULL"):
    rows = [
        # S0: cut 2026-02-09, due before obs, paid before obs
        ("S0", d(-50), d(-25), 80.0, 80.0, 8.0),
        # S1: cut 2026-03-11, due AFTER obs, paid only after obs
        ("S1", d(-20), d(5), 100.0, 100.0, 10.0),
    ]
    return pd.DataFrame({
        "statement_id": [r[0] for r in rows], "account_id": "AC1", "customer_id": "C1",
        "product_id": "P", "statement_date": [r[1] for r in rows],
        "payment_due_date": [r[2] for r in rows], "closing_balance": [r[3] for r in rows],
        "billed_amount": [r[4] for r in rows], "minimum_due": [r[5] for r in rows],
        "opening_balance": 0.0, "purchases_amount": 50.0, "payments_amount": 5.0,
        "interest_charged": 0.0, "fees_charged": 0.0, "credit_limit": 1000.0,
        "utilization_pct": 10.0, "payment_status": final_status,
    })


def _alloc(rows):
    return pd.DataFrame(rows, columns=["statement_id", "allocation_date", "amount_usd"])


BASE_ALLOC = [("S0", d(-30), 80.0), ("S1", d(20), 100.0)]  # S1 settled AFTER obs


def _run(monkeypatch, tmp_path, statements, allocations):
    out = {}
    monkeypatch.setattr(ml_extracts, "write_parquet",
                        lambda df, path, name=None: out.__setitem__(name, df.copy()))
    monkeypatch.setattr(ml_extracts, "ML", tmp_path)
    accounts = pd.DataFrame({"account_id": ["AC1"], "customer_id": ["C1"], "product_id": "P",
                             "open_date": d(-300), "account_status": "OPEN",
                             "closed_date": pd.NaT})
    customers = pd.DataFrame({"customer_id": ["C1"], "credit_score": [700],
                              "income_annual": [5e4], "customer_segment": "A",
                              "risk_tier": "L"})
    delinq = pd.DataFrame({"account_id": ["AC1"], "snapshot_date": [d(-20)],
                           "days_past_due": [0], "delinquency_status": "CURRENT"})
    tx = pd.DataFrame({"transaction_id": ["T1"], "account_id": ["AC1"], "customer_id": ["C1"],
                       "amount_usd": [10.0], "transaction_date": [d(-3)],
                       "is_international": [False]})
    ml_extracts.build_credit_default(tx, accounts, customers, statements, delinq, allocations)
    return out["credit_default_features"].iloc[0]


# ---- (1) confirmation: the final statement status is settled by post-obs cash ----
def test_confirmed_final_status_uses_cash_after_observation():
    stmt = {"statement_date": d(-20), "closing_balance": 100.0}
    post_obs_cash = [(d(20), 100.0)]
    settled = facts._settled_closing_amounts([stmt], post_obs_cash)
    assert settled == [100.0]  # a payment 20 days AFTER obs settles an at-obs statement
    s = dict(closing_balance=100.0, billed=100.0, settled=settled[0], min_eff=10.0,
             alloc_min=10.0, alloc_bal=100.0, due_date=d(5), paid_by_due=True)
    assert facts._stmt_status(s, C.PERIOD_END)[0] == "PAID_FULL"  # what the old feature read


def test_old_feature_would_leak_asof_feature_does_not(monkeypatch, tmp_path):
    f = _run(monkeypatch, tmp_path, _statements("PAID_FULL"), _alloc(BASE_ALLOC))
    assert f.status_at_obs == "PENDING"   # as-of: S1 unpaid at obs, due after obs
    assert f.paid_full_share_3m == pytest.approx(0.5)  # only S0 was paid by obs
    assert _statements("PAID_FULL").payment_status.eq("PAID_FULL").mean() == 1.0  # old value


# ---- (2) invariance: nothing after the observation date may change features ----
@pytest.mark.parametrize("post_obs", [
    [("S1", d(20), 100.0)],                       # baseline
    [("S1", d(45), 100.0)],                       # shifted later (still post-obs)
    [("S1", d(2), 30.0), ("S1", d(60), 70.0)],    # split post-obs payments
    [],                                           # removed entirely
    [("S1", d(1), 999.0)],                        # different amount
])
def test_features_invariant_to_post_observation_payments(monkeypatch, tmp_path, post_obs):
    ref = _run(monkeypatch, tmp_path, _statements("PAID_FULL"),
               _alloc([("S0", d(-30), 80.0)] + [("S1", d(20), 100.0)]))
    got = _run(monkeypatch, tmp_path, _statements("UNPAID"),  # final status also differs
               _alloc([("S0", d(-30), 80.0)] + post_obs))
    pd.testing.assert_series_equal(got, ref)


def test_pre_observation_payment_does_change_features(monkeypatch, tmp_path):
    """Sensitivity: the invariance tests are not vacuous."""
    paid_before = _run(monkeypatch, tmp_path, _statements(),
                       _alloc([("S0", d(-30), 80.0), ("S1", d(-1), 100.0)]))
    paid_after = _run(monkeypatch, tmp_path, _statements(),
                      _alloc([("S0", d(-30), 80.0), ("S1", d(1), 100.0)]))  # shifted past obs
    assert paid_before.status_at_obs == "PAID_FULL" and paid_after.status_at_obs == "PENDING"
    assert paid_before.paid_full_share_3m != paid_after.paid_full_share_3m


# ---- (3) gate -------------------------------------------------------------
def test_max_source_date_tracks_payment_dates(monkeypatch, tmp_path):
    # a payment dated AFTER the latest statement/snapshot (d(-20)) but before obs
    f = _run(monkeypatch, tmp_path, _statements(),
             _alloc([("S0", d(-30), 80.0), ("S1", d(-2), 40.0), ("S1", d(20), 60.0)]))
    assert f.payment_feature_date == d(-2)
    assert f.max_source_date == d(-2)   # payment date, not just statement/snapshot dates
    assert f.max_source_date <= OBS


def test_payment_gate_fails_on_post_observation_event():
    ok, _ = ml_extracts.check_payment_sources_asof(_alloc([("S1", d(-1), 1.0)]), OBS)
    assert ok
    ok, detail = ml_extracts.check_payment_sources_asof(_alloc([("S1", d(1), 1.0)]), OBS)
    assert not ok and "1 payment events after observation" in detail
    # and the shared date gate catches a payment date leaking into max_source_date
    leaky = pd.DataFrame({"observation_date": [OBS], "max_source_date": [d(3)]})
    assert not ml_extracts.check_no_leakage(leaky)[0]


def test_asof_status_matches_facts_rules_on_edge_rows():
    hist = pd.DataFrame({
        "statement_id": list("ABCDEFG"),
        "closing_balance": [-5.0, 0.0, 50.0, 100.0, 100.0, 100.0, 100.0],
        "billed_amount": [10.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0],
        "minimum_due": [0.0, 0.0, 5.0, 10.0, 10.0, 10.0, 10.0],
        "payment_due_date": [d(-5)] * 4 + [d(-5), d(-5), d(9)],
    })
    alloc = _alloc([("D", d(-1), 100.0), ("E", d(-1), 15.0), ("F", d(-1), 4.0),
                    ("G", d(1), 100.0)])
    status, _, _ = ml_extracts.asof_payment_status(hist, alloc, OBS)
    assert status.tolist() == ["CREDIT_BALANCE", "NO_BALANCE", "CARRY_FORWARD", "PAID_FULL",
                               "PAID_PARTIAL", "PARTIAL", "PENDING"]


# ---- (4) audit: churn extract reads no settled/payment-status field -----------
SETTLED_FIELDS = {"payment_status", "paid_amount", "paid_closing_amount",
                  "last_payment_date", "is_late"}


def _churn(monkeypatch, tmp_path, statements):
    out = {}
    monkeypatch.setattr(ml_extracts, "write_parquet",
                        lambda df, path, name=None: out.__setitem__(name, df.copy()))
    monkeypatch.setattr(ml_extracts, "ML", tmp_path)
    customers = pd.DataFrame({"customer_id": ["C1"], "onboarding_date": [d(-400)],
                              "credit_score": [700], "income_annual": [5e4],
                              "customer_segment": "A", "risk_tier": "L",
                              "customer_status": "ACTIVE", "churn_date": [pd.NaT]})
    tx = pd.DataFrame({"transaction_id": ["T1"], "account_id": ["AC1"], "customer_id": ["C1"],
                       "merchant_id": ["M1"], "mcc": ["5411"], "category": ["grocery"],
                       "amount_usd": [10.0], "transaction_date": [d(-3)],
                       "is_international": [False]})
    ml_extracts.build_churn(tx, customers, statements)
    return out["churn_features"].iloc[0]


def test_churn_features_use_no_settled_statement_fields(monkeypatch, tmp_path):
    a = _churn(monkeypatch, tmp_path, _statements("PAID_FULL"))
    scrambled = _statements("UNPAID")
    scrambled["paid_amount"], scrambled["is_late"] = 9e9, True
    b = _churn(monkeypatch, tmp_path, scrambled)
    assert not SETTLED_FIELDS & set(a.index)
    pd.testing.assert_series_equal(a, b)
