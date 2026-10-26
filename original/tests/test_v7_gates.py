"""Mutation tests for the v7 release gates.

Every test below seeds the defect that the gate exists to catch and asserts the
production gate helper fails on it. The tests do not merely recompute the same
formula: each one corrupts an input (cash reuse, a broken roll, an out-of-band
loss, too few charge-offs, a post-cutoff feature date, a post-hash mutation) and
requires the gate to notice.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path

import pandas as pd

from build import config as C
from build import validation
from build.facts import _settled_closing_amounts
from build.ml_extracts import check_drift, check_no_leakage, check_pr_auc
from build.run_all import verify_manifest


def _stmt(rows):
    return pd.DataFrame(rows)


def test_settlement_is_cash_consuming():
    stmts = [
        {"statement_date": dt.date(2026, 1, 1), "closing_balance": 100.0},
        {"statement_date": dt.date(2026, 2, 1), "closing_balance": 200.0},
    ]
    cash = [(dt.date(2026, 1, 15), 50.0), (dt.date(2026, 2, 15), 100.0)]
    assert _settled_closing_amounts(stmts, cash) == [100.0, 50.0]
    assert sum(_settled_closing_amounts(stmts, cash)) <= sum(a for _, a in cash)


def test_settlement_gate_catches_v6_cash_reuse():
    """Seeded defect: v6 reused the whole later-payment suffix per statement.

    The second statement claims $200 although only $150 of cash exists, so the
    seed is an over-report and the gate must return False.
    """
    stmt = _stmt([
        dict(account_id="A", statement_id="S1", statement_date="2026-01-01",
             paid_closing_amount=100.0, closing_balance=100.0),
        dict(account_id="A", statement_id="S2", statement_date="2026-02-01",
             paid_closing_amount=200.0, closing_balance=200.0),
    ])
    pay = pd.DataFrame([
        dict(account_id="A", payment_type="CARD", amount_usd=50.0),
        dict(account_id="A", payment_type="CARD", amount_usd=100.0),
    ])
    ok, detail = validation.check_settlement_cash_conservation(stmt, pay)
    assert not ok, detail


def test_settlement_gate_accepts_conserving_data():
    stmt = _stmt([
        dict(account_id="A", statement_id="S1", statement_date="2026-01-01",
             paid_closing_amount=100.0, closing_balance=100.0),
        dict(account_id="A", statement_id="S2", statement_date="2026-02-01",
             paid_closing_amount=50.0, closing_balance=200.0),
    ])
    pay = pd.DataFrame([dict(account_id="A", payment_type="CARD", amount_usd=150.0)])
    ok, detail = validation.check_settlement_cash_conservation(stmt, pay)
    assert ok, detail


def test_continuity_gate_catches_broken_roll():
    """Seeded defect: opening(i+1) is $1 away from closing(i)."""
    stmt = _stmt([
        dict(account_id="A", statement_date="2026-01-01", opening_balance=0.0,
             closing_balance=100.0),
        dict(account_id="A", statement_date="2026-02-01", opening_balance=101.0,
             closing_balance=120.0),
    ])
    ok, detail = validation.check_statement_continuity(stmt)
    assert not ok, detail


def test_continuity_gate_accepts_correct_roll():
    stmt = _stmt([
        dict(account_id="A", statement_date="2026-01-01", opening_balance=0.0,
             closing_balance=100.0),
        dict(account_id="A", statement_date="2026-02-01", opening_balance=100.0,
             closing_balance=120.0),
    ])
    ok, detail = validation.check_statement_continuity(stmt)
    assert ok, detail


def test_fraud_band_gate_catches_out_of_band_loss():
    """Seeded defect: net loss is 20 bps of approved purchase spend."""
    tx = pd.DataFrame([
        dict(status="APPROVED", transaction_type="PURCHASE", amount_usd=1_000_000.0),
    ])
    fraud = pd.DataFrame([dict(net_loss=2_000.0)])
    ok, detail = validation.check_fraud_loss_band(fraud, tx)
    assert not ok, detail


def test_fraud_band_gate_accepts_7bps():
    tx = pd.DataFrame([
        dict(status="APPROVED", transaction_type="PURCHASE", amount_usd=1_000_000.0),
    ])
    fraud = pd.DataFrame([dict(net_loss=700.0)])
    ok, detail = validation.check_fraud_loss_band(fraud, tx)
    assert ok, detail


def test_chargeoff_gate_catches_v6_incidence():
    """Seeded defect: only 4 charge-offs, as measured on v6."""
    acct = pd.DataFrame(dict(account_status=["CHARGED_OFF"] * 4 + ["ACTIVE"] * 100))
    ok, detail = validation.check_chargeoff_incidence(acct)
    assert not ok, detail


def test_chargeoff_gate_accepts_calibrated_incidence():
    acct = pd.DataFrame(dict(account_status=["CHARGED_OFF"] * C.CHARGE_OFF_MIN_COUNT))
    ok, detail = validation.check_chargeoff_incidence(acct)
    assert ok, detail


def test_accepted_limitation_gate_requires_entry(tmp_path):
    ok, _ = validation.check_accepted_limitations(tmp_path)
    assert not ok
    (tmp_path / "accepted_limitations.md").write_text(
        "# Accepted limitations\n\n## auth_code\n\nAccepted limitation: codes repeat.\n",
        encoding="utf-8")
    ok, detail = validation.check_accepted_limitations(tmp_path)
    assert ok, detail


def test_manifest_verifier_catches_post_hash_mutation(tmp_path):
    """Seeded defect: a hashed file is changed after the manifest is written."""
    out = tmp_path
    (out / "dim").mkdir()
    (out / "docs").mkdir()
    (out / "ml").mkdir()
    f = out / "dim" / "dim_date.parquet"
    f.write_bytes(b"v7")
    import hashlib
    digest = hashlib.sha256(b"v7").hexdigest()
    (out / "release_manifest.json").write_text(json.dumps({
        "files": [{"table": "dim_date", "layer": "dim", "sha256": digest}],
        "docs": [], "ml": [],
    }), encoding="utf-8")
    assert verify_manifest(out) == []
    f.write_bytes(b"v7-mutated")
    assert verify_manifest(out) == ["dim/dim_date.parquet"]


def test_leakage_gate_catches_post_observation_feature():
    """Seeded defect: max_source_date is one day after the observation date."""
    good = pd.DataFrame(dict(observation_date=["2026-03-31"],
                             max_source_date=["2026-03-31"]))
    ok, detail = check_no_leakage(good)
    assert ok, detail
    bad = pd.DataFrame(dict(observation_date=["2026-03-31"],
                            max_source_date=["2026-04-01"]))
    ok, detail = check_no_leakage(bad)
    assert not ok, detail


def test_ml_acceptance_gates_catch_bad_values():
    assert check_pr_auc(0.97)[0] is False
    assert check_pr_auc(0.60)[0] is True
    assert check_drift(0.05)[0] is False
    assert check_drift(0.30)[0] is True


def test_independent_continuity_check_catches_seeded_defect():
    spec = importlib.util.spec_from_file_location(
        "independent_checks", Path(__file__).resolve().parents[1] / "checks"
        / "independent_checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    stmt = pd.DataFrame([
        dict(account_id="A", statement_date=pd.Timestamp("2026-01-01"),
             opening_balance=0.0, closing_balance=100.0),
        dict(account_id="A", statement_date=pd.Timestamp("2026-02-01"),
             opening_balance=100.0, closing_balance=120.0),
    ])
    mod.CHECKS.clear()
    mod.check_continuity(stmt)
    assert mod.CHECKS[-1][1] is True
    mod.CHECKS.clear()
    stmt.loc[1, "opening_balance"] = 101.0
    mod.check_continuity(stmt)
    assert mod.CHECKS[-1][1] is False
