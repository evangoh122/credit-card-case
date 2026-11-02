"""Regressions for the independent reconciliation checks.

These tests feed Polars frames directly so they never depend on the pandas
pipeline. They pin the fail-closed contract of ``check_continuity``: a missing
previous closing balance and a run with zero comparisons must both fail.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl
import pytest


def _load_checks():
    project_root = Path(__file__).resolve().parents[1]
    checks_path = project_root / "checks" / "independent_checks.py"
    spec = importlib.util.spec_from_file_location("independent_checks", checks_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _statements(closes, openings=(0.0, 100.0, 120.0)):
    return pl.DataFrame({
        "account_id": ["A"] * len(closes),
        "statement_date": [f"2026-{month:02d}-01"
                           for month in range(1, len(closes) + 1)],
        "opening_balance": list(openings),
        "closing_balance": list(closes),
    })


@pytest.mark.parametrize("closes", [[None, 120.0, 140.0], [None, None, None]])
def test_missing_previous_closing_balance_cannot_pass(closes):
    """A later statement with no previous closing has no provable continuity."""
    mod = _load_checks()
    mod.CHECKS.clear()
    assert mod.check_continuity(_statements(closes)) is False
    assert mod.CHECKS[-1][1] is False


def test_zero_continuation_comparisons_cannot_pass():
    """One statement per account performs no comparison and must not pass."""
    mod = _load_checks()
    mod.CHECKS.clear()
    assert mod.check_continuity(_statements([100.0], openings=(0.0,))) is False
    assert "0 statement continuations" in mod.CHECKS[-1][2]


def test_complete_continuity_passes_and_reports_comparisons():
    """Positive control: a genuinely continuous account still passes."""
    mod = _load_checks()
    mod.CHECKS.clear()
    assert mod.check_continuity(_statements([100.0, 120.0, 140.0])) is True
    assert "2 statement continuations checked" in mod.CHECKS[-1][2]
