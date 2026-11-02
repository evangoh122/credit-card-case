"""Independent reconciliation checks over the emitted tables.

These checks deliberately do not import ``build.validation`` (or the producer
helpers it uses). They recompute a property straight from the published
tables, so a defect in a producer cannot also define the gate that is meant to
catch it.

``CHECKS`` accumulates ``(name, ok, detail)`` tuples so a caller can inspect
every result instead of only the last one.
"""
from __future__ import annotations

import pandas as pd

CHECKS: list[tuple[str, bool, str]] = []


def _record(name: str, ok: bool, detail: str) -> bool:
    CHECKS.append((name, bool(ok), detail))
    return bool(ok)


def check_continuity(statements: pd.DataFrame, tolerance: float = 0.01) -> bool:
    """Every statement must open at the previous statement's closing balance.

    Statements are grouped per account and ordered by statement date. The first
    statement of an account is not constrained (it may inherit an opening
    balance from before the fact window); every later statement must continue
    the previous closing balance within ``tolerance``.
    """
    ordered = statements.sort_values(["account_id", "statement_date"])
    problems: list[str] = []
    checked = 0
    for account_id, group in ordered.groupby("account_id", sort=False):
        previous_closing = None
        for row in group.itertuples():
            if previous_closing is not None:
                checked += 1
                opening = row.opening_balance
                if pd.isna(previous_closing) or pd.isna(opening):
                    problems.append(
                        f"{account_id}@{row.statement_date}: missing balance")
                elif abs(float(opening) - float(previous_closing)) > tolerance:
                    problems.append(
                        f"{account_id}@{row.statement_date}: opening "
                        f"{float(opening):.2f} != previous closing "
                        f"{float(previous_closing):.2f}")
            previous_closing = row.closing_balance
    ok = not problems
    detail = (f"{checked} statement continuations checked" if ok
              else "; ".join(problems[:10]))
    return _record("statement continuity", ok, detail)
