"""Independent reconciliation checks over the emitted tables.

These checks deliberately do not import ``build.validation`` (or the producer
helpers it uses). They recompute a property straight from the published
tables, so a defect in a producer cannot also define the gate that is meant to
catch it.

``CHECKS`` accumulates ``(name, ok, detail)`` tuples so a caller can inspect
every result instead of only the last one.
"""
from __future__ import annotations

import polars as pl

CHECKS: list[tuple[str, bool, str]] = []


def _record(name: str, ok: bool, detail: str) -> bool:
    CHECKS.append((name, bool(ok), detail))
    return bool(ok)


def check_continuity(statements: pl.DataFrame | object, tolerance: float = 0.01) -> bool:
    """Every statement must open at the previous statement's closing balance.

    Statements are grouped per account and ordered by statement date. The first
    statement of an account is not constrained (it may inherit an opening
    balance from before the fact window); every later statement must continue
    the previous closing balance within ``tolerance``.
    """
    frame = (statements if isinstance(statements, pl.DataFrame)
             else pl.from_pandas(statements))
    ordered = frame.sort(["account_id", "statement_date"]).with_columns(
        pl.col("closing_balance").shift(1).over("account_id").alias("_prev_closing")
    )
    continuations = ordered.filter(pl.col("_prev_closing").is_not_null())
    with_prev = pl.col("_prev_closing")
    bad = continuations.filter(
        pl.col("opening_balance").is_null()
        | with_prev.is_null()
        | ((pl.col("opening_balance") - with_prev).abs() > tolerance)
    )
    problems = [
        f"{row['account_id']}@{row['statement_date']}: opening "
        f"{row['opening_balance']} != previous closing {row['_prev_closing']}"
        for row in bad.iter_rows(named=True)
    ]
    ok = not problems
    detail = (f"{continuations.height} statement continuations checked" if ok
              else "; ".join(problems[:10]))
    return _record("statement continuity", ok, detail)
