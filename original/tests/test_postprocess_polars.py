"""Regression: Polars postprocess must match the former pandas logic exactly.

`_pandas_reference` is the pre-v7-memory-fix implementation, embedded verbatim
as the oracle. The synthetic frame hits every rule plus the edge cases (null
status, unknown merchant / account, null merchant flag, sort ties and dates).
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import pytest

from build import config as C
from build import postprocess
from build.utils import write_parquet


def _pandas_reference(tx, merch, acct, cards):
    approved = tx.status == "APPROVED"
    tx.loc[~approved, "auth_code"] = None
    tx.loc[approved, "decline_reason_code"] = None
    tx.loc[approved, "decline_reason"] = None
    recurring = merch.set_index("merchant_id")["is_recurring"]
    tx["is_recurring"] = tx.merchant_id.map(recurring).fillna(False).astype(bool)
    net = acct.set_index("account_id")["network"]
    tx["network"] = tx.account_id.map(net)
    tx["transaction_date"] = pd.to_datetime(tx.transaction_date)
    tx = tx.sort_values(["transaction_date", "transaction_ts"]).reset_index(drop=True)
    dead = set(acct.loc[acct.account_status.isin(["CLOSED", "CHARGED_OFF"]), "account_id"])
    mask = cards.account_id.isin(dead) & (cards.card_status == "ACTIVE")
    cards.loc[mask, "card_status"] = "CANCELLED"
    return tx, cards, int(mask.sum())


def _frames():
    d = dt.date
    ts = lambda day, h: dt.datetime(2026, 1, day, h)
    tx = pd.DataFrame({
        "transaction_id": [f"T{i}" for i in range(8)],
        "account_id": ["A1", "A2", "A3", "A1", "A9", None, "A2", "A3"],
        "merchant_id": ["M1", "M2", "M3", "M9", "M1", "M2", None, "M1"],
        "transaction_date": [d(2026, 1, 5), d(2026, 1, 3), d(2026, 1, 3), d(2026, 1, 3),
                             d(2026, 1, 4), d(2026, 1, 3), d(2026, 1, 5), d(2026, 1, 5)],
        "transaction_ts": [ts(5, 9), ts(3, 12), ts(3, 8), ts(3, 8), ts(4, 1), ts(3, 20),
                           ts(5, 9), ts(5, 2)],
        "status": ["APPROVED", "DECLINED", "REVERSED", "APPROVED", None, "DECLINED",
                   "APPROVED", "DECLINED"],
        "auth_code": ["AC0", "AC1", "AC2", "AC3", "AC4", "AC5", "AC6", "AC7"],
        "decline_reason_code": ["D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7"],
        "decline_reason": ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"],
        "is_recurring": [True, False, True, False, True, False, True, False],
        "amount_usd": [10.5, 20.0, 30.25, 40.0, 50.0, 60.0, 70.0, 80.75],
    })
    merch = pd.DataFrame({"merchant_id": ["M1", "M2", "M3"],
                          "is_recurring": pd.array([True, False, None], dtype="boolean")})
    acct = pd.DataFrame({
        "account_id": ["A1", "A2", "A3"],
        "network": ["VISA", "MASTERCARD", None],
        "account_status": ["OPEN", "CLOSED", "CHARGED_OFF"],
    })
    cards = pd.DataFrame({
        "card_id": ["C1", "C2", "C3", "C4", "C5"],
        "account_id": ["A1", "A2", "A2", "A3", None],
        "card_status": ["ACTIVE", "ACTIVE", "BLOCKED", "ACTIVE", "ACTIVE"],
    })
    return tx, merch, acct, cards


def _assert_same(got: pd.DataFrame, want: pd.DataFrame):
    assert len(got) == len(want)
    assert list(got.columns) == list(want.columns)
    assert got.dtypes.astype(str).to_dict() == want.dtypes.astype(str).to_dict()
    pd.testing.assert_frame_equal(got, want, check_dtype=False)
    assert got.transaction_id.tolist() == want.transaction_id.tolist()  # ordering
    assert got.amount_usd.sum() == want.amount_usd.sum()


def _run_polars(tx, merch, acct, cards):
    lf = postprocess.hygiene(pl.from_pandas(tx).lazy(),
                             pl.from_pandas(merch.astype({"is_recurring": "object"})),
                             pl.from_pandas(acct))
    out_cards, n = postprocess.cancel_cards(pl.from_pandas(cards), pl.from_pandas(acct))
    return lf.collect().to_pandas(), out_cards.to_pandas(), n


def test_polars_hygiene_matches_pandas_reference():
    want_tx, want_cards, want_n = _pandas_reference(*_frames())
    got_tx, got_cards, got_n = _run_polars(*_frames())
    _assert_same(got_tx, want_tx)
    _assert_same_cards = got_cards.equals(want_cards)
    assert _assert_same_cards and got_n == want_n == 2
    assert got_tx.is_recurring.dtype == bool
    assert got_tx.transaction_date.dtype == "datetime64[ns]"
    # rule spot checks (independent of the oracle)
    approved = got_tx.status == "APPROVED"
    assert got_tx.loc[~approved, "auth_code"].isna().all()
    assert got_tx.loc[approved, ["decline_reason_code", "decline_reason"]].isna().all().all()
    assert got_tx.loc[approved, "auth_code"].notna().all()


def test_main_end_to_end_streams_and_keeps_metadata(tmp_path, monkeypatch):
    tx, merch, acct, cards = _frames()
    monkeypatch.setattr(C, "FACT_DIR", tmp_path / "fact")
    monkeypatch.setattr(C, "DIM_DIR", tmp_path / "dim")
    tx_p = C.FACT_DIR / "fact_transaction.parquet"
    write_parquet(tx.copy(), tx_p, "fact_transaction")
    write_parquet(merch, C.DIM_DIR / "dim_merchant.parquet", "dim_merchant")
    write_parquet(acct, C.DIM_DIR / "dim_account.parquet", "dim_account")
    write_parquet(cards.copy(), C.DIM_DIR / "dim_card.parquet", "dim_card")

    assert postprocess.main() == 0

    want_tx, want_cards, _ = _pandas_reference(*_frames())
    got = pq.read_table(str(tx_p))
    _assert_same(got.to_pandas(), want_tx)
    md = got.schema.metadata
    assert md[b"dataset"] == b"fact_transaction" and md[b"seed"] == str(C.SEED).encode()
    assert pq.ParquetFile(str(tx_p)).metadata.row_group(0).column(0).compression == "SNAPPY"
    assert pq.read_table(str(C.DIM_DIR / "dim_card.parquet")).to_pandas().equals(want_cards)
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("skip", ["auth", "decline", "recurring", "network", "sort", "cards"])
def test_reference_would_catch_each_rule_break(skip):
    """The comparison is sensitive to each rule: break one in the oracle's copy."""
    want_tx, want_cards, _ = _pandas_reference(*_frames())
    got_tx, got_cards, _ = _run_polars(*_frames())
    if skip == "auth":
        got_tx["auth_code"] = _frames()[0].set_index("transaction_id").loc[
            got_tx.transaction_id, "auth_code"].to_numpy()
    elif skip == "decline":
        got_tx["decline_reason"] = "x"
    elif skip == "recurring":
        got_tx["is_recurring"] = ~got_tx.is_recurring
    elif skip == "network":
        got_tx["network"] = None
    elif skip == "sort":
        got_tx = got_tx.iloc[::-1].reset_index(drop=True)
    else:
        got_cards = _frames()[3]
    with pytest.raises(AssertionError):
        if skip == "cards":
            assert got_cards.equals(want_cards)
        else:
            _assert_same(got_tx, want_tx)
