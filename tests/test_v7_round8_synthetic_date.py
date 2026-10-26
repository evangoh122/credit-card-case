"""Round 8: backdated synthetic fraud rows are internally consistent around the OOT boundary."""
from __future__ import annotations

import pandas as pd
import pytest

from build import config as C
from build import ml_extracts

OOT = pd.Timestamp(C.ML_FRAUD_OOT_START)          # 2026-07-01
assert OOT == pd.Timestamp("2026-07-01")


@pytest.mark.parametrize("real_ts,expect_ts", [
    ("2026-07-01 00:30", "2026-06-30 22:00"),   # real txn just after OOT start -> backdated before it
    ("2026-06-30 23:30", "2026-06-29 22:00"),   # real txn just before OOT start
    ("2026-07-02 09:00", "2026-07-01 22:00"),   # backdated row lands on the OOT day itself
])
def test_backdated_timestamp_date_and_gap(real_ts, expect_ts):
    fraud_date = pd.Timestamp(real_ts).normalize()      # build_fraud passes the fraud's date
    ts = ml_extracts._backdated_test_purchase_ts(fraud_date)
    assert ts == pd.Timestamp(expect_ts)
    assert ts.normalize() == fraud_date - pd.Timedelta(days=1)   # one calendar day before the fraud
    assert ts.hour == 22


def _row(ts, txn_date, split_time):
    return {"customer_id": "A", "transaction_id": f"SYN{pd.Timestamp(ts):%m%d%H}", "transaction_ts": pd.Timestamp(ts),
            "transaction_date": pd.Timestamp(txn_date), "amount_usd": 2.0, "hour": 3,
            "is_card_present": True, "is_international": False, "category": "Grocery",
            "customer_day_txn_count": 1, "customer_week_spend": 5.0, "days_since_first_txn": 0,
            "split_time": split_time, "device_risk_score": 0.5}


def _hist():
    return pd.DataFrame({"customer_id": ["A"], "transaction_id": ["h"],
                         "transaction_ts": [pd.Timestamp("2026-06-01 10:00")],
                         "transaction_date": [pd.Timestamp("2026-06-01")], "amount_usd": [10.0]})


def test_inconsistent_inputs_are_rederived_from_the_timestamp():
    """The pre-round-8 shape: date one day AHEAD of ts, hour copied from the fraud row."""
    syn = pd.DataFrame([
        _row("2026-06-30 22:00", "2026-07-01", "test"),     # ts before OOT, date on/after it
        _row("2026-07-01 22:00", "2026-07-02", "test"),     # ts after OOT start
        _row("2026-06-30 23:30", "2026-07-01", "test"),
    ])
    out = ml_extracts._asof_synthetic_features(_hist(), syn)
    assert (out.transaction_date == pd.to_datetime(out.transaction_ts).dt.normalize()).all()
    assert out.hour.tolist() == [22, 22, 23]
    assert out.split_time.tolist() == ["train", "test", "train"]    # from the timestamp
    # device-risk regime shift (+0.42) follows the same timestamp, not the date
    assert out.device_risk_score.iloc[1] - out.device_risk_score.iloc[0] > 0.25
    assert out.device_risk_score.iloc[1] - out.device_risk_score.iloc[2] > 0.25


@pytest.mark.parametrize("fraud_ts", ["2026-07-01 00:30", "2026-06-30 23:30", "2026-07-02 00:00",
                                      "2026-06-15 12:00", "2026-01-01 00:00"])
def test_end_to_end_helper_plus_features_are_self_consistent(fraud_ts):
    fraud_date = pd.Timestamp(fraud_ts).normalize()
    ts = ml_extracts._backdated_test_purchase_ts(fraud_date)
    row = _row(ts, ts.normalize(), "x")
    out = ml_extracts._asof_synthetic_features(_hist(), pd.DataFrame([row])).iloc[0]
    assert out.transaction_date == pd.Timestamp(out.transaction_ts).normalize()
    assert out.hour == pd.Timestamp(out.transaction_ts).hour
    assert out.split_time == ("test" if pd.Timestamp(out.transaction_ts) >= OOT else "train")
    assert out.transaction_date == fraud_date - pd.Timedelta(days=1)
