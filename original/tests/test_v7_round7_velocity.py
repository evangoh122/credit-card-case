"""Round 7: exact (ts-7d, ts] boundaries on real timestamps; as-of synthetic fraud rows."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from build import config as C
from build import ml_extracts

T = pd.Timestamp("2026-02-02 10:00")
D7 = pd.Timedelta(days=7)


def _tx(rows):
    df = pd.DataFrame(rows, columns=["customer_id", "transaction_id", "transaction_ts", "amount_usd"])
    df["transaction_date"] = df.transaction_ts.dt.normalize()
    return df


def _vel(rows):
    return ml_extracts._attach_velocity(_tx(rows)).set_index("transaction_id")


def test_reviewer_reproduction_equal_ts_then_exact_7d_boundary():
    v = _vel([("A", "1", T, 10.0), ("A", "2", T, 20.0), ("A", "3", T + D7, 30.0)])
    # row 3 is exactly 7d after rows 1/2: they are OUTSIDE (ts-7d, ts]
    assert v.customer_week_spend.to_dict() == {"1": 10.0, "2": 30.0, "3": 30.0}


def test_row_one_second_inside_the_boundary_is_included():
    v = _vel([("A", "1", T, 10.0), ("A", "2", T, 20.0), ("A", "3", T + D7 - pd.Timedelta(seconds=1), 30.0)])
    assert v.customer_week_spend["3"] == 60.0


def test_duplicate_timestamps_exactly_7d_before_are_all_excluded():
    rows = [("A", f"{i}", T, 1.0) for i in range(5)] + [("A", "9", T + D7, 100.0),
                                                       ("B", "b", T, 7.0)]
    v = _vel(rows)
    assert v.customer_week_spend["9"] == 100.0
    assert v.customer_week_spend["4"] == 5.0           # the fifth equal-ts row sees ids 0..4
    assert v.customer_week_spend["b"] == 7.0           # other customer never leaks in


def test_equal_timestamps_are_ordered_by_id_not_by_input_order():
    v = ml_extracts._attach_velocity(_tx([("A", "2", T, 20.0), ("A", "1", T, 10.0)]))
    assert list(v.transaction_id) == ["2", "1"]        # input row order preserved
    v = v.set_index("transaction_id")
    assert v.customer_week_spend.to_dict() == {"1": 10.0, "2": 30.0}
    assert v.customer_day_txn_count.to_dict() == {"1": 1, "2": 2}


def test_brute_force_with_duplicate_and_exact_boundary_timestamps():
    r = np.random.default_rng(11)
    # timestamps drawn from a small grid so duplicates and exact 7-day offsets are common
    grid = [T + pd.Timedelta(days=int(k)) for k in range(0, 22, 1)]
    n = 400
    rows = [(f"C{r.integers(0, 5)}", f"T{i:04d}", grid[r.integers(0, len(grid))],
             float(r.integers(1, 50))) for i in range(n)]
    df = _tx(rows)
    got = ml_extracts._attach_velocity(df).set_index("transaction_id")
    boundary_hits = dup_hits = 0
    for tid, row in df.set_index("transaction_id").iterrows():
        mine = df[(df.customer_id == row.customer_id)
                  & ((df.transaction_ts < row.transaction_ts)
                     | ((df.transaction_ts == row.transaction_ts) & (df.transaction_id <= tid)))]
        week = mine[mine.transaction_ts > row.transaction_ts - D7]
        day = mine[mine.transaction_date == row.transaction_date]
        assert got.loc[tid, "customer_week_spend"] == pytest.approx(week.amount_usd.sum())
        assert got.loc[tid, "customer_day_txn_count"] == len(day)
        boundary_hits += int(((mine.transaction_ts == row.transaction_ts - D7)).any())
        dup_hits += int(((df.customer_id == row.customer_id)
                         & (df.transaction_ts == row.transaction_ts)).sum() > 1)
    assert boundary_hits > 20 and dup_hits > 20      # the data really exercises both edges


# ------------------------------------------------ synthetic backdated fraud rows
def _history():
    return _tx([("A", "h1", T - pd.Timedelta(days=3), 40.0),
                ("A", "h2", T - pd.Timedelta(days=1), 60.0),
                ("A", "fraud", T + pd.Timedelta(days=2), 900.0),    # the LATER real fraud row
                ("B", "b1", T, 5.0)])


def _syn():
    s = pd.DataFrame({"customer_id": ["A"], "transaction_id": ["SYNTEST00000001"],
                      "transaction_ts": [T - pd.Timedelta(hours=12)],
                      "transaction_date": [(T - pd.Timedelta(days=1)).normalize()],
                      "amount_usd": [3.0], "hour": [8], "is_card_present": [True],
                      "is_international": [False], "category": ["Grocery"],
                      # values inherited from the later fraud row (must be overwritten)
                      "customer_day_txn_count": [1], "customer_week_spend": [1e6],
                      "days_since_first_txn": [99], "split_time": ["test"],
                      "device_risk_score": [0.99]})
    return s


def test_synthetic_rows_are_recomputed_as_of_their_own_time():
    out = ml_extracts._asof_synthetic_features(_history(), _syn()).iloc[0]
    # earlier real rows h1 (-3d) and h2 (-1d) plus the synthetic row itself
    assert out.customer_week_spend == pytest.approx(40.0 + 60.0 + 3.0)   # not 1e6, not incl. 900
    assert out.customer_week_spend_log == pytest.approx(np.log1p(103.0))
    assert out.days_since_first_txn == 2                                  # synthetic date - first txn date
    assert out.customer_day_txn_count == 2                                # h2 shares the calendar day
    assert out.split_time == "train" and C.ML_FRAUD_OOT_START > T.date()  # own date, not the fraud row's
    assert out.device_risk_score != 0.99 and 0.0 <= out.device_risk_score <= 1.0


def test_synthetic_features_do_not_change_when_later_transactions_are_appended():
    base = ml_extracts._asof_synthetic_features(_history(), _syn())
    later = pd.concat([_history(), _tx([("A", "x1", T + pd.Timedelta(hours=1), 5000.0),
                                        ("A", "x2", T + pd.Timedelta(days=30), 7000.0)])],
                      ignore_index=True)
    got = ml_extracts._asof_synthetic_features(later, _syn())
    cols = ["customer_day_txn_count", "customer_week_spend", "customer_week_spend_log",
            "days_since_first_txn", "split_time", "device_risk_score"]
    pd.testing.assert_frame_equal(got[cols], base[cols])


def test_synthetic_velocity_leaves_real_rows_untouched():
    hist = _history()
    before = ml_extracts._attach_velocity(hist)
    ml_extracts._asof_synthetic_features(hist, _syn())
    pd.testing.assert_frame_equal(ml_extracts._attach_velocity(hist), before)
