"""Round 6: fraud velocity and the forecast extract must be as-of (no look-ahead)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from build import config as C
from build import ml_extracts

T0 = pd.Timestamp("2026-01-05 10:00")


def _tx(rows):
    """rows: (customer, id, ts, amount)"""
    df = pd.DataFrame(rows, columns=["customer_id", "transaction_id", "transaction_ts", "amount_usd"])
    df["transaction_date"] = df.transaction_ts.dt.normalize()
    return df


def _vel(df):
    return ml_extracts._attach_velocity(df).set_index("transaction_id")


BASE = [
    ("A", "1", T0, 10.0),
    ("A", "2", T0, 20.0),                              # same ts as "1": id breaks the tie
    ("A", "3", T0 + pd.Timedelta(days=3), 30.0),
    ("B", "4", T0, 5.0),
    ("A", "5", T0 + pd.Timedelta(days=8), 40.0),
]


def test_velocity_window_definition():
    v = _vel(_tx(BASE))
    assert v.customer_day_txn_count.to_dict() == {"1": 1, "2": 2, "3": 1, "4": 1, "5": 1}
    # trailing 7 days (ts-7d, ts]: "5" at +8d sees +3d and itself only; "1"/"2" do not see each other later
    assert v.customer_week_spend.to_dict() == {"1": 10.0, "2": 30.0, "3": 60.0, "4": 5.0, "5": 70.0}
    assert v.days_since_first_txn.to_dict() == {"1": 0, "2": 0, "3": 3, "4": 0, "5": 8}
    out = ml_extracts._attach_velocity(_tx(BASE))
    assert out.customer_day_txn_count.dtype == "int64"
    assert out.customer_week_spend.dtype == "float64"
    assert out.days_since_first_txn.dtype == "int64"
    assert list(out.transaction_id) == ["1", "2", "3", "4", "5"]  # row order preserved


@pytest.mark.parametrize("later", [
    [("A", "6", T0 + pd.Timedelta(hours=2), 1000.0)],          # later, same day
    [("A", "7", T0 + pd.Timedelta(days=2), 1000.0)],           # later, same week
    [("A", "8", T0, 1000.0)],                                  # same ts, larger id than "2"
    [("A", "9", T0 + pd.Timedelta(days=30), 1000.0), ("C", "10", T0, 1.0)],
])
def test_appending_later_transactions_never_changes_earlier_rows(later):
    base = _tx(BASE)
    before = _vel(base)
    after = _vel(_tx(BASE + later))
    cols = ["customer_day_txn_count", "customer_week_spend", "days_since_first_txn"]
    # a row is only comparable if every appended row of its customer sorts after it by (ts, id)
    new = _tx(later)
    safe = []
    for r in base.itertuples():
        mine = new[new.customer_id == r.customer_id]
        if all((n.transaction_ts, n.transaction_id) > (r.transaction_ts, r.transaction_id)
               for n in mine.itertuples()):
            safe.append(r.transaction_id)
    assert len(safe) >= 3   # the case really exercises "later rows appended"
    pd.testing.assert_frame_equal(after.loc[safe, cols], before.loc[safe, cols])


def test_velocity_on_random_data_matches_bruteforce_asof():
    r = np.random.default_rng(3)
    n = 300
    ts = T0 + pd.to_timedelta(r.integers(0, 20 * 24 * 60, n), unit="min")
    df = _tx([(f"C{r.integers(0, 6)}", f"T{i:04d}", ts[i], float(r.integers(1, 100)))
              for i in range(n)])
    got = _vel(df)
    for tid, row in df.set_index("transaction_id").iterrows():
        mine = df[(df.customer_id == row.customer_id)
                  & ((df.transaction_ts < row.transaction_ts)
                     | ((df.transaction_ts == row.transaction_ts) & (df.transaction_id <= tid)))]
        same_day = mine[mine.transaction_date == row.transaction_date]
        week = mine[mine.transaction_ts > row.transaction_ts - pd.Timedelta(days=7)]
        assert got.loc[tid, "customer_day_txn_count"] == len(same_day)
        assert got.loc[tid, "customer_week_spend"] == pytest.approx(week.amount_usd.sum())


# ---------------------------------------------------------------- forecast
@pytest.fixture
def forecast(monkeypatch, tmp_path):
    out = {}
    monkeypatch.setattr(ml_extracts, "write_parquet",
                        lambda df, path, name=None: out.__setitem__(name, df.copy()))
    monkeypatch.setattr(ml_extracts, "ML", tmp_path)
    return out


def _sales(extra=()):
    """Merchants M1..M3; train Sep-2025..Jun-2026, test Jul..Sep-2026."""
    rows = []
    months = ["2025-09", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06",
              "2026-07", "2026-08", "2026-09"]
    spend = {"M1": 100.0, "M2": 50.0, "M3": 1.0}
    for i, m in enumerate(months):
        for mid, base in spend.items():
            rows.append((mid, f"{mid}-{m}", pd.Timestamp(f"{m}-10"), base * (1 + i % 3)))
    rows += list(extra)
    return pd.DataFrame(rows, columns=["merchant_id", "transaction_id", "transaction_date", "amount_usd"])


def _run(forecast, df, top=2):
    old = C.ML_FORECAST_MERCHANTS
    try:
        C.ML_FORECAST_MERCHANTS = top
        ml_extracts.build_forecast(df)
    finally:
        C.ML_FORECAST_MERCHANTS = old
    return forecast["forecast_merchant_month"]


def test_forecast_split_and_seasonality_use_train_months_only(forecast):
    f = _run(forecast, _sales())
    assert set(f.merchant_id) == {"M1", "M2"}
    assert set(f[f.month < "2026-07"].split_time) == {"train"}
    assert set(f[f.month >= "2026-07"].split_time) == {"test"}
    # July/August never occur in train -> no seasonality evidence
    assert f[f.month.isin(["2026-07", "2026-08"])].seasonal_index.isna().all()
    # September 2025 is in train, so September 2026 has an index equal to the train ratio
    train = _sales()
    train = train[train.transaction_date < "2026-07-01"]
    totals = train.groupby(train.transaction_date.dt.strftime("%Y-%m")).amount_usd.sum()
    assert f[f.month == "2026-09"].seasonal_index.iloc[0] == pytest.approx(
        round(totals["2025-09"] / totals.mean(), 6))
    assert f.month_of_year.dtype == "int32" and f.quarter.dtype == "int32"
    assert f.txn_count.dtype == "int64" and f.is_q4.dtype == "int8"


def test_forecast_unaffected_by_test_window_spend(forecast):
    base = _run(forecast, _sales()).set_index(["merchant_id", "month"])
    # M3 explodes in the TEST window and everyone's test spend changes
    spike = [("M3", "big", pd.Timestamp("2026-08-10"), 1e7),
             ("M1", "big2", pd.Timestamp("2026-09-10"), 5e6)]
    got = _run(forecast, _sales(spike)).set_index(["merchant_id", "month"])
    assert set(got.index.get_level_values(0)) == {"M1", "M2"}      # M3 still not selected
    train_rows = [i for i in base.index if i[1] < "2026-07"]
    pd.testing.assert_frame_equal(got.loc[train_rows], base.loc[train_rows])  # incl. seasonal_index
    assert (got.loc[base.index, "seasonal_index"].fillna(-1) == base.seasonal_index.fillna(-1)).all()
