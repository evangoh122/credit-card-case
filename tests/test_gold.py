from datetime import UTC, date, datetime

import polars as pl
import pytest

from ccc.gold import (
    GoldStore,
    RFMBins,
    count_late_events,
    engagement_features,
    guard_no_truth_columns,
    make_rfm_bins,
    rfm_snapshot,
    score_rfm,
    segment_assignment_snapshot,
)
from ccc.segmentation import fit_tiered
from ccc.sim.run import simulate
from ccc.sim.scenario import Scenario
from ccc.timeutil import et_day_hours

WINDOWS = {"recency": 30, "frequency": 30, "monetary": 30}


def _customers() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "customer_id": ["a", "b"],
            "join_date": [date(2026, 10, 1), date(2026, 10, 25)],
        }
    )


def _transactions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "customer_id": ["a", "a", "a", "a"],
            "txn_type": ["purchase", "purchase", "purchase", "repayment"],
            "amount": [10.0, 20.0, 999.0, 500.0],
            "event_ts_utc": [
                datetime(2026, 10, 20, 15, tzinfo=UTC),
                datetime(2026, 11, 2, 4, 30, tzinfo=UTC),
                datetime(2026, 11, 2, 5, 0, tzinfo=UTC),
                datetime(2026, 10, 30, 12, tzinfo=UTC),
            ],
            "event_date_et": [
                date(2026, 10, 20),
                date(2026, 11, 1),
                date(2026, 11, 2),
                date(2026, 10, 30),
            ],
        }
    )


def test_rfm_is_point_in_time_and_left_joined():
    result = rfm_snapshot(_customers(), _transactions(), date(2026, 11, 1), WINDOWS)
    a = result.filter(pl.col("customer_id") == "a").row(0, named=True)
    b = result.filter(pl.col("customer_id") == "b").row(0, named=True)
    assert (a["recency"], a["frequency"], a["monetary"]) == (0, 2, 30.0)
    assert a["days_since_join"] == 31
    assert (b["recency"], b["frequency"], b["monetary"]) == (None, 0, 0.0)
    assert b["days_since_join"] == 7


def test_rfm_windows_use_calendar_dates_across_dst_fallback():
    assert et_day_hours(date(2026, 11, 1)) == 25
    windows = {"recency": 2, "frequency": 2, "monetary": 2}
    result = rfm_snapshot(_customers(), _transactions(), date(2026, 11, 1), windows)
    assert result.filter(pl.col("customer_id") == "a")["frequency"].item() == 1


def test_saved_bins_are_reused_without_refitting(tmp_path):
    reference = pl.DataFrame(
        {
            "recency": [1, 2, 3, 4, 5],
            "frequency": [1, 2, 3, 4, 5],
            "monetary": [10, 20, 30, 40, 50],
        }
    )
    bins = make_rfm_bins(reference)
    path = tmp_path / "rfm-bins.json"
    bins.save(path)
    before = path.read_text()
    scored = score_rfm(
        pl.DataFrame({"recency": [100], "frequency": [100], "monetary": [1000]}),
        RFMBins.load(path),
    )
    assert scored.select("recency_score", "frequency_score", "monetary_score").row(0) == (
        1,
        5,
        5,
    )
    assert path.read_text() == before


def test_engagement_is_separate_from_rfm():
    events = pl.DataFrame(
        {
            "customer_id": ["a", "a", "a", "a"],
            "event_type": ["app_open", "repayment", "missed_payment", "app_open"],
            "event_date_et": [
                date(2026, 10, 1),
                date(2026, 10, 2),
                date(2026, 10, 2),
                date(2026, 10, 4),
            ],
        }
    )
    result = engagement_features(events, date(2026, 10, 3))
    assert result.select("app_open_count", "repayment_count").row(0) == (1, 1)
    assert not set(("recency", "frequency", "monetary")).intersection(result.columns)


def _segment_features() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(8)],
            "arrears": [0.0, 0.0, 0.0, 0.0, 1.0, 1.1, 0.0, 0.0],
            "utilisation": [0.1, 0.12, 0.4, 0.42, 0.9, 0.92, 0.2, 0.22],
            "spend": [100.0, 110.0, 50.0, 55.0, 10.0, 12.0, 1.0, 2.0],
            "activity": [20.0, 21.0, 10.0, 11.0, 2.0, 2.5, 0.0, 0.5],
            "over_limit": [False, False, False, False, True, True, False, False],
            "days_active": [60, 61, 60, 61, 60, 61, 5, 6],
        }
    )


def test_segment_snapshot_uses_model_version_and_hard_rules():
    features = _segment_features()
    model, _ = fit_tiered(features)
    result = segment_assignment_snapshot(features, model, date(2026, 11, 1))
    assert result.columns == ["customer_id", "snapshot_date", "segment", "model_version"]
    assert result["model_version"].unique().to_list() == [model.version]
    assert result.filter(pl.col("customer_id") == "c4")["segment"].item() == "Decline"
    assert result.filter(pl.col("customer_id") == "c6")["segment"].item() == "Nurture"


def test_store_is_idempotent_frozen_and_adds_loaded_at(tmp_path):
    store = GoldStore(tmp_path)
    frame = pl.DataFrame(
        {"customer_id": ["a"], "snapshot_date": [date(2026, 11, 1)], "frequency": [2]}
    )
    loaded_at = datetime(2026, 11, 2, tzinfo=UTC)
    first = store.write_snapshot(
        "fact_rfm_snapshot", frame, ("customer_id", "snapshot_date"), loaded_at=loaded_at
    )
    second = store.write_snapshot(
        "fact_rfm_snapshot",
        frame,
        ("customer_id", "snapshot_date"),
        loaded_at=datetime(2026, 11, 3, tzinfo=UTC),
    )
    assert first.equals(second)
    assert second["loaded_at"].item() == loaded_at
    changed = frame.with_columns(pl.lit(3).alias("frequency"))
    with pytest.raises(ValueError, match="frozen"):
        store.write_snapshot("fact_rfm_snapshot", changed, ("customer_id", "snapshot_date"))


def test_late_events_are_counted_and_logged_for_next_snapshot(tmp_path):
    events = pl.DataFrame(
        {
            "event_date_et": [date(2026, 11, 1), date(2026, 11, 2)],
            "loaded_at": [
                datetime(2026, 11, 2, 6, tzinfo=UTC),
                datetime(2026, 11, 2, 6, tzinfo=UTC),
            ],
        },
        schema_overrides={"loaded_at": pl.Datetime("us", "UTC")},
    )
    count = count_late_events(events, date(2026, 11, 1))
    assert count == 1
    store = GoldStore(tmp_path)
    next_snapshot = pl.DataFrame(
        {"customer_id": ["a"], "snapshot_date": [date(2026, 11, 2)], "frequency": [3]}
    )
    store.write_snapshot(
        "fact_rfm_snapshot",
        next_snapshot,
        ("customer_id", "snapshot_date"),
        late_event_count=count,
    )
    assert store.read("late_events")["late_event_count"].item() == 1


def test_experiment_assignment_writer_reuses_assignment_contract(tmp_path):
    store = GoldStore(tmp_path)
    segments = pl.DataFrame({"customer_id": ["a", "b"], "segment": ["Prime", "Decline"]})
    result = store.write_experiment_assignment(
        segments, "offer_1", 7, assigned_date=date(2026, 11, 1)
    )
    assert result["customer_id"].to_list() == ["a"]
    assert result["experiment_id"].to_list() == ["offer_1"]
    assert "loaded_at" in result.columns


@pytest.mark.parametrize("column", ["risk", "spend_rate", "welcome_true_lift"])
def test_truth_leakage_guard(column):
    with pytest.raises(ValueError, match="simulation truth"):
        guard_no_truth_columns(pl.DataFrame({"customer_id": ["a"], column: [1.0]}))


def test_end_to_end_from_simulator():
    scenario = Scenario(
        name="gold-test",
        start_date=date(2026, 10, 30),
        initial_customers=12,
        arrivals_per_day=0,
        arrivals_trend=0,
    )
    tables = simulate(42, 4, scenario)
    result = rfm_snapshot(
        tables["dim_customer"], tables["fact_transactions"], date(2026, 11, 2), WINDOWS
    )
    assert result.height == tables["dim_customer"].height
    assert result["customer_id"].n_unique() == result.height
    assert result["frequency"].null_count() == 0
