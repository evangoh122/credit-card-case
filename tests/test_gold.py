from datetime import UTC, date, datetime

import polars as pl
import pytest

from ccc.gold import (
    TRUTH_COLUMNS,
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
from ccc.timeutil import et_day_hours, snapshot_cutoff_utc, utc_to_et_date

WINDOWS = {"recency": 30, "frequency": 30, "monetary": 30}


def _customers() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "customer_id": ["a", "b", "future"],
            "join_date": [date(2026, 10, 1), date(2026, 10, 25), date(2026, 11, 2)],
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
    assert "future" not in result["customer_id"].to_list()


def test_rfm_windows_use_calendar_dates_across_dst_fallback():
    assert et_day_hours(date(2026, 11, 1)) == 25
    transactions = pl.DataFrame(
        {
            "customer_id": ["a", "a", "a"],
            "txn_type": ["purchase"] * 3,
            "amount": [10.0, 20.0, 40.0],
            "event_ts_utc": [
                datetime(2026, 10, 30, 16, tzinfo=UTC),
                datetime(2026, 10, 31, 16, tzinfo=UTC),
                datetime(2026, 11, 2, 4, 30, tzinfo=UTC),
            ],
        }
    ).with_columns(
        pl.col("event_ts_utc")
        .map_elements(utc_to_et_date, return_dtype=pl.Date)
        .alias("event_date_et")
    )
    windows = {"recency": 2, "frequency": 2, "monetary": 2}
    result = rfm_snapshot(_customers(), transactions, date(2026, 11, 1), windows)
    row = result.filter(pl.col("customer_id") == "a").row(0, named=True)
    assert (row["frequency"], row["monetary"]) == (2, 60.0)


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
    assert RFMBins.load(path) == bins
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


def test_rfm_scores_have_five_tie_safe_levels():
    reference = pl.DataFrame(
        {
            "recency": [None, 0, 0, 1, 2, 3, 4, 5, 6, 7],
            "frequency": [0, 0, 0, 0, 1, 2, 3, 4, 5, 6],
            "monetary": [0.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    bins = make_rfm_bins(reference)
    assert all(len(edges) == 6 for edges in bins.edges.values())
    frequency_edge = bins.edges["frequency"][2]
    scored = score_rfm(
        pl.DataFrame(
            {
                "recency": [None, 0, 7],
                "frequency": [0.0, frequency_edge, 100.0],
                "monetary": [0.0, 0.0, 100.0],
            }
        ),
        bins,
    )
    assert scored["recency_score"].to_list() == [1, 5, 1]
    assert scored["frequency_score"].to_list() == [1, 2, 5]
    assert scored["monetary_score"].to_list() == [1, 1, 5]


def test_rfm_scores_constant_reference_at_lowest_level():
    reference = pl.DataFrame({"recency": [3] * 10, "frequency": [0] * 10, "monetary": [0.0] * 10})
    bins = make_rfm_bins(reference)
    scored = score_rfm(reference, bins)
    assert scored["recency_score"].unique().to_list() == [5]
    assert scored["frequency_score"].unique().to_list() == [1]
    assert scored["monetary_score"].unique().to_list() == [1]


def test_engagement_is_separate_from_rfm():
    events = pl.DataFrame(
        {
            "customer_id": ["b", "a", "a", "a", "a"],
            "event_type": ["app_open", "app_open", "repayment", "missed_payment", "app_open"],
            "event_date_et": [
                date(2026, 10, 1),
                date(2026, 10, 1),
                date(2026, 10, 2),
                date(2026, 10, 2),
                date(2026, 10, 4),
            ],
        }
    )
    result = engagement_features(events, date(2026, 10, 3))
    assert result["customer_id"].to_list() == ["a", "b"]
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
    with pytest.raises(ValueError, match="unique"):
        segment_assignment_snapshot(
            pl.concat([features, features.head(1)]), model, date(2026, 11, 1)
        )


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
    changed = frame.with_columns(pl.lit(3, dtype=pl.Int64).alias("frequency"))
    with pytest.raises(ValueError, match="frozen"):
        store.write_snapshot("fact_rfm_snapshot", changed, ("customer_id", "snapshot_date"))

    with pytest.raises(ValueError, match="schemas differ"):
        store.write_snapshot(
            "fact_rfm_snapshot",
            frame.with_columns(pl.col("frequency").cast(pl.Float64)),
            ("customer_id", "snapshot_date"),
        )


def test_late_events_are_counted_and_logged_for_next_snapshot(tmp_path):
    previous = date(2026, 11, 1)
    current = date(2026, 11, 2)
    events = pl.DataFrame(
        {
            "event_date_et": [previous] * 5,
            "loaded_at": [
                snapshot_cutoff_utc(previous),
                datetime(2026, 11, 2, 6, tzinfo=UTC),
                snapshot_cutoff_utc(current),
                datetime(2026, 11, 3, 5, 0, tzinfo=UTC),
                datetime(2026, 11, 1, 12, tzinfo=UTC),
            ],
        },
        schema_overrides={"loaded_at": pl.Datetime("us", "UTC")},
    )
    count = count_late_events(events, previous, current)
    assert count == 2
    store = GoldStore(tmp_path)
    next_snapshot = pl.DataFrame(
        {"customer_id": ["a"], "snapshot_date": [current], "frequency": [3]}
    )
    store.write_snapshot(
        "fact_rfm_snapshot",
        next_snapshot,
        ("customer_id", "snapshot_date"),
        late_event_count=count,
    )
    store.write_snapshot(
        "fact_rfm_snapshot",
        next_snapshot,
        ("customer_id", "snapshot_date"),
        late_event_count=count,
    )
    log = store.read("late_events")
    assert log.height == 1
    assert log["late_event_count"].item() == 2


def test_invalid_late_event_metadata_is_rejected_before_table_write(tmp_path):
    store = GoldStore(tmp_path)
    frame = pl.DataFrame(
        {
            "customer_id": ["a", "b"],
            "snapshot_date": [date(2026, 11, 1), date(2026, 11, 2)],
        }
    )
    with pytest.raises(ValueError, match="one snapshot_date"):
        store.write_snapshot(
            "fact_rfm_snapshot", frame, ("customer_id", "snapshot_date"), late_event_count=1
        )
    assert not (tmp_path / "fact_rfm_snapshot.parquet").exists()


def test_store_sorts_rows_by_merge_keys(tmp_path):
    frame = pl.DataFrame(
        {
            "customer_id": ["b", "a"],
            "snapshot_date": [date(2026, 11, 1)] * 2,
        }
    )
    result = GoldStore(tmp_path).write_snapshot(
        "fact_rfm_snapshot", frame, ("snapshot_date", "customer_id")
    )
    assert result["customer_id"].to_list() == ["a", "b"]


def test_experiment_assignment_writer_reuses_assignment_contract(tmp_path):
    store = GoldStore(tmp_path)
    segments = pl.DataFrame({"customer_id": ["a", "b"], "segment": ["Prime", "Decline"]})
    result = store.write_experiment_assignment(
        segments, "offer_1", 7, assigned_date=date(2026, 11, 1)
    )
    assert result["customer_id"].to_list() == ["a"]
    assert result["experiment_id"].to_list() == ["offer_1"]
    assert result["arm"].to_list() == ["control"] or result["arm"].to_list() == ["test"]
    assert "loaded_at" in result.columns
    again = store.write_experiment_assignment(
        segments, "offer_1", 7, assigned_date=date(2026, 11, 1)
    )
    assert again.equals(result)
    changed = segments.with_columns(
        pl.when(pl.col("customer_id") == "a")
        .then(pl.lit("Watch"))
        .otherwise(pl.col("segment"))
        .alias("segment")
    )
    with pytest.raises(ValueError, match="frozen"):
        store.write_experiment_assignment(changed, "offer_1", 7, assigned_date=date(2026, 11, 1))


@pytest.mark.parametrize(
    "column",
    [
        "risk",
        "spend_rate",
        "cohort_day",
        "responsiveness",
        "welcome_true_lift",
        "sim_truth_marker",
    ],
)
def test_truth_leakage_guard(column):
    with pytest.raises(ValueError, match="simulation truth"):
        guard_no_truth_columns(pl.DataFrame({"customer_id": ["a"], column: [1.0]}))


def test_truth_guard_matches_simulator_truth_columns():
    scenario = Scenario(name="truth", start_date=date(2026, 10, 30), initial_customers=2)
    truth_columns = set(simulate(1, 1, scenario)["sim_truth_customer"].columns)
    assert truth_columns == {
        "customer_id",
        "cohort_day",
        "risk",
        "spend_rate",
        "responsiveness",
        "utilisation",
        "loaded_at",
        "data_label",
    }
    assert TRUTH_COLUMNS <= truth_columns


def test_truth_guard_is_called_by_every_feature_and_writer_entrypoint(tmp_path):
    truth = pl.Series("cohort_day", [0])
    with pytest.raises(ValueError, match="simulation truth"):
        make_rfm_bins(
            pl.DataFrame({"recency": [1], "frequency": [1], "monetary": [1]}).with_columns(truth)
        )
    with pytest.raises(ValueError, match="simulation truth"):
        rfm_snapshot(
            _customers().with_columns(truth.extend_constant(0, 2)),
            _transactions(),
            date(2026, 11, 1),
            WINDOWS,
        )
    events = pl.DataFrame(
        {
            "customer_id": ["a"],
            "event_type": ["app_open"],
            "event_date_et": [date(2026, 11, 1)],
            "responsiveness": [1.0],
        }
    )
    with pytest.raises(ValueError, match="simulation truth"):
        engagement_features(events, date(2026, 11, 1))
    features = _segment_features().with_columns(pl.lit(0).alias("cohort_day"))
    model, _ = fit_tiered(_segment_features())
    with pytest.raises(ValueError, match="simulation truth"):
        segment_assignment_snapshot(features, model, date(2026, 11, 1))
    with pytest.raises(ValueError, match="simulation truth"):
        GoldStore(tmp_path).write_snapshot("features", features, ("customer_id",))


def test_end_to_end_from_simulator():
    scenario = Scenario(
        name="gold-test",
        start_date=date(2026, 10, 30),
        initial_customers=12,
        arrivals_per_day=10,
        arrivals_trend=0,
    )
    tables = simulate(42, 4, scenario)
    snapshot_date = date(2026, 11, 1)
    result = rfm_snapshot(
        tables["dim_customer"], tables["fact_transactions"], snapshot_date, WINDOWS
    )
    expected = tables["dim_customer"].filter(pl.col("join_date") <= snapshot_date).height
    assert tables["dim_customer"].filter(pl.col("join_date") > snapshot_date).height > 0
    assert result.height == expected
    assert result["customer_id"].n_unique() == result.height
    assert result["frequency"].null_count() == 0
