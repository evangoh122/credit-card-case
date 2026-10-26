from datetime import UTC, date, datetime, timedelta

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
from ccc.timeutil import local_date_expr, local_day_hours, snapshot_cutoff_utc

WINDOWS = {"recency": 30, "frequency": 30, "monetary": 30}


def _customers() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "customer_id": ["a", "b", "future"],
            "city": ["New York", "Los Angeles", "New York"],
            "tz": ["America/New_York", "America/Los_Angeles", "America/New_York"],
            "joined_ts_utc": [
                datetime(2026, 10, 1, 12, tzinfo=UTC),
                datetime(2026, 10, 25, 12, tzinfo=UTC),
                datetime(2026, 11, 2, 12, tzinfo=UTC),
            ],
        },
        schema_overrides={"joined_ts_utc": pl.Datetime("us", "UTC")},
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
        }
    )


def test_rfm_is_point_in_time_and_left_joined():
    result = rfm_snapshot(_customers(), _transactions(), date(2026, 11, 1), WINDOWS)
    assert result["customer_id"].to_list() == ["a", "b"]
    a = result.filter(pl.col("customer_id") == "a").row(0, named=True)
    b = result.filter(pl.col("customer_id") == "b").row(0, named=True)
    assert (a["recency"], a["frequency"], a["monetary"]) == (0, 2, 30.0)
    assert a["days_since_join"] == 31
    assert (b["recency"], b["frequency"], b["monetary"]) == (None, 0, 0.0)
    assert b["days_since_join"] == 7
    assert "future" not in result["customer_id"].to_list()


def test_rfm_windows_use_calendar_dates_across_dst_fallback():
    assert local_day_hours(date(2026, 11, 1), "America/New_York") == 25
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
    )
    windows = {"recency": 2, "frequency": 2, "monetary": 2}
    result = rfm_snapshot(_customers(), transactions, date(2026, 11, 1), windows)
    row = result.filter(pl.col("customer_id") == "a").row(0, named=True)
    assert (row["recency"], row["frequency"], row["monetary"]) == (0, 2, 60.0)


def test_rfm_uses_each_customers_local_cutoff():
    dim = pl.DataFrame(
        {
            "customer_id": ["east", "west"],
            "tz": ["America/New_York", "America/Los_Angeles"],
            "joined_ts_utc": [datetime(2026, 10, 1, 12, tzinfo=UTC)] * 2,
        },
        schema_overrides={"joined_ts_utc": pl.Datetime("us", "UTC")},
    )
    transactions = pl.DataFrame(
        {
            "customer_id": ["east", "west"],
            "txn_type": ["purchase", "purchase"],
            "amount": [10.0, 20.0],
            "event_ts_utc": [datetime(2026, 11, 2, 5, 30, tzinfo=UTC)] * 2,
        }
    )
    result = rfm_snapshot(dim, transactions, date(2026, 11, 1), WINDOWS)
    east = result.filter(pl.col("customer_id") == "east").row(0, named=True)
    west = result.filter(pl.col("customer_id") == "west").row(0, named=True)
    assert (east["frequency"], east["recency"]) == (0, None)
    assert (west["frequency"], west["recency"]) == (1, 0)


def test_late_event_is_judged_against_each_customers_cutoff():
    dim = pl.DataFrame(
        {"customer_id": ["east", "west"], "tz": ["America/New_York", "America/Los_Angeles"]}
    )
    events = pl.DataFrame(
        {
            "customer_id": ["east", "west"],
            "event_ts_utc": [datetime(2026, 11, 1, 15, tzinfo=UTC)] * 2,
            "loaded_at": [datetime(2026, 11, 2, 5, 30, tzinfo=UTC)] * 2,
        },
        schema_overrides={
            "event_ts_utc": pl.Datetime("us", "UTC"),
            "loaded_at": pl.Datetime("us", "UTC"),
        },
    )
    assert count_late_events(events, dim, date(2026, 11, 1), date(2026, 11, 2)) == 1


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


def test_rfm_bins_reject_nan_edges():
    valid = (-float("inf"), 1.0, 2.0, 3.0, 4.0, float("inf"))
    invalid = (-float("inf"), 1.0, float("nan"), 3.0, 4.0, float("inf"))
    with pytest.raises(ValueError, match="finite"):
        RFMBins({"recency": invalid, "frequency": valid, "monetary": valid})


def test_engagement_is_separate_from_rfm():
    events = pl.DataFrame(
        {
            "customer_id": ["b", "a", "a", "a", "a"],
            "event_type": ["app_open", "app_open", "repayment", "missed_payment", "app_open"],
            "event_ts_utc": [
                datetime(2026, 10, 1, 15, tzinfo=UTC),
                datetime(2026, 10, 1, 15, tzinfo=UTC),
                datetime(2026, 10, 2, 15, tzinfo=UTC),
                datetime(2026, 10, 2, 15, tzinfo=UTC),
                datetime(2026, 10, 4, 15, tzinfo=UTC),
            ],
        }
    )
    dim = pl.DataFrame({"customer_id": ["a", "b"], "tz": ["America/New_York"] * 2})
    result = engagement_features(events, dim, date(2026, 10, 3))
    assert result["customer_id"].to_list() == ["a", "b"]
    assert result.select("app_open_count", "repayment_count").row(0) == (1, 1)
    assert not set(("recency", "frequency", "monetary")).intersection(result.columns)


def test_engagement_rejects_facts_without_dim_rows():
    events = pl.DataFrame(
        {
            "customer_id": ["a", "ghost"],
            "event_type": ["app_open", "app_open"],
            "event_ts_utc": [datetime(2026, 10, 1, 15, tzinfo=UTC)] * 2,
        }
    )
    dim = pl.DataFrame({"customer_id": ["a"], "tz": ["America/New_York"]})
    with pytest.raises(ValueError, match="missing from dim_customer"):
        engagement_features(events, dim, date(2026, 10, 3))


def test_engagement_rejects_duplicate_dim_rows():
    events = pl.DataFrame(
        {
            "customer_id": ["a"],
            "event_type": ["app_open"],
            "event_ts_utc": [datetime(2026, 10, 1, 15, tzinfo=UTC)],
        }
    )
    dim = pl.DataFrame({"customer_id": ["a", "a"], "tz": ["America/New_York"] * 2})
    with pytest.raises(ValueError, match="customer_id must be unique"):
        engagement_features(events, dim, date(2026, 10, 3))


def test_engagement_rejects_null_time_zone():
    events = pl.DataFrame(
        {
            "customer_id": ["a"],
            "event_type": ["app_open"],
            "event_ts_utc": [datetime(2026, 10, 1, 15, tzinfo=UTC)],
        }
    )
    dim = pl.DataFrame(
        {"customer_id": ["a"], "tz": [None]},
        schema_overrides={"tz": pl.String},
    )
    with pytest.raises(ValueError, match="must not contain nulls"):
        engagement_features(events, dim, date(2026, 10, 3))


def test_rfm_validates_unknown_customers_before_purchase_filtering():
    transactions = pl.concat(
        [
            _transactions().head(1),
            pl.DataFrame(
                {
                    "customer_id": ["ghost"],
                    "txn_type": ["repayment"],
                    "amount": [5.0],
                    "event_ts_utc": [datetime(2026, 10, 20, 15, tzinfo=UTC)],
                }
            ),
        ]
    )
    with pytest.raises(ValueError, match="fact_transactions customer_ids missing"):
        rfm_snapshot(_customers(), transactions, date(2026, 11, 1), WINDOWS)


def test_gold_entry_points_reject_null_customer_ids():
    transactions = pl.DataFrame(
        {
            "customer_id": [None],
            "txn_type": ["purchase"],
            "amount": [5.0],
            "event_ts_utc": [datetime(2026, 10, 20, 15, tzinfo=UTC)],
        },
        schema_overrides={"customer_id": pl.String},
    )
    with pytest.raises(ValueError, match="fact_transactions customer_id must not contain nulls"):
        rfm_snapshot(_customers(), transactions, date(2026, 11, 1), WINDOWS)

    events = pl.DataFrame(
        {
            "customer_id": [None],
            "event_type": ["app_open"],
            "event_ts_utc": [datetime(2026, 10, 20, 15, tzinfo=UTC)],
        },
        schema_overrides={"customer_id": pl.String},
    )
    with pytest.raises(ValueError, match="fact_events customer_id must not contain nulls"):
        engagement_features(events, _customers(), date(2026, 10, 20))

    late = events.with_columns(
        pl.lit(datetime(2026, 11, 2, tzinfo=UTC)).alias("loaded_at")
    ).with_columns(pl.col("loaded_at").cast(pl.Datetime("us", "UTC")))
    with pytest.raises(ValueError, match="events customer_id must not contain nulls"):
        count_late_events(late, _customers(), date(2026, 11, 1), date(2026, 11, 2))

    null_dim = _customers().with_columns(pl.lit(None, dtype=pl.String).alias("customer_id"))
    with pytest.raises(ValueError, match="dim_customer customer_id must not contain nulls"):
        engagement_features(
            events.with_columns(pl.lit("a").alias("customer_id")),
            null_dim,
            date(2026, 10, 20),
        )


def test_gold_features_accept_empty_fact_frames():
    dim = _customers()
    transactions = _transactions().clear()
    result = rfm_snapshot(dim, transactions, date(2026, 11, 1), WINDOWS)
    assert result.height == 2
    assert (result["frequency"] == 0).all()
    assert (result["monetary"] == 0.0).all()

    events = pl.DataFrame(
        schema={
            "customer_id": pl.String,
            "event_type": pl.String,
            "event_ts_utc": pl.Datetime("us", "UTC"),
            "loaded_at": pl.Datetime("us", "UTC"),
        }
    )
    engagement = engagement_features(events, dim, date(2026, 11, 1))
    assert engagement.height == 0
    assert count_late_events(events, dim, date(2026, 11, 1), date(2026, 11, 2)) == 0


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
    previous_cutoff = snapshot_cutoff_utc(previous, "America/New_York")
    current_cutoff = snapshot_cutoff_utc(current, "America/New_York")
    assert previous_cutoff == datetime(2026, 11, 2, 4, 59, 59, tzinfo=UTC)
    assert current_cutoff == datetime(2026, 11, 3, 4, 59, 59, tzinfo=UTC)
    events = pl.DataFrame(
        {
            "customer_id": ["a"] * 6,
            "event_ts_utc": [
                datetime(2026, 11, 1, 15, tzinfo=UTC),
                datetime(2026, 11, 1, 15, tzinfo=UTC),
                datetime(2026, 11, 1, 15, tzinfo=UTC),
                datetime(2026, 11, 1, 15, tzinfo=UTC),
                datetime(2026, 11, 1, 15, tzinfo=UTC),
                datetime(2026, 11, 2, 15, tzinfo=UTC),
            ],
            "loaded_at": [
                previous_cutoff,
                previous_cutoff + timedelta(hours=1),
                current_cutoff,
                current_cutoff + timedelta(seconds=1),
                previous_cutoff - timedelta(hours=1),
                previous_cutoff + timedelta(hours=2),
            ],
        },
        schema_overrides={
            "event_ts_utc": pl.Datetime("us", "UTC"),
            "loaded_at": pl.Datetime("us", "UTC"),
        },
    )
    assert events["loaded_at"].item(0) == previous_cutoff
    assert events["loaded_at"].item(2) == current_cutoff
    count = count_late_events(events, _customers(), previous, current)
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
    assert result["arm"].to_list() == ["control"]
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
    bins = make_rfm_bins(
        pl.DataFrame(
            {
                "recency": [1, 2, 3, 4, 5],
                "frequency": [1, 2, 3, 4, 5],
                "monetary": [1, 2, 3, 4, 5],
            }
        )
    )
    with pytest.raises(ValueError, match="simulation truth"):
        score_rfm(
            pl.DataFrame({"recency": [1], "frequency": [1], "monetary": [1]}).with_columns(truth),
            bins,
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
            "event_ts_utc": [datetime(2026, 11, 1, 15, tzinfo=UTC)],
            "responsiveness": [1.0],
        }
    )
    with pytest.raises(ValueError, match="simulation truth"):
        engagement_features(events, _customers(), date(2026, 11, 1))
    features = _segment_features().with_columns(pl.lit(0).alias("cohort_day"))
    model, _ = fit_tiered(_segment_features())
    with pytest.raises(ValueError, match="simulation truth"):
        segment_assignment_snapshot(features, model, date(2026, 11, 1))
    with pytest.raises(ValueError, match="simulation truth"):
        GoldStore(tmp_path).write_snapshot("features", features, ("customer_id",))
    assignment_features = pl.DataFrame(
        {"customer_id": ["a"], "segment": ["Prime"], "cohort_day": [0]}
    )
    with pytest.raises(ValueError, match="simulation truth"):
        GoldStore(tmp_path).write_experiment_assignment(
            assignment_features,
            "offer",
            1,
            assigned_date=date(2026, 11, 1),
        )


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
    local = tables["dim_customer"].with_columns(
        local_date_expr("joined_ts_utc", "tz").alias("join_local_date")
    )
    expected = local.filter(pl.col("join_local_date") <= snapshot_date).height
    assert local.filter(pl.col("join_local_date") > snapshot_date).height > 0
    assert result.height == expected
    assert result["customer_id"].n_unique() == result.height
    assert result["frequency"].null_count() == 0
