from datetime import UTC, date, datetime

import numpy as np
import polars as pl
import pytest

from ccc.experiment import (
    assign,
    balance_check,
    mde_for_n,
    reseed,
    sample_size_per_arm,
    size_arms_per_segment,
    tenure_outcomes,
    uplift,
)


def customers(n: int = 4_000) -> pl.DataFrame:
    ids = np.arange(n)
    return pl.DataFrame(
        {
            "customer_id": [f"c{i}" for i in ids],
            "segment": np.resize(["Prime", "Watch", "Nurture", "Decline"], n),
            "pre_spend": ids % 31,
            "arrears": ids % 2,
            "utilisation": (ids % 100) / 100,
        }
    )


def test_assignment_is_deterministic_excludes_decline_and_balances_arms() -> None:
    source = customers()
    first = assign(source, "offer-1", 42, assigned_date=date(2026, 1, 1))
    second = assign(source, "offer-1", 42, assigned_date=date(2026, 1, 1))

    assert first.equals(second)
    assert "Decline" not in first.get_column("stratum").to_list()
    proportions = (
        first.group_by("stratum", "arm")
        .len()
        .with_columns(pl.col("len") / pl.col("len").sum().over("stratum"))
    )
    assert proportions.get_column("len").is_between(0.45, 0.55).all()


def test_assignment_supports_multiple_arms_and_is_stable_when_customers_are_added() -> None:
    source = customers(2_000)
    arms = {"control": 0.4, "email": 0.3, "push": 0.3}
    original = assign(source, "offer-2", 9, arms, assigned_date=date(2026, 1, 1))
    extended = assign(customers(2_400), "offer-2", 9, arms, assigned_date=date(2026, 1, 1))

    assert original.equals(
        extended.join(source.select("customer_id"), on="customer_id", how="semi")
    )
    assert set(original["arm"]) == set(arms)


def test_balance_check_detects_deliberately_skewed_split() -> None:
    source = (
        customers(400)
        .filter(pl.col("segment") == "Prime")
        .with_columns(pl.arange(0, pl.len()).cast(pl.Float64).alias("pre_spend"))
    )
    assignment = pl.DataFrame(
        {
            "customer_id": source["customer_id"],
            "experiment_id": ["bad"] * source.height,
            "arm": ["control"] * 50 + ["test"] * 50,
            "assigned_date": [date(2026, 1, 1)] * source.height,
            "stratum": ["Prime"] * source.height,
        }
    )
    result = balance_check(source, assignment, cols=["pre_spend"])

    assert result.row(0, named=True)["smd"] > 1
    assert not result.row(0, named=True)["balanced"]


def test_balance_check_accepts_a_genuinely_balanced_assignment() -> None:
    source = pl.DataFrame(
        {
            "customer_id": ["c1", "c2", "t1", "t2"],
            "pre_spend": [10.0, 20.0, 10.0, 20.0],
        }
    )
    assignment = pl.DataFrame(
        {
            "customer_id": source["customer_id"],
            "experiment_id": ["balanced"] * 4,
            "arm": ["control", "control", "test", "test"],
            "assigned_date": [date(2026, 1, 1)] * 4,
            "stratum": ["Prime"] * 4,
        }
    )

    result = balance_check(source, assignment, cols=["pre_spend"])

    assert result.row(0, named=True)["smd"] == 0
    assert result.row(0, named=True)["balanced"]


def test_reseed_uses_zero_for_balance_and_advances_only_for_imbalance() -> None:
    source = customers(800).filter(pl.col("segment") == "Prime")
    source = source.with_columns(pl.lit(1.0).alias("pre_spend"))
    _, balanced, salt = reseed(
        source,
        "balanced",
        12,
        cols=["pre_spend"],
        assigned_date=date(2026, 1, 1),
    )
    assert salt == 0
    assert balanced.get_column("balanced").all()

    initial = assign(source, "advance", 12, assigned_date=date(2026, 1, 1))
    skewed = source.with_columns(
        pl.col("customer_id")
        .is_in(initial.filter(pl.col("arm") == "test")["customer_id"].implode())
        .cast(pl.Float64)
        .alias("pre_spend")
    )
    _, recovered, salt = reseed(
        skewed,
        "advance",
        12,
        cols=["pre_spend"],
        threshold=0.3,
        assigned_date=date(2026, 1, 1),
    )
    assert salt > 0
    assert recovered.get_column("balanced").all()

    with pytest.raises(RuntimeError, match="no balanced assignment"):
        reseed(
            skewed,
            "advance",
            12,
            cols=["pre_spend"],
            threshold=0.3,
            max_salts=1,
            assigned_date=date(2026, 1, 1),
        )


def test_power_matches_textbook_normal_approximation_and_inverts() -> None:
    required = sample_size_per_arm(0.10, 0.05, alpha=0.05, power=0.80)

    assert required == 686
    assert abs(mde_for_n(0.10, required) - 0.05) < 0.0001


def test_size_arms_per_segment_floors_capacity_and_compares_power() -> None:
    result = size_arms_per_segment(
        {"Prime": 0.333, "Watch": 0.667},
        {"Prime": 0.10, "Watch": 0.10},
        total_customers=3_001,
        mde=0.05,
    ).sort("segment")

    prime, watch = result.iter_rows(named=True)
    assert prime["available_per_arm"] == 499
    assert watch["available_per_arm"] == 1_000
    assert not prime["powered"]
    assert watch["powered"]


@pytest.mark.parametrize(
    ("shares", "rates", "message"),
    [
        ({"Prime": 0.8}, {"Prime": 0.1}, "sum to 1"),
        ({"Prime": 1.1, "Watch": -0.1}, {"Prime": 0.1, "Watch": 0.1}, "between 0 and 1"),
        ({"Prime": 0.8, "Decline": 0.2}, {"Prime": 0.1, "Decline": 0.1}, "Decline"),
    ],
)
def test_size_arms_per_segment_validates_eligible_shares(shares, rates, message) -> None:
    with pytest.raises(ValueError, match=message):
        size_arms_per_segment(shares, rates, mde=0.05)


def test_uplift_confidence_interval_covers_planted_effect_over_seeds() -> None:
    assignment = assign(customers(8_000), "offer-3", 11, assigned_date=date(2026, 1, 1))
    covered = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        probabilities = np.where(assignment["arm"].to_numpy() == "test", 0.25, 0.20)
        outcomes = pl.DataFrame(
            {"customer_id": assignment["customer_id"], "outcome": rng.binomial(1, probabilities)}
        )
        overall = (
            uplift(outcomes, assignment).filter(pl.col("stratum") == "overall").row(0, named=True)
        )
        covered += overall["ci_low"] <= 0.05 <= overall["ci_high"]

    assert covered >= 18


def test_uplift_filters_immature_and_reports_assignment_attrition() -> None:
    assignment = pl.DataFrame(
        {
            "customer_id": ["c1", "c2", "t1", "t2"],
            "experiment_id": ["offer"] * 4,
            "arm": ["control", "control", "test", "test"],
            "assigned_date": [date(2026, 1, 1)] * 4,
            "stratum": ["Prime"] * 4,
        }
    )
    outcomes = pl.DataFrame(
        {
            "customer_id": ["c1", "c2", "t1"],
            "outcome": [False, True, True],
            "immature": [False, True, False],
        }
    )

    result = uplift(outcomes, assignment).filter(pl.col("stratum") == "Prime").row(0, named=True)

    assert result["n_control_assigned"] == 2
    assert result["n_arm_assigned"] == 2
    assert result["n_control_analysed"] == 1
    assert result["n_arm_analysed"] == 1
    assert result["control_rate"] == 0
    assert result["arm_rate"] == 1


def test_overall_uplift_is_weighted_by_stratum_size_not_pooled_arm_mix() -> None:
    rows = []
    outcomes = []
    for stratum, arm, count, successes in [
        ("A", "control", 40, 4),
        ("A", "test", 10, 2),
        ("B", "control", 10, 8),
        ("B", "test", 40, 36),
    ]:
        for index in range(count):
            customer_id = f"{stratum}-{arm}-{index}"
            rows.append((customer_id, "offer", arm, date(2026, 1, 1), stratum))
            outcomes.append((customer_id, index < successes))
    assignment = pl.DataFrame(
        rows,
        schema=["customer_id", "experiment_id", "arm", "assigned_date", "stratum"],
        orient="row",
    )
    outcome_frame = pl.DataFrame(outcomes, schema=["customer_id", "outcome"], orient="row")

    overall = (
        uplift(outcome_frame, assignment).filter(pl.col("stratum") == "overall").row(0, named=True)
    )

    assert overall["uplift"] == pytest.approx(0.1)
    assert overall["uplift"] != pytest.approx(0.52)


def _tenure_frames():
    assignment = pl.DataFrame(
        {
            "customer_id": ["a"],
            "experiment_id": ["offer"],
            "arm": ["test"],
            "assigned_date": [date(2026, 1, 1)],
            "stratum": ["Prime"],
        }
    )
    dim = pl.DataFrame({"customer_id": ["a"], "tz": ["America/New_York"]})
    events = pl.DataFrame(
        {"customer_id": ["a"], "event_ts_utc": [datetime(2026, 1, 2, 15, tzinfo=UTC)]},
        schema_overrides={"event_ts_utc": pl.Datetime("us", "UTC")},
    )
    return events, assignment, dim


def test_tenure_rejects_assignment_without_dim_rows():
    events, assignment, dim = _tenure_frames()
    with pytest.raises(ValueError, match="assignment customer_ids missing from dim_customer"):
        tenure_outcomes(
            events,
            assignment,
            dim.filter(pl.col("customer_id") == "other"),
            5,
            as_of_date=date(2026, 1, 7),
        )


def test_tenure_rejects_events_without_dim_rows():
    events, assignment, dim = _tenure_frames()
    hidden = events.with_columns(pl.lit("ghost").alias("customer_id"))
    with pytest.raises(ValueError, match="events customer_ids missing from dim_customer"):
        tenure_outcomes(hidden, assignment, dim, 5, as_of_date=date(2026, 1, 7))


def test_tenure_rejects_duplicate_dim_rows():
    events, assignment, dim = _tenure_frames()
    with pytest.raises(ValueError, match="customer_id must be unique"):
        tenure_outcomes(events, assignment, pl.concat([dim, dim]), 5, as_of_date=date(2026, 1, 7))


def test_tenure_rejects_null_time_zone():
    events, assignment, dim = _tenure_frames()
    dim = pl.DataFrame(
        {"customer_id": ["a"], "tz": [None]},
        schema_overrides={"tz": pl.String},
    )
    with pytest.raises(ValueError, match="must not contain nulls"):
        tenure_outcomes(events, assignment, dim, 5, as_of_date=date(2026, 1, 7))


def test_tenure_rejects_null_customer_ids():
    events, assignment, dim = _tenure_frames()
    null_events = events.with_columns(pl.lit(None, dtype=pl.String).alias("customer_id"))
    with pytest.raises(ValueError, match="events customer_id must not contain nulls"):
        tenure_outcomes(null_events, assignment, dim, 5, as_of_date=date(2026, 1, 7))

    null_assignment = assignment.with_columns(pl.lit(None, dtype=pl.String).alias("customer_id"))
    with pytest.raises(ValueError, match="assignment customer_id must not contain nulls"):
        tenure_outcomes(events, null_assignment, dim, 5, as_of_date=date(2026, 1, 7))

    null_dim = dim.with_columns(pl.lit(None, dtype=pl.String).alias("customer_id"))
    with pytest.raises(ValueError, match="dim_customer customer_id must not contain nulls"):
        tenure_outcomes(events, assignment, null_dim, 5, as_of_date=date(2026, 1, 7))


def test_tenure_accepts_empty_events_and_assignment():
    _, assignment, dim = _tenure_frames()
    events = pl.DataFrame(
        schema={
            "customer_id": pl.String,
            "event_ts_utc": pl.Datetime("us", "UTC"),
        }
    )
    result = tenure_outcomes(events, assignment, dim, 5, as_of_date=date(2026, 1, 7))
    assert result.row(0, named=True)["event_count"] == 0
    assert result.row(0, named=True)["outcome"] == 0
    empty = tenure_outcomes(events, assignment.clear(), dim, 5, as_of_date=date(2026, 1, 7))
    assert empty.height == 0


def test_tenure_outcomes_enforces_window_and_maturity_boundaries() -> None:
    assignment = pl.DataFrame(
        {
            "customer_id": ["mature", "boundary", "immature"],
            "experiment_id": ["offer"] * 3,
            "arm": ["control", "test", "control"],
            "assigned_date": [date(2026, 1, 1), date(2026, 1, 3), date(2026, 1, 4)],
            "stratum": ["Prime", "Watch", "Nurture"],
        }
    )
    events = pl.DataFrame(
        {
            "customer_id": ["mature", "mature", "mature", "boundary", "immature"],
            "event_ts_utc": [
                datetime(2025, 12, 31, 15, tzinfo=UTC),
                datetime(2026, 1, 5, 15, tzinfo=UTC),
                datetime(2026, 1, 6, 15, tzinfo=UTC),
                datetime(2026, 1, 7, 15, tzinfo=UTC),
                datetime(2026, 1, 4, 15, tzinfo=UTC),
            ],
        },
        schema_overrides={"event_ts_utc": pl.Datetime("us", "UTC")},
    )
    dim = pl.DataFrame(
        {
            "customer_id": ["mature", "boundary", "immature"],
            "tz": ["America/New_York"] * 3,
        }
    )
    result = tenure_outcomes(events, assignment, dim, 5, as_of_date=date(2026, 1, 7))

    by_customer = {row["customer_id"]: row for row in result.iter_rows(named=True)}
    assert by_customer["mature"]["event_count"] == 1
    assert by_customer["mature"]["outcome"] == 1
    assert not by_customer["mature"]["immature"]
    assert by_customer["boundary"]["event_count"] == 1
    assert by_customer["boundary"]["outcome"] == 1
    assert not by_customer["boundary"]["immature"]
    assert by_customer["immature"]["event_count"] == 1
    assert by_customer["immature"]["outcome"] == 1
    assert by_customer["immature"]["immature"]
