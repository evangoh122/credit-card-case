from datetime import date, timedelta

import numpy as np
import polars as pl

from ccc.experiment import (
    assign,
    balance_check,
    mde_for_n,
    sample_size_per_arm,
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


def test_power_matches_textbook_normal_approximation_and_inverts() -> None:
    required = sample_size_per_arm(0.10, 0.05, alpha=0.05, power=0.80)

    assert required == 686
    assert abs(mde_for_n(0.10, required) - 0.05) < 0.0001


def test_uplift_confidence_interval_covers_planted_effect_over_seeds() -> None:
    assignment = assign(customers(8_000), "offer-3", 11, assigned_date=date(2026, 1, 1))
    covered = 0
    for seed in range(10):
        rng = np.random.default_rng(seed)
        probabilities = np.where(assignment["arm"].to_numpy() == "test", 0.25, 0.20)
        outcomes = pl.DataFrame(
            {"customer_id": assignment["customer_id"], "outcome": rng.binomial(1, probabilities)}
        )
        overall = (
            uplift(outcomes, assignment).filter(pl.col("stratum") == "overall").row(0, named=True)
        )
        covered += overall["ci_low"] <= 0.05 <= overall["ci_high"]

    assert covered >= 9


def test_tenure_outcomes_uses_exposure_date_and_flags_immature() -> None:
    assignment = pl.DataFrame(
        {
            "customer_id": ["mature", "immature"],
            "experiment_id": ["offer", "offer"],
            "arm": ["control", "test"],
            "assigned_date": [date(2026, 1, 1), date(2026, 1, 20)],
            "stratum": ["Prime", "Prime"],
        }
    )
    events = pl.DataFrame(
        {
            "customer_id": ["mature", "mature", "immature"],
            "event_date": [
                date(2026, 1, 1) + timedelta(days=3),
                date(2026, 1, 1) + timedelta(days=40),
                date(2026, 1, 25),
            ],
        }
    )
    result = tenure_outcomes(events, assignment, 30, as_of_date=date(2026, 2, 1))

    assert result.filter(pl.col("customer_id") == "mature").row(0, named=True)["event_count"] == 1
    assert not result.filter(pl.col("customer_id") == "mature").row(0, named=True)["immature"]
    assert result.filter(pl.col("customer_id") == "immature").row(0, named=True)["immature"]
