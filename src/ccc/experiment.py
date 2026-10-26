import logging
import math
from collections.abc import Mapping, Sequence
from datetime import date

import polars as pl
from scipy.optimize import brentq
from scipy.stats import norm

from ccc.rng import unit_hash

LOGGER = logging.getLogger(__name__)
DEFAULT_ARMS = {"control": 0.5, "test": 0.5}
DEFAULT_BALANCE_COLS = ("pre_spend", "arrears", "utilisation")


def _validate_arms(arms: Mapping[str, float]) -> list[tuple[str, float]]:
    items = list(arms.items())
    if len(items) < 2 or "control" not in arms:
        raise ValueError("arms must include a no-offer 'control' arm")
    if any(not name or weight <= 0 or not math.isfinite(weight) for name, weight in items):
        raise ValueError("arm names must be non-empty and weights must be positive and finite")
    total = sum(weight for _, weight in items)
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("arm weights must sum to 1")
    return items


def assign(
    df: pl.DataFrame,
    experiment_id: str,
    root_seed: int,
    arms: Mapping[str, float] = DEFAULT_ARMS,
    *,
    salt: int = 0,
    assigned_date: date | None = None,
) -> pl.DataFrame:
    """Assign eligible customers to stable hash buckets within each segment."""
    items = _validate_arms(arms)
    required = {"customer_id", "segment"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if df.get_column("customer_id").n_unique() != df.height:
        raise ValueError("customer_id must be unique")

    eligible = df.filter(pl.col("segment") != "Decline")
    assignment_day = assigned_date or date.today()
    cumulative: list[tuple[str, float]] = []
    running = 0.0
    for name, weight in items:
        running += weight
        cumulative.append((name, running))

    rows = []
    for customer_id, stratum in eligible.select("customer_id", "segment").iter_rows():
        value = unit_hash(root_seed, f"{experiment_id}:{stratum}:{salt}", customer_id)
        arm = cumulative[-1][0]
        for candidate, boundary in cumulative:
            if value < boundary:
                arm = candidate
                break
        rows.append((customer_id, experiment_id, arm, assignment_day, stratum))

    schema = {
        "customer_id": df.schema["customer_id"],
        "experiment_id": pl.String,
        "arm": pl.String,
        "assigned_date": pl.Date,
        "stratum": df.schema["segment"],
    }
    return pl.DataFrame(rows, schema=schema, orient="row")


def balance_check(
    df: pl.DataFrame,
    assignment: pl.DataFrame,
    cols: Sequence[str] = DEFAULT_BALANCE_COLS,
    *,
    threshold: float = 0.1,
) -> pl.DataFrame:
    """Calculate standardized mean differences against control in each stratum."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    missing = {"customer_id", *cols}.difference(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    joined = assignment.join(df.select("customer_id", *cols), on="customer_id", how="left")
    rows = []
    for stratum in joined.get_column("stratum").unique(maintain_order=True).to_list():
        stratum_df = joined.filter(pl.col("stratum") == stratum)
        control = stratum_df.filter(pl.col("arm") == "control")
        for arm in stratum_df.get_column("arm").unique(maintain_order=True).to_list():
            if arm == "control":
                continue
            treatment = stratum_df.filter(pl.col("arm") == arm)
            for column in cols:
                control_values = control.get_column(column).drop_nulls()
                treatment_values = treatment.get_column(column).drop_nulls()
                if control_values.is_empty() or treatment_values.is_empty():
                    smd = float("nan")
                else:
                    control_mean = float(control_values.mean())
                    treatment_mean = float(treatment_values.mean())
                    control_var = float(control_values.var(ddof=1) or 0.0)
                    treatment_var = float(treatment_values.var(ddof=1) or 0.0)
                    pooled_sd = math.sqrt((control_var + treatment_var) / 2)
                    difference = treatment_mean - control_mean
                    smd = (
                        difference / pooled_sd
                        if pooled_sd
                        else (0.0 if difference == 0 else math.inf)
                    )
                rows.append(
                    (stratum, arm, column, smd, math.isfinite(smd) and abs(smd) <= threshold)
                )
    return pl.DataFrame(
        rows,
        schema={
            "stratum": assignment.schema["stratum"],
            "arm": pl.String,
            "column": pl.String,
            "smd": pl.Float64,
            "balanced": pl.Boolean,
        },
        orient="row",
    )


def reseed(
    df: pl.DataFrame,
    experiment_id: str,
    root_seed: int,
    arms: Mapping[str, float] = DEFAULT_ARMS,
    *,
    cols: Sequence[str] = DEFAULT_BALANCE_COLS,
    threshold: float = 0.1,
    max_salts: int = 100,
    assigned_date: date | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """Try successive salts until all requested balance checks pass."""
    if max_salts < 1:
        raise ValueError("max_salts must be at least 1")
    for salt in range(max_salts):
        assignment = assign(
            df,
            experiment_id,
            root_seed,
            arms,
            salt=salt,
            assigned_date=assigned_date,
        )
        balance = balance_check(df, assignment, cols, threshold=threshold)
        if balance.get_column("balanced").all():
            LOGGER.info("experiment %s uses assignment salt %d", experiment_id, salt)
            return assignment, balance, salt
        LOGGER.warning("experiment %s is imbalanced with salt %d", experiment_id, salt)
    raise RuntimeError(f"no balanced assignment found in {max_salts} salts")


def sample_size_per_arm(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Return the normal-approximation sample size for a two-sided two-proportion test."""
    _validate_power_inputs(baseline_rate, mde, alpha, power)
    alternative_rate = baseline_rate + mde
    midpoint = (baseline_rate + alternative_rate) / 2
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    numerator = (
        z_alpha * math.sqrt(2 * midpoint * (1 - midpoint))
        + z_power
        * math.sqrt(baseline_rate * (1 - baseline_rate) + alternative_rate * (1 - alternative_rate))
    ) ** 2
    return math.ceil(numerator / mde**2)


def _validate_power_inputs(baseline_rate: float, mde: float, alpha: float, power: float) -> None:
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be between 0 and 1")
    if not 0 < mde < 1 - baseline_rate:
        raise ValueError("mde must be positive and keep the alternative rate below 1")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must be between 0 and 1")


def mde_for_n(
    baseline_rate: float,
    n_per_arm: int,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Invert sample_size_per_arm to obtain the detectable absolute uplift."""
    if n_per_arm < 2:
        raise ValueError("n_per_arm must be at least 2")
    if not 0 < baseline_rate < 1 or not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("rates, alpha, and power must be valid probabilities")

    def difference(mde: float) -> float:
        alternative_rate = baseline_rate + mde
        midpoint = (baseline_rate + alternative_rate) / 2
        z_alpha = norm.ppf(1 - alpha / 2)
        z_power = norm.ppf(power)
        required = (
            z_alpha * math.sqrt(2 * midpoint * (1 - midpoint))
            + z_power
            * math.sqrt(
                baseline_rate * (1 - baseline_rate) + alternative_rate * (1 - alternative_rate)
            )
        ) ** 2 / mde**2
        return required - n_per_arm

    upper = 1 - baseline_rate - 1e-12
    if difference(upper) > 0:
        raise ValueError("n_per_arm is too small for an attainable MDE")
    return float(brentq(difference, 1e-12, upper))


def size_arms_per_segment(
    segment_shares: Mapping[str, float],
    baseline_rates: Mapping[str, float],
    *,
    total_customers: int = 50_000,
    mde: float,
    arm_count: int = 2,
    alpha: float = 0.05,
    power: float = 0.8,
) -> pl.DataFrame:
    """Compare available and required customers for each segment."""
    if total_customers < 1 or arm_count < 2:
        raise ValueError("total_customers and arm_count must be positive")
    if set(segment_shares) != set(baseline_rates):
        raise ValueError("segment_shares and baseline_rates must have the same keys")
    if not math.isclose(sum(segment_shares.values()), 1.0, abs_tol=1e-12):
        raise ValueError("segment shares must sum to 1")
    rows = []
    for segment, share in segment_shares.items():
        available = math.floor(total_customers * share / arm_count)
        required = sample_size_per_arm(baseline_rates[segment], mde, alpha, power)
        rows.append((segment, available, required, available >= required))
    return pl.DataFrame(
        rows,
        schema={
            "segment": pl.String,
            "available_per_arm": pl.Int64,
            "required_per_arm": pl.Int64,
            "powered": pl.Boolean,
        },
        orient="row",
    )


def uplift(
    outcomes: pl.DataFrame,
    assignment: pl.DataFrame,
    *,
    outcome_col: str = "outcome",
    alpha: float = 0.05,
) -> pl.DataFrame:
    """Estimate binary-outcome uplift against control by stratum and overall."""
    joined = assignment.join(
        outcomes.select("customer_id", outcome_col), on="customer_id", how="inner"
    )
    if joined.is_empty():
        raise ValueError("assignment and outcomes have no customers in common")
    values = joined.get_column(outcome_col).drop_nulls()
    if values.len() != joined.height or not values.is_in([0, 1, False, True]).all():
        raise ValueError("outcome must be non-null and binary")
    z = norm.ppf(1 - alpha / 2)
    rows = []
    strata: list[str | None] = joined.get_column("stratum").unique(maintain_order=True).to_list()
    for stratum in [*strata, None]:
        group = joined if stratum is None else joined.filter(pl.col("stratum") == stratum)
        control = group.filter(pl.col("arm") == "control")
        if control.is_empty():
            continue
        control_rate = float(control.get_column(outcome_col).mean())
        for arm in group.get_column("arm").unique(maintain_order=True).to_list():
            if arm == "control":
                continue
            treatment = group.filter(pl.col("arm") == arm)
            treatment_rate = float(treatment.get_column(outcome_col).mean())
            effect = treatment_rate - control_rate
            standard_error = math.sqrt(
                treatment_rate * (1 - treatment_rate) / treatment.height
                + control_rate * (1 - control_rate) / control.height
            )
            rows.append(
                (
                    stratum or "overall",
                    arm,
                    control.height,
                    treatment.height,
                    control_rate,
                    treatment_rate,
                    effect,
                    effect - z * standard_error,
                    effect + z * standard_error,
                )
            )
    return pl.DataFrame(
        rows,
        schema={
            "stratum": pl.String,
            "arm": pl.String,
            "n_control": pl.Int64,
            "n_arm": pl.Int64,
            "control_rate": pl.Float64,
            "arm_rate": pl.Float64,
            "uplift": pl.Float64,
            "ci_low": pl.Float64,
            "ci_high": pl.Float64,
        },
        orient="row",
    )


def tenure_outcomes(
    events: pl.DataFrame,
    assignment: pl.DataFrame,
    window_days: int,
    *,
    event_date_col: str = "event_date",
    as_of_date: date | None = None,
) -> pl.DataFrame:
    """Reduce events to equal-tenure customer outcomes and flag incomplete windows."""
    if window_days < 1:
        raise ValueError("window_days must be positive")
    if event_date_col not in events.columns:
        raise ValueError(f"missing event date column {event_date_col!r}")
    observed_through = as_of_date or events.get_column(event_date_col).max()
    if observed_through is None:
        raise ValueError("as_of_date is required when events is empty")

    base = assignment.with_columns(
        ((pl.lit(observed_through) - pl.col("assigned_date")).dt.total_days() < window_days).alias(
            "immature"
        )
    )
    in_window = (
        events.join(
            assignment.select("customer_id", "assigned_date"), on="customer_id", how="inner"
        )
        .filter(
            (pl.col(event_date_col) >= pl.col("assigned_date"))
            & (pl.col(event_date_col) < pl.col("assigned_date") + pl.duration(days=window_days))
        )
        .group_by("customer_id")
        .agg(pl.len().alias("event_count"))
    )
    return (
        base.join(in_window, on="customer_id", how="left")
        .with_columns(
            pl.col("event_count").fill_null(0),
            (pl.col("event_count").fill_null(0) > 0).cast(pl.Int8).alias("outcome"),
        )
        .select(
            "customer_id",
            "experiment_id",
            "arm",
            "assigned_date",
            "stratum",
            "event_count",
            "outcome",
            "immature",
        )
    )
