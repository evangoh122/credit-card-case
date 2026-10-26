import logging
import math
from collections.abc import Mapping, Sequence
from datetime import date

import polars as pl
from scipy.optimize import brentq
from scipy.stats import norm

from ccc.rng import unit_hash
from ccc.timeutil import local_date_expr, validate_zones

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
    assigned_date: date,
    salt: int = 0,
) -> pl.DataFrame:
    """Assign eligible customers to stable hash buckets within each segment."""
    items = _validate_arms(arms)
    required = {"customer_id", "segment"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if df.get_column("customer_id").n_unique() != df.height:
        raise ValueError("customer_id must be unique")
    if df.get_column("segment").null_count():
        raise ValueError("segment must not contain nulls")

    eligible = df.filter(pl.col("segment") != "Decline")
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
        rows.append((customer_id, experiment_id, arm, assigned_date, stratum))

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
    assigned_date: date,
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """Choose a balanced salt before exposure; never reassign an active experiment."""
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
    return math.ceil(_required_n(baseline_rate, mde, alpha, power))


def _required_n(baseline_rate: float, mde: float, alpha: float, power: float) -> float:
    alternative_rate = baseline_rate + mde
    midpoint = (baseline_rate + alternative_rate) / 2
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    numerator = (
        z_alpha * math.sqrt(2 * midpoint * (1 - midpoint))
        + z_power
        * math.sqrt(baseline_rate * (1 - baseline_rate) + alternative_rate * (1 - alternative_rate))
    ) ** 2
    return numerator / mde**2


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
        return _required_n(baseline_rate, mde, alpha, power) - n_per_arm

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
    if "Decline" in segment_shares:
        raise ValueError("Decline must be excluded from experiment segment shares")
    if any(not 0 <= share <= 1 for share in segment_shares.values()):
        raise ValueError("segment shares must be between 0 and 1")
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
    if outcomes.get_column("customer_id").n_unique() != outcomes.height:
        raise ValueError("outcomes must contain at most one row per customer")
    outcome_columns = ["customer_id", outcome_col]
    if "immature" in outcomes.columns:
        outcome_columns.append("immature")
    joined = assignment.join(outcomes.select(outcome_columns), on="customer_id", how="left")
    if joined.get_column(outcome_col).null_count() == joined.height:
        raise ValueError("assignment and outcomes have no customers in common")
    joined = joined.with_columns(pl.col(outcome_col).cast(pl.Int8, strict=True))
    values = joined.get_column(outcome_col).drop_nulls()
    if not values.is_in([0, 1]).all():
        raise ValueError("outcome must be binary")
    if "immature" not in joined.columns:
        joined = joined.with_columns(pl.lit(False).alias("immature"))
    analysed = joined.filter(
        pl.col(outcome_col).is_not_null() & ~pl.col("immature").fill_null(False)
    )
    z = norm.ppf(1 - alpha / 2)
    rows = []
    estimates: dict[str, list[dict[str, float | int | str]]] = {}
    strata = assignment.get_column("stratum").unique(maintain_order=True).to_list()
    arms = assignment.filter(pl.col("arm") != "control").get_column("arm").unique().to_list()
    for stratum in strata:
        assigned_group = joined.filter(pl.col("stratum") == stratum)
        analysed_group = analysed.filter(pl.col("stratum") == stratum)
        control = analysed_group.filter(pl.col("arm") == "control")
        for arm in arms:
            treatment = analysed_group.filter(pl.col("arm") == arm)
            if control.is_empty() or treatment.is_empty():
                continue
            control_rate = float(control.get_column(outcome_col).mean())
            treatment_rate = float(treatment.get_column(outcome_col).mean())
            effect = treatment_rate - control_rate
            variance = (
                treatment_rate * (1 - treatment_rate) / treatment.height
                + control_rate * (1 - control_rate) / control.height
            )
            estimate = {
                "stratum": stratum,
                "arm": arm,
                "stratum_size": assigned_group.height,
                "n_control_assigned": assigned_group.filter(pl.col("arm") == "control").height,
                "n_arm_assigned": assigned_group.filter(pl.col("arm") == arm).height,
                "n_control_analysed": control.height,
                "n_arm_analysed": treatment.height,
                "control_rate": control_rate,
                "arm_rate": treatment_rate,
                "effect": effect,
                "variance": variance,
            }
            estimates.setdefault(arm, []).append(estimate)
            rows.append(_uplift_row(estimate, z))

    for arm, arm_estimates in estimates.items():
        total_size = sum(int(item["stratum_size"]) for item in arm_estimates)
        weights = [int(item["stratum_size"]) / total_size for item in arm_estimates]
        effect = sum(
            weight * float(item["effect"])
            for weight, item in zip(weights, arm_estimates, strict=True)
        )
        variance = sum(
            weight**2 * float(item["variance"])
            for weight, item in zip(weights, arm_estimates, strict=True)
        )
        overall = {
            "stratum": "overall",
            "arm": arm,
            "n_control_assigned": sum(int(item["n_control_assigned"]) for item in arm_estimates),
            "n_arm_assigned": sum(int(item["n_arm_assigned"]) for item in arm_estimates),
            "n_control_analysed": sum(int(item["n_control_analysed"]) for item in arm_estimates),
            "n_arm_analysed": sum(int(item["n_arm_analysed"]) for item in arm_estimates),
            "control_rate": sum(
                weight * float(item["control_rate"])
                for weight, item in zip(weights, arm_estimates, strict=True)
            ),
            "arm_rate": sum(
                weight * float(item["arm_rate"])
                for weight, item in zip(weights, arm_estimates, strict=True)
            ),
            "effect": effect,
            "variance": variance,
        }
        rows.append(_uplift_row(overall, z))
    return pl.DataFrame(
        rows,
        schema={
            "stratum": pl.String,
            "arm": pl.String,
            "n_control_assigned": pl.Int64,
            "n_arm_assigned": pl.Int64,
            "n_control_analysed": pl.Int64,
            "n_arm_analysed": pl.Int64,
            "control_rate": pl.Float64,
            "arm_rate": pl.Float64,
            "uplift": pl.Float64,
            "ci_low": pl.Float64,
            "ci_high": pl.Float64,
        },
        orient="row",
    )


def _uplift_row(estimate: Mapping[str, float | int | str], z: float) -> tuple:
    standard_error = math.sqrt(float(estimate["variance"]))
    effect = float(estimate["effect"])
    return (
        estimate["stratum"],
        estimate["arm"],
        estimate["n_control_assigned"],
        estimate["n_arm_assigned"],
        estimate["n_control_analysed"],
        estimate["n_arm_analysed"],
        estimate["control_rate"],
        estimate["arm_rate"],
        effect,
        effect - z * standard_error,
        effect + z * standard_error,
    )


def tenure_outcomes(
    events: pl.DataFrame,
    assignment: pl.DataFrame,
    dim_customer: pl.DataFrame,
    window_days: int,
    *,
    as_of_date: date,
) -> pl.DataFrame:
    """Reduce events to equal-tenure customer outcomes and flag incomplete windows.

    Event dates are each customer's local business date derived from the UTC instant
    and the zone held in dim_customer.
    """
    if window_days < 1:
        raise ValueError("window_days must be positive")
    checks = (
        ("events", events, {"customer_id", "event_ts_utc"}),
        ("dim_customer", dim_customer, {"customer_id", "tz"}),
        (
            "assignment",
            assignment,
            {"customer_id", "experiment_id", "arm", "assigned_date", "stratum"},
        ),
    )
    for name, frame, required in checks:
        if missing := required.difference(frame.columns):
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
    dim_ids = dim_customer.get_column("customer_id")
    if dim_ids.null_count():
        raise ValueError("dim_customer customer_id must not contain nulls")
    if dim_ids.n_unique() != dim_customer.height:
        raise ValueError("dim_customer customer_id must be unique")
    validate_zones(dim_customer.get_column("tz"))
    for name, frame in (("assignment", assignment), ("events", events)):
        if frame.get_column("customer_id").null_count():
            raise ValueError(f"{name} customer_id must not contain nulls")
        unknown = (
            frame.select("customer_id")
            .unique()
            .join(dim_customer.select("customer_id"), on="customer_id", how="anti")
        )
        if unknown.height:
            raise ValueError(
                f"{name} customer_ids missing from dim_customer: "
                f"{sorted(unknown.get_column('customer_id').to_list(), key=str)[:5]}"
            )
    base = assignment.with_columns(
        (pl.lit(as_of_date) < pl.col("assigned_date") + pl.duration(days=window_days - 1)).alias(
            "immature"
        )
    )
    local = events.join(
        dim_customer.select("customer_id", "tz"), on="customer_id", how="inner"
    ).with_columns(local_date_expr("event_ts_utc", "tz").alias("event_local_date"))
    in_window = (
        local.join(assignment.select("customer_id", "assigned_date"), on="customer_id", how="inner")
        .filter(
            (pl.col("event_local_date") >= pl.col("assigned_date"))
            & (pl.col("event_local_date") < pl.col("assigned_date") + pl.duration(days=window_days))
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
