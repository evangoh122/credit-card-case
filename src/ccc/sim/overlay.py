from dataclasses import dataclass

import polars as pl

from ccc.rng import unit_hash
from ccc.sim.scenario import Campaign, Scenario
from ccc.timeutil import utc_to_et_date


@dataclass(frozen=True)
class DayMults:
    purchase: float = 1.0
    app_open: float = 1.0
    repayment: float = 1.0
    miss: float = 1.0


def day_multipliers(sc: Scenario, d) -> DayMults:
    purchase = app_open = repayment = miss = 1.0
    if d.day in sc.payday.days_of_month:
        purchase *= sc.payday.purchase_mult
        app_open *= sc.payday.open_mult
        repayment *= sc.payday.repayment_mult
    if sc.macro_shock and d >= sc.macro_shock.start_date:
        progress = min(1.0, (d - sc.macro_shock.start_date).days / max(1, sc.macro_shock.ramp_days))
        miss *= 1 + (sc.macro_shock.miss_mult - 1) * progress
    if sc.fee_change and d >= sc.fee_change.date:
        purchase *= sc.fee_change.purchase_mult
    return DayMults(purchase, app_open, repayment, miss)


def arrival_multiplier(sc: Scenario, d) -> float:
    mult = 1.0
    for c in sc.campaigns:
        if c.start_date_et <= d < utc_to_et_date(c.end_utc):
            mult *= 1 + c.arrival_boost
    return mult


def _eligible(c: Campaign, start_day: int) -> pl.Expr:
    seg = c.segment
    cond = (pl.col("join_day") <= start_day) & pl.col("risk").is_between(seg.risk_min, seg.risk_max)
    if seg.channels:
        cond &= pl.col("channel").is_in(seg.channels)
    if seg.regions:
        cond &= pl.col("region").is_in(seg.regions)
    return cond


def assign_campaigns(seed: int, sc: Scenario, customers: pl.DataFrame) -> pl.DataFrame:
    """Adds `<name>_arm` (test/control/none) and `<name>_true_lift` per campaign.

    Arm depends only on (seed, campaign, customer_id); true_lift is the planted
    relative increase in the campaign's event rate for that customer if exposed.
    """
    out = customers
    ids = customers["customer_id"].to_list()
    for c in sc.campaigns:
        start_day = (c.start_date_et - sc.start_date).days
        eligible = out.select(_eligible(c, start_day).alias("e"))["e"]
        u = pl.Series([unit_hash(seed, f"campaign:{c.name}", cid) for cid in ids])
        arm = (
            pl.when(~eligible)
            .then(pl.lit("none"))
            .when(u < c.test_share)
            .then(pl.lit("test"))
            .otherwise(pl.lit("control"))
        )
        out = out.with_columns(
            arm.alias(f"{c.name}_arm"),
            pl.when(eligible)
            .then(c.lift * pl.col("responsiveness"))
            .otherwise(0.0)
            .alias(f"{c.name}_true_lift"),
        )
    return out
