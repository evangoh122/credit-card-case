import polars as pl

from ccc.sim import overlay, profiles
from ccc.sim.run import NO_SHIFT
from ccc.sim.scenario import Campaign, FeeChange, MacroShock, Scenario, Segment
from tests.conftest import START


def population(ids):
    prof = profiles.handwritten_table(1, [(i, 0) for i in ids], NO_SHIFT)
    return prof.with_columns(pl.lit(-5).alias("join_day"))


def scenario_with(**campaign_kwargs):
    camp = Campaign("promo", "2026-10-21T09:00:00", 5, 0.4, **campaign_kwargs)
    return Scenario("s", START, campaigns=(camp,))


def test_arm_depends_only_on_customer_and_campaign():
    sc = scenario_with()
    ids = [f"c{i}" for i in range(400)]
    full = overlay.assign_campaigns(11, sc, population(ids))
    part = overlay.assign_campaigns(11, sc, population(ids[100:150]))
    joined = part.join(full, on="customer_id", suffix="_full")
    assert (joined["promo_arm"] == joined["promo_arm_full"]).all()
    assert full["promo_arm"].value_counts().filter(pl.col("promo_arm") == "test")["count"][0] > 150


def test_ineligible_customers_get_no_arm_and_no_lift():
    sc = scenario_with(segment=Segment(risk_max=0.3))
    df = overlay.assign_campaigns(11, sc, population([f"c{i}" for i in range(500)]))
    out = df.filter(pl.col("risk") > 0.3)
    assert (out["promo_arm"] == "none").all() and (out["promo_true_lift"] == 0).all()
    inn = df.filter(pl.col("risk") <= 0.3)
    assert (inn["promo_arm"] != "none").all()


def test_customers_joining_after_start_are_not_eligible():
    df = population(["a", "b"]).with_columns(pl.Series("join_day", [2, 3]))
    out = overlay.assign_campaigns(1, scenario_with(), df)
    assert out["promo_arm"].to_list()[1] == "none"
    assert out["promo_arm"].to_list()[0] != "none"


def test_true_lift_is_planted_lift_times_responsiveness():
    df = overlay.assign_campaigns(11, scenario_with(), population(["a", "b", "c"]))
    assert df["promo_true_lift"].to_list() == [0.4 * r for r in df["responsiveness"]]


def test_day_multipliers():
    sc = Scenario(
        "s",
        START,
        macro_shock=MacroShock(START.replace(day=25), miss_mult=3.0, ramp_days=10),
        fee_change=FeeChange(START.replace(day=30), purchase_mult=0.5),
    )
    quiet = overlay.day_multipliers(sc, START.replace(day=20))
    assert (quiet.purchase, quiet.miss, quiet.repayment) == (1.0, 1.0, 1.0)
    assert overlay.day_multipliers(sc, START.replace(day=30)).miss == 2.0
    assert overlay.day_multipliers(sc, START.replace(day=30)).purchase == 0.5
    assert overlay.day_multipliers(sc, START.replace(month=11, day=1)).repayment == 3.0
    assert overlay.day_multipliers(sc, START.replace(month=12, day=1)).miss == 3.0
