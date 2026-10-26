import math

import polars as pl

from ccc.sim import run
from ccc.timeutil import local_datetime_expr

SEEDS = (101, 102, 103, 104, 105)
Z = 3.0
DAYS = 12


def estimate(tables, campaign, sc):
    c = next(c for c in sc.campaigns if c.name == campaign)
    truth = tables["sim_truth_customer"]
    buys = (
        tables["fact_transactions"]
        .filter(pl.col("txn_type") == "purchase")
        .join(tables["dim_customer"].select("customer_id", "tz"), on="customer_id")
        .with_columns(local_datetime_expr("event_ts_utc", "tz").alias("local_ts"))
        .filter((pl.col("local_ts") >= c.start_local_dt) & (pl.col("local_ts") < c.end_local_dt))
    )
    counts = buys.group_by("customer_id").len()
    df = (
        truth.join(counts, on="customer_id", how="left")
        .with_columns(pl.col("len").fill_null(0))
        .filter(pl.col(f"{campaign}_arm") != "none")
    )
    arms = {a: df.filter(pl.col(f"{campaign}_arm") == a) for a in ("test", "control")}
    m = {a: x["len"].mean() for a, x in arms.items()}
    se = {a: x["len"].std() / math.sqrt(x.height) for a, x in arms.items()}
    ratio = m["test"] / m["control"]
    ratio_se = ratio * math.sqrt(
        (se["test"] / m["test"]) ** 2 + (se["control"] / m["control"]) ** 2
    )
    t = arms["test"]
    planted = (t["spend_rate"] * t[f"{campaign}_true_lift"]).sum() / t["spend_rate"].sum()
    return ratio - 1, ratio_se, planted


def test_planted_lift_recovered_and_null_campaign_shows_nothing(recovery_scenario):
    sc = recovery_scenario
    ests, ses = [], []
    for seed in SEEDS:
        tables = run.simulate(seed, DAYS, sc)
        est, se, planted = estimate(tables, "real", sc)
        assert planted > 0.15
        assert abs(est - planted) < Z * se, (seed, est, planted, se)
        ests.append(est)
        ses.append(se)

        est0, se0, planted0 = estimate(tables, "nothing", sc)
        assert planted0 == 0.0
        assert abs(est0) < Z * se0, (seed, est0, se0)

    pooled = sum(ests) / len(ests)
    pooled_se = math.sqrt(sum(x**2 for x in ses)) / len(ses)
    assert pooled - Z * pooled_se > 0


def test_exposure_split_is_balanced_on_pre_period_traits(recovery_scenario):
    tables = run.simulate(SEEDS[0], 2, recovery_scenario)
    truth = tables["sim_truth_customer"].filter(pl.col("real_arm") != "none")
    means = truth.group_by("real_arm").agg(pl.col("spend_rate").mean(), pl.col("risk").mean())
    spread = means.select(
        (pl.col("spend_rate").max() - pl.col("spend_rate").min()) / pl.col("spend_rate").mean(),
        pl.col("risk").max() - pl.col("risk").min(),
    ).row(0)
    assert spread[0] < 0.2 and spread[1] < 0.08
