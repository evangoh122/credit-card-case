from datetime import date, datetime

import polars as pl
import pytest

from ccc.geo import ZONES
from ccc.sim import events, overlay, run
from ccc.sim.scenario import Campaign, MacroShock, Scenario
from ccc.timeutil import local_date_expr, local_day_bounds_utc, sim_date, utc_to_local_date
from tests.conftest import START

SEED = 21
DAYS = 21


@pytest.fixture(scope="module")
def sim():
    sc = Scenario(
        "small",
        START,
        initial_customers=150,
        arrivals_per_day=4.0,
        campaigns=(),
    )
    return sc, run.simulate(SEED, DAYS, sc)


def test_same_seed_same_output_and_other_seed_differs(sim):
    sc, tables = sim
    again = run.simulate(SEED, DAYS, sc)
    for name, df in tables.items():
        assert df.equals(again[name]), name
    other = run.simulate(SEED + 1, DAYS, sc)
    assert not other["fact_transactions"].equals(tables["fact_transactions"])


def _check_adding_customers(small_scenario, source):
    base = run.simulate(SEED, 10, small_scenario, profile_source=source)
    bigger_sc = Scenario(**{**small_scenario.__dict__, "initial_customers": 220})
    bigger = run.simulate(SEED, 10, bigger_sc, profile_source=source)
    ids = base["dim_customer"]["customer_id"]
    assert ids.is_in(bigger["dim_customer"]["customer_id"].implode()).all()
    for name, key in (
        ("dim_customer", "customer_id"),
        ("sim_truth_customer", "customer_id"),
        ("fact_transactions", "transaction_id"),
        ("fact_events", "event_id"),
    ):
        kept = bigger[name].filter(pl.col("customer_id").is_in(ids.implode())).sort(key)
        assert kept.equals(base[name].sort(key)), name


def test_adding_customers_does_not_change_existing_events(small_scenario):
    _check_adding_customers(small_scenario, "handwritten")


@pytest.mark.slow
def test_adding_customers_does_not_change_existing_events_copula(small_scenario):
    _check_adding_customers(small_scenario, "copula")


def test_naive_loaded_at_rejected(small_scenario):
    with pytest.raises(ValueError, match="timezone-aware"):
        run.simulate(SEED, 2, small_scenario, loaded_at=datetime(2026, 11, 1))
    with pytest.raises(ValueError, match="timezone-aware"):
        events.label(pl.DataFrame({"a": [1]}), datetime(2026, 11, 1))


def test_output_schemas_and_labels(sim):
    _, t = sim
    assert t["dim_customer"].columns == [
        "customer_id",
        "name",
        "city",
        "tz",
        "joined_ts_utc",
        "channel",
        "region",
        "loaded_at",
        "data_label",
    ]
    assert t["dim_customer"]["joined_ts_utc"].dtype == pl.Datetime("us", "UTC")
    assert not any(
        name.endswith("_et") or name.endswith("_local") or name == "join_date"
        for name in t["dim_customer"].columns
    )
    for dtype in t["dim_customer"].schema.values():
        if dtype == pl.Datetime:
            assert dtype.time_zone == "UTC"
    latent = {"risk", "spend_rate", "responsiveness", "utilisation"}
    assert not latent & set(t["dim_customer"].columns)
    assert "segment" not in t["dim_customer"].columns
    for df in t.values():
        assert (df["data_label"] == "synthetic").all()
        assert df["loaded_at"].null_count() == 0
    assert t["dim_customer"]["customer_id"].is_unique().all()
    assert set(t["fact_events"]["event_type"]) == {"app_open", "missed_payment"}
    assert set(t["fact_transactions"]["txn_type"]) == {"purchase", "repayment"}
    assert (t["fact_transactions"]["amount"] > 0).all()
    assert t["fact_events"]["event_id"].is_unique().all()
    assert t["fact_transactions"]["transaction_id"].is_unique().all()


def _local(event_frame: pl.DataFrame, dim: pl.DataFrame) -> pl.DataFrame:
    return event_frame.join(dim.select("customer_id", "tz"), on="customer_id").with_columns(
        local_date_expr("event_ts_utc", "tz").alias("local_date")
    )


def test_event_dates_are_utc_and_inside_each_local_day(sim):
    sc, t = sim
    local_days = {sim_date(sc.start_date, i) for i in range(DAYS)}
    for name in ("fact_transactions", "fact_events"):
        df = t[name]
        assert df["event_ts_utc"].dtype == pl.Datetime("us", "UTC")
        assert df["event_date_utc"].dtype == pl.Date
        assert [ts.date() for ts in df["event_ts_utc"]] == df["event_date_utc"].to_list(), name
        derived = _local(df, t["dim_customer"])
        assert set(derived["local_date"].to_list()) <= local_days, name


def test_events_not_before_join(sim):
    _, t = sim
    dim = t["dim_customer"].with_columns(
        local_date_expr("joined_ts_utc", "tz").alias("join_local_date")
    )
    j = (
        t["fact_events"]
        .join(dim.select("customer_id", "tz", "join_local_date"), on="customer_id")
        .with_columns(local_date_expr("event_ts_utc", "tz").alias("local_date"))
    )
    assert (j["local_date"] >= j["join_local_date"]).all()


DST_ZONES = (
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Los_Angeles",
)


@pytest.mark.parametrize("tz", ZONES)
@pytest.mark.parametrize(("day", "dst_hours"), [(date(2026, 11, 1), 25), (date(2026, 3, 8), 23)])
def test_local_day_generation_covers_exact_utc_hour_buckets(tz, day, dst_hours):
    expected = dst_hours if tz in DST_ZONES else 24
    lo, hi = local_day_bounds_utc(day, tz)
    total_s = int((hi - lo).total_seconds())
    assert total_s == expected * 3600
    step = 60
    stamps = [
        events._stamp(lo, total_s, (i + 0.5) * step / total_s)[0] for i in range(total_s // step)
    ]
    assert len(set(stamps)) == len(stamps)
    assert all(lo <= ts < hi for ts in stamps)
    buckets = {int((ts - lo).total_seconds() // 3600) for ts in stamps}
    assert buckets == set(range(expected))
    assert {utc_to_local_date(ts, tz) for ts in stamps} == {day}


def test_events_layer_alone_on_hand_built_customers():
    sc = Scenario("bare", date(2026, 10, 30), campaigns=())
    cust = pl.DataFrame(
        {
            "customer_id": ["a", "b"],
            "join_day": [0, 2],
            "risk": [0.9, 0.1],
            "spend_rate": [2.0, 0.5],
            "utilisation": [0.5, 0.5],
            "tz": ["America/New_York", "America/Los_Angeles"],
        }
    )
    txns, evts = events.simulate_events(3, sc, cust, 4)
    assert txns.schema == pl.Schema(events.TXN_SCHEMA)
    assert evts.schema == pl.Schema(events.EVENT_SCHEMA)
    b_stamps = [
        ts
        for frame in (txns, evts)
        for ts in frame.filter(pl.col("customer_id") == "b")["event_ts_utc"]
    ]
    assert b_stamps
    assert min(utc_to_local_date(ts, "America/Los_Angeles") for ts in b_stamps) >= date(2026, 11, 1)
    assert txns.filter(pl.col("customer_id") == "a").height > 0


def test_payday_spike_and_macro_shock_are_visible():
    sc = Scenario(
        "p",
        date(2026, 11, 2),
        initial_customers=1500,
        arrivals_per_day=0.0,
        macro_shock=MacroShock(date(2026, 11, 10), miss_mult=4.0, ramp_days=1),
    )
    t = run.simulate(5, 16, sc)
    buys = (
        _local(
            t["fact_transactions"].filter(pl.col("txn_type") == "purchase"),
            t["dim_customer"],
        )
        .group_by("local_date")
        .len()
    )
    daily = dict(zip(buys["local_date"], buys["len"], strict=True))
    payday, comparison = daily[date(2026, 11, 15)], daily[date(2026, 11, 8)]
    assert payday > comparison * 1.15
    misses = _local(
        t["fact_events"].filter(pl.col("event_type") == "missed_payment"), t["dim_customer"]
    )
    before = misses.filter(pl.col("local_date") < date(2026, 11, 10)).height / 8
    after = misses.filter(pl.col("local_date") >= date(2026, 11, 11)).height / 5
    assert after > before * 2


def test_arrival_spike_from_campaign(small_scenario):
    quiet = Scenario(**{**small_scenario.__dict__, "campaigns": (), "arrivals_per_day": 60.0})
    loud_camp = Campaign("big", "2026-10-21T09:00:00", 6, 0.0, arrival_boost=1.0)
    loud = Scenario(**{**quiet.__dict__, "campaigns": (loud_camp,)})
    q = events.daily_arrivals(1, quiet, 12)
    ld = events.daily_arrivals(1, loud, 12)
    assert sum(ld[2:8]) > sum(q[2:8]) * 1.4
    assert ld[:2] == q[:2] and ld[8:] == q[8:]
    assert overlay.arrival_multiplier(loud, date(2026, 10, 20)) == 1.0
