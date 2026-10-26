from datetime import date, datetime, timedelta

import polars as pl
import pytest

from ccc.sim import events, overlay, run
from ccc.sim.scenario import Campaign, MacroShock, Scenario
from ccc.timeutil import utc_to_et_date, window_bounds_utc
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
        "join_date",
        "channel",
        "region",
        "loaded_at",
        "data_label",
    ]
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


def test_no_event_date_et_mismatch_and_all_inside_window(sim):
    sc, t = sim
    lo, hi = window_bounds_utc(sc.start_date, DAYS)
    for name in ("fact_transactions", "fact_events"):
        df = t[name]
        derived = [utc_to_et_date(ts) for ts in df["event_ts_utc"]]
        assert derived == df["event_date_et"].to_list(), name
        assert df["event_ts_utc"].min() >= lo and df["event_ts_utc"].max() < hi
        assert df["event_ts_utc"].dtype == pl.Datetime("us", "UTC")


def test_events_not_before_join(sim):
    _, t = sim
    j = t["fact_events"].join(t["dim_customer"], on="customer_id")
    assert (j["event_date_et"] >= j["join_date"]).all()


def test_dst_fall_back_day_counts():
    sc = Scenario("dst", date(2026, 10, 31), initial_customers=800, arrivals_per_day=0.0)
    t = run.simulate(SEED, 3, sc)
    fall = date(2026, 11, 1)
    lo, hi = window_bounds_utc(fall, 1)
    assert hi - lo == timedelta(hours=25)
    ev = pl.concat(
        [
            t["fact_events"].select("event_ts_utc", "event_date_et"),
            t["fact_transactions"].select("event_ts_utc", "event_date_et"),
        ]
    )
    day = ev.filter(pl.col("event_date_et") == fall)
    in_span = ev.filter(pl.col("event_ts_utc").is_between(lo, hi, closed="left"))
    assert day.height == in_span.height
    hours = {int((ts - lo).total_seconds() // 3600) for ts in day["event_ts_utc"]}
    assert hours == set(range(25))
    per_day = ev.group_by("event_date_et").len().sort("event_date_et")
    assert per_day["event_date_et"].to_list() == [date(2026, 10, 31), fall, date(2026, 11, 2)]
    assert per_day["len"].min() > 0.7 * per_day["len"].max()


def test_events_layer_alone_on_hand_built_customers():
    sc = Scenario("bare", date(2026, 10, 30), campaigns=())
    cust = pl.DataFrame(
        {
            "customer_id": ["a", "b"],
            "join_day": [0, 2],
            "risk": [0.9, 0.1],
            "spend_rate": [2.0, 0.5],
            "utilisation": [0.5, 0.5],
        }
    )
    txns, evts = events.simulate_events(3, sc, cust, 4)
    assert txns.schema == pl.Schema(events.TXN_SCHEMA)
    assert evts.schema == pl.Schema(events.EVENT_SCHEMA)
    b_first = min(
        txns.filter(pl.col("customer_id") == "b")["event_date_et"].min(),
        evts.filter(pl.col("customer_id") == "b")["event_date_et"].min(),
    )
    assert b_first >= date(2026, 11, 1)
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
        t["fact_transactions"]
        .filter(pl.col("txn_type") == "purchase")
        .group_by("event_date_et")
        .len()
    )
    daily = dict(zip(buys["event_date_et"], buys["len"], strict=True))
    payday, comparison = daily[date(2026, 11, 15)], daily[date(2026, 11, 8)]
    assert payday > comparison * 1.15
    misses = t["fact_events"].filter(pl.col("event_type") == "missed_payment")
    before = misses.filter(pl.col("event_date_et") < date(2026, 11, 10)).height / 8
    after = misses.filter(pl.col("event_date_et") >= date(2026, 11, 11)).height / 5
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
