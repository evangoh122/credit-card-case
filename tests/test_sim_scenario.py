from pathlib import Path

import pytest

from ccc.sim import scenario
from ccc.sim.scenario import Campaign

DEFAULT = Path(__file__).parent.parent / "scenarios" / "default.toml"


def test_default_scenario_loads():
    sc = scenario.load(DEFAULT)
    assert [c.name for c in sc.campaigns] == ["welcome_cashback", "null_email"]
    assert sc.campaigns[1].lift == 0.0
    assert sc.macro_shock.miss_mult == 2.0


def test_campaign_start_is_local_0900_in_each_customer_zone():
    c = Campaign("a", "2026-10-24T09:00:00", 3, 0.1)
    assert c.local_bounds_utc("America/New_York")[0].isoformat() == "2026-10-24T13:00:00+00:00"
    assert c.local_bounds_utc("America/Los_Angeles")[0].isoformat() == ("2026-10-24T16:00:00+00:00")
    assert c.local_bounds_utc("Pacific/Honolulu")[0].isoformat() == ("2026-10-24T19:00:00+00:00")
    after_dst = Campaign("b", "2026-11-02T09:00:00", 3, 0.1)
    assert after_dst.local_bounds_utc("America/New_York")[0].isoformat() == (
        "2026-11-02T14:00:00+00:00"
    )


def test_campaign_end_uses_wall_clock_days_across_dst():
    c = Campaign("a", "2026-10-30T09:00:00", 3, 0.1)
    assert c.local_bounds_utc("America/New_York")[1].isoformat() == ("2026-11-02T14:00:00+00:00")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_local": "2026-10-24T09:00:00-04:00"},
        {"event": "refund"},
        {"test_share": 1.0},
        {"duration_days": 0},
        {"lift": -0.1},
        {"name": "has space"},
    ],
)
def test_campaign_validation(kwargs):
    base = {"name": "a", "start_local": "2026-10-24T09:00:00", "duration_days": 3, "lift": 0.1}
    with pytest.raises(ValueError):
        Campaign(**{**base, **kwargs})


def test_unknown_keys_and_duplicate_names_rejected():
    with pytest.raises(ValueError, match="unknown"):
        scenario.from_dict({"name": "x", "start_date": "2026-10-19", "lifts": 1})
    camp = {"name": "a", "start_local": "2026-10-24T09:00:00", "duration_days": 3, "lift": 0.1}
    with pytest.raises(ValueError, match="unique"):
        scenario.from_dict({"name": "x", "start_date": "2026-10-19", "campaigns": [camp, camp]})
