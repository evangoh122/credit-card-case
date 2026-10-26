from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from ccc.geo import ZONES, city_zone
from ccc.timeutil import (
    add_business_days,
    is_business_day,
    local_date_expr,
    local_day_hours,
    local_to_utc,
    sim_date,
    snapshot_cutoff_utc,
    utc_to_local_date,
    validate_zones,
    window_bounds_utc,
    window_dates,
    zone,
)


def utc(*args):
    return datetime(*args, tzinfo=UTC)


def test_same_utc_instant_has_different_local_dates_by_zone():
    instant = utc(2026, 11, 2, 5, 30)
    assert utc_to_local_date(instant, "America/New_York") == date(2026, 11, 2)
    assert utc_to_local_date(instant, "America/Los_Angeles") == date(2026, 11, 1)
    assert utc_to_local_date(instant, "Pacific/Honolulu") == date(2026, 11, 1)


def test_vectorised_local_dates_match_scalar_conversion():
    frame = pl.DataFrame(
        {
            "event_ts_utc": [
                utc(2026, 11, 2, 5, 30),
                utc(2026, 11, 2, 5, 30),
                utc(2026, 11, 2, 5, 30),
                utc(2026, 3, 8, 9, 30),
            ],
            "tz": [
                "America/New_York",
                "America/Los_Angeles",
                "Pacific/Honolulu",
                "America/Denver",
            ],
        }
    )
    derived = frame.with_columns(local_date_expr("event_ts_utc", "tz").alias("local_date"))
    assert derived["local_date"].to_list() == [
        utc_to_local_date(ts, tz)
        for ts, tz in zip(frame["event_ts_utc"].to_list(), frame["tz"].to_list(), strict=True)
    ]
    assert derived["local_date"].to_list()[:3] == [
        date(2026, 11, 2),
        date(2026, 11, 1),
        date(2026, 11, 1),
    ]


def test_dst_day_lengths_per_zone():
    fall = date(2026, 11, 1)
    spring = date(2026, 3, 8)
    summer = date(2026, 6, 15)
    for tz in ("America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles"):
        assert local_day_hours(fall, tz) == 25
        assert local_day_hours(spring, tz) == 23
        assert local_day_hours(summer, tz) == 24
    for tz in ("America/Phoenix", "Pacific/Honolulu"):
        assert local_day_hours(fall, tz) == 24
        assert local_day_hours(spring, tz) == 24
        assert local_day_hours(summer, tz) == 24


def test_local_to_utc_spring_forward_gap_shifts_forward():
    # 02:30 does not exist in New York on 2026-03-08; it resolves as 03:30 EDT.
    assert local_to_utc(datetime(2026, 3, 8, 2, 30), "America/New_York") == utc(2026, 3, 8, 7, 30)
    assert utc_to_local_date(utc(2026, 3, 8, 7, 30), "America/New_York") == date(2026, 3, 8)


def test_local_to_utc_fall_back_ambiguity_uses_first_occurrence():
    first = local_to_utc(datetime(2026, 11, 1, 1, 30), "America/New_York")
    assert first == utc(2026, 11, 1, 5, 30)
    assert first.astimezone(ZoneInfo("America/New_York")).utcoffset() == timedelta(hours=-4)
    second = datetime(2026, 11, 1, 1, 30, tzinfo=ZoneInfo("America/New_York"), fold=1)
    assert second.astimezone(UTC) == utc(2026, 11, 1, 6, 30)


def test_local_to_utc_rejects_offsets_and_unknown_zones():
    with pytest.raises(ValueError, match="offset"):
        local_to_utc(datetime(2026, 10, 24, 9, tzinfo=UTC), "America/New_York")
    with pytest.raises(ValueError, match="unknown IANA"):
        local_to_utc(datetime(2026, 10, 24, 9), "Mars/Olympus")


def test_wall_clock_campaign_time_is_zone_aware():
    assert local_to_utc(datetime(2026, 10, 24, 9), "America/New_York") == utc(2026, 10, 24, 13)
    assert local_to_utc(datetime(2026, 11, 2, 9), "America/New_York") == utc(2026, 11, 2, 14)
    assert local_to_utc(datetime(2026, 11, 2, 9), "America/Phoenix") == utc(2026, 11, 2, 16)
    assert local_to_utc(datetime(2026, 11, 2, 9), "Pacific/Honolulu") == utc(2026, 11, 2, 19)


def test_zone_and_zone_series_require_valid_iana_names():
    for value in (None, 3):
        with pytest.raises(ValueError, match="unknown IANA time zone"):
            zone(value)
    with pytest.raises(ValueError, match="unknown IANA time zone"):
        zone("Mars/Olympus")
    with pytest.raises(ValueError, match="must not contain nulls"):
        validate_zones(pl.Series("tz", ["America/New_York", None]))
    with pytest.raises(ValueError, match="unknown IANA time zone"):
        validate_zones(pl.Series("tz", ["America/New_York", "Mars/Olympus"]))


def test_naive_timestamp_rejected():
    with pytest.raises(ValueError):
        utc_to_local_date(datetime(2026, 1, 1), "America/New_York")


def test_calendar_window_across_fall_back_is_not_24h_arithmetic():
    start = date(2026, 10, 28)
    dates = window_dates(start, 7)
    assert dates == [date(2026, 10, 28) + timedelta(days=i) for i in range(7)]
    assert date(2026, 11, 1) in dates

    lo, hi = window_bounds_utc(start, 7, "America/New_York")
    assert (hi - lo) == timedelta(days=7, hours=1)
    assert utc_to_local_date(lo, "America/New_York") == date(2026, 10, 28)
    assert utc_to_local_date(hi - timedelta(seconds=1), "America/New_York") == date(2026, 11, 3)
    assert utc_to_local_date(hi, "America/New_York") == date(2026, 11, 4)

    naive_hi = lo + timedelta(days=7)
    assert naive_hi != hi
    assert utc_to_local_date(naive_hi, "America/New_York") == date(2026, 11, 3)


def test_calendar_window_across_spring_forward():
    lo, hi = window_bounds_utc(date(2026, 3, 5), 7, "America/New_York")
    assert hi - lo == timedelta(days=7) - timedelta(hours=1)


def test_snapshot_cutoff_is_local_235959_per_zone():
    ny = snapshot_cutoff_utc(date(2026, 11, 1), "America/New_York")
    la = snapshot_cutoff_utc(date(2026, 11, 1), "America/Los_Angeles")
    phx = snapshot_cutoff_utc(date(2026, 11, 1), "America/Phoenix")
    hnl = snapshot_cutoff_utc(date(2026, 11, 1), "Pacific/Honolulu")
    assert ny == utc(2026, 11, 2, 4, 59, 59)
    assert la == utc(2026, 11, 2, 7, 59, 59)
    assert phx == utc(2026, 11, 2, 6, 59, 59)
    assert hnl == utc(2026, 11, 2, 9, 59, 59)
    for tz, cutoff in (("America/New_York", ny), ("America/Los_Angeles", la)):
        assert utc_to_local_date(cutoff, tz) == date(2026, 11, 1)
        assert utc_to_local_date(cutoff + timedelta(seconds=1), tz) == date(2026, 11, 2)


def test_event_before_pacific_cutoff_but_after_eastern_cutoff():
    instant = utc(2026, 11, 2, 5, 30)
    assert instant < snapshot_cutoff_utc(date(2026, 11, 1), "America/Los_Angeles")
    assert instant > snapshot_cutoff_utc(date(2026, 11, 1), "America/New_York")
    assert utc_to_local_date(instant, "America/Los_Angeles") == date(2026, 11, 1)
    assert utc_to_local_date(instant, "America/New_York") == date(2026, 11, 2)


def test_dst_cutoffs_on_both_sides_of_each_transition():
    assert snapshot_cutoff_utc(date(2026, 10, 31), "America/New_York") == utc(
        2026, 11, 1, 3, 59, 59
    )
    assert snapshot_cutoff_utc(date(2026, 3, 7), "America/New_York") == utc(2026, 3, 8, 4, 59, 59)
    assert snapshot_cutoff_utc(date(2026, 3, 8), "America/New_York") == utc(2026, 3, 9, 3, 59, 59)
    # A fixed -5h offset would put the Nov 1 cutoff an hour late.
    assert snapshot_cutoff_utc(date(2026, 10, 31), "America/New_York") != utc(
        2026, 11, 1, 4, 59, 59
    )


def test_sim_date_uses_calendar_days():
    assert sim_date(date(2026, 10, 30), 3) == date(2026, 11, 2)
    assert sim_date(date(2026, 1, 1), 0) == date(2026, 1, 1)


def test_business_days():
    fri = date(2026, 10, 30)
    assert not is_business_day(date(2026, 10, 31))
    assert is_business_day(fri)
    assert add_business_days(fri, 1) == date(2026, 11, 2)
    assert add_business_days(fri, 5) == date(2026, 11, 6)
    assert add_business_days(date(2026, 11, 2), -1) == fri
    assert add_business_days(fri, 0) == fri


def test_window_requires_positive_days():
    for days in (0, -1):
        with pytest.raises(ValueError):
            window_dates(date(2026, 1, 1), days)
        with pytest.raises(ValueError):
            window_bounds_utc(date(2026, 1, 1), days, "America/New_York")


def test_geo_city_zone_mapping_and_unknown_city():
    assert city_zone("New York") == "America/New_York"
    assert city_zone("Chicago") == "America/Chicago"
    assert city_zone("Denver") == "America/Denver"
    assert city_zone("Los Angeles") == "America/Los_Angeles"
    assert city_zone("Phoenix") == "America/Phoenix"
    assert set(ZONES) == {
        "America/New_York",
        "America/Chicago",
        "America/Denver",
        "America/Phoenix",
        "America/Los_Angeles",
        "Pacific/Honolulu",
    }
    with pytest.raises(ValueError, match="unknown city"):
        city_zone("Atlantis")
