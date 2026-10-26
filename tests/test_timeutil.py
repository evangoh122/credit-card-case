from datetime import UTC, date, datetime, timedelta

import pytest

from ccc.timeutil import (
    ET,
    add_business_days,
    et_day_hours,
    is_business_day,
    sim_date,
    snapshot_cutoff_utc,
    utc_to_et_date,
    window_bounds_utc,
    window_dates,
)


def utc(*args):
    return datetime(*args, tzinfo=UTC)


def test_dst_day_lengths():
    assert et_day_hours(date(2026, 11, 1)) == 25
    assert et_day_hours(date(2026, 3, 8)) == 23
    assert et_day_hours(date(2026, 6, 15)) == 24
    # Naive 24h arithmetic from local midnight lands on the wrong wall-clock time.
    naive_end = datetime(2026, 11, 1, tzinfo=ET).astimezone(UTC) + timedelta(hours=24)
    assert naive_end.astimezone(ET).hour == 23
    assert naive_end != window_bounds_utc(date(2026, 11, 1), 1)[1]


def test_fall_back_boundary_maps_to_correct_et_date():
    # 04:30Z is 00:30 EDT, 05:30Z is 01:30 EDT, 06:30Z is the repeated 01:30, now EST.
    for hour, minute in [(4, 30), (5, 30), (6, 30)]:
        assert utc_to_et_date(utc(2026, 11, 1, hour, minute)) == date(2026, 11, 1)
    assert utc(2026, 11, 1, 5, 30).astimezone(ET).utcoffset() == timedelta(hours=-4)
    assert utc(2026, 11, 1, 6, 30).astimezone(ET).utcoffset() == timedelta(hours=-5)
    # 03:59Z is still Oct 31 evening ET; 04:00Z is the first instant of Nov 1.
    assert utc_to_et_date(utc(2026, 11, 1, 3, 59)) == date(2026, 10, 31)
    assert utc_to_et_date(utc(2026, 11, 1, 4, 0)) == date(2026, 11, 1)
    # End of the 25-hour day: 04:59Z Nov 2 is 23:59 EST, 05:00Z is Nov 2.
    assert utc_to_et_date(utc(2026, 11, 2, 4, 59)) == date(2026, 11, 1)
    assert utc_to_et_date(utc(2026, 11, 2, 5, 0)) == date(2026, 11, 2)


def test_naive_timestamp_rejected():
    with pytest.raises(ValueError):
        utc_to_et_date(datetime(2026, 1, 1))


def test_seven_calendar_day_window_across_fall_back():
    start = date(2026, 10, 28)
    dates = window_dates(start, 7)
    assert dates == [date(2026, 10, 28) + timedelta(days=i) for i in range(7)]
    assert date(2026, 11, 1) in dates

    lo, hi = window_bounds_utc(start, 7)
    assert (hi - lo) == timedelta(days=7, hours=1)
    assert utc_to_et_date(lo) == date(2026, 10, 28)
    assert utc_to_et_date(hi - timedelta(seconds=1)) == date(2026, 11, 3)
    assert utc_to_et_date(hi) == date(2026, 11, 4)

    # Naive 7*24h window is an hour short and ends a day early in ET.
    naive_hi = lo + timedelta(days=7)
    assert naive_hi != hi
    assert utc_to_et_date(naive_hi) == date(2026, 11, 3)


def test_seven_calendar_day_window_across_spring_forward():
    lo, hi = window_bounds_utc(date(2026, 3, 5), 7)
    assert hi - lo == timedelta(days=7) - timedelta(hours=1)
    assert utc_to_et_date(hi) == date(2026, 3, 12)


def test_snapshot_cutoff_either_side_of_dst():
    assert snapshot_cutoff_utc(date(2026, 10, 31)) == utc(2026, 11, 1, 3, 59, 59)  # EDT, UTC-4
    assert snapshot_cutoff_utc(date(2026, 11, 1)) == utc(2026, 11, 2, 4, 59, 59)  # EST, UTC-5
    assert snapshot_cutoff_utc(date(2026, 3, 7)) == utc(2026, 3, 8, 4, 59, 59)  # EST
    assert snapshot_cutoff_utc(date(2026, 3, 8)) == utc(2026, 3, 9, 3, 59, 59)  # EDT
    for d in (date(2026, 11, 1), date(2026, 3, 8)):
        cutoff = snapshot_cutoff_utc(d)
        assert utc_to_et_date(cutoff) == d
        assert utc_to_et_date(cutoff + timedelta(seconds=1)) == d + timedelta(days=1)
    # A fixed -5h offset would put the Nov 1 cutoff in the wrong instant before the change.
    assert snapshot_cutoff_utc(date(2026, 10, 31)) != utc(2026, 11, 1, 4, 59, 59)


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
            window_bounds_utc(date(2026, 1, 1), days)
