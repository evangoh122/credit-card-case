from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def utc_to_et_date(ts: datetime) -> date:
    if ts.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return ts.astimezone(ET).date()


def _et_instant(d: date, t: time = time()) -> datetime:
    # Midnight and 23:59:59 are never ambiguous or skipped in ET (transitions are at 02:00).
    return datetime.combine(d, t, tzinfo=ET).astimezone(UTC)


def sim_date(start: date, offset: int) -> date:
    return start + timedelta(days=offset)


def is_business_day(d: date) -> bool:
    return d.weekday() < 5


def add_business_days(start: date, n: int) -> date:
    d = start
    step = 1 if n >= 0 else -1
    remaining = abs(n)
    while remaining:
        d += timedelta(days=step)
        if is_business_day(d):
            remaining -= 1
    return d


def _check_days(days: int) -> None:
    if days < 1:
        raise ValueError("days must be >= 1")


def window_dates(start: date, days: int) -> list[date]:
    _check_days(days)
    return [start + timedelta(days=i) for i in range(days)]


def window_bounds_utc(start: date, days: int) -> tuple[datetime, datetime]:
    """Half-open [start of first ET day, start of day after the last) as UTC instants."""
    _check_days(days)
    return _et_instant(start), _et_instant(start + timedelta(days=days))


def snapshot_cutoff_utc(snapshot_date: date) -> datetime:
    return _et_instant(snapshot_date, time(23, 59, 59))


def et_day_hours(d: date) -> float:
    lo, hi = window_bounds_utc(d, 1)
    return (hi - lo).total_seconds() / 3600
