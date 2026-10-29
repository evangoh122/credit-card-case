from datetime import UTC, date, datetime, time, timedelta
from functools import cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import polars as pl


@cache
def zone(tz: str) -> ZoneInfo:
    if not isinstance(tz, str):
        raise ValueError(f"unknown IANA time zone: {tz!r}")
    try:
        return ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"unknown IANA time zone: {tz!r}") from exc


def validate_zones(zones: pl.Series) -> None:
    if zones.null_count():
        raise ValueError("time zone column must not contain nulls")
    for name in zones.unique(maintain_order=True):
        zone(name)


def _aware(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return ts


def utc_to_local(ts: datetime, tz: str) -> datetime:
    return _aware(ts).astimezone(zone(tz))


def utc_to_local_date(ts: datetime, tz: str) -> date:
    return utc_to_local(ts, tz).date()


def local_to_utc(wall: datetime, tz: str) -> datetime:
    """Resolve a naive local wall time in an IANA zone to a UTC instant.

    Ambiguous fall-back wall times resolve to their first (earlier) occurrence.
    Nonexistent spring-forward wall times shift forward by the DST gap.
    """
    if wall.tzinfo is not None:
        raise ValueError("local wall time must not carry a UTC offset")
    tzinfo = zone(tz)

    def instant(fold: int) -> datetime:
        return wall.replace(tzinfo=tzinfo, fold=fold).astimezone(UTC)

    def valid(candidate: datetime) -> bool:
        return candidate.astimezone(tzinfo).replace(tzinfo=None) == wall

    first, second = instant(0), instant(1)
    if valid(first) or valid(second):
        return min(first, second)

    before = (wall - timedelta(hours=3)).replace(tzinfo=tzinfo)
    after = (wall + timedelta(hours=3)).replace(tzinfo=tzinfo)
    gap = after.utcoffset() - before.utcoffset()
    return local_to_utc(wall + gap, tz)


def _per_zone(values: pl.Series, *, dates: bool) -> pl.Series:
    stamps = values.struct.field("ts")
    zones = values.struct.field("tz")
    if stamps.dtype.time_zone is None:
        raise ValueError("timestamp column must be timezone-aware")
    validate_zones(zones)
    parts = []
    for name in zones.unique(maintain_order=True):
        mask = zones == name
        local = stamps.filter(mask).dt.convert_time_zone(name)
        if dates:
            local = local.dt.date()
        else:
            local = local.dt.replace_time_zone(None)
        parts.append(
            pl.DataFrame(
                {
                    "_row": pl.Series(range(len(stamps))).filter(mask),
                    "_value": local,
                }
            )
        )
    if not parts:
        return pl.Series("_value", [], dtype=pl.Date if dates else pl.Datetime("us"))
    return pl.concat(parts).sort("_row")["_value"]


def local_date_expr(ts_col: str, tz_col: str) -> pl.Expr:
    return pl.struct(
        pl.col(ts_col).alias("ts"),
        pl.col(tz_col).alias("tz"),
    ).map_batches(lambda values: _per_zone(values, dates=True), return_dtype=pl.Date)


def local_datetime_expr(ts_col: str, tz_col: str) -> pl.Expr:
    return pl.struct(
        pl.col(ts_col).alias("ts"),
        pl.col(tz_col).alias("tz"),
    ).map_batches(
        lambda values: _per_zone(values, dates=False),
        return_dtype=pl.Datetime("us"),
    )


def local_day_bounds_utc(d: date, tz: str) -> tuple[datetime, datetime]:
    """Half-open [local midnight, next local midnight) for one zone, as UTC instants."""
    start = local_to_utc(datetime.combine(d, time()), tz)
    end = local_to_utc(datetime.combine(d + timedelta(days=1), time()), tz)
    return start, end


def local_day_hours(d: date, tz: str) -> float:
    lo, hi = local_day_bounds_utc(d, tz)
    return (hi - lo).total_seconds() / 3600


def snapshot_cutoff_utc(snapshot_date: date, tz: str) -> datetime:
    """Exclusive end of a customer-local calendar day as a UTC instant.

    The snapshot for ``snapshot_date`` includes every instant in that local day,
    so the half-open interval ends at the next local midnight. Building it from
    local wall time keeps 23/25-hour DST days exact and preserves fractional
    seconds instead of rounding down to 23:59:59.
    """
    return local_to_utc(datetime.combine(snapshot_date + timedelta(days=1), time()), tz)


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


def window_bounds_utc(start: date, days: int, tz: str) -> tuple[datetime, datetime]:
    """Half-open [start of first local day, start after the last) as UTC instants."""
    _check_days(days)
    lo = local_to_utc(datetime.combine(start, time()), tz)
    hi = local_to_utc(datetime.combine(start + timedelta(days=days), time()), tz)
    return lo, hi
