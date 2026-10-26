import json
import tomllib
from dataclasses import dataclass, field, fields
from datetime import date, datetime, timedelta
from pathlib import Path

from ccc.timeutil import local_to_utc

EVENTS = ("purchase", "app_open")


@dataclass(frozen=True)
class Segment:
    channels: tuple[str, ...] = ()
    regions: tuple[str, ...] = ()
    risk_min: float = 0.0
    risk_max: float = 1.0


@dataclass(frozen=True)
class Campaign:
    name: str
    start_local: str
    duration_days: int
    lift: float
    event: str = "purchase"
    test_share: float = 0.5
    arrival_boost: float = 0.0
    segment: Segment = field(default_factory=Segment)

    def __post_init__(self):
        if not self.name.isidentifier():
            raise ValueError(f"campaign name {self.name!r} must be an identifier")
        if self.event not in EVENTS:
            raise ValueError(f"campaign event must be one of {EVENTS}")
        if not 0 < self.test_share < 1:
            raise ValueError("test_share must be in (0, 1)")
        if self.duration_days < 1 or self.lift < 0 or self.arrival_boost < 0:
            raise ValueError("duration_days >= 1, lift >= 0, arrival_boost >= 0 required")
        if self.start_local_dt.tzinfo is not None:
            raise ValueError("start_local is a local wall time and must not carry an offset")
        if self.end_local_dt <= self.start_local_dt:
            raise ValueError("campaign must end after it starts")

    @property
    def start_local_dt(self) -> datetime:
        return datetime.fromisoformat(self.start_local)

    @property
    def end_local_dt(self) -> datetime:
        return self.start_local_dt + timedelta(days=self.duration_days)

    @property
    def start_local_date(self) -> date:
        return self.start_local_dt.date()

    @property
    def end_local_date(self) -> date:
        return self.end_local_dt.date()

    def local_bounds_utc(self, tz: str) -> tuple[datetime, datetime]:
        return (
            local_to_utc(self.start_local_dt, tz),
            local_to_utc(self.end_local_dt, tz),
        )


@dataclass(frozen=True)
class Payday:
    days_of_month: tuple[int, ...] = (1, 15)
    purchase_mult: float = 1.3
    repayment_mult: float = 3.0
    open_mult: float = 1.15


@dataclass(frozen=True)
class MacroShock:
    start_date: date
    miss_mult: float = 2.0
    ramp_days: int = 14


@dataclass(frozen=True)
class FeeChange:
    date: date
    purchase_mult: float = 0.92


@dataclass(frozen=True)
class CohortShift:
    paid_share_per_day: float = 0.004
    risk_z_per_day: float = 0.01


@dataclass(frozen=True)
class Scenario:
    name: str
    start_date: date
    initial_customers: int = 1000
    arrivals_per_day: float = 40.0
    arrivals_trend: float = 0.005
    campaigns: tuple[Campaign, ...] = ()
    payday: Payday = field(default_factory=Payday)
    macro_shock: MacroShock | None = None
    fee_change: FeeChange | None = None
    cohort_shift: CohortShift = field(default_factory=CohortShift)

    def __post_init__(self):
        names = [c.name for c in self.campaigns]
        if len(set(names)) != len(names):
            raise ValueError("campaign names must be unique")


def _date(v) -> date:
    return v if isinstance(v, date) else date.fromisoformat(v)


def from_dict(raw: dict) -> Scenario:
    raw = dict(raw)
    known = {f.name for f in fields(Scenario)}
    if unknown := set(raw) - known:
        raise ValueError(f"unknown scenario keys: {sorted(unknown)}")
    raw["start_date"] = _date(raw["start_date"])
    camps = []
    for c in raw.pop("campaigns", []):
        c = dict(c)
        seg = {k: tuple(v) if isinstance(v, list) else v for k, v in c.pop("segment", {}).items()}
        camps.append(Campaign(segment=Segment(**seg), **c))
    raw["campaigns"] = tuple(camps)
    if "payday" in raw:
        p = dict(raw["payday"])
        p["days_of_month"] = tuple(p.get("days_of_month", Payday.days_of_month))
        raw["payday"] = Payday(**p)
    if "macro_shock" in raw:
        m = dict(raw["macro_shock"])
        raw["macro_shock"] = MacroShock(**{**m, "start_date": _date(m["start_date"])})
    if "fee_change" in raw:
        f = dict(raw["fee_change"])
        raw["fee_change"] = FeeChange(**{**f, "date": _date(f["date"])})
    if "cohort_shift" in raw:
        raw["cohort_shift"] = CohortShift(**raw["cohort_shift"])
    return Scenario(**raw)


def load(path: str | Path) -> Scenario:
    path = Path(path)
    if path.suffix == ".json":
        raw = json.loads(path.read_text())
    else:
        raw = tomllib.loads(path.read_text())
    return from_dict(raw)
