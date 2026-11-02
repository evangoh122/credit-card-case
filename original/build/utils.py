"""Shared helpers: RNG, IDs, date maths, parquet IO, logging."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config as C

# --------------------------------------------------------------------------
# RNG
# --------------------------------------------------------------------------
_MASTER = np.random.default_rng(C.SEED)


def rng_for(name: str) -> np.random.Generator:
    """Deterministic, independent stream per build step."""
    tag = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    return np.random.default_rng(np.random.SeedSequence([C.SEED, tag]))


# --------------------------------------------------------------------------
# IDs and hashing
# --------------------------------------------------------------------------
_SALT = "synthetic-us-cc-2026"


def hash_id(value: str, length: int = 64) -> str:
    return hashlib.sha256(f"{_SALT}|{value}".encode()).hexdigest()[:length]


def mask_id(value: str, keep: int = 4) -> str:
    s = str(value)
    return "*" * max(0, len(s) - keep) + s[-keep:]


def synthetic_ssn(rng: np.random.Generator) -> str:
    """Structurally valid but non-issuable SSN (900-999 area range)."""
    area = rng.integers(900, 1000)
    group = rng.integers(1, 100)
    serial = rng.integers(1, 10000)
    return f"{area:03d}-{group:02d}-{serial:04d}"


def synthetic_phone(rng: np.random.Generator) -> str:
    area = int(rng.choice([201, 212, 305, 312, 404, 415, 469, 512, 602, 617, 704, 713, 718,
                           786, 801, 813, 917, 919, 972, 206]))
    exch = int(rng.integers(200, 1000))
    line = int(rng.integers(0, 10000))
    return f"+1-{area}-{exch:03d}-{line:04d}"


def synthetic_ip(rng: np.random.Generator, foreign: bool = False) -> str:
    if foreign:
        first = int(rng.choice([81, 86, 91, 103, 141, 185, 193, 196, 203, 5]))
    else:
        first = int(rng.choice([24, 47, 50, 66, 67, 68, 71, 72, 73, 75, 98, 99, 108, 174, 184]))
    return f"{first}.{int(rng.integers(0,256))}.{int(rng.integers(0,256))}.{int(rng.integers(1,255))}"


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------
def daterange(a: dt.date, b: dt.date):
    d = a
    while d <= b:
        yield d
        d += dt.timedelta(days=1)


def month_iter(a: dt.date, b: dt.date):
    """Yield (year, month) tuples inclusive."""
    y, m = a.year, a.month
    while (y, m) <= (b.year, b.month):
        yield y, m
        m += 1
        if m == 13:
            m = 1
            y += 1


def us_holidays(year: int) -> set[dt.date]:
    """US federal holidays with weekend observation."""
    def nth_weekday(y, m, weekday, n):
        d = dt.date(y, m, 1)
        while d.weekday() != weekday:
            d += dt.timedelta(days=1)
        return d + dt.timedelta(days=7 * (n - 1))

    def last_weekday(y, m, weekday):
        d = dt.date(y, m + 1, 1) - dt.timedelta(days=1) if m < 12 else dt.date(y, 12, 31)
        while d.weekday() != weekday:
            d -= dt.timedelta(days=1)
        return d

    def observed(d):
        if d.weekday() == 5:
            return d - dt.timedelta(days=1)
        if d.weekday() == 6:
            return d + dt.timedelta(days=1)
        return d

    raw = [
        dt.date(year, 1, 1),
        nth_weekday(year, 1, 0, 3),
        nth_weekday(year, 2, 0, 3),
        last_weekday(year, 5, 0),
        dt.date(year, 6, 19),
        dt.date(year, 7, 4),
        nth_weekday(year, 9, 0, 1),
        nth_weekday(year, 10, 0, 2),
        dt.date(year, 11, 11),
        nth_weekday(year, 11, 3, 4),
        dt.date(year, 12, 25),
    ]
    out = set()
    for d in raw:
        out.add(d)
        out.add(observed(d))
    return out


HOLIDAYS = set()
for _y in range(C.HISTORY_START.year, C.PERIOD_END.year + 2):
    HOLIDAYS |= us_holidays(_y)


# --------------------------------------------------------------------------
# Parquet IO
# --------------------------------------------------------------------------
def ensure_dirs():
    for d in (C.DIM_DIR, C.FACT_DIR, C.RAW_DIR, C.DOC_DIR, C.OUT / C.ML_DIR_NAME):
        Path(d).mkdir(parents=True, exist_ok=True)


def parquet_metadata(table_name: str | None, path: Path) -> dict[str, str]:
    """Table-level lineage metadata shared by every parquet writer."""
    return {
        "dataset": table_name or Path(path).stem,
        "generator": "synthetic-us-credit-card/1.0",
        "seed": str(C.SEED),
        "period": f"{C.PERIOD_START}..{C.PERIOD_END}",
    }


def write_parquet(df: pd.DataFrame, path: Path, table_name: str | None = None):
    """Write with snappy + table-level metadata for downstream lineage."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    md = {k.encode(): v.encode() for k, v in parquet_metadata(table_name, path).items()}
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    existing = tbl.schema.metadata or {}
    existing.update(md)
    tbl = tbl.replace_schema_metadata(existing)
    pq.write_table(tbl, str(path), compression="snappy")
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    return pq.read_table(str(path)).to_pandas()


# --------------------------------------------------------------------------
# Column buffer (fast append for the sequential simulator)
# --------------------------------------------------------------------------
class Buf:
    __slots__ = ("d",)

    def __init__(self):
        self.d: dict[str, list] = {}

    def add(self, **kw):
        for k, v in kw.items():
            self.d.setdefault(k, []).append(v)

    def frame(self, columns=None) -> pd.DataFrame:
        df = pd.DataFrame(self.d)
        if columns:
            for c in columns:
                if c not in df.columns:
                    df[c] = None
            df = df[columns]
        return df


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
_T0 = time.time()


def log(msg: str):
    elapsed = time.time() - _T0
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)


def run_step(name: str, fn, *a, **kw):
    log(f"START {name}")
    try:
        res = fn(*a, **kw)
        log(f"DONE  {name}")
        return res
    except Exception:
        log(f"FAILED {name}")
        traceback.print_exc()
        raise


def json_dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), default=str)


def lognormal_amount(rng: np.random.Generator, median: float, sigma: float,
                     lo: float, hi: float, n: int) -> np.ndarray:
    vals = rng.lognormal(mean=np.log(median), sigma=sigma, size=n)
    return np.clip(np.round(vals, 2), lo, hi)
