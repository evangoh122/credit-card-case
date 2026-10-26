from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
from scipy.stats import beta

from ccc.rng import entity_rng
from ccc.sim.overlay import arrival_multiplier, day_multipliers
from ccc.sim.scenario import Scenario
from ccc.timeutil import sim_date, utc_to_et_date, window_bounds_utc

DATA_LABEL = "synthetic"

_GRID = np.linspace(0.0, 1.0, 2001)
_CDF = beta(2.2, 2.6).cdf(_GRID)

PURCHASE_WEEKDAY = (0.90, 0.90, 0.95, 1.00, 1.15, 1.30, 1.10)
OPEN_WEEKDAY = (1.00, 1.00, 1.00, 1.00, 1.05, 0.90, 0.90)
WEEKEND_PAYMENT_MULT = 0.3

EVENT_SCHEMA = {
    "event_id": pl.String,
    "customer_id": pl.String,
    "event_type": pl.String,
    "event_ts_utc": pl.Datetime("us", "UTC"),
    "event_date_et": pl.Date,
}
TXN_SCHEMA = {
    "transaction_id": pl.String,
    "customer_id": pl.String,
    "txn_type": pl.String,
    "amount": pl.Float64,
    "event_ts_utc": pl.Datetime("us", "UTC"),
    "event_date_et": pl.Date,
}


def purchase_rate(spend_rate: float) -> float:
    return spend_rate


def open_rate(spend_rate: float) -> float:
    return 0.5 + 1.5 * spend_rate


def daily_arrivals(seed: int, sc: Scenario, days: int) -> list[int]:
    out = []
    for d in range(days):
        lam = sc.arrivals_per_day * (1 + sc.arrivals_trend * d)
        lam *= arrival_multiplier(sc, sim_date(sc.start_date, d))
        out.append(int(entity_rng(seed, "arrivals", "cohort", d).poisson(lam)))
    return out


def _ppf(q):
    return np.interp(q, _CDF, _GRID)


def _cdf(x):
    return np.interp(x, _GRID, _CDF)


def _frac(instant: datetime, lo: datetime, hi: datetime) -> float:
    return min(1.0, max(0.0, (instant - lo) / (hi - lo)))


def _stamp(lo: datetime, total_s: int, u: float) -> tuple[datetime, date]:
    ts = lo + timedelta(seconds=min(total_s - 1, int(u * total_s)))
    return ts, utc_to_et_date(ts)


def simulate_events(
    seed: int, sc: Scenario, customers: pl.DataFrame, days: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """customers needs: customer_id, join_day, risk, spend_rate, utilisation, and the
    `<campaign>_arm` / `<campaign>_true_lift` columns from overlay.assign_campaigns."""
    rows = customers.to_dicts()
    events: list[tuple] = []
    txns: list[tuple] = []

    for offset in range(days):
        d = sim_date(sc.start_date, offset)
        lo, hi = window_bounds_utc(d, 1)
        total_s = int((hi - lo).total_seconds())
        mult = day_multipliers(sc, d)
        wd = d.weekday()
        pay_mult = 1.0 if wd < 5 else WEEKEND_PAYMENT_MULT
        windows = []
        for c in sc.campaigns:
            a = float(_cdf(_frac(c.start_utc, lo, hi)))
            b = float(_cdf(_frac(c.end_utc, lo, hi)))
            if b > a:
                windows.append((c, a, b))

        for r in rows:
            if r["join_day"] > offset:
                continue
            cid = r["customer_id"]
            rng = entity_rng(seed, "behavior", cid, d)
            lam_open = open_rate(r["spend_rate"]) * OPEN_WEEKDAY[wd] * mult.app_open
            lam_buy = purchase_rate(r["spend_rate"]) * PURCHASE_WEEKDAY[wd] * mult.purchase
            n_open = rng.poisson(lam_open)
            n_buy = rng.poisson(lam_buy)
            u_open = _ppf(rng.random(n_open))
            u_buy = _ppf(rng.random(n_buy))
            amounts = rng.lognormal(np.log(35) + 0.4 * (r["utilisation"] - 0.5), 0.7, n_buy)
            p_repay = min(1.0, 0.04 * (1 - 0.6 * r["risk"]) * mult.repayment * pay_mult)
            repay = rng.random() < p_repay
            repay_amt = rng.lognormal(np.log(150) + 0.5 * (r["utilisation"] - 0.5), 0.6)
            u_repay = _ppf(rng.random())
            p_miss = min(1.0, (0.01 + 0.05 * r["risk"]) * mult.miss * pay_mult)
            miss = rng.random() < p_miss
            u_miss = _ppf(rng.random())

            extra_open: list[float] = []
            extra_buy: list[float] = []
            extra_amt: list[float] = []
            for c, a, b in windows:
                if r[f"{c.name}_arm"] != "test":
                    continue
                lift = r[f"{c.name}_true_lift"]
                if c.event == "purchase":
                    n = rng.poisson(lam_buy * lift * (b - a))
                    extra_buy += _ppf(a + (b - a) * rng.random(n)).tolist()
                    extra_amt += rng.lognormal(
                        np.log(35) + 0.4 * (r["utilisation"] - 0.5), 0.7, n
                    ).tolist()
                else:
                    n = rng.poisson(lam_open * lift * (b - a))
                    extra_open += _ppf(a + (b - a) * rng.random(n)).tolist()

            key = f"{cid}:{d:%Y%m%d}"
            k = 0
            for u in [*u_open, *extra_open]:
                ts, ed = _stamp(lo, total_s, u)
                events.append((f"{key}:{k}", cid, "app_open", ts, ed))
                k += 1
            if miss:
                ts, ed = _stamp(lo, total_s, u_miss)
                events.append((f"{key}:{k}", cid, "missed_payment", ts, ed))
                k += 1
            for u, amt in zip([*u_buy, *extra_buy], [*amounts, *extra_amt], strict=True):
                ts, ed = _stamp(lo, total_s, u)
                txns.append((f"{key}:{k}", cid, "purchase", float(amt), ts, ed))
                k += 1
            if repay:
                ts, ed = _stamp(lo, total_s, u_repay)
                txns.append((f"{key}:{k}", cid, "repayment", float(repay_amt), ts, ed))
                k += 1

    fact_events = pl.DataFrame(events, schema=EVENT_SCHEMA, orient="row")
    fact_txns = pl.DataFrame(txns, schema=TXN_SCHEMA, orient="row")
    return (
        fact_txns.sort("event_ts_utc", "transaction_id"),
        fact_events.sort("event_ts_utc", "event_id"),
    )


def label(df: pl.DataFrame, loaded_at: datetime) -> pl.DataFrame:
    return df.with_columns(
        pl.lit(loaded_at.astimezone(UTC)).cast(pl.Datetime("us", "UTC")).alias("loaded_at"),
        pl.lit(DATA_LABEL).alias("data_label"),
    )
