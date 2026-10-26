import argparse
from datetime import datetime
from pathlib import Path

import polars as pl

from ccc.geo import ZONES
from ccc.sim import events, overlay, profiles, scenario
from ccc.timeutil import local_day_bounds_utc, sim_date

SEED_TABLE_ROWS = 2000
NO_SHIFT = scenario.CohortShift(0.0, 0.0)


def cohort_specs(seed: int, sc: scenario.Scenario, days: int) -> list[tuple[str, int, int]]:
    """(customer_id, cohort_day, join_day); IDs depend only on cohort day and index."""
    specs = [
        (cid, 0, profiles.initial_join_day(seed, cid))
        for cid in (f"I-{i:06d}" for i in range(sc.initial_customers))
    ]
    for d, n in enumerate(events.daily_arrivals(seed, sc, days)):
        specs += [(f"D{d:03d}-{i:05d}", d, d) for i in range(n)]
    return specs


def profile_model(seed: int, profile_source: str):
    """Fitted copula for profile_source='copula' (shared across chunks), else None."""
    if profile_source not in ("handwritten", "copula"):
        raise ValueError("profile_source must be 'handwritten' or 'copula'")
    if profile_source == "handwritten":
        return None
    seed_pairs = [(f"S-{i:05d}", 0) for i in range(SEED_TABLE_ROWS)]
    seed_table = profiles.handwritten_table(seed, seed_pairs, NO_SHIFT)
    return profiles.fit_copula(seed_table)


def default_loaded_at(sc: scenario.Scenario, days: int) -> datetime:
    last_local_date = sim_date(sc.start_date, days)
    return max(local_day_bounds_utc(last_local_date, tz)[0] for tz in ZONES)


def simulate_chunk(
    seed: int,
    days: int,
    sc: scenario.Scenario,
    specs: list[tuple[str, int, int]],
    loaded_at: datetime,
    synth=None,
    locale: str = "en_US",
) -> dict[str, pl.DataFrame]:
    """Simulate one batch of `cohort_specs` rows. Every output row depends only on
    (seed, its own customer), so any partition of the specs yields the same union of rows."""
    pairs = [(c, d) for c, d, _ in specs]
    if synth is None:
        prof = profiles.handwritten_table(seed, pairs, sc.cohort_shift)
    else:
        prof = profiles.copula_table(synth, seed, pairs, sc.cohort_shift)
    ident = profiles.identities(seed, [(c, j) for c, _, j in specs], sc.start_date, locale)
    customers = ident.join(prof, on="customer_id", how="inner", maintain_order="left")
    customers = customers.with_columns(
        pl.Series("cohort_day", [d for _, d, _ in specs], dtype=pl.Int64)
    )
    customers = overlay.assign_campaigns(seed, sc, customers)

    txns, evts = events.simulate_events(seed, sc, customers, days)

    campaign_cols = [col for c in sc.campaigns for col in (f"{c.name}_arm", f"{c.name}_true_lift")]
    dim = customers.select(
        "customer_id", "name", "city", "tz", "joined_ts_utc", "channel", "region"
    )
    truth = customers.select(
        "customer_id",
        "cohort_day",
        "risk",
        "spend_rate",
        "responsiveness",
        "utilisation",
        *campaign_cols,
    )
    return {
        "dim_customer": events.label(dim, loaded_at),
        "sim_truth_customer": events.label(truth, loaded_at),
        "fact_transactions": events.label(txns, loaded_at),
        "fact_events": events.label(evts, loaded_at),
    }


def simulate(
    seed: int,
    days: int,
    sc: scenario.Scenario,
    profile_source: str = "handwritten",
    locale: str = "en_US",
    loaded_at: datetime | None = None,
    batch_size: int | None = None,
) -> dict[str, pl.DataFrame]:
    """batch_size only bounds peak memory; the result is identical for every batch size."""
    if loaded_at is not None and loaded_at.tzinfo is None:
        raise ValueError("loaded_at must be timezone-aware")
    synth = profile_model(seed, profile_source)
    specs = cohort_specs(seed, sc, days)
    if loaded_at is None:
        loaded_at = default_loaded_at(sc, days)
    step = batch_size or max(1, len(specs))
    batches = [specs[i : i + step] for i in range(0, len(specs), step)] or [[]]
    chunks = [simulate_chunk(seed, days, sc, b, loaded_at, synth) for b in batches]
    if len(chunks) == 1:
        return chunks[0]
    out = {name: pl.concat([c[name] for c in chunks]) for name in chunks[0]}
    out["fact_transactions"] = out["fact_transactions"].sort("event_ts_utc", "transaction_id")
    out["fact_events"] = out["fact_events"].sort("event_ts_utc", "event_id")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m ccc.sim")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profiles", choices=("handwritten", "copula"), default="handwritten")
    args = ap.parse_args(argv)
    sc = scenario.load(args.scenario)
    tables = simulate(args.seed, args.days, sc, profile_source=args.profiles)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.write_parquet(out / f"{name}.parquet")
        print(f"{name}: {df.height} rows")
