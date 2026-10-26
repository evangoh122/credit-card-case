import argparse
from datetime import datetime
from pathlib import Path

import polars as pl

from ccc.sim import events, overlay, profiles, scenario
from ccc.timeutil import window_bounds_utc

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


def simulate(
    seed: int,
    days: int,
    sc: scenario.Scenario,
    profile_source: str = "handwritten",
    locale: str = "en_US",
    loaded_at: datetime | None = None,
) -> dict[str, pl.DataFrame]:
    if profile_source not in ("handwritten", "copula"):
        raise ValueError("profile_source must be 'handwritten' or 'copula'")
    specs = cohort_specs(seed, sc, days)
    pairs = [(c, d) for c, d, _ in specs]
    if profile_source == "handwritten":
        prof = profiles.handwritten_table(seed, pairs, sc.cohort_shift)
    else:
        seed_pairs = [(f"S-{i:05d}", 0) for i in range(SEED_TABLE_ROWS)]
        seed_table = profiles.handwritten_table(seed, seed_pairs, NO_SHIFT)
        prof = profiles.copula_table(profiles.fit_copula(seed_table), seed, pairs, sc.cohort_shift)
    ident = profiles.identities(seed, [(c, j) for c, _, j in specs], sc.start_date, locale)
    customers = ident.join(prof, on="customer_id", how="inner", maintain_order="left")
    customers = customers.with_columns(
        pl.Series("cohort_day", [d for _, d, _ in specs], dtype=pl.Int64)
    )
    customers = overlay.assign_campaigns(seed, sc, customers)

    txns, evts = events.simulate_events(seed, sc, customers, days)
    if loaded_at is None:
        loaded_at = window_bounds_utc(sc.start_date, days)[1]

    campaign_cols = [col for c in sc.campaigns for col in (f"{c.name}_arm", f"{c.name}_true_lift")]
    dim = customers.select("customer_id", "name", "join_date", "channel", "region")
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
