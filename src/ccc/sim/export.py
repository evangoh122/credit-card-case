"""Chunked export of the simulation to partitioned Parquet via DuckDB.

    python -m ccc.sim.export --target-transactions 15000000 --out DIR --seed 1

Customer batches are simulated in worker processes and staged as Parquet, so memory is
bounded by the batch size. DuckDB then writes the final zstd Parquet, with the fact tables
partitioned by event_date_utc. sim_truth_customer holds the planted effects and is only
written with --include-truth.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from importlib import metadata
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from ccc.sim import events, run, scenario

DEFAULT_SCENARIO = Path(__file__).resolve().parents[3] / "scenarios" / "default.toml"
FACTS = {
    "fact_transactions": ("event_ts_utc", "transaction_id"),
    "fact_events": ("event_ts_utc", "event_id"),
}
DIMS = {"dim_customer": "customer_id", "sim_truth_customer": "customer_id"}
PILOT_CUSTOMERS = 2000


def _write_batch(job: tuple) -> dict[str, int]:
    idx, seed, days, sc, specs, loaded_at, staging, tables = job
    out = run.simulate_chunk(seed, days, sc, specs, loaded_at)
    counts = {}
    for name in tables:
        out[name].write_parquet(staging / name / f"batch-{idx:06d}.parquet", compression="zstd")
        counts[name] = out[name].height
    return counts


def estimate_initial_customers(seed: int, sc: scenario.Scenario, days: int, target: int) -> int:
    """Initial-customer count expected to produce about `target` transactions.

    Measures transactions per customer-day on the first PILOT_CUSTOMERS initial customers,
    then adds the exposure of the (fixed) daily arrival cohorts.
    """
    pilot_sc = replace(sc, initial_customers=PILOT_CUSTOMERS, arrivals_per_day=0.0)
    specs = run.cohort_specs(seed, pilot_sc, days)
    loaded_at = run.default_loaded_at(pilot_sc, days)
    txns = run.simulate_chunk(seed, days, pilot_sc, specs, loaded_at)["fact_transactions"].height
    rate = txns / (PILOT_CUSTOMERS * days)
    arrival_days = sum(n * (days - d) for d, n in enumerate(events.daily_arrivals(seed, sc, days)))
    n = math.ceil((target / rate - arrival_days) / days)
    if n < 1:
        raise ValueError(
            f"arrival cohorts alone give about {rate * arrival_days:.0f} transactions; "
            "lower --arrivals-per-day or raise the target"
        )
    return n


def _version_info() -> dict:
    root = Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = None
    return {
        "code_version": metadata.version("ccc") if _installed() else None,
        "git_commit": sha,
        "python": sys.version.split()[0],
        "polars": pl.__version__,
        "duckdb": duckdb.__version__,
        "numpy": np.__version__,
    }


def _installed() -> bool:
    try:
        metadata.version("ccc")
    except metadata.PackageNotFoundError:
        return False
    return True


def export(
    out: Path,
    seed: int,
    days: int,
    sc: scenario.Scenario,
    target_transactions: int | None = None,
    batch_size: int = 500,
    workers: int | None = None,
    include_truth: bool = False,
    memory_limit: str = "4GB",
) -> dict:
    """Write the dataset under `out` and return the manifest (also saved as manifest.json).

    `out` must be absent or empty. On any failure everything this run wrote is removed.
    """
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    created = not out.exists()
    try:
        return _export(
            out,
            seed,
            days,
            sc,
            target_transactions,
            batch_size,
            workers,
            include_truth,
            memory_limit,
        )
    except BaseException:
        if created:
            shutil.rmtree(out, ignore_errors=True)
        else:  # out was empty on entry, so every child was written by this run
            for child in out.iterdir():
                shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink()
        raise


def _export(
    out: Path,
    seed: int,
    days: int,
    sc: scenario.Scenario,
    target_transactions: int | None,
    batch_size: int,
    workers: int | None,
    include_truth: bool,
    memory_limit: str,
) -> dict:
    t0 = time.time()
    if target_transactions is not None:
        sc = replace(
            sc, initial_customers=estimate_initial_customers(seed, sc, days, target_transactions)
        )
    specs = run.cohort_specs(seed, sc, days)
    loaded_at = run.default_loaded_at(sc, days)
    tables = [*FACTS, "dim_customer", *(["sim_truth_customer"] if include_truth else [])]

    staging = out / ".staging"
    con = None
    try:
        for name in tables:
            (staging / name).mkdir(parents=True)
        batches = [specs[i : i + batch_size] for i in range(0, len(specs), batch_size)]
        jobs = [(i, seed, days, sc, b, loaded_at, staging, tables) for i, b in enumerate(batches)]
        workers = workers or os.cpu_count() or 1
        staged = dict.fromkeys(tables, 0)
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
            for counts in pool.map(_write_batch, jobs):
                for name, n in counts.items():
                    staged[name] += n

        con = duckdb.connect()
        con.execute(f"SET memory_limit = '{memory_limit}'")
        con.execute(f"SET temp_directory = '{staging / 'duckdb_tmp'}'")
        con.execute("SET TimeZone = 'UTC'")
        con.execute("SET threads = 1")  # one writer per partition: one file per day
        for name in tables:
            src = f"read_parquet('{staging / name}/*.parquet')"
            keys = ", ".join(FACTS.get(name) or (DIMS[name],))
            if name in FACTS:
                con.execute(
                    f"COPY (SELECT * FROM {src} ORDER BY {keys}) TO '{out / name}' "
                    "(FORMAT parquet, COMPRESSION zstd, PARTITION_BY (event_date_utc), "
                    "WRITE_PARTITION_COLUMNS true)"
                )
            else:
                con.execute(
                    f"COPY (SELECT * FROM {src} ORDER BY {keys}) TO '{out / name}.parquet' "
                    "(FORMAT parquet, COMPRESSION zstd)"
                )
        shutil.rmtree(staging)

        counts = {}
        for name in tables:
            glob = f"{out / name}/*/*.parquet" if name in FACTS else f"{out / name}.parquet"
            counts[name] = con.execute(f"SELECT count(*) FROM read_parquet('{glob}')").fetchone()[0]
        if counts != staged:
            raise RuntimeError(f"written row counts {counts} differ from generated {staged}")
        dates = con.execute(
            f"SELECT min(event_date_utc), max(event_date_utc) "
            f"FROM read_parquet('{out / 'fact_transactions'}/*/*.parquet')"
        ).fetchone()
    finally:
        if con is not None:
            con.close()
        shutil.rmtree(staging, ignore_errors=True)
    manifest = {
        "data_label": events.DATA_LABEL,
        "timezone": "UTC",
        "seed": seed,
        "days": days,
        "scenario": sc.name,
        "start_date": sc.start_date.isoformat(),
        "initial_customers": sc.initial_customers,
        "customers": counts["dim_customer"],
        "target_transactions": target_transactions,
        "batch_size": batch_size,
        "loaded_at": loaded_at.isoformat(),
        "include_truth": include_truth,
        "row_counts": counts,
        "event_date_utc_range": [d.isoformat() for d in dates],
        "partition_by": {name: "event_date_utc" for name in FACTS},
        "compression": "zstd",
        "versions": _version_info(),
        "generation_seconds": round(time.time() - t0, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m ccc.sim.export")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--target-transactions", type=int, required=True)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--scenario", default=str(DEFAULT_SCENARIO))
    ap.add_argument("--batch-size", type=int, default=500, help="customers per batch")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--arrivals-per-day", type=float, help="override the scenario value")
    ap.add_argument("--memory-limit", default="4GB", help="DuckDB memory limit for the final write")
    ap.add_argument(
        "--include-truth",
        action="store_true",
        help="also write sim_truth_customer (planted effects); never publish it",
    )
    args = ap.parse_args(argv)
    sc = scenario.load(args.scenario)
    if args.arrivals_per_day is not None:
        sc = replace(sc, arrivals_per_day=args.arrivals_per_day)
    manifest = export(
        args.out,
        args.seed,
        args.days,
        sc,
        args.target_transactions,
        args.batch_size,
        args.workers,
        args.include_truth,
        args.memory_limit,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
