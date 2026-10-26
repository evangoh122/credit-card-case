import argparse
import dataclasses
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl

from ccc import gold
from ccc.sim import run, scenario
from ccc.timeutil import local_date_expr, sim_date

NUMERIC_FEATURES = (
    "recency",
    "frequency",
    "monetary",
    "days_since_join",
    "app_open_count",
    "repayment_count",
)
CATEGORICAL_FEATURES = ("channel", "region", "tz")
FEATURES = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
WINDOWS = {"recency": 30, "frequency": 30, "monetary": 30}
PROJECT_NAME = "ccc-drift"


def drift_frames(
    tables: dict[str, pl.DataFrame], start_date: date, days: int, cohort_day: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Observable features at the end of the run. Reference is customers who joined before the
    simulation start; current is customers who joined from cohort_day on."""
    if cohort_day < 1 or days <= cohort_day:
        raise ValueError("cohort_day must be in [1, days)")
    dim = tables["dim_customer"]
    snapshot = sim_date(start_date, days - 1)
    rfm = gold.rfm_snapshot(dim, tables["fact_transactions"], snapshot, WINDOWS)
    engagement = gold.engagement_features(tables["fact_events"], dim, snapshot)
    df = (
        dim.select(
            "customer_id",
            *CATEGORICAL_FEATURES,
            local_date_expr("joined_ts_utc", "tz").alias("join_date"),
        )
        .join(rfm.drop("snapshot_date"), on="customer_id", how="inner")
        .join(engagement.drop("snapshot_date"), on="customer_id", how="left")
        .with_columns(pl.col("app_open_count", "repayment_count").fill_null(0))
    )
    reference = df.filter(pl.col("join_date") < start_date)
    current = df.filter(pl.col("join_date") >= sim_date(start_date, cohort_day))
    if not reference.height or not current.height:
        raise ValueError(f"need customers who joined before start and from cohort day {cohort_day}")
    out = []
    for frame in (reference, current):
        frame = frame.select(FEATURES)
        gold.guard_no_truth_columns(frame)
        out.append(frame.to_pandas())
    return out[0], out[1]


def write_drift_snapshot(
    reference: pd.DataFrame, current: pd.DataFrame, workspace: str | Path
) -> str:
    from evidently import Report
    from evidently.presets import DataDriftPreset
    from evidently.ui.workspace import Workspace

    ws = Workspace.create(str(workspace))
    found = ws.search_project(PROJECT_NAME)
    project = found[0] if found else ws.create_project(PROJECT_NAME)
    snapshot = Report([DataDriftPreset()]).run(current, reference)
    ws.add_run(project.id, snapshot)
    return str(project.id)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ccc.evidently_data")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--n-initial", type=int)
    ap.add_argument("--workspace", default="evidently_workspace")
    args = ap.parse_args(argv)
    sc = scenario.load(args.scenario)
    if args.n_initial is not None:
        sc = dataclasses.replace(sc, initial_customers=args.n_initial)
    tables = run.simulate(args.seed, args.days, sc)
    reference, current = drift_frames(tables, sc.start_date, args.days, args.days // 2)
    project_id = write_drift_snapshot(reference, current, args.workspace)
    print(f"snapshot written to {args.workspace} (project {project_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
