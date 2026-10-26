import argparse
import dataclasses
import sys
from pathlib import Path

import pandas as pd
import polars as pl

from ccc import tracking
from ccc.sim import run, scenario

FEATURES = (*tracking.NUMERIC_FEATURES, *tracking.CATEGORICAL_FEATURES)
PROJECT_NAME = "ccc-drift"


def drift_frames(
    tables: dict[str, pl.DataFrame], cohort_day: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reference is the initial cohort (day 0); current is cohorts arriving from cohort_day on."""
    if cohort_day < 1:
        raise ValueError("cohort_day must be at least 1")
    df = tracking.feature_table(tables).select("cohort_day", *FEATURES)
    reference = df.filter(pl.col("cohort_day") == 0).drop("cohort_day")
    current = df.filter(pl.col("cohort_day") >= cohort_day).drop("cohort_day")
    if not reference.height or not current.height:
        raise ValueError(f"need customers in cohort 0 and at cohort day >= {cohort_day}")
    return reference.to_pandas(), current.to_pandas()


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
    reference, current = drift_frames(tables, args.days // 2)
    project_id = write_drift_snapshot(reference, current, args.workspace)
    print(f"snapshot written to {args.workspace} (project {project_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
