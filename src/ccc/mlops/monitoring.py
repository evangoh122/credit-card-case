"""Evidently drift helpers for the MLOps skeleton.

The conventions match :mod:`ccc.evidently_data`: drift frames stay Polars until the
Evidently ``Report.run`` boundary, where they are converted to pandas.
"""

from pathlib import Path

import polars as pl

PROJECT_NAME = "ccc-mlops-drift"


def prepare_drift_frames(
    reference: pl.DataFrame, current: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Validate Polars drift inputs without converting them yet."""
    for name, frame in (("reference", reference), ("current", current)):
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(f"{name} drift frame must be a Polars DataFrame")
    if reference.columns != current.columns:
        raise ValueError("reference and current drift frames must share columns")
    if reference.is_empty() or current.is_empty():
        raise ValueError("reference and current drift frames must not be empty")
    return reference, current


def write_drift_snapshot(
    reference: pl.DataFrame, current: pl.DataFrame, workspace: str | Path
) -> str:
    """Write one Evidently DataDriftPreset run to a local workspace and return its id."""
    from evidently import Report
    from evidently.presets import DataDriftPreset
    from evidently.ui.workspace import Workspace

    reference, current = prepare_drift_frames(reference, current)
    ws = Workspace.create(str(workspace))
    found = ws.search_project(PROJECT_NAME)
    project = found[0] if found else ws.create_project(PROJECT_NAME)
    snapshot = Report([DataDriftPreset()]).run(current.to_pandas(), reference.to_pandas())
    ws.add_run(project.id, snapshot)
    return str(project.id)
