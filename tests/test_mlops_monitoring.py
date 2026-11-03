import ast
from pathlib import Path

import polars as pl
import pytest
from evidently.ui.workspace import Workspace

from ccc.mlops import monitoring


def test_prepare_drift_frames_keeps_polars_and_validates():
    reference = pl.DataFrame({"x": [1.0, 2.0], "channel": ["web", "app"]})
    current = pl.DataFrame({"x": [1.5, 2.5], "channel": ["web", "web"]})
    got_reference, got_current = monitoring.prepare_drift_frames(reference, current)
    assert isinstance(got_reference, pl.DataFrame)
    assert isinstance(got_current, pl.DataFrame)
    with pytest.raises(TypeError, match="Polars"):
        monitoring.prepare_drift_frames([1.0, 2.0], current)
    with pytest.raises(ValueError, match="share columns"):
        monitoring.prepare_drift_frames(pl.DataFrame({"y": [1.0]}), current)
    with pytest.raises(ValueError, match="must not be empty"):
        monitoring.prepare_drift_frames(reference, pl.DataFrame({"x": [], "channel": []}))


def test_monitoring_module_never_imports_pandas():
    tree = ast.parse(Path(monitoring.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "pandas" not in imported


def test_prepare_drift_frames_rejects_truth_columns_in_either_frame():
    clean = pl.DataFrame({"x": [1.0, 2.0]})
    leaked = pl.DataFrame({"x": [1.0, 2.0], "sim_truth_risk": [0.1, 0.2]})
    with pytest.raises(ValueError, match="sim_truth_risk"):
        monitoring.prepare_drift_frames(leaked, clean)
    with pytest.raises(ValueError, match="sim_truth_risk"):
        monitoring.prepare_drift_frames(clean, leaked)


def test_write_drift_snapshot_returns_the_project_id(tmp_path):
    reference = pl.DataFrame({"x": [1.0, 2.0], "channel": ["web", "app"]})
    current = pl.DataFrame({"x": [1.5, 2.5], "channel": ["web", "web"]})
    project_id = monitoring.write_drift_snapshot(reference, current, tmp_path / "ws")
    ws = Workspace.create(str(tmp_path / "ws"))
    found = ws.search_project(monitoring.PROJECT_NAME)
    assert found[0].id is not None
    assert project_id == str(found[0].id)
