import ast
from pathlib import Path

import polars as pl
import pytest

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
