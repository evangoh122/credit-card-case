"""The source-extract path defaults to the bundled layout and honours CC_SOURCE_DIR."""
from __future__ import annotations

import importlib

from build import config as C


def test_source_dir_defaults_to_bundled_layout():
    assert C.SOURCE_DIR == C.ROOT / "company_z_data_download" / "company_z"


def test_source_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CC_SOURCE_DIR", str(tmp_path))
    try:
        reloaded = importlib.reload(C)
        assert reloaded.SOURCE_DIR == tmp_path
    finally:
        monkeypatch.delenv("CC_SOURCE_DIR", raising=False)
        importlib.reload(C)
