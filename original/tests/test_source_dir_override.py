"""The source-extract path defaults to the bundled layout and honours CC_SOURCE_DIR."""
from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from build import config as C

DEFAULT_SOURCE_DIR = C.ROOT / "company_z_data_download" / "company_z"


@contextmanager
def _restore_source_dir_environment():
    """Restore CC_SOURCE_DIR on exit, then reload config to match it.

    The shared ``build.config`` module caches SOURCE_DIR at import time, so the
    environment must be put back *before* reloading. Otherwise a caller that
    started with CC_SOURCE_DIR set would get its environment restored but leave
    C.SOURCE_DIR pointing at the default, leaking inconsistent state.
    """
    original = os.environ.get("CC_SOURCE_DIR")
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("CC_SOURCE_DIR", None)
        else:
            os.environ["CC_SOURCE_DIR"] = original
        importlib.reload(C)


@pytest.fixture(autouse=True)
def _isolated_source_dir():
    with _restore_source_dir_environment():
        yield


def test_source_dir_defaults_to_bundled_layout(monkeypatch):
    monkeypatch.delenv("CC_SOURCE_DIR", raising=False)
    assert importlib.reload(C).SOURCE_DIR == DEFAULT_SOURCE_DIR


def test_source_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CC_SOURCE_DIR", str(tmp_path))
    assert importlib.reload(C).SOURCE_DIR == tmp_path


def test_config_matches_environment_on_entry():
    expected = os.environ.get("CC_SOURCE_DIR")
    if expected is None:
        assert C.SOURCE_DIR == DEFAULT_SOURCE_DIR
    else:
        assert C.SOURCE_DIR == Path(expected)


def test_teardown_restores_preset_source_dir(tmp_path, monkeypatch):
    """Regression: an environment that was already set must be restored intact.

    This exercises the same restore-then-reload path the autouse fixture uses,
    with CC_SOURCE_DIR preset before the context is entered.
    """
    preset = tmp_path / "preset-source"
    other = tmp_path / "other-source"
    monkeypatch.setenv("CC_SOURCE_DIR", str(preset))
    importlib.reload(C)
    assert C.SOURCE_DIR == preset

    with _restore_source_dir_environment():
        monkeypatch.setenv("CC_SOURCE_DIR", str(other))
        assert importlib.reload(C).SOURCE_DIR == other

    assert os.environ.get("CC_SOURCE_DIR") == str(preset)
    assert C.SOURCE_DIR == preset
