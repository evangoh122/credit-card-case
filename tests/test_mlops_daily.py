import socket
from datetime import date

import mlflow
import pytest
from mlflow import MlflowClient

from ccc.mlops import daily


@pytest.fixture(autouse=True)
def local_mlflow(monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    previous = mlflow.get_tracking_uri()

    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    yield
    mlflow.set_tracking_uri(previous)


def test_append_only_guard_refuses_existing_target(tmp_path):
    target = tmp_path / "2026-11-03"
    target.mkdir()
    with pytest.raises(daily.AppendOnlyError, match="already exists"):
        daily.ensure_append_only(target)


def test_append_only_guard_accepts_a_new_path(tmp_path):
    target = tmp_path / "2026-11-03"
    assert daily.ensure_append_only(target) == target


def test_run_daily_logs_params_to_one_local_file_experiment(tmp_path):
    tracking = tmp_path / "mlruns"
    targets = tmp_path / "targets"
    first = daily.run_daily(date(2026, 11, 3), 11, target_dir=targets, tracking_dir=tracking)
    second = daily.run_daily(date(2026, 11, 4), 11, target_dir=targets, tracking_dir=tracking)

    assert first["experiment_id"] == second["experiment_id"]
    assert mlflow.get_tracking_uri() == tracking.resolve().as_uri()
    assert (tracking / first["experiment_id"] / "meta.yaml").is_file()
    client = MlflowClient(tracking.as_uri())
    matching = [e for e in client.search_experiments() if e.name == daily.EXPERIMENT]
    assert len(matching) == 1
    runs = client.search_runs([first["experiment_id"]])
    assert len(runs) == 2
    params = {run.data.params["run_date"]: run.data.params for run in runs}
    assert params["2026-11-03"]["root_seed"] == "11"
    assert params["2026-11-03"]["derived_seed"] == str(first["derived_seed"])
    assert params["2026-11-04"]["derived_seed"] == str(second["derived_seed"])
    assert params["2026-11-03"]["derived_seed"] != params["2026-11-04"]["derived_seed"]
    assert params["2026-11-03"]["tracking_store"] == "local-file"


def test_run_daily_refuses_to_overwrite_an_existing_date(tmp_path):
    tracking = tmp_path / "mlruns"
    targets = tmp_path / "targets"
    first = daily.run_daily(date(2026, 11, 3), 1, target_dir=targets, tracking_dir=tracking)
    with pytest.raises(daily.AppendOnlyError, match="already exists"):
        daily.run_daily(date(2026, 11, 3), 2, target_dir=targets, tracking_dir=tracking)
    client = MlflowClient(tracking.as_uri())
    assert len(client.search_runs([first["experiment_id"]])) == 1
