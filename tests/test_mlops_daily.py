import json
import multiprocessing as mp
import os
import socket
import time
from datetime import date, datetime
from pathlib import Path

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


def test_run_daily_publishes_a_complete_marker(tmp_path):
    targets = tmp_path / "targets"
    result = daily.run_daily(
        date(2026, 11, 3), 7, target_dir=targets, tracking_dir=tmp_path / "mlruns"
    )
    marker = json.loads((targets / "2026-11-03" / "run.json").read_text())
    assert marker == {"run_date": "2026-11-03", "derived_seed": result["derived_seed"]}
    assert [p.name for p in targets.iterdir()] == ["2026-11-03"]


def test_interrupted_marker_write_does_not_reserve_the_date(tmp_path, monkeypatch):
    targets = tmp_path / "targets"
    tracking = tmp_path / "mlruns"
    real_write_text = Path.write_text
    attempts = {"count": 0}

    def fail_first_marker_write(self, *args, **kwargs):
        if self.name == "run.json" and attempts["count"] == 0:
            attempts["count"] += 1
            raise OSError("disk full")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_first_marker_write)
    with pytest.raises(OSError, match="disk full"):
        daily.run_daily(date(2026, 11, 3), 7, target_dir=targets, tracking_dir=tracking)

    assert not (targets / "2026-11-03").exists()
    assert list(targets.iterdir()) == []

    retried = daily.run_daily(date(2026, 11, 3), 7, target_dir=targets, tracking_dir=tracking)
    assert (targets / "2026-11-03" / "run.json").is_file()
    assert retried["target"] == str(targets / "2026-11-03")


def test_run_daily_rejects_datetime_before_creating_the_target(tmp_path):
    targets = tmp_path / "targets"
    with pytest.raises(TypeError, match="datetime"):
        daily.run_daily(
            datetime(2026, 11, 3, 12, 30),
            7,
            target_dir=targets,
            tracking_dir=tmp_path / "mlruns",
        )
    assert not targets.exists()


def test_empty_target_created_after_check_is_refused(tmp_path, monkeypatch):
    target = tmp_path / "2026-11-03"
    real_check = daily.ensure_append_only

    def create_empty_target_after_check(path):
        result = real_check(path)
        Path(path).mkdir()
        return result

    monkeypatch.setattr(daily, "ensure_append_only", create_empty_target_after_check)
    with pytest.raises(daily.AppendOnlyError, match="already exists"):
        daily._publish_marker(target, {"derived_seed": 7})
    assert list(target.iterdir()) == []
    assert [p.name for p in tmp_path.iterdir()] == ["2026-11-03"]


def _publisher_after_synchronized_check(target, barrier, queue, seed):
    original = daily.ensure_append_only

    def synchronized(path):
        result = original(path)
        barrier.wait(timeout=20)
        return result

    daily.ensure_append_only = synchronized
    try:
        daily._publish_marker(Path(target), {"derived_seed": seed})
        queue.put(("ok", seed))
    except daily.AppendOnlyError:
        queue.put(("exists", seed))
    except BaseException as exc:  # pragma: no cover - surfaced through the queue
        queue.put((type(exc).__name__, seed))


def test_concurrent_publishers_produce_exactly_one_complete_winner(tmp_path):
    ctx = mp.get_context("fork")
    barrier = ctx.Barrier(2)
    queue = ctx.Queue()
    target = tmp_path / "2026-11-03"
    workers = [
        ctx.Process(
            target=_publisher_after_synchronized_check,
            args=(str(target), barrier, queue, seed),
        )
        for seed in (1, 2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(20)
        if worker.is_alive():
            worker.kill()
            worker.join()
        assert worker.exitcode == 0
    results = [queue.get(timeout=5) for _ in workers]
    assert sorted(status for status, _ in results) == ["exists", "ok"]
    winner = next(seed for status, seed in results if status == "ok")
    assert json.loads((target / "run.json").read_text()) == {"derived_seed": winner}
    assert [p.name for p in tmp_path.iterdir()] == ["2026-11-03"]


def _crash_before_publication(target):
    daily._rename_noreplace = lambda *args: os._exit(77)
    daily._publish_marker(Path(target), {"derived_seed": 1})


def test_crash_before_publication_does_not_reserve_the_date(tmp_path):
    target = tmp_path / "2026-11-03"
    worker = mp.get_context("fork").Process(target=_crash_before_publication, args=(str(target),))
    worker.start()
    worker.join(20)
    if worker.is_alive():
        worker.kill()
        worker.join()
    assert worker.exitcode == 77
    assert not target.exists()
    daily._publish_marker(target, {"derived_seed": 2})
    assert json.loads((target / "run.json").read_text()) == {"derived_seed": 2}
    with pytest.raises(daily.AppendOnlyError, match="already exists"):
        daily._publish_marker(target, {"derived_seed": 3})
    assert json.loads((target / "run.json").read_text()) == {"derived_seed": 2}


def test_portable_fallback_refuses_an_existing_empty_target(tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "_rename_noreplace_at2", lambda *args: False)
    target = tmp_path / "2026-11-03"
    target.mkdir()
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "run.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(daily.AppendOnlyError, match="already exists"):
        daily._rename_noreplace(staging, target)
    assert list(target.iterdir()) == []


def test_portable_fallback_reclaims_a_stale_empty_reservation(tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "_rename_noreplace_at2", lambda *args: False)
    target = tmp_path / "2026-11-03"
    target.mkdir()
    stale = time.time() - daily._INCOMPLETE_RESERVATION_SECONDS - 1
    os.utime(target, (stale, stale))
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "run.json").write_text("{}\n", encoding="utf-8")
    daily._rename_noreplace(staging, target)
    assert (target / "run.json").read_text(encoding="utf-8") == "{}\n"
