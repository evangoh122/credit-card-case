import ctypes
import errno
import json
import multiprocessing as mp
import os
import socket
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


def _public_publisher(target_dir, tracking_dir, seed, barrier, queue):
    try:
        barrier.wait(timeout=20)
        result = daily.run_daily(
            date(2026, 11, 3), seed, target_dir=target_dir, tracking_dir=tracking_dir
        )
        queue.put(("ok", seed, result["derived_seed"]))
    except daily.AppendOnlyError:
        queue.put(("exists", seed, None))
    except BaseException as exc:  # pragma: no cover - surfaced through the queue
        queue.put((type(exc).__name__, seed, str(exc)))


def test_concurrent_public_publishers_produce_exactly_one_winner(tmp_path):
    ctx = mp.get_context("fork")
    barrier = ctx.Barrier(2)
    queue = ctx.Queue()
    targets = tmp_path / "targets"
    workers = [
        ctx.Process(
            target=_public_publisher,
            args=(str(targets), str(tmp_path / f"mlruns-{seed}"), seed, barrier, queue),
        )
        for seed in (1, 2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(30)
        if worker.is_alive():
            worker.kill()
            worker.join()
        assert worker.exitcode == 0
    results = [queue.get(timeout=10) for _ in workers]
    assert sorted(status for status, _, _ in results) == ["exists", "ok"]
    winner = next((seed, derived) for status, seed, derived in results if status == "ok")
    marker = json.loads((targets / "2026-11-03" / "run.json").read_text())
    assert marker == {"run_date": "2026-11-03", "derived_seed": winner[1]}
    assert [p.name for p in targets.iterdir()] == ["2026-11-03"]


def _crash_in_publication(target_dir, tracking_dir):
    from ccc.mlops import daily as child_daily

    if hasattr(child_daily, "_load_renameat2"):
        child_daily._rename_noreplace = lambda *args, **kwargs: os._exit(77)
    else:
        child_daily._rename_noreplace_at2 = lambda *args, **kwargs: False
        os.rename = lambda *args, **kwargs: os._exit(77)
    child_daily.run_daily(date(2026, 11, 3), 7, target_dir=target_dir, tracking_dir=tracking_dir)


def test_crash_between_staging_and_rename_leaves_the_date_retryable(tmp_path):
    targets = tmp_path / "targets"
    tracking = tmp_path / "mlruns"
    worker = mp.get_context("fork").Process(
        target=_crash_in_publication, args=(str(targets), str(tracking))
    )
    worker.start()
    worker.join(30)
    if worker.is_alive():
        worker.kill()
        worker.join()
    assert worker.exitcode == 77
    target = targets / "2026-11-03"
    assert not target.exists()
    result = daily.run_daily(date(2026, 11, 3), 7, target_dir=targets, tracking_dir=tracking)
    assert result["target"] == str(target)
    assert json.loads((target / "run.json").read_text()) == {
        "run_date": "2026-11-03",
        "derived_seed": result["derived_seed"],
    }


class _FakeRenameat2:
    argtypes = None
    restype = None

    def __init__(self, code):
        self.code = code

    def __call__(self, *args):
        ctypes.set_errno(self.code)
        return -1


class _FakeLibc:
    def __init__(self, code):
        self.renameat2 = _FakeRenameat2(code)


def test_missing_renameat2_raises_before_creating_anything(tmp_path, monkeypatch):
    def missing_libc(*args, **kwargs):
        raise OSError("libc unavailable")

    monkeypatch.setattr(daily.ctypes, "CDLL", missing_libc)
    targets = tmp_path / "targets"
    tracking = tmp_path / "mlruns"
    with pytest.raises(RuntimeError, match="renameat2") as excinfo:
        daily.run_daily(date(2026, 11, 3), 7, target_dir=targets, tracking_dir=tracking)
    assert type(excinfo.value).__name__ == "UnsupportedPlatformError"
    assert not targets.exists()
    assert not tracking.exists()


@pytest.mark.parametrize("code", [errno.ENOSYS, errno.EINVAL])
def test_unavailable_renameat2_raises_before_creating_anything(tmp_path, monkeypatch, code):
    monkeypatch.setattr(daily.ctypes, "CDLL", lambda *args, **kwargs: _FakeLibc(code))
    targets = tmp_path / "targets"
    tracking = tmp_path / "mlruns"
    with pytest.raises(RuntimeError, match="renameat2") as excinfo:
        daily.run_daily(date(2026, 11, 3), 7, target_dir=targets, tracking_dir=tracking)
    assert type(excinfo.value).__name__ == "UnsupportedPlatformError"
    assert not targets.exists()
    assert not tracking.exists()
