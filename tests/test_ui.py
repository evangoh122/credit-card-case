import signal
import socket
import subprocess
import sys
from unittest import mock

import pytest

from ccc import tracking, ui


def test_mlflow_command_uses_study_store(tmp_path):
    cmd = ui.mlflow_command(tmp_path, 5000)
    assert cmd[:4] == [sys.executable, "-m", "mlflow", "ui"]
    assert cmd[cmd.index("--backend-store-uri") + 1] == tracking.default_tracking_uri(tmp_path)
    assert cmd[cmd.index("--port") + 1] == "5000"


def test_mlflow_command_explicit_uri_wins(tmp_path):
    cmd = ui.mlflow_command(tmp_path, 5001, "sqlite:///x.db")
    assert cmd[cmd.index("--backend-store-uri") + 1] == "sqlite:///x.db"
    assert cmd[cmd.index("--port") + 1] == "5001"


def test_evidently_command(tmp_path):
    cmd = ui.evidently_command(tmp_path / "ws", 8000)
    assert cmd[cmd.index("-c") + 2] == "ui"
    assert cmd[cmd.index("--workspace") + 1] == str(tmp_path / "ws")
    assert cmd[cmd.index("--port") + 1] == "8000"


def test_defaults():
    assert (ui.MLFLOW_PORT, ui.EVIDENTLY_PORT) == (5000, 8000)


def test_port_in_use_is_clear_error():
    with socket.socket() as busy:
        busy.bind((ui.HOST, 0))
        port = busy.getsockname()[1]
        with pytest.raises(SystemExit, match=f"port {port}.*already in use"):
            ui.check_port_free(port)


def test_main_builds_both_commands_and_prints_urls(tmp_path, capsys):
    with (
        mock.patch.object(ui, "check_port_free"),
        mock.patch.object(ui, "serve", return_value=0) as serve,
    ):
        rc = ui.main(["--tracking-dir", str(tmp_path), "--workspace", str(tmp_path / "ws")])
    assert rc == 0
    mlflow_cmd, evidently_cmd = serve.call_args.args[0]
    assert mlflow_cmd[mlflow_cmd.index("--port") + 1] == "5000"
    assert evidently_cmd[evidently_cmd.index("--port") + 1] == "8000"
    out = capsys.readouterr().out
    assert "http://127.0.0.1:5000" in out and "http://127.0.0.1:8000" in out
    assert (tmp_path / "ws").is_dir()


def test_main_rejects_same_port():
    with pytest.raises(SystemExit):
        ui.main(["--port", "7000", "--evidently-port", "7000"])


def _fake_proc(pid):
    p = mock.Mock(pid=pid, returncode=0)
    p.poll.return_value = None
    return p


def test_serve_stops_every_child_even_if_one_already_exited():
    procs = [_fake_proc(101), _fake_proc(102)]
    procs[0].wait.side_effect = [KeyboardInterrupt, None, None]
    sent = []

    def killpg(pid, sig):
        sent.append((pid, sig))
        if pid == 101:
            raise ProcessLookupError

    with (
        mock.patch.object(subprocess, "Popen", side_effect=procs) as popen,
        mock.patch.object(ui.os, "killpg", side_effect=killpg),
    ):
        assert ui.serve([["a"], ["b"]]) == 0
    assert all(c.kwargs["start_new_session"] for c in popen.call_args_list)
    assert (102, signal.SIGTERM) in sent and (102, signal.SIGKILL) in sent
    procs[1].wait.assert_called()


def test_serve_cleans_up_first_child_when_second_spawn_fails():
    first = _fake_proc(201)
    with (
        mock.patch.object(subprocess, "Popen", side_effect=[first, OSError("boom")]),
        mock.patch.object(ui.os, "killpg") as killpg,
        pytest.raises(OSError),
    ):
        ui.serve([["a"], ["b"]])
    killpg.assert_any_call(201, signal.SIGTERM)
    first.wait.assert_called()


def test_stop_kills_group_when_child_ignores_sigterm():
    p = _fake_proc(301)
    p.wait.side_effect = [subprocess.TimeoutExpired("x", 1), None]
    with mock.patch.object(ui.os, "killpg") as killpg:
        ui._stop([p])
    assert [c.args for c in killpg.call_args_list] == [
        (301, signal.SIGTERM),
        (301, signal.SIGKILL),
    ]
