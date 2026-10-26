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


def test_serve_terminates_children_on_ctrl_c():
    procs = [mock.Mock(), mock.Mock()]
    for p in procs:
        p.poll.return_value = None
        p.returncode = 0
    procs[0].wait.side_effect = [KeyboardInterrupt, None]
    with mock.patch.object(subprocess, "Popen", side_effect=procs):
        assert ui.serve([["a"], ["b"]]) == 0
    for p in procs:
        p.terminate.assert_called_once()
