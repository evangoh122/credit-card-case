import argparse
import contextlib
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

from ccc import tracking

HOST = "127.0.0.1"
MLFLOW_PORT = 5000
EVIDENTLY_PORT = 8000
EVIDENTLY_CLI = "from evidently.cli import app; app()"
STOP_TIMEOUT_S = 10


def check_port_free(port: int, host: str = HOST) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError as exc:
            raise SystemExit(f"port {port} on {host} is already in use; pick another") from exc


def mlflow_command(
    tracking_dir: str | Path, port: int, tracking_uri: str | None = None
) -> list[str]:
    uri = tracking_uri or tracking.default_tracking_uri(tracking_dir)
    return [
        sys.executable, "-m", "mlflow", "ui",
        "--backend-store-uri", uri, "--host", HOST, "--port", str(port),
    ]  # fmt: skip


def evidently_command(workspace: str | Path, port: int) -> list[str]:
    return [
        sys.executable, "-c", EVIDENTLY_CLI, "ui",
        "--workspace", str(workspace), "--host", HOST, "--port", str(port),
    ]  # fmt: skip


IS_WINDOWS = sys.platform == "win32"


def _popen_kwargs() -> dict:
    if IS_WINDOWS:
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)}
    return {"start_new_session": True}


def _taskkill_tree(p: subprocess.Popen) -> None:
    """Force-kill the child's whole process tree; fall back to the direct child on failure."""
    try:
        result = subprocess.run(
            ["taskkill", "/PID", str(p.pid), "/F", "/T"], check=False, capture_output=True
        )
        failed, detail = result.returncode != 0, f"exit code {result.returncode}"
    except OSError as exc:  # taskkill missing
        failed, detail = True, f"unavailable ({exc})"
    if failed and p.poll() is None:  # nonzero while the child is already gone is fine
        print(
            f"taskkill failed for child pid {p.pid}: {detail}; killing it directly", file=sys.stderr
        )
        with contextlib.suppress(OSError):
            p.kill()


def _signal_group(p: subprocess.Popen, force: bool) -> None:
    """SIGTERM/SIGKILL the child's process group (Unix only)."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(p.pid, signal.SIGKILL if force else signal.SIGTERM)


def _wait_or_report(p: subprocess.Popen) -> None:
    """Bounded final wait; a child that outlives it is reported, never raised or waited on."""
    try:
        p.wait(timeout=STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(f"failed to stop child pid {p.pid}; it may still be running", file=sys.stderr)


def _stop(procs: list[subprocess.Popen]) -> list[int | None]:
    """Stop every child. Windows: force the whole tree at once (a graceful taskkill does nothing
    for console processes). Unix: SIGTERM the group, wait, then SIGKILL what is left.

    Returns each child's return code as observed immediately before we first signalled it, so
    a child that died on its own keeps its real code and one we killed reads None.
    """
    seen: list[int | None] = []
    for p in procs:
        seen.append(p.poll())  # before any signal: our own kills must not look like crashes
        _taskkill_tree(p) if IS_WINDOWS else _signal_group(p, force=False)
    if IS_WINDOWS:
        for p in procs:
            _wait_or_report(p)
        return seen
    for p in procs:
        with contextlib.suppress(subprocess.TimeoutExpired):
            p.wait(timeout=STOP_TIMEOUT_S)
        _signal_group(p, force=True)
        _wait_or_report(p)
    return seen


def serve(commands: list[list[str]]) -> int:
    procs: list[subprocess.Popen] = []
    try:
        for cmd in commands:
            procs.append(subprocess.Popen(cmd, **_popen_kwargs()))
        while all(p.poll() is None for p in procs):
            with contextlib.suppress(subprocess.TimeoutExpired):
                procs[0].wait(timeout=1)
    except KeyboardInterrupt:
        pass
    finally:
        codes = _stop(procs)  # forced-stop codes are never counted
    return next((c for c in codes if c not in (0, None, -15)), 0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ccc.ui")
    ap.add_argument("--tracking-dir", default="mlruns")
    ap.add_argument("--tracking-uri")
    ap.add_argument("--mlflow-port", "--port", type=int, default=MLFLOW_PORT)
    ap.add_argument("--evidently-port", type=int, default=EVIDENTLY_PORT)
    ap.add_argument("--workspace", default="evidently_workspace")
    args = ap.parse_args(argv)
    if args.mlflow_port == args.evidently_port:
        ap.error("MLflow and Evidently ports must differ")
    for port in (args.mlflow_port, args.evidently_port):
        check_port_free(port)
    Path(args.workspace).mkdir(parents=True, exist_ok=True)
    print(f"MLflow:    http://{HOST}:{args.mlflow_port}")
    print(f"Evidently: http://{HOST}:{args.evidently_port}  (Ctrl-C stops both)")
    return serve(
        [
            mlflow_command(args.tracking_dir, args.mlflow_port, args.tracking_uri),
            evidently_command(args.workspace, args.evidently_port),
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
