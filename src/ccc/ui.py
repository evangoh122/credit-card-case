import argparse
import contextlib
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


def _stop(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try:
            p.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def serve(commands: list[list[str]]) -> int:
    procs: list[subprocess.Popen] = []
    try:
        for cmd in commands:
            procs.append(subprocess.Popen(cmd))
        while all(p.poll() is None for p in procs):
            with contextlib.suppress(subprocess.TimeoutExpired):
                procs[0].wait(timeout=1)
    except KeyboardInterrupt:
        pass
    finally:
        _stop(procs)
    return next((p.returncode for p in procs if p.returncode not in (0, None, -15)), 0)


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
