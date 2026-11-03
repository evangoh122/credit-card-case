"""Daily MLOps run stub: deterministic seed, append-only target, local MLflow store."""

import argparse
import ctypes
import ctypes.util
import errno
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import mlflow

from ccc.mlops.seed import derive_seed, validate_run_date

EXPERIMENT = "ccc-mlops-daily"
DEFAULT_TARGET_DIR = Path("data/mlops_daily")
DEFAULT_TRACKING_DIR = Path("mlruns/mlops")
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_EXISTING_ERRNOS = frozenset({errno.EEXIST, errno.ENOTEMPTY, errno.EISDIR, errno.ENOTDIR})
_INCOMPLETE_RESERVATION_SECONDS = 3600


class AppendOnlyError(FileExistsError):
    """Raised when an append-only daily target already exists."""


def ensure_append_only(target: str | Path) -> Path:
    """Return ``target`` after refusing to reuse an existing artefact."""
    path = Path(target)
    if path.exists() or path.is_symlink():
        raise AppendOnlyError(f"append-only target already exists: {path}")
    return path


def _experiment_id(experiment: str, tracking_dir: Path) -> str:
    found = mlflow.get_experiment_by_name(experiment)
    if found is not None:
        return found.experiment_id
    return mlflow.create_experiment(
        experiment, artifact_location=(tracking_dir / "artifacts").as_uri()
    )


def _rename_noreplace_at2(source: Path, destination: Path) -> bool:
    """Try ``renameat2(RENAME_NOREPLACE)``; return ``False`` when unavailable."""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        renameat2 = libc.renameat2
    except (OSError, AttributeError):
        return False
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            _AT_FDCWD,
            os.fsencode(source),
            _AT_FDCWD,
            os.fsencode(destination),
            _RENAME_NOREPLACE,
        )
        == 0
    ):
        return True
    code = ctypes.get_errno()
    if code in (errno.ENOSYS, errno.EINVAL):
        return False
    if code in _EXISTING_ERRNOS:
        raise AppendOnlyError(f"append-only target already exists: {destination}") from OSError(
            code, os.strerror(code), str(destination)
        )
    raise OSError(code, os.strerror(code), str(destination))


def _is_abandoned_reservation(destination: Path) -> bool:
    """Return whether ``destination`` is a stale, empty reservation from a crash."""
    if destination.is_symlink() or not destination.is_dir():
        return False
    try:
        empty = not any(destination.iterdir())
    except OSError:
        return False
    return empty and time.time() - destination.stat().st_mtime >= _INCOMPLETE_RESERVATION_SECONDS


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically move ``source`` to ``destination``, refusing any existing path.

    On Linux this is ``renameat2`` with ``RENAME_NOREPLACE``, so an empty
    directory created by another process after a check can never be replaced.
    The portable fallback reserves ``destination`` with ``mkdir`` first, so any
    existing file, symlink or directory is still refused, and reclaims only an
    abandoned empty reservation that is old enough to belong to a crash.
    """
    if _rename_noreplace_at2(source, destination):
        return
    try:
        destination.mkdir()
    except FileExistsError as exc:
        if not _is_abandoned_reservation(destination):
            raise AppendOnlyError(f"append-only target already exists: {destination}") from exc
        shutil.rmtree(destination, ignore_errors=True)
        try:
            destination.mkdir()
        except FileExistsError as retry_exc:
            raise AppendOnlyError(
                f"append-only target already exists: {destination}"
            ) from retry_exc
    try:
        os.rename(source, destination)
    except OSError:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _publish_marker(target: Path, marker: dict[str, str | int]) -> None:
    """Publish ``run.json`` inside ``target`` atomically and without clobbering.

    The marker is written to a complete staging directory first, then the whole
    directory is moved into place with an atomic no-replace rename. An existing
    target, including a concurrently created empty directory, is refused, while
    a crash before publication leaves no target at all, so a retry succeeds.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    ensure_append_only(target)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        (staging / "run.json").write_text(
            json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
        )
        _rename_noreplace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run_daily(
    run_date: date,
    seed: int,
    *,
    target_dir: str | Path = DEFAULT_TARGET_DIR,
    tracking_dir: str | Path = DEFAULT_TRACKING_DIR,
    experiment: str = EXPERIMENT,
) -> dict[str, str | int]:
    """Log one daily run to a single local-file MLflow experiment.

    The run date is validated and the target directory is checked before
    anything is written; an existing run is never overwritten, so the pipeline
    stays append-only. The ``run.json`` marker is prepared before the target is
    published, so an interrupted attempt never reserves the date.
    """
    run_date = validate_run_date(run_date)
    target = ensure_append_only(Path(target_dir) / run_date.isoformat())
    derived_seed = derive_seed(run_date, seed)
    tracking_root = Path(tracking_dir).resolve()
    tracking_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_root.as_uri())
    experiment_id = _experiment_id(experiment, tracking_root)
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=f"daily-{run_date.isoformat()}"
    ) as run:
        mlflow.log_params(
            {
                "run_date": run_date.isoformat(),
                "root_seed": seed,
                "derived_seed": derived_seed,
                "append_only": "true",
                "tracking_store": "local-file",
            }
        )
        mlflow.set_tag("data_label", "synthetic")
        _publish_marker(target, {"run_date": run_date.isoformat(), "derived_seed": derived_seed})
        return {
            "run_id": run.info.run_id,
            "experiment_id": experiment_id,
            "derived_seed": derived_seed,
            "target": str(target),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ccc.mlops.daily")
    parser.add_argument("--run-date", required=True, type=date.fromisoformat)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--target-dir", default=str(DEFAULT_TARGET_DIR))
    parser.add_argument("--tracking-dir", default=str(DEFAULT_TRACKING_DIR))
    args = parser.parse_args(argv)
    result = run_daily(
        args.run_date,
        args.seed,
        target_dir=args.target_dir,
        tracking_dir=args.tracking_dir,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
