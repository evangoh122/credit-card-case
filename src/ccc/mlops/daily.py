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


class AppendOnlyError(FileExistsError):
    """Raised when an append-only daily target already exists."""


class UnsupportedPlatformError(RuntimeError):
    """Raised when atomic publication via ``renameat2`` is unavailable."""


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


def _probe_renameat2(renameat2) -> None:
    """Raise unless the kernel accepts ``RENAME_NOREPLACE``.

    Passing null paths yields ``EFAULT`` when the syscall and flag are
    supported and ``ENOSYS``/``EINVAL`` when they are not, so the probe has no
    filesystem side effects. It runs before any target or staging directory is
    created.
    """
    if renameat2(_AT_FDCWD, None, _AT_FDCWD, None, _RENAME_NOREPLACE) == 0:
        raise UnsupportedPlatformError("renameat2(RENAME_NOREPLACE) probe unexpectedly succeeded")
    code = ctypes.get_errno()
    if code in (errno.ENOSYS, errno.EINVAL):
        raise UnsupportedPlatformError(
            "renameat2(RENAME_NOREPLACE) is unavailable "
            f"({errno.errorcode.get(code, code)}); daily runs require Linux renameat2"
        )


def _load_renameat2():
    """Return ``libc.renameat2`` or raise :class:`UnsupportedPlatformError`.

    Availability is checked before anything is written, so an unsupported
    platform never leaves a target or staging directory behind.
    """
    if not sys.platform.startswith("linux"):
        raise UnsupportedPlatformError(
            "daily publication requires Linux renameat2(RENAME_NOREPLACE); "
            f"this platform is {sys.platform!r}"
        )
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        renameat2 = libc.renameat2
    except (OSError, AttributeError) as exc:
        raise UnsupportedPlatformError(
            "libc does not expose renameat2; daily publication requires Linux renameat2"
        ) from exc
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    _probe_renameat2(renameat2)
    return renameat2


def _rename_noreplace(source: Path, destination: Path, renameat2=None) -> None:
    """Atomically move ``source`` to ``destination``, refusing any existing path.

    This is only ``renameat2`` with ``RENAME_NOREPLACE``: an existing file,
    symlink or directory (including one created concurrently after a check) can
    never be replaced. There is no fallback publication path.
    """
    if renameat2 is None:
        renameat2 = _load_renameat2()
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
        return
    code = ctypes.get_errno()
    if code in (errno.ENOSYS, errno.EINVAL):
        raise UnsupportedPlatformError(
            "renameat2(RENAME_NOREPLACE) is unavailable "
            f"({errno.errorcode.get(code, code)}); daily runs require Linux renameat2"
        ) from OSError(code, os.strerror(code), str(destination))
    if code in _EXISTING_ERRNOS:
        raise AppendOnlyError(f"append-only target already exists: {destination}") from OSError(
            code, os.strerror(code), str(destination)
        )
    raise OSError(code, os.strerror(code), str(destination))


def _publish_marker(target: Path, marker: dict[str, str | int]) -> None:
    """Publish ``run.json`` inside ``target`` atomically and without clobbering.

    The marker is written to a unique staging directory first, then the whole
    directory is moved into place with an atomic no-replace rename. An existing
    target, including a concurrently created empty directory, is refused. A
    crash before publication leaves only an orphan staging directory, never the
    target, so a retry of the date always succeeds.
    """
    renameat2 = _load_renameat2()
    target.parent.mkdir(parents=True, exist_ok=True)
    ensure_append_only(target)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        (staging / "run.json").write_text(
            json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
        )
        _rename_noreplace(staging, target, renameat2)
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
    stays append-only. Atomic publication requires Linux ``renameat2``; an
    unsupported platform raises before any target or tracking directory is
    created. The ``run.json`` marker is prepared before the target is published,
    so an interrupted attempt never reserves the date.
    """
    run_date = validate_run_date(run_date)
    _load_renameat2()
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
