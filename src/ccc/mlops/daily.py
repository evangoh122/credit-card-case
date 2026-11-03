"""Daily MLOps run stub: deterministic seed, append-only target, local MLflow store."""

import argparse
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


def _publish_marker(target: Path, marker: dict[str, str | int]) -> None:
    """Publish ``run.json`` inside ``target`` atomically and without clobbering.

    The marker is written to a staging directory first, so a crash before
    publication never reserves the daily target. Publication is a single rename,
    and an existing target is refused.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        (staging / "run.json").write_text(
            json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
        )
        ensure_append_only(target)
        try:
            os.rename(staging, target)
        except OSError as exc:
            if target.exists() or target.is_symlink():
                raise AppendOnlyError(f"append-only target already exists: {target}") from exc
            raise
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
