"""Upload an exported dataset directory to a Hugging Face dataset repo.

    HF_TOKEN=... python -m ccc.sim.upload --dir DIR --repo-id USER/NAME

The token is read only from the HF_TOKEN environment variable. The repo must already exist
unless --create is given. Needs the optional extra: pip install 'ccc[hf]'.
"""

import argparse
import os
from pathlib import Path

TRUTH_FILE = "sim_truth_customer.parquet"


def upload(folder: Path, repo_id: str, token: str, create: bool = False) -> None:
    if not (folder / "manifest.json").is_file():
        raise FileNotFoundError(f"{folder} has no manifest.json; run ccc.sim.export first")
    if (folder / TRUTH_FILE).exists():
        raise ValueError(f"{TRUTH_FILE} holds the planted truth and must not be published")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    if create:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_large_folder(repo_id=repo_id, repo_type="dataset", folder_path=folder)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m ccc.sim.upload")
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--repo-id", required=True, help="e.g. user/credit-card-case-synthetic")
    ap.add_argument("--create", action="store_true", help="create the dataset repo if missing")
    args = ap.parse_args(argv)
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set")
    upload(args.dir, args.repo_id, token, args.create)


if __name__ == "__main__":
    main()
