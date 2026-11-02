# ccc

Credit card case: customer segmentation, randomized card-offer experiments and drift
monitoring, tested on a seeded 60-day simulation. See `docs/design.md`.

## How this was built

- **Data:** all data is synthetic. It is produced with the help of AI and relevant libraries
  (Faker, SDV and the seeded simulator in `ccc.sim`). No real customer data is used.
  Additional synthetic customer data will be brought in as part of CI/CD.
- **Front end:** built by AI.
- **Analysis and MLOps:** done by a human.

Python 3.12, managed with [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
```

- `ccc.rng`: per-customer, per-stream generators derived from a stable hash of the root seed.
- `ccc.timeutil`: UTC and per-city IANA zones, calendar-day windows, snapshot cutoffs.
- `ccc.geo`: US city to IANA time-zone mapping.
- `ccc.sim`: seeded simulator (profiles, events, scenario overlay). Run
  `uv run python -m ccc.sim --seed 1 --days 60 --scenario scenarios/default.toml --out data/sim`.
  `uv run python -m ccc.sim.export --seed 1 --target-transactions 15000000 --out data/export`
  writes chunked, zstd, date-partitioned Parquet via DuckDB (`manifest.json` has counts and
  versions; `sim_truth_customer` only with `--include-truth`). `ccc.sim.upload` pushes it to a
  Hugging Face dataset (`HF_TOKEN` env var, `pip install '.[hf]'`);
  `notebooks/colab_generate_15m.ipynb` does both on Colab.
  Tests that fit SDV are marked `slow` (`uv run pytest -m slow`).
- Build connected DuckDB files with `uv run python -m ccc.duck.build --parquet-dir DIR --out-dir DIR`.
  After `from ccc.duck import connect`, `with connect(DIR) as con:` attaches `dim`, `facts`, and `gold`, so
  `con.sql("SELECT * FROM facts.fact_transactions JOIN dim.dim_customer USING (customer_id)")`
  works across files; DuckDB CLI/UI users can run `current/connect.sql` after changing to the
  warehouse output directory. That script pins every `ATTACH` to the immutable `build-*` directory it
  was generated for, so moving `current` during the run cannot mix generations.
- Every build publishes an immutable `build-*` directory and atomically moves the `current` symlink;
  normal builds never delete old builds, because an existing reader may still have one open. To reclaim
  space, stop every consumer that may use candidate builds and run the explicit maintenance command
  `uv run python -m ccc.duck.build --out-dir DIR --prune-builds --keep-last 2 --acknowledge-readers-stopped`.
  Pruning refuses without that acknowledgement and preserves the `current` target, `.stage-*` directories,
  symlinks, malformed `build-*` names and unrelated files. It cannot detect open file handles itself.
- `ccc.tracking`, `python -m ccc.run_study`: logs a seeded study to MLflow (parent run per study,
  child per seed) and checks planted campaign effects are recovered across seeds. Run
  `uv run python -m ccc.run_study --scenario scenarios/default.toml --seeds 1 2 3 --n-initial 300`;
  the store is `./mlruns` (gitignored); `--sdv-scores` needs `--profiles copula`.

## CI and secret scanning

- `ci.yml` runs `uv sync --locked`, tests and ruff; `dependency-audit.yml` exports the
  locked runtime dependencies with `uv export` (plus a Windows-targeted view so
  Windows-only pins such as `waitress` are covered), audits the exported pins directly
  with pip-audit, and fails only on vulnerabilities that have a fix available. Scheduled
  runs fail the workflow as well, so failures trigger notifications.
- `secrets.yml` scans pull requests from the base branch's copy of the workflow
  (`pull_request_target`), so a pull request cannot replace the scan with a no-op.
  The pull request head is only fetched as git data, never executed, and
  `.gitleaks.toml` / `.gitleaksignore` always come from the base ref: ignore rules and
  workflow edits apply after merge. The job keeps `contents: read`, no secrets, and a
  pinned gitleaks binary; do not add checkout-and-run steps to it. It needs the
  repository's Actions event policy to allow `pull_request_target` (GitHub's default
  policy for public repositories blocks that event, enforced from 2026-12-02).
