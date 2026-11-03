# MLOps skeleton

Minimal, versioned MLOps scaffolding for the case. It contains no data, no heavy
model logic, no network access and no remote writes. Python code lives in the
installed package (`src/ccc/mlops/`), but run every command below from the
repository root (or use `uv run --directory /path/to/repo python -m ...`), since
the SQL and warehouse paths are relative; SQL, configs and runbooks stay under
`mlops/`.

## Layout

| Path | Purpose |
| --- | --- |
| `mlops/sql/` | Versioned `.sql` checks over the DuckDB warehouse (`dim`, `facts`, `gold`). |
| `mlops/features/` | Placeholder for feature definitions (see `README.md`). |
| `mlops/training/` | Placeholder for training jobs (see `README.md`). |
| `mlops/monitoring/` | Placeholder for drift runbooks; helper is `ccc.mlops.monitoring`. |
| `mlops/configs/` | Scenario and model TOML placeholders. |
| `mlops/pipelines/` | Daily pipeline runbook; no scheduled workflow is committed yet. |

## Run SQL on DuckDB

Build a warehouse first (see the root `README.md`), then:

```bash
uv run python -m ccc.mlops.sql_runner --db data/warehouse --sql mlops/sql/001_row_counts.sql
```

`--db` accepts a warehouse directory (opened through `ccc.duck.connect`, so
`dim`, `facts` and `gold` attach by alias) or a plain DuckDB database file.
The final result comes back as a Polars `DataFrame`; no pandas conversion
happens in the runner.

Add a new check by adding a numbered file such as `mlops/sql/003_new_check.sql`;
never rewrite an existing numbered file after it has run.

## Daily run stub

```bash
uv run python -m ccc.mlops.daily \
  --run-date 2026-11-03 --seed 7 \
  --target-dir data/mlops_daily \
  --tracking-dir mlruns/mlops
```

- Derives a deterministic per-date seed from `(run_date, seed)`.
- Refuses to continue when the target path already exists (append-only); it
  never overwrites a previous run.
- Logs parameters to one local-file MLflow experiment (`ccc-mlops-daily`).
  The tracking URI is a `file://` store; nothing is sent to a server.
- Does no ingestion, training or data writes beyond a small `run.json` marker.

## Drift helper

`ccc.mlops.monitoring.prepare_drift_frames` validates reference/current frames and
keeps them Polars. `ccc.mlops.monitoring.write_drift_snapshot` converts to pandas
only at the Evidently `Report.run` boundary, matching `ccc.evidently_data`.

## Scheduled workflow (stub only)

No GitHub Actions workflow is committed yet. The intended shape is a weekday cron
that runs the daily module once per UTC date, passing `--target-dir` and
`--tracking-dir` under CI scratch space. It must not push to a remote MLflow
server, call Hugging Face, or write to a shared warehouse; see
`mlops/pipelines/README.md`.
