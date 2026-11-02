# Pipelines (runbook only)

The daily entrypoint is the installed module `ccc.mlops.daily`:

```bash
uv run python -m ccc.mlops.daily --run-date 2026-11-03 --seed 7
```

No scheduled workflow is committed yet. A future workflow should:

1. Trigger on a weekday cron in UTC and pass a single `--run-date`.
2. Pass `--seed` from repository configuration and `--target-dir` /
   `--tracking-dir` under CI scratch space.
3. Fail (not retry-overwrite) when the target date already exists.
4. Use only the local file MLflow store; never contact a remote MLflow server,
   Hugging Face or a scheduled warehouse job.
