# Training (placeholder)

Training jobs are deliberately out of scope for this skeleton. When added, they
will:

- read features only from the warehouse or gold snapshots,
- take a seed from `ccc.mlops.seed.derive_seed(run_date, root_seed)`,
- log a single run to the local-file MLflow experiment, and
- write model artefacts to a new append-only path instead of overwriting one.
