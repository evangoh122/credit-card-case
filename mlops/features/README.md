# Features (placeholder)

Feature definitions for the MLOps pipeline will live here as versioned modules or
SQL. Until then, reuse the observable features in `ccc.evidently_data.FEATURES`
and the gold snapshots built by `ccc.duck.build`.

Rules for anything added later:

- Polars only; no pandas conversions before the Evidently boundary.
- No simulation-truth columns (`ccc.gold.guard_no_truth_columns`).
- One numbered file per change; never rewrite an already-published definition.
