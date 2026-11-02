# Monitoring (placeholder)

Drift helpers live in `src/ccc/mlops/monitoring.py`:

- `prepare_drift_frames(reference, current)` validates two Polars frames and keeps
  them Polars; it rejects anything else and mismatched columns.
- `write_drift_snapshot(reference, current, workspace)` writes one Evidently
  `DataDriftPreset` run to a local workspace. The pandas conversion happens only
  inside `Report.run`, matching `ccc.evidently_data`.

This folder holds only runbooks and future check definitions. Do not add
scheduled jobs or remote writers here.
