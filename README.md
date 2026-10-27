# ccc

Credit card case: customer segmentation, randomized card-offer experiments and drift
monitoring, tested on a seeded 60-day simulation. See `docs/design.md`.

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
- Build connected DuckDB files with `python -m ccc.duck.build --parquet-dir DIR --out-dir DIR`.
  After `from ccc.duck import connect`, `with connect(DIR) as con:` attaches `dim`, `facts`, and `gold`, so
  `con.sql("SELECT * FROM facts.fact_transactions JOIN dim.dim_customer USING (customer_id)")`
  works across files; DuckDB CLI/UI users can run the generated `connect.sql`.
- `ccc.tracking`, `python -m ccc.run_study`: logs a seeded study to MLflow (parent run per study,
  child per seed) and checks planted campaign effects are recovered across seeds. Run
  `uv run python -m ccc.run_study --scenario scenarios/default.toml --seeds 1 2 3 --n-initial 300`;
  the store is `./mlruns` (gitignored); `--sdv-scores` needs `--profiles copula`.
