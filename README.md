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
  Tests that fit SDV are marked `slow` (`uv run pytest -m slow`).
- `ccc.tracking`, `python -m ccc.run_study`: logs a seeded study to MLflow (parent run per study,
  child per seed) and checks planted campaign effects are recovered across seeds. Run
  `uv run python -m ccc.run_study --scenario scenarios/default.toml --seeds 1 2 3 --n-initial 300`;
  the store is `./mlruns` (gitignored); `--sdv-scores` needs `--profiles copula`.
