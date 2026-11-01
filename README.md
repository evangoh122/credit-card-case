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
  Tests that fit SDV are marked `slow` (`uv run pytest -m slow`).
- `ccc.tracking`, `python -m ccc.run_study`: logs a seeded study to MLflow (parent run per study,
  child per seed) and checks planted campaign effects are recovered across seeds. Run
  `uv run python -m ccc.run_study --scenario scenarios/default.toml --seeds 1 2 3 --n-initial 300`;
  the store is `./mlruns` (gitignored); `--sdv-scores` needs `--profiles copula`.

## CI and secret scanning

- `ci.yml` runs `uv sync --locked`, tests and ruff; `dependency-audit.yml` exports the
  locked runtime dependencies with `uv export` and fails only on vulnerabilities that
  have a fix available.
- `secrets.yml` scans pull requests from the base branch's copy of the workflow
  (`pull_request_target`), so a pull request cannot replace the scan with a no-op.
  The pull request head is only fetched as git data, never executed, and
  `.gitleaks.toml` / `.gitleaksignore` always come from the base ref: ignore rules and
  workflow edits apply after merge. The job keeps `contents: read`, no secrets, and a
  pinned gitleaks binary; do not add checkout-and-run steps to it. It needs the
  repository's Actions event policy to allow `pull_request_target` (GitHub's default
  policy for public repositories blocks that event, enforced from 2026-12-02).
