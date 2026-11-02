# v7 fix1 (`fix/v7-polars-memory`): consolidated evidence, rounds 1-8

This file replaces the per-round evidence files (`v7_polars_memory_*`, `v7_round5..8_*`), which were folded in and removed; their text is in git history up to commit c5bd837.

**Roles.** Coder: `claude_code` (Claude Sonnet 5.5, `claude-sonnet-5-5`). Reviewers: `codex` (GPT) in every round, and Kimi `kimi-k3` from round 5. Verdicts below are as relayed in the task prompts for each round; this author did not independently re-run the reviewers.
**Scope.** The v7 build OOM'd in postprocessing (the 470 MB `fact_transaction.parquet` was loaded whole into pandas). The branch moves the large-table paths to Polars and, through review, closed several label/as-of leakage gaps in the ML extracts. `facts.py` simulation/RNG behaviour was never changed; the full 55-minute build was never run. All test runs: `.venv/bin/python -m pytest tests -q -p no:cacheprovider -m "not spark and not lakebase and not databricks"`, in the foreground.

| Round | Commit | Tests passed | Reviewer verdict on that commit |
|---|---|---|---|
| 1 | 59dc54f | 30 | codex CHANGES_REQUESTED (3 blockers + whitespace) |
| 2 | 0d04acd | 37 | codex CHANGES_REQUESTED |
| 3 | ce457a5 | 37 | codex CHANGES_REQUESTED (1 blocker: fixture drift) |
| 4 | e4795f4 | 38 | codex: blocker closed, everything else verified OK; a separate reviewer then flagged the payment as-of gap |
| 5 | 3f8c0a1 | 50 | codex and Kimi CHANGES_REQUESTED |
| 6 | b0b1600 | 60 | codex and Kimi CHANGES_REQUESTED |
| 7 | 834f0b8 | 68 | Kimi APPROVED; codex CHANGES_REQUESTED (1 blocker) |
| 8 | c5bd837 | 77 | approved by both reviewers (per the cleanup request) |

## Round 1 - Polars postprocess (59dc54f)
- **Done:** `build/postprocess.py` rewritten with Polars (lazy scan, streaming sink to a temp file, atomic replace); semantics identical to the old pandas logic (auth_code nulled unless APPROVED; decline fields nulled when APPROVED; `is_recurring` from `dim_merchant`, missing -> False; `network` from `dim_account`; `transaction_date` datetime; sort by date, ts; cancel ACTIVE cards of CLOSED/CHARGED_OFF accounts). Shared `utils.parquet_metadata`; snappy + dataset/generator/seed/period metadata kept. `ml_extracts` transaction read and `validation.source_percentiles` moved to Polars; `run_all` frees dimension/fact/raw frames (`del` + `gc.collect()`) before postprocess; `polars>=2.0,<3` pinned (2.0.0 installed).
- **Tests:** `tests/test_postprocess_polars.py` (8 tests) compares against the pandas logic embedded as an oracle (counts, dtypes, sums, ordering, end-to-end metadata/compression).
- **Mutation proofs:** skipping the auth_code nulling, reversing the sort, and defaulting a missing merchant to True each fail 2 tests; unmutated 8 passed.
- **Reviewer findings:** `refresh_docs` still loaded every table, including `fact_transaction`, into pandas; churn extract carried `customer_status`/`churn_date` (label-defining); credit-default extract carried `account_status`/`closed_date`; trailing whitespace in `build/config.py`.

## Round 2 - lazy refresh/validation and label leakage (0d04acd)
- **Done:** `refresh_docs` keeps `fact_transaction` as a lazy Polars scan; `validation.tx_metrics` computes every transaction-derived check lazily (anti-joins, group-bys); the data dictionary profiles lazily; inventory row counts come from parquet metadata. Churn and credit-default features no longer contain the label-defining columns; new `check_no_label_columns` gate fails the build if they return.
- **Tests:** `tests/test_v7_polars_fixes.py` (7 tests): label columns, the gate, `tx_metrics` vs pandas, a spy test that `refresh_docs.load_all` never calls the pandas reader on `fact_transaction`, dictionary lazy == pandas.
- **Mutation proofs (5, each fails the matching test):** re-add churn columns; re-add credit-default columns; neuter the gate (2 tests); pandas-load `fact_transaction` in `refresh_docs`; break the FK metric. Unmutated 7 passed.
- **Real-data check (on-disk `credit_card_data_v7`, measurement only):** `tx_metrics` over 12,656,688 rows in 12 s; 12 sampled metrics matched the pandas values.
- **Reviewer findings:** Polars `ne_missing` treats null/null as equal while pandas `!=` counts it as a mismatch (`bad_geo` 270 vs 197); mixed CRLF/LF line endings in `config.py`, `dictionary.py`, `refresh_docs.py`; tests should compare all metrics exactly.

## Round 3 - pandas null semantics, CRLF (ce457a5)
- **Done:** mismatch counts follow pandas `!=` (a null on either side is a mismatch) for `bad_mcc/cat/geo/intl`; the three files normalised to CRLF (diff vs master unchanged under `--ignore-cr-at-eol`); the test compares `tx_metrics` to the old pandas expressions with `==`.
- **Mutation proof:** switching back to `ne_missing` fails the metrics test.
- **Reviewer finding:** the test had edited the committed fixture (`M2.geo_id` G1 -> None, `bad_geo` 330), hiding the 270 baseline.

## Round 4 - fixture baseline restored (e4795f4)
- **Done:** original fixture restored (`bad_geo == 270` asserted); null-bearing cases added as appended rows and a merchant M3 in a separate test; exact comparison of all metrics kept.
- **Tests:** 38 passed. **Mutation proof:** `ne_missing` fails both metric tests; unmutated 8 passed in the file.
- **Reviewer:** blocker closed, everything else verified OK.

## Round 5 - credit-default payment features as-of (3f8c0a1)
- **Finding (CONFIRMED):** `fact_statement.payment_status` is settled with all later card cash (`facts._settled_closing_amounts`, `_stmt_status`), so `status_at_obs` and `paid_full_share_3m` used payments dated after the observation date; `max_source_date` did not track payment dates. Not leaky: `payments_amount`, `closing_balance`, `minimum_due`, `utilization_pct`, delinquency days-past-due (all as-of the statement cut).
- **Done:** `asof_payment_status` recomputes status from payments dated <= observation date; `payment_feature_date` feeds `max_source_date`; new `check_payment_sources_asof` gate.
- **Tests:** `tests/test_v7_asof_payments.py` (12 tests): confirmation demo, invariance to post-observation payments (5 variants), sensitivity, gate, churn audit. **Mutations (5):** drop `<= obs` filter 9 failed; read final status again 7; payment dates out of `max_source_date` 1; neuter gate 1; churn reads final status 1.
- **Reviewers:** codex and Kimi CHANGES_REQUESTED (status compared with `billed_amount` not `closing_balance`; fraud velocity not as-of; forecast uses full-period spend).

## Round 6 - mirror the simulator, as-of velocity and forecast (b0b1600)
- **Done:** (1) status now replays card payments <= obs through the simulator's own `_waterfall`, `_settled_closing_amounts`, `_stmt_status` (closing balance, rule order, `min_eff`), with defined null behaviour (null closing -> null status; null billed/minimum_due -> 0; null due -> already due; null/non-CARD payments ignored). (2) `_attach_velocity` rewritten in Polars: running same-day count and trailing-7-day spend. (3) `build_forecast` rewritten in Polars with a train/test split at `ML_FORECAST_TEST_START` (2026-07-01); merchant selection and `seasonal_index` from train months only; new `split_time` column.
- **Tests:** 60 passed; carry-forward regression, equality with the simulator functions, null-edge test, velocity and forecast tests. **Mutations:** billed instead of closing 3 failed; null closing 1; null `minimum_due` -> 1e9 survived (payment too small), so the fixture was strengthened and the 1.0 mutation re-run failed 1; day count 4; week total 6; tiebreak 2; forecast merchant selection 1; forecast seasonal index 2.
- **Scale (stale on-disk v7, measurement):** status replay over 289k statements / 1.01M payments 79 s; velocity over 10.43M purchases 45 s; the as-of status never beat the final status.
- **Reviewers:** codex and Kimi CHANGES_REQUESTED (the 1 ns tiebreak shifted the window boundary; synthetic fraud rows copied later-row velocity).

## Round 7 - exact window, as-of synthetic rows (834f0b8)
- **Done:** week spend = per-customer cumulative sum in (ts, id) order minus the cumulative sum through `ts <= current - 7d` (`join_asof` on real timestamps); exact `(ts-7d, ts]`, equal timestamps ordered by id. Backdated synthetic fraud rows recompute velocity (inserted into the real customer timeline), `split_time` and `device_risk_score` as-of their own time (dedicated RNG stream, main stream unchanged).
- **Tests:** 68 passed (`tests/test_v7_round7_velocity.py`): the reviewer's reproduction (10/30/30), boundary and duplicate timestamps, brute force on a coarse grid with >20 duplicate and >20 exact-boundary hits, synthetic as-of tests. **Mutations (6):** restore the 1 ns offset 3 failed; boundary row included 3; copied velocity 1; copied `split_time` 1; copied `device_risk_score` 1; future-timeline synthetic velocity 2.
- **Scale:** 10.43M purchases in 39 s, 0 mismatches against a brute force on 300 random rows.
- **Reviewers:** Kimi APPROVED; codex CHANGES_REQUESTED (synthetic `transaction_date` one calendar day ahead of its timestamp).

## Round 8 - consistent synthetic date/timestamp (c5bd837)
- **Done:** `_backdated_test_purchase_ts` places the test purchase at 22:00 the day before the fraud; `transaction_date` is the date of that timestamp (unchanged value for every row), `hour` from it; `_asof_synthetic_features` re-derives date and hour from the timestamp and computes `split_time` and the regime from it.
- **Tests:** 77 passed; `tests/test_v7_round8_synthetic_date.py` (9 tests) covers the 2026-07-01 boundary. **Mutations:** old +1 day shift 8 failed; no date re-derivation 1; no hour re-derivation 6; split/regime from date 2; `transaction_date` +1 day inside `build_fraud` is neutralised by the re-derivation (equivalent mutation, recorded as such).
- **Reviewers:** approved by codex and Kimi.

## Standing notes
- Earlier approved behaviour is kept in every later round (label-column gate, lazy refresh, pandas null semantics, payment as-of gate, exact velocity window).
- Final state before this cleanup: 77 tests passed. `evidence/builder_provenance.txt` and `evidence/deepseek-BUILD-v7.out` are untracked and untouched. A git-ignored `evidence/v7_polars_memory_round2_realdata_check.md` (the real-data check above) may remain on disk.
- Housekeeping commit (this one): old logs removed, hand-off/feedback documents moved to `docs/archive/`, per-round evidence folded into this file; no code or test behaviour changed.
