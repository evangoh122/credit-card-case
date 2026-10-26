# Round 5: credit-default as-of payment features

Branch `fix/v7-polars-memory`, base e4795f4. Reviewer: Claude Sonnet 5.5 (`claude-sonnet-5-5`), code inspection plus tests (no full build run).

## Verdict: CONFIRMED (for `status_at_obs`, `paid_full_share_3m`); NOT_CONFIRMED (for `payments_3m` and the other statement fields)

### Evidence (code inspection)
- `build/facts.py:1437-1465` `_settled_closing_amounts`: settles each statement with every card payment dated after its cut, with no upper date bound (only `PAYMENT_HORIZON` 2026-09-30, `build/config.py:36`).
- `build/facts.py:1985-1987` `s["settled"]` is filled after ALL payments are generated; `build/facts.py:1481-1499` `_stmt_status` turns it (and the final `alloc_min`/`alloc_bal` from `_waterfall`, `facts.py:1398-1434`) into `payment_status`; written at `facts.py:2010-2013`.
- `build/ml_extracts.py` (before this round) read that final `payment_status` for `status_at_obs` and `paid_full_share_3m`. A statement cut on or before 2026-03-31 could therefore read PAID_FULL because of a payment made after the observation date. `max_source_date` only tracked `statement_date` and the delinquency snapshot date, so `check_no_leakage` could not see it.
- Not leaky: `payments_amount` is `paid_this_cycle`, the sum of payments dated <= `cycle_end` = `min(cut, eff_end)` (`facts.py:1683-1698`, `1856`), so it is as-of the statement cut. `closing_balance`, `minimum_due`, `utilization_pct`, `credit_limit` are cut-time values. Delinquency `days_past_due` (`facts.py:1898-1912`) uses `alloc_min` at the cut, so it is as-of too.

### Demonstration (synthetic, in `tests/test_v7_asof_payments.py`)
- `test_confirmed_final_status_uses_cash_after_observation`: a statement cut 20 days before obs, closing 100, paid by a payment 20 days AFTER obs -> `_settled_closing_amounts` returns `[100.0]` and `_stmt_status` returns `PAID_FULL`.
- `test_old_feature_would_leak_asof_feature_does_not`: old feature value PAID_FULL; as-of value is PENDING (unpaid at obs, due after obs), and `paid_full_share_3m` is 0.5 instead of 1.0.

### Fix
- `ml_extracts.asof_payment_status` recomputes status from `fact_payment_allocation` rows with `allocation_date <= observation_date` (same rule order as `facts._stmt_status`, using allocated-so-far in place of final settlement and `minimum_due` for `min_eff`). `build_credit_default` takes a new `allocations` argument; `build_all` reads `fact_payment_allocation` (`statement_id, allocation_date, amount_usd`) plus `billed_amount, payment_due_date` from `fact_statement`.
- New feature column `payment_feature_date` (latest payment event used, per account) is part of `max_source_date`, so `check_no_leakage` now covers payment dates.
- New gate `check_payment_sources_asof` raises if any payment event used is dated after obs.
- Value change: `status_at_obs` and `paid_full_share_3m` are now as-of values (statements due after obs show PENDING/partial instead of their final outcome). Column names and dtypes are unchanged; `payment_feature_date` is a new datetime column.

### Audit of other extracts
- Churn: reads only `closing_balance`/`utilization_pct` from statements (cut-time values). `test_churn_features_use_no_settled_statement_fields` shows the features are unchanged when `payment_status`, `paid_amount`, `is_late` are scrambled and that no settled field is in the extract.
- Fraud: no statement/payment fields. Note (not changed, out of scope): `customer_day_txn_count` and `customer_week_spend` (`_attach_velocity`) count the whole day/week, including later transactions of that period; these are documented as point-in-time proxies.
- Segmentation and forecast: descriptive full-period aggregates with no observation date or label; `seasonal_index` uses whole-period totals.
- `facts.py` RNG behaviour is untouched.

### Mutation proofs (tests/test_v7_asof_payments.py, 12 tests; unmutated: 12 passed)
Source: `/tmp/mut5.md` run on this commit's `ml_extracts.py`, restored after each.

| Mutation | Result |
|---|---|
| drop the `<= obs` filter on payment events | 9 failed, 3 passed |
| features read the final statement `payment_status` again | 7 failed, 5 passed |
| payment dates left out of `max_source_date` | 1 failed (`test_max_source_date_tracks_payment_dates`) |
| neuter `check_payment_sources_asof` | 1 failed (`test_payment_gate_fails_on_post_observation_event`) |
| churn reads the final statement `payment_status` | 1 failed (`test_churn_features_use_no_settled_statement_fields`) |

Full per-test output:

### Mutation: drop the <= obs filter on payment events
```
FAILED tests/test_v7_asof_payments.py::test_old_feature_would_leak_asof_feature_does_not
FAILED tests/test_v7_asof_payments.py::test_features_invariant_to_post_observation_payments[post_obs0]
FAILED tests/test_v7_asof_payments.py::test_features_invariant_to_post_observation_payments[post_obs1]
FAILED tests/test_v7_asof_payments.py::test_features_invariant_to_post_observation_payments[post_obs2]
FAILED tests/test_v7_asof_payments.py::test_features_invariant_to_post_observation_payments[post_obs3]
FAILED tests/test_v7_asof_payments.py::test_features_invariant_to_post_observation_payments[post_obs4]
FAILED tests/test_v7_asof_payments.py::test_pre_observation_payment_does_change_features
FAILED tests/test_v7_asof_payments.py::test_max_source_date_tracks_payment_dates
FAILED tests/test_v7_asof_payments.py::test_asof_status_matches_facts_rules_on_edge_rows
9 failed, 3 passed, 120 warnings in 23.14s
```
### Mutation: features read the final statement payment_status again
```
FAILED tests/test_v7_asof_payments.py::test_old_feature_would_leak_asof_feature_does_not
FAILED tests/test_v7_asof_payments.py::test_features_invariant_to_post_observation_payments[post_obs0]
FAILED tests/test_v7_asof_payments.py::test_features_invariant_to_post_observation_payments[post_obs1]
FAILED tests/test_v7_asof_payments.py::test_features_invariant_to_post_observation_payments[post_obs2]
FAILED tests/test_v7_asof_payments.py::test_features_invariant_to_post_observation_payments[post_obs3]
FAILED tests/test_v7_asof_payments.py::test_features_invariant_to_post_observation_payments[post_obs4]
FAILED tests/test_v7_asof_payments.py::test_pre_observation_payment_does_change_features
7 failed, 5 passed, 201 warnings in 23.14s
```
### Mutation: payment dates left out of max_source_date
```
FAILED tests/test_v7_asof_payments.py::test_max_source_date_tracks_payment_dates
1 failed, 11 passed, 205 warnings in 21.10s
```
### Mutation: neuter the payment-source gate
```
FAILED tests/test_v7_asof_payments.py::test_payment_gate_fails_on_post_observation_event
1 failed, 11 passed, 204 warnings in 22.43s
```
### Mutation: churn reads the final statement payment_status
```
FAILED tests/test_v7_asof_payments.py::test_churn_features_use_no_settled_statement_fields
1 failed, 11 passed, 205 warnings in 22.13s
```
