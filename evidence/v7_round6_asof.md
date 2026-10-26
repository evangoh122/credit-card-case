# Round 6: payment-status mirror, as-of fraud velocity, as-of forecast

Branch `fix/v7-polars-memory`, base 3f8c0a1. Author: Claude Sonnet 5.5 (`claude-sonnet-5-5`); evidence = code inspection + tests (no full build).
Reviewer reports: `/tmp/polly-review/v7fix1_r5_codex_validation.md` (codex), Kimi findings as relayed in the task.

## 1. Statement status now mirrors `facts._stmt_status` (blocking)
Cause: `asof_payment_status` compared allocations with `billed_amount`; `facts._stmt_status` settles against `closing_balance` (`facts.py:1489`). They differ on carry-forward statements (closing 150, billed 50, cash 50 returned PAID_FULL instead of PAID_PARTIAL).
Fix (`build/ml_extracts.py`, `asof_payment_status`): card payments dated <= obs are replayed through the simulator's own `facts._waterfall` (allocation to minimums/balances, statements eligible only if cut < payment date), `facts._settled_closing_amounts` (cash that settles each closing balance) and `facts._stmt_status` (rule order, thresholds, `min_eff = min(minimum_due, billed)`), with obs as the period end. The function now takes `fact_payment` (CARD rows) instead of allocations; `build_all` reads `account_id, payment_date, amount_usd, payment_type`.
Null behaviour (defined and tested): null `closing_balance` -> null status; null `billed_amount`/`minimum_due` -> 0; null `payment_due_date` -> already due (UNPAID, never PENDING); payments with null account/date/amount and non-CARD payments are ignored; no payments -> no cash.
Tests (`tests/test_v7_asof_payments.py`): `test_asof_status_rules_including_carry_forward` (row I is the carry-forward regression), `test_asof_status_equals_simulator_when_nothing_is_cut_off` (replay == facts functions), `test_asof_status_null_edges_are_defined`; all earlier invariance/gate tests kept and moved to payments.

## 2. Fraud velocity strictly as-of (blocking)
Cause: `_attach_velocity` attached complete customer-day / calendar-week totals to every row.
Fix (Polars, no pandas full-load; only five slim columns are converted): ordered per customer by (`transaction_ts`, `transaction_id`).
- `customer_day_txn_count` (int64): running count of the customer's transactions on the same `transaction_date`, including the current one.
- `customer_week_spend` (float64): sum of `amount_usd` over the trailing 7 days `(ts - 7d, ts]`, including the current transaction (rolling window; the calendar-week bucket cannot be made as-of). Equal timestamps get 1 ns steps so the id tiebreak is honoured.
- `days_since_first_txn` (int64) unchanged in meaning (first transaction is always earlier).
Names/dtypes/row order preserved. Documented in the ML README.
Tests (`tests/test_v7_round6_asof_extracts.py`): exact window values, appending later rows (same day, same week, same ts larger id, 30 days later, other customer) leaves earlier rows unchanged, and a brute-force per-row as-of check on 300 random transactions.

## 3. Forecast extract as-of (blocking Kimi / non-blocking codex)
Choice: train/test split at `C.ML_FORECAST_TEST_START` (2026-07-01, new config constant, same date as the fraud OOT start). Merchant selection (top 5,000 by spend) and `seasonal_index` use train months only. `seasonal_index` = train total for the row's calendar month / mean train monthly total; null for a test month whose calendar month never occurs in train (July, August 2026; September 2026 uses September 2025). New column `split_time` (train/test). Lags unchanged. `build_forecast` rewritten in Polars; dtypes of the existing columns preserved.
Tests: split/seasonality definition and dtypes; test-window spend explosions change neither merchant selection nor any train row/seasonal_index.

## Mutation proofs (same ml_extracts.py restored after each; unmutated: all pass)
Run: `tests/test_v7_asof_payments.py tests/test_v7_round6_asof_extracts.py`.

### Mutation: PAID_FULL measured against billed_amount instead of closing_balance
```
test_v7_asof_payments.py::test_asof_status_rules_including_carry_forward
test_v7_asof_payments.py::test_asof_status_equals_simulator_when_nothing_is_cut_off
test_v7_asof_payments.py::test_asof_status_null_edges_are_defined
3 failed, 19 passed, 537 warnings in 22.17s
```
### Mutation: null closing_balance no longer yields a null status
```
test_v7_asof_payments.py::test_asof_status_null_edges_are_defined
1 failed, 21 passed, 541 warnings in 22.81s
```
### Mutation: day count uses the complete day total
```
test_v7_round6_asof_extracts.py::test_velocity_window_definition
test_v7_round6_asof_extracts.py::test_appending_later_transactions_never_changes_earlier_rows[later0]
test_v7_round6_asof_extracts.py::test_appending_later_transactions_never_changes_earlier_rows[later2]
test_v7_round6_asof_extracts.py::test_velocity_on_random_data_matches_bruteforce_asof
4 failed, 18 passed, 243 warnings in 22.44s
```
### Mutation: week spend uses the customer total (look-ahead)
```
E             comparison failed
test_v7_round6_asof_extracts.py::test_velocity_window_definition
test_v7_round6_asof_extracts.py::test_appending_later_transactions_never_changes_earlier_rows[later0]
test_v7_round6_asof_extracts.py::test_appending_later_transactions_never_changes_earlier_rows[later1]
test_v7_round6_asof_extracts.py::test_appending_later_transactions_never_changes_earlier_rows[later2]
test_v7_round6_asof_extracts.py::test_appending_later_transactions_never_changes_earlier_rows[later3]
test_v7_round6_asof_extracts.py::test_velocity_on_random_data_matches_bruteforce_asof
6 failed, 16 passed, 242 warnings in 22.25s
```
### Mutation: no (ts, id) tiebreak for equal timestamps
```
test_v7_round6_asof_extracts.py::test_velocity_window_definition
test_v7_round6_asof_extracts.py::test_appending_later_transactions_never_changes_earlier_rows[later2]
2 failed, 20 passed, 541 warnings in 22.10s
```
### Mutation: forecast merchant selection uses full-period spend
```
test_v7_round6_asof_extracts.py::test_forecast_unaffected_by_test_window_spend
1 failed, 21 passed, 541 warnings in 23.05s
```
### Mutation: forecast seasonal_index uses full-period totals
```
test_v7_round6_asof_extracts.py::test_forecast_split_and_seasonality_use_train_months_only
test_v7_round6_asof_extracts.py::test_forecast_unaffected_by_test_window_spend
2 failed, 20 passed, 541 warnings in 21.49s
```

### Mutation: null minimum_due coerced to 1.0 instead of 0 (re-run after strengthening N3)
```
test_v7_asof_payments.py::test_asof_status_null_edges_are_defined
1 failed, 13 passed, 236 warnings in 20.43s
```
Note: the first null-minimum_due mutation (-> 1e9) survived because the N3 payment (3.0) was too small to distinguish it; N3 was raised to 20.0 and the mutation re-run (last block, now fails).

## Scale check on the on-disk credit_card_data_v7 (stale build; measurement, not a gate)
asof_payment_status over 289,045 statements <= obs and 1,013,762 payments: 79 s. _attach_velocity over 10.43M approved purchases: 45 s.
Final vs as-of status (rows = final statement payment_status, columns = as-of status): the as-of status never moves toward a better outcome than the final one, and CREDIT_BALANCE / NO_BALANCE / CARRY_FORWARD agree exactly; later cash explains every PAID_FULL -> PAID_PARTIAL/PENDING/UNPAID move.
```
col_0           CARRY_FORWARD  CREDIT_BALANCE  NO_BALANCE  PAID_FULL  PAID_PARTIAL  PARTIAL  PENDING  UNPAID
payment_status                                                                                              
CARRY_FORWARD              63               0           0          0             0        0        0       0
CREDIT_BALANCE              0            5415           0          0             0        0        0       0
NO_BALANCE                  0               0        4896          0             0        0        0       0
PAID_FULL                   0               0           0     162555         69288      473    15604    2834
PAID_PARTIAL                0               0           0          0         14554      250     4458     820
PARTIAL                     0               0           0          0             0       16        4       3
UNPAID                      0               0           0          0             0        0      985    6827
velocity 45 s 10431009
```
