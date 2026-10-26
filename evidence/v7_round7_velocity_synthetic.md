# Round 7: exact 7-day window on real timestamps; as-of synthetic fraud rows

Branch `fix/v7-polars-memory`, base b0b1600. Author: Claude Sonnet 5.5 (`claude-sonnet-5-5`); evidence = code inspection + tests + a scale measurement (no full build). `facts.py` untouched.

## 1. `_attach_velocity` boundary defect (blocking, both reviewers)
Cause: the 1 ns per-equal-timestamp key offset moved the rolling window's lower edge, so rows exactly 7 days earlier could be included (A/1 T 10, A/2 T 20, A/3 T+7d 30 gave 10, 30, 50 instead of 10, 30, 30). Reproduced as a failing test under mutation 1 below.
Fix (`build/ml_extracts.py`): per customer, sort by (`transaction_ts`, `transaction_id`), take the inclusive cumulative sum, and subtract the cumulative sum through the last row with `ts <= current_ts - 7 days` (`join_asof`, backward, on the REAL timestamps, with the end-of-timestamp-group cumulative sum so equal-ts rows are all treated alike). Window is exactly `(ts - 7d, ts]`; equal-timestamp rows see smaller ids through the running sum and never larger ids. `customer_day_txn_count` is a running count over (customer, date) in the same order, so the same tiebreak holds. Names, dtypes (int64/float64/int64) and row order are unchanged.
Tests (`tests/test_v7_round7_velocity.py`): reviewer reproduction; row one second inside the boundary included; duplicate timestamps exactly 7d before excluded and other customers never leak; equal timestamps ordered by id regardless of input order; 400-row brute-force check on a coarse timestamp grid that asserts it really hits duplicate and exact-7d-boundary cases (>20 each).

## 2. Backdated synthetic fraud rows leaked later-row features (blocking, Kimi)
Cause: `build_fraud` copied the later fraud row (`syn = dict(r)`), moved its date back, and kept its `customer_day_txn_count`, `customer_week_spend` (-> `_log`), `days_since_first_txn`, `split_time` and `device_risk_score`.
Choice (most defensible): recompute, do not null. New `_asof_synthetic_features`:
- velocity: the synthetic row is inserted into its customer's REAL timeline (`_synthetic_velocity`: customer's operational purchases + synthetic rows, same as-of function) and only the synthetic rows' values are read back; real rows are never altered (tested). Later transactions, including the fraud row it was copied from, cannot affect it.
- `split_time`: from the row's own backdated date (it could cross the OOT start).
- `device_risk_score`: same formula on the row's own hour and regime; its noise uses a dedicated `rng_for("fraud_synthetic_device_risk")` stream so the main ML RNG stream is unchanged.
Non-time attributes (card-present, international, category, merchant) are properties of the copied purchase, not of time, and are kept.
Pre-existing quirk, unchanged: the synthetic `transaction_ts` is `fraud_date - 1d - 2h`, i.e. the calendar day before `transaction_date`; day counts follow `transaction_date`.
Tests: features recomputed as-of own time (week spend 103.0, not 1e6 and not incl. the 900 fraud row; day count 2; days since first 2; split_time train; score not copied); appending later transactions leaves all synthetic features unchanged; real-row features untouched by the insertion.

## Mutation proofs (ml_extracts.py restored after each; unmutated suite: 68 passed)
Run: `tests/test_v7_round6_asof_extracts.py tests/test_v7_round7_velocity.py` (16 tests). Raw log below.

### Mutation: restore the 1 ns per-equal-timestamp offset (rolling window on nudged keys)
```
3 failed, 13 passed, 373 warnings in 24.93s
E             comparison failed
E       AssertionError: assert {'1': 10.0, '....0, '3': 50.0} == {'1': 10.0, '....0, '3': 30.0}
test_v7_round7_velocity.py::test_brute_force_with_duplicate_and_exact_boundary_timestamps
test_v7_round7_velocity.py::test_duplicate_timestamps_exactly_7d_before_are_all_excluded
test_v7_round7_velocity.py::test_reviewer_reproduction_equal_ts_then_exact_7d_boundary
```
### Mutation: boundary row included (exact matches off in the asof probe)
```
3 failed, 13 passed, 373 warnings in 26.59s
E             comparison failed
E       AssertionError: assert {'1': 10.0, '....0, '3': 60.0} == {'1': 10.0, '....0, '3': 30.0}
test_v7_round7_velocity.py::test_brute_force_with_duplicate_and_exact_boundary_timestamps
test_v7_round7_velocity.py::test_duplicate_timestamps_exactly_7d_before_are_all_excluded
test_v7_round7_velocity.py::test_reviewer_reproduction_equal_ts_then_exact_7d_boundary
```
### Mutation: synthetic rows keep velocity copied from the later fraud row
```
1 failed, 15 passed, 373 warnings in 26.21s
E         comparison failed
test_v7_round7_velocity.py::test_synthetic_rows_are_recomputed_as_of_their_own_time
```
### Mutation: synthetic rows keep the later row split_time
```
1 failed, 15 passed, 373 warnings in 35.36s
E       AssertionError: assert ('test' == 'train'
test_v7_round7_velocity.py::test_synthetic_rows_are_recomputed_as_of_their_own_time
```
### Mutation: synthetic rows keep the later row device_risk_score
```
1 failed, 15 passed, 373 warnings in 31.33s
test_v7_round7_velocity.py::test_synthetic_rows_are_recomputed_as_of_their_own_time
```
### Mutation: synthetic velocity sees the full future timeline (week sum over whole customer)
```
2 failed, 14 passed, 373 warnings in 42.48s
E         comparison failed
E   AssertionError: DataFrame.iloc[:, 1] (column name="customer_week_spend") are different
test_v7_round7_velocity.py::test_synthetic_features_do_not_change_when_later_transactions_are_appended
test_v7_round7_velocity.py::test_synthetic_rows_are_recomputed_as_of_their_own_time
```

## Scale check (on-disk credit_card_data_v7, measurement)
```
/mnt/d/Get shit Done/Data Science/DS Work/Credit Card Case/build/ml_extracts.py:186: UserWarning: Sortedness of columns cannot be checked when 'by' groups provided
  before = probe.join_asof(
/tmp/claude-1000/-home-jianj/d5393740-c9e3-4570-b8f3-1644bda45814/scratchpad/scale7.py:14: DeprecationWarning: The 'generic' unit for NumPy timedelta is deprecated, and will raise an error in the future. This includes implicit conversion of bare integers (e.g. `+ 1`). Please use a specific unit instead.
  wk=mine[mine.transaction_ts>row.transaction_ts-pd.Timedelta(days=7)].amount_usd.sum()
velocity 39 s 10431009
bruteforce mismatches 0 of 300
```
