# Round 8: synthetic fraud rows are internally consistent (date, hour, split, regime)

Branch `fix/v7-polars-memory`, base 834f0b8. Author: Claude Sonnet 5.5 (`claude-sonnet-5-5`); evidence = code inspection + tests (no full build). `facts.py` untouched.

## Defect (codex, blocking)
`build_fraud` built each ML-only test purchase with `transaction_date = fraud_date - 1d` but `transaction_ts = fraud_date - 1d - 2h`, i.e. the timestamp fell on the calendar day BEFORE `transaction_date` (and `hour` came from it). At the 2026-07-01 OOT boundary `split_time` and the device-risk regime (+0.42) were taken from the date while the timestamp was a day earlier (fraud on 2026-07-02 gave date 07-01 -> `test`, ts 06-30 22:00 -> should be `train`).

## Choice
Keep the intended backdating (the test purchase is on the calendar day BEFORE the fraud, recorded `test_purchase_gap_days` = 1.0, hour 22) and the downstream contract (`transaction_date` still = fraud date - 1 day), and move the TIMESTAMP onto that date:
`_backdated_test_purchase_ts(fraud_date) = fraud_date.normalize() - 2h` (22:00 the day before). `transaction_date` is `ts.normalize()`, `hour` is `ts.hour`. Effect on values: only `transaction_ts` (not an output column) and `hour`/velocity inputs move; `transaction_date` is unchanged for every row.
`_asof_synthetic_features` additionally re-derives `transaction_date` and `hour` from `transaction_ts` (so they cannot disagree), computes the OOT regime / `split_time` from the timestamp (`ts >= ML_FRAUD_OOT_START`), and recomputes velocity and `device_risk_score` from that same moment (earlier fixes kept).

## Tests (`tests/test_v7_round8_synthetic_date.py`, 9 tests; `_syn()` fixture in the round-7 file made consistent)
- helper: real txn 2026-07-01 00:30 -> synthetic 2026-06-30 22:00; 2026-06-30 23:30 -> 2026-06-29 22:00; 2026-07-02 -> 2026-07-01 22:00; always date == fraud date - 1 day, hour 22.
- pre-round-8 shaped input (date a day ahead of ts, copied hour): output date == date(ts), hour == ts.hour, `split_time` train/test/train from the timestamp, and device-risk regime gap > 0.25 between rows on either side of the boundary.
- five fraud timestamps incl. both boundary cases: helper + features self-consistent (date == date(ts), hour, split, date == fraud date - 1).

## Mutation proofs (ml_extracts.py restored after each; unmutated suite: see pytest file)
Run: `tests/test_v7_round8_synthetic_date.py tests/test_v7_round7_velocity.py` (17 tests).

### Mutation: restore the old shift (timestamp a further day before transaction_date)
```
test_v7_round8_synthetic_date.py::test_backdated_timestamp_date_and_gap[2026-07-01 00:30-2026-06-30 22:00]
test_v7_round8_synthetic_date.py::test_backdated_timestamp_date_and_gap[2026-06-30 23:30-2026-06-29 22:00]
test_v7_round8_synthetic_date.py::test_backdated_timestamp_date_and_gap[2026-07-02 09:00-2026-07-01 22:00]
test_v7_round8_synthetic_date.py::test_end_to_end_helper_plus_features_are_self_consistent[2026-07-01 00:30]
test_v7_round8_synthetic_date.py::test_end_to_end_helper_plus_features_are_self_consistent[2026-06-30 23:30]
test_v7_round8_synthetic_date.py::test_end_to_end_helper_plus_features_are_self_consistent[2026-07-02 00:00]
test_v7_round8_synthetic_date.py::test_end_to_end_helper_plus_features_are_self_consistent[2026-06-15 12:00]
test_v7_round8_synthetic_date.py::test_end_to_end_helper_plus_features_are_self_consistent[2026-01-01 00:00]
8 failed, 9 passed, 76 warnings in 27.86s
```
### Mutation: date no longer re-derived from the timestamp
```
test_v7_round8_synthetic_date.py::test_inconsistent_inputs_are_rederived_from_the_timestamp
1 failed, 16 passed, 79 warnings in 23.78s
```
### Mutation: hour no longer re-derived from the timestamp
```
test_v7_round8_synthetic_date.py::test_inconsistent_inputs_are_rederived_from_the_timestamp
test_v7_round8_synthetic_date.py::test_end_to_end_helper_plus_features_are_self_consistent[2026-07-01 00:30]
test_v7_round8_synthetic_date.py::test_end_to_end_helper_plus_features_are_self_consistent[2026-06-30 23:30]
test_v7_round8_synthetic_date.py::test_end_to_end_helper_plus_features_are_self_consistent[2026-07-02 00:00]
test_v7_round8_synthetic_date.py::test_end_to_end_helper_plus_features_are_self_consistent[2026-06-15 12:00]
test_v7_round8_synthetic_date.py::test_end_to_end_helper_plus_features_are_self_consistent[2026-01-01 00:00]
6 failed, 11 passed, 74 warnings in 28.04s
```
### Mutation: split_time/regime from the (possibly inconsistent) date instead of the timestamp
```
test_v7_round8_synthetic_date.py::test_inconsistent_inputs_are_rederived_from_the_timestamp
test_v7_round8_synthetic_date.py::test_end_to_end_helper_plus_features_are_self_consistent[2026-07-01 00:30]
2 failed, 15 passed, 88 warnings in 24.64s
```
### Mutation: build_fraud keeps transaction_date a day ahead of the timestamp
```
17 passed, 79 warnings in 19.05s
```

Note on the last mutation: setting `transaction_date` a day ahead inside `build_fraud` is neutralised by the re-derivation in `_asof_synthetic_features` (defence in depth), so no unit test can observe it; it is an equivalent mutation, not a coverage gap in the shipped output. The original +1 day shift (first mutation) fails 8 tests.
