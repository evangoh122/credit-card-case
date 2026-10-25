# Round 2 mutation proofs

Test file: tests/test_v7_polars_fixes.py (run with -p no:cacheprovider)

## Mutation: re-add customer_status/churn_date to churn features
```
FAILED tests/test_v7_polars_fixes.py::test_churn_features_exclude_label_columns
1 failed, 6 passed, 18 warnings in 20.57s
```
## Mutation: re-add account_status/closed_date to credit-default features
```
FAILED tests/test_v7_polars_fixes.py::test_credit_default_features_exclude_label_columns
1 failed, 6 passed, 18 warnings in 21.65s
```
## Mutation: neuter the label-column gate
```
FAILED tests/test_v7_polars_fixes.py::test_label_column_gate_fails_on_every_defining_column[churn]
FAILED tests/test_v7_polars_fixes.py::test_label_column_gate_fails_on_every_defining_column[credit_default]
2 failed, 5 passed, 18 warnings in 24.67s
```
## Mutation: refresh_docs pandas-loads fact_transaction again
```
FAILED tests/test_v7_polars_fixes.py::test_refresh_never_pandas_loads_fact_transaction
1 failed, 6 passed, 18 warnings in 25.20s
```
## Mutation: break the Polars FK metric
```
FAILED tests/test_v7_polars_fixes.py::test_tx_metrics_match_pandas_reference
1 failed, 6 passed, 18 warnings in 24.63s
```
## Unmutated
```
7 passed, 18 warnings in 14.56s
```
