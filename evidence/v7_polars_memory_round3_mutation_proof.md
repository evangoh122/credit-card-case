# Round 3/4 mutation proof

Round 4: original fixture restored (bad_geo == 270 asserted); null cases are additional appended rows/merchant M3.

## Mutation: ne_missing semantics (null/null equal) in tx_metrics
```
FAILED tests/test_v7_polars_fixes.py::test_tx_metrics_match_pandas_reference_original_fixture
FAILED tests/test_v7_polars_fixes.py::test_tx_metrics_match_pandas_reference_with_null_rows
2 failed, 6 passed, 18 warnings in 21.81s
```
## Unmutated
```
8 passed, 18 warnings in 13.61s
```
