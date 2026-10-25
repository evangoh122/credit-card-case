# Round 3 mutation proof

## Mutation: ne_missing semantics (null/null equal) in tx_metrics
```
323:        return a.ne_missing(b)
FAILED tests/test_v7_polars_fixes.py::test_tx_metrics_match_pandas_reference
1 failed, 6 passed, 18 warnings in 24.89s
```
## Unmutated
```
7 passed, 18 warnings in 13.49s
```
