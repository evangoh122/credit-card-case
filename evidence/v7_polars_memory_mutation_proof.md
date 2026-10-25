# v7 polars postprocess: mutation proof

Command: .venv/bin/python -m pytest tests/test_postprocess_polars.py -q -p no:cacheprovider

## Mutation: skip nulling auth_code
```
FAILED tests/test_postprocess_polars.py::test_polars_hygiene_matches_pandas_reference
FAILED tests/test_postprocess_polars.py::test_main_end_to_end_streams_and_keeps_metadata
2 failed, 6 deselected in 24.04s
```
## Mutation: reverse sort
```
FAILED tests/test_postprocess_polars.py::test_polars_hygiene_matches_pandas_reference
FAILED tests/test_postprocess_polars.py::test_main_end_to_end_streams_and_keeps_metadata
2 failed, 6 deselected in 22.69s
```
## Mutation: missing merchant -> True
```
FAILED tests/test_postprocess_polars.py::test_polars_hygiene_matches_pandas_reference
FAILED tests/test_postprocess_polars.py::test_main_end_to_end_streams_and_keeps_metadata
2 failed, 6 deselected in 25.02s
```
## Unmutated
```
8 passed, 3 warnings in 14.84s
```
