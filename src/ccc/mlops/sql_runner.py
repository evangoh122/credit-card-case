"""Run versioned SQL files against a DuckDB warehouse and return Polars frames."""

import argparse
import sys
from pathlib import Path

import duckdb
import polars as pl

from ccc.gold import guard_no_truth_columns


def _guard_utc_timestamps(frame: pl.DataFrame) -> None:
    """Reject timestamp columns that are not timezone-aware UTC.

    DuckDB returns ``TIMESTAMPTZ`` in the session timezone, which ``_result``
    pins to UTC, but a bare ``TIMESTAMP`` stays naive and would silently mix
    with aware warehouse columns. Naive (or non-UTC) results are an error.
    """
    naive = sorted(
        name
        for name, dtype in frame.schema.items()
        if isinstance(dtype, pl.Datetime) and dtype.time_zone is None
    )
    non_utc = sorted(
        name
        for name, dtype in frame.schema.items()
        if isinstance(dtype, pl.Datetime) and dtype.time_zone not in (None, "UTC")
    )
    if naive or non_utc:
        raise ValueError(
            f"SQL timestamp columns must be timezone-aware UTC; naive: {naive}, non-UTC: {non_utc}"
        )


def _result(con: duckdb.DuckDBPyConnection, sql: str) -> pl.DataFrame:
    """Run ``sql`` in UTC and return its result after dtype and truth guards."""
    con.execute("SET TimeZone = 'UTC'")
    frame = con.sql(sql).pl()
    _guard_utc_timestamps(frame)
    guard_no_truth_columns(frame)
    return frame


def run_sql(
    db: str | Path | duckdb.DuckDBPyConnection,
    sql_file: str | Path,
    *,
    read_only: bool = True,
) -> pl.DataFrame:
    """Execute a ``.sql`` file and return its final result as a Polars DataFrame.

    ``db`` may be an open DuckDB connection, a warehouse directory built by
    :mod:`ccc.duck.build`, or a plain DuckDB database file. File databases are
    opened read-only unless ``read_only`` is cleared, and a missing file is an
    error instead of a newly created database. Every branch runs in UTC and
    rejects naive or non-UTC timestamp columns, then applies
    :func:`ccc.gold.guard_no_truth_columns` before returning.
    """
    sql = Path(sql_file).read_text(encoding="utf-8")
    if isinstance(db, duckdb.DuckDBPyConnection):
        return _result(db, sql)
    path = Path(db)
    if path.is_dir():
        from ccc.duck import connect

        with connect(path) as con:
            return _result(con, sql)
    if not path.is_file():
        raise FileNotFoundError(f"duckdb database file is missing: {path}")
    with duckdb.connect(str(path), read_only=read_only) as con:
        return _result(con, sql)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ccc.mlops.sql_runner")
    parser.add_argument("--db", required=True, help="warehouse directory or DuckDB file")
    parser.add_argument("--sql", required=True, help="versioned .sql file to run")
    args = parser.parse_args(argv)
    frame = run_sql(args.db, args.sql)
    with pl.Config(tbl_rows=50, tbl_cols=20):
        print(frame)
    return 0


if __name__ == "__main__":
    sys.exit(main())
