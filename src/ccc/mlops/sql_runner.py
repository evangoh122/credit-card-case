"""Run versioned SQL files against a DuckDB warehouse and return Polars frames."""

import argparse
import sys
from pathlib import Path

import duckdb
import polars as pl


def run_sql(db: str | Path | duckdb.DuckDBPyConnection, sql_file: str | Path) -> pl.DataFrame:
    """Execute a ``.sql`` file and return its final result as a Polars DataFrame.

    ``db`` may be an open DuckDB connection, a warehouse directory built by
    :mod:`ccc.duck.build`, or a plain DuckDB database file.
    """
    sql = Path(sql_file).read_text(encoding="utf-8")
    if isinstance(db, duckdb.DuckDBPyConnection):
        return db.sql(sql).pl()
    path = Path(db)
    if path.is_dir():
        from ccc.duck import connect

        with connect(path) as con:
            return con.sql(sql).pl()
    with duckdb.connect(str(path)) as con:
        return con.sql(sql).pl()


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
