from pathlib import Path

import duckdb
import polars as pl
import pytest

from ccc.mlops import sql_runner

SQL_DIR = Path(__file__).resolve().parents[1] / "mlops" / "sql"
ROW_COUNTS = SQL_DIR / "001_row_counts.sql"

EXPECTED = [
    {"table_name": "dim.dim_customer", "row_count": 2},
    {"table_name": "facts.fact_events", "row_count": 1},
    {"table_name": "facts.fact_transactions", "row_count": 3},
    {"table_name": "gold.fact_engagement_snapshot", "row_count": 2},
    {"table_name": "gold.fact_rfm_snapshot", "row_count": 4},
]


def _seed_warehouse(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE SCHEMA dim")
    con.execute("CREATE SCHEMA facts")
    con.execute("CREATE SCHEMA gold")
    con.execute("CREATE TABLE dim.dim_customer(customer_id VARCHAR)")
    con.execute("INSERT INTO dim.dim_customer VALUES ('c1'), ('c2')")
    con.execute("CREATE TABLE facts.fact_transactions(transaction_id VARCHAR)")
    con.execute("INSERT INTO facts.fact_transactions VALUES ('t1'), ('t2'), ('t3')")
    con.execute("CREATE TABLE facts.fact_events(event_id VARCHAR, event_type VARCHAR)")
    con.execute("INSERT INTO facts.fact_events VALUES ('e1', 'app_open')")
    con.execute("CREATE TABLE gold.fact_rfm_snapshot(customer_id VARCHAR)")
    con.execute("INSERT INTO gold.fact_rfm_snapshot VALUES ('c1'), ('c2'), ('c3'), ('c4')")
    con.execute("CREATE TABLE gold.fact_engagement_snapshot(customer_id VARCHAR)")
    con.execute("INSERT INTO gold.fact_engagement_snapshot VALUES ('c1'), ('c2')")


@pytest.fixture
def memory_db():
    con = duckdb.connect(":memory:")
    _seed_warehouse(con)
    yield con
    con.close()


def test_run_sql_returns_expected_rows_as_polars(memory_db):
    frame = sql_runner.run_sql(memory_db, ROW_COUNTS)
    assert isinstance(frame, pl.DataFrame)
    assert frame.to_dicts() == EXPECTED


def test_run_sql_reads_a_duckdb_file(tmp_path):
    db = tmp_path / "warehouse.duckdb"
    with duckdb.connect(str(db)) as con:
        _seed_warehouse(con)
    frame = sql_runner.run_sql(db, ROW_COUNTS)
    assert frame.to_dicts() == EXPECTED


def test_cli_prints_the_result(tmp_path, capsys):
    db = tmp_path / "warehouse.duckdb"
    with duckdb.connect(str(db)) as con:
        _seed_warehouse(con)
    assert sql_runner.main(["--db", str(db), "--sql", str(ROW_COUNTS)]) == 0
    out = capsys.readouterr().out
    assert "dim.dim_customer" in out
    assert "gold.fact_rfm_snapshot" in out
