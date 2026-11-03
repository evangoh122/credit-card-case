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


def _write_sql(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_run_sql_rejects_truth_columns_from_a_connection(memory_db, tmp_path):
    sql = _write_sql(tmp_path, "truth.sql", "SELECT 1 AS sim_truth_risk;\n")
    with pytest.raises(ValueError, match="sim_truth_risk"):
        sql_runner.run_sql(memory_db, sql)


def test_run_sql_rejects_truth_columns_from_a_file(tmp_path):
    db = tmp_path / "warehouse.duckdb"
    with duckdb.connect(str(db)) as con:
        _seed_warehouse(con)
    sql = _write_sql(tmp_path, "truth.sql", "SELECT 1 AS sim_truth_risk;\n")
    with pytest.raises(ValueError, match="sim_truth_risk"):
        sql_runner.run_sql(db, sql)


def test_run_sql_rejects_truth_columns_from_a_warehouse_directory(tmp_path):
    build = tmp_path / "build-1"
    build.mkdir()
    for name in ("hub", "dim", "facts", "gold"):
        with duckdb.connect(str(build / f"{name}.duckdb")):
            pass
    (tmp_path / "current").symlink_to(build)
    sql = _write_sql(tmp_path, "truth.sql", "SELECT 1 AS sim_truth_risk;\n")
    with pytest.raises(ValueError, match="sim_truth_risk"):
        sql_runner.run_sql(tmp_path, sql)


def test_run_sql_does_not_create_a_missing_database_file(tmp_path):
    db = tmp_path / "typo.duckdb"
    with pytest.raises(FileNotFoundError, match="missing"):
        sql_runner.run_sql(db, ROW_COUNTS)
    assert not db.exists()


def test_run_sql_opens_existing_files_read_only_by_default(tmp_path):
    db = tmp_path / "warehouse.duckdb"
    with duckdb.connect(str(db)) as con:
        _seed_warehouse(con)
    write = _write_sql(tmp_path, "write.sql", "INSERT INTO dim.dim_customer VALUES ('c3');\n")
    with pytest.raises(duckdb.Error, match="read-only|read only"):
        sql_runner.run_sql(db, write)
    frame = sql_runner.run_sql(db, write, read_only=False)
    assert frame["Count"].to_list() == [1]


def test_run_sql_sets_the_utc_session_timezone(memory_db, tmp_path):
    sql = _write_sql(tmp_path, "tz.sql", "SELECT current_setting('TimeZone') AS tz;\n")
    memory_db.execute("SET TimeZone = 'America/Los_Angeles'")
    frame = sql_runner.run_sql(memory_db, sql)
    assert frame["tz"].to_list() == ["UTC"]


NAIVE_TIMESTAMP_SQL = "SELECT TIMESTAMP '2026-11-03 12:00:00' AS event_ts_utc;\n"


def test_run_sql_rejects_naive_timestamps_from_a_connection(tmp_path):
    sql = _write_sql(tmp_path, "naive.sql", NAIVE_TIMESTAMP_SQL)
    with duckdb.connect(":memory:") as con:
        with pytest.raises(ValueError, match="naive|timestamp|UTC"):
            sql_runner.run_sql(con, sql)


def test_run_sql_rejects_naive_timestamps_from_a_file(tmp_path):
    db = tmp_path / "warehouse.duckdb"
    duckdb.connect(str(db)).close()
    sql = _write_sql(tmp_path, "naive.sql", NAIVE_TIMESTAMP_SQL)
    with pytest.raises(ValueError, match="naive|timestamp|UTC"):
        sql_runner.run_sql(db, sql)


def test_run_sql_rejects_naive_timestamps_from_a_warehouse_directory(tmp_path):
    build = tmp_path / "build-1"
    build.mkdir()
    for name in ("hub", "dim", "facts", "gold"):
        duckdb.connect(str(build / f"{name}.duckdb")).close()
    (tmp_path / "current").symlink_to(build)
    sql = _write_sql(tmp_path, "naive.sql", NAIVE_TIMESTAMP_SQL)
    with pytest.raises(ValueError, match="naive|timestamp|UTC"):
        sql_runner.run_sql(tmp_path, sql)


def test_run_sql_returns_aware_utc_timestamps_and_keeps_dates(memory_db, tmp_path):
    sql = _write_sql(
        tmp_path,
        "aware.sql",
        "SELECT TIMESTAMPTZ '2026-11-03 00:30:00+00' AS ts, "
        "CAST(TIMESTAMPTZ '2026-11-03 00:30:00+00' AS DATE) AS day;\n",
    )
    memory_db.execute("SET TimeZone = 'America/Los_Angeles'")
    frame = sql_runner.run_sql(memory_db, sql)
    assert frame.schema["ts"].time_zone == "UTC"
    assert frame.schema["day"] == pl.Date
    assert frame["ts"].to_list()[0].isoformat() == "2026-11-03T00:30:00+00:00"
    assert str(frame["day"][0]) == "2026-11-03"
