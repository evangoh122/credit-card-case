"""Build connected DuckDB files from a synthetic Parquet export."""

import argparse
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from ccc.duck import DATABASES, _sql_string, connect

FACT_TABLES = ("fact_transactions", "fact_events")
OUTPUT_FILES = tuple(f"{name}.duckdb" for name in (*DATABASES, "hub")) + ("connect.sql",)


def _parquet_source(root: Path, table: str) -> str:
    path = root / table
    if table in FACT_TABLES:
        pattern = path / "*" / "*.parquet"
    else:
        pattern = root / f"{table}.parquet"
    return _sql_string(pattern)


def _read_manifest(parquet_dir: Path) -> dict:
    manifest_path = parquet_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing export manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("timezone") != "UTC":
        raise ValueError("Parquet export must use UTC timestamps")
    if manifest.get("data_label") != "synthetic":
        raise ValueError("Parquet export must be labelled synthetic")
    if manifest.get("include_truth") or "sim_truth_customer" in manifest.get("row_counts", {}):
        raise ValueError("sim_truth_customer exports are refused")
    if (parquet_dir / "sim_truth_customer.parquet").exists():
        raise ValueError("sim_truth_customer must never be loaded")
    return manifest


def _configure(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("SET TimeZone = 'UTC'")


def _build_dim(path: Path, parquet_dir: Path) -> None:
    with duckdb.connect(str(path)) as con:
        _configure(con)
        source = _parquet_source(parquet_dir, "dim_customer")
        columns = {
            row[0]
            for row in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{source}')").fetchall()
        }
        forbidden = {
            name for name in columns if "local" in name.lower() or name.startswith("sim_truth_")
        }
        if forbidden:
            raise ValueError(f"dim_customer contains forbidden columns: {sorted(forbidden)}")
        con.execute(
            f"CREATE TABLE dim_customer AS SELECT * FROM read_parquet('{source}') "
            "ORDER BY customer_id"
        )
        con.execute("ALTER TABLE dim_customer ADD PRIMARY KEY (customer_id)")


def _build_facts(path: Path, parquet_dir: Path) -> None:
    with duckdb.connect(str(path)) as con:
        _configure(con)
        for table in FACT_TABLES:
            source = _parquet_source(parquet_dir, table)
            con.execute(
                f"CREATE TABLE {table} AS SELECT * FROM read_parquet('{source}') "
                "ORDER BY event_ts_utc"
            )


def _build_gold(path: Path, dim_path: Path, facts_path: Path, snapshot_date: str) -> None:
    with duckdb.connect(str(path)) as con:
        _configure(con)
        con.execute(f"ATTACH '{_sql_string(dim_path)}' AS dim (READ_ONLY)")
        con.execute(f"ATTACH '{_sql_string(facts_path)}' AS facts (READ_ONLY)")
        con.execute(
            """
            CREATE TABLE fact_rfm_snapshot AS
            WITH customers AS (
                SELECT customer_id, tz,
                       CAST(timezone(tz, joined_ts_utc) AS DATE) AS joined_date
                FROM dim.dim_customer
            ), purchases AS (
                SELECT t.customer_id, t.amount,
                       CAST(timezone(c.tz, t.event_ts_utc) AS DATE) AS event_date
                FROM facts.fact_transactions t
                JOIN customers c USING (customer_id)
                WHERE t.txn_type = 'purchase'
            ), metrics AS (
                SELECT c.customer_id,
                       date_diff('day', max(p.event_date) FILTER (
                           WHERE p.event_date BETWEEN $snapshot::DATE - INTERVAL 29 DAY
                                                  AND $snapshot::DATE), $snapshot::DATE) AS recency,
                       count(p.event_date) FILTER (
                           WHERE p.event_date BETWEEN $snapshot::DATE - INTERVAL 89 DAY
                                                  AND $snapshot::DATE) AS frequency,
                       coalesce(sum(p.amount) FILTER (
                           WHERE p.event_date BETWEEN $snapshot::DATE - INTERVAL 89 DAY
                                                  AND $snapshot::DATE), 0) AS monetary
                FROM customers c LEFT JOIN purchases p USING (customer_id)
                WHERE c.joined_date <= $snapshot::DATE
                GROUP BY c.customer_id
            )
            SELECT customer_id, $snapshot::DATE AS snapshot_date, recency,
                   frequency::BIGINT AS frequency, monetary
            FROM metrics ORDER BY customer_id
            """,
            {"snapshot": snapshot_date},
        )
        con.execute(
            """
            CREATE TABLE fact_engagement_snapshot AS
            SELECT e.customer_id, $snapshot::DATE AS snapshot_date,
                   count(*) FILTER (WHERE event_type = 'app_open')::BIGINT AS app_open_count,
                   count(*) FILTER (WHERE event_type = 'repayment')::BIGINT AS repayment_count
            FROM facts.fact_events e
            JOIN dim.dim_customer c USING (customer_id)
            WHERE CAST(timezone(c.tz, e.event_ts_utc) AS DATE) <= $snapshot::DATE
            GROUP BY e.customer_id ORDER BY e.customer_id
            """,
            {"snapshot": snapshot_date},
        )


def _build_hub(path: Path, built_at: datetime) -> None:
    with duckdb.connect(str(path)) as con:
        _configure(con)
        con.execute(
            "CREATE TABLE connections(alias VARCHAR PRIMARY KEY, file_name VARCHAR NOT NULL)"
        )
        con.executemany(
            "INSERT INTO connections VALUES (?, ?)",
            [(name, f"{name}.duckdb") for name in DATABASES],
        )
        con.execute(
            "CREATE MACRO connection_files() AS TABLE "
            "SELECT alias, file_name FROM connections ORDER BY alias"
        )
        con.execute("CREATE TABLE build_metadata(built_at TIMESTAMPTZ)")
        con.execute("INSERT INTO build_metadata VALUES (?)", [built_at])


def _write_connect_sql(path: Path) -> None:
    path.write_text(
        "-- Run from the warehouse output directory.\n"
        "ATTACH 'dim.duckdb' AS dim (READ_ONLY);\n"
        "ATTACH 'facts.duckdb' AS facts (READ_ONLY);\n"
        "ATTACH 'gold.duckdb' AS gold (READ_ONLY);\n"
    )


def check_integrity(out_dir: str | Path) -> None:
    """Reject null or orphan customer IDs in every fact-like warehouse table."""
    with connect(out_dir) as con:
        tables = [
            ("facts", "fact_transactions"),
            ("facts", "fact_events"),
            ("gold", "fact_rfm_snapshot"),
            ("gold", "fact_engagement_snapshot"),
        ]
        for database, table in tables:
            nulls = con.execute(
                f"SELECT count(*) FROM {database}.{table} WHERE customer_id IS NULL"
            ).fetchone()[0]
            orphans = con.execute(
                f"SELECT count(*) FROM (SELECT customer_id FROM {database}.{table} "
                "EXCEPT SELECT customer_id FROM dim.dim_customer)"
            ).fetchone()[0]
            if nulls or orphans:
                raise ValueError(
                    f"{database}.{table} failed customer integrity: "
                    f"{nulls} null IDs, {orphans} orphan IDs"
                )


def build(parquet_dir: str | Path, out_dir: str | Path) -> None:
    parquet_root = Path(parquet_dir).expanduser().resolve()
    output_root = Path(out_dir).expanduser().resolve()
    manifest = _read_manifest(parquet_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".ccc-duckdb-", dir=output_root))
    try:
        dim_path = stage / "dim.duckdb"
        facts_path = stage / "facts.duckdb"
        gold_path = stage / "gold.duckdb"
        _build_dim(dim_path, parquet_root)
        _build_facts(facts_path, parquet_root)
        dates = manifest.get("event_date_utc_range") or []
        if len(dates) != 2:
            raise ValueError("manifest event_date_utc_range must contain two dates")
        _build_gold(gold_path, dim_path, facts_path, dates[1])
        _build_hub(stage / "hub.duckdb", datetime.now(UTC))
        _write_connect_sql(stage / "connect.sql")
        check_integrity(stage)
        for file_name in OUTPUT_FILES:
            os.replace(stage / file_name, output_root / file_name)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m ccc.duck.build")
    parser.add_argument("--parquet-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("~/data/ccc-duckdb"))
    args = parser.parse_args(argv)
    build(args.parquet_dir, args.out_dir)


if __name__ == "__main__":
    main()
