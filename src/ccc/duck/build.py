"""Build connected DuckDB files from a synthetic Parquet export."""

import argparse
import json
import os
import re
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import duckdb

from ccc.duck import DATABASES, _sql_string

FACT_TABLES = ("fact_transactions", "fact_events")
_PUBLISHED_BUILD_NAME = re.compile(r"build-\d{8}T\d{12}Z-[0-9a-f]{8}\Z")


def _escape_glob_path(path: Path) -> str:
    replacements = {"[": "[[]", "]": "[]]", "*": "[*]", "?": "[?]"}
    return "".join(replacements.get(character, character) for character in str(path))


def _parquet_source(root: Path, table: str) -> str:
    path = root / table
    if table in FACT_TABLES:
        pattern = f"{_escape_glob_path(path)}/*/*.parquet"
    else:
        pattern = _escape_glob_path(root / f"{table}.parquet")
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
    required_counts = {"dim_customer", *FACT_TABLES}
    if missing := required_counts.difference(manifest.get("row_counts", {})):
        raise ValueError(f"manifest row_counts missing tables: {sorted(missing)}")
    if manifest.get("include_truth") or "sim_truth_customer" in manifest.get("row_counts", {}):
        raise ValueError("sim_truth_customer exports are refused")
    if (parquet_dir / "sim_truth_customer.parquet").exists() or (
        parquet_dir / "sim_truth_customer"
    ).exists():
        raise ValueError("sim_truth_customer must never be loaded")
    return manifest


def _configure(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("SET TimeZone = 'UTC'")


def _assert_utc_timestamps(con: duckdb.DuckDBPyConnection, source: str, table: str) -> None:
    schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{source}')").fetchall()
    invalid = [
        name
        for name, data_type, *_ in schema
        if name.endswith("_ts_utc") and data_type != "TIMESTAMP WITH TIME ZONE"
    ]
    if invalid:
        raise ValueError(f"{table} UTC timestamp columns must be TIMESTAMPTZ: {invalid}")


def _build_dim(path: Path, parquet_dir: Path) -> None:
    with duckdb.connect(str(path)) as con:
        _configure(con)
        source = _parquet_source(parquet_dir, "dim_customer")
        schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{source}')").fetchall()
        columns = {row[0] for row in schema}
        forbidden = {
            name for name in columns if "local" in name.lower() or name.startswith("sim_truth_")
        }
        if forbidden:
            raise ValueError(f"dim_customer contains forbidden columns: {sorted(forbidden)}")
        _assert_utc_timestamps(con, source, "dim_customer")
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
            _assert_utc_timestamps(con, source, table)
            tie_breaker = "transaction_id" if table == "fact_transactions" else "event_id"
            con.execute(
                f"CREATE TABLE {table} AS SELECT * FROM read_parquet('{source}') "
                f"ORDER BY event_ts_utc, {tie_breaker}"
            )


def _build_gold(path: Path, dim_path: Path, facts_path: Path, snapshot_date: str) -> None:
    """Build gold for a customer-local calendar date, matching ``gold.rfm_snapshot``.

    Each zone's cutoff is its next local midnight (exclusive). When ``snapshot_date`` is
    the manifest's maximum UTC date, events late in that UTC date may be on the next date
    east of UTC and are intentionally excluded.
    """
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
                       CAST(timezone(c.tz, t.event_ts_utc) AS DATE) AS event_date,
                       t.event_ts_utc,
                       timezone(c.tz, ($snapshot::DATE + INTERVAL 1 DAY)::TIMESTAMP) AS cutoff_utc
                FROM facts.fact_transactions t
                JOIN customers c USING (customer_id)
                WHERE t.txn_type = 'purchase' AND t.event_ts_utc < cutoff_utc
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
              AND e.event_ts_utc < timezone(
                  c.tz, ($snapshot::DATE + INTERVAL 1 DAY)::TIMESTAMP
              )
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


def _write_connect_sql(path: Path, build_id: str) -> None:
    build_relative = Path(build_id)
    path.write_text(
        "-- Run from the warehouse output directory.\n"
        "-- Every ATTACH pins this immutable published build, so swapping the\n"
        "-- current symlink cannot mix files from different generations.\n"
        "SET TimeZone = 'UTC';\n"
        f"ATTACH '{_sql_string((build_relative / 'dim.duckdb').as_posix())}' "
        "AS dim (READ_ONLY);\n"
        f"ATTACH '{_sql_string((build_relative / 'facts.duckdb').as_posix())}' "
        "AS facts (READ_ONLY);\n"
        f"ATTACH '{_sql_string((build_relative / 'gold.duckdb').as_posix())}' "
        "AS gold (READ_ONLY);\n"
    )


def _check_integrity_dir(root: Path, expected_counts: dict[str, int] | None = None) -> None:
    with duckdb.connect(str(root / "hub.duckdb"), read_only=True) as con:
        _configure(con)
        for alias in DATABASES:
            con.execute(f"ATTACH '{_sql_string(root / f'{alias}.duckdb')}' AS {alias} (READ_ONLY)")
        tables = (
            ("dim", "dim_customer"),
            ("facts", "fact_transactions"),
            ("facts", "fact_events"),
            ("gold", "fact_rfm_snapshot"),
            ("gold", "fact_engagement_snapshot"),
        )
        for database, table in tables:
            count = con.execute(f"SELECT count(*) FROM {database}.{table}").fetchone()[0]
            if not count:
                raise ValueError(f"{database}.{table} must not be empty")
            if expected_counts is not None and table in expected_counts:
                expected = expected_counts[table]
                if count != expected:
                    raise ValueError(
                        f"{table} row count {count} does not match manifest {expected}"
                    )
            nulls = con.execute(
                f"SELECT count(*) FROM {database}.{table} WHERE customer_id IS NULL"
            ).fetchone()[0]
            orphans = 0
            if database != "dim":
                orphans = con.execute(
                    f"SELECT count(*) FROM (SELECT customer_id FROM {database}.{table} "
                    "EXCEPT SELECT customer_id FROM dim.dim_customer)"
                ).fetchone()[0]
            if nulls or orphans:
                raise ValueError(
                    f"{database}.{table} failed customer integrity: "
                    f"{nulls} null IDs, {orphans} orphan IDs"
                )


def check_integrity(out_dir: str | Path) -> None:
    """Reject empty tables and null or orphan customer IDs in the current build."""
    root = (Path(out_dir).expanduser().resolve() / "current").resolve(strict=True)
    _check_integrity_dir(root)


def build(
    parquet_dir: str | Path, out_dir: str | Path, *, snapshot_date: str | None = None
) -> None:
    parquet_root = Path(parquet_dir).expanduser().resolve()
    output_root = Path(out_dir).expanduser().resolve()
    manifest = _read_manifest(parquet_root)
    output_root.mkdir(parents=True, exist_ok=True)
    current = output_root / "current"
    if current.exists() and not current.is_symlink():
        raise ValueError(f"warehouse current must be a symlink, not a directory: {current}")
    if legacy := sorted(output_root.glob("*.duckdb")):
        raise ValueError(f"legacy flat DuckDB files must be moved before building: {legacy}")
    for abandoned in output_root.glob(".stage-*"):
        if (
            abandoned.is_dir()
            and not abandoned.is_symlink()
            and time.time() - abandoned.stat().st_mtime >= 3600
        ):
            shutil.rmtree(abandoned)
    build_id = f"build-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}"
    stage = output_root / f".stage-{build_id}"
    stage.mkdir()
    try:
        dim_path = stage / "dim.duckdb"
        facts_path = stage / "facts.duckdb"
        gold_path = stage / "gold.duckdb"
        _build_dim(dim_path, parquet_root)
        _build_facts(facts_path, parquet_root)
        dates = manifest.get("event_date_utc_range") or []
        if len(dates) != 2:
            raise ValueError("manifest event_date_utc_range must contain two dates")
        _build_gold(gold_path, dim_path, facts_path, snapshot_date or dates[1])
        _build_hub(stage / "hub.duckdb", datetime.now(UTC))
        _write_connect_sql(stage / "connect.sql", build_id)
        _check_integrity_dir(stage, manifest.get("row_counts", {}))
        for wal in stage.rglob("*.wal"):
            wal.unlink()
        final = output_root / build_id
        os.replace(stage, final)
        pending_link = output_root / f".current-{uuid4().hex}"
        try:
            pending_link.symlink_to(build_id, target_is_directory=True)
            os.replace(pending_link, output_root / "current")
        finally:
            pending_link.unlink(missing_ok=True)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _published_builds(root: Path) -> list[Path]:
    builds = []
    for path in root.glob("build-*"):
        if path.is_symlink() or not path.is_dir():
            continue
        if _PUBLISHED_BUILD_NAME.fullmatch(path.name) is None:
            continue
        builds.append(path)
    return sorted(builds, key=lambda path: path.name)


def prune_builds(
    out_dir: str | Path,
    *,
    keep_last: int = 1,
    acknowledge_readers_stopped: bool = False,
) -> list[Path]:
    """Explicitly delete older published ``build-*`` directories.

    Ordinary :func:`build` never calls this function: a reader may still have an
    older build open even after ``current`` moves on. Only run it after stopping
    every consumer of candidate builds and passing
    ``acknowledge_readers_stopped=True``. The ``current`` target, ``.stage-*``
    directories, symlinks, malformed ``build-*`` names and unrelated files are
    preserved. This cannot detect open file handles itself, so the operator
    acknowledgement is a required precondition, not a safety check.
    """
    if acknowledge_readers_stopped is not True:
        raise ValueError(
            "pruning requires acknowledge_readers_stopped=True after stopping all readers"
        )
    if isinstance(keep_last, bool) or not isinstance(keep_last, int) or keep_last < 1:
        raise ValueError("keep_last must be a positive integer")
    root = Path(out_dir).expanduser().resolve()
    current = root / "current"
    if not current.is_symlink():
        raise ValueError(f"warehouse current must be a symlink before pruning: {current}")
    try:
        current_target = current.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"warehouse current target is missing: {current}") from exc
    if current_target.parent != root:
        raise ValueError("warehouse current must point at a build inside the output root")
    builds = _published_builds(root)
    protected = set(builds[-keep_last:]) | {current_target}
    removed = []
    for path in builds:
        if path in protected:
            continue
        shutil.rmtree(path)
        removed.append(path)
    return removed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m ccc.duck.build")
    parser.add_argument("--parquet-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("~/data/ccc-duckdb"))
    parser.add_argument("--snapshot-date", help="customer-local gold snapshot date (YYYY-MM-DD)")
    parser.add_argument(
        "--prune-builds",
        action="store_true",
        help="delete older published builds after an explicit reader-stop acknowledgement",
    )
    parser.add_argument(
        "--keep-last",
        type=int,
        default=1,
        help="number of most recent published builds to keep when pruning",
    )
    parser.add_argument(
        "--acknowledge-readers-stopped",
        action="store_true",
        help="assert every consumer of candidate older builds has stopped",
    )
    args = parser.parse_args(argv)
    if args.prune_builds:
        removed = prune_builds(
            args.out_dir,
            keep_last=args.keep_last,
            acknowledge_readers_stopped=args.acknowledge_readers_stopped,
        )
        for path in removed:
            print(f"removed {path}")
        return
    if args.parquet_dir is None:
        parser.error("--parquet-dir is required unless --prune-builds is used")
    build(args.parquet_dir, args.out_dir, snapshot_date=args.snapshot_date)


if __name__ == "__main__":
    main()
