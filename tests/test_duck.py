import importlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import polars as pl
import pytest

from ccc.duck import connect
from ccc.duck.build import build, check_integrity


def _write_export(root: Path) -> None:
    root.mkdir()
    dim = pl.DataFrame(
        {
            "customer_id": ["c1", "c2"],
            "city": ["New York", "Los Angeles"],
            "tz": ["America/New_York", "America/Los_Angeles"],
            "joined_ts_utc": [datetime(2026, 1, 1, tzinfo=UTC)] * 2,
            "loaded_at": [datetime(2026, 1, 5, tzinfo=UTC)] * 2,
        }
    )
    dim.write_parquet(root / "dim_customer.parquet")
    tables = {
        "fact_transactions": pl.DataFrame(
            {
                "transaction_id": ["t2", "t1", "t4", "t3", "t5"],
                "customer_id": ["c2", "c1", "c2", "c1", "c2"],
                "event_ts_utc": [
                    datetime(2026, 1, 3, 12, tzinfo=UTC),
                    datetime(2026, 1, 2, 12, tzinfo=UTC),
                    datetime(2026, 1, 4, 5, 30, tzinfo=UTC),
                    datetime(2026, 1, 4, 5, 30, tzinfo=UTC),
                    datetime(2026, 1, 4, 7, 59, 59, 500000, tzinfo=UTC),
                ],
                "event_date_utc": [
                    datetime(2026, 1, 3).date(),
                    datetime(2026, 1, 2).date(),
                    datetime(2026, 1, 4).date(),
                    datetime(2026, 1, 4).date(),
                    datetime(2026, 1, 4).date(),
                ],
                "txn_type": ["purchase"] * 5,
                "amount": [20.0, 10.0, 30.0, 99.0, 40.0],
                "loaded_at": [datetime(2026, 1, 5, tzinfo=UTC)] * 5,
            }
        ),
        "fact_events": pl.DataFrame(
            {
                "event_id": ["e2", "e1", "e3"],
                "customer_id": ["c2", "c1", "c2"],
                "event_ts_utc": [
                    datetime(2026, 1, 3, 13, tzinfo=UTC),
                    datetime(2026, 1, 2, 13, tzinfo=UTC),
                    datetime(2026, 1, 4, 7, 59, 59, 500000, tzinfo=UTC),
                ],
                "event_date_utc": [
                    datetime(2026, 1, 3).date(),
                    datetime(2026, 1, 2).date(),
                    datetime(2026, 1, 4).date(),
                ],
                "event_type": ["repayment", "app_open", "app_open"],
                "loaded_at": [datetime(2026, 1, 5, tzinfo=UTC)] * 3,
            }
        ),
    }
    for name, frame in tables.items():
        for day in frame["event_date_utc"].unique():
            folder = root / name / f"event_date_utc={day}"
            folder.mkdir(parents=True)
            frame.filter(pl.col("event_date_utc") == day).write_parquet(folder / "data.parquet")
    manifest = {
        "data_label": "synthetic",
        "timezone": "UTC",
        "include_truth": False,
        "row_counts": {"dim_customer": 2, "fact_transactions": 5, "fact_events": 3},
        "event_date_utc_range": ["2026-01-02", "2026-01-04"],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


@pytest.fixture
def parquet_export(tmp_path):
    root = tmp_path / "parquet[?]"
    _write_export(root)
    return root


@pytest.fixture
def warehouse(parquet_export, tmp_path):
    out = tmp_path / "warehouse"
    build(parquet_export, out, snapshot_date="2026-01-03")
    return out


def _current(out: Path) -> Path:
    return (out / "current").resolve(strict=True)


def _all_data(out: Path) -> dict[str, list[tuple]]:
    with connect(out) as con:
        return {
            table: con.execute(f"SELECT * FROM {database}.{table}").fetchall()
            for database, table in (
                ("dim", "dim_customer"),
                ("facts", "fact_transactions"),
                ("facts", "fact_events"),
                ("gold", "fact_rfm_snapshot"),
                ("gold", "fact_engagement_snapshot"),
            )
        }


def test_cross_file_join_stored_order_and_exact_rfm(warehouse):
    with connect(warehouse) as con:
        stored = con.execute("SELECT transaction_id FROM facts.fact_transactions").fetchall()
        assert stored == [("t1",), ("t2",), ("t3",), ("t4",), ("t5",)]
        events = con.execute("SELECT event_id FROM facts.fact_events").fetchall()
        assert events == [("e1",), ("e2",), ("e3",)]
        joined = con.execute(
            "SELECT count(*) FROM facts.fact_transactions t "
            "JOIN dim.dim_customer c USING (customer_id)"
        ).fetchone()
        assert joined == (5,)
        rfm = con.execute(
            "SELECT customer_id, recency, frequency, monetary FROM gold.fact_rfm_snapshot"
        ).fetchall()
        # 05:30 UTC is Jan 4 in New York but Jan 3 in Los Angeles; 07:59:59.5
        # is still inside Los Angeles's half-open local-day cutoff.
        assert rfm == [("c1", 1, 1, 10.0), ("c2", 0, 3, 90.0)]
        engagement = con.execute(
            "SELECT customer_id, app_open_count, repayment_count FROM gold.fact_engagement_snapshot"
        ).fetchall()
        assert engagement == [("c1", 1, 0), ("c2", 1, 1)]


def test_connect_sql_runs_from_output_directory(warehouse, monkeypatch):
    monkeypatch.chdir(warehouse)
    sql = (warehouse / "current/connect.sql").read_text()
    with duckdb.connect() as con:
        con.execute(sql)
        assert con.execute(
            "SELECT count(*) FROM facts.fact_transactions JOIN dim.dim_customer USING (customer_id)"
        ).fetchone() == (5,)


def test_orphan_and_null_customer_fail_integrity(warehouse):
    facts = _current(warehouse) / "facts.duckdb"
    with duckdb.connect(str(facts)) as con:
        con.execute(
            "UPDATE fact_transactions SET customer_id = 'orphan' WHERE transaction_id = 't1'"
        )
    with pytest.raises(ValueError, match="1 orphan IDs"):
        check_integrity(warehouse)
    with duckdb.connect(str(facts)) as con:
        con.execute("UPDATE fact_transactions SET customer_id = NULL WHERE transaction_id = 't1'")
    with pytest.raises(ValueError, match="1 null IDs"):
        check_integrity(warehouse)


def test_empty_table_fails_integrity(warehouse):
    with duckdb.connect(str(_current(warehouse) / "facts.duckdb")) as con:
        con.execute("DELETE FROM fact_events")
    with pytest.raises(ValueError, match="facts.fact_events must not be empty"):
        check_integrity(warehouse)


def test_manifest_row_count_mismatch_fails(parquet_export, tmp_path):
    manifest_path = parquet_export / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["row_counts"]["fact_transactions"] = 6
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not match manifest"):
        build(parquet_export, tmp_path / "warehouse")


def test_read_only_attach_rejects_writes_and_sets_utc(warehouse):
    with connect(warehouse, read_only=True) as con:
        assert con.execute("SELECT current_setting('TimeZone')").fetchone() == ("UTC",)
        with pytest.raises(duckdb.Error):
            con.execute("DELETE FROM facts.fact_transactions")


def test_rebuild_is_idempotent_and_preserves_unrelated_files(parquet_export, warehouse):
    unrelated = warehouse / "keep.txt"
    unrelated.write_text("mine")
    before = _all_data(warehouse)
    build(parquet_export, warehouse, snapshot_date="2026-01-03")
    abandoned = warehouse / ".stage-killed"
    abandoned.mkdir()
    (abandoned / "partial.wal").write_text("partial")
    old = time.time() - 3700
    os.utime(abandoned, (old, old))
    stage_target = warehouse / "do-not-delete"
    stage_target.mkdir()
    stage_link = warehouse / ".stage-link"
    stage_link.symlink_to(stage_target.name, target_is_directory=True)
    young_stage = warehouse / ".stage-young"
    young_stage.mkdir()
    build(parquet_export, warehouse, snapshot_date="2026-01-03")
    assert _all_data(warehouse) == before
    assert unrelated.read_text() == "mine"
    assert not abandoned.exists()
    assert stage_link.is_symlink()
    assert stage_target.is_dir()
    assert young_stage.is_dir()
    assert not list(_current(warehouse).glob("*.wal"))


def test_failed_rebuild_keeps_current_and_cleans_stage(parquet_export, warehouse, monkeypatch):
    old_current = _current(warehouse)
    old_data = _all_data(warehouse)
    module = importlib.import_module("ccc.duck.build")

    def fail_gold(*args, **kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(module, "_build_gold", fail_gold)
    with pytest.raises(RuntimeError, match="injected failure"):
        module.build(parquet_export, warehouse, snapshot_date="2026-01-03")
    assert _current(warehouse) == old_current
    assert _all_data(warehouse) == old_data
    assert not list(warehouse.glob(".stage-*"))


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_truth_on_disk_is_refused(parquet_export, tmp_path, kind):
    truth = parquet_export / (
        "sim_truth_customer.parquet" if kind == "file" else "sim_truth_customer"
    )
    truth.touch() if kind == "file" else truth.mkdir()
    with pytest.raises(ValueError, match="sim_truth_customer must never be loaded"):
        build(parquet_export, tmp_path / "warehouse")


def test_truth_manifest_is_refused(parquet_export, tmp_path):
    manifest_path = parquet_export / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["include_truth"] = True
    manifest["row_counts"]["sim_truth_customer"] = 2
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="sim_truth_customer exports are refused"):
        build(parquet_export, tmp_path / "warehouse")


def test_non_utc_manifest_is_refused(parquet_export, tmp_path):
    manifest_path = parquet_export / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["timezone"] = "America/New_York"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="must use UTC"):
        build(parquet_export, tmp_path / "warehouse")


def test_local_time_dim_column_is_refused(parquet_export, tmp_path):
    path = parquet_export / "dim_customer.parquet"
    pl.read_parquet(path, glob=False).with_columns(
        pl.lit("noon").alias("joined_local_time")
    ).write_parquet(path)
    with pytest.raises(ValueError, match="forbidden columns"):
        build(parquet_export, tmp_path / "warehouse")


def test_naive_utc_timestamp_is_refused(parquet_export, tmp_path):
    path = parquet_export / "dim_customer.parquet"
    pl.read_parquet(path, glob=False).with_columns(
        pl.col("joined_ts_utc").dt.replace_time_zone(None)
    ).write_parquet(path)
    with pytest.raises(ValueError, match="must be TIMESTAMPTZ"):
        build(parquet_export, tmp_path / "warehouse")


def test_naive_fact_utc_timestamp_is_refused(parquet_export, tmp_path):
    for path in (parquet_export / "fact_events").rglob("*.parquet"):
        pl.read_parquet(path, glob=False).with_columns(
            pl.col("event_ts_utc").dt.replace_time_zone(None)
        ).write_parquet(path)
    with pytest.raises(ValueError, match="must be TIMESTAMPTZ"):
        build(parquet_export, tmp_path / "warehouse")


@pytest.mark.parametrize("legacy", ["current-directory", "flat-duckdb"])
def test_legacy_output_layout_is_rejected(parquet_export, tmp_path, legacy):
    out = tmp_path / "warehouse"
    out.mkdir()
    if legacy == "current-directory":
        (out / "current").mkdir()
        match = "current must be a symlink"
    else:
        (out / "facts.duckdb").touch()
        match = "legacy flat DuckDB files"
    with pytest.raises(ValueError, match=match):
        build(parquet_export, out)


def test_missing_hub_has_explicit_error(tmp_path):
    out = tmp_path / "warehouse"
    build_dir = out / "build-empty"
    build_dir.mkdir(parents=True)
    (out / "current").symlink_to(build_dir.name, target_is_directory=True)
    with pytest.raises(FileNotFoundError, match="warehouse hub is missing"):
        connect(out)
