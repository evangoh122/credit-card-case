import importlib
import json
import os
import time
from datetime import UTC, date, datetime, timedelta
from datetime import time as wall_time
from pathlib import Path

import duckdb
import polars as pl
import pytest

from ccc.duck import connect
from ccc.duck.build import build, check_integrity, main, prune_builds
from ccc.gold import engagement_features, rfm_snapshot
from ccc.timeutil import local_to_utc, snapshot_cutoff_utc

GOLD_WINDOWS = {"recency": 30, "frequency": 90, "monetary": 90}


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


def _write_boundary_export(root: Path, snapshot: date) -> None:
    root.mkdir()
    zones = {"la": "America/Los_Angeles", "ny": "America/New_York"}
    dim_customers, tz_names, txn_ids, txn_customers, txn_times, txn_amounts = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    event_ids, event_customers, event_types, event_times = [], [], [], []
    for customer_id, tz in zones.items():
        dim_customers.append(customer_id)
        tz_names.append(tz)
        just_before = local_to_utc(datetime.combine(snapshot, wall_time(23, 59, 59, 500000)), tz)
        at_midnight = snapshot_cutoff_utc(snapshot, tz)
        after_midnight = at_midnight + timedelta(microseconds=500000)
        for index, (instant, amount) in enumerate(
            ((just_before, 10.0), (at_midnight, 11.0), (after_midnight, 12.0))
        ):
            txn_ids.append(f"{customer_id}-txn-{index}")
            txn_customers.append(customer_id)
            txn_times.append(instant)
            txn_amounts.append(amount)
            event_ids.append(f"{customer_id}-open-{index}")
            event_customers.append(customer_id)
            event_types.append("app_open")
            event_times.append(instant)
        older = local_to_utc(datetime.combine(snapshot - timedelta(days=3), wall_time(12)), tz)
        event_ids.extend([f"{customer_id}-open-old", f"{customer_id}-repay"])
        event_customers.extend([customer_id, customer_id])
        event_types.extend(["app_open", "repayment"])
        event_times.extend([older, older])
    dim = pl.DataFrame(
        {
            "customer_id": dim_customers,
            "city": ["Los Angeles", "New York"],
            "tz": tz_names,
            "joined_ts_utc": [datetime(2025, 12, 1, 12, tzinfo=UTC)] * 2,
            "loaded_at": [datetime(2026, 12, 1, tzinfo=UTC)] * 2,
        }
    )
    dim.write_parquet(root / "dim_customer.parquet")
    tables = {
        "fact_transactions": pl.DataFrame(
            {
                "transaction_id": txn_ids,
                "customer_id": txn_customers,
                "event_ts_utc": txn_times,
                "event_date_utc": [instant.date() for instant in txn_times],
                "txn_type": ["purchase"] * len(txn_ids),
                "amount": txn_amounts,
                "loaded_at": [datetime(2026, 12, 1, tzinfo=UTC)] * len(txn_ids),
            }
        ),
        "fact_events": pl.DataFrame(
            {
                "event_id": event_ids,
                "customer_id": event_customers,
                "event_ts_utc": event_times,
                "event_date_utc": [instant.date() for instant in event_times],
                "event_type": event_types,
                "loaded_at": [datetime(2026, 12, 1, tzinfo=UTC)] * len(event_ids),
            }
        ),
    }
    for name, frame in tables.items():
        for day in frame["event_date_utc"].unique():
            folder = root / name / f"event_date_utc={day}"
            folder.mkdir(parents=True, exist_ok=True)
            frame.filter(pl.col("event_date_utc") == day).write_parquet(folder / "data.parquet")
    event_dates = tables["fact_events"]["event_date_utc"]
    manifest = {
        "data_label": "synthetic",
        "timezone": "UTC",
        "include_truth": False,
        "row_counts": {
            "dim_customer": dim.height,
            "fact_transactions": len(txn_ids),
            "fact_events": len(event_ids),
        },
        "event_date_utc_range": [
            str(event_dates.min()),
            str(event_dates.max()),
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


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
        # is inside the half-open local day ending at next local midnight, which
        # is the same cutoff ccc.gold.rfm_snapshot now applies.
        assert rfm == [("c1", 1, 1, 10.0), ("c2", 0, 3, 90.0)]
        engagement = con.execute(
            "SELECT customer_id, app_open_count, repayment_count FROM gold.fact_engagement_snapshot"
        ).fetchall()
        assert engagement == [("c1", 1, 0), ("c2", 1, 1)]


@pytest.mark.parametrize(
    "snapshot",
    [date(2026, 1, 3), date(2026, 11, 1), date(2026, 3, 8)],
    ids=["winter", "fall-back-dst", "spring-forward-dst"],
)
def test_duck_gold_matches_canonical_boundary_rfm_and_engagement(tmp_path, snapshot):
    parquet_dir = tmp_path / "parquet"
    out = tmp_path / "warehouse"
    _write_boundary_export(parquet_dir, snapshot)
    build(parquet_dir, out, snapshot_date=snapshot.isoformat())

    with connect(out) as con:
        duck_rfm = [
            (customer_id, str(day), recency, frequency, float(monetary))
            for customer_id, day, recency, frequency, monetary in con.execute(
                "SELECT customer_id, snapshot_date, recency, frequency, monetary "
                "FROM gold.fact_rfm_snapshot ORDER BY customer_id"
            ).fetchall()
        ]
        duck_engagement = con.execute(
            "SELECT customer_id, app_open_count, repayment_count "
            "FROM gold.fact_engagement_snapshot ORDER BY customer_id"
        ).fetchall()

    dim = pl.read_parquet(parquet_dir / "dim_customer.parquet")
    transactions = pl.read_parquet(parquet_dir / "fact_transactions" / "**/*.parquet")
    events = pl.read_parquet(parquet_dir / "fact_events" / "**/*.parquet")
    canonical_rfm = rfm_snapshot(dim, transactions, snapshot, GOLD_WINDOWS)
    canonical_rfm_rows = [
        (
            row["customer_id"],
            row["snapshot_date"].isoformat(),
            row["recency"],
            row["frequency"],
            float(row["monetary"]),
        )
        for row in canonical_rfm.select(
            "customer_id", "snapshot_date", "recency", "frequency", "monetary"
        ).iter_rows(named=True)
    ]
    canonical_engagement = [
        (row["customer_id"], row["app_open_count"], row["repayment_count"])
        for row in engagement_features(events, dim, snapshot).iter_rows(named=True)
    ]

    assert duck_rfm == canonical_rfm_rows
    assert duck_rfm == [
        ("la", snapshot.isoformat(), 0, 1, 10.0),
        ("ny", snapshot.isoformat(), 0, 1, 10.0),
    ]
    assert duck_engagement == canonical_engagement == [("la", 2, 1), ("ny", 2, 1)]


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


def _published(warehouse: Path) -> list[Path]:
    return sorted(
        (path for path in warehouse.glob("build-*") if path.is_dir() and not path.is_symlink()),
        key=lambda path: path.name,
    )


def test_normal_rebuild_retains_all_published_builds(parquet_export, warehouse):
    first = _current(warehouse)
    build(parquet_export, warehouse, snapshot_date="2026-01-03")
    second = _current(warehouse)
    build(parquet_export, warehouse, snapshot_date="2026-01-03")
    third = _current(warehouse)
    builds = _published(warehouse)
    assert len(builds) == 3
    assert first in builds and second in builds and third in builds
    assert (warehouse / "current").resolve() == third


def test_prune_builds_requires_reader_stop_acknowledgement(parquet_export, warehouse):
    before = _published(warehouse)
    with pytest.raises(ValueError, match="acknowledge_readers_stopped=True"):
        prune_builds(warehouse)
    assert _published(warehouse) == before


def test_prune_builds_preserves_current_stages_and_unrelated_paths(parquet_export, warehouse):
    build(parquet_export, warehouse, snapshot_date="2026-01-03")
    build(parquet_export, warehouse, snapshot_date="2026-01-03")
    builds = _published(warehouse)
    current = _current(warehouse)
    assert len(builds) == 3

    live_stage = warehouse / ".stage-live"
    live_stage.mkdir()
    stage_link = warehouse / ".stage-link"
    stage_link.symlink_to(live_stage.name, target_is_directory=True)
    unrelated = warehouse / "keep.txt"
    unrelated.write_text("mine")
    malformed = warehouse / "build-notes"
    malformed.mkdir()
    malformed_build = warehouse / "build-20260101T000000000000Z-nothex"
    malformed_build.mkdir()

    removed = prune_builds(warehouse, keep_last=2, acknowledge_readers_stopped=True)

    assert removed == [builds[0]]
    assert not builds[0].exists()
    assert builds[1].is_dir() and builds[2].is_dir()
    assert current == builds[2]
    assert _current(warehouse) == current
    assert live_stage.is_dir()
    assert stage_link.is_symlink()
    assert unrelated.read_text() == "mine"
    assert malformed.is_dir()
    assert malformed_build.is_dir()


def test_prune_builds_cli_requires_and_records_acknowledgement(parquet_export, warehouse):
    build(parquet_export, warehouse, snapshot_date="2026-01-03")
    builds = _published(warehouse)
    with pytest.raises(ValueError, match="acknowledge_readers_stopped=True"):
        main(["--out-dir", str(warehouse), "--prune-builds"])
    main(
        [
            "--out-dir",
            str(warehouse),
            "--prune-builds",
            "--acknowledge-readers-stopped",
        ]
    )
    assert not builds[0].exists()
    assert builds[1].is_dir()
    assert _current(warehouse) == builds[1]


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
