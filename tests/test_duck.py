import json
from datetime import UTC, datetime

import duckdb
import polars as pl
import pytest

from ccc.duck import connect
from ccc.duck.build import build, check_integrity


@pytest.fixture
def parquet_export(tmp_path):
    root = tmp_path / "parquet"
    root.mkdir()
    dim = pl.DataFrame(
        {
            "customer_id": ["c1", "c2"],
            "city": ["New York", "Los Angeles"],
            "tz": ["America/New_York", "America/Los_Angeles"],
            "joined_ts_utc": [datetime(2026, 1, 1, tzinfo=UTC)] * 2,
            "loaded_at": [datetime(2026, 1, 4, tzinfo=UTC)] * 2,
        }
    )
    dim.write_parquet(root / "dim_customer.parquet")
    schemas = {
        "fact_transactions": pl.DataFrame(
            {
                "transaction_id": ["t2", "t1"],
                "customer_id": ["c2", "c1"],
                "event_ts_utc": [
                    datetime(2026, 1, 3, 12, tzinfo=UTC),
                    datetime(2026, 1, 2, 12, tzinfo=UTC),
                ],
                "event_date_utc": [datetime(2026, 1, 3).date(), datetime(2026, 1, 2).date()],
                "txn_type": ["purchase", "purchase"],
                "amount": [20.0, 10.0],
                "loaded_at": [datetime(2026, 1, 4, tzinfo=UTC)] * 2,
            }
        ),
        "fact_events": pl.DataFrame(
            {
                "event_id": ["e1", "e2"],
                "customer_id": ["c1", "c2"],
                "event_ts_utc": [
                    datetime(2026, 1, 2, 13, tzinfo=UTC),
                    datetime(2026, 1, 3, 13, tzinfo=UTC),
                ],
                "event_date_utc": [datetime(2026, 1, 2).date(), datetime(2026, 1, 3).date()],
                "event_type": ["app_open", "repayment"],
                "loaded_at": [datetime(2026, 1, 4, tzinfo=UTC)] * 2,
            }
        ),
    }
    for name, frame in schemas.items():
        for day in frame["event_date_utc"].unique():
            folder = root / name / f"event_date_utc={day}"
            folder.mkdir(parents=True)
            frame.filter(pl.col("event_date_utc") == day).write_parquet(folder / "data.parquet")
    manifest = {
        "data_label": "synthetic",
        "timezone": "UTC",
        "include_truth": False,
        "row_counts": {"dim_customer": 2, "fact_transactions": 2, "fact_events": 2},
        "event_date_utc_range": ["2026-01-02", "2026-01-03"],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


@pytest.fixture
def warehouse(parquet_export, tmp_path):
    out = tmp_path / "warehouse"
    build(parquet_export, out)
    return out


def test_cross_file_join_and_order(warehouse):
    with connect(warehouse) as con:
        rows = con.execute(
            "SELECT t.customer_id, c.city FROM facts.fact_transactions t "
            "JOIN dim.dim_customer c USING (customer_id) ORDER BY t.event_ts_utc"
        ).fetchall()
        assert rows == [("c1", "New York"), ("c2", "Los Angeles")]
        assert con.execute("SELECT count(*) FROM gold.fact_rfm_snapshot").fetchone() == (2,)


def test_orphan_customer_fails_integrity(warehouse):
    with duckdb.connect(str(warehouse / "facts.duckdb")) as con:
        con.execute(
            "UPDATE fact_transactions SET customer_id = 'orphan' WHERE transaction_id = 't1'"
        )
    with pytest.raises(ValueError, match="1 orphan IDs"):
        check_integrity(warehouse)


def test_read_only_attach_rejects_writes(warehouse):
    with connect(warehouse, read_only=True) as con:
        with pytest.raises(duckdb.Error):
            con.execute("DELETE FROM facts.fact_transactions")


def test_rebuild_is_idempotent(parquet_export, warehouse):
    build(parquet_export, warehouse)
    build(parquet_export, warehouse)
    with connect(warehouse) as con:
        assert con.execute("SELECT count(*) FROM facts.fact_transactions").fetchone() == (2,)
        assert con.execute("SELECT count(*) FROM hub.connections").fetchone() == (3,)


def test_truth_export_is_refused(parquet_export, tmp_path):
    manifest_path = parquet_export / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["include_truth"] = True
    manifest["row_counts"]["sim_truth_customer"] = 2
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="sim_truth_customer exports are refused"):
        build(parquet_export, tmp_path / "warehouse")
