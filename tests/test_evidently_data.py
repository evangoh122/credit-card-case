from datetime import UTC, date, datetime

import polars as pl
import pytest

from ccc import evidently_data
from ccc.gold import TRUTH_COLUMNS

START = date(2026, 1, 1)
TZ = "America/New_York"


def _tables():
    ids = ["old1", "old2", "mid", "new1", "new2"]
    joined = [
        datetime(2025, 11, 1, 12, tzinfo=UTC),
        datetime(2025, 12, 1, 12, tzinfo=UTC),
        datetime(2026, 1, 6, 12, tzinfo=UTC),
        datetime(2026, 1, 21, 12, tzinfo=UTC),
        datetime(2026, 1, 22, 12, tzinfo=UTC),
    ]
    dim = pl.DataFrame(
        {
            "customer_id": ids,
            "channel": ["web"] * 5,
            "region": ["east"] * 5,
            "tz": [TZ] * 5,
            "joined_ts_utc": joined,
        },
        schema_overrides={"joined_ts_utc": pl.Datetime("us", "UTC")},
    )
    txns = pl.DataFrame(
        {
            "customer_id": ["old1", "new1", "new1"],
            "txn_type": ["purchase"] * 3,
            "amount": [10.0, 5.0, 7.0],
            "event_ts_utc": [datetime(2026, 1, 25, 12, tzinfo=UTC)] * 3,
        }
    )
    events = pl.DataFrame(
        {
            "customer_id": ["old2"],
            "event_type": ["app_open"],
            "event_ts_utc": [datetime(2026, 1, 26, 12, tzinfo=UTC)],
        }
    )
    truth = pl.DataFrame({"customer_id": ids, "risk": [0.1] * 5})
    return {
        "dim_customer": dim,
        "fact_transactions": txns,
        "fact_events": events,
        "sim_truth_customer": truth,
    }


def test_reference_is_pre_start_and_current_is_late_joiners():
    ref, cur = evidently_data.drift_frames(_tables(), START, days=30, cohort_day=15)
    assert list(ref["frequency"]) == [1, 0]
    assert list(cur["frequency"]) == [2, 0]
    assert list(cur["monetary"]) == [12.0, 0.0]
    assert list(ref["app_open_count"]) == [0, 1]


def test_mid_joiners_belong_to_neither_frame():
    ref, cur = evidently_data.drift_frames(_tables(), START, days=30, cohort_day=15)
    assert len(ref) == 2 and len(cur) == 2


def test_frames_have_only_observable_features():
    for frame in evidently_data.drift_frames(_tables(), START, days=30, cohort_day=15):
        assert list(frame.columns) == list(evidently_data.FEATURES)
        assert not TRUTH_COLUMNS & set(frame.columns)


def test_truth_column_in_features_is_rejected(monkeypatch):
    monkeypatch.setattr(evidently_data, "FEATURES", (*evidently_data.FEATURES, "risk"))
    tables = _tables()
    tables["dim_customer"] = tables["dim_customer"].with_columns(risk=pl.lit(0.1))
    with pytest.raises(ValueError, match="truth columns"):
        evidently_data.drift_frames(tables, START, days=30, cohort_day=15)


def test_empty_current_raises():
    with pytest.raises(ValueError, match="cohort day 29"):
        evidently_data.drift_frames(_tables(), START, days=30, cohort_day=29)


def test_snapshot_written_with_drift_results(tmp_path):
    from evidently.ui.workspace import Workspace

    ref, cur = evidently_data.drift_frames(_tables(), START, days=30, cohort_day=15)
    project_id = evidently_data.write_drift_snapshot(ref, cur, tmp_path / "ws")
    ws = Workspace.create(str(tmp_path / "ws"))
    (run_id,) = ws.list_runs(project_id)
    text = ws.get_run(project_id, run_id).json()
    assert "DriftedColumnsCount" in text and "ValueDrift" in text
