import polars as pl
import pytest

from ccc import evidently_data


def _tables():
    n = 6
    dim = pl.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(n)],
            "channel": ["web"] * n,
            "region": ["east"] * n,
        }
    )
    truth = pl.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(n)],
            "cohort_day": [0, 0, 0, 5, 40, 41],
            "risk": [0.1, 0.2, 0.3, 0.4, 0.9, 0.8],
            "spend_rate": [1.0] * n,
            "responsiveness": [0.5] * n,
            "utilisation": [0.2] * n,
        }
    )
    return {"dim_customer": dim, "sim_truth_customer": truth}


def test_reference_is_initial_and_current_is_late_cohorts():
    ref, cur = evidently_data.drift_frames(_tables(), cohort_day=30)
    assert list(ref["risk"]) == [0.1, 0.2, 0.3]
    assert list(cur["risk"]) == [0.9, 0.8]
    assert list(ref.columns) == list(evidently_data.FEATURES)
    assert "cohort_day" not in cur.columns


def test_mid_cohorts_belong_to_neither_frame():
    ref, cur = evidently_data.drift_frames(_tables(), cohort_day=30)
    assert 0.4 not in set(ref["risk"]) | set(cur["risk"])


def test_empty_current_raises():
    with pytest.raises(ValueError, match="cohort day >= 100"):
        evidently_data.drift_frames(_tables(), cohort_day=100)


def test_snapshot_written_to_workspace(tmp_path):
    ref, cur = evidently_data.drift_frames(_tables(), cohort_day=30)
    evidently_data.write_drift_snapshot(ref, cur, tmp_path / "ws")
    assert list((tmp_path / "ws").rglob("*.json"))
