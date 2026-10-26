import json
from dataclasses import replace

import duckdb
import polars as pl
import pytest

from ccc.sim import export, run
from ccc.sim.scenario import Scenario
from tests.conftest import START

SEED = 5
DAYS = 12


@pytest.fixture(scope="module")
def sc():
    return Scenario("tiny", START, initial_customers=60, arrivals_per_day=3.0)


@pytest.fixture(scope="module")
def out(tmp_path_factory, sc):
    path = tmp_path_factory.mktemp("export") / "ds"
    export.export(path, SEED, DAYS, sc, batch_size=25, workers=2)
    return path


def _facts(out, name):
    return pl.read_parquet(out / name / "*" / "*.parquet")


def test_chunked_equals_unchunked(sc):
    whole = run.simulate(SEED, DAYS, sc)
    for size in (1, 7, 40):
        chunked = run.simulate(SEED, DAYS, sc, batch_size=size)
        for name, df in whole.items():
            assert df.equals(chunked[name]), (name, size)


def test_exported_facts_equal_unchunked_simulation(out, sc):
    whole = run.simulate(SEED, DAYS, sc)
    for name, keys in export.FACTS.items():
        got = _facts(out, name).select(whole[name].columns).sort(*keys)
        assert got.equals(whole[name].sort(*keys)), name


def test_manifest_counts_match_duckdb(out):
    manifest = json.loads((out / "manifest.json").read_text())
    counts = manifest["row_counts"]
    for name in export.FACTS:
        n = duckdb.sql(f"SELECT count(*) FROM read_parquet('{out / name}/*/*.parquet')").fetchall()
        assert counts[name] == n[0][0] > 0
    n = duckdb.sql(f"SELECT count(*) FROM read_parquet('{out / 'dim_customer.parquet'}')")
    assert counts["dim_customer"] == n.fetchall()[0][0] == manifest["customers"]
    assert manifest["seed"] == SEED
    assert manifest["data_label"] == "synthetic"
    assert {"python", "polars", "duckdb"} <= set(manifest["versions"])


def test_truth_excluded_by_default(out, tmp_path, sc):
    assert not (out / "sim_truth_customer.parquet").exists()
    assert "sim_truth_customer" not in json.loads((out / "manifest.json").read_text())["row_counts"]
    assert not list(out.rglob("*truth*"))
    with_truth = tmp_path / "t"
    export.export(with_truth, SEED, DAYS, sc, batch_size=100, workers=1, include_truth=True)
    truth = pl.read_parquet(with_truth / "sim_truth_customer.parquet")
    assert {"risk", "spend_rate"} <= set(truth.columns)


def test_partition_layout_and_utc_dates(out):
    for name in export.FACTS:
        dirs = sorted(p.name for p in (out / name).iterdir())
        assert dirs and all(d.startswith("event_date_utc=") for d in dirs)
        df = _facts(out, name)
        assert df["event_ts_utc"].dtype == pl.Datetime("us", "UTC")
        assert df["event_date_utc"].dtype == pl.Date
        assert (df["event_ts_utc"].dt.date() == df["event_date_utc"]).all()
        assert {f"event_date_utc={d}" for d in df["event_date_utc"].unique()} == set(dirs)
        assert set(df["data_label"]) == {"synthetic"}
    assert not (out / ".staging").exists()


def test_refuses_non_empty_output_dir(out, sc):
    with pytest.raises(FileExistsError):
        export.export(out, SEED, DAYS, sc)


def test_target_estimate_lands_near_target(tmp_path):
    sc = Scenario("tiny", START, arrivals_per_day=2.0)
    target = 30_000
    n = export.estimate_initial_customers(SEED, sc, 10, target)
    txns = run.simulate(SEED, 10, replace(sc, initial_customers=n), batch_size=500)
    assert abs(txns["fact_transactions"].height - target) < 0.1 * target


def test_upload_refuses_truth_file(tmp_path):
    from ccc.sim import upload

    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / upload.TRUTH_FILE).write_bytes(b"")
    with pytest.raises(ValueError, match="planted truth"):
        upload.upload(tmp_path, "u/r", "token")
