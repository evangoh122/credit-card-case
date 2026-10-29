import json
from dataclasses import replace
from datetime import date, timedelta

import duckdb
import polars as pl
import pytest

from ccc.sim import export, profiles, run
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


def test_simulate_forwards_locale_for_unchunked_and_chunked():
    sc = Scenario("locale", START, initial_customers=8, arrivals_per_day=0.0)
    specs = run.cohort_specs(SEED, sc, 2)
    expected = profiles.identities(
        SEED, [(cid, join_day) for cid, _, join_day in specs], sc.start_date, "ja_JP"
    )
    expected_names = dict(zip(expected["customer_id"], expected["name"], strict=True))
    english = run.simulate(SEED, 2, sc)["dim_customer"]
    english_names = dict(zip(english["customer_id"], english["name"], strict=True))
    assert english_names != expected_names
    for batch_size in (None, 3):
        actual = run.simulate(SEED, 2, sc, locale="ja_JP", batch_size=batch_size)["dim_customer"]
        actual_names = dict(zip(actual["customer_id"], actual["name"], strict=True))
        assert actual_names == expected_names, batch_size


def test_zero_transaction_export_is_rejected_with_clear_error(tmp_path):
    sc = Scenario("empty", START, initial_customers=0, arrivals_per_day=0.0)
    assert run.cohort_specs(SEED, sc, 2) == []
    out = tmp_path / "empty"
    with pytest.raises(ValueError, match="produced zero transactions"):
        export.export(out, SEED, 2, sc, batch_size=10, workers=1)
    assert not out.exists() or not any(out.iterdir())
    assert not (out / ".staging").exists()


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


def _all_tables(out):
    tables = {name: _facts(out, name).sort(*keys) for name, keys in export.FACTS.items()}
    tables["dim_customer"] = pl.read_parquet(out / "dim_customer.parquet").sort("customer_id")
    return tables


def test_export_identical_across_workers_and_batch_sizes(out, sc, tmp_path):
    other = tmp_path / "other"
    export.export(other, SEED, DAYS, sc, batch_size=7, workers=1)
    base, again = _all_tables(out), _all_tables(other)
    for name, df in base.items():
        assert df.equals(again[name]), name


def test_more_initial_customers_keep_existing_rows(out, sc, tmp_path):
    bigger = tmp_path / "bigger"
    export.export(bigger, SEED, DAYS, replace(sc, initial_customers=90), batch_size=40, workers=1)
    base, more = _all_tables(out), _all_tables(bigger)
    ids = base["dim_customer"]["customer_id"].to_list()
    assert more["dim_customer"].height > base["dim_customer"].height
    for name, df in base.items():
        kept = more[name].filter(pl.col("customer_id").is_in(ids))
        assert kept.equals(df), name


@pytest.mark.parametrize("start", [date(2026, 3, 5), date(2026, 10, 30)])
def test_dst_transitions_keep_partitions_and_offsets_consistent(tmp_path, start):
    sc = Scenario("dst", start, initial_customers=150, arrivals_per_day=0.0)
    path = tmp_path / "ds"
    export.export(path, SEED, 6, sc, batch_size=100, workers=1)
    txns = _facts(path, "fact_transactions")
    dim = pl.read_parquet(path / "dim_customer.parquet").select("customer_id", "tz")
    assert (txns["event_ts_utc"].dt.date() == txns["event_date_utc"]).all()
    assert {f"event_date_utc={d}" for d in txns["event_date_utc"].unique()} == {
        p.name for p in (path / "fact_transactions").iterdir()
    }
    transitions = 0
    for tz, grp in txns.join(dim, on="customer_id").sort("event_ts_utc").group_by("tz"):
        local = grp["event_ts_utc"].dt.convert_time_zone(tz[0])
        offset = local.dt.replace_time_zone(None) - grp["event_ts_utc"].dt.replace_time_zone(None)
        hours = offset.dt.total_hours().to_list()
        assert len(set(hours)) <= 2
        changes = sum(a != b for a, b in zip(hours, hours[1:], strict=False))
        assert changes <= 1  # offset is monotone in time: at most one DST switch
        transitions += changes
        days_local = local.dt.date()
        assert days_local.min() >= start and days_local.max() < start + timedelta(days=6)
    assert transitions >= 1
    # shifting an offset moves events across the UTC day boundary, never across partitions
    hours_utc = txns.group_by(txns["event_ts_utc"].dt.hour().alias("h")).len()
    assert hours_utc["len"].sum() == txns.height


class _InlinePool:
    """Stand-in for ProcessPoolExecutor; `spawn` workers cannot see monkeypatches."""

    counts_delta = 0

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, jobs):
        for counts in map(fn, jobs):
            counts["fact_transactions"] += self.counts_delta
            yield counts


class _WorkerDies(_InlinePool):
    def map(self, fn, jobs):
        yield fn(next(iter(jobs)))
        raise RuntimeError("worker died")


class _CountsDrift(_InlinePool):
    counts_delta = 1


def _failing_connect(real_connect, made):
    class Con:
        def __init__(self):
            self.real = real_connect()
            self.closed = False
            made.append(self)

        def execute(self, sql, *args):
            if sql.startswith("COPY") and "fact_events" in sql:
                raise duckdb.Error("disk full")
            return self.real.execute(sql, *args)

        def close(self):
            self.closed = True
            self.real.close()

    return Con


@pytest.mark.parametrize("failure", ["worker", "duckdb", "counts"])
@pytest.mark.parametrize("preexisting_empty_dir", [False, True])
def test_failed_export_cleans_up_and_retry_succeeds(
    tmp_path, monkeypatch, sc, failure, preexisting_empty_dir
):
    out = tmp_path / "ds"
    if preexisting_empty_dir:
        out.mkdir()
    made = []
    if failure == "worker":
        monkeypatch.setattr(export, "ProcessPoolExecutor", _WorkerDies)
    elif failure == "counts":
        monkeypatch.setattr(export, "ProcessPoolExecutor", _CountsDrift)
    else:
        monkeypatch.setattr(export, "ProcessPoolExecutor", _InlinePool)
        monkeypatch.setattr(export.duckdb, "connect", _failing_connect(duckdb.connect, made))
    with pytest.raises((RuntimeError, duckdb.Error)):
        export.export(out, SEED, DAYS, sc, batch_size=40, workers=1)
    assert not out.exists() or not any(out.iterdir())
    assert all(con.closed for con in made)
    monkeypatch.undo()
    manifest = export.export(out, SEED, DAYS, sc, batch_size=40, workers=1)
    assert manifest["row_counts"]["fact_transactions"] > 0


def test_non_empty_output_dir_is_left_untouched(tmp_path, sc):
    out = tmp_path / "ds"
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    with pytest.raises(FileExistsError):
        export.export(out, SEED, DAYS, sc)
    assert (out / "keep.txt").read_text() == "mine"
