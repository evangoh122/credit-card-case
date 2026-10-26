import socket
from datetime import date

import mlflow
import pytest
from mlflow import MlflowClient

from ccc import run_study, tracking
from ccc.sim import run
from ccc.sim.scenario import Campaign, Scenario, Segment

SEEDS = [101, 102, 103]
DAYS = 12
START = date(2026, 10, 19)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    sc = Scenario(
        name="recovery",
        start_date=START,
        initial_customers=800,
        arrivals_per_day=0.0,
        arrivals_trend=0.0,
        campaigns=(
            Campaign("real", "2026-10-20T09:00:00", 10, 0.5, segment=Segment(risk_max=0.8)),
            Campaign("nothing", "2026-10-20T09:00:00", 10, 0.0),
        ),
    )
    result = tracking.run_simulation_study(
        sc,
        SEEDS,
        DAYS,
        tracking_dir=tmp_path_factory.mktemp("shared"),
        locale="en_GB",
        alpha=0.01,
        min_recovery=0.6,
    )
    return result, None


def study(sc, tmp_path, name="store", seeds=SEEDS, **kw):
    return tracking.run_simulation_study(sc, seeds, DAYS, tracking_dir=tmp_path / name, **kw)


def children(result):
    return MlflowClient(result["tracking_uri"]).search_runs(
        [result["experiment_id"]],
        filter_string=f"tags.mlflow.parentRunId = '{result['parent_run_id']}'",
    )


def test_parent_and_child_runs(shared):
    result, _ = shared
    client = MlflowClient(result["tracking_uri"])
    parent = client.get_run(result["parent_run_id"])
    kids = children(result)
    assert len(kids) == len(SEEDS)
    assert {k.data.params["root_seed"] for k in kids} == {str(s) for s in SEEDS}
    assert parent.data.params["scenario_config"] == "recovery"
    assert "recovery.real.share" in parent.data.metrics
    assert "recovery_summary.json" in {a.path for a in client.list_artifacts(parent.info.run_id)}
    kid = kids[0]
    assert kid.data.metrics["rows.dim_customer"] == 800
    assert "null_rate.fact_transactions.amount" in kid.data.metrics
    assert "psi.max" in kid.data.metrics
    assert "lift.real.estimated" in kid.data.metrics
    names = {a.path for a in client.list_artifacts(kid.info.run_id)}
    assert names == {f"seed_{kid.data.params['root_seed']}"}
    inner = {a.path for a in client.list_artifacts(kid.info.run_id, names.copy().pop())}
    assert any(p.endswith("drift_report.json") for p in inner)
    assert any(p.endswith("data_dictionary.json") for p in inner)


def test_params_and_tags(shared):
    result, _ = shared
    client = MlflowClient(result["tracking_uri"])
    for r in [client.get_run(result["parent_run_id"]), *children(result)]:
        assert r.data.tags["data_label"] == "synthetic"
        assert len(r.data.tags["git_sha"]) == 40 or r.data.tags["git_sha"] == "unknown"
        assert r.data.tags["git_dirty"] in ("true", "false", "unknown")
        for lib in ("faker", "sdv", "mlflow", "polars"):
            assert r.data.tags[f"{lib}_version"]
        assert r.data.params["faker_locale"] == "en_GB"
        assert r.data.params["customer_count"] == "800"
        assert r.data.params["cohort_day"] == str(DAYS // 2)


def test_data_hash_reproducible(recovery_scenario, tmp_path):
    a = study(recovery_scenario, tmp_path, "a", seeds=[101])
    b = study(recovery_scenario, tmp_path, "b", seeds=[101])
    c = study(recovery_scenario, tmp_path, "c", seeds=[102])
    assert a["data_hashes"] == b["data_hashes"]
    assert a["data_hashes"][101] != c["data_hashes"][102]
    kid = children(b)[0]
    assert (
        kid.data.params["data_hash.fact_transactions"]
        == (a["data_hashes"][101]["fact_transactions"])
    )


def test_recovery_on_planted_effect_and_null(shared):
    result, _ = shared
    real, null = result["summary"]["real"], result["summary"]["nothing"]
    assert real["kind"] == "planted" and real["share"] >= 0.6 and real["ok"]
    assert null["kind"] == "null" and null["share"] == 0.0 and null["ok"]
    assert result["ok"]


def test_estimator_flags_a_missed_effect(recovery_scenario):
    tables = run.simulate(101, DAYS, recovery_scenario)
    real = recovery_scenario.campaigns[0]
    good = tracking.estimate_lift(tables, real, 0.01)
    assert good["true"] > 0.15 and good["covered"] and good["detected"]
    flipped = {
        **tables,
        "sim_truth_customer": tables["sim_truth_customer"].with_columns(
            real_arm=tables["sim_truth_customer"]["real_arm"].replace(
                {"test": "control", "control": "test"}
            )
        ),
    }
    bad = tracking.estimate_lift(flipped, real, 0.01)
    assert bad["estimated"] < 0 and not bad["covered"]


def test_summary_thresholds(recovery_scenario):
    real, null = recovery_scenario.campaigns
    hit = {"covered": True, "detected": True}
    miss = {"covered": False, "detected": False}
    false_pos = {"covered": False, "detected": True}
    seeds = [
        {"real": hit, "nothing": miss},
        {"real": hit, "nothing": miss},
        {"real": miss, "nothing": false_pos},
    ]
    s = tracking.summarize_recovery(seeds, (real, null), min_recovery=0.6, alpha=0.05)
    assert s["real"]["share"] == pytest.approx(2 / 3) and s["real"]["ok"]
    assert s["nothing"]["share"] == pytest.approx(1 / 3) and not s["nothing"]["ok"]
    strict = tracking.summarize_recovery(seeds, (real, null), min_recovery=0.9)
    assert not strict["real"]["ok"]


def test_cli_prints_summary_and_exit_code(tmp_path, capsys):
    from ccc.sim import scenario

    path = tmp_path / "s.json"
    path.write_text(
        '{"name": "cli", "start_date": "2026-10-19", "initial_customers": 60,'
        ' "arrivals_per_day": 0.0, "campaigns": ['
        '{"name": "nothing", "start_et": "2026-10-20T09:00:00", "duration_days": 5, "lift": 0.0}]}'
    )
    assert scenario.load(path).name == "cli"
    code = run_study.main(
        [
            "--scenario",
            str(path),
            "--seeds",
            "1",
            "2",
            "--days",
            "7",
            "--tracking-dir",
            str(tmp_path / "store"),
        ]
    )
    assert code == 0
    assert "nothing: null campaign detected" in capsys.readouterr().out
    assert mlflow.get_tracking_uri().startswith("sqlite:///")


@pytest.mark.slow
def test_sdv_scores_logged_for_copula(tmp_path):
    sc = Scenario(name="copula", start_date=START, initial_customers=150, arrivals_per_day=0.0)
    result = tracking.run_simulation_study(
        sc,
        [1],
        3,
        profile_source="copula",
        tracking_dir=tmp_path / "store",
        with_sdv_scores=True,
    )
    metrics = children(result)[0].data.metrics
    assert 0 <= metrics["sdv_quality_score"] <= 1
    assert 0 <= metrics["sdv_diagnostic_score"] <= 1
    with pytest.raises(ValueError):
        tracking.run_simulation_study(sc, [1], 3, tracking_dir=tmp_path / "x", with_sdv_scores=True)
