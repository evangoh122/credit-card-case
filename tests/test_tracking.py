import math
import socket
from datetime import UTC, date, datetime

import polars as pl
import pytest
from mlflow import MlflowClient
from scipy.stats import norm

from ccc import run_study, tracking
from ccc.sim import run
from ccc.sim.scenario import Campaign, Scenario, Segment

SEEDS = [101, 102, 103]
DAYS = 8
START = date(2026, 10, 19)
ALPHA = 0.05
WINDOW = ("2026-10-20T09:00:00", 6)


def scenario_with(*campaigns, customers=300, arrivals=0.0):
    return Scenario(
        name="recovery",
        start_date=START,
        initial_customers=customers,
        arrivals_per_day=arrivals,
        arrivals_trend=0.0,
        campaigns=campaigns,
    )


REAL = Campaign("real", *WINDOW, 1.0, segment=Segment(risk_max=0.8))
NOTHING = Campaign("nothing", *WINDOW, 0.0)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    store = tmp_path_factory.mktemp("shared")
    with pytest.MonkeyPatch.context() as mp:
        # the sqlite store's schema migration alone costs ~10s per fresh store
        mp.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
        result = tracking.run_simulation_study(
            scenario_with(REAL, NOTHING, arrivals=2.0),
            SEEDS,
            DAYS,
            tracking_dir=store,
            tracking_uri=(store / "file-store").as_uri(),
            locale="en_GB",
            alpha=ALPHA,
            min_recovery=0.6,
        )
        client = MlflowClient(result["tracking_uri"])
        parent = client.get_run(result["parent_run_id"])
        kids = client.search_runs(
            [result["experiment_id"]],
            filter_string=f"tags.mlflow.parentRunId = '{result['parent_run_id']}'",
        )
    return result, client, parent, kids


def artifact_paths(client, run_id, path=None):
    return {a.path for a in client.list_artifacts(run_id, path)}


def test_parent_and_child_runs(shared):
    result, client, parent, kids = shared
    assert len(kids) == len(SEEDS)
    assert {k.data.params["root_seed"] for k in kids} == {str(s) for s in SEEDS}
    assert parent.data.params["scenario_config"] == "recovery"
    assert "recovery.real.share" in parent.data.metrics
    assert "recovery.null_family.share" in parent.data.metrics
    parent_files = artifact_paths(client, parent.info.run_id)
    assert {"scenario.json", "recovery_summary.json"} <= parent_files
    kid = kids[0]
    assert kid.data.metrics["rows.dim_customer"] >= 300
    assert "null_rate.fact_transactions.amount" in kid.data.metrics
    assert "psi.max" in kid.data.metrics
    assert "lift.real.estimated" in kid.data.metrics
    folder = f"seed_{kid.data.params['root_seed']}"
    assert artifact_paths(client, kid.info.run_id) == {folder}
    inner = artifact_paths(client, kid.info.run_id, folder)
    assert {f"{folder}/drift_report.json", f"{folder}/data_dictionary.json"} <= inner
    assert f"{folder}/sample" in inner
    sample = artifact_paths(client, kid.info.run_id, f"{folder}/sample")
    assert f"{folder}/sample/fact_transactions.parquet" in sample


def test_params_and_tags(shared):
    _, _, parent, kids = shared
    for r in [parent, *kids]:
        assert r.data.tags["data_label"] == "synthetic"
        assert len(r.data.tags["git_sha"]) == 40 or r.data.tags["git_sha"] == "unknown"
        assert r.data.tags["git_dirty"] in ("true", "false", "unknown")
        for lib in ("faker", "sdv", "mlflow", "polars"):
            assert r.data.tags[f"{lib}_version"]
        assert r.data.params["faker_locale"] == "en_GB"
        assert r.data.params["customer_count"] == "300"
        assert r.data.params["cohort_day"] == str(DAYS // 2)
        assert r.data.params["profile_source"] == "handwritten"
        assert r.data.params["synthesizer"] == "handwritten"
    assert parent.data.params["multiple_comparison_policy"] == "bonferroni_within_seed"
    assert float(parent.data.params["alpha_per_test"]) == pytest.approx(ALPHA / 2)
    assert {k.data.params["drift_basis"] for k in kids} == {"late_cohorts"}


def test_bonferroni_applied_to_real_estimator_output(shared):
    result, *_ = shared
    z = norm.ppf(1 - ALPHA / 2 / 2)
    assert result["alpha_per_test"] == pytest.approx(ALPHA / 2)
    for lifts in result["per_seed_lifts"]:
        for lift in lifts.values():
            assert lift["ci_high"] - lift["ci_low"] == pytest.approx(2 * z * lift["se"])


def test_recovery_on_planted_effect_and_null_family(shared):
    result, *_ = shared
    real = result["summary"]["real"]
    null = result["summary"]["null_family"]
    assert real["kind"] == "planted" and real["share"] >= 0.6 and real["ok"]
    assert null["campaigns"] == ["nothing"] and null["share"] == 0.0 and null["ok"]
    assert result["ok"]


def test_family_share_matches_independent_count(shared):
    result, *_ = shared
    per_seed = result["per_seed_lifts"]
    expected = sum(s["nothing"]["detected"] for s in per_seed) / len(per_seed)
    assert result["summary"]["null_family"]["share"] == expected
    planted = sum(s["real"]["covered"] and s["real"]["detected"] for s in per_seed)
    assert result["summary"]["real"]["share"] == planted / len(per_seed)


def test_data_hash_reproducible(shared):
    result, *_ = shared
    sc = scenario_with(REAL, NOTHING, arrivals=2.0)
    again = {
        n: tracking.table_hash(df)
        for n, df in run.simulate(SEEDS[0], DAYS, sc, locale="en_GB").items()
    }
    other = {n: tracking.table_hash(df) for n, df in run.simulate(SEEDS[1], DAYS, sc).items()}
    assert result["data_hashes"][SEEDS[0]] == again
    assert result["data_hashes"][SEEDS[0]] != other
    assert set(again) == {"dim_customer", "sim_truth_customer", "fact_transactions", "fact_events"}


def ts(day):
    return datetime(2026, 10, day, 15, tzinfo=UTC)


def hand_tables():
    test_counts = {"t1": 1, "t2": 2, "t3": 3, "t4": 2}
    control_counts = {"c1": 1, "c2": 1, "c3": 2, "c4": 0}
    ids = [*test_counts, *control_counts]
    arms = ["test"] * 4 + ["control"] * 4
    truth = pl.DataFrame(
        {
            "customer_id": ids,
            "spend_rate": [0.5, 1.0, 1.5, 2.0, 1.0, 1.0, 1.0, 1.0],
            "hand_arm": arms,
            "hand_true_lift": [0.5] * 4 + [0.0] * 4,
        }
    )
    rows = [
        (cid, "purchase", ts(21))
        for cid, n in {**test_counts, **control_counts}.items()
        for _ in range(n)
    ]
    rows += [("t1", "purchase", ts(30)), ("c4", "repayment", ts(21))]
    txns = pl.DataFrame(
        rows,
        schema={
            "customer_id": pl.String,
            "txn_type": pl.String,
            "event_ts_utc": pl.Datetime("us", "UTC"),
        },
        orient="row",
    )
    return {"sim_truth_customer": truth, "fact_transactions": txns}


def test_estimate_lift_matches_hand_computed_delta_method_ci():
    camp = Campaign("hand", "2026-10-20T09:00:00", 5, 0.5)
    got = tracking.estimate_lift(hand_tables(), camp, alpha=0.05)
    # test counts 1,2,3,2: mean 2, sample var 2/3; control 1,1,2,0: mean 1, sample var 2/3
    se_test, se_control = math.sqrt(2 / 3 / 4), math.sqrt(2 / 3 / 4)
    ratio_se = 2 * math.sqrt((se_test / 2) ** 2 + (se_control / 1) ** 2)
    assert ratio_se == pytest.approx(2 * math.sqrt(5 / 24))
    half = 1.959963984540054 * ratio_se
    assert got["estimated"] == pytest.approx(1.0)
    assert got["se"] == pytest.approx(ratio_se)
    assert got["ci_low"] == pytest.approx(1.0 - half)
    assert got["ci_high"] == pytest.approx(1.0 + half)
    assert got["true"] == pytest.approx(0.5)
    assert got["covered"] and not got["detected"]
    assert tracking.estimate_lift(hand_tables(), camp, alpha=0.5)["detected"]


def test_estimate_lift_degenerate_inputs_do_not_detect():
    tables = hand_tables()
    tables["fact_transactions"] = tables["fact_transactions"].clear()
    got = tracking.estimate_lift(tables, Campaign("hand", "2026-10-20T09:00:00", 5, 0.5))
    assert math.isnan(got["estimated"]) and not got["detected"] and not got["covered"]


def test_estimator_flags_a_missed_effect():
    sc = scenario_with(REAL)
    tables = run.simulate(101, DAYS, sc)
    good = tracking.estimate_lift(tables, REAL, 0.01)
    assert good["true"] > 0.15 and good["covered"] and good["detected"]
    truth = tables["sim_truth_customer"]
    flipped = {
        **tables,
        "sim_truth_customer": truth.with_columns(
            real_arm=truth["real_arm"].replace({"test": "control", "control": "test"})
        ),
    }
    bad = tracking.estimate_lift(flipped, REAL, 0.01)
    assert bad["estimated"] < 0 and not bad["covered"]


def test_null_campaign_false_positive_rate_is_nominal():
    sc = scenario_with(NOTHING, customers=300)
    seeds = range(200, 208)
    detected = [
        tracking.estimate_lift(run.simulate(s, 6, sc), NOTHING, ALPHA)["detected"] for s in seeds
    ]
    assert sum(detected) <= 2


NULL_SUMMARY = {"kind": "null_family", "campaigns": ["nothing"], "share": 0.0, "threshold": 0.1}


def lift(covered, detected):
    return {"covered": covered, "detected": detected}


def test_summary_family_wise_null_and_recovery_thresholds():
    null2 = Campaign("null2", *WINDOW, 0.0)
    per_seed = [
        {"real": lift(True, True), "nothing": lift(False, False), "null2": lift(False, False)},
        {"real": lift(True, True), "nothing": lift(False, True), "null2": lift(False, True)},
        {"real": lift(False, False), "nothing": lift(False, False), "null2": lift(False, True)},
    ]
    s = tracking.summarize_recovery(per_seed, (REAL, NOTHING, null2), min_recovery=0.6, alpha=0.5)
    assert s["real"]["share"] == pytest.approx(2 / 3) and s["real"]["ok"]
    assert s["null_family"]["share"] == pytest.approx(2 / 3)
    assert not s["null_family"]["ok"]
    loose = tracking.summarize_recovery(per_seed, (REAL, NOTHING, null2), 0.9, alpha=0.7)
    assert loose["null_family"]["ok"] and not loose["real"]["ok"]
    assert "nothing, null2" in tracking.format_summary(s)


@pytest.mark.parametrize(
    ("alpha", "min_recovery"), [(0, 0.9), (1, 0.9), (-0.1, 0.9), (0.05, 1.1), (0.05, -0.1)]
)
def test_invalid_thresholds_rejected(alpha, min_recovery, tmp_path):
    with pytest.raises(ValueError):
        tracking.summarize_recovery([{"real": lift(True, True)}], (REAL,), min_recovery, alpha)
    with pytest.raises(ValueError):
        tracking.run_simulation_study(
            scenario_with(REAL),
            [1],
            3,
            tracking_dir=tmp_path,
            alpha=alpha,
            min_recovery=min_recovery,
        )


def test_invalid_study_arguments(tmp_path):
    sc = scenario_with(REAL)
    with pytest.raises(ValueError):
        tracking.run_simulation_study(sc, [], 3, tracking_dir=tmp_path)
    with pytest.raises(ValueError):
        tracking.run_simulation_study(sc, [1], 0, tracking_dir=tmp_path)


def test_drift_without_late_cohort_reports_fallback(caplog):
    tables = run.simulate(1, 4, scenario_with(NOTHING, customers=60))
    ref = tracking.make_reference(tables)
    with caplog.at_level("WARNING"):
        _, basis = tracking.drift_for(ref, tables, cohort_day=2)
    assert basis == "all_customers_no_late_cohort"
    assert "no customers at cohort day" in caplog.text


def test_cli_prints_summary_and_exit_code(monkeypatch, capsys):
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        summary = {"null_family": {**NULL_SUMMARY, "ok": len(calls) == 1}}
        return {"summary": summary, "ok": len(calls) == 1}

    monkeypatch.setattr(tracking, "run_simulation_study", fake)
    argv = ["--scenario", "s.toml", "--seeds", "1", "2", "--days", "5", "--alpha", "0.1"]
    assert run_study.main(argv) == 0
    assert "null campaigns nothing: any detected in 0% of seeds" in capsys.readouterr().out
    assert run_study.main(argv) == 1
    args, kwargs = calls[0]
    assert args == ("s.toml", [1, 2], 5, None) and kwargs["alpha"] == 0.1


@pytest.mark.slow
def test_sdv_scores_logged_for_copula(tmp_path):
    sc = scenario_with(customers=150)
    result = tracking.run_simulation_study(
        sc,
        [1],
        3,
        profile_source="copula",
        tracking_dir=tmp_path / "store",
        with_sdv_scores=True,
    )
    kid = MlflowClient(result["tracking_uri"]).search_runs(
        [result["experiment_id"]],
        filter_string=f"tags.mlflow.parentRunId = '{result['parent_run_id']}'",
    )[0]
    assert kid.data.params["synthesizer"] == "gaussian_copula"
    assert 0 <= kid.data.metrics["sdv_quality_score"] <= 1
    assert 0 <= kid.data.metrics["sdv_diagnostic_score"] <= 1
    with pytest.raises(ValueError):
        tracking.run_simulation_study(sc, [1], 3, tracking_dir=tmp_path / "x", with_sdv_scores=True)
