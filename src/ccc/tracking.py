import dataclasses
import hashlib
import importlib.metadata
import json
import math
import subprocess
import tempfile
from datetime import date, datetime
from pathlib import Path

import mlflow
import polars as pl
from scipy.stats import norm

from ccc import drift
from ccc.sim import events, profiles, run, scenario

REPO_ROOT = Path(__file__).resolve().parents[2]
NUMERIC_FEATURES = ("risk", "spend_rate", "responsiveness", "utilisation")
CATEGORICAL_FEATURES = ("channel", "region")
SAMPLE_ROWS = 100
PSI_BINS = 5
DEFAULT_MIN_RECOVERY = 0.9
DEFAULT_ALPHA = 0.05
EXPERIMENT = "ccc-simulation"
EVENT_RATE = {"purchase": events.purchase_rate, "app_open": events.open_rate}


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def code_version() -> dict[str, str]:
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "git_sha": sha or "unknown",
        "git_dirty": "unknown" if status is None else str(bool(status)).lower(),
    }


def library_versions() -> dict[str, str]:
    return {
        f"{lib}_version": importlib.metadata.version(lib)
        for lib in ("faker", "sdv", "mlflow", "polars")
    }


def table_hash(df: pl.DataFrame) -> str:
    with tempfile.TemporaryFile() as fh:
        df.write_ipc(fh, compression="uncompressed")
        fh.seek(0)
        return hashlib.sha256(fh.read()).hexdigest()


def null_rates(df: pl.DataFrame) -> dict[str, float]:
    n = df.height
    return {c: (df[c].null_count() / n if n else 0.0) for c in df.columns}


def estimate_lift(
    tables: dict[str, pl.DataFrame], campaign: scenario.Campaign, alpha: float = DEFAULT_ALPHA
) -> dict:
    """Test/control ratio of event counts in the campaign window against the planted lift."""
    name = campaign.name
    truth = tables["sim_truth_customer"]
    if campaign.event == "purchase":
        source = tables["fact_transactions"].filter(pl.col("txn_type") == "purchase")
    else:
        source = tables["fact_events"].filter(pl.col("event_type") == campaign.event)
    window = source.filter(
        (pl.col("event_ts_utc") >= campaign.start_utc) & (pl.col("event_ts_utc") < campaign.end_utc)
    )
    counts = window.group_by("customer_id").len()
    df = (
        truth.join(counts, on="customer_id", how="left")
        .with_columns(pl.col("len").fill_null(0))
        .filter(pl.col(f"{name}_arm") != "none")
    )
    arms = {a: df.filter(pl.col(f"{name}_arm") == a) for a in ("test", "control")}
    test = arms["test"]
    weight = test["spend_rate"].map_elements(EVENT_RATE[campaign.event], return_dtype=pl.Float64)
    total = weight.sum()
    true_lift = float((weight * test[f"{name}_true_lift"]).sum() / total) if total else 0.0
    out = {
        "true": true_lift,
        "estimated": math.nan,
        "se": math.nan,
        "ci_low": math.nan,
        "ci_high": math.nan,
        "covered": False,
        "detected": False,
    }
    if min(a.height for a in arms.values()) < 2:
        return out
    mean = {a: x["len"].mean() for a, x in arms.items()}
    if mean["control"] <= 0 or mean["test"] <= 0:
        return out
    se = {a: x["len"].std() / math.sqrt(x.height) for a, x in arms.items()}
    ratio = mean["test"] / mean["control"]
    ratio_se = ratio * math.sqrt(
        (se["test"] / mean["test"]) ** 2 + (se["control"] / mean["control"]) ** 2
    )
    z = float(norm.ppf(1 - alpha / 2))
    est = ratio - 1
    low, high = est - z * ratio_se, est + z * ratio_se
    out.update(
        estimated=est,
        se=ratio_se,
        ci_low=low,
        ci_high=high,
        covered=low <= true_lift <= high,
        detected=low > 0 or high < 0,
    )
    return out


def summarize_recovery(
    per_seed: list[dict[str, dict]],
    campaigns: tuple[scenario.Campaign, ...],
    min_recovery: float = DEFAULT_MIN_RECOVERY,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, dict]:
    """Planted campaigns must be recovered (CI covers truth and excludes zero) in at least
    min_recovery of seeds; null campaigns must be detected in at most alpha of seeds."""
    summary = {}
    for c in campaigns:
        rows = [s[c.name] for s in per_seed]
        n = len(rows)
        if c.lift > 0:
            share = sum(r["covered"] and r["detected"] for r in rows) / n
            summary[c.name] = {
                "kind": "planted",
                "share": share,
                "threshold": min_recovery,
                "ok": share >= min_recovery,
            }
        else:
            share = sum(r["detected"] for r in rows) / n
            summary[c.name] = {
                "kind": "null",
                "share": share,
                "threshold": alpha,
                "ok": share <= alpha,
            }
    return summary


def feature_table(tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    return tables["dim_customer"].join(tables["sim_truth_customer"], on="customer_id", how="inner")


def make_reference(tables: dict[str, pl.DataFrame]) -> drift.Reference:
    initial = feature_table(tables).filter(pl.col("cohort_day") == 0)
    return drift.make_reference(initial, NUMERIC_FEATURES, CATEGORICAL_FEATURES, n_bins=PSI_BINS)


def drift_for(ref: drift.Reference, tables: dict[str, pl.DataFrame], cohort_day: int) -> dict:
    df = feature_table(tables)
    late = df.filter(pl.col("cohort_day") >= cohort_day)
    return drift.drift_report(ref, late if late.height else df)


def data_dictionary(tables: dict[str, pl.DataFrame], hashes: dict[str, str]) -> dict:
    return {
        name: {
            "rows": df.height,
            "sha256": hashes[name],
            "columns": {
                c: {"dtype": str(df.schema[c]), "null_count": df[c].null_count()}
                for c in df.columns
            },
        }
        for name, df in tables.items()
    }


def sdv_scores(seed: int, tables: dict[str, pl.DataFrame]) -> dict[str, float]:
    from sdv.evaluation import evaluate_quality, run_diagnostic

    cols = [*profiles.LATENT, *CATEGORICAL_FEATURES]
    seed_pairs = [(f"S-{i:05d}", 0) for i in range(run.SEED_TABLE_ROWS)]
    real = profiles.handwritten_table(seed, seed_pairs, run.NO_SHIFT).select(cols).to_pandas()
    synthetic = feature_table(tables).filter(pl.col("cohort_day") == 0).select(cols).to_pandas()
    meta = profiles.copula_metadata()
    real, synthetic = {"seed": real}, {"seed": synthetic}
    quality = evaluate_quality(real, synthetic, meta, verbose=False)
    diagnostic = run_diagnostic(real, synthetic, meta, verbose=False)
    return {
        "sdv_quality_score": float(quality.get_score()),
        "sdv_diagnostic_score": float(diagnostic.get_score()),
    }


def _json_default(o):
    if isinstance(o, date | datetime):
        return o.isoformat()
    raise TypeError(type(o).__name__)


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default))


def _finite(metrics: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in metrics.items() if math.isfinite(v)}


def _setup(tracking_dir: Path, tracking_uri: str | None, experiment: str) -> str:
    tracking_dir = Path(tracking_dir).resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri or f"sqlite:///{tracking_dir / 'mlflow.db'}")
    found = mlflow.get_experiment_by_name(experiment)
    if found is not None:
        return found.experiment_id
    return mlflow.create_experiment(
        experiment, artifact_location=(tracking_dir / "artifacts").as_uri()
    )


def _log_seed(
    seed: int,
    tables: dict[str, pl.DataFrame],
    lifts: dict[str, dict],
    report: dict,
    out: Path,
    scores: dict[str, float],
) -> dict[str, str]:
    hashes = {name: table_hash(df) for name, df in tables.items()}
    mlflow.log_params({f"data_hash.{n}": h for n, h in hashes.items()})
    for name, df in tables.items():
        mlflow.log_metric(f"rows.{name}", df.height)
        mlflow.log_metrics({f"null_rate.{name}.{c}": r for c, r in null_rates(df).items()})
    psi = {c: f["psi"] for c, f in report["features"].items()}
    mlflow.log_metrics({f"psi.{c}": v for c, v in psi.items()})
    mlflow.log_metric("psi.max", max(psi.values()))
    for cname, lift in lifts.items():
        mlflow.log_metrics(
            _finite(
                {
                    f"lift.{cname}.{k}": float(v)
                    for k, v in lift.items()
                    if k not in ("covered", "detected")
                }
                | {
                    f"lift.{cname}.ci_covers_true": float(lift["covered"]),
                    f"lift.{cname}.detected": float(lift["detected"]),
                }
            )
        )
    mlflow.log_metrics(scores)

    sample = out / "sample"
    sample.mkdir(parents=True)
    for name, df in tables.items():
        df.head(SAMPLE_ROWS).write_parquet(sample / f"{name}.parquet")
    _write_json(out / "data_dictionary.json", data_dictionary(tables, hashes))
    _write_json(out / "drift_report.json", report)
    _write_json(out / "lift_estimates.json", lifts)
    mlflow.log_artifacts(str(out), artifact_path=f"seed_{seed}")
    return hashes


def run_simulation_study(
    sc: scenario.Scenario | str | Path,
    seeds: list[int],
    days: int,
    n_initial: int | None = None,
    *,
    profile_source: str = "handwritten",
    locale: str = "en_US",
    tracking_dir: str | Path = "mlruns",
    tracking_uri: str | None = None,
    experiment: str = EXPERIMENT,
    min_recovery: float = DEFAULT_MIN_RECOVERY,
    alpha: float = DEFAULT_ALPHA,
    with_sdv_scores: bool = False,
) -> dict:
    if not seeds:
        raise ValueError("seeds must not be empty")
    if with_sdv_scores and profile_source != "copula":
        raise ValueError("SDV scores require profile_source='copula'")
    if not isinstance(sc, scenario.Scenario):
        sc = scenario.load(sc)
    if n_initial is not None:
        sc = dataclasses.replace(sc, initial_customers=n_initial)
    cohort_day = days // 2
    exp_id = _setup(Path(tracking_dir), tracking_uri, experiment)
    tags = {
        "data_label": events.DATA_LABEL,
        **code_version(),
        **library_versions(),
    }

    ref = None
    per_seed: list[dict[str, dict]] = []
    seed_hashes: dict[int, dict[str, str]] = {}
    with mlflow.start_run(experiment_id=exp_id, run_name=f"study-{sc.name}") as parent:
        mlflow.set_tags(tags)
        mlflow.log_params(
            {
                "seeds": ",".join(map(str, seeds)),
                "faker_locale": locale,
                "profile_source": profile_source,
                "synthesizer": "gaussian_copula" if profile_source == "copula" else "handwritten",
                "customer_count": sc.initial_customers,
                "cohort_day": cohort_day,
                "days": days,
                "scenario_config": sc.name,
                "min_recovery": min_recovery,
                "alpha": alpha,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "scenario.json"
            _write_json(cfg, dataclasses.asdict(sc))
            mlflow.log_artifact(str(cfg))
        for seed in seeds:
            tables = run.simulate(seed, days, sc, profile_source=profile_source, locale=locale)
            if ref is None:
                ref = make_reference(tables)
            lifts = {c.name: estimate_lift(tables, c, alpha) for c in sc.campaigns}
            per_seed.append(lifts)
            scores = sdv_scores(seed, tables) if with_sdv_scores else {}
            with (
                mlflow.start_run(experiment_id=exp_id, run_name=f"seed-{seed}", nested=True),
                tempfile.TemporaryDirectory() as tmp,
            ):
                mlflow.set_tags(tags)
                mlflow.log_params(
                    {
                        "root_seed": seed,
                        "faker_locale": locale,
                        "profile_source": profile_source,
                        "customer_count": sc.initial_customers,
                        "cohort_day": cohort_day,
                        "scenario_config": sc.name,
                    }
                )
                report = drift_for(ref, tables, cohort_day)
                seed_hashes[seed] = _log_seed(
                    seed, tables, lifts, report, Path(tmp) / "out", scores
                )
        summary = summarize_recovery(per_seed, sc.campaigns, min_recovery, alpha)
        for cname, s in summary.items():
            mlflow.log_metrics(
                {f"recovery.{cname}.share": s["share"], f"recovery.{cname}.ok": s["ok"]}
            )
        mlflow.log_metric("recovery.all_ok", all(s["ok"] for s in summary.values()))
        mlflow.log_dict(summary, "recovery_summary.json")
    return {
        "parent_run_id": parent.info.run_id,
        "experiment_id": exp_id,
        "tracking_uri": mlflow.get_tracking_uri(),
        "summary": summary,
        "data_hashes": seed_hashes,
        "ok": all(s["ok"] for s in summary.values()),
    }


def format_summary(summary: dict[str, dict]) -> str:
    lines = []
    for name, s in summary.items():
        if s["kind"] == "planted":
            lines.append(
                f"{name}: planted effect recovered in {s['share']:.0%} of seeds "
                f"(need {s['threshold']:.0%}) {'ok' if s['ok'] else 'FAIL'}"
            )
        else:
            lines.append(
                f"{name}: null campaign detected in {s['share']:.0%} of seeds "
                f"(allowed {s['threshold']:.0%}) {'ok' if s['ok'] else 'FAIL'}"
            )
    return "\n".join(lines)
