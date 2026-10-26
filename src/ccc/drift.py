import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

EPSILON = 1e-4
FLOAT_TOL = 1e-9
WATCH_PSI = 0.10
ALERT_PSI = 0.25
MIX_SHIFT_POINTS = 0.05
FAR_QUANTILE = 0.95

Values = pl.Series | np.ndarray | Sequence[float] | Sequence[str]


@dataclass(frozen=True)
class Reference:
    numeric_edges: dict[str, list[float]]
    numeric_shares: dict[str, list[float]]
    categorical_shares: dict[str, dict[str, float]]
    segment_shares: dict[str, float] = field(default_factory=dict)
    distance_mean: float | None = None
    distance_far_threshold: float | None = None

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "Reference":
        return cls(**json.loads(Path(path).read_text()))


def _floats(values: Values) -> np.ndarray:
    arr = values.to_numpy() if isinstance(values, pl.Series) else np.asarray(values)
    arr = arr.astype(float)
    return arr[~np.isnan(arr)]


def _is_null(v: object) -> bool:
    return v is None or (isinstance(v, float | np.floating) and bool(np.isnan(v)))


def _labels(values: Values) -> list[str]:
    items = values.to_list() if isinstance(values, pl.Series) else list(values)
    return [str(v) for v in items if not _is_null(v)]


def _shares(counts: Counter, keys: Sequence[str]) -> list[float]:
    total = sum(counts[k] for k in keys)
    if total == 0:
        raise ValueError("no non-null values")
    return [counts[k] / total for k in keys]


def _bin_counts(x: np.ndarray, edges: Sequence[float]) -> Counter:
    return Counter(np.searchsorted(edges, x, side="right").tolist())


def _psi(expected: Sequence[float], actual: Sequence[float]) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPSILON, None)
    a = np.clip(np.asarray(actual, dtype=float), EPSILON, None)
    return float(np.sum((a - e) * np.log(a / e)))


def make_reference(
    df: pl.DataFrame,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str] = (),
    n_bins: int = 10,
    segment_col: str | None = None,
    distance_col: str | None = None,
) -> Reference:
    """Frozen snapshot of the training data; never refreshed by this module."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    edges: dict[str, list[float]] = {}
    num_shares: dict[str, list[float]] = {}
    for col in numeric_cols:
        x = _floats(df[col])
        if x.size == 0:
            raise ValueError(f"column {col!r} has no non-null values")
        cuts = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))
        edges[col] = cuts.tolist()
        counts = _bin_counts(x, cuts)
        num_shares[col] = _shares(counts, range(len(cuts) + 1))

    cat_shares: dict[str, dict[str, float]] = {}
    for col in categorical_cols:
        counts = Counter(_labels(df[col]))
        keys = sorted(counts)
        cat_shares[col] = dict(zip(keys, _shares(counts, keys), strict=True))

    segment_shares: dict[str, float] = {}
    if segment_col is not None:
        counts = Counter(_labels(df[segment_col]))
        keys = sorted(counts)
        segment_shares = dict(zip(keys, _shares(counts, keys), strict=True))

    mean = threshold = None
    if distance_col is not None:
        d = _floats(df[distance_col])
        if d.size == 0:
            raise ValueError(f"column {distance_col!r} has no non-null values")
        mean = float(d.mean())
        threshold = float(np.quantile(d, FAR_QUANTILE))

    return Reference(edges, num_shares, cat_shares, segment_shares, mean, threshold)


def psi_numeric(ref: Reference, column: str, current: Values) -> float:
    cuts = ref.numeric_edges[column]
    x = _floats(current)
    if x.size == 0:
        raise ValueError("current has no non-null values")
    actual = _shares(_bin_counts(x, cuts), range(len(cuts) + 1))
    return _psi(ref.numeric_shares[column], actual)


def _psi_shares(expected: dict[str, float], current: Values) -> float:
    counts = Counter(_labels(current))
    if not counts:
        raise ValueError("current has no non-null values")
    total = sum(counts.values())
    keys = sorted(expected)
    unseen = sum(n for k, n in counts.items() if k not in expected) / total
    exp = [expected[k] for k in keys] + [0.0]
    act = [counts[k] / total for k in keys] + [unseen]
    return _psi(exp, act)


def psi_categorical(ref: Reference, column: str, current: Values) -> float:
    return _psi_shares(ref.categorical_shares[column], current)


def psi_status(psi: float) -> str:
    if psi > ALERT_PSI:
        return "alert"
    if psi >= WATCH_PSI:
        return "watch"
    return "ok"


def segment_mix_drift(ref: Reference, segments: Values) -> dict:
    counts = Counter(_labels(segments))
    total = sum(counts.values())
    if total == 0:
        raise ValueError("segments has no non-null values")
    keys = sorted(set(ref.segment_shares) | set(counts))
    shares = {k: counts[k] / total for k in keys}
    moves = {k: shares[k] - ref.segment_shares.get(k, 0.0) for k in keys}
    flagged = [k for k in keys if abs(moves[k]) >= MIX_SHIFT_POINTS - FLOAT_TOL]
    return {
        "shares": shares,
        "moves": moves,
        "psi": _psi_shares(ref.segment_shares, segments),
        "flagged": flagged,
    }


def cluster_fit(ref: Reference, distances: Values) -> dict:
    if ref.distance_far_threshold is None:
        raise ValueError("reference was built without distance_col")
    d = _floats(distances)
    if d.size == 0:
        raise ValueError("distances has no values")
    return {
        "mean_distance": float(d.mean()),
        "mean_ratio": float(d.mean() / ref.distance_mean) if ref.distance_mean else None,
        "far_share": float(np.mean(d > ref.distance_far_threshold)),
    }


def structure_agreement(
    X: np.ndarray, frozen_labels: Values, n_clusters: int | None = None, seed: int = 0
) -> float:
    """Adjusted Rand index between frozen-model labels and a KMeans refit on X."""
    labels = np.asarray(frozen_labels)
    k = n_clusters or len(np.unique(labels))
    refit = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(np.asarray(X))
    return float(adjusted_rand_score(labels, refit))


def sustained_alert(history: Sequence[str], min_periods: int = 3) -> bool:
    """True only when the last min_periods statuses are all 'watch' or 'alert'."""
    if min_periods < 1:
        raise ValueError("min_periods must be at least 1")
    recent = list(history)[-min_periods:]
    return len(recent) == min_periods and all(s in ("watch", "alert") for s in recent)


def drift_report(
    ref: Reference,
    current: pl.DataFrame,
    segment_col: str | None = None,
    distance_col: str | None = None,
) -> dict:
    """Per-period report of feature PSI, segment mix and cluster fit. Never refits."""
    features = {}
    for col in ref.numeric_edges:
        features[col] = psi_numeric(ref, col, current[col])
    for col in ref.categorical_shares:
        features[col] = psi_categorical(ref, col, current[col])
    report: dict = {
        "features": {c: {"psi": v, "status": psi_status(v)} for c, v in features.items()}
    }
    if segment_col is not None:
        mix = segment_mix_drift(ref, current[segment_col])
        report["segment_mix"] = {**mix, "status": psi_status(mix["psi"])}
    if distance_col is not None:
        report["cluster_fit"] = cluster_fit(ref, current[distance_col])
    report["alerts"] = [c for c, f in report["features"].items() if f["status"] == "alert"]
    if report.get("segment_mix", {}).get("flagged"):
        report["alerts"].append("segment_mix")
    return report
