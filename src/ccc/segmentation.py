import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ("arrears", "utilisation", "spend", "activity")
RFM_COLUMNS = ("recency", "frequency", "monetary")
TIERS = ("Prime", "Watch", "Nurture", "Decline")
MODEL_VERSION = "seg-1.0.0"
NEW_CUSTOMER_DAYS = 30


def _reject_leaky_columns(columns: Sequence[str]) -> None:
    for name in columns:
        if name.startswith("offer") or name.startswith("post_"):
            raise ValueError(f"column {name!r} is measured after an offer and must not be used")


def _numeric_frame(features: pl.DataFrame, names: Sequence[str]) -> pl.DataFrame:
    if not isinstance(features, pl.DataFrame):
        raise TypeError("features must be a polars DataFrame")
    _reject_leaky_columns(features.columns)
    missing = [name for name in names if name not in features.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    frame = features.select(list(names))
    for name in names:
        if frame[name].null_count():
            raise ValueError(f"feature {name!r} must not contain nulls")
        dtype = frame.schema[name]
        if dtype == pl.Boolean or not dtype.is_numeric():
            raise TypeError(f"feature {name!r} must be numeric")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("features must be finite")
    return frame


def _feature_names(include_rfm: bool) -> tuple[str, ...]:
    return FEATURE_COLUMNS + RFM_COLUMNS if include_rfm else FEATURE_COLUMNS


def _matrix(features: pl.DataFrame, names: Sequence[str]) -> np.ndarray:
    return _numeric_frame(features, names).to_numpy()


def _make_version(
    feature_names: Sequence[str],
    mean: Sequence[float],
    scale: Sequence[float],
    centroids: Sequence[Sequence[float]],
) -> str:
    payload = json.dumps(
        {
            "feature_names": list(feature_names),
            "mean": list(mean),
            "scale": list(scale),
            "centroids": [list(row) for row in centroids],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{MODEL_VERSION}+{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


@dataclass(frozen=True)
class RuleThresholds:
    arrears: float
    utilisation: float
    spend: float
    activity: float


@dataclass(frozen=True)
class SegmentModel:
    version: str
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    centroids: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        width = len(self.feature_names)
        if width == 0 or len(set(self.feature_names)) != width:
            raise ValueError("feature_names must be non-empty and unique")
        if len(self.mean) != width or len(self.scale) != width:
            raise ValueError("mean and scale must match feature_names")
        if not self.centroids:
            raise ValueError("centroids must be non-empty")
        flat = [*self.mean, *self.scale, *(v for row in self.centroids for v in row)]
        if not all(isinstance(v, float) and np.isfinite(v) for v in flat):
            raise ValueError("mean, scale and centroids must be finite floats")
        if any(s <= 0 for s in self.scale):
            raise ValueError("scale must be positive")
        for row in self.centroids:
            if len(row) != width:
                raise ValueError("centroid width must match feature_names")
        expected = _make_version(self.feature_names, self.mean, self.scale, self.centroids)
        if self.version != expected:
            raise ValueError(f"version {self.version!r} does not match model contents {expected!r}")

    def predict(self, features: pl.DataFrame) -> pl.Series:
        x = _matrix(features, self.feature_names)
        mean = np.asarray(self.mean, dtype=float)
        scale = np.asarray(self.scale, dtype=float)
        centroids = np.asarray(self.centroids, dtype=float)
        z = (x - mean) / scale
        d2 = ((z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        return pl.Series("cluster", d2.argmin(axis=1).tolist())

    def save(self, path: str | Path) -> None:
        payload = {
            "version": self.version,
            "feature_names": list(self.feature_names),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "centroids": [list(row) for row in self.centroids],
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "SegmentModel":
        try:
            payload = json.loads(Path(path).read_text())
            return cls(
                version=payload["version"],
                feature_names=tuple(payload["feature_names"]),
                mean=tuple(float(v) for v in payload["mean"]),
                scale=tuple(float(v) for v in payload["scale"]),
                centroids=tuple(tuple(float(v) for v in row) for row in payload["centroids"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid segment model file: {path}") from exc


def fit_segmenter(
    features: pl.DataFrame,
    k: int,
    include_rfm: bool = False,
    random_state: int = 0,
) -> SegmentModel:
    names = _feature_names(include_rfm)
    x = _matrix(features, names)
    if isinstance(k, bool) or not isinstance(k, int) or not 2 <= k <= x.shape[0]:
        raise ValueError(f"k must be an integer in [2, {x.shape[0]}], got {k!r}")
    scaler = StandardScaler()
    z = scaler.fit_transform(x)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(z)
    mean = tuple(float(v) for v in scaler.mean_)
    scale = tuple(float(v) for v in scaler.scale_)
    centroids = tuple(tuple(float(v) for v in row) for row in km.cluster_centers_)
    return SegmentModel(
        version=_make_version(names, mean, scale, centroids),
        feature_names=names,
        mean=mean,
        scale=scale,
        centroids=centroids,
    )


def choose_k(
    features: pl.DataFrame,
    k_range: Iterable[int],
    include_rfm: bool = False,
    random_state: int = 0,
) -> tuple[pl.DataFrame, int]:
    names = _feature_names(include_rfm)
    x = _matrix(features, names)
    n = x.shape[0]
    ks = list(k_range)
    if not ks:
        raise ValueError("k_range must not be empty")
    z = StandardScaler().fit_transform(x)
    scores = []
    for k in ks:
        if isinstance(k, bool) or not isinstance(k, int) or not 2 <= k <= n - 1:
            raise ValueError(f"k must be an integer in [2, {n - 1}], got {k!r}")
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(z)
        scores.append(float(silhouette_score(z, labels)))
    table = pl.DataFrame({"k": ks, "silhouette": scores})
    best = ks[max(range(len(ks)), key=lambda i: (scores[i], -ks[i]))]
    return table, best


def map_clusters_to_tiers(
    centroids: Sequence[Sequence[float]], feature_names: Sequence[str]
) -> tuple[str, ...]:
    """Map four clusters to tiers from the scaled centroid profile.

    Decline: highest arrears + utilisation.
    Nurture: lowest spend + activity among the rest.
    Prime: highest spend + activity - arrears - utilisation among the rest.
    Watch: the leftover cluster.
    Ties resolve to the lowest cluster index.
    """
    arr = np.asarray(centroids, dtype=float)
    names = tuple(feature_names)
    if arr.ndim != 2 or arr.shape[0] != 4:
        raise ValueError("tier mapping requires exactly 4 clusters")
    if arr.shape[1] != len(names):
        raise ValueError("centroid width must match feature_names")
    needed = ("arrears", "utilisation", "spend", "activity")
    missing = [name for name in needed if name not in names]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    a, u, s, act = (names.index(name) for name in needed)
    risk = arr[:, a] + arr[:, u]
    engage = arr[:, s] + arr[:, act]
    tiers = [""] * arr.shape[0]
    remaining = list(range(arr.shape[0]))
    decline = max(remaining, key=lambda i: (risk[i], -i))
    remaining.remove(decline)
    tiers[decline] = "Decline"
    nurture = min(remaining, key=lambda i: (engage[i], i))
    remaining.remove(nurture)
    tiers[nurture] = "Nurture"
    prime = max(remaining, key=lambda i: (engage[i] - risk[i], -i))
    remaining.remove(prime)
    tiers[prime] = "Prime"
    tiers[remaining[0]] = "Watch"
    return tuple(tiers)


def rule_based_tiers(features: pl.DataFrame, thresholds: RuleThresholds) -> pl.Series:
    """Tier by fixed thresholds, in order:

    Decline: arrears > thresholds.arrears or utilisation > thresholds.utilisation.
    Nurture: activity < thresholds.activity and spend < thresholds.spend.
    Prime: spend >= thresholds.spend and activity >= thresholds.activity.
    Watch: everything else.
    """
    frame = _numeric_frame(features, FEATURE_COLUMNS)
    t = thresholds
    return frame.select(
        pl.when((pl.col("arrears") > t.arrears) | (pl.col("utilisation") > t.utilisation))
        .then(pl.lit("Decline"))
        .when((pl.col("activity") < t.activity) & (pl.col("spend") < t.spend))
        .then(pl.lit("Nurture"))
        .when((pl.col("spend") >= t.spend) & (pl.col("activity") >= t.activity))
        .then(pl.lit("Prime"))
        .otherwise(pl.lit("Watch"))
        .alias("tier")
    ).to_series()


def hard_credit_rules(df: pl.DataFrame) -> pl.Series:
    """Hard credit override, in order:

    Decline: arrears > 0 or over_limit, regardless of cluster.
    Nurture: days_active < 30 (new customers start in Nurture).
    No override (null): everything else.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError("df must be a polars DataFrame")
    _reject_leaky_columns(df.columns)
    needed = ("arrears", "over_limit", "days_active")
    missing = [name for name in needed if name not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    frame = df.select(list(needed))
    for name in needed:
        if frame[name].null_count():
            raise ValueError(f"{name} must not contain nulls")
    if frame.schema["over_limit"] != pl.Boolean:
        raise TypeError("over_limit must be boolean")
    for name in ("arrears", "days_active"):
        if not frame.schema[name].is_numeric():
            raise TypeError(f"{name} must be numeric")
    decline = (pl.col("arrears") > 0) | pl.col("over_limit")
    nurture = pl.col("days_active") < NEW_CUSTOMER_DAYS
    return frame.select(
        pl.when(decline)
        .then(pl.lit("Decline"))
        .when(nurture)
        .then(pl.lit("Nurture"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
        .alias("override")
    ).to_series()


def final_tier(cluster_tier: pl.Series, override: pl.Series) -> pl.Series:
    """Hard-rule override wins where set; otherwise the cluster tier."""
    if len(cluster_tier) != len(override):
        raise ValueError("cluster_tier and override must have the same length")
    if cluster_tier.null_count():
        raise ValueError("cluster_tier must not contain nulls")
    for name, series in (("cluster_tier", cluster_tier), ("override", override)):
        bad = [v for v in series.drop_nulls().unique().to_list() if v not in TIERS]
        if bad:
            raise ValueError(f"{name} contains values outside {TIERS}: {bad}")
    combined = [
        o if o is not None else c
        for o, c in zip(override.to_list(), cluster_tier.to_list(), strict=True)
    ]
    return pl.Series("final_tier", combined)


def experiment_eligible(tiers: pl.Series) -> pl.Series:
    """Decline is never experiment-eligible."""
    if tiers.null_count():
        raise ValueError("tiers must not contain nulls")
    bad = [v for v in tiers.unique().to_list() if v not in TIERS]
    if bad:
        raise ValueError(f"tiers contains values outside {TIERS}: {bad}")
    return (tiers != "Decline").alias("eligible")
