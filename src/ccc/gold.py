import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from ccc.experiment import DEFAULT_ARMS, assign
from ccc.segmentation import (
    SegmentModel,
    final_tier,
    hard_credit_rules,
    map_clusters_to_tiers,
)
from ccc.timeutil import snapshot_cutoff_utc

LOGGER = logging.getLogger(__name__)
RFM_METRICS = ("recency", "frequency", "monetary")
TRUTH_COLUMNS = {"cohort_day", "risk", "spend_rate", "responsiveness"}


def guard_no_truth_columns(frame: pl.DataFrame) -> None:
    leaked = sorted(
        name
        for name in frame.columns
        if name in TRUTH_COLUMNS or name.endswith("_true_lift") or name.startswith("sim_truth_")
    )
    if leaked:
        raise ValueError(
            f"simulation truth columns must not be used in features or models: {leaked}"
        )


def _snapshot_date(value: date) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError("snapshot_date must be a date")
    return value


def _validate_windows(windows: Mapping[str, int]) -> dict[str, int]:
    if set(windows) != set(RFM_METRICS):
        raise ValueError(f"windows must contain exactly {list(RFM_METRICS)}")
    invalid = (
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in windows.values()
    )
    if any(invalid):
        raise ValueError("window lengths must be positive integers")
    return dict(windows)


@dataclass(frozen=True)
class RFMBins:
    edges: Mapping[str, tuple[float, ...]]

    def __post_init__(self) -> None:
        if set(self.edges) != set(RFM_METRICS):
            raise ValueError(f"bin edges must contain exactly {list(RFM_METRICS)}")
        for metric, edges in self.edges.items():
            if len(edges) < 2 or any(a >= b for a, b in zip(edges, edges[1:], strict=False)):
                raise ValueError(f"{metric} bin edges must be strictly increasing")

    def save(self, path: str | Path) -> None:
        payload = {metric: [None, *self.edges[metric][1:-1], None] for metric in RFM_METRICS}
        Path(path).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "RFMBins":
        try:
            payload = json.loads(Path(path).read_text())
            edges = {}
            for metric in RFM_METRICS:
                values = payload[metric]
                if len(values) < 2 or values[0] is not None or values[-1] is not None:
                    raise ValueError("outer bin edges must be null")
                edges[metric] = tuple(
                    [float("-inf"), *(float(value) for value in values[1:-1]), float("inf")]
                )
            return cls(edges)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid RFM bins file: {path}") from exc


def make_rfm_bins(
    reference_snapshot: pl.DataFrame,
    quantiles: Sequence[float] = (0.2, 0.4, 0.6, 0.8),
) -> RFMBins:
    guard_no_truth_columns(reference_snapshot)
    if any(not 0 < q < 1 for q in quantiles) or list(quantiles) != sorted(set(quantiles)):
        raise ValueError("quantiles must be unique, increasing values between 0 and 1")
    edges: dict[str, tuple[float, ...]] = {}
    for metric in RFM_METRICS:
        if metric not in reference_snapshot.columns:
            raise ValueError(f"missing RFM column: {metric}")
        values = reference_snapshot.get_column(metric).drop_nulls().cast(pl.Float64)
        if values.is_empty():
            raise ValueError(f"cannot fit {metric} bins without values")
        boundaries = sorted({float(values.quantile(q, interpolation="linear")) for q in quantiles})
        edges[metric] = tuple([float("-inf"), *boundaries, float("inf")])
    return RFMBins(edges)


def score_rfm(snapshot: pl.DataFrame, bins: RFMBins) -> pl.DataFrame:
    guard_no_truth_columns(snapshot)
    expressions = []
    for metric in RFM_METRICS:
        if metric not in snapshot.columns:
            raise ValueError(f"missing RFM column: {metric}")
        edges = bins.edges[metric]
        score = pl.lit(1, dtype=pl.Int8)
        for index, boundary in enumerate(edges[1:-1], start=2):
            score = pl.when(pl.col(metric) >= boundary).then(pl.lit(index)).otherwise(score)
        if metric == "recency":
            score = pl.when(pl.col(metric).is_null()).then(pl.lit(1)).otherwise(len(edges) - score)
        else:
            score = pl.when(pl.col(metric).is_null()).then(pl.lit(1)).otherwise(score)
        expressions.append(score.cast(pl.Int8).alias(f"{metric}_score"))
    return snapshot.with_columns(expressions)


def rfm_snapshot(
    dim_customer: pl.DataFrame,
    fact_transactions: pl.DataFrame,
    snapshot_date: date,
    windows: Mapping[str, int],
    *,
    bins: RFMBins | None = None,
) -> pl.DataFrame:
    snapshot_date = _snapshot_date(snapshot_date)
    windows = _validate_windows(windows)
    guard_no_truth_columns(dim_customer)
    guard_no_truth_columns(fact_transactions)
    required_dim = {"customer_id", "join_date"}
    required_txn = {"customer_id", "txn_type", "amount", "event_date_et"}
    if missing := required_dim.difference(dim_customer.columns):
        raise ValueError(f"dim_customer missing columns: {sorted(missing)}")
    if missing := required_txn.difference(fact_transactions.columns):
        raise ValueError(f"fact_transactions missing columns: {sorted(missing)}")
    if dim_customer.get_column("customer_id").n_unique() != dim_customer.height:
        raise ValueError("dim_customer customer_id must be unique")

    purchases = fact_transactions.filter(
        (pl.col("txn_type") == "purchase") & (pl.col("event_date_et") <= snapshot_date)
    )
    result = dim_customer.select("customer_id", "join_date").with_columns(
        pl.lit(snapshot_date).cast(pl.Date).alias("snapshot_date"),
        (pl.lit(snapshot_date) - pl.col("join_date")).dt.total_days().alias("days_since_join"),
    )
    for metric in RFM_METRICS:
        start = snapshot_date - timedelta(days=windows[metric] - 1)
        in_window = purchases.filter(pl.col("event_date_et") >= start)
        if metric == "recency":
            aggregate = in_window.group_by("customer_id").agg(
                (pl.lit(snapshot_date) - pl.col("event_date_et").max())
                .dt.total_days()
                .alias(metric)
            )
        elif metric == "frequency":
            aggregate = in_window.group_by("customer_id").agg(pl.len().cast(pl.Int64).alias(metric))
        else:
            aggregate = in_window.group_by("customer_id").agg(pl.col("amount").sum().alias(metric))
        result = result.join(aggregate, on="customer_id", how="left")
    result = result.with_columns(
        pl.col("frequency").fill_null(0),
        pl.col("monetary").fill_null(0.0),
    ).drop("join_date")
    return score_rfm(result, bins) if bins is not None else result


def engagement_features(fact_events: pl.DataFrame, snapshot_date: date) -> pl.DataFrame:
    snapshot_date = _snapshot_date(snapshot_date)
    guard_no_truth_columns(fact_events)
    required = {"customer_id", "event_type", "event_date_et"}
    if missing := required.difference(fact_events.columns):
        raise ValueError(f"fact_events missing columns: {sorted(missing)}")
    filtered = fact_events.filter(pl.col("event_date_et") <= snapshot_date)
    return (
        filtered.group_by("customer_id")
        .agg(
            (pl.col("event_type") == "app_open").sum().cast(pl.Int64).alias("app_open_count"),
            (pl.col("event_type") == "repayment").sum().cast(pl.Int64).alias("repayment_count"),
        )
        .with_columns(pl.lit(snapshot_date).cast(pl.Date).alias("snapshot_date"))
    )


def segment_assignment_snapshot(
    features: pl.DataFrame, model: SegmentModel, snapshot_date: date
) -> pl.DataFrame:
    snapshot_date = _snapshot_date(snapshot_date)
    guard_no_truth_columns(features)
    if "customer_id" not in features.columns:
        raise ValueError("features missing customer_id")
    clusters = model.predict(features)
    tier_map = map_clusters_to_tiers(model.centroids, model.feature_names)
    cluster_tiers = pl.Series("cluster_tier", [tier_map[value] for value in clusters])
    segments = final_tier(cluster_tiers, hard_credit_rules(features))
    return features.select("customer_id").with_columns(
        pl.lit(snapshot_date).cast(pl.Date).alias("snapshot_date"),
        segments.alias("segment"),
        pl.lit(model.version).alias("model_version"),
    )


def count_late_events(events: pl.DataFrame, previous_snapshot_date: date) -> int:
    previous_snapshot_date = _snapshot_date(previous_snapshot_date)
    required = {"event_date_et", "loaded_at"}
    if missing := required.difference(events.columns):
        raise ValueError(f"events missing columns: {sorted(missing)}")
    cutoff = snapshot_cutoff_utc(previous_snapshot_date)
    return events.filter(
        (pl.col("event_date_et") <= previous_snapshot_date) & (pl.col("loaded_at") >= cutoff)
    ).height


class GoldStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def read(self, table_name: str) -> pl.DataFrame:
        path = self.root / f"{table_name}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        return pl.read_parquet(path)

    def write_snapshot(
        self,
        table_name: str,
        frame: pl.DataFrame,
        keys: Sequence[str],
        *,
        loaded_at: datetime | None = None,
        late_event_count: int = 0,
    ) -> pl.DataFrame:
        if not table_name.isidentifier():
            raise ValueError("table_name must be an identifier")
        missing = set(keys).difference(frame.columns)
        if not keys or missing:
            raise ValueError(
                f"invalid keys; missing columns: {sorted(missing) if keys else list(keys)}"
            )
        if frame.select(list(keys)).is_duplicated().any():
            raise ValueError("incoming keys must be unique")
        if late_event_count < 0:
            raise ValueError("late_event_count must be non-negative")
        guard_no_truth_columns(frame)
        now = loaded_at or datetime.now(UTC)
        if now.tzinfo is None:
            raise ValueError("loaded_at must be timezone-aware")
        incoming = frame.drop("loaded_at", strict=False).with_columns(
            pl.lit(now.astimezone(UTC)).cast(pl.Datetime("us", "UTC")).alias("loaded_at")
        )
        path = self.root / f"{table_name}.parquet"
        if path.exists():
            current = pl.read_parquet(path)
            self._check_frozen(current, incoming, keys)
            old_keys = current.select(list(keys))
            additions = incoming.join(old_keys, on=list(keys), how="anti")
            result = pl.concat([current, additions], how="diagonal_relaxed")
        else:
            result = incoming
        result.write_parquet(path)
        if late_event_count:
            self._log_late_events(table_name, incoming, late_event_count, now)
        return result

    @staticmethod
    def _check_frozen(current: pl.DataFrame, incoming: pl.DataFrame, keys: Sequence[str]) -> None:
        shared = incoming.join(current.select(list(keys)), on=list(keys), how="semi")
        if shared.is_empty():
            return
        old = current.join(shared.select(list(keys)), on=list(keys), how="semi")
        compare = [name for name in incoming.columns if name != "loaded_at"]
        left = old.select(compare).sort(list(keys))
        right = shared.select(compare).sort(list(keys))
        if not left.equals(right, null_equal=True):
            raise ValueError("frozen snapshot rows cannot be rewritten with different content")

    def _log_late_events(
        self, table_name: str, incoming: pl.DataFrame, count: int, loaded_at: datetime
    ) -> None:
        if "snapshot_date" not in incoming.columns or incoming.is_empty():
            raise ValueError("late events require a non-empty frame with snapshot_date")
        snapshot_dates = incoming.get_column("snapshot_date").unique()
        if len(snapshot_dates) != 1:
            raise ValueError("late events must be logged for one snapshot_date")
        row = pl.DataFrame(
            {
                "table_name": [table_name],
                "snapshot_date": snapshot_dates,
                "late_event_count": [count],
                "loaded_at": [loaded_at],
            },
            schema_overrides={"loaded_at": pl.Datetime("us", "UTC")},
        )
        path = self.root / "late_events.parquet"
        log = pl.concat([pl.read_parquet(path), row]) if path.exists() else row
        log.write_parquet(path)
        LOGGER.info("%d late events included in %s for %s", count, table_name, snapshot_dates[0])

    def write_experiment_assignment(
        self,
        features: pl.DataFrame,
        experiment_id: str,
        root_seed: int,
        *,
        assigned_date: date,
        arms: Mapping[str, float] = DEFAULT_ARMS,
        loaded_at: datetime | None = None,
    ) -> pl.DataFrame:
        guard_no_truth_columns(features)
        assignment = assign(
            features,
            experiment_id,
            root_seed,
            arms,
            assigned_date=assigned_date,
        )
        return self.write_snapshot(
            "fact_experiment_assignment",
            assignment,
            ("customer_id", "experiment_id"),
            loaded_at=loaded_at,
        )
