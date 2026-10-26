import json

import numpy as np
import polars as pl
import pytest

from ccc.segmentation import (
    FEATURE_COLUMNS,
    LEAKAGE_PATTERNS,
    MODEL_VERSION,
    RFM_COLUMNS,
    RuleThresholds,
    SegmentModel,
    _make_version,
    choose_k,
    experiment_eligible,
    final_tier,
    fit_segmenter,
    fit_tiered,
    hard_credit_rules,
    map_clusters_to_tiers,
    rule_based_tiers,
)

CENTERS = {
    "Prime": (0.0, 0.3, 2000.0, 40.0),
    "Watch": (10.0, 0.6, 800.0, 20.0),
    "Nurture": (0.0, 0.05, 50.0, 2.0),
    "Decline": (500.0, 1.2, 400.0, 10.0),
}
TIERS_ALL = ("Prime", "Watch", "Nurture", "Decline")
THRESHOLDS = RuleThresholds(arrears=0.0, utilisation=1.0, spend=500.0, activity=5.0)


def blob_data(seed=0, per_cluster=60):
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for label, (arrears, utilisation, spend, activity) in CENTERS.items():
        for _ in range(per_cluster):
            rows.append(
                (
                    arrears + rng.normal(0, 4.0),
                    utilisation + rng.normal(0, 0.02),
                    spend + rng.normal(0, 25.0),
                    activity + rng.normal(0, 1.0),
                )
            )
            labels.append(label)
    features = pl.DataFrame(rows, schema=list(FEATURE_COLUMNS), orient="row")
    return features, pl.Series("true_tier", labels)


def test_blobs_recover_known_clusters():
    features, truth = blob_data()
    model = fit_segmenter(features, 4)
    pred = model.predict(features).to_numpy()
    labels = truth.to_numpy()
    cluster_to_blob = {}
    for cluster in range(4):
        seen = set(labels[pred == cluster].tolist())
        assert len(seen) == 1
        cluster_to_blob[cluster] = seen.pop()
    assert sorted(cluster_to_blob.values()) == sorted(CENTERS)
    tiers = map_clusters_to_tiers(model.centroids, model.feature_names)
    for cluster, blob in cluster_to_blob.items():
        assert tiers[cluster] == blob


def test_choose_k_recovers_blob_count():
    features, _ = blob_data()
    table, best = choose_k(features, range(2, 7))
    assert table.columns == ["k", "silhouette"]
    assert table["k"].to_list() == [2, 3, 4, 5, 6]
    assert all(-1.0 <= v <= 1.0 for v in table["silhouette"].to_list())
    assert best == 4
    assert table["silhouette"][table["k"].to_list().index(best)] == table["silhouette"].max()


def test_save_load_roundtrip_identical_predictions(tmp_path):
    features, _ = blob_data()
    model = fit_segmenter(features, 4)
    query = features.head(20)
    before = model.predict(query).to_list()
    path = tmp_path / "model.json"
    model.save(path)
    payload = json.loads(path.read_text())
    assert payload["version"] == model.version
    assert payload["version"].startswith(f"{MODEL_VERSION}+")
    assert tuple(payload["feature_names"]) == model.feature_names
    loaded = SegmentModel.load(path)
    assert loaded == model
    assert loaded.predict(query).to_list() == before


def test_model_unaffected_by_fit_data_changes(tmp_path):
    features1, _ = blob_data(seed=1)
    features2, _ = blob_data(seed=2)
    model = fit_segmenter(features1, 4)
    query = pl.DataFrame(
        {
            "arrears": [5.0, 0.0, 300.0],
            "utilisation": [0.4, 0.1, 1.1],
            "spend": [1500.0, 10.0, 600.0],
            "activity": [30.0, 1.0, 12.0],
        }
    )
    expected = model.predict(query).to_list()
    path = tmp_path / "model.json"
    model.save(path)
    loaded = SegmentModel.load(path)
    other = fit_segmenter(features2, 4)
    assert other.version != model.version
    features1.drop_in_place("spend")
    assert model.predict(query).to_list() == expected
    assert loaded.predict(query).to_list() == expected


def test_predict_uses_saved_scaler_and_centroids_only():
    features, _ = blob_data()
    model = fit_segmenter(features, 4)
    query = pl.DataFrame(
        {
            "arrears": [30.0, 250.0, 0.0, 480.0],
            "utilisation": [0.85, 0.75, 0.2, 1.15],
            "spend": [1500.0, 1200.0, 1800.0, 350.0],
            "activity": [32.0, 25.0, 36.0, 8.0],
        }
    )
    x = query.to_numpy()
    z = (x - np.asarray(model.mean)) / np.asarray(model.scale)
    d2 = ((z[:, None, :] - np.asarray(model.centroids)[None, :, :]) ** 2).sum(axis=2)
    assert model.predict(query).to_list() == d2.argmin(axis=1).tolist()


def test_predict_independent_of_batch_composition():
    features, _ = blob_data()
    model = fit_segmenter(features, 4)
    query = features[::20]
    batch = model.predict(query).to_list()
    assert len(set(batch)) == 4
    per_row = [model.predict(query.slice(i, 1)).item() for i in range(len(query))]
    assert per_row == batch


def test_hard_rule_overrides_cluster():
    df = pl.DataFrame(
        {
            "arrears": [0.0, 120.0, 0.0, 0.0],
            "over_limit": [False, False, True, False],
            "days_active": [400, 400, 400, 400],
        }
    )
    cluster_tier = pl.Series("cluster_tier", ["Prime", "Prime", "Watch", "Nurture"])
    out = final_tier(cluster_tier, hard_credit_rules(df))
    assert out.to_list() == ["Prime", "Decline", "Decline", "Nurture"]


def test_new_customers_are_nurture():
    df = pl.DataFrame(
        {
            "arrears": [0.0, 0.0, 0.0, 50.0],
            "over_limit": [False, False, False, False],
            "days_active": [1, 29, 30, 5],
        }
    )
    cluster_tier = pl.Series("cluster_tier", ["Prime", "Prime", "Prime", "Prime"])
    out = final_tier(cluster_tier, hard_credit_rules(df))
    assert out.to_list() == ["Nurture", "Nurture", "Prime", "Decline"]


def test_decline_never_experiment_eligible():
    tiers = pl.Series("final_tier", ["Prime", "Watch", "Nurture", "Decline"])
    assert experiment_eligible(tiers).to_list() == [True, True, True, False]
    features, _ = blob_data()
    model = fit_segmenter(features, 4)
    labels = map_clusters_to_tiers(model.centroids, model.feature_names)
    cluster_tier = pl.Series("cluster_tier", [labels[c] for c in model.predict(features).to_list()])
    df = features.with_columns(pl.lit(400).alias("days_active"), pl.lit(False).alias("over_limit"))
    out = final_tier(cluster_tier, hard_credit_rules(df))
    eligible = experiment_eligible(out).to_list()
    assert any(t == "Decline" for t in out.to_list())
    assert not any(t == "Decline" and e for t, e in zip(out.to_list(), eligible, strict=True))


def test_rule_based_tiers():
    features = pl.DataFrame(
        {
            "arrears": [0.0, 0.0, 0.0, 0.0, 0.0],
            "utilisation": [0.4, 0.5, 1.1, 0.1, 0.2],
            "spend": [2000.0, 400.0, 300.0, 20.0, 700.0],
            "activity": [30, 15, 10, 1, 3],
        }
    )
    tiers = rule_based_tiers(features, THRESHOLDS)
    assert tiers.to_list() == ["Prime", "Watch", "Decline", "Nurture", "Watch"]
    decline = rule_based_tiers(
        features.head(1).with_columns(pl.lit(50.0).alias("arrears")), THRESHOLDS
    )
    assert decline.to_list() == ["Decline"]


def test_rfm_features_optional():
    features, _ = blob_data()
    with_rfm = features.with_columns(
        recency=pl.Series([10.0] * len(features)),
        frequency=pl.Series([3.0] * len(features)),
        monetary=pl.Series([400.0] * len(features)),
    )
    base = fit_segmenter(features, 4)
    assert base.feature_names == FEATURE_COLUMNS
    assert base.predict(with_rfm).len() == len(features)
    full = fit_segmenter(with_rfm, 4, include_rfm=True)
    assert full.feature_names == FEATURE_COLUMNS + RFM_COLUMNS
    assert full.predict(with_rfm).len() == len(features)
    with pytest.raises(ValueError):
        fit_segmenter(features, 4, include_rfm=True)
    with pytest.raises(ValueError):
        choose_k(features, [2, 3, 4], include_rfm=True)


@pytest.mark.parametrize(
    "bad_column",
    [
        "offer",
        "offer_date",
        "post_spend",
        "post_utilisation",
        "OfferAccepted",
        "POST_spend",
        "postOfferSpend",
        "accepted_offer",
        "Response_flag",
        "outcome",
        "treatment_arm",
        "ExposureDays",
        "preoffer_spend",  # policy: pre-offer names are rejected, the name cannot prove timing
    ],
)
def test_rejects_post_offer_columns(bad_column):
    features, _ = blob_data()
    bad = features.with_columns(pl.lit(1.0).alias(bad_column))
    with pytest.raises(ValueError):
        fit_segmenter(bad, 4)
    with pytest.raises(ValueError):
        choose_k(bad, [2, 3, 4])
    with pytest.raises(ValueError):
        rule_based_tiers(bad, THRESHOLDS)
    model = fit_segmenter(features, 4)
    with pytest.raises(ValueError):
        model.predict(bad)
    credit = pl.DataFrame(
        {"arrears": [0.0], "over_limit": [False], "days_active": [400]}
    ).with_columns(pl.lit(1.0).alias(bad_column))
    with pytest.raises(ValueError):
        hard_credit_rules(credit)


def test_input_validation():
    features, _ = blob_data()
    for bad_k in (0, 1, -3, 2.0, True, len(features) + 1):
        with pytest.raises(ValueError):
            fit_segmenter(features, bad_k)
    for bad_k in (1, len(features)):
        with pytest.raises(ValueError):
            choose_k(features, [2, 4, bad_k])
    with pytest.raises(ValueError):
        choose_k(features, [])
    model = fit_segmenter(features, 4)
    with pytest.raises(ValueError):
        map_clusters_to_tiers(model.centroids[:3], model.feature_names)
    with pytest.raises(ValueError):
        map_clusters_to_tiers(model.centroids, model.feature_names[:-1])
    with pytest.raises(ValueError):
        fit_segmenter(features.drop("spend"), 4)
    with pytest.raises(ValueError):
        fit_segmenter(features.with_columns(pl.lit(None).alias("spend")), 4)
    with pytest.raises(TypeError):
        fit_segmenter(features.with_columns(pl.lit(True).alias("spend")), 4)
    with pytest.raises(ValueError):
        final_tier(pl.Series([None]), pl.Series([None]))
    with pytest.raises(ValueError):
        final_tier(pl.Series(["Prime"]), pl.Series(["Decline", None]))
    with pytest.raises(ValueError):
        final_tier(pl.Series(["Primed"]), pl.Series([None]))
    with pytest.raises(ValueError):
        experiment_eligible(pl.Series([None]))
    with pytest.raises(ValueError):
        experiment_eligible(pl.Series(["Excluded"]))
    with pytest.raises(TypeError):
        hard_credit_rules(pl.DataFrame({"arrears": [0.0], "over_limit": [1], "days_active": [400]}))
    with pytest.raises(ValueError):
        hard_credit_rules(
            pl.DataFrame({"arrears": [0.0], "over_limit": [False], "days_active": [None]})
        )
    with pytest.raises(ValueError):
        hard_credit_rules(pl.DataFrame({"arrears": [0.0]}))


@pytest.mark.parametrize("benign", ["postcode", "tenure_months", "region", "customer_id"])
def test_benign_column_names_pass(benign):
    features, _ = blob_data()
    assert not any(p in benign.lower() for p in LEAKAGE_PATTERNS)
    extra = features.with_columns(pl.lit(1.0).alias(benign))
    assert fit_segmenter(extra, 4).predict(extra).len() == len(features)


def test_choose_k_range_must_include_four():
    features, _ = blob_data()
    with pytest.raises(ValueError, match="must include 4"):
        choose_k(features, [2, 3, 5])


def test_end_to_end_choose_k_fit_map():
    features, truth = blob_data()
    table, best = choose_k(features, range(2, 7))
    assert 4 in table["k"].to_list()
    assert best == 4
    model = fit_segmenter(features, best)
    tiers = map_clusters_to_tiers(model.centroids, model.feature_names)
    assert sorted(tiers) == sorted(TIERS_ALL)
    per_customer = [tiers[c] for c in model.predict(features).to_list()]
    assert per_customer == truth.to_list()
    fitted, fitted_tiers = fit_tiered(features)
    assert fitted == model
    assert fitted_tiers == tiers


def test_tiers_independent_of_cluster_permutation():
    features, _ = blob_data()
    model = fit_segmenter(features, 4)
    base = map_clusters_to_tiers(model.centroids, model.feature_names)
    base_per_customer = [base[c] for c in model.predict(features).to_list()]
    for perm in ([3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]):
        centroids = [model.centroids[i] for i in perm]
        tiers = map_clusters_to_tiers(centroids, model.feature_names)
        permuted = SegmentModel(
            version=_make_version(model.feature_names, model.mean, model.scale, centroids),
            feature_names=model.feature_names,
            mean=model.mean,
            scale=model.scale,
            centroids=tuple(centroids),
        )
        got = [tiers[c] for c in permuted.predict(features).to_list()]
        assert got == base_per_customer


def test_tier_ties_do_not_depend_on_index():
    # Two clusters tie exactly on engagement; the result must follow centroid values.
    names = FEATURE_COLUMNS
    rows = [
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 0.0, -1.0, 1.0),
        (0.0, 0.0, 1.0, -1.0),
        (5.0, 5.0, 0.0, 0.0),
    ]
    base = dict(zip(rows, map_clusters_to_tiers(rows, names), strict=True))
    for perm in ([3, 1, 0, 2], [2, 3, 1, 0], [1, 0, 3, 2]):
        shuffled = [rows[i] for i in perm]
        got = dict(zip(shuffled, map_clusters_to_tiers(shuffled, names), strict=True))
        assert got == base


def test_degenerate_clustering_rejected():
    identical = pl.DataFrame(
        {name: [1.0] * 20 for name in FEATURE_COLUMNS},
    )
    with pytest.raises(ValueError, match="distinct"):
        fit_segmenter(identical, 4)
    three = pl.DataFrame(
        {
            "arrears": [0.0, 1.0, 2.0] * 10,
            "utilisation": [0.1, 0.2, 0.3] * 10,
            "spend": [10.0, 20.0, 30.0] * 10,
            "activity": [1.0, 2.0, 3.0] * 10,
        }
    )
    with pytest.raises(ValueError, match="distinct"):
        fit_segmenter(three, 4)
    with pytest.raises(ValueError, match="duplicate centroids"):
        map_clusters_to_tiers(
            [(0, 0, 0, 0), (0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2)], FEATURE_COLUMNS
        )
    with pytest.raises(ValueError, match="duplicate centroids"):
        map_clusters_to_tiers(
            [(0, 0, 0, 0), (0, 0, 0, 1e-6), (1, 1, 1, 1), (2, 2, 2, 2)], FEATURE_COLUMNS
        )


def test_zero_variance_column_is_allowed_with_unit_scale():
    features, _ = blob_data()
    flat = features.with_columns(pl.lit(7.0).alias("activity"))
    model = fit_segmenter(flat, 4)
    assert model.scale[FEATURE_COLUMNS.index("activity")] == 1.0
    assert len(set(model.predict(flat).to_list())) == 4
    shifted = flat.with_columns(pl.lit(99.0).alias("activity"))
    # the constant column adds the same distance to every centroid, so labels are unchanged
    assert shifted.pipe(model.predict).to_list() == model.predict(flat).to_list()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("column", ["arrears", "days_active"])
def test_hard_credit_rules_reject_non_finite(bad, column):
    df = pl.DataFrame(
        {"arrears": [0.0, 1.0], "over_limit": [False, False], "days_active": [400.0, 5.0]}
    )
    df = df.with_columns(pl.Series(column, [bad, 1.0]))
    with pytest.raises(ValueError, match="finite"):
        hard_credit_rules(df)


@pytest.mark.parametrize("field", ["scale", "mean", "centroids"])
def test_tampered_model_json_rejected(tmp_path, field):
    features, _ = blob_data()
    model = fit_segmenter(features, 4)
    path = tmp_path / "model.json"
    model.save(path)
    payload = json.loads(path.read_text())
    if field == "centroids":
        payload[field][0][0] += 0.5
    else:
        payload[field][0] *= 1.01
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid segment model file"):
        SegmentModel.load(path)
    payload["version"] = "seg-1.0.0+deadbeef0000"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid segment model file"):
        SegmentModel.load(path)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), "5", None, True])
def test_rule_thresholds_reject_bad_values(bad):
    for name in ("arrears", "utilisation", "spend", "activity"):
        kwargs = {"arrears": 0.0, "utilisation": 1.0, "spend": 500.0, "activity": 5.0}
        kwargs[name] = bad
        with pytest.raises(ValueError):
            RuleThresholds(**kwargs)
