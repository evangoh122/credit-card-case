import math

import numpy as np
import polars as pl
import pytest

from ccc.drift import (
    Reference,
    cluster_fit,
    drift_report,
    make_reference,
    psi_categorical,
    psi_numeric,
    psi_status,
    segment_mix_drift,
    structure_agreement,
    sustained_alert,
)


@pytest.fixture
def train():
    rng = np.random.default_rng(0)
    n = 5000
    return pl.DataFrame(
        {
            "spend": rng.normal(100, 15, n),
            "channel": rng.choice(["organic", "paid", "referral"], n, p=[0.5, 0.3, 0.2]),
            "tier": rng.choice(["prime", "watch"], n),
            "dist": rng.gamma(4, 0.5, n),
        }
    )


@pytest.fixture
def ref(train):
    return make_reference(train, ["spend"], ["channel"], segment_col="tier", distance_col="dist")


def test_identical_data_psi_near_zero(train, ref):
    assert psi_numeric(ref, "spend", train["spend"]) < 1e-3
    assert psi_categorical(ref, "channel", train["channel"]) < 1e-9
    assert psi_status(psi_numeric(ref, "spend", train["spend"])) == "ok"


def test_shifted_mean_alerts(train, ref):
    shifted = train["spend"] + 20
    psi = psi_numeric(ref, "spend", shifted)
    assert psi > 0.25
    assert psi_status(psi) == "alert"


def test_hand_computed_psi():
    df = pl.DataFrame({"x": [float(i) for i in range(1, 11)]})
    ref = make_reference(df, ["x"], n_bins=2)
    assert ref.numeric_edges["x"] == [5.5]
    current = np.array([1, 2, 3, 4, 5, 1, 2, 3, 9, 10], dtype=float)
    expected = 0.3 * math.log(0.8 / 0.5) + (-0.3) * math.log(0.2 / 0.5)
    assert psi_numeric(ref, "x", current) == pytest.approx(expected)


def test_out_of_range_lands_in_end_bins():
    df = pl.DataFrame({"x": [float(i) for i in range(1, 11)]})
    ref = make_reference(df, ["x"], n_bins=2)
    far = np.array([-1e9] * 5 + [1e9] * 5)
    assert psi_numeric(ref, "x", far) == pytest.approx(0.0)


def test_unseen_category_counts_as_drift(ref):
    seen = pl.Series(["organic"] * 50 + ["paid"] * 30 + ["referral"] * 20)
    with_new = pl.Series(["organic"] * 40 + ["paid"] * 30 + ["referral"] * 10 + ["tiktok"] * 20)
    assert psi_categorical(ref, "channel", seen) < 0.01
    psi = psi_categorical(ref, "channel", with_new)
    assert math.isfinite(psi) and psi > 0.25


def test_status_thresholds():
    assert psi_status(0.0999) == "ok"
    assert psi_status(0.10) == "watch"
    assert psi_status(0.25) == "watch"
    assert psi_status(0.2501) == "alert"


def mix_ref(a_share):
    labels = ["a"] * int(a_share * 1000) + ["b"] * int((1 - a_share) * 1000)
    return make_reference(pl.DataFrame({"s": labels}), [], segment_col="s")


def test_mix_shift_of_exactly_five_points_flagged():
    ref = mix_ref(0.5)
    flagged = segment_mix_drift(ref, ["a"] * 550 + ["b"] * 450)["flagged"]
    assert flagged == ["a", "b"]


def test_mix_shift_of_4_9_points_not_flagged():
    ref = mix_ref(0.5)
    assert segment_mix_drift(ref, ["a"] * 549 + ["b"] * 451)["flagged"] == []


def test_mix_psi_zero_when_unchanged():
    ref = mix_ref(0.5)
    assert segment_mix_drift(ref, ["a"] * 500 + ["b"] * 500)["psi"] == pytest.approx(0.0)


def test_cluster_fit_far_share_and_mean(train, ref):
    same = cluster_fit(ref, train["dist"])
    assert same["mean_ratio"] == pytest.approx(1.0)
    assert same["far_share"] == pytest.approx(0.05, abs=0.005)
    worse = cluster_fit(ref, train["dist"] * 2)
    assert worse["mean_ratio"] == pytest.approx(2.0)
    assert worse["far_share"] > 0.4


def blobs(seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [10, 0], [0, 10]])
    X = np.vstack([c + rng.normal(0, 0.5, (100, 2)) for c in centers])
    return X, np.repeat([0, 1, 2], 100)


def test_ari_one_for_same_structure():
    X, labels = blobs()
    assert structure_agreement(X, labels) == pytest.approx(1.0)
    assert structure_agreement(X, (labels + 1) % 3) == pytest.approx(1.0)


def test_ari_low_for_unrelated_labels():
    X, _ = blobs()
    random_labels = np.random.default_rng(1).integers(0, 3, len(X))
    assert structure_agreement(X, random_labels) < 0.1


def test_single_noisy_period_not_alerted():
    assert not sustained_alert(["ok", "ok", "alert"])
    assert not sustained_alert(["alert", "ok", "alert"])
    assert not sustained_alert(["alert", "alert"])


def test_three_consecutive_alerts_fire():
    assert sustained_alert(["ok", "alert", "alert", "alert"])
    assert not sustained_alert(["alert", "alert", "watch"])


def test_reference_roundtrip(tmp_path, train, ref):
    path = tmp_path / "ref.json"
    ref.save(path)
    assert Reference.load(path) == ref


def test_report_flags_shift_without_refit(train, ref):
    ok = drift_report(ref, train, "tier", "dist")
    assert ok["alerts"] == []
    bad = drift_report(ref, train.with_columns(pl.col("spend") + 30), "tier", "dist")
    assert bad["alerts"] == ["spend"]
