from datetime import date

import numpy as np
import polars as pl
import pytest

from ccc.sim import profiles
from ccc.sim.run import NO_SHIFT
from ccc.sim.scenario import CohortShift


def table(seed, ids, day=0, shift=NO_SHIFT):
    return profiles.handwritten_table(seed, [(i, day) for i in ids], shift)


def test_profile_ranges_and_categories():
    df = table(1, [f"c{i}" for i in range(500)])
    for col in ("risk", "responsiveness", "utilisation"):
        assert df[col].min() > 0 and df[col].max() < 1
    assert df["spend_rate"].min() > 0
    assert set(df["channel"]) <= set(profiles.CHANNELS)
    assert set(df["region"]) <= set(profiles.REGIONS)


def test_latent_correlations_have_expected_signs():
    df = table(2, [f"c{i}" for i in range(4000)])
    corr = df.select(profiles.LATENT).to_numpy().T
    c = np.corrcoef(corr)
    assert c[0, 3] > 0.3
    assert c[0, 2] < -0.1
    assert c[1, 3] > 0.1


def test_profile_depends_only_on_own_id():
    alone = table(3, ["c1"])
    crowd = table(3, ["c0", "c1", "c2"]).filter(pl.col("customer_id") == "c1")
    assert alone.equals(crowd)
    assert not alone.equals(table(4, ["c1"]))


def test_later_cohorts_shift_to_paid_and_higher_risk():
    shift = CohortShift(paid_share_per_day=0.006, risk_z_per_day=0.015)
    ids = [f"c{i}" for i in range(3000)]
    early, late = table(5, ids, 0, shift), table(5, ids, 55, shift)
    share = lambda df: (df["channel"] == "paid").mean()  # noqa: E731
    assert share(late) > share(early) + 0.15
    assert late["risk"].mean() > early["risk"].mean() + 0.1


def test_channel_probs_are_a_distribution():
    p = profiles.channel_probs(1000, CohortShift(0.01, 0.0))
    assert p.sum() == pytest.approx(1.0) and p.max() <= 0.8 + 1e-9


def test_identities_are_seeded_and_dated():
    ids = [("c1", -10), ("c2", 3)]
    a = profiles.identities(7, ids, date(2026, 10, 19))
    assert a.equals(profiles.identities(7, ids, date(2026, 10, 19)))
    assert a["join_date"].to_list() == [date(2026, 10, 9), date(2026, 10, 22)]
    assert not a["name"].equals(profiles.identities(8, ids, date(2026, 10, 19))["name"])
    assert a["name"][0] == profiles.identities(7, ids[:1], date(2026, 10, 19))["name"][0]


@pytest.mark.slow
def test_copula_cohort_matches_seed_table_and_shifts():
    seed_table = table(9, [f"s{i}" for i in range(1500)])
    synth = profiles.fit_copula(seed_table)
    ids = [f"c{i}" for i in range(1500)]
    shift = CohortShift(paid_share_per_day=0.005, risk_z_per_day=0.01)
    day0 = profiles.copula_table(synth, 9, [(i, 0) for i in ids], shift)
    again = profiles.copula_table(synth, 9, [(i, 0) for i in ids], shift)
    late = profiles.copula_table(synth, 9, [(i, 50) for i in ids], shift)
    assert day0.columns == list(profiles._PROFILE_SCHEMA)
    assert day0.equals(again)
    assert abs(day0["risk"].mean() - seed_table["risk"].mean()) < 0.05
    seed_corr = np.corrcoef(seed_table["risk"], seed_table["utilisation"])[0, 1]
    assert abs(np.corrcoef(day0["risk"], day0["utilisation"])[0, 1] - seed_corr) < 0.1
    assert (late["channel"] == "paid").mean() > (day0["channel"] == "paid").mean() + 0.1
