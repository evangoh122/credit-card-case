import numpy as np
import polars as pl
from faker import Faker
from scipy.special import ndtr

from ccc.rng import entity_rng, unit_hash
from ccc.sim.scenario import CohortShift
from ccc.timeutil import sim_date

CHANNELS = ("organic", "referral", "paid", "partner")
REGIONS = ("northeast", "south", "midwest", "west")
LATENT = ("risk", "spend_rate", "responsiveness", "utilisation")

_BASE_CHANNEL_P = np.array([0.35, 0.20, 0.30, 0.15])
_REGION_P = np.array([0.18, 0.38, 0.21, 0.23])
_CHANNEL_RISK_Z = {"organic": 0.0, "referral": -0.25, "paid": 0.35, "partner": 0.1}
_CHANNEL_RESP_Z = {"organic": 0.0, "referral": 0.1, "paid": 0.25, "partner": -0.1}
_REGION_SPEND_Z = {"northeast": 0.15, "south": -0.05, "midwest": -0.1, "west": 0.1}

# order: risk, spend, responsiveness, utilisation
_CORR = np.array(
    [
        [1.0, -0.20, -0.30, 0.50],
        [-0.20, 1.0, 0.15, 0.30],
        [-0.30, 0.15, 1.0, 0.00],
        [0.50, 0.30, 0.00, 1.0],
    ]
)
_CHOL = np.linalg.cholesky(_CORR)

SPEND_MEDIAN = 0.35
SPEND_SIGMA = 0.6


def channel_probs(cohort_day: int, shift: CohortShift) -> np.ndarray:
    paid = min(0.8, _BASE_CHANNEL_P[2] + shift.paid_share_per_day * cohort_day)
    rest = np.delete(_BASE_CHANNEL_P, 2)
    rest = rest / rest.sum() * (1 - paid)
    return np.insert(rest, 2, paid)


def initial_join_day(seed: int, customer_id: str) -> int:
    return -1 - int(entity_rng(seed, "arrivals", customer_id, -2).random() * 365)


def handwritten_profile(
    seed: int, customer_id: str, cohort_day: int, shift: CohortShift
) -> dict[str, float | str]:
    rng = entity_rng(seed, "arrivals", customer_id, -1)
    channel = CHANNELS[rng.choice(4, p=channel_probs(cohort_day, shift))]
    region = REGIONS[rng.choice(4, p=_REGION_P)]
    z = _CHOL @ rng.standard_normal(4)
    risk = ndtr(z[0] + _CHANNEL_RISK_Z[channel] + shift.risk_z_per_day * cohort_day)
    spend = SPEND_MEDIAN * np.exp(SPEND_SIGMA * (z[1] + _REGION_SPEND_Z[region]))
    resp = ndtr(z[2] + _CHANNEL_RESP_Z[channel])
    return {
        "channel": channel,
        "region": region,
        "risk": float(risk),
        "spend_rate": float(spend),
        "responsiveness": float(resp),
        "utilisation": float(ndtr(z[3])),
    }


def handwritten_table(seed: int, specs: list[tuple[str, int]], shift: CohortShift) -> pl.DataFrame:
    """specs: (customer_id, cohort_day). Each row depends only on its own id."""
    rows = [
        {"customer_id": cid, **handwritten_profile(seed, cid, day, shift)} for cid, day in specs
    ]
    return pl.DataFrame(rows, schema=_PROFILE_SCHEMA)


_PROFILE_SCHEMA = {
    "customer_id": pl.String,
    "channel": pl.String,
    "region": pl.String,
    "risk": pl.Float64,
    "spend_rate": pl.Float64,
    "responsiveness": pl.Float64,
    "utilisation": pl.Float64,
}


def copula_metadata():
    from sdv.metadata import Metadata

    m = Metadata()
    m.add_table("seed")
    for col in LATENT:
        m.add_column(col, table_name="seed", sdtype="numerical")
    for col in ("channel", "region"):
        m.add_column(col, table_name="seed", sdtype="categorical")
    return m


def fit_copula(seed_table: pl.DataFrame):
    import pandas as pd
    from sdv.single_table import GaussianCopulaSynthesizer

    cols = [*LATENT, "channel", "region"]
    synth = GaussianCopulaSynthesizer(copula_metadata())
    synth.fit({"seed": pd.DataFrame(seed_table.select(cols).to_dict(as_series=False))})
    return synth


def _copula_rows(synth, seed: int, ids: list[str]) -> pl.DataFrame:
    """Per-customer draws through the fitted copula: seeded normals, correlated with the
    fitted matrix, mapped through the fitted marginals, then the SDV reverse transform."""
    import pandas as pd
    from scipy.special import ndtr as cdf

    model = synth._model
    chol = np.linalg.cholesky(np.asarray(model.correlation))
    normals = np.array(
        [entity_rng(seed, "arrivals", c, -4).standard_normal(len(chol)) for c in ids]
    )
    u = cdf(normals @ chol.T)
    processed = pd.DataFrame(
        {
            col: uni.percent_point(u[:, i])
            for i, (col, uni) in enumerate(zip(model.columns, model.univariates, strict=True))
        }
    )
    original = synth._data_processor.reverse_transform(processed)
    return pl.DataFrame(original.to_dict("list"))


def copula_table(
    synth, seed: int, specs: list[tuple[str, int]], shift: CohortShift
) -> pl.DataFrame:
    """A customer's row depends only on (seed, customer_id, cohort_day) and the fitted copula."""
    sampled = _copula_rows(synth, seed, [c for c, _ in specs])
    df = sampled.with_columns(
        pl.Series("customer_id", [c for c, _ in specs]),
        pl.Series("cohort_day", [d for _, d in specs]),
    )
    flips = np.array([entity_rng(seed, "arrivals", c, -3).random() for c, _ in specs])
    days = df["cohort_day"].to_numpy()
    p_paid = np.minimum(0.8, shift.paid_share_per_day * days)
    to_paid = (flips < p_paid) & (df["channel"].to_numpy() != "paid")
    risk = np.clip(df["risk"].to_numpy() + 0.4 * shift.risk_z_per_day * days, 0.001, 0.999)
    return df.with_columns(
        pl.when(pl.Series(to_paid))
        .then(pl.lit("paid"))
        .otherwise(pl.col("channel"))
        .alias("channel"),
        pl.Series("risk", risk),
        pl.col("spend_rate").clip(lower_bound=0.01),
        pl.col("responsiveness").clip(0.0, 1.0),
        pl.col("utilisation").clip(0.0, 1.0),
    ).select(list(_PROFILE_SCHEMA))


def identities(
    seed: int, ids_join_day: list[tuple[str, int]], start, locale: str = "en_US"
) -> pl.DataFrame:
    fake = Faker(locale)
    names = []
    for cid, _ in ids_join_day:
        fake.seed_instance(int(unit_hash(seed, "faker", cid) * 2**32))
        names.append(fake.name())
    return pl.DataFrame(
        {
            "customer_id": [c for c, _ in ids_join_day],
            "name": names,
            "join_day": [j for _, j in ids_join_day],
            "join_date": [sim_date(start, j) for _, j in ids_join_day],
        },
        schema={
            "customer_id": pl.String,
            "name": pl.String,
            "join_day": pl.Int64,
            "join_date": pl.Date,
        },
    )
