from datetime import date

import pytest

from ccc.sim.scenario import Campaign, MacroShock, Scenario, Segment

START = date(2026, 10, 19)


@pytest.fixture
def small_scenario():
    return Scenario(
        name="small",
        start_date=START,
        initial_customers=150,
        arrivals_per_day=4.0,
        campaigns=(
            Campaign(
                name="promo",
                start_et="2026-10-21T09:00:00",
                duration_days=6,
                lift=0.4,
                arrival_boost=0.5,
            ),
        ),
        macro_shock=MacroShock(start_date=date(2026, 11, 3), miss_mult=3.0, ramp_days=1),
    )


@pytest.fixture
def recovery_scenario():
    return Scenario(
        name="recovery",
        start_date=START,
        initial_customers=800,
        arrivals_per_day=0.0,
        arrivals_trend=0.0,
        campaigns=(
            Campaign(
                name="real",
                start_et="2026-10-20T09:00:00",
                duration_days=10,
                lift=0.5,
                segment=Segment(risk_max=0.8),
            ),
            Campaign(name="nothing", start_et="2026-10-20T09:00:00", duration_days=10, lift=0.0),
        ),
    )
