from datetime import UTC, date, datetime, timedelta

import pytest

from ccc.mlops.seed import MAX_SEED, derive_seed, validate_run_date

DAY = date(2026, 11, 3)


def test_seed_is_deterministic_and_varies_with_inputs():
    assert derive_seed(DAY, 7) == derive_seed(DAY, 7)
    assert derive_seed(DAY, 7) != derive_seed(DAY + timedelta(days=1), 7)
    assert derive_seed(DAY, 7) != derive_seed(DAY, 8)
    assert 0 <= derive_seed(DAY, 7) <= MAX_SEED


def test_seed_rejects_invalid_inputs():
    with pytest.raises(TypeError, match="run_date"):
        derive_seed("2026-11-03", 1)
    with pytest.raises(ValueError, match="root_seed"):
        derive_seed(DAY, -1)
    with pytest.raises(ValueError, match="root_seed"):
        derive_seed(DAY, True)


def test_seed_rejects_datetime_run_dates():
    with pytest.raises(TypeError, match="datetime"):
        derive_seed(datetime(2026, 11, 3, 12, 30), 7)
    with pytest.raises(TypeError, match="datetime"):
        derive_seed(datetime(2026, 11, 3, 12, 30, tzinfo=UTC), 7)


def test_validate_run_date_returns_plain_dates():
    assert validate_run_date(DAY) is DAY
