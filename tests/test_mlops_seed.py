from datetime import date, timedelta

import pytest

from ccc.mlops.seed import MAX_SEED, derive_seed

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
