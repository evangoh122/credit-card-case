"""Deterministic seeds for scheduled MLOps runs."""

import hashlib
from datetime import date

MAX_SEED = 2**63 - 1


def derive_seed(run_date: date, root_seed: int) -> int:
    """Return a stable 63-bit seed keyed on the run date and root seed.

    Re-running the same ``run_date`` with the same ``root_seed`` always yields the
    same value, while a different date or root seed yields a different stream.
    """
    if isinstance(run_date, bool) or not isinstance(run_date, date):
        raise TypeError("run_date must be a date")
    if isinstance(root_seed, bool) or not isinstance(root_seed, int) or root_seed < 0:
        raise ValueError("root_seed must be a non-negative integer")
    payload = f"{run_date.isoformat()}|{root_seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & MAX_SEED
