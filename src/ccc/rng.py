import hashlib
from datetime import date

import numpy as np

STREAMS = ("arrivals", "behavior", "campaign_response")


def _digest(*parts: object) -> bytes:
    return hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).digest()


def entity_rng(
    root_seed: int, stream: str, customer_id: str | int, day: int | date = 0
) -> np.random.Generator:
    """Generator keyed only on (root_seed, stream, customer_id, day)."""
    if stream not in STREAMS:
        raise ValueError(f"unknown stream {stream!r}; expected one of {STREAMS}")
    words = np.frombuffer(_digest(root_seed, stream, customer_id, day), dtype=np.uint32)
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(words.tolist())))


def unit_hash(root_seed: int, name: str, customer_id: str | int) -> float:
    """Stable value in [0, 1) for stratified assignment."""
    return int.from_bytes(_digest(root_seed, name, customer_id)[:8], "big") / 2**64
