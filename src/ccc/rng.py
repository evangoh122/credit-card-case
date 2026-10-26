import hashlib
from datetime import date

import numpy as np

STREAMS = ("arrivals", "behavior", "campaign_response")


def _encode(part: int | str | date) -> bytes:
    match part:
        case bool():
            raise TypeError("bool is not a valid key part")
        case int():
            tag, body = b"i", str(part).encode()
        case str():
            tag, body = b"s", part.encode()
        case date():
            tag, body = b"d", part.isoformat().encode()
        case _:
            raise TypeError(f"unsupported key part type {type(part).__name__}")
    return tag + len(body).to_bytes(8, "big") + body


def _digest(*parts: int | str | date) -> bytes:
    return hashlib.sha256(b"".join(_encode(p) for p in parts)).digest()


def entity_rng(
    root_seed: int, stream: str, customer_id: str | int, day: int | date = 0
) -> np.random.Generator:
    """Generator keyed only on (root_seed, stream, customer_id, day)."""
    if stream not in STREAMS:
        raise ValueError(f"unknown stream {stream!r}; expected one of {STREAMS}")
    words = np.frombuffer(_digest(root_seed, stream, customer_id, day), dtype="<u4")
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(words.tolist())))


def unit_hash(root_seed: int, name: str, customer_id: str | int) -> float:
    """Stable value in [0, 1) for stratified assignment."""
    return int.from_bytes(_digest(root_seed, name, customer_id)[:8], "big") / 2**64
