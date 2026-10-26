import numpy as np
import pytest

from ccc.rng import STREAMS, entity_rng, unit_hash

PIN_UNIT = 0.04396681854015525
PIN_DRAW = 0.19074376189935172


def draw(*args, n=5):
    return entity_rng(*args).random(n)


def test_deterministic():
    assert np.array_equal(draw(42, "behavior", "c1", 3), draw(42, "behavior", "c1", 3))


def test_varies_with_each_key():
    base = draw(42, "behavior", "c1", 3)
    for other in [
        (43, "behavior", "c1", 3),
        (42, "arrivals", "c1", 3),
        (42, "behavior", "c2", 3),
        (42, "behavior", "c1", 4),
    ]:
        assert not np.array_equal(base, draw(*other))


def test_independent_of_other_customers():
    alone = draw(7, "behavior", "c1", 0)
    for cid in ("c0", "c2", "c999"):
        entity_rng(7, "behavior", cid, 0).random(100)
    assert np.array_equal(alone, draw(7, "behavior", "c1", 0))


def test_streams_are_separate():
    draws = [tuple(draw(1, s, "c1", 0)) for s in STREAMS]
    assert len(set(draws)) == len(STREAMS)


def test_unknown_stream_rejected():
    with pytest.raises(ValueError):
        entity_rng(1, "nope", "c1")


def test_unit_hash_stable_and_in_range():
    assert unit_hash(1, "assign", "c1") == unit_hash(1, "assign", "c1")
    vals = np.array([unit_hash(1, "assign", f"c{i}") for i in range(5000)])
    assert vals.min() >= 0 and vals.max() < 1
    assert abs(vals.mean() - 0.5) < 0.02
    assert unit_hash(1, "assign", "c1") != unit_hash(2, "assign", "c1")
    assert unit_hash(1, "assign", "c1") != unit_hash(1, "other", "c1")


def test_int_and_str_ids_differ():
    assert unit_hash(1, "assign", 1) != unit_hash(1, "assign", "1")
    assert not np.array_equal(draw(1, "behavior", 1, 0), draw(1, "behavior", "1", 0))


def test_delimiter_in_ids_does_not_collide():
    assert unit_hash(1, "a\x1fb", "c") != unit_hash(1, "a", "b\x1fc")
    assert unit_hash(1, "ab", "c") != unit_hash(1, "a", "bc")
    assert unit_hash(1, "assign", "1|2") != unit_hash(1, "assign|1", "2")


def test_unsupported_key_types_rejected():
    for bad in (1.5, True, None):
        with pytest.raises(TypeError):
            unit_hash(1, "assign", bad)


def test_pinned_values():
    # Guards the hash scheme; changing it would reshuffle every assignment and draw.
    assert unit_hash(1, "assign", "c1") == PIN_UNIT
    assert entity_rng(1, "behavior", "c1", 0).random() == PIN_DRAW
