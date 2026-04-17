"""Unit tests for SpeakerNamePool in experiments/utils/speaker_names.py.

Covers determinism, uniqueness, idempotency, pool exhaustion with suffix
fallback, save/load round-trip, and different-seed divergence.

No GPU, no model load, no network access. Runs in under a second.
"""

import json
from pathlib import Path

import pytest

from experiments.utils.speaker_names import (
    _NAME_POOL,
    SpeakerNamePool,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ids(n: int, prefix: str = "speaker") -> list[str]:
    """Return a list of ``n`` distinct speaker ID strings."""
    return [f"{prefix}:{i}" for i in range(n)]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same seed + same IDs always produce the same mapping."""

    def test_same_seed_same_single_id(self):
        """Two pools with the same seed assign the same name to the same ID."""
        pool_a = SpeakerNamePool(seed=42)
        pool_b = SpeakerNamePool(seed=42)
        assert pool_a.get("alice") == pool_b.get("alice")

    def test_same_seed_same_order_multiple_ids(self):
        """Same seed + same insertion order produces identical full mapping."""
        ids = _make_ids(20)
        pool_a = SpeakerNamePool(seed=42)
        pool_b = SpeakerNamePool(seed=42)
        for sid in ids:
            pool_a.get(sid)
            pool_b.get(sid)
        assert pool_a.to_dict() == pool_b.to_dict()

    def test_determinism_across_seeds_differs(self):
        """Two different seeds assign different names to the same set of IDs."""
        ids = _make_ids(10)
        pool_a = SpeakerNamePool(seed=1)
        pool_b = SpeakerNamePool(seed=2)
        names_a = [pool_a.get(sid) for sid in ids]
        names_b = [pool_b.get(sid) for sid in ids]
        # With 10 IDs drawn from ~600 names, the probability of all 10
        # coincidentally matching two independent shuffles is astronomically
        # small.  If this ever triggers, investigate the shuffle implementation.
        assert names_a != names_b, (
            "Different seeds produced identical name assignments for 10 IDs; "
            "the shuffle may be broken."
        )


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    """Calling get() twice with the same ID always returns the same name."""

    def test_same_id_twice(self):
        """get() on the same ID returns identical name both times."""
        pool = SpeakerNamePool(seed=42)
        name1 = pool.get("longmemeval:gpt4_2655b836")
        name2 = pool.get("longmemeval:gpt4_2655b836")
        assert name1 == name2

    def test_interleaved_ids_stable(self):
        """Interleaved get() calls for the same ID return the same name."""
        pool = SpeakerNamePool(seed=99)
        name_a_first = pool.get("id_a")
        pool.get("id_b")
        pool.get("id_c")
        name_a_again = pool.get("id_a")
        assert name_a_first == name_a_again


# ---------------------------------------------------------------------------
# Uniqueness
# ---------------------------------------------------------------------------


class TestUniqueness:
    """N different IDs receive N different names (for N <= pool size)."""

    @pytest.mark.parametrize("n", [1, 10, 50, 100, 500])
    def test_unique_names_for_n_ids(self, n: int):
        """get() for n distinct IDs produces n distinct names."""
        pool = SpeakerNamePool(seed=7)
        names = [pool.get(f"id:{i}") for i in range(n)]
        assert len(set(names)) == n, (
            f"Expected {n} unique names for {n} IDs, got {len(set(names))} unique"
        )

    def test_full_pool_uniqueness(self):
        """All pool-sized IDs receive unique names (no collision within pool)."""
        pool_size = len(_NAME_POOL)
        pool = SpeakerNamePool(seed=42)
        names = [pool.get(f"id:{i}") for i in range(pool_size)]
        assert len(set(names)) == pool_size, (
            f"Pool of size {pool_size} produced only {len(set(names))} unique names"
        )


# ---------------------------------------------------------------------------
# Pool exhaustion / suffix fallback
# ---------------------------------------------------------------------------


class TestExhaustion:
    """When the pool runs out, suffix names are generated without raising."""

    def test_pool_plus_one_does_not_raise(self):
        """pool_size + 1 IDs can be registered without raising an exception."""
        pool = SpeakerNamePool(seed=42)
        n = len(_NAME_POOL) + 1
        names = [pool.get(f"id:{i}") for i in range(n)]
        assert len(names) == n

    def test_suffix_name_has_underscore_digit(self):
        """The (pool_size + 1)-th name contains an underscore followed by a digit."""
        pool = SpeakerNamePool(seed=42)
        n = len(_NAME_POOL)
        for i in range(n):
            pool.get(f"id:{i}")
        overflow_name = pool.get(f"id:{n}")
        # Suffix format: "<base>_<int>"
        assert "_" in overflow_name, (
            f"Overflow name {overflow_name!r} expected to contain '_' suffix"
        )
        parts = overflow_name.rsplit("_", 1)
        assert parts[1].isdigit(), (
            f"Suffix in {overflow_name!r} is not a digit string: {parts[1]!r}"
        )

    def test_overflow_id_idempotent(self):
        """The suffix name for an overflow ID is stable across multiple get() calls."""
        pool = SpeakerNamePool(seed=42)
        n = len(_NAME_POOL)
        for i in range(n):
            pool.get(f"id:{i}")
        overflow_id = f"id:{n}"
        name1 = pool.get(overflow_id)
        name2 = pool.get(overflow_id)
        assert name1 == name2, (
            f"Overflow ID {overflow_id!r} returned different names: {name1!r} vs {name2!r}"
        )


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """save() and load() preserve the full mapping state."""

    def test_save_produces_valid_json(self, tmp_path: Path):
        """save() writes a valid JSON file with 'seed' and 'mapping' keys."""
        pool = SpeakerNamePool(seed=42)
        pool.get("alice")
        pool.get("bob")
        path = tmp_path / "names.json"
        pool.save(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "seed" in payload
        assert "mapping" in payload
        assert payload["seed"] == 42
        assert payload["mapping"]["alice"] == pool.get("alice")
        assert payload["mapping"]["bob"] == pool.get("bob")

    def test_load_restores_mapping(self, tmp_path: Path):
        """load() returns a pool with the same name assignments as the original."""
        pool = SpeakerNamePool(seed=7)
        ids = _make_ids(30)
        original_names = {sid: pool.get(sid) for sid in ids}

        path = tmp_path / "names.json"
        pool.save(path)

        restored = SpeakerNamePool.load(path)
        for sid, expected in original_names.items():
            assert restored.get(sid) == expected, (
                f"After load, {sid!r} → {restored.get(sid)!r}, expected {expected!r}"
            )

    def test_load_restores_seed(self, tmp_path: Path):
        """load() preserves the seed value."""
        pool = SpeakerNamePool(seed=123)
        path = tmp_path / "names.json"
        pool.save(path)
        restored = SpeakerNamePool.load(path)
        assert restored._seed == 123

    def test_load_continues_new_assignments_correctly(self, tmp_path: Path):
        """After load, new IDs receive the same names as if the pool were never saved.

        Registers 10 IDs, saves, loads, then compares new-ID assignments
        against a reference pool that was never saved.
        """
        seed = 55
        initial_ids = _make_ids(10, "initial")
        new_ids = _make_ids(5, "new")

        # Reference: never saved.
        ref = SpeakerNamePool(seed=seed)
        for sid in initial_ids:
            ref.get(sid)
        ref_new_names = [ref.get(sid) for sid in new_ids]

        # Saved-and-loaded.
        pool = SpeakerNamePool(seed=seed)
        for sid in initial_ids:
            pool.get(sid)
        path = tmp_path / "names.json"
        pool.save(path)
        restored = SpeakerNamePool.load(path)
        restored_new_names = [restored.get(sid) for sid in new_ids]

        assert restored_new_names == ref_new_names, (
            f"New assignments after load differ from reference:\n"
            f"  restored: {restored_new_names}\n"
            f"  reference: {ref_new_names}"
        )

    def test_round_trip_empty_pool(self, tmp_path: Path):
        """save/load of an empty pool (no IDs registered) works without error."""
        pool = SpeakerNamePool(seed=0)
        path = tmp_path / "empty.json"
        pool.save(path)
        restored = SpeakerNamePool.load(path)
        assert restored.to_dict() == {}

    def test_to_dict_returns_copy(self):
        """to_dict() returns a copy; mutating it does not affect the pool."""
        pool = SpeakerNamePool(seed=42)
        pool.get("alice")
        d = pool.to_dict()
        d["alice"] = "MODIFIED"
        assert pool.get("alice") != "MODIFIED"


# ---------------------------------------------------------------------------
# Different seeds
# ---------------------------------------------------------------------------


class TestSeedDivergence:
    """Different seeds produce different mappings for the same set of IDs."""

    def test_two_seeds_differ(self):
        """Seed 42 and seed 1 assign different names to a set of 50 IDs."""
        ids = _make_ids(50)
        pool_a = SpeakerNamePool(seed=42)
        pool_b = SpeakerNamePool(seed=1)
        map_a = {sid: pool_a.get(sid) for sid in ids}
        map_b = {sid: pool_b.get(sid) for sid in ids}
        # At least one assignment must differ.
        differences = {sid for sid in ids if map_a[sid] != map_b[sid]}
        assert differences, (
            "Seeds 42 and 1 produced identical mappings for 50 IDs; "
            "the seed is not influencing the shuffle."
        )

    @pytest.mark.parametrize("seed_pair", [(42, 43), (0, 1), (100, 200), (7, 77)])
    def test_seed_pairs_differ(self, seed_pair: tuple[int, int]):
        """Parametrized check: each seed pair produces at least one different assignment."""
        seed_a, seed_b = seed_pair
        ids = _make_ids(30)
        pool_a = SpeakerNamePool(seed=seed_a)
        pool_b = SpeakerNamePool(seed=seed_b)
        names_a = [pool_a.get(sid) for sid in ids]
        names_b = [pool_b.get(sid) for sid in ids]
        assert names_a != names_b, (
            f"Seeds {seed_a} and {seed_b} produced identical mappings for 30 IDs."
        )
