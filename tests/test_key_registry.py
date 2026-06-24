"""Tests for the key registry."""

import json as _json

from paramem.training.key_registry import KeyRegistry


class TestKeyRegistry:
    def test_add_and_list(self):
        reg = KeyRegistry()
        reg.add("session_001")
        reg.add("session_002")
        assert reg.list_active() == ["session_001", "session_002"]

    def test_add_duplicate_ignored(self):
        reg = KeyRegistry()
        reg.add("session_001")
        reg.add("session_001")
        assert len(reg) == 1

    def test_remove(self):
        reg = KeyRegistry()
        reg.add("session_001")
        reg.add("session_002")
        reg.remove("session_001")
        assert reg.list_active() == ["session_002"]
        assert "session_001" not in reg

    def test_remove_nonexistent(self):
        reg = KeyRegistry()
        reg.remove("nonexistent")  # should not raise

    def test_contains(self):
        reg = KeyRegistry()
        reg.add("session_001")
        assert "session_001" in reg
        assert "session_002" not in reg

    def test_len(self):
        reg = KeyRegistry()
        assert len(reg) == 0
        reg.add("a")
        reg.add("b")
        assert len(reg) == 2


class TestFidelityTracking:
    def test_update_and_get_history(self):
        reg = KeyRegistry()
        reg.add("key_a")
        reg.update_fidelity("key_a", 0.9)
        reg.update_fidelity("key_a", 0.85)
        assert reg.get_fidelity_history("key_a") == [0.9, 0.85]

    def test_latest_fidelity(self):
        reg = KeyRegistry()
        reg.add("key_a")
        assert reg.get_latest_fidelity("key_a") is None
        reg.update_fidelity("key_a", 0.9)
        reg.update_fidelity("key_a", 0.7)
        assert reg.get_latest_fidelity("key_a") == 0.7

    def test_empty_history(self):
        reg = KeyRegistry()
        assert reg.get_fidelity_history("unknown") == []

    def test_remove_clears_fidelity(self):
        reg = KeyRegistry()
        reg.add("key_a")
        reg.update_fidelity("key_a", 0.5)
        reg.remove("key_a")
        assert reg.get_fidelity_history("key_a") == []


class TestRetirement:
    def test_should_retire_sustained_low(self):
        reg = KeyRegistry()
        reg.add("key_a")
        reg.update_fidelity("key_a", 0.05)
        reg.update_fidelity("key_a", 0.08)
        reg.update_fidelity("key_a", 0.03)
        assert reg.should_retire("key_a", threshold=0.1, consecutive_cycles=3)

    def test_should_not_retire_not_enough_cycles(self):
        reg = KeyRegistry()
        reg.add("key_a")
        reg.update_fidelity("key_a", 0.05)
        reg.update_fidelity("key_a", 0.08)
        assert not reg.should_retire("key_a", threshold=0.1, consecutive_cycles=3)

    def test_should_not_retire_recent_recovery(self):
        reg = KeyRegistry()
        reg.add("key_a")
        reg.update_fidelity("key_a", 0.05)
        reg.update_fidelity("key_a", 0.05)
        reg.update_fidelity("key_a", 0.5)  # recovered
        assert not reg.should_retire("key_a", threshold=0.1, consecutive_cycles=3)

    def test_should_not_retire_unknown_key(self):
        reg = KeyRegistry()
        assert not reg.should_retire("unknown")


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / "registry.json"

        reg = KeyRegistry()
        reg.add("session_001")
        reg.add("session_002")
        reg.update_fidelity("session_001", 0.9)
        reg.update_fidelity("session_001", 0.85)
        reg.save(path)

        loaded = KeyRegistry.load(path)
        assert loaded.list_active() == ["session_001", "session_002"]
        assert loaded.get_fidelity_history("session_001") == [0.9, 0.85]
        assert loaded.get_fidelity_history("session_002") == []

    def test_load_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        loaded = KeyRegistry.load(path)
        assert len(loaded) == 0

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "registry.json"
        reg = KeyRegistry()
        reg.add("key")
        reg.save(path)
        assert path.exists()

    def test_roundtrip_preserves_order(self, tmp_path):
        path = tmp_path / "registry.json"
        reg = KeyRegistry()
        for i in range(10):
            reg.add(f"key_{i:02d}")
        reg.save(path)
        loaded = KeyRegistry.load(path)
        assert loaded.list_active() == [f"key_{i:02d}" for i in range(10)]


class TestPerTierSchema:
    """Per-tier KeyRegistry: each registry owns one tier's keys.

    The adapter_id concept is now encoded by the tier name in the store's
    per-tier registries (``MemoryStore.registry(tier)``), not by a field
    on the registry record.  These tests verify the single-tier registry
    behaviours that the per-tier pattern relies on.
    """

    def test_add_no_adapter_id_kwarg(self):
        """add(key) with no kwargs works; old adapter_id kwarg is gone."""
        reg = KeyRegistry()
        reg.add("k1")
        assert "k1" in reg

    def test_add_accepts_only_key_positional(self):
        """Positional-only add(key) matches every production call site."""
        reg = KeyRegistry()
        reg.add("graph1")
        assert "graph1" in reg
        # Duplicate add must remain idempotent.
        reg.add("graph1")
        assert len(reg) == 1

    def test_list_active_scoped_to_this_tier(self):
        """list_active() returns only the keys in THIS tier's registry."""
        ep_reg = KeyRegistry()
        ep_reg.add("graph1")
        ep_reg.add("graph2")

        sem_reg = KeyRegistry()
        sem_reg.add("graph3")

        # Each registry is isolated.
        assert ep_reg.list_active() == ["graph1", "graph2"]
        assert sem_reg.list_active() == ["graph3"]

    def test_contains_scoped_to_tier(self):
        """Membership check is local to this tier's registry."""
        ep_reg = KeyRegistry()
        ep_reg.add("graph1")

        sem_reg = KeyRegistry()
        sem_reg.add("graph2")

        assert "graph1" in ep_reg
        assert "graph1" not in sem_reg
        assert "graph2" not in ep_reg
        assert "graph2" in sem_reg

    def test_roundtrip_per_tier(self, tmp_path):
        """Save + load preserves keys for a single-tier registry."""
        path = tmp_path / "registry.json"
        reg = KeyRegistry()
        reg.add("graph1")
        reg.add("graph2")
        reg.update_fidelity("graph1", 0.95)
        reg.save(path)

        loaded = KeyRegistry.load(path)
        assert loaded.list_active() == ["graph1", "graph2"]
        assert loaded.get_fidelity_history("graph1") == [0.95]
        assert "graph1" in loaded
        assert "graph2" in loaded

    def test_load_legacy_file_no_health_field(self, tmp_path):
        """Load a JSON file written before the health field existed (no 'health' key)."""
        path = tmp_path / "legacy_registry.json"
        legacy = {
            "active_keys": ["graph1", "graph2", "graph3"],
            "fidelity_history": {"graph1": [0.9, 0.85]},
        }
        path.write_text(_json.dumps(legacy))

        loaded = KeyRegistry.load(path)
        assert loaded.list_active() == ["graph1", "graph2", "graph3"]
        assert loaded.is_healthy()  # no health record = healthy


class TestStaleSemantics:
    """KeyRegistry stale partition semantics.

    Covers: stale(key) moves a key from active to stale; list_active excludes
    it; list_stale includes it; is_stale returns True; remove purges from BOTH
    active and stale; get_reclaimable returns stale keys past the cycle
    threshold; save/load round-trips the "stale" field; loading a pre-existing
    registry JSON with no "stale" key yields zero stale keys (backward-compat);
    stale_cycles starts at 0 and increment_stale_cycles advances it (a newly
    staled key has stale_cycles=0 at the durable write, unobservable until
    the second fold reads it back and increments).
    """

    def test_stale_moves_key_from_active_to_stale(self):
        """stale(key) removes from active, adds to stale partition."""
        reg = KeyRegistry()
        reg.add("graph1")
        reg.add("graph2")
        reg.stale("graph1")
        assert "graph1" not in reg.list_active()
        assert "graph1" in reg.list_stale()
        assert reg.is_stale("graph1")
        # graph2 unchanged
        assert "graph2" in reg.list_active()
        assert "graph2" not in reg.list_stale()
        assert not reg.is_stale("graph2")

    def test_stale_excludes_from_len_and_contains(self):
        """Staled keys are excluded from __len__ and __contains__."""
        reg = KeyRegistry()
        reg.add("graph1")
        reg.add("graph2")
        reg.stale("graph1")
        assert len(reg) == 1
        assert "graph1" not in reg  # __contains__ checks active only
        assert "graph2" in reg

    def test_stale_idempotent_on_already_stale(self):
        """Calling stale() on an already-stale key is a no-op."""
        reg = KeyRegistry()
        reg.add("graph1")
        reg.stale("graph1")
        first_record = dict(reg._stale["graph1"])
        reg.stale("graph1")  # second call — must not change stale_since
        assert dict(reg._stale["graph1"]) == first_record

    def test_stale_idempotent_on_absent_key(self):
        """stale() on a key that was never added is a no-op."""
        reg = KeyRegistry()
        reg.stale("nonexistent")  # must not raise
        assert reg.list_stale() == []

    def test_stale_drops_fidelity_history(self):
        """stale() removes the key from fidelity history."""
        reg = KeyRegistry()
        reg.add("graph1")
        reg.update_fidelity("graph1", 0.9)
        reg.stale("graph1")
        assert reg.get_fidelity_history("graph1") == []

    def test_remove_purges_from_both_active_and_stale(self):
        """remove() hard-erases a stale key — gone from active and stale."""
        reg = KeyRegistry()
        reg.add("graph1")
        reg.stale("graph1")
        assert "graph1" in reg.list_stale()
        reg.remove("graph1")
        assert "graph1" not in reg.list_active()
        assert "graph1" not in reg.list_stale()
        assert not reg.is_stale("graph1")

    def test_remove_active_key_also_pops_stale(self):
        """remove() on an active key leaves stale clean too."""
        reg = KeyRegistry()
        reg.add("graph1")
        reg.add("graph2")
        reg.stale("graph2")  # graph2 is stale
        reg.remove("graph1")
        assert "graph1" not in reg.list_active()
        assert "graph2" in reg.list_stale()

    def test_get_reclaimable_threshold(self):
        """get_reclaimable returns stale keys with stale_cycles >= threshold."""
        reg = KeyRegistry()
        reg.add("graph1")
        reg.add("graph2")
        reg.stale("graph1")
        reg.stale("graph2")
        # Both start at stale_cycles=0.
        assert reg.get_reclaimable(min_stale_cycles=1) == []
        # Advance graph1 to stale_cycles=1.
        reg._stale["graph1"]["stale_cycles"] = 1
        reclaimable = reg.get_reclaimable(min_stale_cycles=1)
        assert reclaimable == ["graph1"]
        assert "graph2" not in reclaimable

    def test_stale_cycles_starts_at_zero(self):
        """A freshly staled key has stale_cycles=0."""
        reg = KeyRegistry()
        reg.add("graph1")
        reg.stale("graph1")
        assert reg._stale["graph1"]["stale_cycles"] == 0

    def test_increment_stale_cycles_advances_all_stale_keys(self):
        """increment_stale_cycles advances stale_cycles for every stale key.

        A key staled in fold N has stale_cycles=0 at the durable write;
        stale_cycles=1 after increment_stale_cycles (unobservable until
        the NEXT fold reads it from disk).
        """
        reg = KeyRegistry()
        reg.add("graph1")
        reg.add("graph2")
        reg.stale("graph1")
        reg.stale("graph2")
        reg.increment_stale_cycles()
        assert reg._stale["graph1"]["stale_cycles"] == 1
        assert reg._stale["graph2"]["stale_cycles"] == 1
        # Second increment → 2.
        reg.increment_stale_cycles()
        assert reg._stale["graph1"]["stale_cycles"] == 2

    def test_save_load_roundtrip_stale_field(self, tmp_path):
        """save_bytes / load round-trips the "stale" partition."""
        reg = KeyRegistry()
        reg.add("graph1")
        reg.add("graph2")
        reg.stale("graph2")
        path = tmp_path / "registry.json"
        reg.save(path)

        loaded = KeyRegistry.load(path)
        assert loaded.list_active() == ["graph1"]
        assert loaded.list_stale() == ["graph2"]
        assert loaded.is_stale("graph2")
        assert not loaded.is_stale("graph1")

    def test_load_legacy_file_no_stale_key_yields_empty_stale(self, tmp_path):
        """Loading a pre-existing registry JSON with no "stale" key gives zero stale keys.

        Backward-compat: old on-disk files lack the "stale" field; load must
        default to {} (all keys active).
        """
        path = tmp_path / "legacy.json"
        legacy = {
            "active_keys": ["graph1", "graph2"],
            "fidelity_history": {},
            "health": None,
            # no "stale" key
        }
        path.write_text(_json.dumps(legacy))
        loaded = KeyRegistry.load(path)
        assert loaded.list_active() == ["graph1", "graph2"]
        assert loaded.list_stale() == []

    def test_two_fold_stale_cycle_sequence(self, tmp_path):
        """stale_cycles=0 at durable write; =1 after increment.

        Two-fold sequence: fold N stales a key and saves (stale_cycles=0 on
        disk); increment_stale_cycles runs after the durable write (stale_cycles=1
        in-memory); fold N+1 reads from disk (stale_cycles=0), then increment
        advances to 1.
        """
        reg = KeyRegistry()
        reg.add("graph1")
        reg.stale("graph1")
        path = tmp_path / "registry.json"
        # Fold N: durable write with stale_cycles=0.
        reg.save(path)
        loaded_at_write = KeyRegistry.load(path)
        assert loaded_at_write._stale["graph1"]["stale_cycles"] == 0

        # After durable write, increment runs (in-memory only).
        reg.increment_stale_cycles()
        assert reg._stale["graph1"]["stale_cycles"] == 1

        # Fold N+1: load from disk (still 0 — disk was saved before increment).
        loaded_fold_n1 = KeyRegistry.load(path)
        assert loaded_fold_n1._stale["graph1"]["stale_cycles"] == 0
        # Then increment again to simulate fold N+1 post-durable-write.
        loaded_fold_n1.increment_stale_cycles()
        assert loaded_fold_n1._stale["graph1"]["stale_cycles"] == 1


class TestKnownPredicate:
    """Unit tests for KeyRegistry.knows() and KeyRegistry.list_known().

    Verifies the KNOWN-legitimacy predicates (active ∪ stale) introduced to
    canonicalize orphan-check and bookkeeping-retention consumers.  Distinct
    from __contains__ (active-only, SERVE semantics).
    """

    def test_knows_active_key(self):
        """knows() returns True for an active key."""
        reg = KeyRegistry()
        reg.add("graph1")
        assert reg.knows("graph1")

    def test_knows_stale_key(self):
        """knows() returns True for a key that has been staled."""
        reg = KeyRegistry()
        reg.add("graph1")
        reg.stale("graph1")
        # Active predicate no longer sees it.
        assert "graph1" not in reg
        # Known predicate does.
        assert reg.knows("graph1")

    def test_knows_false_for_absent_key(self):
        """knows() returns False for a key never added."""
        reg = KeyRegistry()
        assert not reg.knows("unknown")

    def test_knows_false_after_remove(self):
        """knows() returns False after hard-remove (key purged from both partitions)."""
        reg = KeyRegistry()
        reg.add("graph1")
        reg.stale("graph1")
        reg.remove("graph1")
        assert not reg.knows("graph1")

    def test_list_known_active_and_stale(self):
        """list_known() = active keys first, then stale keys."""
        reg = KeyRegistry()
        reg.add("graph1")
        reg.add("graph2")
        reg.add("graph3")
        reg.stale("graph2")
        result = reg.list_known()
        # graph1 and graph3 are active (graph2 staled out of active).
        assert result == ["graph1", "graph3", "graph2"]

    def test_list_known_active_only(self):
        """list_known() == list_active() when no stale keys."""
        reg = KeyRegistry()
        reg.add("a")
        reg.add("b")
        assert reg.list_known() == reg.list_active()

    def test_list_known_stale_only(self):
        """list_known() == list_stale() when all keys have been staled."""
        reg = KeyRegistry()
        reg.add("a")
        reg.add("b")
        reg.stale("a")
        reg.stale("b")
        assert reg.list_known() == reg.list_stale()

    def test_staling_flips_contains_but_not_knows(self):
        """Staling a key moves it from __contains__=True to knows()=True, __contains__=False.

        This is the exact active/known divergence that the orphan-check bug relied on.
        """
        reg = KeyRegistry()
        reg.add("proc52")
        assert "proc52" in reg  # active
        assert reg.knows("proc52")  # known

        reg.stale("proc52")
        assert "proc52" not in reg  # no longer active
        assert reg.knows("proc52")  # still known


class TestSaveBytesBoundary:
    def test_save_bytes_payload_is_bookkeeping_free(self):
        """HARD CONSTRAINT: KeyRegistry.save_bytes() must contain exactly
        {active_keys, fidelity_history, health, stale, simhash} — no bookkeeping fields.

        registry_sha256 hashes the save_bytes payload; bookkeeping must never
        enter it or slot identity breaks on restart.  The ``"simhash"`` field
        (unified SimHash storage, simhash-unification refactor) holds the
        active∪stale fingerprint map.  The ``"stale"`` field was added in the
        soft-stale extension.
        """
        reg = KeyRegistry()
        reg.add("graph1")
        reg.add("graph2")
        reg.update_fidelity("graph1", 0.9)
        payload = set(_json.loads(reg.save_bytes()))
        assert payload == {"active_keys", "fidelity_history", "health", "stale", "simhash"}
        assert "speaker_id" not in payload
        assert "first_seen_cycle" not in payload
        assert "bookkeeping" not in payload
