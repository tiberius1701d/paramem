"""Unit tests for :class:`paramem.memory.store.MemoryStore`.

Locks the per-tier {quads, simhash, registry} contract that replaced the
flat ``indexed_key_cache`` + three flat ``*_simhash`` dicts on
:class:`ConsolidationLoop`.
"""

from __future__ import annotations

import threading

import pytest

from paramem.memory.store import MemoryStore
from paramem.training.key_registry import KeyRegistry


def _entry(
    key: str, subject: str = "Alice", predicate: str = "lives_in", obj: str = "Berlin"
) -> dict:
    return {
        "key": key,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "speaker_id": "spk-alice",
    }


# ---------------------------------------------------------------------------
# Quad payload — read / write / membership
# ---------------------------------------------------------------------------


class TestQuadPayload:
    def test_put_then_get_round_trip(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        assert s.get("graph1") == _entry("graph1")

    def test_get_miss_returns_none(self):
        assert MemoryStore().get("graph999") is None

    def test_has_membership(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        assert s.has("graph1")
        assert "graph1" in s
        assert not s.has("graph999")

    def test_tier_of_returns_owning_tier(self):
        s = MemoryStore()
        s.put("semantic", "graph42", _entry("graph42"))
        assert s.tier_of("graph42") == "semantic"
        assert s.tier_of("graph999") is None

    def test_quads_in_tier_returns_only_that_tier(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        s.put("semantic", "graph2", _entry("graph2"))
        assert set(s.entries_in_tier("episodic")) == {"graph1"}
        assert set(s.entries_in_tier("semantic")) == {"graph2"}
        assert s.entries_in_tier("procedural") == {}

    def test_iter_quads_yields_tier_key_quad(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        s.put("semantic", "graph2", _entry("graph2"))
        out = sorted((t, k) for t, k, _ in s.iter_entries())
        assert out == [("episodic", "graph1"), ("semantic", "graph2")]

    def test_len_counts_all_tiers(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        s.put("episodic", "graph2", _entry("graph2"))
        s.put("semantic", "graph3", _entry("graph3"))
        assert len(s) == 3

    def test_bookkeeping_round_trips_speaker_id(self):
        """set_bookkeeping then bookkeeping_for_key returns the correct fields.

        Replaces the deleted setdefault_entry test — bookkeeping is now the
        canonical owner of speaker_id/relation_type/reinforcement_count/last_seen."""
        s = MemoryStore()
        s.set_bookkeeping("graph1", speaker_id="spk-a", relation_type="factual", first_seen="")
        bk = s.bookkeeping_for_key("graph1")
        assert bk is not None
        assert bk["speaker_id"] == "spk-a"
        assert bk["relation_type"] == "factual"
        # Second call must return updated values (idempotent overwrite).
        s.set_bookkeeping("graph1", speaker_id="spk-b", relation_type="preference", first_seen="")
        bk2 = s.bookkeeping_for_key("graph1")
        assert bk2["speaker_id"] == "spk-b"
        assert bk2["relation_type"] == "preference"


# ---------------------------------------------------------------------------
# SimHash fingerprints — separated from the quad
# ---------------------------------------------------------------------------


class TestSimHash:
    def test_put_via_put_writes_simhash_when_supplied(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"), simhash=0xCAFE)
        assert s.simhash("episodic", "graph1") == 0xCAFE

    def test_put_simhash_independently(self):
        s = MemoryStore()
        s.put_simhash("episodic", "graph1", 0xBEEF)
        assert s.simhash("episodic", "graph1") == 0xBEEF
        assert s.has_simhash("episodic", "graph1")

    def test_simhash_miss_returns_none(self):
        assert MemoryStore().simhash("episodic", "graph1") is None
        assert not MemoryStore().has_simhash("episodic", "graph1")

    def test_tier_simhashes_is_dict_str_int_for_verify_confidence(self):
        """tier_simhashes must return the same shape as the old *_simhash dicts
        — key -> 64-bit int — so verify_confidence callers see a consistent shape."""
        s = MemoryStore()
        s.put_simhash("episodic", "graph1", 0xCAFE)
        s.put_simhash("episodic", "graph2", 0xBEEF)
        view = s.tier_simhashes("episodic", include_stale=True)
        assert view == {"graph1": 0xCAFE, "graph2": 0xBEEF}
        assert all(isinstance(v, int) for v in view.values())

    def test_replace_simhashes_in_tier_bulk_overwrite(self):
        s = MemoryStore()
        s.put_simhash("episodic", "graph_old", 0xAA)
        s.replace_simhashes_in_tier("episodic", {"graph_new": 0xBB})
        assert s.tier_simhashes("episodic", include_stale=True) == {"graph_new": 0xBB}

    def test_simhash_count_in_tier(self):
        s = MemoryStore()
        assert s.simhash_count_in_tier("episodic") == 0
        s.put_simhash("episodic", "graph1", 1)
        s.put_simhash("episodic", "graph2", 2)
        assert s.simhash_count_in_tier("episodic") == 2

    def test_delete_simhash_only(self):
        s = MemoryStore()
        s.put_simhash("episodic", "graph1", 1)
        s.delete_simhash("episodic", "graph1")
        assert not s.has_simhash("episodic", "graph1")


# ---------------------------------------------------------------------------
# KeyRegistry simhash methods — unit tests for the simhash primitives
# ---------------------------------------------------------------------------


class TestKeyRegistrySimhash:
    """Unit tests for the KeyRegistry simhash primitives.

    Locks the contract: set_simhash, drop_simhash, simhash_for,
    has_simhash, _active_simhashes, _known_simhashes, stale auto-carry,
    remove auto-drop.
    """

    def test_set_and_simhash_for_active(self):
        """set_simhash on an active key is returned by simhash_for."""
        reg = KeyRegistry()
        reg.add("graph1")
        reg.set_simhash("graph1", 0xCAFE)
        assert reg.simhash_for("graph1") == 0xCAFE

    def test_simhash_for_stale_key(self):
        """simhash_for returns the fingerprint from the stale record."""
        reg = KeyRegistry()
        reg.add("graph1")
        reg.set_simhash("graph1", 0xBEEF)
        reg.stale("graph1")
        # Key is now stale — simhash must still be accessible.
        assert reg.simhash_for("graph1") == 0xBEEF

    def test_stale_carries_active_simhash_atomically(self):
        """stale() moves the active simhash into the stale record automatically.

        The caller does NOT need to manually move the fingerprint.
        """
        reg = KeyRegistry()
        reg.add("graph1")
        reg.set_simhash("graph1", 0xDEAD)
        reg.stale("graph1")
        # Active _simhash must be empty.
        assert "graph1" not in reg._simhash
        # Stale record must carry the fingerprint.
        assert reg._stale["graph1"].get("simhash") == 0xDEAD

    def test_remove_drops_simhash(self):
        """remove() erases the fingerprint from both active and stale partitions."""
        reg = KeyRegistry()
        reg.add("graph1")
        reg.set_simhash("graph1", 0xAAAA)
        reg.remove("graph1")
        assert reg.simhash_for("graph1") is None
        assert not reg.has_simhash("graph1")

    def test_has_simhash_active_and_stale(self):
        """has_simhash returns True for both active and stale partitions."""
        reg = KeyRegistry()
        reg.add("graph1")
        reg.set_simhash("graph1", 0x1111)
        assert reg.has_simhash("graph1")
        reg.stale("graph1")
        assert reg.has_simhash("graph1")

    def test_active_simhashes_excludes_stale(self):
        """_active_simhashes returns only active-partition fingerprints."""
        reg = KeyRegistry()
        reg.add("graph1")
        reg.add("graph2")
        reg.set_simhash("graph1", 0x1111)
        reg.set_simhash("graph2", 0x2222)
        reg.stale("graph2")
        active = reg._active_simhashes()
        assert "graph1" in active
        assert "graph2" not in active

    def test_known_simhashes_includes_stale(self):
        """_known_simhashes returns active∪stale fingerprints."""
        reg = KeyRegistry()
        reg.add("graph1")
        reg.add("graph2")
        reg.set_simhash("graph1", 0x1111)
        reg.set_simhash("graph2", 0x2222)
        reg.stale("graph2")
        known = reg._known_simhashes()
        assert "graph1" in known
        assert "graph2" in known
        assert known["graph1"] == 0x1111
        assert known["graph2"] == 0x2222

    def test_drop_simhash_clears_both_partitions(self):
        """drop_simhash removes the fingerprint regardless of partition."""
        reg = KeyRegistry()
        reg.add("graph1")
        reg.set_simhash("graph1", 0xCAFE)
        reg.drop_simhash("graph1")
        assert not reg.has_simhash("graph1")
        assert reg.simhash_for("graph1") is None

    def test_save_bytes_load_roundtrip_with_simhash(self):
        """save_bytes/load round-trip preserves active and stale fingerprints."""
        import json

        reg = KeyRegistry()
        reg.add("graph1")
        reg.add("graph2")
        reg.set_simhash("graph1", 0xCAFE)
        reg.set_simhash("graph2", 0xDEAD)
        reg.stale("graph2")

        payload = reg.save_bytes()
        data = json.loads(payload.decode("utf-8"))

        # "simhash" key must be present in the new schema.
        assert "simhash" in data
        assert data["simhash"]["graph1"] == 0xCAFE
        assert data["simhash"]["graph2"] == 0xDEAD

    def test_load_reads_simhash_from_new_schema(self, tmp_path):
        """KeyRegistry.load reads the 'simhash' field from the new schema (no .get fallback).

        A registry file without 'simhash' raises KeyError — fresh-start mandate.
        """
        import json

        path = tmp_path / "indexed_key_registry.json"
        # New schema with simhash.
        data = {
            "active_keys": ["graph1"],
            "fidelity_history": {},
            "stale": {},
            "simhash": {"graph1": 0xABCDEF},
        }
        path.write_text(json.dumps(data))

        reg = KeyRegistry.load(path)
        assert reg.simhash_for("graph1") == 0xABCDEF

    def test_load_old_schema_without_simhash_loads_as_empty_fingerprints(self, tmp_path):
        """KeyRegistry.load gracefully handles old-schema files without 'simhash'.

        The fresh-start mandate means no legacy-file fallback (no reading
        simhash_registry.json), but the load still succeeds — the registry
        is loaded with an empty fingerprint map rather than crashing.
        """
        import json

        path = tmp_path / "indexed_key_registry.json"
        # Old schema — no simhash key.
        data = {
            "active_keys": ["graph1"],
            "fidelity_history": {},
            "stale": {},
        }
        path.write_text(json.dumps(data))

        reg = KeyRegistry.load(path)
        # Registry loads the active keys.
        assert reg.list_active() == ["graph1"]
        # No fingerprints (old schema had none).
        assert reg.simhash_for("graph1") is None
        assert not reg.has_simhash("graph1")


class TestTierSimhashes:
    """Unit tests for MemoryStore.tier_simhashes.

    The mandatory ``include_stale`` keyword makes active-vs-known confusion
    structurally impossible.  This class locks that contract.
    """

    def test_tier_simhashes_active_only(self):
        """tier_simhashes(include_stale=False) returns only active fingerprints."""
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"), simhash=0x1111)
        s.put("episodic", "graph2", _entry("graph2"), simhash=0x2222)
        s.discard_keys(["graph2"], mode="stale")
        active = s.tier_simhashes("episodic", include_stale=False)
        assert "graph1" in active
        assert "graph2" not in active

    def test_tier_simhashes_include_stale(self):
        """tier_simhashes(include_stale=True) returns active∪stale fingerprints."""
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"), simhash=0x1111)
        s.put("episodic", "graph2", _entry("graph2"), simhash=0x2222)
        s.discard_keys(["graph2"], mode="stale")
        known = s.tier_simhashes("episodic", include_stale=True)
        assert "graph1" in known
        assert "graph2" in known

    def test_tier_simhashes_requires_keyword_argument(self):
        """include_stale is a keyword-only argument — positional call must fail."""
        s = MemoryStore()
        with pytest.raises(TypeError):
            s.tier_simhashes("episodic", True)  # type: ignore[call-arg]

    def test_tier_simhashes_empty_tier_returns_empty_dict(self):
        """Requesting simhashes for a non-existent tier returns an empty dict."""
        s = MemoryStore()
        result = s.tier_simhashes("nonexistent", include_stale=False)
        assert result == {}


# ---------------------------------------------------------------------------
# Lifecycle registry — Optional gate
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_replay_enabled_default(self):
        s = MemoryStore()
        assert s.replay_enabled
        assert s.registry("episodic") is not None
        assert isinstance(s.registry("episodic"), KeyRegistry)

    def test_replay_disabled_registry_is_not_none(self):
        s = MemoryStore(replay_enabled=False)
        assert not s.replay_enabled
        # registry() never returns None; replay_enabled is a behaviour flag only.
        assert isinstance(s.registry("episodic"), KeyRegistry)

    def test_put_registers_when_replay_enabled(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        assert "graph1" in s.registry("episodic")

    def test_put_does_not_register_when_replay_disabled(self):
        s = MemoryStore(replay_enabled=False)
        s.put("episodic", "graph1", _entry("graph1"))
        # registry() never returns None; replay_enabled governs lifecycle only.
        # The key was NOT added to the registry (register=False path under replay-off).
        assert "graph1" not in s.registry("episodic")
        # Quad is still there.
        assert s.get("graph1") is not None

    def test_register_false_skips_registry(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"), register=False)
        assert "graph1" not in s.registry("episodic")
        assert s.get("graph1") is not None

    def test_load_registry_installs_preloaded_instance(self):
        s = MemoryStore()
        reg = KeyRegistry()
        reg.add("graph_preloaded")
        s.load_registry("episodic", reg)
        assert s.registry("episodic") is reg
        assert "graph_preloaded" in s.registry("episodic")

    def test_load_registry_raises_when_replay_disabled(self):
        s = MemoryStore(replay_enabled=False)
        with pytest.raises(RuntimeError, match="replay is disabled"):
            s.load_registry("episodic", KeyRegistry())

    def test_active_keys_in_tier(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        s.put("episodic", "graph2", _entry("graph2"))
        s.put("semantic", "graph3", _entry("graph3"))
        assert sorted(s.active_keys_in_tier("episodic")) == ["graph1", "graph2"]
        assert s.active_keys_in_tier("procedural") == []

    def test_stale_keys_in_tier(self):
        """Per-tier analogue of active_keys_in_tier for the stale partition
        (fold-telemetry: measures encoded-vs-active divergence)."""
        s = MemoryStore(replay_enabled=True)
        s.put("episodic", "graph1", _entry("graph1"), register=True)
        s.put("episodic", "graph2", _entry("graph2"), register=True)
        s.put("semantic", "graph3", _entry("graph3"), register=True)

        # Fresh store: nothing stale yet, and an unregistered tier reports
        # an empty list rather than raising (mirrors active_keys_in_tier).
        assert s.stale_keys_in_tier("episodic") == []
        assert s.stale_keys_in_tier("procedural") == []

        s.discard_keys(["graph1"], mode="stale")

        assert s.stale_keys_in_tier("episodic") == ["graph1"]
        assert s.stale_keys_in_tier("semantic") == []
        # A fully erased key (never soft-staled) is invisible to this count
        # by design -- it is removed from the registry entirely.
        s.discard_keys(["graph2"], mode="erase")
        assert s.stale_keys_in_tier("episodic") == ["graph1"]

    def test_all_active_keys(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        s.put("semantic", "graph2", _entry("graph2"))
        s.put("procedural", "proc1", _entry("proc1"))
        assert sorted(s.all_active_keys()) == ["graph1", "graph2", "proc1"]

    def test_tier_for_active_key(self):
        s = MemoryStore()
        s.put("semantic", "graph42", _entry("graph42"))
        assert s.tier_for_active_key("graph42") == "semantic"
        assert s.tier_for_active_key("graph999") is None

    def test_drop_registry_removes_tier(self):
        s = MemoryStore()
        s.put("episodic_interim_20260514T0000", "graph1", _entry("graph1"))
        assert s.has_registry("episodic_interim_20260514T0000")
        dropped = s.drop_registry("episodic_interim_20260514T0000")
        assert dropped is not None
        assert "graph1" in dropped
        assert not s.has_registry("episodic_interim_20260514T0000")


# ---------------------------------------------------------------------------
# Known-legitimacy predicates — is_known, tier_for_known_key, all_known_keys
# ---------------------------------------------------------------------------


class TestKnownPredicates:
    """Unit tests for MemoryStore.is_known, tier_for_known_key, all_known_keys.

    Mirrors test_all_active_keys / test_tier_for_active_key with the KNOWN
    (active ∪ stale) semantics.  SERVE predicates (tier_for_active_key,
    all_active_keys) must remain unaffected — verified here too.
    """

    def _store_with_stale_key(self) -> MemoryStore:
        """Build a store that has 'graph1' active and 'proc52' stale."""
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        s.put("procedural", "proc52", _entry("proc52"))
        s.discard_keys(["proc52"], mode="stale")
        return s

    def test_is_known_active_key(self):
        """is_known() returns True for an active key."""
        s = self._store_with_stale_key()
        assert s.is_known("graph1")

    def test_is_known_stale_key(self):
        """is_known() returns True for a stale key."""
        s = self._store_with_stale_key()
        assert s.is_known("proc52")

    def test_is_known_absent_key(self):
        """is_known() returns False for a key not in any tier."""
        s = self._store_with_stale_key()
        assert not s.is_known("ghost")

    def test_is_known_replay_disabled(self):
        """is_known() returns False when replay is disabled."""
        s = MemoryStore(replay_enabled=False)
        assert not s.is_known("anything")

    def test_tier_for_known_key_active(self):
        """tier_for_known_key() returns the owning tier for an active key."""
        s = MemoryStore()
        s.put("semantic", "graph42", _entry("graph42"))
        assert s.tier_for_known_key("graph42") == "semantic"

    def test_tier_for_known_key_stale(self):
        """tier_for_known_key() returns the owning tier for a stale key.

        tier_for_active_key() on the same key must return None (SERVE unchanged).
        """
        s = self._store_with_stale_key()
        # KNOWN sees it
        assert s.tier_for_known_key("proc52") == "procedural"
        # SERVE does not
        assert s.tier_for_active_key("proc52") is None

    def test_tier_for_known_key_absent(self):
        """tier_for_known_key() returns None for an absent key."""
        s = self._store_with_stale_key()
        assert s.tier_for_known_key("ghost") is None

    def test_all_known_keys_includes_stale(self):
        """all_known_keys() includes both active and stale keys."""
        s = self._store_with_stale_key()
        known = sorted(s.all_known_keys())
        assert "graph1" in known
        assert "proc52" in known

    def test_all_known_keys_replay_disabled(self):
        """all_known_keys() returns [] when replay is disabled."""
        s = MemoryStore(replay_enabled=False)
        assert s.all_known_keys() == []

    def test_all_active_keys_excludes_stale(self):
        """all_active_keys() (SERVE) must not include stale keys after refactor."""
        s = self._store_with_stale_key()
        active = s.all_active_keys()
        assert "graph1" in active
        assert "proc52" not in active

    def test_delete_stale_only_key(self):
        """delete() removes a key that is ONLY stale (registry + simhash)."""
        s = MemoryStore()
        s.put("procedural", "proc52", _entry("proc52"), simhash=0xDEAD)
        s.discard_keys(["proc52"], mode="stale")
        # Before: proc52 is stale-only
        assert s.is_stale("proc52")
        assert not s.tier_for_active_key("proc52")
        former = s.delete("proc52")
        # After: fully gone
        assert former == "procedural"
        assert not s.is_known("proc52")
        assert s.registry("procedural") is not None
        assert "proc52" not in s.registry("procedural")


# ---------------------------------------------------------------------------
# Move + delete — preserve cross-structure consistency
# ---------------------------------------------------------------------------


class TestMoveDelete:
    def test_delete_clears_all_three_structures(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"), simhash=0xCAFE)
        former = s.delete("graph1")
        assert former == "episodic"
        assert s.get("graph1") is None
        assert not s.has_simhash("episodic", "graph1")
        assert "graph1" not in s.registry("episodic")

    def test_delete_unknown_returns_none(self):
        assert MemoryStore().delete("graph999") is None

    def test_move_relocates_quad_simhash_registry(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"), simhash=0xCAFE)
        s.move("graph1", "semantic")
        assert s.tier_of("graph1") == "semantic"
        assert s.entries_in_tier("episodic") == {}
        assert s.tier_simhashes("episodic", include_stale=True) == {}
        assert s.tier_simhashes("semantic", include_stale=True) == {"graph1": 0xCAFE}
        assert "graph1" not in s.registry("episodic")
        assert "graph1" in s.registry("semantic")

    def test_move_to_same_tier_is_noop_but_keeps_registry(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        s.move("graph1", "episodic")
        assert s.tier_of("graph1") == "episodic"
        assert "graph1" in s.registry("episodic")


# ---------------------------------------------------------------------------
# drop_tier — tier-granularity rollback primitive (Part B)
# ---------------------------------------------------------------------------


class TestDropTier:
    def test_drop_tier_removes_entries_registry_and_bookkeeping(self):
        s = MemoryStore()
        s.put("episodic_interim_20260417T0000", "graph1", _entry("graph1"), simhash=0xCAFE)
        s.set_bookkeeping("graph1", speaker_id="speaker0", relation_type="factual", first_seen="t0")
        s.drop_tier("episodic_interim_20260417T0000")
        assert s.entries_in_tier("episodic_interim_20260417T0000") == {}
        assert s.has_registry("episodic_interim_20260417T0000") is False
        assert s.bookkeeping_for_key("graph1") is None

    def test_drop_tier_does_not_touch_other_tiers(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"), simhash=1)
        s.set_bookkeeping("graph1", speaker_id="speaker0", relation_type="factual", first_seen="t0")
        s.put("episodic_interim_20260417T0000", "graph2", _entry("graph2"), simhash=2)
        s.drop_tier("episodic_interim_20260417T0000")
        assert s.get("graph1") == _entry("graph1")
        assert s.bookkeeping_for_key("graph1") is not None
        assert s.entries_in_tier("episodic_interim_20260417T0000") == {}

    def test_drop_tier_unknown_tier_is_noop(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        s.drop_tier("does_not_exist")  # must not raise
        assert s.get("graph1") == _entry("graph1")

    def test_drop_tier_drops_bookkeeping_for_stale_keys_too(self):
        """A soft-staled key's bookkeeping must not survive its tier being
        dropped — only active_keys_in_tier was swept before, leaving a stale
        key's bookkeeping record orphaned (re-indexed by router.reload
        against a tier that no longer exists)."""
        s = MemoryStore()
        s.put("episodic_interim_20260417T0000", "graph1", _entry("graph1"), simhash=1)
        s.put("episodic_interim_20260417T0000", "graph2", _entry("graph2"), simhash=2)
        s.set_bookkeeping("graph1", speaker_id="speaker0", relation_type="factual", first_seen="t0")
        s.set_bookkeeping("graph2", speaker_id="speaker0", relation_type="factual", first_seen="t0")
        s.registry("episodic_interim_20260417T0000").stale("graph2")

        s.drop_tier("episodic_interim_20260417T0000")

        assert dict(s.iter_bookkeeping()) == {}
        assert s.bookkeeping_for_key("graph1") is None
        assert s.bookkeeping_for_key("graph2") is None


# ---------------------------------------------------------------------------
# reactivate — dual of discard_keys(mode="stale") (Part B rollback primitive)
# ---------------------------------------------------------------------------


class TestReactivate:
    def test_reactivate_restores_active_and_simhash(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"), simhash=0xBEEF)
        s.discard_keys(["graph1"], mode="stale")
        assert s.is_stale("graph1")
        assert "graph1" not in s.registry("episodic")

        s.reactivate("episodic", "graph1")
        assert "graph1" in s.registry("episodic")
        assert not s.is_stale("graph1")
        assert s.simhash("episodic", "graph1") == 0xBEEF

    def test_reactivate_noop_when_key_not_stale(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"), simhash=1)
        s.reactivate("episodic", "graph1")  # already active — must not raise
        assert "graph1" in s.registry("episodic")

    def test_reactivate_noop_when_tier_unknown(self):
        s = MemoryStore()
        s.reactivate("does_not_exist", "graph1")  # must not raise

    def test_reactivate_key_with_no_simhash(self):
        """A staled key that never had a fingerprint reactivates cleanly (fp=None branch)."""
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))  # no simhash= kwarg
        s.discard_keys(["graph1"], mode="stale")
        assert s.is_stale("graph1")
        s.reactivate("episodic", "graph1")
        assert "graph1" in s.registry("episodic")
        assert not s.is_stale("graph1")
        assert s.simhash("episodic", "graph1") is None

    def test_reactivate_noop_when_replay_disabled(self):
        """reactivate() is a no-op guard, matching discard_keys' replay_enabled gate."""
        s = MemoryStore(replay_enabled=False)
        s.put("episodic", "graph1", _entry("graph1"), simhash=1, register=False)
        s.reactivate("episodic", "graph1")  # must not raise


# ---------------------------------------------------------------------------
# Snapshot / restore — cycle-resume rollback rope
# ---------------------------------------------------------------------------


class TestSnapshotRestore:
    def test_snapshot_then_mutate_then_restore(self):
        """Snapshot captures both entries and simhash; restore brings both back.

        This is the core cycle-resume rollback rope contract.  Simhash is now
        captured from the registry so fingerprints survive
        rollback correctly even after the registry simhash moved off _simhash.
        """
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"), simhash=0xCAFE)
        snap = s.snapshot()
        # Mutate
        s.put("episodic", "graph2", _entry("graph2"))
        s.delete("graph1")
        assert s.get("graph2") is not None
        assert s.get("graph1") is None
        # Restore
        s.restore(snap)
        assert s.get("graph1") == _entry("graph1")
        # Simhash must survive restore.
        assert s.simhash("episodic", "graph1") == 0xCAFE
        assert s.get("graph2") is None

    def test_snapshot_does_not_share_mutable_state(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        snap = s.snapshot()
        # Mutate snapshot — store must be unaffected.
        snap["entries"]["episodic"]["graph1"]["subject"] = "Eve"
        assert s.get("graph1")["subject"] == "Alice"

    def test_snapshot_excludes_registry_lifecycle_but_includes_fingerprints(self):
        """Registries persist via their own KeyRegistry.save/load lifecycle.

        The snapshot DOES carry the fingerprint map so fingerprints
        survive rollback, but the registry lifecycle (active/stale partitions)
        is NOT included — that is restored from disk separately.
        """
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"), simhash=0xDEAD)
        snap = s.snapshot()
        assert "entries" in snap
        assert "simhash" in snap  # fingerprints ARE included
        assert "registry" not in snap  # lifecycle partitions are NOT
        # The fingerprint for graph1 must be in the snapshot map.
        assert snap["simhash"].get("episodic", {}).get("graph1") == 0xDEAD

    def test_snapshot_stale_fingerprint_included(self):
        """Stale fingerprints are included in the snapshot simhash map.

        When a key is staled its fingerprint moves to the stale record; the
        snapshot must capture it from _known_simhashes (active∪stale) so the
        stale-echo confidence gate still works after rollback.
        """
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"), simhash=0xBEEF)
        s.discard_keys(["graph1"], mode="stale")
        snap = s.snapshot()
        # The stale fingerprint must appear in the snapshot.
        assert snap["simhash"].get("episodic", {}).get("graph1") == 0xBEEF


# ---------------------------------------------------------------------------
# Stats — diagnostic surface
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_per_tier_counts(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"), simhash=1)
        s.put("episodic", "graph2", _entry("graph2"), simhash=2)
        s.put("semantic", "graph3", _entry("graph3"), simhash=3)
        stats = s.stats()
        assert stats["episodic"]["key_count"] == 2
        assert stats["episodic"]["simhash_count"] == 2
        assert stats["episodic"]["registry_active"] == 2
        assert stats["semantic"]["key_count"] == 1

    def test_stats_replay_disabled_no_registry_keys(self):
        s = MemoryStore(replay_enabled=False)
        s.put("episodic", "graph1", _entry("graph1"), simhash=1)
        stats = s.stats()
        assert "registry_active" not in stats["episodic"]
        assert stats["episodic"]["key_count"] == 1


# ---------------------------------------------------------------------------
# Bookkeeping — speaker_id / relation_type / reinforcement_count / last_seen
# ---------------------------------------------------------------------------


class TestBookkeeping:
    def test_bookkeeping_does_not_create_content_hit(self):
        """REGRESSION LOCK: set_bookkeeping must NOT create a content cache hit.

        Under preload_cache=False this is the key correctness invariant:
        bookkeeping presence must never mask a cache miss."""
        s = MemoryStore()
        s.set_bookkeeping("k", speaker_id="alice", relation_type="factual", first_seen="")
        # No put — _entries is empty.
        assert s.get("k") is None
        results = s.probe({"episodic": ["k"]}, source=None)
        assert results["k"] is None

    def test_new_key_write_back_round_trips_speaker_id(self):
        """set_bookkeeping + bookkeeping_for_key round-trips all fields."""
        s = MemoryStore()
        s.set_bookkeeping("graph1", speaker_id="bob", relation_type="factual", first_seen="")
        bk = s.bookkeeping_for_key("graph1")
        assert bk is not None
        assert bk["speaker_id"] == "bob"
        assert bk["relation_type"] == "factual"

    def test_bookkeeping_absent_returns_none(self):
        s = MemoryStore()
        assert s.bookkeeping_for_key("nonexistent") is None

    def test_iter_bookkeeping_yields_all_keys(self):
        s = MemoryStore()
        s.set_bookkeeping("k1", speaker_id="alice", relation_type="factual", first_seen="")
        s.set_bookkeeping("k2", speaker_id="bob", relation_type="preference", first_seen="")
        items = dict(s.iter_bookkeeping())
        assert items["k1"]["speaker_id"] == "alice"
        assert items["k2"]["speaker_id"] == "bob"
        assert s.bookkeeping_count() == 2

    def test_delete_also_drops_bookkeeping(self):
        """store.delete must retire bookkeeping automatically."""
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        s.set_bookkeeping("graph1", speaker_id="alice", relation_type="factual", first_seen="")
        s.delete("graph1")
        assert s.bookkeeping_for_key("graph1") is None

    def test_probe_results_never_carry_speaker_id(self):
        """No dict returned by probe carries ``speaker_id`` — no memory source
        emits it (content-only source-result contract), so there is nothing
        to strip.  Covers cache-hit (graph1), plain source-served (graph2),
        confidence-patched source-served (graphC — a stored fingerprint
        routes through the patch arm's ``src = dict(src)`` copy) and
        failure-dict pass-through (graphF)."""
        from paramem.memory.entry import entry_simhash

        carol = {"key": "graphC", "subject": "Carol", "predicate": "lives_in", "object": "Rome"}
        fp = entry_simhash(carol)

        class _StubSource:
            def probe(self, keys_by_tier):
                return {
                    "graph2": {
                        "key": "graph2",
                        "subject": "Bob",
                        "predicate": "lives_in",
                        "object": "Paris",
                        "fact_text": "Bob lives_in Paris",
                    },
                    "graphC": {
                        **carol,
                        "fact_text": "Carol lives_in Rome",
                    },
                    "graphF": {"raw_output": "garbled", "failure_reason": "parse_error"},
                }

        s = MemoryStore()
        # Cache-hit key: bookkeeping already present.
        s.put(
            "episodic",
            "graph1",
            {"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"},
        )
        s.set_bookkeeping("graph1", speaker_id="spk-alice", relation_type="factual", first_seen="")
        s.put_simhash("episodic", "graphC", fp)  # forces the confidence-patch arm

        results = s.probe(
            {"episodic": ["graph1", "graph2", "graphC", "graphF"]}, source=_StubSource()
        )

        for key in ("graph1", "graph2", "graphC"):
            assert "speaker_id" not in results[key]
        assert results["graph1"]["fact_text"] == "Alice lives_in Berlin"
        assert results["graph2"]["subject"] == "Bob"
        assert results["graphC"]["confidence"] == 1.0
        # Failure dicts never carry speaker_id; nothing disturbs them.
        assert results["graphF"] == {"raw_output": "garbled", "failure_reason": "parse_error"}

    def test_probe_cold_disk_source_key_does_not_backfill_bookkeeping(self):
        """A cold (previously-unbookkept) key served by a disk source on a
        cache miss must NOT gain a bookkeeping record from that probe — the
        fold publishes bookkeeping durably itself, and probe is no longer a
        bookkeeping writer."""
        from paramem.memory.entry import entry_simhash

        entry = {"key": "graph9", "subject": "Dana", "predicate": "lives_in", "object": "Oslo"}

        class _DiskLikeSource:
            def probe(self, keys_by_tier):
                # Content-only — matches DiskMemorySource's post-fix contract.
                return {"graph9": dict(entry)}

        s = MemoryStore()
        s.put_simhash("episodic", "graph9", entry_simhash(entry))
        assert s.bookkeeping_for_key("graph9") is None

        results = s.probe({"episodic": ["graph9"]}, source=_DiskLikeSource(), memoize=True)

        assert results["graph9"]["subject"] == "Dana"
        assert s.get("graph9") is not None  # content still memoized
        assert s.bookkeeping_for_key("graph9") is None  # no backfill

    def test_probe_has_no_speaker_id_parameter(self):
        """MemoryStore.probe takes no speaker_id kwarg — passing one raises."""
        s = MemoryStore()
        with pytest.raises(TypeError):
            s.probe({"episodic": []}, speaker_id="x")

    def test_cache_off_empty_store_always_probes(self):
        """Under cache-off the store has no entries.  Every key must miss
        and be delegated to the source — restores preload_cache=False contract."""
        probed: list = []

        class _FakeSource:
            def probe(self, keys_by_tier):
                probed.extend(k for keys in keys_by_tier.values() for k in keys)
                return {
                    k: {"key": k, "subject": "X", "predicate": "p", "object": "Y"}
                    for keys in keys_by_tier.values()
                    for k in keys
                }

        s = MemoryStore()
        # Bookkeeping loaded but NO entries (cache-off scenario).
        s.set_bookkeeping("k1", speaker_id="alice", relation_type="factual", first_seen="")
        s.set_bookkeeping("k2", speaker_id="alice", relation_type="factual", first_seen="")
        # Register keys in the registry so tier resolution works.
        from paramem.training.key_registry import KeyRegistry

        reg = KeyRegistry()
        reg.add("k1")
        reg.add("k2")
        s.load_registry("episodic", reg)
        results = s.probe({"episodic": ["k1", "k2"]}, source=_FakeSource())
        assert set(probed) == {"k1", "k2"}
        assert results["k1"] is not None
        assert results["k2"] is not None

    def test_snapshot_excludes_bookkeeping(self):
        """_bookkeeping must not enter snapshot — it is boot-loaded from disk."""
        s = MemoryStore()
        s.set_bookkeeping("graph1", speaker_id="alice", relation_type="factual", first_seen="")
        snap = s.snapshot()
        assert "_bookkeeping" not in snap
        assert "bookkeeping" not in snap

    # -- relation_type round-trip tests --

    def test_relation_type_round_trips(self):
        """set_bookkeeping with relation_type='preference' → bookkeeping_for_key returns it."""
        s = MemoryStore()
        s.set_bookkeeping("graph1", speaker_id="alice", relation_type="preference", first_seen="")
        bk = s.bookkeeping_for_key("graph1")
        assert bk is not None
        assert bk["relation_type"] == "preference"

    def test_relation_type_overwrite_idempotent(self):
        """A second set_bookkeeping call updates relation_type in place."""
        s = MemoryStore()
        s.set_bookkeeping("graph1", speaker_id="alice", relation_type="factual", first_seen="")
        s.set_bookkeeping("graph1", speaker_id="alice", relation_type="preference", first_seen="")
        bk = s.bookkeeping_for_key("graph1")
        assert bk is not None
        assert bk["relation_type"] == "preference"

    def test_load_bookkeeping_from_disk_reads_relation_type(self, tmp_path):
        """load_bookkeeping_from_disk reads relation_type from key_metadata.json."""
        import json

        path = tmp_path / "key_metadata.json"
        path.write_text(
            json.dumps(
                {
                    "keys": {
                        "graph1": {
                            "reinforcement_count": 1,
                            "last_reinforced_cycle": 0,
                            "last_seen": "",
                            "first_seen": "",
                            "speaker_id": "alice",
                            "relation_type": "preference",
                        }
                    }
                }
            )
        )
        s = MemoryStore()
        # Register graph1 so it is not treated as an orphan.
        reg = KeyRegistry()
        reg.add("graph1")
        s.load_registry("episodic", reg)
        stats = s.load_bookkeeping_from_disk(path)
        assert stats["loaded"] == 1
        assert stats["orphaned"] == 0
        bk = s.bookkeeping_for_key("graph1")
        assert bk is not None
        assert bk["relation_type"] == "preference"

    def test_load_bookkeeping_from_disk_incomplete_record_fails_loud(self, tmp_path):
        """A persisted record missing a mandatory bookkeeping field raises —
        no legacy-fill tolerance, no backward compatibility. The splat read
        (``set_bookkeeping(key, **key_meta, ...)``) requires the on-disk
        record to carry every field ``set_bookkeeping`` needs."""
        import json

        path = tmp_path / "key_metadata.json"
        path.write_text(
            json.dumps(
                {
                    "keys": {
                        "graph1": {
                            "speaker_id": "alice",
                            "first_seen": "",
                            # relation_type deliberately absent — no tolerance.
                        }
                    }
                }
            )
        )
        s = MemoryStore()
        reg = KeyRegistry()
        reg.add("graph1")
        s.load_registry("episodic", reg)
        with pytest.raises(TypeError):
            s.load_bookkeeping_from_disk(path)


# ---------------------------------------------------------------------------
# SimHash confidence gate — Bug-B read-time gate tests
# ---------------------------------------------------------------------------


class TestConfidenceGate:
    """Locks the SimHash confidence gate applied at probe-time on both
    cache-hit and source-result paths (Bug B fix).

    The gate must satisfy three invariants:
    1. Cache-on and cache-off serve the SAME key set for the same registry.
    2. replay_enabled=False is an unconditional pass-through (no fingerprints).
    3. A key whose fingerprint matches its content is served with real confidence.
    """

    def _spo_entry(
        self,
        key: str,
        subject: str = "Alice",
        predicate: str = "lives_in",
        obj: str = "Berlin",
    ) -> dict:
        return {"key": key, "subject": subject, "predicate": predicate, "object": obj}

    def _correct_fingerprint(self, entry: dict) -> int:
        """Compute the expected SimHash fingerprint for *entry*.

        Routed through :func:`entry_simhash` — the production registration
        primitive — so the space/underscore fold it applies matches what
        :func:`verify_confidence` recomputes at recall time.
        """
        from paramem.memory.entry import entry_simhash

        return entry_simhash(entry)

    def _mismatched_fingerprint(self, entry: dict) -> int:
        """Return a fingerprint that will NOT match *entry* (different content)."""
        from paramem.memory.entry import compute_simhash

        # Compute for a completely different triple so the Hamming distance is large.
        return compute_simhash("wrong_key", "Eve", "hates", "Brussels")

    def test_cache_on_off_parity(self):
        """A key whose stored fingerprint mismatches its content is DROPPED on
        the cache-hit path exactly as on the source path.

        Reproduces the live 239-vs-234 divergence: cache-on served 5 extra
        keys that had been ungated when preloaded without a registry.  After
        the fix, both paths must serve the same set.

        Build:
        - key "graph1": CORRECT fingerprint (should be served on both paths).
        - key "graph2": MISMATCHED fingerprint (should be dropped on both paths).

        Cache-off path: entries absent → source probe called → gate applied to
        source result BEFORE returning.
        Cache-on path: entries present → _render called → gate applied to cached entry.
        Both must produce the same result: graph1 served, graph2 None."""
        entry_good = self._spo_entry("graph1")
        entry_bad = self._spo_entry("graph2")
        fp_good = self._correct_fingerprint(entry_good)
        fp_bad = self._mismatched_fingerprint(entry_bad)  # wrong hash for graph2's content

        # Cache-OFF path: store is empty, source must return the entries.
        probed: list = []

        class _FakeSource:
            def probe(self, keys_by_tier):
                probed.extend(k for keys in keys_by_tier.values() for k in keys)
                # Returns both entries (source itself does NOT gate here — the
                # store boundary gate is the authority).
                return {
                    "graph1": {**entry_good, "confidence": 1.0},
                    "graph2": {**entry_bad, "confidence": 1.0},
                }

        s_off = MemoryStore()
        s_off.put_simhash("episodic", "graph1", fp_good)
        s_off.put_simhash("episodic", "graph2", fp_bad)
        off_results = s_off.probe({"episodic": ["graph1", "graph2"]}, source=_FakeSource())
        assert "graph1" in probed and "graph2" in probed, "source must be called for misses"
        assert off_results["graph1"] is not None, "good key must be served on cache-off"
        assert off_results["graph2"] is None, "bad key must be dropped on cache-off"

        # Cache-ON path: entries pre-populated (simulating boot preload).
        s_on = MemoryStore()
        s_on.put_simhash("episodic", "graph1", fp_good)
        s_on.put_simhash("episodic", "graph2", fp_bad)
        s_on.put("episodic", "graph1", entry_good, register=False)
        s_on.put("episodic", "graph2", entry_bad, register=False)
        on_results = s_on.probe({"episodic": ["graph1", "graph2"]}, source=None)
        assert on_results["graph1"] is not None, "good key must be served on cache-on"
        assert on_results["graph2"] is None, "bad key must be dropped on cache-on"

        # The served sets must be identical (parity).
        cache_on_served = {k for k, v in on_results.items() if v is not None}
        cache_off_served = {k for k, v in off_results.items() if v is not None}
        assert cache_on_served == cache_off_served, (
            f"cache-on and cache-off must serve the same set; "
            f"on={cache_on_served} off={cache_off_served}"
        )

    def test_replay_off_passthrough(self):
        """With replay_enabled=False (no _simhash), all entries are served.

        The gate must be a complete no-op when replay is disabled — no
        fingerprints exist by design, and the store must not over-drop."""
        entry = self._spo_entry("graph1")
        s = MemoryStore(replay_enabled=False)
        s.put("episodic", "graph1", entry, register=False)
        # Explicitly verify no simhash is set (replay-off means no fingerprints).
        assert not s.has_simhash("episodic", "graph1")
        results = s.probe({"episodic": ["graph1"]}, source=None)
        assert results["graph1"] is not None, "replay-off must serve all entries ungated"
        assert results["graph1"]["confidence"] == 1.0, (
            "replay-off confidence must be 1.0 (pass-through)"
        )

    def test_confident_key_served(self):
        """A key with a matching fingerprint passes the gate and is served.

        The rendered result must include the real computed confidence (not
        the hardcoded 1.0 that existed before the fix)."""
        entry = self._spo_entry("graph1")
        fp = self._correct_fingerprint(entry)
        s = MemoryStore()
        s.put("episodic", "graph1", entry, register=False)
        s.put_simhash("episodic", "graph1", fp)
        results = s.probe({"episodic": ["graph1"]}, source=None)
        assert results["graph1"] is not None, "correctly-fingerprinted key must be served"
        # Confidence must be exactly 1.0 since the fingerprint was computed from
        # the same content (identical simhash → Hamming distance 0).
        assert results["graph1"]["confidence"] == 1.0, (
            f"matching fingerprint must yield confidence 1.0, got {results['graph1']['confidence']}"
        )


# ---------------------------------------------------------------------------
# discard_keys helper — erase and stale mode variants
# ---------------------------------------------------------------------------


class TestDiscardKeys:
    """MemoryStore.discard_keys(keys, mode=) — erase and stale variants.

    Covers: mode="erase" removes from active + drops simhash across tiers
    (registry over tiers_with_registry, simhash over three main tiers only);
    mode="stale" moves to stale + RETAINS simhash; idempotent on absent key;
    no-op when replay disabled.
    """

    def _make_store_with_key(self, key: str = "graph1", tier: str = "episodic") -> MemoryStore:
        """Build a MemoryStore with one registered key plus its simhash."""
        s = MemoryStore(replay_enabled=True)
        s.put(tier, key, _entry(key), simhash=0xDEADBEEF, register=True)
        return s

    def test_erase_removes_from_active_and_drops_simhash(self):
        """mode='erase': key removed from active list; simhash dropped."""
        s = self._make_store_with_key("graph1", "episodic")
        s.discard_keys(["graph1"], mode="erase")

        reg = s.registry("episodic")
        assert "graph1" not in reg.list_active(), "Erased key must not be in active"
        assert "graph1" not in reg.list_stale(), "Erased key must not be in stale either"
        assert not s.has_simhash("episodic", "graph1"), "Erased key simhash must be dropped"

    def test_erase_drops_simhash_via_registry_remove(self):
        """mode='erase': registry.remove drops the simhash from all tiers the key
        is known in.

        After unification, discard_keys(mode='erase') uses registry.remove, which
        drops the fingerprint from _simhash atomically.  A key registered in an
        interim tier's registry loses its fingerprint there too.
        """
        s = MemoryStore(replay_enabled=True)
        # Register key in the interim tier (which owns both its registry and simhash).
        s.put(
            "episodic_interim_20260101",
            "graph_interim",
            _entry("graph_interim"),
            simhash=0x2222,
            register=True,
        )

        s.discard_keys(["graph_interim"], mode="erase")

        # Interim tier simhash dropped (registry.remove covers all tiers_with_registry).
        assert not s.has_simhash("episodic_interim_20260101", "graph_interim"), (
            "Registry-owned simhash must be dropped by erase"
        )

    def test_stale_moves_to_stale_and_retains_simhash(self):
        """mode='stale': key moved to stale partition; simhash retained."""
        s = self._make_store_with_key("graph1", "episodic")
        s.discard_keys(["graph1"], mode="stale")

        reg = s.registry("episodic")
        assert "graph1" not in reg.list_active(), "Staled key must not be in active"
        assert "graph1" in reg.list_stale(), "Staled key must be in stale partition"
        assert s.has_simhash("episodic", "graph1"), "Staled key simhash must be RETAINED"

    def test_erase_idempotent_on_absent_key(self):
        """discard_keys(mode='erase') on a non-existent key does not raise."""
        s = MemoryStore(replay_enabled=True)
        # Should not raise
        s.discard_keys(["nonexistent_key"], mode="erase")

    def test_stale_idempotent_on_absent_key(self):
        """discard_keys(mode='stale') on a non-existent key does not raise."""
        s = MemoryStore(replay_enabled=True)
        s.discard_keys(["nonexistent_key"], mode="stale")

    def test_noop_when_replay_disabled(self):
        """When replay is disabled, discard_keys is a no-op."""
        s = MemoryStore(replay_enabled=False)
        s.put_simhash("episodic", "graph1", 0xAAAA)
        # Replay disabled → must not raise and must leave simhash intact.
        s.discard_keys(["graph1"], mode="erase")
        assert s.has_simhash("episodic", "graph1"), (
            "Replay-disabled discard_keys must not touch simhashes"
        )

    def test_invalid_mode_raises_value_error(self):
        """An unrecognised mode string raises ValueError."""
        s = MemoryStore(replay_enabled=True)
        with pytest.raises(ValueError, match="unknown mode"):
            s.discard_keys(["graph1"], mode="invalid")

    def test_is_stale_delegates_to_registry(self):
        """MemoryStore.is_stale(key) returns True iff the owning tier marks it stale."""
        s = self._make_store_with_key("graph1", "episodic")
        assert not s.is_stale("graph1")
        s.discard_keys(["graph1"], mode="stale")
        assert s.is_stale("graph1")
        assert not s.is_stale("nonexistent")

    def test_erase_drops_entry_and_bookkeeping_sibling_key_keeps_both(self):
        """mode='erase' is full retirement: entry payload and bookkeeping
        record are dropped, delegating to :meth:`delete` per key.  A sibling
        key in the same tier keeps both."""
        s = self._make_store_with_key("graph1", "episodic")
        s.set_bookkeeping(
            "graph1", speaker_id="speaker1", relation_type="factual", first_seen="2026-01-01"
        )
        s.put("episodic", "graph2", _entry("graph2"), simhash=0xC0FFEE, register=True)
        s.set_bookkeeping(
            "graph2", speaker_id="speaker1", relation_type="factual", first_seen="2026-01-01"
        )

        s.discard_keys(["graph1"], mode="erase")

        assert s.get("graph1") is None, "erase must drop the entry payload"
        assert s.bookkeeping_for_key("graph1") is None, "erase must drop the bookkeeping record"
        assert s.get("graph2") is not None, "sibling key's entry must survive"
        assert s.bookkeeping_for_key("graph2") is not None, (
            "sibling key's bookkeeping record must survive"
        )

    def test_stale_keeps_entry_and_bookkeeping(self):
        """mode='stale' must NOT leak the erase-mode entry/bookkeeping drop
        into the soft path — entry and bookkeeping both survive."""
        s = self._make_store_with_key("graph1", "episodic")
        s.set_bookkeeping(
            "graph1", speaker_id="speaker1", relation_type="factual", first_seen="2026-01-01"
        )

        s.discard_keys(["graph1"], mode="stale")

        assert s.get("graph1") is not None, "stale must keep the entry payload"
        assert s.bookkeeping_for_key("graph1") is not None, "stale must keep the bookkeeping record"

    def test_all_stale_keys_aggregates_across_tiers(self):
        """MemoryStore.all_stale_keys() returns stale keys across all registered tiers."""
        s = MemoryStore(replay_enabled=True)
        s.put("episodic", "ep1", _entry("ep1"), register=True)
        s.put("semantic", "sem1", _entry("sem1"), register=True)
        s.put("episodic", "ep2", _entry("ep2"), register=True)

        s.discard_keys(["ep1"], mode="stale")
        s.discard_keys(["sem1"], mode="stale")

        stale = s.all_stale_keys()
        assert "ep1" in stale
        assert "sem1" in stale
        assert "ep2" not in stale


# ---------------------------------------------------------------------------
# Thread-safety concurrency contract
# ---------------------------------------------------------------------------


class TestConcurrencyContract:
    """Verify the RLock concurrency contract.

    These tests do NOT test for race conditions (that would require a stress
    harness with tight timing).  Instead they verify the observable contract:
    - iter_entries / iter_bookkeeping snapshot under the lock → their result
      is unaffected by a concurrent (but serialised) mutation.
    - swap() atomically rebinds all three structures.
    - entries_in_tier() returns a copy, not a live internal dict.
    - The store exposes an RLock (not plain Lock) on ``_lock``.
    """

    def test_rlock_is_reentrant(self):
        """_lock is an RLock; the same thread can acquire it twice without deadlock."""
        s = MemoryStore()
        # RLock allows re-entry from the same thread.
        acquired_inner = False
        with s._lock:
            with s._lock:
                acquired_inner = True
        assert acquired_inner, "_lock must be an RLock (re-entrant)"

    def test_entries_in_tier_returns_copy_not_live(self):
        """entries_in_tier must return a snapshot copy.

        Mutations to the returned dict must NOT propagate to the store."""
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        copy_dict = s.entries_in_tier("episodic")
        # Mutate the returned copy.
        copy_dict["injected"] = {"key": "injected"}
        # Store must be unaffected.
        assert "injected" not in s.entries_in_tier("episodic")
        assert s.get("injected") is None

    def test_iter_entries_snapshot_unaffected_by_subsequent_put(self):
        """iter_entries snapshot is taken before yielding.

        A put() that happens after the iterator is created must NOT appear
        in the iteration (because the snapshot is fixed at creation time)."""
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        it = s.iter_entries()
        # Materialise the generator to exhaust the snapshot.
        snap = list(it)
        # Now add a second key.
        s.put("episodic", "graph2", _entry("graph2"))
        # The snapshot must only contain graph1 (the generator was already
        # exhausted before graph2 was inserted).
        keys_in_snap = {k for _, k, _ in snap}
        assert "graph1" in keys_in_snap
        assert "graph2" not in keys_in_snap

    def test_iter_bookkeeping_snapshot_unaffected_by_subsequent_set(self):
        """iter_bookkeeping snapshot is taken before yielding."""
        s = MemoryStore()
        s.set_bookkeeping("k1", speaker_id="alice", relation_type="factual", first_seen="")
        it = s.iter_bookkeeping()
        snap = list(it)
        # Add a second key after the iterator is exhausted.
        s.set_bookkeeping("k2", speaker_id="bob", relation_type="factual", first_seen="")
        keys_in_snap = {k for k, _ in snap}
        assert "k1" in keys_in_snap
        assert "k2" not in keys_in_snap

    def test_swap_rebinds_all_three_structures_atomically(self):
        """swap() replaces entries, registry, and bookkeeping in one operation.

        After swap, get/iter_entries/bookkeeping_for_key all reflect the new
        state and the old keys are gone."""
        s = MemoryStore()
        s.put("episodic", "old_key", _entry("old_key"))
        s.set_bookkeeping("old_key", speaker_id="alice", relation_type="factual", first_seen="")

        new_reg = KeyRegistry()
        new_reg.add("new_key")
        new_entries: dict[str, dict[str, dict]] = {"semantic": {"new_key": _entry("new_key")}}
        new_bookkeeping: dict[str, dict] = {
            "new_key": {
                "speaker_id": "bob",
                "relation_type": "preference",
                "reinforcement_count": 1,
                "last_reinforced_cycle": 0,
                "last_seen": "",
            }
        }
        s.swap(new_entries, {"semantic": new_reg}, new_bookkeeping)

        # Old key is gone.
        assert s.get("old_key") is None
        assert s.bookkeeping_for_key("old_key") is None
        # New key is present.
        assert s.get("new_key") == _entry("new_key")
        assert s.tier_of("new_key") == "semantic"
        bk = s.bookkeeping_for_key("new_key")
        assert bk is not None
        assert bk["speaker_id"] == "bob"
        # Registry reflects the swap.
        assert "new_key" in s.registry("semantic")
        assert not s.has_registry("episodic")

    def test_swap_visible_to_concurrent_reader_after_release(self):
        """A thread that acquires the lock after swap() sees the new state.

        This is a serialised (not truly concurrent) test: it proves the lock
        is not bypassed by swap() — the new state is visible to the next
        acquirer."""
        s = MemoryStore()
        s.put("episodic", "before", _entry("before"))

        results: list = []

        def reader():
            results.append(s.get("before"))
            results.append(s.get("after"))

        new_entries: dict[str, dict[str, dict]] = {"episodic": {"after": _entry("after")}}
        s.swap(new_entries, {}, {})

        t = threading.Thread(target=reader)
        t.start()
        t.join()

        assert results[0] is None, "old key must be gone after swap"
        assert results[1] == _entry("after"), "new key must be visible after swap"


# ---------------------------------------------------------------------------
# read_registries_from_disk — store-free static builder (phase-2)
# ---------------------------------------------------------------------------


class TestReadRegistriesFromDisk:
    """read_registries_from_disk returns a fresh dict without touching any store."""

    def test_returns_dict_with_main_tiers_on_empty_dir(self, tmp_path) -> None:
        """All three main tiers appear even when their registry files are absent.

        KeyRegistry.load on a missing file returns an empty registry — so the
        returned dict always has "episodic", "semantic", "procedural" keys."""
        # Create the adapter dir structure but leave registry files absent.
        for tier in ("episodic", "semantic", "procedural"):
            (tmp_path / tier).mkdir()

        result = MemoryStore.read_registries_from_disk(tmp_path)

        assert set(result.keys()) == {"episodic", "semantic", "procedural"}
        for tier in ("episodic", "semantic", "procedural"):
            assert result[tier].list_active() == []

    def test_does_not_touch_any_live_store(self, tmp_path) -> None:
        """read_registries_from_disk is store-free — the live store is unchanged."""
        for tier in ("episodic", "semantic", "procedural"):
            (tmp_path / tier).mkdir()

        live = MemoryStore()
        live.put("episodic", "live_key", _entry("live_key"))

        # Call the static method — should not touch the live store.
        MemoryStore.read_registries_from_disk(tmp_path)

        # live store must be unmodified.
        assert live.get("live_key") == _entry("live_key")

    def test_load_registries_from_disk_delegates_to_static(self, tmp_path) -> None:
        """load_registries_from_disk (instance method) produces the same registry
        as read_registries_from_disk (static) on the same adapter dir."""
        for tier in ("episodic", "semantic", "procedural"):
            (tmp_path / tier).mkdir()

        static_result = MemoryStore.read_registries_from_disk(tmp_path)

        store = MemoryStore()
        store.load_registries_from_disk(tmp_path)

        for tier in ("episodic", "semantic", "procedural"):
            assert list(static_result[tier].list_active()) == list(
                store.registry(tier).list_active()
            ), f"Tier '{tier}' active keys differ between static and instance method"


# ---------------------------------------------------------------------------
# set_bookkeeping guard — no-unattributed-keys invariant
# ---------------------------------------------------------------------------


class TestSetBookkeepingGuard:
    """set_bookkeeping raises on an empty speaker_id unless the caller opts out.

    ``MemoryStore.set_bookkeeping`` rejects ``speaker_id=""`` with ValueError
    (no-unattributed-keys invariant) unless ``allow_empty_speaker=True`` is
    passed — the carve-out reserved for reload paths and keyless concept-node
    edges that genuinely have no speaker.
    """

    def test_empty_speaker_id_raises_without_allow_flag(self):
        """set_bookkeeping(speaker_id='') raises ValueError by default."""
        s = MemoryStore()
        with pytest.raises(ValueError, match="no-unattributed-keys invariant"):
            s.set_bookkeeping(
                "graph1",
                speaker_id="",
                relation_type="factual",
                first_seen="",
            )

    def test_empty_speaker_id_allowed_with_flag(self):
        """set_bookkeeping(speaker_id='', allow_empty_speaker=True) succeeds."""
        s = MemoryStore()
        s.set_bookkeeping(
            "graph2",
            speaker_id="",
            relation_type="factual",
            allow_empty_speaker=True,
            first_seen="",
        )
        bk = s.bookkeeping_for_key("graph2")
        assert bk is not None
        assert bk["speaker_id"] == ""

    def test_nonempty_speaker_id_succeeds_without_flag(self):
        """Non-empty speaker_id always succeeds (no flag needed).

        A cased ``Speaker0`` is accepted and normalized to lowercase ``speaker0``
        by :meth:`set_bookkeeping`'s ``is_speaker_id`` gate."""
        s = MemoryStore()
        s.set_bookkeeping(
            "graph3",
            speaker_id="Speaker0",
            relation_type="factual",
            first_seen="",
        )
        bk = s.bookkeeping_for_key("graph3")
        assert bk is not None
        assert bk["speaker_id"] == "speaker0"

    def test_cased_speaker_id_normalized_to_lowercase(self):
        """set_bookkeeping normalizes is_speaker_id values to lowercase.

        Cased ``Speaker0`` is coerced to ``speaker0``; the router's
        ``_speaker_key_index`` receives the normalized form, eliminating the
        silent-drop regression where legacy key_metadata.json held cased ids."""
        s = MemoryStore()
        s.set_bookkeeping("g1", speaker_id="Speaker0", relation_type="factual", first_seen="")
        bk = s.bookkeeping_for_key("g1")
        assert bk is not None
        assert bk["speaker_id"] == "speaker0", (
            "set_bookkeeping must lowercase Speaker0 → speaker0 to match the router index."
        )

    def test_empty_speaker_id_passes_through_with_flag(self):
        """Empty speaker_id passes through (allow_empty_speaker=True); not lowercased."""
        s = MemoryStore()
        s.set_bookkeeping(
            "g2",
            speaker_id="",
            relation_type="factual",
            allow_empty_speaker=True,
            first_seen="",
        )
        bk = s.bookkeeping_for_key("g2")
        assert bk is not None
        assert bk["speaker_id"] == ""

    def test_non_speaker_value_passes_through_unchanged(self):
        """A non-speaker_id value that is non-empty passes through without lowercasing."""
        s = MemoryStore()
        s.set_bookkeeping("g3", speaker_id="alice", relation_type="factual", first_seen="")
        bk = s.bookkeeping_for_key("g3")
        assert bk is not None
        assert bk["speaker_id"] == "alice"

    def test_load_bookkeeping_legacy_cased_speaker_id_normalized(self, tmp_path):
        """Legacy key_metadata.json with cased Speaker0 is lowercased at boot.

        :meth:`load_bookkeeping_from_disk` calls :meth:`set_bookkeeping`, which
        normalizes ``Speaker0`` → ``speaker0`` via ``is_speaker_id`` — the same
        normalization the router's ``_speaker_key_index`` (the sole privacy
        boundary) relies on to route legacy-cased ids without dropping the key
        (the live regression this fix targets: 158 keys silently dropped after
        the speaker-identity refactor)."""
        import json

        path = tmp_path / "key_metadata.json"
        path.write_text(
            json.dumps(
                {
                    "keys": {
                        "graph42": {
                            "speaker_id": "Speaker0",
                            "relation_type": "factual",
                            "reinforcement_count": 1,
                            "last_reinforced_cycle": 0,
                            "last_seen": "",
                            "first_seen": "",
                        }
                    }
                }
            )
        )
        s = MemoryStore()
        from paramem.training.key_registry import KeyRegistry

        reg = KeyRegistry()
        reg.add("graph42")
        s.load_registry("episodic", reg)
        s.put(
            "episodic",
            "graph42",
            {"key": "graph42", "subject": "speaker0", "predicate": "lives_in", "object": "London"},
        )
        s.load_bookkeeping_from_disk(path)

        bk = s.bookkeeping_for_key("graph42")
        assert bk is not None
        assert bk["speaker_id"] == "speaker0", (
            "Legacy cased Speaker0 must be normalized to speaker0 at boot."
        )

    def test_router_routes_by_legacy_cased_speaker_id_after_normalization(self, monkeypatch):
        """Cased-bookkeeping → normalized → routable: ``QueryRouter``'s
        ``_speaker_key_index`` (sole privacy boundary) is built from the
        same normalized bookkeeping the preceding test locks, so
        ``route(speaker_id="speaker0")`` finds a key bookkept as
        ``Speaker0``."""
        from pathlib import Path
        from unittest.mock import MagicMock

        from paramem.server.router import Intent, QueryRouter

        s = MemoryStore()
        s.put(
            "episodic",
            "graph42",
            {"key": "graph42", "subject": "speaker0", "predicate": "lives_in", "object": "London"},
        )
        s.set_bookkeeping("graph42", speaker_id="Speaker0", relation_type="factual", first_seen="")

        monkeypatch.setattr(
            "paramem.server.intent.classify_intent", MagicMock(return_value=Intent.PERSONAL)
        )
        router = QueryRouter(adapter_dir=Path("/nonexistent"), memory_store=s)
        plan = router.route("Where do I live?", speaker_id="speaker0")

        all_keys = [k for step in plan.steps for k in step.keys_to_probe]
        assert "graph42" in all_keys, (
            "route(speaker_id='speaker0') must find a key bookkept under cased "
            "'Speaker0' — the router index is the sole privacy boundary."
        )


# ---------------------------------------------------------------------------
# probe renders speaker{N} tokens verbatim — no resolver in probe
# ---------------------------------------------------------------------------


class TestProbeRendersTokensVerbatim:
    """MemoryStore.probe renders ``fact_text`` with raw ``speaker{N}`` tokens
    on both the cache-hit and cache-miss/source-passthrough paths.  There is
    no resolver parameter on ``probe`` — display-name resolution happens
    exactly once, at the reply boundary, via
    :func:`paramem.server.speaker.resolve_speaker_tokens`, never inside probe.
    """

    def test_cache_hit_subject_token_verbatim(self) -> None:
        """Cache-hit path: fact_text carries the raw subject token as-is."""
        s = MemoryStore()
        s.put(
            "episodic",
            "graph1",
            {"key": "graph1", "subject": "speaker0", "predicate": "lives_in", "object": "Berlin"},
        )
        s.set_bookkeeping("graph1", speaker_id="speaker0", relation_type="factual", first_seen="")
        results = s.probe({"episodic": ["graph1"]})
        assert results["graph1"]["fact_text"] == "speaker0 lives_in Berlin"

    def test_cache_hit_object_token_verbatim(self) -> None:
        """Cache-hit path: fact_text carries the raw object token as-is."""
        s = MemoryStore()
        s.put(
            "episodic",
            "graph2",
            {"key": "graph2", "subject": "speaker0", "predicate": "knows", "object": "speaker9"},
        )
        s.set_bookkeeping("graph2", speaker_id="speaker0", relation_type="factual", first_seen="")
        results = s.probe({"episodic": ["graph2"]})
        assert results["graph2"]["fact_text"] == "speaker0 knows speaker9"

    def test_source_miss_path_verbatim(self) -> None:
        """Cache-MISS / source passthrough: the source's own fact_text (already
        rendered verbatim by the source layer) passes through unmodified —
        probe does not re-render or resolve it."""

        class _FakeSource:
            def probe(self, keys_by_tier):
                return {
                    "graph9": {
                        "key": "graph9",
                        "subject": "speaker9",
                        "predicate": "lives_in",
                        "object": "Paris",
                        "fact_text": "speaker9 lives_in Paris",
                    }
                }

        s = MemoryStore()
        # No cache entry for graph9 — forces source path.
        results = s.probe(
            {"episodic": ["graph9"]},
            source=_FakeSource(),
            memoize=False,
        )
        assert results["graph9"]["fact_text"] == "speaker9 lives_in Paris"

    def test_memoized_stash_reconstructs_from_spo(self) -> None:
        """A source-miss result, once memoized, is re-rendered from its raw SPO
        fields on the next cache hit — the stash holds SPO, not a frozen
        fact_text string."""

        class _FakeSource:
            def probe(self, keys_by_tier):
                return {
                    "graphM": {
                        "key": "graphM",
                        "subject": "speaker9",
                        "predicate": "works_at",
                        "object": "Acme",
                        "fact_text": "speaker9 works_at Acme",
                    }
                }

        s = MemoryStore()
        s.probe({"episodic": ["graphM"]}, source=_FakeSource(), memoize=True)

        # Second probe is a cache hit — re-rendered from the memoized SPO.
        results2 = s.probe({"episodic": ["graphM"]})
        assert results2["graphM"]["fact_text"] == "speaker9 works_at Acme"

    def test_probe_has_no_resolver_parameter(self) -> None:
        """MemoryStore.probe takes no speaker_resolver kwarg — passing one raises."""
        import pytest

        s = MemoryStore()
        with pytest.raises(TypeError):
            s.probe({"episodic": []}, speaker_resolver=lambda t: t)


# ---------------------------------------------------------------------------
# read_simhash_registry_from_disk — opt-in cache
# ---------------------------------------------------------------------------


def _write_episodic_registry(adapter_dir, fingerprints: dict[str, int]) -> None:
    """Write a minimal ``episodic/indexed_key_registry.json`` under *adapter_dir*."""
    reg = KeyRegistry()
    for key, fp in fingerprints.items():
        reg.add(key)
        reg.set_simhash(key, fp)
    tier_dir = adapter_dir / "episodic"
    tier_dir.mkdir(parents=True, exist_ok=True)
    (tier_dir / "indexed_key_registry.json").write_bytes(reg.save_bytes())


@pytest.fixture(autouse=True)
def _clear_simhash_registry_cache():
    """Every test in this module gets a clean process-wide cache — a hit
    left behind by one test must never leak into the next."""
    from paramem.memory.store import invalidate_simhash_registry_cache

    invalidate_simhash_registry_cache()
    yield
    invalidate_simhash_registry_cache()


class TestReadSimhashRegistryFromDiskCache:
    """``cached=True`` is opt-in: default behaviour is unchanged (always
    disk-truth), and the cache is fresh until explicitly invalidated."""

    def test_default_uncached_reflects_disk_every_call(self, tmp_path) -> None:
        _write_episodic_registry(tmp_path, {"graph1": 111})
        first = MemoryStore.read_simhash_registry_from_disk(tmp_path)
        assert first == {"graph1": 111}

        _write_episodic_registry(tmp_path, {"graph1": 222})
        second = MemoryStore.read_simhash_registry_from_disk(tmp_path)
        assert second == {"graph1": 222}

    def test_cached_read_matches_fresh_read(self, tmp_path) -> None:
        _write_episodic_registry(tmp_path, {"graph1": 111, "graph2": 222})
        fresh = MemoryStore.read_simhash_registry_from_disk(tmp_path, cached=False)
        cached = MemoryStore.read_simhash_registry_from_disk(tmp_path, cached=True)
        assert cached == fresh == {"graph1": 111, "graph2": 222}

    def test_cached_read_survives_disk_change_until_invalidated(self, tmp_path) -> None:
        from paramem.memory.store import invalidate_simhash_registry_cache

        _write_episodic_registry(tmp_path, {"graph1": 111})
        first = MemoryStore.read_simhash_registry_from_disk(tmp_path, cached=True)
        assert first == {"graph1": 111}

        # Mutate the registry file on disk without invalidating the cache.
        _write_episodic_registry(tmp_path, {"graph1": 999})
        stale = MemoryStore.read_simhash_registry_from_disk(tmp_path, cached=True)
        assert stale == {"graph1": 111}, "cache must serve the old mapping until invalidated"

        invalidate_simhash_registry_cache()
        fresh = MemoryStore.read_simhash_registry_from_disk(tmp_path, cached=True)
        assert fresh == {"graph1": 999}

    def test_invalidate_clears_whole_cache(self, tmp_path) -> None:
        from paramem.memory.store import invalidate_simhash_registry_cache

        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        _write_episodic_registry(dir_a, {"graph1": 1})
        _write_episodic_registry(dir_b, {"graph2": 2})
        MemoryStore.read_simhash_registry_from_disk(dir_a, cached=True)
        MemoryStore.read_simhash_registry_from_disk(dir_b, cached=True)

        _write_episodic_registry(dir_a, {"graph1": 999})
        _write_episodic_registry(dir_b, {"graph2": 999})
        invalidate_simhash_registry_cache()

        # invalidate_simhash_registry_cache() takes no argument — it always
        # clears the whole process-wide cache, every adapter_dir at once.
        assert MemoryStore.read_simhash_registry_from_disk(dir_a, cached=True) == {"graph1": 999}
        assert MemoryStore.read_simhash_registry_from_disk(dir_b, cached=True) == {"graph2": 999}

    def test_invalidate_takes_no_argument(self) -> None:
        """Signature guard: passing an argument is a TypeError, not a
        per-adapter_dir invalidation — there is no such arm any more."""
        from paramem.memory.store import invalidate_simhash_registry_cache

        with pytest.raises(TypeError):
            invalidate_simhash_registry_cache("/some/adapter/dir")

    def test_invalidation_mid_walk_is_returned_but_not_published(
        self, tmp_path, monkeypatch
    ) -> None:
        """An invalidation landing while a disk walk is in flight must not
        corrupt the cache: the walked result still reaches this call's own
        caller (a walk in progress owes its result to whoever started it),
        but the generation guard discards the publish — the next cached
        read re-walks instead of ever serving that now-possibly-stale
        value from the cache."""
        from paramem.memory.store import invalidate_simhash_registry_cache

        _write_episodic_registry(tmp_path, {"graph1": 111})

        # Captured before monkeypatching — stays bound to the real
        # implementation regardless of the patch below.
        original_load_simhashes = KeyRegistry.load_simhashes
        calls = {"n": 0}

        def _racing_load_simhashes(cls, path):
            calls["n"] += 1
            if calls["n"] == 1:
                # A concurrent registry mutation + invalidation landing
                # while this walk is still in flight.
                invalidate_simhash_registry_cache()
            return original_load_simhashes(path)

        monkeypatch.setattr(KeyRegistry, "load_simhashes", classmethod(_racing_load_simhashes))

        result = MemoryStore.read_simhash_registry_from_disk(tmp_path, cached=True)
        assert result == {"graph1": 111}, "the walked result is still returned to the caller"

        # The race must have discarded the publish: a second cached read
        # re-walks (observable via the call counter) rather than serving a
        # cache hit from the raced walk.
        calls["n"] = 0
        MemoryStore.read_simhash_registry_from_disk(tmp_path, cached=True)
        assert calls["n"] > 0, "cache entry was not published after the race — must re-walk"


class TestBuildMemorySourceCachedRegistry:
    """``build_memory_source(..., cached_registry=...)`` threads through to
    :meth:`MemoryStore.read_simhash_registry_from_disk`."""

    def test_default_cached_registry_is_disk_truth(self, tmp_path) -> None:
        from paramem.memory.source import build_memory_source

        _write_episodic_registry(tmp_path, {"graph1": 111})
        source = build_memory_source(
            mode="train",
            adapter_dir=tmp_path,
            batch_size=4,
            model=object(),
            tokenizer=object(),
        )
        assert source.registry == {"graph1": 111}

        _write_episodic_registry(tmp_path, {"graph1": 222})
        source2 = build_memory_source(
            mode="train",
            adapter_dir=tmp_path,
            batch_size=4,
            model=object(),
            tokenizer=object(),
        )
        assert source2.registry == {"graph1": 222}

    def test_cached_registry_true_serves_stale_until_router_reload(self, tmp_path) -> None:
        from paramem.memory.source import build_memory_source

        _write_episodic_registry(tmp_path, {"graph1": 111})
        source = build_memory_source(
            mode="train",
            adapter_dir=tmp_path,
            batch_size=4,
            model=object(),
            tokenizer=object(),
            cached_registry=True,
        )
        assert source.registry == {"graph1": 111}

        _write_episodic_registry(tmp_path, {"graph1": 999})
        source2 = build_memory_source(
            mode="train",
            adapter_dir=tmp_path,
            batch_size=4,
            model=object(),
            tokenizer=object(),
            cached_registry=True,
        )
        assert source2.registry == {"graph1": 111}, "stale cache entry survives the disk write"

        from paramem.memory.store import invalidate_simhash_registry_cache

        invalidate_simhash_registry_cache()
        source3 = build_memory_source(
            mode="train",
            adapter_dir=tmp_path,
            batch_size=4,
            model=object(),
            tokenizer=object(),
            cached_registry=True,
        )
        assert source3.registry == {"graph1": 999}


class TestRouterReloadInvalidatesSimhashRegistryCache:
    """``QueryRouter.reload()`` invalidates the process-wide simhash-registry
    cache so a stale opt-in read can never survive past the reload every
    registry-mutating server path ends in."""

    def test_reload_calls_invalidate_simhash_registry_cache(self, tmp_path, monkeypatch) -> None:
        import paramem.server.router as router_module

        calls: list[object] = []
        monkeypatch.setattr(
            router_module,
            "invalidate_simhash_registry_cache",
            lambda *a, **k: calls.append((a, k)),
        )

        router = router_module.QueryRouter(adapter_dir=tmp_path, memory_store=None)
        assert len(calls) == 1, "__init__ calls reload() once"

        calls.clear()
        router.reload()
        assert len(calls) == 1

    def test_reload_actually_clears_a_populated_cache(self, tmp_path) -> None:
        import paramem.server.router as router_module

        _write_episodic_registry(tmp_path, {"graph1": 111})
        MemoryStore.read_simhash_registry_from_disk(tmp_path, cached=True)

        _write_episodic_registry(tmp_path, {"graph1": 999})
        router = router_module.QueryRouter(adapter_dir=tmp_path, memory_store=None)
        router.reload()

        assert MemoryStore.read_simhash_registry_from_disk(tmp_path, cached=True) == {"graph1": 999}
