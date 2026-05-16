"""Unit tests for :class:`paramem.training.memory_store.MemoryStore`.

Locks the per-tier {quads, simhash, registry} contract that replaced the
flat ``indexed_key_cache`` + three flat ``*_simhash`` dicts on
:class:`ConsolidationLoop`.
"""

from __future__ import annotations

import pytest

from paramem.training.key_registry import KeyRegistry
from paramem.training.memory_store import MemoryStore


def _entry(
    key: str, subject: str = "Alice", predicate: str = "lives_in", obj: str = "Berlin"
) -> dict:
    return {
        "key": key,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "speaker_id": "spk-alice",
        "first_seen_cycle": 1,
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

    def test_setdefault_quad_lazily_creates_slot(self):
        s = MemoryStore()
        first = s.setdefault_entry("episodic", "graph1", {"speaker_id": "spk-a"})
        first["first_seen_cycle"] = 5
        # Second call must return the SAME mutable dict (no overwrite)
        second = s.setdefault_entry("episodic", "graph1", {"speaker_id": "spk-b"})
        assert second is first
        assert second["speaker_id"] == "spk-a"
        assert second["first_seen_cycle"] == 5


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

    def test_simhashes_in_tier_is_dict_str_int_for_verify_confidence(self):
        """The view exposed to verify_confidence must be the same shape the
        legacy *_simhash dicts had — key -> 64-bit int."""
        s = MemoryStore()
        s.put_simhash("episodic", "graph1", 0xCAFE)
        s.put_simhash("episodic", "graph2", 0xBEEF)
        view = s.simhashes_in_tier("episodic")
        assert view == {"graph1": 0xCAFE, "graph2": 0xBEEF}
        assert all(isinstance(v, int) for v in view.values())

    def test_replace_simhashes_in_tier_bulk_overwrite(self):
        s = MemoryStore()
        s.put_simhash("episodic", "graph_old", 0xAA)
        s.replace_simhashes_in_tier("episodic", {"graph_new": 0xBB})
        assert s.simhashes_in_tier("episodic") == {"graph_new": 0xBB}

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
# Lifecycle registry — Optional gate
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_replay_enabled_default(self):
        s = MemoryStore()
        assert s.replay_enabled
        assert s.registry("episodic") is not None
        assert isinstance(s.registry("episodic"), KeyRegistry)

    def test_replay_disabled_registry_is_none(self):
        s = MemoryStore(replay_enabled=False)
        assert not s.replay_enabled
        assert s.registry("episodic") is None

    def test_put_registers_when_replay_enabled(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        assert "graph1" in s.registry("episodic")

    def test_put_does_not_register_when_replay_disabled(self):
        s = MemoryStore(replay_enabled=False)
        s.put("episodic", "graph1", _entry("graph1"))
        # registry() is None; the legacy gate check `registry is None` survives.
        assert s.registry("episodic") is None
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
        assert s.simhashes_in_tier("episodic") == {}
        assert s.simhashes_in_tier("semantic") == {"graph1": 0xCAFE}
        assert "graph1" not in s.registry("episodic")
        assert "graph1" in s.registry("semantic")

    def test_move_to_same_tier_is_noop_but_keeps_registry(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        s.move("graph1", "episodic")
        assert s.tier_of("graph1") == "episodic"
        assert "graph1" in s.registry("episodic")


# ---------------------------------------------------------------------------
# Snapshot / restore — cycle-resume rollback rope
# ---------------------------------------------------------------------------


class TestSnapshotRestore:
    def test_snapshot_then_mutate_then_restore(self):
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
        assert s.simhash("episodic", "graph1") == 0xCAFE
        assert s.get("graph2") is None

    def test_snapshot_does_not_share_mutable_state(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        snap = s.snapshot()
        # Mutate snapshot — store must be unaffected.
        snap["entries"]["episodic"]["graph1"]["subject"] = "Eve"
        assert s.get("graph1")["subject"] == "Alice"

    def test_snapshot_excludes_registries(self):
        """Registries persist via their own KeyRegistry.save/load lifecycle."""
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        snap = s.snapshot()
        assert "entries" in snap
        assert "simhash" in snap
        assert "registry" not in snap


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
