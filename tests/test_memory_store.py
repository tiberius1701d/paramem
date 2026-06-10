"""Unit tests for :class:`paramem.memory.store.MemoryStore`.

Locks the per-tier {quads, simhash, registry} contract that replaced the
flat ``indexed_key_cache`` + three flat ``*_simhash`` dicts on
:class:`ConsolidationLoop`.
"""

from __future__ import annotations

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

    def test_bookkeeping_round_trips_speaker_id(self):
        """set_bookkeeping then bookkeeping_for_key returns the correct fields.

        Replaces the deleted setdefault_entry test — bookkeeping is now the
        canonical owner of speaker_id/first_seen_cycle/relation_type."""
        s = MemoryStore()
        s.set_bookkeeping("graph1", speaker_id="spk-a", first_seen_cycle=5, relation_type="factual")
        bk = s.bookkeeping_for_key("graph1")
        assert bk is not None
        assert bk["speaker_id"] == "spk-a"
        assert bk["first_seen_cycle"] == 5
        # Second call must return updated values (idempotent overwrite).
        s.set_bookkeeping(
            "graph1", speaker_id="spk-b", first_seen_cycle=7, relation_type="preference"
        )
        bk2 = s.bookkeeping_for_key("graph1")
        assert bk2["speaker_id"] == "spk-b"
        assert bk2["first_seen_cycle"] == 7


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


# ---------------------------------------------------------------------------
# Bookkeeping — speaker_id / first_seen_cycle owner
# ---------------------------------------------------------------------------


class TestBookkeeping:
    def test_bookkeeping_does_not_create_content_hit(self):
        """REGRESSION LOCK: set_bookkeeping must NOT create a content cache hit.

        Under preload_cache=False this is the key correctness invariant:
        bookkeeping presence must never mask a cache miss."""
        s = MemoryStore()
        s.set_bookkeeping("k", speaker_id="alice", first_seen_cycle=1, relation_type="factual")
        # No put — _entries is empty.
        assert s.get("k") is None
        results = s.probe({"episodic": ["k"]}, source=None)
        assert results["k"] is None

    def test_new_key_write_back_round_trips_speaker_id(self):
        """set_bookkeeping + bookkeeping_for_key round-trips all fields."""
        s = MemoryStore()
        s.set_bookkeeping("graph1", speaker_id="bob", first_seen_cycle=3, relation_type="factual")
        bk = s.bookkeeping_for_key("graph1")
        assert bk is not None
        assert bk["speaker_id"] == "bob"
        assert bk["first_seen_cycle"] == 3

    def test_bookkeeping_absent_returns_none(self):
        s = MemoryStore()
        assert s.bookkeeping_for_key("nonexistent") is None

    def test_iter_bookkeeping_yields_all_keys(self):
        s = MemoryStore()
        s.set_bookkeeping("k1", speaker_id="alice", first_seen_cycle=1, relation_type="factual")
        s.set_bookkeeping("k2", speaker_id="bob", first_seen_cycle=2, relation_type="preference")
        items = dict(s.iter_bookkeeping())
        assert items["k1"]["speaker_id"] == "alice"
        assert items["k2"]["speaker_id"] == "bob"
        assert s.bookkeeping_count() == 2

    def test_delete_also_drops_bookkeeping(self):
        """store.delete must retire bookkeeping automatically."""
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        s.set_bookkeeping("graph1", speaker_id="alice", first_seen_cycle=1, relation_type="factual")
        s.delete("graph1")
        assert s.bookkeeping_for_key("graph1") is None

    def test_probe_hit_render_joins_store_bookkeeping(self):
        """Fake source returns SPO with NO speaker_id.  _bookkeeping is the
        authoritative render-time source (mirrors WeightMemorySource path).
        The store has an entry for graph1 (cache-hit path)."""

        class _FakeSource:
            def probe(self, keys_by_tier):
                # Weight-source style: SPO only, no speaker_id.
                result = {}
                for keys in keys_by_tier.values():
                    for k in keys:
                        result[k] = {
                            "key": k,
                            "subject": "Alice",
                            "predicate": "lives_in",
                            "object": "Berlin",
                        }
                return result

        s = MemoryStore()
        s.put(
            "episodic",
            "graph1",
            {"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"},
        )
        s.set_bookkeeping(
            "graph1", speaker_id="spk-alice", first_seen_cycle=2, relation_type="factual"
        )
        results = s.probe({"episodic": ["graph1"]}, source=_FakeSource())
        assert results["graph1"]["speaker_id"] == "spk-alice"
        assert results["graph1"]["first_seen_cycle"] == 2

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
        s.set_bookkeeping("k1", speaker_id="alice", first_seen_cycle=1, relation_type="factual")
        s.set_bookkeeping("k2", speaker_id="alice", first_seen_cycle=2, relation_type="factual")
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
        s.set_bookkeeping("graph1", speaker_id="alice", first_seen_cycle=1, relation_type="factual")
        snap = s.snapshot()
        assert "_bookkeeping" not in snap
        assert "bookkeeping" not in snap

    # -- Slice 1: relation_type round-trip tests --

    def test_relation_type_round_trips(self):
        """set_bookkeeping with relation_type='preference' → bookkeeping_for_key returns it."""
        s = MemoryStore()
        s.set_bookkeeping(
            "graph1", speaker_id="alice", first_seen_cycle=1, relation_type="preference"
        )
        bk = s.bookkeeping_for_key("graph1")
        assert bk is not None
        assert bk["relation_type"] == "preference"

    def test_relation_type_overwrite_idempotent(self):
        """A second set_bookkeeping call updates relation_type in place."""
        s = MemoryStore()
        s.set_bookkeeping("graph1", speaker_id="alice", first_seen_cycle=1, relation_type="factual")
        s.set_bookkeeping(
            "graph1", speaker_id="alice", first_seen_cycle=1, relation_type="preference"
        )
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
                            "recurrence_count": 1,
                            "speaker_id": "alice",
                            "first_seen_cycle": 2,
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

    def test_load_bookkeeping_from_disk_legacy_without_relation_type(self, tmp_path):
        """Legacy key_metadata.json without relation_type loads with 'unknown' default.

        No crash; legacy_upgraded counter incremented."""
        import json

        path = tmp_path / "key_metadata.json"
        path.write_text(
            json.dumps(
                {
                    "keys": {
                        "graph1": {
                            "recurrence_count": 1,
                            "speaker_id": "alice",
                            "first_seen_cycle": 2,
                            # relation_type deliberately absent (legacy file)
                        }
                    }
                }
            )
        )
        s = MemoryStore()
        reg = KeyRegistry()
        reg.add("graph1")
        s.load_registry("episodic", reg)
        stats = s.load_bookkeeping_from_disk(path)
        assert stats["legacy_upgraded"] == 1
        bk = s.bookkeeping_for_key("graph1")
        assert bk is not None
        assert bk["relation_type"] == "unknown"


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
        """Compute the expected SimHash fingerprint for *entry*."""
        from paramem.memory.entry import compute_simhash

        return compute_simhash(
            entry["key"],
            entry["subject"],
            entry["predicate"],
            entry["object"],
        )

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
# discard_keys helper (§6.2.7 / §3.5)
# ---------------------------------------------------------------------------


class TestDiscardKeys:
    """MemoryStore.discard_keys(keys, mode=) — erase and stale variants.

    Covers: mode="erase" removes from active + drops simhash across tiers
    (registry over tiers_with_registry, simhash over three main tiers only);
    mode="stale" moves to stale + RETAINS simhash; idempotent on absent key;
    no-op when replay disabled (W5 guard).
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
        assert "graph1" not in s.simhashes_in_tier("episodic"), "Erased key simhash must be dropped"

    def test_erase_drops_simhash_on_main_tiers_only(self):
        """mode='erase': simhash drop is limited to the three main tiers (not interim).

        This preserves /forget's tier asymmetry: registry mutation over
        tiers_with_registry() (all tiers), simhash drop over main tiers only.
        """
        s = MemoryStore(replay_enabled=True)
        # Put key in an interim-style tier in the registry only.
        s.put("episodic_interim_20260101", "graph_interim", _entry("graph_interim"), register=True)
        # Also put simhash manually in episodic (main tier) and the interim tier.
        s.put_simhash("episodic", "graph_interim", 0x1111)
        s.put_simhash("episodic_interim_20260101", "graph_interim", 0x2222)

        s.discard_keys(["graph_interim"], mode="erase")

        # episodic simhash dropped (main tier).
        assert "graph_interim" not in s.simhashes_in_tier("episodic"), (
            "Main-tier simhash must be dropped by erase"
        )
        # Interim simhash NOT dropped (discard_keys only touches main tiers).
        assert "graph_interim" in s.simhashes_in_tier("episodic_interim_20260101"), (
            "Interim-tier simhash must NOT be dropped by discard_keys"
        )

    def test_stale_moves_to_stale_and_retains_simhash(self):
        """mode='stale': key moved to stale partition; simhash retained."""
        s = self._make_store_with_key("graph1", "episodic")
        s.discard_keys(["graph1"], mode="stale")

        reg = s.registry("episodic")
        assert "graph1" not in reg.list_active(), "Staled key must not be in active"
        assert "graph1" in reg.list_stale(), "Staled key must be in stale partition"
        assert "graph1" in s.simhashes_in_tier("episodic"), "Staled key simhash must be RETAINED"

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
        """W5: when _registry is None (replay disabled), discard_keys is a no-op."""
        s = MemoryStore(replay_enabled=False)
        s.put_simhash("episodic", "graph1", 0xAAAA)
        # No registry → must not raise and must leave simhash intact.
        s.discard_keys(["graph1"], mode="erase")
        assert "graph1" in s.simhashes_in_tier("episodic"), (
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
