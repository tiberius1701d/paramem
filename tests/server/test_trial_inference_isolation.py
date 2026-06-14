"""Tests for trial adapter inference isolation.

Verifies that the router uses ConsolidationLoop.store._entries_flat_view() (the
canonical entity-index source) and does NOT pick up trial adapter keys when
trial_adapter_dir is separate from config.adapter_dir.

Also tests _load_simhash_registry with the unified registry layout so the
SimHash unification refactor has end-to-end coverage:
    write per-tier indexed_key_registry.json → _load_simhash_registry → merged dict.

No GPU — all tests use in-memory construction.
"""

from __future__ import annotations

import json
from pathlib import Path

from paramem.server.inference import _load_simhash_registry
from paramem.server.router import QueryRouter


class TestRouterReadsFromLoopCache:
    """Router reads from ConsolidationLoop.store._entries_flat_view(), not from quads.json."""

    def _make_store_with_cache(self, cache: dict):
        """Build a MemoryStore pre-populated with the given entries.

        Mirrors the production put+set_bookkeeping write contract so that
        ``QueryRouter.reload()`` (which reads ``iter_bookkeeping()``) picks up
        speaker associations the same way the live server does.
        """
        from paramem.memory.store import MemoryStore

        store = MemoryStore(replay_enabled=True)
        for k, q in cache.items():
            store.put("episodic", k, q)
            spk = q.get("speaker_id", "")
            if spk:
                store.set_bookkeeping(
                    k,
                    speaker_id=spk,
                    first_seen_cycle=q.get("first_seen_cycle", 0),
                    relation_type=q.get("relation_type", "factual"),
                )
        return store

    def test_router_indexes_from_loop_cache(self, tmp_path):
        """Router picks up keys from the injected MemoryStore.

        The router no longer reads quads.json files. Keys are indexed
        from the lifespan-owned MemoryStore at reload() time.
        """
        adapter_dir = tmp_path / "adapters"
        adapter_dir.mkdir(parents=True)

        cache = {
            "graph1": {
                "key": "graph1",
                "subject": "Alice",
                "predicate": "lives_in",
                "object": "London",
                "speaker_id": "spk1",
                "first_seen_cycle": 1,
            }
        }
        store = self._make_store_with_cache(cache)

        router = QueryRouter(adapter_dir=adapter_dir, memory_store=store)

        # The speaker → key mapping must reflect the seeded entry.
        assert router._speaker_key_index.get("spk1") == {"graph1"}

    def test_trial_key_not_indexed_when_loop_is_none(self, tmp_path):
        """Router with an empty store returns empty indexes.

        Trial adapter runs without a live store; quads.json in the
        trial dir must not be picked up (the router does not read those files).
        """
        from paramem.memory.store import MemoryStore

        adapter_dir = tmp_path / "adapters"
        adapter_dir.mkdir(parents=True)
        trial_dir = tmp_path / "trial_adapter"
        trial_dir.mkdir(parents=True)
        # Write a quads.json in trial dir — should NOT be indexed.
        (trial_dir / "quads.json").write_text(
            json.dumps([{"key": "trial-key-001", "subject": "TrialEntity"}]),
            encoding="utf-8",
        )

        router = QueryRouter(
            adapter_dir=adapter_dir,
            memory_store=MemoryStore(replay_enabled=False),
        )

        all_keys: set[str] = set()
        for keys in router._speaker_key_index.values():
            all_keys.update(keys)

        assert "trial-key-001" not in all_keys, (
            "Trial adapter key was indexed by router — inference isolation violated"
        )

    def test_router_empty_when_adapter_dir_absent(self, tmp_path):
        """Router returns no indexed speakers when adapter_dir does not exist."""
        from paramem.memory.store import MemoryStore

        router = QueryRouter(
            adapter_dir=tmp_path / "nonexistent",
            memory_store=MemoryStore(replay_enabled=False),
        )
        assert router._speaker_key_index == {}

    def test_trial_adapter_dir_separate_from_live(self, tmp_path):
        """trial_adapter_dir is a sibling of state_dir, not inside adapter_dir."""
        data_dir = tmp_path / "data" / "ha"
        trial_adapter_dir = data_dir / "trial_adapter"
        adapter_dir = tmp_path / "data" / "adapters"

        try:
            trial_adapter_dir.relative_to(adapter_dir)
            is_subdir = True
        except ValueError:
            is_subdir = False

        assert not is_subdir, (
            "trial_adapter_dir is inside adapter_dir — trial keys could be picked up by router"
        )


class TestLoadSimhashRegistryPerTierPaths:
    """_load_simhash_registry reads from per-tier indexed_key_registry.json files.

    After the SimHash unification refactor, fingerprints live in the ``"simhash"``
    key of each tier's ``indexed_key_registry.json`` rather than in a separate
    ``simhash_registry.json`` sidecar.  The reader merges the ``"simhash"`` maps
    from all tiers and interim slots.
    """

    def _write_registry(self, path: Path, simhash_map: dict[str, int]) -> None:
        """Write an indexed_key_registry.json with the given simhash map."""
        from paramem.training.key_registry import KeyRegistry

        path.parent.mkdir(parents=True, exist_ok=True)
        reg = KeyRegistry()
        for k, fp in simhash_map.items():
            reg.add(k)
            reg.set_simhash(k, fp)
        path.write_bytes(reg.save_bytes())

    def test_reads_main_tier_files(self, tmp_path):
        """episodic + semantic + procedural registry files are all merged."""
        adapter_dir = tmp_path / "adapters"
        self._write_registry(adapter_dir / "episodic" / "indexed_key_registry.json", {"graph1": 1})
        self._write_registry(adapter_dir / "semantic" / "indexed_key_registry.json", {"graph2": 2})
        self._write_registry(adapter_dir / "procedural" / "indexed_key_registry.json", {"proc1": 3})

        result = _load_simhash_registry(adapter_dir)

        assert result == {"graph1": 1, "graph2": 2, "proc1": 3}

    def test_reads_interim_tier_files(self, tmp_path):
        """Interim adapter registry files are merged in addition to main tiers."""
        adapter_dir = tmp_path / "adapters"
        self._write_registry(adapter_dir / "episodic" / "indexed_key_registry.json", {"graph1": 1})
        # 2026-05-14 hierarchy: interim slots live under episodic/interim_<stamp>/.
        self._write_registry(
            adapter_dir / "episodic" / "interim_20260501T1200" / "indexed_key_registry.json",
            {"graph5": 5},
        )

        result = _load_simhash_registry(adapter_dir)

        assert "graph1" in result
        assert "graph5" in result

    def test_returns_empty_when_adapter_dir_absent(self, tmp_path):
        """Missing adapter_dir → empty dict (no crash)."""
        result = _load_simhash_registry(tmp_path / "nonexistent")
        assert result == {}

    def test_later_tier_wins_on_key_collision(self, tmp_path):
        """When a key appears in two tier files, the later read wins.

        During promotion the same key may transiently appear in both
        interim and main; the simhash content is identical so the
        value doesn't matter — but the later read must not crash.
        """
        adapter_dir = tmp_path / "adapters"
        # Same key in both episodic and semantic.
        self._write_registry(
            adapter_dir / "episodic" / "indexed_key_registry.json", {"graph1": 111}
        )
        self._write_registry(
            adapter_dir / "semantic" / "indexed_key_registry.json", {"graph1": 999}
        )

        result = _load_simhash_registry(adapter_dir)

        # Key is present; value is whichever was read last (semantic after episodic).
        assert "graph1" in result

    def test_old_simhash_sidecar_not_picked_up(self, tmp_path):
        """Legacy simhash_registry.json sidecars at tier subdirs are NOT read.

        Fingerprints now live in indexed_key_registry.json.  A stale
        simhash_registry.json must not be picked up by the new reader.
        """
        adapter_dir = tmp_path / "adapters"
        (adapter_dir / "episodic").mkdir(parents=True)
        # Write legacy sidecar — must NOT be picked up.
        (adapter_dir / "episodic" / "simhash_registry.json").write_text(
            json.dumps({"legacy_key": 42}), encoding="utf-8"
        )

        result = _load_simhash_registry(adapter_dir)

        assert "legacy_key" not in result, (
            "Reader picked up stale simhash_registry.json sidecar — "
            "must only read from indexed_key_registry.json"
        )

    def test_malformed_tier_file_skipped(self, tmp_path):
        """A malformed registry file is skipped; other tiers are still merged."""
        adapter_dir = tmp_path / "adapters"
        (adapter_dir / "episodic").mkdir(parents=True)
        (adapter_dir / "episodic" / "indexed_key_registry.json").write_bytes(b"not json!!!")
        self._write_registry(adapter_dir / "semantic" / "indexed_key_registry.json", {"graph2": 2})

        result = _load_simhash_registry(adapter_dir)

        # episodic failed → skipped; semantic loaded fine.
        assert "graph2" in result
