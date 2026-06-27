"""Regression: an interim adapter's keys must never be permanently deleted by
orphan-pruning when the interim slot is only *transiently* absent from disk.

Root-cause reproduction for the 2026-06-02 data-loss incident:

* boot 10:57 mounted ``episodic_interim_20260601T1200`` (27 keys, live).
* A mid-session in-process reload (migration / swap / release) deleted the
  interim slot directory WITHOUT first folding its keys into a persisted
  main-tier slot.
* ``prune_key_metadata_orphans`` then ran on the reload and treated the now
  registry-less keys as permanent orphans — deleting all 27 from
  ``key_metadata.json``.  The affected keys' facts were never in a durable
  main-tier adapter, so they were lost for good.

These tests pin the two halves of the bug and the fix:

1. ``prune_key_metadata_orphans`` must NOT delete metadata for keys whose owning
   interim slot is merely off-disk at prune time (transient absence on a reload).
2. The boot-time torn-write cleanup must not ``rmtree`` an interim slot whose
   registry still carries active keys that are present in ``key_metadata.json``
   (i.e. keys that were never folded into a main tier) — that is the silent
   delete path that armed the orphan-prune.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from paramem.server.consolidation import prune_key_metadata_orphans
from paramem.training.key_registry import KeyRegistry

INTERIM_KEYS = [f"graph{i}" for i in range(27)]


def _write_registry(reg_path: Path, keys: list[str]) -> None:
    reg = KeyRegistry()
    for k in keys:
        reg.add(k)
    reg.save(reg_path)


def _write_key_metadata(path: Path, keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cycle_count": 1,
        "promoted_keys": [],
        "keys": {k: {"reinforcement_count": 1, "speaker_id": "Speaker0"} for k in keys},
    }
    from paramem.backup.encryption import write_infra_bytes

    write_infra_bytes(path, json.dumps(payload, indent=2).encode("utf-8"))


def _read_key_metadata(path: Path) -> dict:
    from paramem.backup.encryption import read_maybe_encrypted

    return json.loads(read_maybe_encrypted(path).decode("utf-8"))


def _make_config(adapter_dir: Path, key_metadata_path: Path) -> MagicMock:
    cfg = MagicMock()
    cfg.adapter_dir = adapter_dir
    cfg.key_metadata_path = key_metadata_path
    return cfg


class TestInterimOrphanPruneDataLoss:
    def test_prune_keeps_interim_backed_keys_on_disk(self, tmp_path):
        """Sanity baseline: prune spares interim keys when the slot IS on disk."""
        adapter_dir = tmp_path / "adapters"
        # Main episodic slot present but empty; interim slot carries the 27 keys.
        (adapter_dir / "episodic").mkdir(parents=True, exist_ok=True)
        _write_registry(adapter_dir / "episodic" / "indexed_key_registry.json", [])
        interim_dir = adapter_dir / "episodic" / "interim_20260601T1200"
        interim_dir.mkdir(parents=True, exist_ok=True)
        _write_registry(interim_dir / "indexed_key_registry.json", INTERIM_KEYS)

        km_path = tmp_path / "registry" / "key_metadata.json"
        _write_key_metadata(km_path, INTERIM_KEYS)

        cfg = _make_config(adapter_dir, km_path)
        removed = prune_key_metadata_orphans(cfg)

        assert removed == 0
        kept = _read_key_metadata(km_path)["keys"]
        assert set(kept) == set(INTERIM_KEYS)

    def test_prune_must_not_delete_keys_when_interim_transiently_absent(self, tmp_path):
        """The data-loss bug: interim slot vanished from disk this reload, but its
        keys are still referenced by key_metadata.json and were never folded into a
        main tier.  prune_key_metadata_orphans must NOT permanently delete them.

        Pre-fix, ``active`` (built only from on-disk registries) does not contain
        the interim keys, so all 27 are pruned — the exact ``removed 27 orphan
        key(s)`` line from the incident journal.
        """
        adapter_dir = tmp_path / "adapters"
        # Main episodic slot exists with a registry, but the interim DIR IS GONE
        # (deleted earlier this session by a migration/release reload).
        (adapter_dir / "episodic").mkdir(parents=True, exist_ok=True)
        _write_registry(adapter_dir / "episodic" / "indexed_key_registry.json", [])
        # NOTE: no interim_* dir on disk — simulates the transient-absence reload.

        km_path = tmp_path / "registry" / "key_metadata.json"
        _write_key_metadata(km_path, INTERIM_KEYS)

        cfg = _make_config(adapter_dir, km_path)
        removed = prune_key_metadata_orphans(cfg)

        # The fix: keys not present in any tier registry but still carried in
        # key_metadata.json are NOT permanently dropped — a missing interim is a
        # transient mount condition, not proof of permanent orphanhood.
        assert removed == 0, "interim-backed keys were permanently orphan-pruned"
        kept = _read_key_metadata(km_path)["keys"]
        assert set(INTERIM_KEYS).issubset(set(kept)), "interim keys were lost by prune"

    def test_prune_still_drops_genuine_orphans(self, tmp_path):
        """The fix must remain destructive for genuine orphans: a key that is in
        NEITHER any tier registry NOR was ever an interim/promoted key is stale
        bookkeeping and should still be pruned.
        """
        adapter_dir = tmp_path / "adapters"
        (adapter_dir / "episodic").mkdir(parents=True, exist_ok=True)
        # Episodic registry holds key "live0"; key_metadata also carries "stale0"
        # which exists in no registry and is not interim/promoted → genuine orphan.
        _write_registry(adapter_dir / "episodic" / "indexed_key_registry.json", ["live0"])

        km_path = tmp_path / "registry" / "key_metadata.json"
        path = km_path
        path.parent.mkdir(parents=True, exist_ok=True)
        from paramem.backup.encryption import write_infra_bytes

        payload = {
            "cycle_count": 1,
            "promoted_keys": [],
            "keys": {
                "live0": {"reinforcement_count": 1, "speaker_id": "Speaker0"},
                "stale0": {"reinforcement_count": 1, "speaker_id": "Speaker0"},
            },
        }
        write_infra_bytes(path, json.dumps(payload, indent=2).encode("utf-8"))

        cfg = _make_config(adapter_dir, km_path)
        removed = prune_key_metadata_orphans(cfg)

        assert removed == 1
        kept = _read_key_metadata(km_path)["keys"]
        assert set(kept) == {"live0"}


class TestSoftStaleBookkeepingRetention:
    """Soft-staled keys' bookkeeping must survive prune_key_metadata_orphans
    and be persisted by _save_key_metadata.

    A stale key is absent from list_active() but present in list_stale().  Before
    the soft-stale fix, the retention union only included active keys, so a
    soft-staled key's bookkeeping was silently pruned after the next consolidation
    cycle — breaking the stale-echo seam (no speaker/relation_type to resolve).
    """

    def test_prune_does_not_delete_stale_key_metadata(self, tmp_path):
        """prune_key_metadata_orphans retains bookkeeping for a soft-staled key.

        Setup: episodic registry has one active key ("active1") and one stale key
        ("stale1").  key_metadata.json carries both.  After prune, both survive.
        """
        adapter_dir = tmp_path / "adapters"
        (adapter_dir / "episodic").mkdir(parents=True, exist_ok=True)

        # Build a registry with active1 active and stale1 stale.
        reg = KeyRegistry()
        reg.add("active1")
        reg.add("stale1")
        reg.stale("stale1")
        reg.save(adapter_dir / "episodic" / "indexed_key_registry.json")

        km_path = tmp_path / "registry" / "key_metadata.json"
        km_path.parent.mkdir(parents=True, exist_ok=True)
        from paramem.backup.encryption import write_infra_bytes

        payload = {
            "cycle_count": 1,
            "promoted_keys": [],
            "keys": {
                "active1": {
                    "reinforcement_count": 1,
                    "speaker_id": "Speaker0",
                    "relation_type": "factual",
                    "last_reinforced_cycle": 1,
                    "last_seen": "",
                },
                "stale1": {
                    "reinforcement_count": 1,
                    "speaker_id": "Speaker0",
                    "relation_type": "factual",
                    "last_reinforced_cycle": 1,
                    "last_seen": "",
                },
            },
        }
        write_infra_bytes(km_path, json.dumps(payload, indent=2).encode("utf-8"))

        cfg = _make_config(adapter_dir, km_path)
        removed = prune_key_metadata_orphans(cfg)

        assert removed == 0, (
            f"prune removed {removed} key(s) but expected 0 — stale1 must be retained"
        )
        kept = _read_key_metadata(km_path)["keys"]
        assert "active1" in kept, "active1 must be retained by prune"
        assert "stale1" in kept, "stale1 (soft-staled) must NOT be pruned"

    def test_save_key_metadata_persists_stale_key_bookkeeping(self, tmp_path):
        """_save_key_metadata persists bookkeeping for both active and stale keys.

        Builds a MemoryStore with one active key and one stale key, sets bookkeeping
        for both, then calls _save_key_metadata and verifies the output file carries
        both.
        """
        from unittest.mock import MagicMock

        from paramem.memory.store import MemoryStore
        from paramem.server.consolidation import _save_key_metadata

        store = MemoryStore(replay_enabled=True)
        store.put(
            "episodic",
            "active1",
            {"subject": "Alice", "predicate": "lives_in", "object": "Berlin"},
            register=True,
        )
        store.put(
            "episodic",
            "stale1",
            {"subject": "Bob", "predicate": "works_at", "object": "Acme"},
            register=True,
        )
        store.set_bookkeeping("active1", speaker_id="Speaker0", relation_type="factual")
        store.set_bookkeeping("stale1", speaker_id="Speaker0", relation_type="preference")
        # Soft-stale stale1 via the registry.
        store.discard_keys(["stale1"], mode="stale")

        km_path = tmp_path / "registry" / "key_metadata.json"
        km_path.parent.mkdir(parents=True, exist_ok=True)

        # Build a minimal ConsolidationLoop mock.
        loop = MagicMock()
        loop.store = store
        loop.cycle_count = 1
        loop.promoted_keys = set()
        loop.trial_key_metadata_path = None

        cfg = MagicMock()
        cfg.key_metadata_path = km_path

        _save_key_metadata(loop, cfg)

        saved = _read_key_metadata(km_path)
        keys = saved["keys"]
        assert "active1" in keys, "active1 must be persisted by _save_key_metadata"
        assert "stale1" in keys, "stale1 (soft-staled) must be persisted by _save_key_metadata"
        assert keys["stale1"]["relation_type"] == "preference", (
            "stale1 bookkeeping must include the original relation_type"
        )
