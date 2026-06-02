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
2. The boot-time I5 torn-write cleanup must not ``rmtree`` an interim slot whose
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
        "keys": {k: {"sessions_seen": 1, "speaker_id": "Speaker0"} for k in keys},
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
                "live0": {"sessions_seen": 1, "speaker_id": "Speaker0"},
                "stale0": {"sessions_seen": 1, "speaker_id": "Speaker0"},
            },
        }
        write_infra_bytes(path, json.dumps(payload, indent=2).encode("utf-8"))

        cfg = _make_config(adapter_dir, km_path)
        removed = prune_key_metadata_orphans(cfg)

        assert removed == 1
        kept = _read_key_metadata(km_path)["keys"]
        assert set(kept) == {"live0"}
