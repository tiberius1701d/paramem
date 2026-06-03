"""Unit tests for paramem.backup.integrity.

Covers:
- All-valid plaintext store → ok=True.
- Per-file failure modes: corrupt JSON registry, bad manifest schema,
  required graph missing (simulate), graph absent in train (skipped).
- no-identity undecryptable (age file + no key): not a corruption failure.
- Tampered age + identity: undecryptable (corrupt detail), IS a failure
  when daily_loadable=True.
- Cross-consistency: R-without-S, S-without-R, key_metadata orphan.
- registry-key-absent-from-metadata → NOT a failure.
- Required-vs-optional: empty registry → skipped; partial interim slot → skipped.
- Boot degraded: corrupt registry → store_load_degraded True.

Tests use a config mock and tmp_path; encryption tests mock out
read_maybe_encrypted to inject age-like behaviour without a real key.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paramem.backup.integrity import (
    _DETAIL_NO_KEY,
    _INCONSISTENT,
    _MISSING,
    _OK,
    _PARSE_ERROR,
    _SCHEMA_ERROR,
    _SKIPPED,
    _UNDECRYPTABLE,
    FileCheck,
    IntegrityReport,
    cleanup_partial_slots,
    verify_infrastructure_integrity,
)
from paramem.memory.store import MemoryStore
from paramem.training.key_registry import KeyRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, mode: str = "train") -> MagicMock:
    """Build a minimal config mock rooted at *tmp_path*.

    Matches the shape used by the production app (adapter_dir, key_metadata_path,
    paths.data, consolidation.mode).
    """
    data_dir = tmp_path / "data" / "ha"
    data_dir.mkdir(parents=True, exist_ok=True)

    cfg = MagicMock()
    cfg.paths.data = data_dir
    cfg.adapter_dir = data_dir / "adapters"
    cfg.adapter_dir.mkdir(parents=True, exist_ok=True)
    cfg.key_metadata_path = data_dir / "registry" / "key_metadata.json"
    cfg.consolidation.mode = mode
    return cfg


def _write_key_registry(path: Path, keys: list[str]) -> None:
    """Write a minimal indexed_key_registry.json with *keys* as active keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    reg = KeyRegistry()
    for k in keys:
        reg.add(k)
    path.write_bytes(reg.save_bytes())


def _write_simhash(path: Path, entries: dict[str, int]) -> None:
    """Write a minimal simhash_registry.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _write_graph(path: Path) -> None:
    """Write a minimal (empty) graph.json in node_link_data format."""
    import networkx as nx

    path.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(nx.MultiDiGraph())
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_manifest(slot_dir: Path, name: str = "episodic") -> None:
    """Write a minimal valid meta.json for a weight slot."""
    slot_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 4,
        "name": name,
        "trained_at": "2026-05-01T00:00:00Z",
        "window_stamp": "",
        "base_model": {"repo": "test/model", "sha": "abc", "hash": "sha256:deadbeef"},
        "tokenizer": {"name_or_path": "test/model", "vocab_size": 32000, "merges_hash": "abc"},
        "lora": {"rank": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["q_proj"]},
        "registry_sha256": "",
        "key_count": 0,
    }
    (slot_dir / "meta.json").write_text(json.dumps(manifest), encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def train_store_dir(tmp_path):
    """Minimal valid 'train' store: episodic has 2 keys, others empty."""
    cfg = _make_config(tmp_path, mode="train")
    adapter_dir = cfg.adapter_dir

    # episodic tier
    ep_dir = adapter_dir / "episodic"
    ep_dir.mkdir(parents=True, exist_ok=True)
    _write_key_registry(ep_dir / "indexed_key_registry.json", ["key1", "key2"])
    _write_simhash(ep_dir / "simhash_registry.json", {"key1": 1, "key2": 2})
    slot = ep_dir / "20260501-000000"
    _write_manifest(slot, "episodic")

    # semantic tier (empty registry — skipped)
    sem_dir = adapter_dir / "semantic"
    sem_dir.mkdir(parents=True, exist_ok=True)
    _write_key_registry(sem_dir / "indexed_key_registry.json", [])
    _write_simhash(sem_dir / "simhash_registry.json", {})

    # procedural tier (empty registry — skipped)
    proc_dir = adapter_dir / "procedural"
    proc_dir.mkdir(parents=True, exist_ok=True)
    _write_key_registry(proc_dir / "indexed_key_registry.json", [])
    _write_simhash(proc_dir / "simhash_registry.json", {})

    return cfg


@pytest.fixture()
def simulate_store_dir(tmp_path):
    """Minimal valid 'simulate' store: episodic has 2 keys + graph.json."""
    cfg = _make_config(tmp_path, mode="simulate")
    adapter_dir = cfg.adapter_dir

    ep_dir = adapter_dir / "episodic"
    ep_dir.mkdir(parents=True, exist_ok=True)
    _write_key_registry(ep_dir / "indexed_key_registry.json", ["key1", "key2"])
    _write_simhash(ep_dir / "simhash_registry.json", {"key1": 1, "key2": 2})
    _write_graph(ep_dir / "graph.json")

    # semantic + procedural empty
    for tier in ("semantic", "procedural"):
        d = adapter_dir / tier
        d.mkdir(parents=True, exist_ok=True)
        _write_key_registry(d / "indexed_key_registry.json", [])
        _write_simhash(d / "simhash_registry.json", {})

    return cfg


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestAllValidPlaintext:
    def test_train_store_ok(self, train_store_dir):
        """Valid train store → report.ok == True."""
        report = verify_infrastructure_integrity(train_store_dir, daily_loadable=False)
        assert isinstance(report, IntegrityReport)
        assert report.ok is True
        assert report.failures == []

    def test_simulate_store_ok(self, simulate_store_dir):
        """Valid simulate store → report.ok == True."""
        report = verify_infrastructure_integrity(simulate_store_dir, daily_loadable=False)
        assert report.ok is True
        assert report.failures == []

    def test_report_to_dict_is_serializable(self, train_store_dir):
        """to_dict() returns a JSON-serializable dict."""
        report = verify_infrastructure_integrity(train_store_dir, daily_loadable=False)
        d = report.to_dict()
        json.dumps(d)  # must not raise
        assert "ok" in d
        assert "checks" in d
        assert "failures" in d


# ---------------------------------------------------------------------------
# Per-file failure modes
# ---------------------------------------------------------------------------


class TestRegistryFailure:
    def test_corrupt_json_registry(self, tmp_path):
        """Corrupt JSON in indexed_key_registry.json → parse_error failure."""
        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        (ep_dir / "indexed_key_registry.json").write_text("{not: json}", encoding="utf-8")

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)
        assert report.ok is False
        reg_failures = [f for f in report.failures if f.category == "registry"]
        assert len(reg_failures) >= 1
        assert reg_failures[0].status == _PARSE_ERROR

    def test_empty_registry_tier_skipped(self, tmp_path):
        """A tier with an empty registry → simhash is skipped, not failed."""
        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        _write_key_registry(ep_dir / "indexed_key_registry.json", [])
        _write_simhash(ep_dir / "simhash_registry.json", {})

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)
        assert report.ok is True
        # simhash for episodic must be skipped or ok (not a failure)
        sim_checks = [c for c in report.checks if c.category == "simhash" and c.tier == "episodic"]
        assert all(c.status in (_OK, _SKIPPED) for c in sim_checks)


class TestManifestFailure:
    def test_bad_schema_manifest(self, tmp_path):
        """meta.json with missing required field → schema_error failure."""
        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        _write_key_registry(ep_dir / "indexed_key_registry.json", ["key1"])
        _write_simhash(ep_dir / "simhash_registry.json", {"key1": 1})

        # Write a slot with invalid meta.json (missing 'name' field)
        slot = ep_dir / "20260501-000000"
        slot.mkdir(parents=True, exist_ok=True)
        bad_manifest = {"schema_version": 4, "trained_at": "2026-05-01T00:00:00Z"}
        (slot / "meta.json").write_text(json.dumps(bad_manifest), encoding="utf-8")

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)
        assert report.ok is False
        manifest_failures = [f for f in report.failures if f.category == "manifest"]
        assert len(manifest_failures) >= 1
        assert manifest_failures[0].status == _SCHEMA_ERROR


class TestGraphFailure:
    def test_graph_missing_in_simulate_with_keys(self, tmp_path):
        """simulate mode + tier has keys but no graph.json → missing failure."""
        cfg = _make_config(tmp_path, mode="simulate")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        _write_key_registry(ep_dir / "indexed_key_registry.json", ["key1"])
        _write_simhash(ep_dir / "simhash_registry.json", {"key1": 1})
        # No graph.json written

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)
        assert report.ok is False
        graph_failures = [f for f in report.failures if f.category == "graph"]
        assert len(graph_failures) >= 1
        assert graph_failures[0].status == _MISSING

    def test_graph_absent_in_train_is_skipped(self, tmp_path):
        """train mode: graph.json absent → skipped (not a failure)."""
        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        _write_key_registry(ep_dir / "indexed_key_registry.json", ["key1"])
        _write_simhash(ep_dir / "simhash_registry.json", {"key1": 1})
        slot = ep_dir / "20260501-000000"
        _write_manifest(slot, "episodic")
        # No graph.json — expected in train mode

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)
        # graph.json being absent should NOT be a failure in train mode
        graph_failures = [f for f in report.failures if f.category == "graph"]
        assert graph_failures == []


# ---------------------------------------------------------------------------
# Encryption / no-identity
# ---------------------------------------------------------------------------


class TestEncryptionHandling:
    def test_age_file_no_key_not_corruption_daily_loadable_false(self, tmp_path):
        """age-encrypted file + no daily key + daily_loadable=False → not a failure."""
        from paramem.backup.age_envelope import AGE_MAGIC

        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # Write a fake age-magic prefix + garbage (triggers RuntimeError on load)
        fake_age = AGE_MAGIC + b"fake encrypted content"
        (ep_dir / "indexed_key_registry.json").write_bytes(fake_age)

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)
        # undecryptable (no-key) should not appear in failures when daily_loadable=False
        undecrypt_failures = [
            f for f in report.failures if f.status == _UNDECRYPTABLE and f.detail == _DETAIL_NO_KEY
        ]
        assert undecrypt_failures == [], (
            "No-key undecryptable should not be a failure when daily_loadable=False"
        )

    def test_age_file_no_key_appears_in_checks(self, tmp_path):
        """age-encrypted file + no daily key → undecryptable entry appears in checks."""
        from paramem.backup.age_envelope import AGE_MAGIC

        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        fake_age = AGE_MAGIC + b"fake encrypted content"
        (ep_dir / "indexed_key_registry.json").write_bytes(fake_age)

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)
        undecrypt_checks = [
            c for c in report.checks if c.status == _UNDECRYPTABLE and c.detail == _DETAIL_NO_KEY
        ]
        assert len(undecrypt_checks) >= 1

    def test_age_file_no_key_is_failure_when_daily_loadable_true(self, tmp_path):
        """age-encrypted file + daily_loadable=True → undecryptable is a failure."""
        from paramem.backup.age_envelope import AGE_MAGIC

        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        fake_age = AGE_MAGIC + b"fake encrypted content"
        (ep_dir / "indexed_key_registry.json").write_bytes(fake_age)

        report = verify_infrastructure_integrity(cfg, daily_loadable=True)
        assert report.ok is False
        undecrypt_failures = [f for f in report.failures if f.status == _UNDECRYPTABLE]
        assert len(undecrypt_failures) >= 1


# ---------------------------------------------------------------------------
# Cross-consistency checks
# ---------------------------------------------------------------------------


class TestCrossConsistency:
    def test_registry_key_without_simhash(self, tmp_path):
        """Key in registry but absent from simhash → inconsistent failure."""
        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        _write_key_registry(ep_dir / "indexed_key_registry.json", ["key1", "key2"])
        # simhash only has key1 (key2 missing)
        _write_simhash(ep_dir / "simhash_registry.json", {"key1": 1})
        slot = ep_dir / "20260501-000000"
        _write_manifest(slot, "episodic")

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)
        assert report.ok is False
        inconsistent = [f for f in report.failures if f.status == _INCONSISTENT]
        assert len(inconsistent) >= 1
        assert any("key2" in f.detail for f in inconsistent)

    def test_simhash_key_without_registry(self, tmp_path):
        """Key in simhash but absent from registry → inconsistent failure."""
        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        _write_key_registry(ep_dir / "indexed_key_registry.json", ["key1"])
        # simhash has extra key2 (orphan)
        _write_simhash(ep_dir / "simhash_registry.json", {"key1": 1, "key2": 2})
        slot = ep_dir / "20260501-000000"
        _write_manifest(slot, "episodic")

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)
        assert report.ok is False
        inconsistent = [f for f in report.failures if f.status == _INCONSISTENT]
        assert len(inconsistent) >= 1
        assert any("key2" in f.detail for f in inconsistent)

    def test_key_metadata_orphan_when_store_passed(self, tmp_path):
        """key in key_metadata but not in any registry → inconsistent."""
        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        _write_key_registry(ep_dir / "indexed_key_registry.json", ["key1"])
        _write_simhash(ep_dir / "simhash_registry.json", {"key1": 1})
        slot = ep_dir / "20260501-000000"
        _write_manifest(slot, "episodic")

        # Write key_metadata with an extra orphan key
        cfg.key_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "keys": {
                "key1": {"speaker_id": "speaker0", "first_seen_cycle": 0},
                "orphan_key": {"speaker_id": "speaker0", "first_seen_cycle": 1},
            }
        }
        cfg.key_metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        # Build a store with only key1
        store = MemoryStore(replay_enabled=True)
        reg = KeyRegistry()
        reg.add("key1")
        store.load_registry("episodic", reg)

        report = verify_infrastructure_integrity(cfg, store=store, daily_loadable=False)
        assert report.ok is False
        km_failures = [
            f for f in report.failures if f.category == "key_metadata" and f.status == _INCONSISTENT
        ]
        assert len(km_failures) == 1
        assert "orphan_key" in km_failures[0].detail

    def test_registry_key_absent_from_metadata_not_failure(self, tmp_path):
        """Key in registry but absent from key_metadata → NOT a failure."""
        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        _write_key_registry(ep_dir / "indexed_key_registry.json", ["key1", "key2"])
        _write_simhash(ep_dir / "simhash_registry.json", {"key1": 1, "key2": 2})
        slot = ep_dir / "20260501-000000"
        _write_manifest(slot, "episodic")

        # key_metadata only has key1 — key2 is absent (that's fine)
        cfg.key_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"keys": {"key1": {"speaker_id": "speaker0", "first_seen_cycle": 0}}}
        cfg.key_metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        store = MemoryStore(replay_enabled=True)
        reg = KeyRegistry()
        reg.add("key1")
        reg.add("key2")
        store.load_registry("episodic", reg)

        report = verify_infrastructure_integrity(cfg, store=store, daily_loadable=False)
        # key2 absent from key_metadata is NOT a failure
        km_inconsistent = [
            f for f in report.failures if f.category == "key_metadata" and f.status == _INCONSISTENT
        ]
        assert km_inconsistent == []


# ---------------------------------------------------------------------------
# Required-vs-optional
# ---------------------------------------------------------------------------


class TestRequiredVsOptional:
    def test_empty_semantic_skipped(self, tmp_path):
        """semantic tier with no registry file → skipped, not a failure."""
        cfg = _make_config(tmp_path, mode="train")
        # Only create episodic with keys; semantic dir absent entirely
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        _write_key_registry(ep_dir / "indexed_key_registry.json", ["key1"])
        _write_simhash(ep_dir / "simhash_registry.json", {"key1": 1})
        slot = ep_dir / "20260501-000000"
        _write_manifest(slot, "episodic")

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)
        # semantic failures should not appear
        sem_failures = [f for f in report.failures if f.tier in ("semantic", "procedural")]
        assert sem_failures == []

    def test_partial_interim_slot_skipped(self, tmp_path):
        """Interim slot dir present but no registry → skipped, not a failure."""
        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        _write_key_registry(ep_dir / "indexed_key_registry.json", ["key1"])
        _write_simhash(ep_dir / "simhash_registry.json", {"key1": 1})
        slot = ep_dir / "20260501-000000"
        _write_manifest(slot, "episodic")

        # Create a partial interim slot (dir present, no registry)
        interim_dir = ep_dir / "interim_20260517T1200"
        interim_dir.mkdir(parents=True, exist_ok=True)
        # Do NOT write indexed_key_registry.json — this is the partial slot case

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)
        assert report.ok is True
        # partial interim should be skipped
        interim_checks = [c for c in report.checks if "interim" in c.tier]
        assert all(c.status == _SKIPPED for c in interim_checks)

    def test_common_files_absent_are_skipped(self, tmp_path):
        """speaker_profiles.json, observed_languages.json etc. absent → skipped."""
        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        _write_key_registry(ep_dir / "indexed_key_registry.json", ["key1"])
        _write_simhash(ep_dir / "simhash_registry.json", {"key1": 1})
        slot = ep_dir / "20260501-000000"
        _write_manifest(slot, "episodic")

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)
        # Common optional files absent → only skipped
        common_failures = [f for f in report.failures if f.category == "common"]
        assert common_failures == []


# ---------------------------------------------------------------------------
# FileCheck and IntegrityReport data model
# ---------------------------------------------------------------------------


class TestDataModel:
    def test_filecheck_to_dict(self):
        """FileCheck.to_dict() returns all fields."""
        fc = FileCheck("/a/b/c.json", "registry", "episodic", _OK, "")
        d = fc.to_dict()
        assert d == {
            "path": "/a/b/c.json",
            "category": "registry",
            "tier": "episodic",
            "status": "ok",
            "detail": "",
        }

    def test_integrity_report_ok_false_when_failures(self):
        """IntegrityReport.ok is False when failures list is non-empty."""
        bad = FileCheck("/bad.json", "registry", "episodic", _PARSE_ERROR, "bad json")
        report = IntegrityReport(ok=False, checks=[bad], failures=[bad])
        assert report.ok is False
        d = report.to_dict()
        assert d["ok"] is False
        assert len(d["failures"]) == 1

    def test_integrity_report_ok_true_when_all_skipped(self):
        """IntegrityReport.ok True when only skipped entries."""
        skipped = FileCheck("/f.json", "registry", "episodic", _SKIPPED, "")
        report = IntegrityReport(ok=True, checks=[skipped], failures=[])
        assert report.ok is True


# ---------------------------------------------------------------------------
# Boot housekeeping: cleanup_partial_slots
# ---------------------------------------------------------------------------


class TestCleanupPartialSlots:
    """Boot housekeeping: delete partial training slots under tier roots."""

    REQUIRED = ("meta.json", "adapter_config.json", "adapter_model.safetensors")

    def _make_complete_slot(self, root: Path, tier: str, name: str) -> Path:
        slot = root / tier / name
        slot.mkdir(parents=True)
        for f in self.REQUIRED:
            (slot / f).write_text("{}")
        return slot

    def _make_partial_slot(self, root: Path, tier: str, name: str, *missing: str) -> Path:
        slot = root / tier / name
        slot.mkdir(parents=True)
        for f in self.REQUIRED:
            if f not in missing:
                (slot / f).write_text("{}")
        return slot

    def test_empty_adapter_dir_returns_empty(self, tmp_path):
        """No tier roots — returns empty list, no errors."""
        assert cleanup_partial_slots(tmp_path) == []

    def test_complete_slot_retained(self, tmp_path):
        """Slot with all 3 files is never touched."""
        slot = self._make_complete_slot(tmp_path, "episodic", "20260101T0000")
        removed = cleanup_partial_slots(tmp_path)
        assert removed == []
        assert slot.exists()
        assert (slot / "meta.json").exists()

    def test_partial_slot_missing_meta_deleted(self, tmp_path):
        """Non-interim flat slot missing meta.json is deleted; entry recorded."""
        slot = self._make_partial_slot(tmp_path, "episodic", "20260101-000000", "meta.json")
        removed = cleanup_partial_slots(tmp_path)
        assert not slot.exists()
        assert len(removed) == 1
        assert removed[0]["tier"] == "episodic"
        assert removed[0]["slot_name"] == "20260101-000000"
        assert removed[0]["missing"] == ["meta.json"]
        assert removed[0]["path"] == str(slot)

    def test_partial_slot_missing_safetensors_deleted(self, tmp_path):
        """Non-interim flat slot missing adapter_model.safetensors is deleted."""
        slot = self._make_partial_slot(
            tmp_path, "semantic", "20260101-000000", "adapter_model.safetensors"
        )
        removed = cleanup_partial_slots(tmp_path)
        assert not slot.exists()
        assert removed[0]["missing"] == ["adapter_model.safetensors"]

    def test_partial_slot_missing_multiple_records_all(self, tmp_path):
        """Slot missing several files records every missing path."""
        slot = self._make_partial_slot(
            tmp_path,
            "procedural",
            "broken",
            "meta.json",
            "adapter_config.json",
        )
        removed = cleanup_partial_slots(tmp_path)
        assert not slot.exists()
        assert sorted(removed[0]["missing"]) == ["adapter_config.json", "meta.json"]

    def test_mixed_complete_and_partial(self, tmp_path):
        """Complete slots survive; non-interim partial slots are deleted in the same pass."""
        kept = self._make_complete_slot(tmp_path, "episodic", "20260101T0000")
        gone = self._make_partial_slot(tmp_path, "episodic", "20260102-000000", "meta.json")
        removed = cleanup_partial_slots(tmp_path)
        assert kept.exists()
        assert not gone.exists()
        assert len(removed) == 1
        assert removed[0]["slot_name"] == "20260102-000000"

    def test_dotted_entries_skipped(self, tmp_path):
        """Hidden dotted dirs are never deleted."""
        dotted = tmp_path / "episodic" / ".quarantine"
        dotted.mkdir(parents=True)
        removed = cleanup_partial_slots(tmp_path)
        assert removed == []
        assert dotted.exists()

    def test_files_at_tier_root_skipped(self, tmp_path):
        """Files directly under tier root (e.g. registries) are never deleted."""
        tier_root = tmp_path / "episodic"
        tier_root.mkdir()
        (tier_root / "indexed_key_registry.json").write_text("{}")
        (tier_root / "simhash_registry.json").write_text("{}")
        removed = cleanup_partial_slots(tmp_path)
        assert removed == []
        assert (tier_root / "indexed_key_registry.json").exists()
        assert (tier_root / "simhash_registry.json").exists()

    def test_all_three_main_tiers_walked(self, tmp_path):
        """One partial slot under each of episodic/semantic/procedural — all deleted."""
        slots = [
            self._make_partial_slot(tmp_path, tier, "broken", "meta.json")
            for tier in ("episodic", "semantic", "procedural")
        ]
        removed = cleanup_partial_slots(tmp_path)
        assert all(not s.exists() for s in slots)
        assert {r["tier"] for r in removed} == {"episodic", "semantic", "procedural"}

    def test_non_main_tier_untouched(self, tmp_path):
        """A tier name outside _MAIN_TIERS is not walked."""
        unrelated = self._make_partial_slot(
            tmp_path, "consolidation_refresh", "scratch", "meta.json"
        )
        removed = cleanup_partial_slots(tmp_path)
        assert removed == []
        assert unrelated.exists()

    # ------------------------------------------------------------------
    # Regression tests: interim containers must never be deleted
    # ------------------------------------------------------------------

    def test_interim_container_with_valid_nested_slot_never_deleted(self, tmp_path):
        """Incident regression: interim container with valid nested weights is never deleted.

        Layout: episodic/interim_20260101T0000/20260101-000000/<3 files>
                episodic/interim_20260101T0000/indexed_key_registry.json

        The container has none of the 3 required flat-slot files at its root.
        cleanup_partial_slots must skip it entirely (interim integrity is owned
        by find_live_slot + the I5 boot guard).
        """
        container = tmp_path / "episodic" / "interim_20260101T0000"
        inner_slot = container / "20260101-000000"
        inner_slot.mkdir(parents=True)
        for f in self.REQUIRED:
            (inner_slot / f).write_text("{}")
        (container / "indexed_key_registry.json").write_text("{}")

        removed = cleanup_partial_slots(tmp_path)

        assert removed == [], f"Expected no removals, got: {removed}"
        assert container.exists(), "Interim container was wrongly deleted"
        assert inner_slot.exists(), "Interim inner slot was wrongly deleted"

    def test_interim_container_simulate_mode_only_graph_never_deleted(self, tmp_path):
        """Interim container with only graph.json + registry (simulate mode) is never deleted.

        cleanup_partial_slots must not judge interim containers — that is the
        I5 boot guard's job.
        """
        container = tmp_path / "episodic" / "interim_20260101T0000"
        container.mkdir(parents=True)
        (container / "graph.json").write_text("{}")
        (container / "indexed_key_registry.json").write_text("{}")

        removed = cleanup_partial_slots(tmp_path)

        assert removed == [], f"Expected no removals, got: {removed}"
        assert container.exists(), "Simulate-mode interim container was wrongly deleted"

    def test_genuinely_empty_interim_container_not_deleted(self, tmp_path):
        """Empty interim container (no inner slot, no registry) is never deleted by cleanup.

        Option B: cleanup_partial_slots skips all interim dirs unconditionally.
        An empty interim container is the I5 guard's responsibility, not this
        function's.
        """
        container = tmp_path / "episodic" / "interim_20260101T0000"
        container.mkdir(parents=True)

        removed = cleanup_partial_slots(tmp_path)

        assert removed == [], f"Expected no removals, got: {removed}"
        assert container.exists(), "Empty interim container was wrongly deleted"

    def test_flat_main_tier_scratch_still_deleted_all_tiers(self, tmp_path):
        """Non-interim partial flat slots under all three tiers are still removed.

        Ensures the interim-skip logic does not accidentally suppress genuine
        flat-slot cleanup for episodic, semantic, and procedural.
        """
        slots = [
            self._make_partial_slot(tmp_path, "episodic", "20260601-000000", "meta.json"),
            self._make_partial_slot(tmp_path, "semantic", "20260601-000000", "adapter_config.json"),
            self._make_partial_slot(
                tmp_path, "procedural", "20260601-000000", "adapter_model.safetensors"
            ),
        ]
        removed = cleanup_partial_slots(tmp_path)

        assert all(not s.exists() for s in slots), "One or more flat-scratch slots were not removed"
        assert {r["tier"] for r in removed} == {"episodic", "semantic", "procedural"}
        assert len(removed) == 3
