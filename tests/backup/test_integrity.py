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
- Boot degraded: corrupt registry → integrity_check_failed True.

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


def _write_key_registry(
    path: Path,
    keys: list[str] | None = None,
    *,
    active: list[str] | None = None,
    stale: list[str] | None = None,
    simhash: dict[str, int] | None = None,
) -> None:
    """Write a minimal indexed_key_registry.json (new unified schema).

    Supports two call shapes:
    - Legacy positional: ``_write_key_registry(path, ["k1", "k2"])`` — all active.
    - Explicit partitions: ``_write_key_registry(path, active=["k1"], stale=["k2"])``.

    The optional ``simhash`` kwarg writes fingerprints into the registry's
    unified ``"simhash"`` field (keys are routed to the active or stale partition
    automatically).  When absent, no fingerprints are written.

    The ``keys`` positional argument is treated as ``active`` when provided; it
    must not be combined with the keyword forms.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if keys is not None:
        active = keys
    active = active or []
    stale = stale or []
    reg = KeyRegistry()
    for k in active:
        reg.add(k)
    for k in stale:
        reg.add(k)
        reg.stale(k)
    if simhash:
        for k, fp in simhash.items():
            reg.set_simhash(k, fp)
    path.write_bytes(reg.save_bytes())


def _write_graph(path: Path) -> None:
    """Write a minimal (empty) graph.json in node_link_data format."""
    import networkx as nx

    path.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(nx.MultiDiGraph())
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_manifest(
    slot_dir: Path,
    name: str = "episodic",
    *,
    registry_path: Path | None = None,
    registry_sha256: str | None = None,
    key_count: int | None = None,
) -> None:
    """Write a minimal valid meta.json for a weight slot.

    Defaults to the legacy no-binding shape (``registry_sha256=""``,
    ``key_count=0``) — sufficient for schema-only tests that never resolve
    the slot through :func:`~paramem.adapters.registry_binding.verify_tier_binding`.

    When *registry_path* is given, ``registry_sha256`` is derived as the
    live plaintext hash of that file (mirroring what
    :func:`~paramem.adapters.manifest.tier_registry_sha256` computes in
    production) and ``key_count`` defaults to the registry's active-key
    count — this is what makes the written slot the tier's LIVE weight slot
    under the binding, i.e. resolves to
    :data:`~paramem.adapters.registry_binding.VERIFIED`.

    *registry_sha256* / *key_count* explicit overrides always win over the
    derived values — used to construct a deliberate hash or key-count
    mismatch against a real registry.
    """
    slot_dir.mkdir(parents=True, exist_ok=True)
    resolved_sha = registry_sha256 if registry_sha256 is not None else ""
    resolved_count = key_count if key_count is not None else 0
    if registry_path is not None:
        from paramem.backup.hashing import plaintext_sha256
        from paramem.training.key_registry import KeyRegistry

        if registry_sha256 is None:
            resolved_sha = plaintext_sha256(registry_path)
        if key_count is None:
            resolved_count = len(KeyRegistry.load(registry_path).list_active())
    manifest = {
        "schema_version": 4,
        "name": name,
        "trained_at": "2026-05-01T00:00:00Z",
        "window_stamp": "",
        "base_model": {"repo": "test/model", "sha": "abc", "hash": "sha256:deadbeef"},
        "tokenizer": {"name_or_path": "test/model", "vocab_size": 32000, "merges_hash": "abc"},
        "lora": {"rank": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["q_proj"]},
        "registry_sha256": resolved_sha,
        "key_count": resolved_count,
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
    _write_key_registry(
        ep_dir / "indexed_key_registry.json", ["key1", "key2"], simhash={"key1": 1, "key2": 2}
    )
    slot = ep_dir / "20260501-000000"
    _write_manifest(slot, "episodic", registry_path=ep_dir / "indexed_key_registry.json")

    # semantic tier (empty registry — skipped)
    sem_dir = adapter_dir / "semantic"
    sem_dir.mkdir(parents=True, exist_ok=True)
    _write_key_registry(sem_dir / "indexed_key_registry.json", [])

    # procedural tier (empty registry — skipped)
    proc_dir = adapter_dir / "procedural"
    proc_dir.mkdir(parents=True, exist_ok=True)
    _write_key_registry(proc_dir / "indexed_key_registry.json", [])

    return cfg


@pytest.fixture()
def simulate_store_dir(tmp_path):
    """Minimal valid 'simulate' store: episodic has 2 keys + graph.json."""
    cfg = _make_config(tmp_path, mode="simulate")
    adapter_dir = cfg.adapter_dir

    ep_dir = adapter_dir / "episodic"
    ep_dir.mkdir(parents=True, exist_ok=True)
    _write_key_registry(
        ep_dir / "indexed_key_registry.json", ["key1", "key2"], simhash={"key1": 1, "key2": 2}
    )
    _write_graph(ep_dir / "graph.json")

    # semantic + procedural empty
    for tier in ("semantic", "procedural"):
        d = adapter_dir / tier
        d.mkdir(parents=True, exist_ok=True)
        _write_key_registry(d / "indexed_key_registry.json", [])

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

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)
        assert report.ok is True
        # simhash for episodic must be skipped or ok (not a failure)
        sim_checks = [c for c in report.checks if c.category == "simhash" and c.tier == "episodic"]
        assert all(c.status in (_OK, _SKIPPED) for c in sim_checks)

    def test_foreign_shaped_registry_is_schema_error(self, tmp_path):
        """A registry file that fails KeyRegistry.load's strict shape check
        (foreign JSON, missing 'simhash') reports schema_error for the
        registry category — the ValueError -> _SCHEMA_ERROR arm in
        _check_registry, the live-half counterpart to the manifest one
        pinned below
        (TestManifestFailure.test_malformed_manifest_is_non_ok_via_binding)."""
        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        # Foreign-shaped: has active_keys but no simhash section.
        (ep_dir / "indexed_key_registry.json").write_text(
            json.dumps({"active_keys": ["key1"]}), encoding="utf-8"
        )

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)
        assert report.ok is False
        registry_failures = [f for f in report.failures if f.category == "registry"]
        assert len(registry_failures) >= 1
        assert registry_failures[0].status == _SCHEMA_ERROR


class TestManifestFailure:
    def test_malformed_manifest_is_non_ok_via_binding(self, tmp_path):
        """meta.json with a missing required field → non-ok, via the binding.

        verify_tier_binding's slot resolution (find_live_slot) reads and
        validates every candidate's manifest to compare it against the live
        registry hash; a candidate that fails to parse is skipped rather
        than matched, so the tier resolves NO_MATCHING_SLOT — reported here
        as an inconsistent binding row, not a manifest schema_error (there
        is no longer a single resolved slot left to schema-check once the
        binding itself cannot match one)."""
        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        _write_key_registry(ep_dir / "indexed_key_registry.json", ["key1"], simhash={"key1": 1})

        # Write a slot with invalid meta.json (missing 'name' field)
        slot = ep_dir / "20260501-000000"
        slot.mkdir(parents=True, exist_ok=True)
        bad_manifest = {"schema_version": 4, "trained_at": "2026-05-01T00:00:00Z"}
        (slot / "meta.json").write_text(json.dumps(bad_manifest), encoding="utf-8")

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)
        assert report.ok is False
        manifest_failures = [f for f in report.failures if f.category == "manifest"]
        assert len(manifest_failures) == 1, manifest_failures
        assert manifest_failures[0].status == _INCONSISTENT


class TestGraphFailure:
    def test_graph_missing_in_simulate_with_keys(self, tmp_path):
        """simulate mode + tier has keys but no graph.json → missing failure."""
        cfg = _make_config(tmp_path, mode="simulate")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        _write_key_registry(ep_dir / "indexed_key_registry.json", ["key1"], simhash={"key1": 1})
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
        _write_key_registry(ep_dir / "indexed_key_registry.json", ["key1"], simhash={"key1": 1})
        slot = ep_dir / "20260501-000000"
        _write_manifest(slot, "episodic", registry_path=ep_dir / "indexed_key_registry.json")
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
        # simhash only has key1 (key2 missing from the unified registry)
        _write_key_registry(
            ep_dir / "indexed_key_registry.json", ["key1", "key2"], simhash={"key1": 1}
        )
        slot = ep_dir / "20260501-000000"
        _write_manifest(slot, "episodic", registry_path=ep_dir / "indexed_key_registry.json")

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
        # simhash has extra key2 (orphan — not in active or stale partition)
        _write_key_registry(
            ep_dir / "indexed_key_registry.json", ["key1"], simhash={"key1": 1, "key2": 2}
        )
        slot = ep_dir / "20260501-000000"
        _write_manifest(slot, "episodic", registry_path=ep_dir / "indexed_key_registry.json")

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
        _write_key_registry(ep_dir / "indexed_key_registry.json", ["key1"], simhash={"key1": 1})
        slot = ep_dir / "20260501-000000"
        _write_manifest(slot, "episodic", registry_path=ep_dir / "indexed_key_registry.json")

        # Write key_metadata with an extra orphan key
        cfg.key_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "keys": {
                "key1": {"speaker_id": "speaker0"},
                "orphan_key": {"speaker_id": "speaker0"},
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
        _write_key_registry(
            ep_dir / "indexed_key_registry.json",
            ["key1", "key2"],
            simhash={"key1": 1, "key2": 2},
        )
        slot = ep_dir / "20260501-000000"
        _write_manifest(slot, "episodic", registry_path=ep_dir / "indexed_key_registry.json")

        # key_metadata only has key1 — key2 is absent (that's fine)
        cfg.key_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"keys": {"key1": {"speaker_id": "speaker0"}}}
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
# Stale-key cross-consistency (known = active ∪ stale)
# ---------------------------------------------------------------------------


class TestStaleKeyCrossConsistency:
    """Cross-consistency checks respect the stale partition.

    (a) A stale key with a simhash is NOT flagged as an orphan fingerprint.
    (b) A key absent from both active AND stale IS still flagged.
    (c) A stale key without a simhash does NOT trigger missing_from_sh
        (that sub-check is active-only by design).
    (d) A stale key in key_metadata is NOT flagged as an orphan.
    (e) A wholly-unknown key in key_metadata IS flagged.
    (f) A stale-only tier keeps simhash/manifest OPTIONAL (has_keys=False).
    """

    def test_stale_simhash_not_flagged_orphan(self, tmp_path):
        """Stale key with simhash → NOT an orphan fingerprint (reproduces live proc52 bug)."""
        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        _write_key_registry(
            ep_dir / "indexed_key_registry.json",
            active=["key1"],
            stale=["proc52"],
            simhash={"key1": 1, "proc52": 2},
        )
        slot = ep_dir / "20260501-000000"
        _write_manifest(slot, "episodic", registry_path=ep_dir / "indexed_key_registry.json")

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)
        # No inconsistent failure may mention proc52.
        proc52_fails = [f for f in report.failures if "proc52" in f.detail]
        assert proc52_fails == [], f"Unexpected failures for stale key proc52: {proc52_fails}"

    def test_genuine_orphan_simhash_still_flagged(self, tmp_path):
        """Key in simhash but absent from both active AND stale → inconsistent."""
        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        _write_key_registry(
            ep_dir / "indexed_key_registry.json",
            active=["key1"],
            simhash={"key1": 1, "ghost": 99},
        )
        slot = ep_dir / "20260501-000000"
        _write_manifest(slot, "episodic", registry_path=ep_dir / "indexed_key_registry.json")

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)
        inconsistent = [f for f in report.failures if f.status == _INCONSISTENT]
        assert len(inconsistent) >= 1, "Genuine orphan simhash key must still be flagged"
        assert any("ghost" in f.detail for f in inconsistent)

    def test_stale_key_without_simhash_not_flagged_missing(self, tmp_path):
        """Stale key without a simhash does NOT trigger missing_from_sh.

        missing_from_sh is active-only (every SERVED key must have a fingerprint).
        A stale key may or may not have a retained simhash without triggering a
        missing-fingerprint failure.
        """
        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        # simhash only has key1 — old1 (stale) has no simhash
        _write_key_registry(
            ep_dir / "indexed_key_registry.json",
            active=["key1"],
            stale=["old1"],
            simhash={"key1": 1},
        )
        slot = ep_dir / "20260501-000000"
        _write_manifest(slot, "episodic", registry_path=ep_dir / "indexed_key_registry.json")

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)
        old1_fails = [f for f in report.failures if "old1" in f.detail]
        assert old1_fails == [], f"Stale key without simhash must not be flagged: {old1_fails}"

    def test_stale_key_metadata_not_flagged(self, tmp_path):
        """key_metadata entry for a stale key is NOT flagged as orphan."""
        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        _write_key_registry(
            ep_dir / "indexed_key_registry.json",
            active=["key1"],
            stale=["proc52"],
            simhash={"key1": 1, "proc52": 2},
        )
        slot = ep_dir / "20260501-000000"
        _write_manifest(slot, "episodic", registry_path=ep_dir / "indexed_key_registry.json")

        cfg.key_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "keys": {
                "key1": {"speaker_id": "spk0"},
                "proc52": {"speaker_id": "spk0"},
            }
        }
        cfg.key_metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        store = MemoryStore(replay_enabled=True)
        reg = KeyRegistry()
        reg.add("key1")
        reg.add("proc52")
        reg.stale("proc52")
        store.load_registry("episodic", reg)

        report = verify_infrastructure_integrity(cfg, store=store, daily_loadable=False)
        km_fails = [
            f for f in report.failures if f.category == "key_metadata" and f.status == _INCONSISTENT
        ]
        assert km_fails == [], f"Stale key in key_metadata must not be flagged: {km_fails}"

    def test_wholly_unknown_key_metadata_still_flagged(self, tmp_path):
        """key_metadata entry absent from both active and stale → still flagged."""
        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        _write_key_registry(
            ep_dir / "indexed_key_registry.json", active=["key1"], simhash={"key1": 1}
        )
        slot = ep_dir / "20260501-000000"
        _write_manifest(slot, "episodic", registry_path=ep_dir / "indexed_key_registry.json")

        cfg.key_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "keys": {
                "key1": {"speaker_id": "spk0"},
                "ghost_key": {"speaker_id": "spk0"},
            }
        }
        cfg.key_metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        store = MemoryStore(replay_enabled=True)
        reg = KeyRegistry()
        reg.add("key1")
        store.load_registry("episodic", reg)

        report = verify_infrastructure_integrity(cfg, store=store, daily_loadable=False)
        km_fails = [
            f for f in report.failures if f.category == "key_metadata" and f.status == _INCONSISTENT
        ]
        assert len(km_fails) == 1
        assert "ghost_key" in km_fails[0].detail

    def test_stale_only_tier_simhash_optional(self, tmp_path):
        """A tier with ONLY stale keys keeps has_keys=False → simhash/manifest OPTIONAL.

        A stale-only tier serves nothing; its simhash and manifest must stay
        optional (the has_keys regression guard).
        """
        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        # Registry has only a stale key — no active keys.
        _write_key_registry(ep_dir / "indexed_key_registry.json", active=[], stale=["old1"])
        # No simhash file and no manifest slot — should not be required.

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)
        failures = [f for f in report.failures if f.tier == "episodic"]
        assert failures == [], f"Stale-only tier must not require simhash/manifest: {failures}"


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
        _write_key_registry(ep_dir / "indexed_key_registry.json", ["key1"], simhash={"key1": 1})
        slot = ep_dir / "20260501-000000"
        _write_manifest(slot, "episodic", registry_path=ep_dir / "indexed_key_registry.json")

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)
        # semantic failures should not appear
        sem_failures = [f for f in report.failures if f.tier in ("semantic", "procedural")]
        assert sem_failures == []

    def test_partial_interim_slot_skipped(self, tmp_path):
        """Interim slot dir present but no registry → skipped, not a failure."""
        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        _write_key_registry(ep_dir / "indexed_key_registry.json", ["key1"], simhash={"key1": 1})
        slot = ep_dir / "20260501-000000"
        _write_manifest(slot, "episodic", registry_path=ep_dir / "indexed_key_registry.json")

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
        _write_key_registry(ep_dir / "indexed_key_registry.json", ["key1"], simhash={"key1": 1})
        slot = ep_dir / "20260501-000000"
        _write_manifest(slot, "episodic", registry_path=ep_dir / "indexed_key_registry.json")

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)
        # Common optional files absent → only skipped
        common_failures = [f for f in report.failures if f.category == "common"]
        assert common_failures == []


# ---------------------------------------------------------------------------
# Interim tier manifest check — nested weight-slot resolution
# ---------------------------------------------------------------------------


class TestInterimManifestResolution:
    """Regression: manifest resolution must use interim's NESTED slot root.

    Interim adapter dirs are NESTED at ``<adapter_dir>/episodic/interim_<stamp>/``
    rather than flat at ``<adapter_dir>/episodic_interim_<stamp>/``. The
    manifest check resolves its slot through
    :func:`~paramem.adapters.registry_binding.verify_tier_binding`, called
    with the already-resolved interim tier root — passing the flat tier
    *name* instead would produce a non-existent path, causing every interim
    tier with keys to emit a spurious ``no weight slot`` failure.
    """

    def _build_interim_dir(
        self,
        adapter_dir: Path,
        stamp: str,
        keys: list[str],
        include_slot: bool = True,
    ) -> tuple[str, Path]:
        """Create an interim tier under ``adapter_dir/episodic/interim_<stamp>/``.

        Returns ``(tier_name, interim_dir)`` where *tier_name* is the PEFT
        adapter name used internally (``episodic_interim_<stamp>``).
        """
        from paramem.memory.interim_adapter import INTERIM_NAME_PREFIX

        interim_dir = adapter_dir / "episodic" / f"interim_{stamp}"
        interim_dir.mkdir(parents=True, exist_ok=True)

        reg_path = interim_dir / "indexed_key_registry.json"
        _write_key_registry(
            reg_path,
            keys,
            simhash={k: i for i, k in enumerate(keys)},
        )

        if include_slot:
            slot = interim_dir / "20260603-000000"
            _write_manifest(slot, f"{INTERIM_NAME_PREFIX}{stamp}", registry_path=reg_path)

        tier_name = f"{INTERIM_NAME_PREFIX}{stamp}"
        return tier_name, interim_dir

    def test_interim_weight_slot_found_no_spurious_failure(self, tmp_path):
        """train mode: interim tier with nested slot → NO 'no weight slot' failure.

        Regression for: ``Integrity failure [manifest/episodic_interim_<stamp>]
        data/ha/adapters/episodic_interim_<stamp>/meta.json: no weight slot``
        """
        cfg = _make_config(tmp_path, mode="train")
        adapter_dir = cfg.adapter_dir

        # Main episodic tier with keys + slot
        ep_dir = adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        ep_reg = ep_dir / "indexed_key_registry.json"
        _write_key_registry(ep_reg, ["key1"], simhash={"key1": 1})
        _write_manifest(ep_dir / "20260603-000000", "episodic", registry_path=ep_reg)

        # Nested interim slot with keys + nested weight slot
        self._build_interim_dir(adapter_dir, "20260603T0000", ["ikey1"], include_slot=True)

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)

        assert report.ok is True, f"Unexpected failures: {report.failures}"
        manifest_failures = [
            f for f in report.failures if f.category == "manifest" and "interim" in f.tier
        ]
        assert manifest_failures == [], f"Spurious interim manifest failure(s): {manifest_failures}"

    def test_interim_missing_weight_slot_still_reported(self, tmp_path):
        """train mode: interim tier with keys but NO nested slot → non-ok, reported.

        Ensures the binding-based resolution does not mask genuine
        missing-slot failures for interim tiers. Exactly ONE manifest-category
        row appears for this tier — the binding's NO_CANDIDATES verdict — with
        the same detail text verify_tier_binding always uses for "no
        weight-slot candidates on disk".
        """
        cfg = _make_config(tmp_path, mode="train")
        adapter_dir = cfg.adapter_dir

        # Main episodic with keys + slot (must be valid so it doesn't mask interim)
        ep_dir = adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        ep_reg = ep_dir / "indexed_key_registry.json"
        _write_key_registry(ep_reg, ["key1"], simhash={"key1": 1})
        _write_manifest(ep_dir / "20260603-000000", "episodic", registry_path=ep_reg)

        # Interim with keys but NO nested weight slot
        self._build_interim_dir(adapter_dir, "20260603T0000", ["ikey1"], include_slot=False)

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)

        assert report.ok is False
        manifest_failures = [
            f for f in report.failures if f.category == "manifest" and "interim" in f.tier
        ]
        assert len(manifest_failures) == 1, (
            f"Expected exactly one interim manifest failure, got: {manifest_failures}"
        )
        assert manifest_failures[0].status == _MISSING
        assert "no weight-slot candidates" in manifest_failures[0].detail

    def test_main_tier_manifest_still_resolves_flat(self, tmp_path):
        """train mode: main tier slot is still found correctly after the refactor.

        Verifies the binding-based resolution does not break main-tier
        (flat) weight-slot resolution. Exactly ONE ok manifest-category row
        is expected — the binding verdict, with no second, independently
        re-derived row.
        """
        cfg = _make_config(tmp_path, mode="train")
        adapter_dir = cfg.adapter_dir

        ep_dir = adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        ep_reg = ep_dir / "indexed_key_registry.json"
        _write_key_registry(
            ep_reg,
            ["key1", "key2"],
            simhash={"key1": 1, "key2": 2},
        )
        # Flat slot under episodic/
        slot = ep_dir / "20260501-000000"
        _write_manifest(slot, "episodic", registry_path=ep_reg)

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)

        assert report.ok is True, f"Main-tier manifest should resolve: {report.failures}"
        manifest_checks = [
            c for c in report.checks if c.category == "manifest" and c.tier == "episodic"
        ]
        assert len(manifest_checks) == 1, manifest_checks
        assert manifest_checks[0].status == _OK
        # The row's path is the resolved slot's meta.json, not the registry file.
        assert manifest_checks[0].path == str(slot / "meta.json")


# ---------------------------------------------------------------------------
# Registry↔slot binding — verify_tier_binding consumption
# ---------------------------------------------------------------------------


class TestRegistryBinding:
    """verify_infrastructure_integrity resolves the manifest check's slot
    through verify_tier_binding and reports the binding verdict as THE one
    manifest-category row for a keyed train-mode tier — no second,
    independently re-derived row — so a hash-mismatched, count-mismatched,
    or slot-less tier is a visible non-ok check even when a candidate
    manifest is otherwise well-formed JSON, and never alongside a false
    second row (e.g. "no weight slot" when a slot exists but doesn't bind)."""

    def test_hash_mismatched_slot_is_non_ok(self, tmp_path):
        """A slot whose manifest registry_sha256 does not match the live
        registry hash is a visible non-ok check (NO_MATCHING_SLOT) —
        exactly ONE failure, not a second row falsely claiming no slot
        exists (one real candidate slot IS on disk, it just doesn't bind)."""
        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        _write_key_registry(ep_dir / "indexed_key_registry.json", ["key1"], simhash={"key1": 1})
        slot = ep_dir / "20260501-000000"
        # Deliberately wrong hash — does not correspond to the live registry.
        _write_manifest(slot, "episodic", registry_sha256="0" * 64)

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)

        assert report.ok is False
        assert len(report.failures) == 1, report.failures
        binding_failures = [
            f for f in report.failures if f.category == "manifest" and f.status == _INCONSISTENT
        ]
        assert len(binding_failures) == 1, report.failures
        assert binding_failures[0].tier == "episodic"

    def test_key_count_mismatch_is_non_ok(self, tmp_path):
        """A slot whose registry_sha256 matches but key_count disagrees with
        the registry's active count is a visible non-ok check
        (KEY_COUNT_MISMATCH) — exactly ONE manifest-category row total, the
        binding row itself (no second, independently re-parsed schema row)."""
        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        reg_path = ep_dir / "indexed_key_registry.json"
        _write_key_registry(reg_path, ["key1"], simhash={"key1": 1})
        slot = ep_dir / "20260501-000000"
        _write_manifest(slot, "episodic", registry_path=reg_path, key_count=99)

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)

        assert report.ok is False
        manifest_checks = [
            c for c in report.checks if c.category == "manifest" and c.tier == "episodic"
        ]
        assert len(manifest_checks) == 1, manifest_checks
        assert manifest_checks[0].status == _INCONSISTENT
        assert "key_count" in manifest_checks[0].detail
        # The row's path is the resolved (mismatched-count, but hash-matched) slot.
        assert manifest_checks[0].path == str(slot / "meta.json")

    def test_no_candidates_on_keyed_tier_is_failure(self, tmp_path):
        """A keyed train-mode tier with NO weight-slot candidates at all
        (never trained, or the slot was fully removed) is a FAILURE row via
        the binding — NO_CANDIDATES maps to missing, never skipped. Pins
        against a mutation that would downgrade NO_CANDIDATES to skipped,
        which would otherwise pass silently now that there is no longer a
        second, independently-derived fallback row also carrying the
        failure signal."""
        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        _write_key_registry(ep_dir / "indexed_key_registry.json", ["key1"], simhash={"key1": 1})
        # No slot directory at all.

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)

        assert report.ok is False
        manifest_checks = [
            c for c in report.checks if c.category == "manifest" and c.tier == "episodic"
        ]
        assert len(manifest_checks) == 1, manifest_checks
        assert manifest_checks[0].status == _MISSING
        assert manifest_checks[0] in report.failures

    def test_registry_absent_with_slots_is_inconsistent(self, tmp_path):
        """REGISTRY_ABSENT_WITH_SLOTS -- a keyed tier whose weight slots
        remain on disk but whose registry file was deleted (a registry
        loss, distinct from a tier that was simply never trained) -- maps
        to inconsistent via _binding_check.

        Constructed directly against a synthetic TierBinding rather than
        through verify_infrastructure_integrity's full per-tier walk: that
        walk short-circuits (registry-absent -> skipped, not a failure)
        before ever calling verify_tier_binding whenever the on-disk
        registry is absent, by design -- a tier that was NEVER trained
        looks identical on disk to one whose registry was lost, and the
        walk cannot tell them apart. This pins _binding_check's own mapping
        table for the corruption case, which app.py's boot mount path
        (verify_tier_binding called directly, no registry-absence gate)
        does reach in production.
        """
        from paramem.adapters.registry_binding import REGISTRY_ABSENT_WITH_SLOTS, TierBinding
        from paramem.backup.integrity import _binding_check
        from paramem.training.key_registry import KeyRegistry

        tier_root = tmp_path / "episodic"
        tier_root.mkdir(parents=True, exist_ok=True)
        registry = KeyRegistry()
        registry.add("key1")
        binding = TierBinding(
            tier="episodic",
            tier_root=tier_root,
            status=REGISTRY_ABSENT_WITH_SLOTS,
            registry=registry,
            registry_present=False,
            slot=None,
            manifest=None,
            candidate_count=1,
            detail=(
                "1 candidate slot(s) present but no indexed_key_registry.json exists for this tier"
            ),
        )

        check = _binding_check(binding, "episodic")

        assert check.category == "manifest"
        assert check.status == _INCONSISTENT
        assert check.detail == binding.detail

    def test_foreign_shaped_registry_with_slot_present_is_schema_error(self, tmp_path):
        """A registry that fails KeyRegistry.load's shape check reports
        schema_error even when a weight-slot manifest sits alongside it —
        has_keys stays False for an unreadable registry, so the binding is
        never computed and cannot mask the registry failure as ok."""
        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        # Foreign-shaped: has active_keys but no simhash section.
        (ep_dir / "indexed_key_registry.json").write_text(
            json.dumps({"active_keys": ["key1"]}), encoding="utf-8"
        )
        slot = ep_dir / "20260501-000000"
        _write_manifest(slot, "episodic")

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)

        assert report.ok is False
        registry_failures = [f for f in report.failures if f.category == "registry"]
        assert len(registry_failures) == 1
        assert registry_failures[0].status == _SCHEMA_ERROR
        manifest_checks = [
            c for c in report.checks if c.category == "manifest" and c.tier == "episodic"
        ]
        assert manifest_checks == [], (
            f"No binding/manifest row expected for an unreadable registry: {manifest_checks}"
        )

    def test_donor_store_contributes_no_checks(self, tmp_path):
        """A donor-store directory beside the memory tiers contributes no
        checks.

        Two independent reasons, either one sufficient on its own:
        iter_tier_roots enumerates only the three literal main tiers plus
        interim dirs found under episodic/interim_*, so a donor directory
        sitting alongside them is structurally never yielded to the
        per-tier loop at all. And even if it somehow were, a donor store
        carries no indexed_key_registry.json by design (donor stores are
        keyless), so has_keys would stay False and the loop would exit via
        the registry-absent skip/continue before verify_tier_binding's
        donor guard (which raises ValueError for a donor tier_root) is
        ever called."""
        from paramem.training.donor import DONOR_STORE_PREFIX

        cfg = _make_config(tmp_path, mode="train")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        ep_reg = ep_dir / "indexed_key_registry.json"
        _write_key_registry(ep_reg, ["key1"], simhash={"key1": 1})
        _write_manifest(ep_dir / "20260501-000000", "episodic", registry_path=ep_reg)

        donor_dir = cfg.adapter_dir / f"{DONOR_STORE_PREFIX}20260501-000000"
        donor_slot = donor_dir / "20260501-000000"
        donor_slot.mkdir(parents=True, exist_ok=True)
        donor_manifest = {
            "schema_version": 4,
            "name": donor_dir.name,
            "trained_at": "2026-05-01T00:00:00Z",
            "window_stamp": "",
            "base_model": {"repo": "test/model", "sha": "abc", "hash": "sha256:deadbeef"},
            "tokenizer": {"name_or_path": "test/model", "vocab_size": 32000, "merges_hash": "abc"},
            "lora": {"rank": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["q_proj"]},
            "registry_sha256": "",
            "key_count": 3,
        }
        (donor_slot / "meta.json").write_text(json.dumps(donor_manifest), encoding="utf-8")

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)

        assert report.ok is True, report.failures
        assert all(donor_dir.name != c.tier for c in report.checks)


# ---------------------------------------------------------------------------
# Simulate mode's manifest row — unconditional, oracle-free skip
# ---------------------------------------------------------------------------


class TestSimulateManifestRow:
    """simulate mode's manifest check is one unconditional, oracle-free
    'skipped' row per committed tier — no slot resolution happens, so its
    presence (and detail) never depends on whether a weight slot happens to
    exist on disk, whether its hash would match, or whether the tier has
    any active keys."""

    def test_skip_row_present_with_no_slot_on_disk(self, tmp_path):
        """simulate mode, tier has keys, NO weight slot anywhere on disk →
        still exactly one skipped manifest row (not absent, not a failure)."""
        cfg = _make_config(tmp_path, mode="simulate")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        _write_key_registry(ep_dir / "indexed_key_registry.json", ["key1"], simhash={"key1": 1})
        _write_graph(ep_dir / "graph.json")

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)

        assert report.ok is True, report.failures
        manifest_checks = [
            c for c in report.checks if c.category == "manifest" and c.tier == "episodic"
        ]
        assert len(manifest_checks) == 1, manifest_checks
        assert manifest_checks[0].status == _SKIPPED
        assert manifest_checks[0].detail == "simulate mode"

    def test_skip_row_present_with_matching_slot_on_disk(self, tmp_path):
        """simulate mode, a weight slot happens to exist and matches the
        live registry hash → the row is still just 'skipped', unaffected by
        the slot's presence or hash match (no oracle call is made)."""
        cfg = _make_config(tmp_path, mode="simulate")
        ep_dir = cfg.adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        reg_path = ep_dir / "indexed_key_registry.json"
        _write_key_registry(reg_path, ["key1"], simhash={"key1": 1})
        _write_graph(ep_dir / "graph.json")
        _write_manifest(ep_dir / "20260501-000000", "episodic", registry_path=reg_path)

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)

        assert report.ok is True, report.failures
        manifest_checks = [
            c for c in report.checks if c.category == "manifest" and c.tier == "episodic"
        ]
        assert len(manifest_checks) == 1, manifest_checks
        assert manifest_checks[0].status == _SKIPPED
        assert manifest_checks[0].detail == "simulate mode"
        assert manifest_checks[0].path == str(reg_path)

    def test_skip_row_present_even_when_registry_empty(self, tmp_path):
        """simulate mode, a tier with a present-but-empty registry (no
        active keys) still gets the unconditional skip row — it is not
        gated on has_keys."""
        cfg = _make_config(tmp_path, mode="simulate")
        sem_dir = cfg.adapter_dir / "semantic"
        sem_dir.mkdir(parents=True, exist_ok=True)
        _write_key_registry(sem_dir / "indexed_key_registry.json", [])

        report = verify_infrastructure_integrity(cfg, daily_loadable=False)

        assert report.ok is True, report.failures
        manifest_checks = [
            c for c in report.checks if c.category == "manifest" and c.tier == "semantic"
        ]
        assert len(manifest_checks) == 1, manifest_checks
        assert manifest_checks[0].status == _SKIPPED


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
        expected_check = {
            "path": "/bad.json",
            "category": "registry",
            "tier": "episodic",
            "status": _PARSE_ERROR,
            "detail": "bad json",
        }
        assert d == {
            "ok": False,
            "checks": [expected_check],
            "failures": [expected_check],
        }

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
        (tier_root / "key_metadata.json").write_text("{}")
        removed = cleanup_partial_slots(tmp_path)
        assert removed == []
        assert (tier_root / "indexed_key_registry.json").exists()
        assert (tier_root / "key_metadata.json").exists()

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
        """A tier name outside MAIN_TIERS is not walked."""
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
        by find_live_slot + the boot registry-consistency sweep).
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
        boot registry-consistency sweep's job.
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
        An empty interim container is the boot registry-consistency sweep's
        responsibility, not this function's.
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
