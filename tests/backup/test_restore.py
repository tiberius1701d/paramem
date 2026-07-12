"""Tests for restore_bundle() in paramem.backup.backup.

Round-trip tests: write_bundle → restore_bundle into a scratch data_dir.

Coverage:
(a) find_live_slot resolves each restored adapter slot via its per-tier hash.
(b) registry + per-tier registries + simhash + speaker_profiles present in
    scratch dir after restore.
(c) restart_required is True.
(d) Restored adapter slot names are fresh (_slot_name_now timestamps, not the
    bundle's original timestamps).
(e) restore_config=False leaves target config untouched;
    restore_config=True writes it.
(f) Corrupt bundle (hash mismatch) → FingerprintMismatchError raised loud;
    no mutation in scratch dir.
(g) Safety bundle created when target has an episodic slot; skipped gracefully
    when empty (no adapters at all).
(h) Missing bundle.meta.json → BundleManifestError.
(i) Forward schema version in bundle.meta.json → BundleManifestError.
(j) [ENCRYPTED] Full round-trip with a real daily identity — encryption ON:
    write_bundle byte-copies age envelopes, plaintext carve-outs stay
    plaintext; restore_bundle preserves on-disk state exactly; assert_mode_consistency
    does not raise; find_live_slot resolves.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from paramem.backup.backup import _atomic_write_file, restore_bundle, write_bundle
from paramem.backup.types import (
    BundleManifestError,
    FingerprintMismatchError,
    RestoreAbortedError,
    RestoreResult,
)

_AGE_MAGIC = b"age-encryption.org/v1\n"


# ---------------------------------------------------------------------------
# _atomic_write_file — S2 stale-temp robustness
# ---------------------------------------------------------------------------


class TestAtomicWriteFile:
    """Unit tests for _atomic_write_file (S2: stale-temp robustness).

    The pre-S2 implementation used a fixed ``.restore-pending`` suffix with
    ``O_CREAT | O_EXCL``: a crash between create and rename left a stale temp
    that wedged the next restore with ``FileExistsError``.  The S2 fix uses
    ``tempfile.mkstemp`` which generates a unique name, so stale temps from a
    prior crash never block a subsequent restore.
    """

    def test_happy_path_writes_content(self, tmp_path) -> None:
        """_atomic_write_file writes the expected bytes to dst."""
        dst = tmp_path / "target.json"
        _atomic_write_file(b'{"ok": true}', dst)
        assert dst.read_bytes() == b'{"ok": true}'

    def test_parent_created_automatically(self, tmp_path) -> None:
        """dst.parent is created if absent."""
        dst = tmp_path / "subdir" / "target.json"
        _atomic_write_file(b"data", dst)
        assert dst.read_bytes() == b"data"

    def test_stale_temp_does_not_wedge(self, tmp_path) -> None:
        """A stale .restore-pending temp left by a prior crash must not cause FileExistsError.

        Pre-S2: the fixed suffix + O_EXCL would raise FileExistsError if the
        same-named temp already exists.  S2: mkstemp always picks a unique name
        so the stale file is ignored.
        """
        dst = tmp_path / "target.json"
        # Plant a stale temp file with the old fixed-suffix naming scheme.
        stale_temp = tmp_path / "target.json.restore-pending"
        stale_temp.write_bytes(b"stale-content")

        # This must NOT raise even though a stale temp exists at the old path.
        _atomic_write_file(b"fresh-content", dst)

        assert dst.read_bytes() == b"fresh-content"
        # The stale temp is untouched (mkstemp uses a unique name; the old file is orphaned).
        assert stale_temp.exists()

    def test_mode_0o600_applied(self, tmp_path) -> None:
        """dst has mode 0o600 after write (no world-readable plaintext window)."""
        import stat

        dst = tmp_path / "secret.json"
        _atomic_write_file(b"secret", dst)
        file_mode = stat.S_IMODE(dst.stat().st_mode)
        assert file_mode == 0o600, f"Expected mode 0o600, got 0o{file_mode:o}"

    def test_overwrites_existing_dst(self, tmp_path) -> None:
        """_atomic_write_file atomically replaces an existing dst file."""
        dst = tmp_path / "target.json"
        dst.write_bytes(b"old-content")
        _atomic_write_file(b"new-content", dst)
        assert dst.read_bytes() == b"new-content"


# ---------------------------------------------------------------------------
# Helpers / fixtures — reuse the same layout as test_bundle.py
# ---------------------------------------------------------------------------

_FAKE_AGE_MAGIC = b"age-encryption.org/v1\n"


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _make_adapter_slot(
    parent_dir: Path,
    slot_name: str,
    registry_sha256: str,
    adapter_name: str = "episodic",
    weight_bytes: bytes = b"fake_plain_weights",
) -> Path:
    """Create a minimal adapter slot directory with meta.json + adapter files."""
    slot = parent_dir / slot_name
    slot.mkdir(parents=True, exist_ok=True)

    meta = {
        "schema_version": 4,
        "name": adapter_name,
        "trained_at": "2026-05-20T12:00:00Z",
        "window_stamp": "",
        "base_model": {
            "repo": "mistralai/Mistral-7B-Instruct-v0.3",
            "sha": "c170c708abc",
            "hash": "sha256:f4b6abc",
        },
        "tokenizer": {
            "name_or_path": "mistralai/Mistral-7B-Instruct-v0.3",
            "vocab_size": 32000,
            "merges_hash": "d" * 64,
        },
        "lora": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"],
        },
        "registry_sha256": registry_sha256,
        "key_count": 10,
        "synthesized": False,
    }
    (slot / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (slot / "adapter_model.safetensors").write_bytes(weight_bytes)
    (slot / "adapter_config.json").write_bytes(b'{"peft_type": "LORA"}')
    return slot


def _make_source_store(tmp_path: Path) -> dict:
    """Build a minimal source data directory for write_bundle.

    Returns a dict with all paths needed to call write_bundle and
    subsequently call restore_bundle.

    Contains:
    - procedural main tier: finalized main slot with indexed_key_registry.json
    - episodic interim tier: interim family (no main slot)
    - key_metadata.json (global registry)
    - speaker_profiles.json
    - server.yaml
    """
    src = tmp_path / "src_store"
    src.mkdir()

    # Config
    config_path = src / "server.yaml"
    config_path.write_bytes(b"model: mistral\nport: 8420\n")

    # Global registry
    registry_dir = src / "registry"
    registry_dir.mkdir()
    registry_path = registry_dir / "key_metadata.json"
    registry_path.write_bytes(b'{"speakers": {}, "cycles": []}')

    # Speaker profiles
    speaker_profiles_path = src / "speaker_profiles.json"
    speaker_profiles_path.write_bytes(b'{"profiles": [{"id": "speaker_1"}]}')

    adapters_base = src / "adapters"

    # --- Procedural: finalized main slot ---
    proc_content = b'{"keys": {"proc_key": 1}}'
    proc_hash = _sha256(proc_content)
    proc_dir = adapters_base / "procedural"
    proc_dir.mkdir(parents=True)
    _make_adapter_slot(proc_dir, "20260520-100000", proc_hash, adapter_name="procedural")
    (proc_dir / "indexed_key_registry.json").write_bytes(proc_content)

    # --- Episodic: NO main slot, only an interim family ---
    ep_interim_content = b'{"keys": {"ep_interim_key": "val"}}'
    ep_interim_hash = _sha256(ep_interim_content)
    ep_dir = adapters_base / "episodic"
    ep_dir.mkdir(parents=True)
    interim_family_dir = ep_dir / "interim_20260517T1200"
    interim_family_dir.mkdir()
    _make_adapter_slot(
        interim_family_dir,
        "20260517-180430",
        ep_interim_hash,
        adapter_name="episodic_interim_20260517T1200",
        weight_bytes=b"episodic_weights_data",
    )
    (interim_family_dir / "indexed_key_registry.json").write_bytes(ep_interim_content)

    bundle_base = src / "backups" / "snapshot"
    bundle_base.mkdir(parents=True)

    return {
        "src": src,
        "config_path": config_path,
        "registry_path": registry_path,
        "speaker_profiles_path": speaker_profiles_path,
        "adapter_dirs": {
            "episodic": ep_dir,
            "procedural": proc_dir,
        },
        "bundle_base": bundle_base,
        "proc_hash": proc_hash,
        "ep_interim_hash": ep_interim_hash,
    }


def _build_bundle(tmp_path: Path) -> tuple[Path, dict]:
    """Build a bundle from _make_source_store; return (bundle_slot_dir, source_dict)."""
    src_dict = _make_source_store(tmp_path)
    slot = write_bundle(
        config_path=src_dict["config_path"],
        registry_path=src_dict["registry_path"],
        adapter_dirs=src_dict["adapter_dirs"],
        base_dir=src_dict["bundle_base"],
        meta_fields={"tier": "manual", "label": "test"},
        adapter_scope="live",
        speaker_profiles_path=src_dict["speaker_profiles_path"],
    )
    return slot, src_dict


# ---------------------------------------------------------------------------
# (a) find_live_slot resolves restored adapter slots
# ---------------------------------------------------------------------------


class TestRestoreFindLiveSlot:
    def test_find_live_slot_resolves_procedural_main(self, tmp_path) -> None:
        """After restore_bundle, find_live_slot resolves the procedural main slot."""
        from paramem.adapters.manifest import find_live_slot

        bundle_slot, src = _build_bundle(tmp_path)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        restore_bundle(
            bundle_slot,
            data_dir=scratch,
            config_path=config_path,
            restore_config=False,
        )

        # The procedural tier was captured in the bundle with its own registry hash.
        # After restore, find_live_slot should resolve a slot in the restored dir.
        proc_tier_root = scratch / "adapters" / "procedural"
        proc_reg_bytes = (proc_tier_root / "indexed_key_registry.json").read_bytes()
        proc_hash = _sha256(proc_reg_bytes)
        live_slot = find_live_slot(proc_tier_root, proc_hash)
        assert live_slot is not None, (
            "find_live_slot must resolve procedural slot after restore_bundle"
        )
        assert live_slot.is_dir()
        assert (live_slot / "meta.json").exists()

    def test_find_live_slot_resolves_episodic_interim(self, tmp_path) -> None:
        """After restore_bundle, find_live_slot resolves the episodic interim slot."""
        from paramem.adapters.manifest import find_live_slot

        bundle_slot, src = _build_bundle(tmp_path)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        restore_bundle(
            bundle_slot,
            data_dir=scratch,
            config_path=config_path,
            restore_config=False,
        )

        # Episodic interim: tier-root is the interim_family dir.
        interim_tier_root = scratch / "adapters" / "episodic" / "interim_20260517T1200"
        assert interim_tier_root.is_dir(), f"Interim family dir not restored: {interim_tier_root}"
        ep_reg_bytes = (interim_tier_root / "indexed_key_registry.json").read_bytes()
        ep_hash = _sha256(ep_reg_bytes)
        live_slot = find_live_slot(interim_tier_root, ep_hash)
        assert live_slot is not None, (
            "find_live_slot must resolve episodic interim slot after restore_bundle"
        )
        assert live_slot.is_dir()


# ---------------------------------------------------------------------------
# (b) Files present in scratch dir after restore
# ---------------------------------------------------------------------------


class TestRestoreFilesPresent:
    def test_registry_present(self, tmp_path) -> None:
        """key_metadata.json is restored into scratch/registry/."""
        bundle_slot, src = _build_bundle(tmp_path)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        assert (scratch / "registry" / "key_metadata.json").exists()

    def test_speaker_profiles_present(self, tmp_path) -> None:
        """speaker_profiles.json is restored into scratch/."""
        bundle_slot, src = _build_bundle(tmp_path)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        assert (scratch / "speaker_profiles.json").exists()
        content = json.loads((scratch / "speaker_profiles.json").read_bytes())
        assert "profiles" in content

    def test_per_tier_indexed_key_registry_present(self, tmp_path) -> None:
        """indexed_key_registry.json is restored at the tier-root for each adapter."""
        bundle_slot, src = _build_bundle(tmp_path)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        # Procedural tier root: scratch/adapters/procedural/
        proc_root = scratch / "adapters" / "procedural"
        assert (proc_root / "indexed_key_registry.json").exists(), (
            "procedural indexed_key_registry.json must be at tier root"
        )
        # Episodic interim: scratch/adapters/episodic/interim_20260517T1200/
        ep_interim_root = scratch / "adapters" / "episodic" / "interim_20260517T1200"
        assert (ep_interim_root / "indexed_key_registry.json").exists(), (
            "episodic interim indexed_key_registry.json must be at interim-family root"
        )

    def test_no_simhash_sidecar_at_tier_root(self, tmp_path) -> None:
        """simhash_registry.json is NOT present after restore.

        Simhashes live in indexed_key_registry.json after the SimHash unification
        refactor; the separate sidecar is no longer bundled or restored.
        """
        bundle_slot, src = _build_bundle(tmp_path)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        proc_root = scratch / "adapters" / "procedural"
        assert not (proc_root / "simhash_registry.json").exists(), (
            "simhash_registry.json sidecar must not be restored"
        )
        ep_interim_root = scratch / "adapters" / "episodic" / "interim_20260517T1200"
        assert not (ep_interim_root / "simhash_registry.json").exists(), (
            "simhash_registry.json sidecar must not be restored for interim families"
        )

    def test_weight_files_present_in_slot(self, tmp_path) -> None:
        """adapter_model.safetensors + adapter_config.json + meta.json present in slot."""
        from paramem.adapters.manifest import find_live_slot

        bundle_slot, src = _build_bundle(tmp_path)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        proc_root = scratch / "adapters" / "procedural"
        proc_reg_bytes = (proc_root / "indexed_key_registry.json").read_bytes()
        proc_hash = _sha256(proc_reg_bytes)
        live_slot = find_live_slot(proc_root, proc_hash)
        assert live_slot is not None
        assert (live_slot / "adapter_model.safetensors").exists()
        assert (live_slot / "adapter_config.json").exists()
        assert (live_slot / "meta.json").exists()


# ---------------------------------------------------------------------------
# (c) restart_required is True
# ---------------------------------------------------------------------------


class TestRestartRequired:
    def test_restart_required_is_true(self, tmp_path) -> None:
        """RestoreResult.restart_required must always be True."""
        bundle_slot, src = _build_bundle(tmp_path)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        result = restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        assert isinstance(result, RestoreResult)
        assert result.restart_required is True


# ---------------------------------------------------------------------------
# (d) Restored slot names are fresh (not the bundle's original timestamps)
# ---------------------------------------------------------------------------


class TestFreshSlotNames:
    def test_restored_slot_names_differ_from_bundle_original(self, tmp_path) -> None:
        """New slot dirs get fresh _slot_name_now timestamps, not the bundle's originals.

        The bundle captures the source slot name (e.g. '20260520-100000').
        After restore_bundle the new slot in the scratch dir should have a
        different name (a fresh timestamp from _promote_slot).
        """
        bundle_slot, src = _build_bundle(tmp_path)

        # Record the source slot names from the bundle manifest.
        import json as _json

        manifest = _json.loads((bundle_slot / "bundle.meta.json").read_text(encoding="utf-8"))
        original_slot_sources = {
            k: Path(v["slot_source"]).name for k, v in manifest["adapters"].items()
        }

        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        # Find the restored slot dirs and assert they have different names.
        from paramem.adapters.manifest import find_live_slot

        proc_root = scratch / "adapters" / "procedural"
        proc_reg = (proc_root / "indexed_key_registry.json").read_bytes()
        proc_hash = _sha256(proc_reg)
        proc_live = find_live_slot(proc_root, proc_hash)
        assert proc_live is not None

        original_proc_name = original_slot_sources.get("procedural")
        if original_proc_name:
            assert proc_live.name != original_proc_name, (
                "Restored slot must have a fresh timestamp, not the bundle's original."
            )


# ---------------------------------------------------------------------------
# (e) restore_config=False / True
# ---------------------------------------------------------------------------


class TestRestoreConfig:
    def test_restore_config_false_leaves_config_untouched(self, tmp_path) -> None:
        """restore_config=False (default) must not overwrite the target config."""
        bundle_slot, src = _build_bundle(tmp_path)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        original_config_content = b"model: ORIGINAL_CONTENT\n"
        config_path = scratch / "server.yaml"
        config_path.write_bytes(original_config_content)

        result = restore_bundle(
            bundle_slot,
            data_dir=scratch,
            config_path=config_path,
            restore_config=False,
        )

        assert config_path.read_bytes() == original_config_content, (
            "restore_config=False must leave config untouched"
        )
        assert result.restored_config is False

    def test_restore_config_true_writes_bundle_config(self, tmp_path) -> None:
        """restore_config=True atomically writes the bundle's server.yaml."""
        bundle_slot, src = _build_bundle(tmp_path)
        original_config_content = src["config_path"].read_bytes()

        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: OVERWRITTEN\n")

        result = restore_bundle(
            bundle_slot,
            data_dir=scratch,
            config_path=config_path,
            restore_config=True,
        )

        assert config_path.exists()
        restored_bytes = config_path.read_bytes()
        assert restored_bytes == original_config_content, (
            "restore_config=True must write the bundle's server.yaml content"
        )
        assert result.restored_config is True


# ---------------------------------------------------------------------------
# (f) Corrupt bundle → FingerprintMismatchError, no mutation
# ---------------------------------------------------------------------------


class TestCorruptBundle:
    def test_hash_mismatch_raises_before_mutation(self, tmp_path) -> None:
        """Corrupt one file in the bundle → FingerprintMismatchError, scratch untouched."""
        bundle_slot, src = _build_bundle(tmp_path)

        # Tamper with one of the captured files (the global registry).
        bundle_registry = bundle_slot / "registry" / "key_metadata.json"
        if bundle_registry.exists():
            original = bundle_registry.read_bytes()
            bundle_registry.write_bytes(original + b"TAMPERED")

        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        with pytest.raises(FingerprintMismatchError):
            restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        # No adapter dirs should have been created in scratch (no mutation).
        adapter_base = scratch / "adapters"
        assert not adapter_base.exists() or not any(adapter_base.rglob("meta.json")), (
            "No adapter slots should exist in scratch after a corrupt-bundle abort"
        )

    def test_hash_mismatch_message_is_actionable(self, tmp_path) -> None:
        """FingerprintMismatchError message names the corrupt file."""
        bundle_slot, src = _build_bundle(tmp_path)

        bundle_registry = bundle_slot / "registry" / "key_metadata.json"
        if bundle_registry.exists():
            bundle_registry.write_bytes(b"COMPLETELY_WRONG_CONTENT")

        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        with pytest.raises(FingerprintMismatchError) as exc_info:
            restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)
        msg = str(exc_info.value)
        # Message should reference the corrupt path.
        assert "registry" in msg or "hash" in msg.lower(), (
            f"Error message should name the corrupt file: {msg}"
        )

    def test_missing_bundle_manifest_raises(self, tmp_path) -> None:
        """Missing bundle.meta.json → BundleManifestError."""
        bundle_slot, src = _build_bundle(tmp_path)
        (bundle_slot / "bundle.meta.json").unlink()

        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        with pytest.raises(BundleManifestError):
            restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

    def test_forward_schema_version_raises(self, tmp_path) -> None:
        """bundle.meta.json with future bundle_schema_version → BundleManifestError."""
        bundle_slot, src = _build_bundle(tmp_path)
        manifest_path = bundle_slot / "bundle.meta.json"
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw["bundle_schema_version"] = 9999  # future version
        manifest_path.write_text(json.dumps(raw), encoding="utf-8")

        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        with pytest.raises(BundleManifestError, match="bundle_schema_version"):
            restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)


# ---------------------------------------------------------------------------
# (g) Safety bundle created / skipped gracefully
# ---------------------------------------------------------------------------


class TestSafetyBundle:
    def test_safety_bundle_created_when_target_has_episodic_slot(self, tmp_path) -> None:
        """Safety bundle is written when the target data_dir already has an episodic slot.

        Strategy: restore the bundle once (populates scratch), then restore
        again using scratch as both source and target — the second restore
        finds the episodic slot from the first restore and takes a safety bundle.
        """
        bundle_slot, src = _build_bundle(tmp_path)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        # First restore: populates scratch with an episodic interim slot.
        restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        # Second restore: safety bundle must be taken from scratch's current state.
        safety_base = scratch / "backups" / "snapshot"
        slots_before = set(safety_base.iterdir()) if safety_base.exists() else set()

        result = restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        assert result.safety_slot is not None, (
            "safety_slot must not be None when target has an episodic slot"
        )
        assert result.safety_slot.is_dir(), "safety_slot must be a directory"
        slots_after = set(safety_base.iterdir()) if safety_base.exists() else set()
        new_slots = slots_after - slots_before
        assert len(new_slots) >= 1, "At least one new safety bundle slot must have been created"

    def test_safety_bundle_skipped_gracefully_when_empty_target(self, tmp_path) -> None:
        """Safety bundle is skipped gracefully (safety_slot=None) on fresh empty target.

        A fresh scratch dir has no episodic slot → write_bundle raises BackupError
        → restore_bundle must catch it and continue with safety_slot=None.
        """
        bundle_slot, src = _build_bundle(tmp_path)
        scratch = tmp_path / "scratch_empty"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        # scratch has no adapters dir at all → safety bundle will fail (no episodic)
        result = restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        # Restore must succeed despite the safety bundle failure.
        assert isinstance(result, RestoreResult)
        assert result.safety_slot is None, (
            "safety_slot should be None when safety bundle skipped on fresh target"
        )
        # Actual restore artifacts must still be present.
        assert (scratch / "registry" / "key_metadata.json").exists()


# ---------------------------------------------------------------------------
# (h) restored_adapters list populated correctly
# ---------------------------------------------------------------------------


class TestRestoredAdaptersList:
    def test_restored_adapters_populated(self, tmp_path) -> None:
        """RestoreResult.restored_adapters lists all adapter names restored."""
        bundle_slot, src = _build_bundle(tmp_path)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        result = restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        # Our fixture has procedural + episodic_interim_20260517T1200.
        assert "procedural" in result.restored_adapters
        assert "episodic_interim_20260517T1200" in result.restored_adapters

    def test_restored_config_false_by_default(self, tmp_path) -> None:
        """RestoreResult.restored_config is False by default (restore_config not set)."""
        bundle_slot, src = _build_bundle(tmp_path)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        result = restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        assert result.restored_config is False


# ---------------------------------------------------------------------------
# RestoreAbortedError surfaces safety_slot on mid-restore failure
# ---------------------------------------------------------------------------


class TestRestoreAbortedError:
    """Verify that RestoreAbortedError carries safety_slot when step 5 fails.

    S3 requirement: when restore_bundle raises after the safety bundle is
    captured (step 4) due to a filesystem error in step 5, the exception
    must carry the safety_slot path so the caller (app.py handler or operator)
    can present a concrete recovery target.
    """

    def test_restore_aborted_error_carries_safety_slot(self, tmp_path, monkeypatch) -> None:
        """RestoreAbortedError.safety_slot is a valid path when step 5 fails.

        Strategy: restore the bundle once (populates scratch with an episodic
        slot so the safety bundle will be captured on a second restore), then
        inject an OSError in _atomic_write_file to simulate a step-5 failure
        on the second restore.  Assert that RestoreAbortedError is raised and
        its safety_slot is a real directory.
        """
        import paramem.backup.backup as _backup_module

        bundle_slot, src = _build_bundle(tmp_path)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        # First restore: populate scratch with an episodic slot so the safety
        # bundle is captured on the next restore attempt.
        restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        # Now inject a failure in the atomic write path during step 5.
        _call_count = {"n": 0}
        _real_atomic = _backup_module._atomic_write_file

        def _failing_atomic(src_bytes: bytes, dst, mode: int = 0o600) -> None:
            _call_count["n"] += 1
            if _call_count["n"] == 1:
                # Fail on the first call to _atomic_write_file in step 5.
                raise OSError("injected step-5 failure for S3 test")
            return _real_atomic(src_bytes, dst, mode)

        monkeypatch.setattr(_backup_module, "_atomic_write_file", _failing_atomic)

        with pytest.raises(RestoreAbortedError) as exc_info:
            restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        err = exc_info.value
        # safety_slot must be surfaced (not None) because scratch has an episodic slot.
        assert err.safety_slot is not None, (
            "RestoreAbortedError.safety_slot must not be None when the live store "
            "had an episodic slot (safety bundle should have been captured in step 4)"
        )
        assert err.safety_slot.is_dir(), f"safety_slot {err.safety_slot} must be a real directory"
        # The cause must be the original OSError.
        assert isinstance(err.cause, OSError), (
            f"RestoreAbortedError.cause must be the original OSError; got {type(err.cause)}"
        )

    def test_restore_aborted_error_safety_slot_none_on_fresh_target(
        self, tmp_path, monkeypatch
    ) -> None:
        """RestoreAbortedError.safety_slot is None when the target is fresh (empty).

        On a fresh target there is no episodic slot → the safety bundle is
        skipped (safety_slot=None).  A step-5 failure on a fresh target must
        still surface a RestoreAbortedError, but with safety_slot=None.
        """
        import paramem.backup.backup as _backup_module

        bundle_slot, src = _build_bundle(tmp_path)
        # Fresh scratch — no prior adapters.
        scratch = tmp_path / "fresh_scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        _call_count = {"n": 0}
        _real_atomic = _backup_module._atomic_write_file

        def _failing_atomic(src_bytes: bytes, dst, mode: int = 0o600) -> None:
            _call_count["n"] += 1
            if _call_count["n"] == 1:
                raise OSError("injected step-5 failure — fresh target")
            return _real_atomic(src_bytes, dst, mode)

        monkeypatch.setattr(_backup_module, "_atomic_write_file", _failing_atomic)

        with pytest.raises(RestoreAbortedError) as exc_info:
            restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        err = exc_info.value
        # On a fresh target the safety bundle is skipped → safety_slot is None.
        assert err.safety_slot is None, (
            "RestoreAbortedError.safety_slot must be None on a fresh/empty target"
        )


# ---------------------------------------------------------------------------
# (j) Encrypted round-trip regression test — encryption ON
# ---------------------------------------------------------------------------
# This class catches the bug class that Security-OFF tests miss:
#   - write_bundle must NOT encrypt plaintext carve-outs (meta.json,
#     adapter_config.json) that the live store keeps plaintext.
#   - restore_bundle must NOT decrypt key_metadata.json (and other age
#     envelopes) — it must byte-copy them so the restored store has the same
#     on-disk encryption state as the source.
#   - assert_mode_consistency(scratch) must not raise after restore (proves
#     the store is consistent: all age envelopes, no plaintext infra files).
#   - find_live_slot resolves the restored slot via the per-tier registry hash.
# ---------------------------------------------------------------------------


def _mint_and_wire_daily_identity(tmp_path: Path, monkeypatch) -> None:
    """Mint a fresh daily identity, write it to a tmp key file, wire env vars.

    Pattern from tests/server/test_backup_endpoints.py lines 539-557.
    """
    from paramem.backup.key_store import (
        DAILY_PASSPHRASE_ENV_VAR,
        _clear_daily_identity_cache,
        mint_daily_identity,
        wrap_daily_identity,
        write_daily_key_file,
    )

    ident = mint_daily_identity()
    key_path = tmp_path / "daily_key.age"
    write_daily_key_file(wrap_daily_identity(ident, "test-pw"), key_path)
    monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "test-pw")
    monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path)
    _clear_daily_identity_cache()
    return ident


def _make_encrypted_source_store(tmp_path: Path) -> dict:
    """Build a fixture store where infra files are age-encrypted.

    Mirrors _make_source_store but writes registries and speaker_profiles
    as age envelopes (using ``envelope_encrypt_bytes``) so the fixture
    exercises the live-encryption-ON code path.  Adapter weight files
    are also written with the age magic to simulate the live encrypted state.

    The daily identity MUST be wired (via _mint_and_wire_daily_identity)
    before calling this function.
    """
    from paramem.backup.encryption import envelope_encrypt_bytes

    src = tmp_path / "enc_src_store"
    src.mkdir()

    # Config (plaintext — server.yaml is not in infra_paths as encrypted)
    config_path = src / "server.yaml"
    config_path.write_bytes(b"model: mistral\nport: 8420\n")

    # Global registry — encrypted
    registry_dir = src / "registry"
    registry_dir.mkdir()
    registry_path = registry_dir / "key_metadata.json"
    reg_plaintext = b'{"speakers": {}, "cycles": []}'
    registry_path.write_bytes(envelope_encrypt_bytes(reg_plaintext))

    # Speaker profiles — encrypted
    speaker_profiles_path = src / "speaker_profiles.json"
    sp_plaintext = b'{"profiles": [{"id": "spk1"}]}'
    speaker_profiles_path.write_bytes(envelope_encrypt_bytes(sp_plaintext))

    adapters_base = src / "adapters"

    # --- Procedural: finalized main slot ---
    proc_reg_plaintext = b'{"keys": {"proc_key": 1}}'
    proc_reg_ciphertext = envelope_encrypt_bytes(proc_reg_plaintext)
    proc_hash = hashlib.sha256(proc_reg_plaintext).hexdigest()
    proc_dir = adapters_base / "procedural"
    proc_dir.mkdir(parents=True)

    proc_slot = proc_dir / "20260520-100000"
    proc_slot.mkdir()
    # Adapter weights — age-encrypted (simulates the live encrypted state)
    weight_plaintext = b"fake_weight_data_for_proc"
    proc_slot_meta = {
        "schema_version": 4,
        "name": "procedural",
        "trained_at": "2026-05-20T12:00:00Z",
        "window_stamp": "",
        "base_model": {
            "repo": "mistralai/Mistral-7B-Instruct-v0.3",
            "sha": "c170abc",
            "hash": "sha256:f4b6abc",
        },
        "tokenizer": {
            "name_or_path": "mistralai/Mistral-7B-Instruct-v0.3",
            "vocab_size": 32000,
            "merges_hash": "d" * 64,
        },
        "lora": {"rank": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
        "registry_sha256": proc_hash,
        "key_count": 10,
        "synthesized": False,
    }
    # meta.json and adapter_config.json are PLAINTEXT carve-outs
    (proc_slot / "meta.json").write_text(json.dumps(proc_slot_meta), encoding="utf-8")
    (proc_slot / "adapter_config.json").write_bytes(b'{"peft_type": "LORA"}')
    # adapter_model.safetensors is age-encrypted
    (proc_slot / "adapter_model.safetensors").write_bytes(envelope_encrypt_bytes(weight_plaintext))

    # Per-tier registries — encrypted
    proc_dir.joinpath("indexed_key_registry.json").write_bytes(proc_reg_ciphertext)

    # --- Episodic: interim family ---
    ep_reg_plaintext = b'{"keys": {"ep_interim_key": "val"}}'
    ep_reg_ciphertext = envelope_encrypt_bytes(ep_reg_plaintext)
    ep_interim_hash = hashlib.sha256(ep_reg_plaintext).hexdigest()
    ep_dir = adapters_base / "episodic"
    ep_dir.mkdir(parents=True)
    interim_family_dir = ep_dir / "interim_20260517T1200"
    interim_family_dir.mkdir()

    ep_interim_slot = interim_family_dir / "20260517-180430"
    ep_interim_slot.mkdir()
    ep_interim_meta = {
        "schema_version": 4,
        "name": "episodic_interim_20260517T1200",
        "trained_at": "2026-05-17T18:04:30Z",
        "window_stamp": "",
        "base_model": {
            "repo": "mistralai/Mistral-7B-Instruct-v0.3",
            "sha": "c170abc",
            "hash": "sha256:f4b6abc",
        },
        "tokenizer": {
            "name_or_path": "mistralai/Mistral-7B-Instruct-v0.3",
            "vocab_size": 32000,
            "merges_hash": "d" * 64,
        },
        "lora": {"rank": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
        "registry_sha256": ep_interim_hash,
        "key_count": 5,
        "synthesized": False,
    }
    (ep_interim_slot / "meta.json").write_text(json.dumps(ep_interim_meta), encoding="utf-8")
    (ep_interim_slot / "adapter_config.json").write_bytes(b'{"peft_type": "LORA"}')
    (ep_interim_slot / "adapter_model.safetensors").write_bytes(
        envelope_encrypt_bytes(b"episodic_weights")
    )

    interim_family_dir.joinpath("indexed_key_registry.json").write_bytes(ep_reg_ciphertext)

    bundle_base = src / "backups" / "snapshot"
    bundle_base.mkdir(parents=True)

    return {
        "src": src,
        "config_path": config_path,
        "registry_path": registry_path,
        "speaker_profiles_path": speaker_profiles_path,
        "adapter_dirs": {
            "episodic": ep_dir,
            "procedural": proc_dir,
        },
        "bundle_base": bundle_base,
        "proc_hash": proc_hash,
        "ep_interim_hash": ep_interim_hash,
    }


class TestEncryptedRoundTrip:
    """Regression guard for the encrypted write_bundle / restore_bundle round-trip.

    Verifies that:
    - write_bundle byte-copies age envelopes verbatim (no double-encrypt).
    - Plaintext carve-outs (meta.json, adapter_config.json) remain plaintext
      in the bundle and are restored plaintext.
    - restore_bundle byte-copies encrypted files verbatim (no decrypt-then-write).
    - assert_mode_consistency(scratch, daily_identity_loadable=True) does NOT
      raise after restore (proves the restored store is consistent).
    - find_live_slot resolves via the per-tier registry hash after restore.
    """

    def test_encrypted_round_trip(self, tmp_path, monkeypatch) -> None:
        """Full write_bundle → restore_bundle round-trip with encryption ON."""
        from paramem.backup.encryption import assert_mode_consistency

        # Wire daily identity so write/restore paths use encryption.
        _mint_and_wire_daily_identity(tmp_path, monkeypatch)

        # Build an encrypted source store.
        src = _make_encrypted_source_store(tmp_path)

        # --- write_bundle ---
        bundle_slot = write_bundle(
            config_path=src["config_path"],
            registry_path=src["registry_path"],
            adapter_dirs=src["adapter_dirs"],
            base_dir=src["bundle_base"],
            meta_fields={"tier": "manual", "label": "enc-test"},
            adapter_scope="live",
            speaker_profiles_path=src["speaker_profiles_path"],
        )
        assert bundle_slot.is_dir(), "Bundle slot must exist after write_bundle"

        # Assert that weight blobs and registries are age envelopes in the bundle.
        for rel in [
            "adapters/procedural/adapter_model.safetensors",
            "adapters/procedural/indexed_key_registry.json",
            "adapters/episodic_interim_20260517T1200/adapter_model.safetensors",
            "adapters/episodic_interim_20260517T1200/indexed_key_registry.json",
            "registry/key_metadata.json",
            "speaker_profiles.json",
        ]:
            bundle_file = bundle_slot / rel
            if bundle_file.exists():
                raw = bundle_file.read_bytes()
                assert raw.startswith(_AGE_MAGIC), (
                    f"Expected age envelope in bundle for {rel!r}; first 40 bytes: {raw[:40]!r}"
                )

        # Assert plaintext carve-outs are NOT age envelopes in the bundle.
        for rel in [
            "adapters/procedural/meta.json",
            "adapters/procedural/adapter_config.json",
            "adapters/episodic_interim_20260517T1200/meta.json",
            "adapters/episodic_interim_20260517T1200/adapter_config.json",
            "config/server.yaml",
        ]:
            bundle_file = bundle_slot / rel
            if bundle_file.exists():
                raw = bundle_file.read_bytes()
                assert not raw.startswith(_AGE_MAGIC), (
                    f"Carve-out {rel!r} must be PLAINTEXT in the bundle, not an age envelope. "
                    f"First 40 bytes: {raw[:40]!r}"
                )

        # --- restore_bundle into a scratch dir ---
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        result = restore_bundle(
            bundle_slot,
            data_dir=scratch,
            config_path=config_path,
            restore_config=False,
        )
        assert isinstance(result, RestoreResult)
        assert result.restart_required is True

        # Assert that restored infra files are still age envelopes (byte-faithful).
        for rel_path in [
            "registry/key_metadata.json",
            "speaker_profiles.json",
            "adapters/procedural/indexed_key_registry.json",
        ]:
            restored_file = scratch / rel_path
            if restored_file.exists():
                raw = restored_file.read_bytes()
                assert raw.startswith(_AGE_MAGIC), (
                    f"Restored file {rel_path!r} must be age-encrypted after restore; "
                    f"first 40 bytes: {raw[:40]!r}"
                )

        # Restored weight blobs must also be age envelopes.
        for adapter_name in result.restored_adapters:
            from paramem.memory.interim_adapter import adapter_slot_root_for_name

            tier_root = adapter_slot_root_for_name(scratch / "adapters", adapter_name)
            reg_path = tier_root / "indexed_key_registry.json"
            if reg_path.exists():
                reg_plaintext = None
                if reg_path.read_bytes().startswith(_AGE_MAGIC):
                    from paramem.backup.encryption import read_maybe_encrypted

                    reg_plaintext = read_maybe_encrypted(reg_path)
                else:
                    reg_plaintext = reg_path.read_bytes()
                reg_hash = hashlib.sha256(reg_plaintext).hexdigest()
                from paramem.adapters.manifest import find_live_slot

                live_slot = find_live_slot(tier_root, reg_hash)
                assert live_slot is not None, (
                    f"find_live_slot must resolve {adapter_name!r} slot after encrypted restore"
                )
                # meta.json must be plaintext (carve-out).
                meta_file = live_slot / "meta.json"
                if meta_file.exists():
                    raw = meta_file.read_bytes()
                    assert not raw.startswith(_AGE_MAGIC), (
                        f"meta.json in restored slot {live_slot} must be PLAINTEXT, "
                        "not an age envelope"
                    )
                # adapter_model.safetensors must be an age envelope.
                weight_file = live_slot / "adapter_model.safetensors"
                if weight_file.exists():
                    raw = weight_file.read_bytes()
                    assert raw.startswith(_AGE_MAGIC), (
                        f"adapter_model.safetensors in restored slot {live_slot} must be "
                        "age-encrypted"
                    )

        # assert_mode_consistency on the scratch data_dir — must NOT raise.
        # This is the primary regression guard: it catches the over-encrypt bug
        # (plaintext carve-outs written as age envelopes → FatalConfigError because
        # plaintext mixes with age envelopes) and the under-encrypt bug (age
        # envelopes decrypted and re-written as plaintext → wrong posture).
        from paramem.backup.key_store import DAILY_KEY_PATH_DEFAULT, daily_identity_loadable

        is_loadable = daily_identity_loadable(DAILY_KEY_PATH_DEFAULT)
        assert_mode_consistency(scratch, daily_identity_loadable=is_loadable)
        # If we reach this line, no FatalConfigError was raised — the store is consistent.


# ---------------------------------------------------------------------------
# Candidate sidecar — preserved-not-restored invariant
# ---------------------------------------------------------------------------


class TestCandidateSidecarPreservedNotRestored:
    """write_bundle with candidate_config_path → restore_bundle → sidecar preserved, not restored.

    The sidecar (server.yaml.candidate) must survive in the bundle slot with
    its original bytes intact.  It must NOT be written to the live config path.
    """

    def _build_bundle_with_candidate(self, tmp_path: Path) -> tuple[Path, bytes, bytes]:
        """Build a bundle with a candidate sidecar.

        Returns
        -------
        tuple[Path, bytes, bytes]
            (bundle_slot_dir, live_config_bytes, candidate_bytes)
        """
        src_dict = _make_source_store(tmp_path)

        live_bytes = b"model: mistral\nport: 8420\n"
        src_dict["config_path"].write_bytes(live_bytes)

        candidate_path = tmp_path / "candidate.yaml"
        candidate_bytes = b"model: qwen3\nport: 8420\n"
        candidate_path.write_bytes(candidate_bytes)

        slot = write_bundle(
            config_path=src_dict["config_path"],
            registry_path=src_dict["registry_path"],
            adapter_dirs=src_dict["adapter_dirs"],
            base_dir=src_dict["bundle_base"],
            meta_fields={"tier": "pre_base_swap", "label": "test_candidate"},
            adapter_scope="live",
            speaker_profiles_path=src_dict["speaker_profiles_path"],
            candidate_config_path=candidate_path,
        )
        return slot, live_bytes, candidate_bytes

    def test_candidate_sidecar_intact_after_restore(self, tmp_path) -> None:
        """Sidecar bytes in the bundle slot are unmodified after restore_bundle."""
        bundle_slot, _live_bytes, candidate_bytes = self._build_bundle_with_candidate(tmp_path)

        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: old\n")

        restore_bundle(
            bundle_slot,
            data_dir=scratch,
            config_path=config_path,
            restore_config=True,
        )

        # Sidecar must still exist in the bundle slot with original bytes.
        sidecar_in_bundle = bundle_slot / "server.yaml.candidate"
        assert sidecar_in_bundle.exists(), (
            "server.yaml.candidate must remain in the bundle slot after restore"
        )
        assert sidecar_in_bundle.read_bytes() == candidate_bytes, (
            "server.yaml.candidate bytes in bundle slot were modified by restore_bundle"
        )

    def test_live_config_equals_bundle_live_config_not_candidate(self, tmp_path) -> None:
        """restore_config=True writes the live config (config/server.yaml), not the candidate."""
        bundle_slot, live_bytes, candidate_bytes = self._build_bundle_with_candidate(tmp_path)

        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: old\n")

        restore_bundle(
            bundle_slot,
            data_dir=scratch,
            config_path=config_path,
            restore_config=True,
        )

        restored_config = config_path.read_bytes()
        assert restored_config == live_bytes, (
            "restore_bundle wrote candidate bytes to the live config path — "
            "it must write the bundle's config/server.yaml, not server.yaml.candidate"
        )
        assert restored_config != candidate_bytes, (
            "Candidate bytes must never be restored to the live config path"
        )


# ---------------------------------------------------------------------------
# Orphan interim pruning during restore
# ---------------------------------------------------------------------------


class TestOrphanInterimPruning:
    """restore_bundle prunes on-disk interim families absent from the bundle.

    Boot mounts every internally-consistent interim_* dir unconditionally (gated
    only against the interim's own indexed_key_registry.json), so orphan interims
    inflate the live store.  restore_bundle must remove them and report each
    removal in RestoreResult.pruned_orphans.
    """

    def _add_orphan_interim(
        self,
        data_dir: Path,
        stamp: str,
        active_key_content: bytes | None = None,
    ) -> Path:
        """Plant an interim family dir under data_dir/adapters/episodic/interim_<stamp>/.

        Writes a minimal indexed_key_registry.json using the real KeyRegistry
        on-disk schema (``{"active_keys": [...], ...}``) so KeyRegistry.load can
        parse it.  Returns the family dir path.

        Pass ``active_key_content`` to override with arbitrary bytes (e.g. a
        corrupt payload to exercise the fallback path).
        """
        family_dir = data_dir / "adapters" / "episodic" / f"interim_{stamp}"
        family_dir.mkdir(parents=True, exist_ok=True)
        if active_key_content is None:
            # Real KeyRegistry schema: "active_keys" is a list of key strings.
            # Two known keys so the count assertion can verify a non-zero value.
            active_key_content = json.dumps(
                {
                    "active_keys": ["orphan_key_1", "orphan_key_2"],
                    "fidelity_history": {},
                    "health": None,
                    "stale": {},
                    "simhash": {},
                }
            ).encode("utf-8")
        (family_dir / "indexed_key_registry.json").write_bytes(active_key_content)
        # Minimal slot so the directory looks structurally valid on disk.
        slot_dir = family_dir / "20260614-120000"
        slot_dir.mkdir()
        (slot_dir / "adapter_model.safetensors").write_bytes(b"fake_orphan_weights")
        (slot_dir / "adapter_config.json").write_bytes(b'{"peft_type": "LORA"}')
        meta = {
            "schema_version": 4,
            "name": f"episodic_interim_{stamp}",
            "trained_at": "2026-06-14T12:00:00Z",
            "window_stamp": "",
            "base_model": {
                "repo": "mistralai/Mistral-7B-Instruct-v0.3",
                "sha": "abc",
                "hash": "sha256:abc",
            },
            "tokenizer": {
                "name_or_path": "mistralai/Mistral-7B-Instruct-v0.3",
                "vocab_size": 32000,
                "merges_hash": "d" * 64,
            },
            "lora": {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.0,
                "target_modules": ["q_proj", "v_proj"],
            },
            "registry_sha256": _sha256(active_key_content),
            "key_count": 1,
            "synthesized": False,
        }
        import json as _json

        (slot_dir / "meta.json").write_text(_json.dumps(meta), encoding="utf-8")
        return family_dir

    def test_orphan_interim_removed_after_restore(self, tmp_path) -> None:
        """An interim family absent from the bundle is removed from disk after restore.

        The bundle captures episodic_interim_20260517T1200 (from _make_source_store).
        We plant an additional orphan interim_20260614T1200 in the scratch dir BEFORE
        calling restore_bundle.  After restore, the orphan must not exist on disk.
        """
        bundle_slot, src = _build_bundle(tmp_path)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        # First restore to populate scratch with a known-good state (gives us
        # the episodic adapter so the safety bundle is captured on the second restore).
        restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        # Plant the orphan interim in the already-restored scratch dir.
        orphan_dir = self._add_orphan_interim(scratch, "20260614T1200")
        assert orphan_dir.exists(), "Orphan dir must exist before restore"

        # Second restore: the bundle does NOT include episodic_interim_20260614T1200.
        result = restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        assert not orphan_dir.exists(), "Orphan interim dir must be removed by restore_bundle"
        assert len(result.pruned_orphans) >= 1, (
            "result.pruned_orphans must contain at least one entry"
        )
        orphan_entries = {e["name"]: e for e in result.pruned_orphans}
        assert "episodic_interim_20260614T1200" in orphan_entries, (
            f"pruned_orphans must name the removed interim; got {list(orphan_entries)}"
        )
        assert orphan_entries["episodic_interim_20260614T1200"]["kind"] == "interim", (
            "pruned_orphans entry for an interim family must carry kind='interim'"
        )

    def test_orphan_pruned_entry_has_correct_active_keys(self, tmp_path) -> None:
        """pruned_orphans entry carries the exact active_keys count from the interim's registry.

        Uses the real KeyRegistry on-disk schema (``{"active_keys": [...], ...}``) so
        KeyRegistry.load + list_active produces the correct non-zero count.  The orphan
        is planted with 2 active keys; after restore the pruned_orphans entry must
        record active_keys == 2.
        """
        bundle_slot, src = _build_bundle(tmp_path)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        # Populate scratch so the safety bundle is captured on the second restore.
        restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        # Plant an orphan with 2 real active keys (real KeyRegistry schema).
        self._add_orphan_interim(scratch, "20260614T1200")

        result = restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        pruned = [e for e in result.pruned_orphans if e["name"] == "episodic_interim_20260614T1200"]
        assert len(pruned) == 1, (
            f"Expected one pruned_orphans entry for the orphan; got {result.pruned_orphans}"
        )
        assert pruned[0]["active_keys"] == 2, (
            f"pruned_orphans[*].active_keys must equal 2 (the number of active keys planted); "
            f"got {pruned[0]['active_keys']}"
        )
        assert pruned[0]["kind"] == "interim", (
            f"pruned_orphans entry for interim must carry kind='interim'; got {pruned[0]['kind']}"
        )

    def test_orphan_pruned_entry_active_keys_fallback_on_corrupt_registry(self, tmp_path) -> None:
        """Unreadable/corrupt registry → active_keys count falls back to 0.

        When the indexed_key_registry.json cannot be parsed (e.g. it contains
        invalid JSON), the except branch in the prune sub-step must record
        active_keys=0 rather than aborting restore.
        """
        bundle_slot, src = _build_bundle(tmp_path)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        # Populate scratch so the safety bundle is captured on the second restore.
        restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        # Plant an orphan whose registry is corrupt (not valid JSON).
        self._add_orphan_interim(
            scratch, "20260614T1200", active_key_content=b"not-a-valid-registry"
        )

        result = restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        pruned = [e for e in result.pruned_orphans if e["name"] == "episodic_interim_20260614T1200"]
        assert len(pruned) == 1, (
            f"Expected one pruned_orphans entry for the corrupt-registry orphan; "
            f"got {result.pruned_orphans}"
        )
        assert pruned[0]["active_keys"] == 0, (
            f"Corrupt registry must fall back to active_keys=0; got {pruned[0]['active_keys']}"
        )
        assert pruned[0]["kind"] == "interim", (
            "pruned_orphans entry for interim must carry kind='interim'; "
            f"got {pruned[0].get('kind')}"
        )

    def test_bundle_interim_survives_restore(self, tmp_path) -> None:
        """An interim family present in the bundle is NOT pruned — it survives restore.

        _make_source_store plants episodic_interim_20260517T1200 and the bundle
        captures it under adapter_scope='live'.  After restore, that family must
        still exist on disk.
        """
        bundle_slot, src = _build_bundle(tmp_path)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        result = restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        # episodic_interim_20260517T1200 is in the bundle — must be restored, not pruned.
        interim_dir = scratch / "adapters" / "episodic" / "interim_20260517T1200"
        assert interim_dir.is_dir(), "Interim family present in bundle must survive restore_bundle"
        pruned_names = [e["name"] for e in result.pruned_orphans]
        assert "episodic_interim_20260517T1200" not in pruned_names, (
            "Bundle-included interim must not appear in pruned_orphans"
        )

    def test_orphan_main_tier_removed_after_restore(self, tmp_path) -> None:
        """A main tier absent from the bundle is removed from disk after restore.

        The bundle captures episodic (interim) and procedural.  We add a
        'procedural' tier dir to the scratch dir with a real registry (≥1 active
        key) BEFORE calling restore_bundle.  However, the bundle DOES include
        procedural — so we build a bundle WITHOUT procedural to exercise the
        orphan-main-tier path.

        Strategy: build a bundle that only captures the episodic interim (not
        procedural), then restore into a scratch that has a stale procedural dir
        on disk.  After restore, adapters/procedural/ must not exist and
        pruned_orphans must carry {"name": "procedural", "kind": "main", "active_keys": N}.
        """
        import json as _json

        # Build a bundle from a source that has only the episodic interim (no procedural).
        src = tmp_path / "ep_only_src"
        src.mkdir()
        config_path_src = src / "server.yaml"
        config_path_src.write_bytes(b"model: mistral\nport: 8420\n")

        registry_dir = src / "registry"
        registry_dir.mkdir()
        registry_path_src = registry_dir / "key_metadata.json"
        registry_path_src.write_bytes(b'{"speakers": {}, "cycles": []}')

        speaker_profiles_src = src / "speaker_profiles.json"
        speaker_profiles_src.write_bytes(b'{"profiles": []}')

        adapters_base = src / "adapters"
        ep_interim_content = b'{"keys": {"ep_key": "val"}}'
        ep_interim_hash = _sha256(ep_interim_content)
        ep_dir = adapters_base / "episodic"
        ep_dir.mkdir(parents=True)
        interim_family_dir = ep_dir / "interim_20260515T0900"
        interim_family_dir.mkdir()
        _make_adapter_slot(
            interim_family_dir,
            "20260515-090000",
            ep_interim_hash,
            adapter_name="episodic_interim_20260515T0900",
            weight_bytes=b"ep_weights",
        )
        (interim_family_dir / "indexed_key_registry.json").write_bytes(ep_interim_content)

        bundle_base = src / "backups" / "snapshot"
        bundle_base.mkdir(parents=True)

        # Bundle captures only episodic (via interim); procedural is NOT in the bundle.
        bundle_slot = write_bundle(
            config_path=config_path_src,
            registry_path=registry_path_src,
            adapter_dirs={"episodic": ep_dir},
            base_dir=bundle_base,
            meta_fields={"tier": "manual", "label": "ep-only-test"},
            adapter_scope="live",
            speaker_profiles_path=speaker_profiles_src,
        )

        # Build the scratch dir: plant a stale procedural tier with ≥1 active key.
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path_scratch = scratch / "server.yaml"
        config_path_scratch.write_bytes(b"model: mistral\n")

        proc_content = _json.dumps(
            {
                "active_keys": ["proc_key_1", "proc_key_2"],
                "fidelity_history": {},
                "health": None,
                "stale": {},
                "simhash": {},
            }
        ).encode("utf-8")
        proc_hash = _sha256(proc_content)
        proc_dir_scratch = scratch / "adapters" / "procedural"
        proc_dir_scratch.mkdir(parents=True)
        (proc_dir_scratch / "indexed_key_registry.json").write_bytes(proc_content)
        proc_slot = proc_dir_scratch / "20260520-100000"
        proc_slot.mkdir()
        _make_adapter_slot(
            proc_dir_scratch, "20260520-100000", proc_hash, adapter_name="procedural"
        )

        assert proc_dir_scratch.is_dir(), "Stale procedural dir must exist before restore"

        # Restore: procedural is NOT in the bundle → must be pruned.
        result = restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path_scratch)

        assert not proc_dir_scratch.exists(), (
            "adapters/procedural/ must be removed after restore when absent from the bundle"
        )
        pruned_map = {e["name"]: e for e in result.pruned_orphans}
        assert "procedural" in pruned_map, (
            f"pruned_orphans must contain an entry for 'procedural'; got {list(pruned_map)}"
        )
        assert pruned_map["procedural"]["kind"] == "main", (
            "pruned 'procedural' entry must carry kind='main'; "
            f"got {pruned_map['procedural']['kind']}"
        )
        assert pruned_map["procedural"]["active_keys"] == 2, (
            "pruned 'procedural' active_keys must be 2; "
            f"got {pruned_map['procedural']['active_keys']}"
        )

    def test_main_tier_in_bundle_survives_restore(self, tmp_path) -> None:
        """A main tier present in the bundle is NOT pruned — its tier dir survives restore.

        _make_source_store captures procedural as a main tier.  After restore into
        a clean scratch dir, adapters/procedural/ must still be present and must
        NOT appear in pruned_orphans.
        """
        bundle_slot, src = _build_bundle(tmp_path)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        result = restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        # procedural is in the bundle — its tier dir must exist after restore.
        proc_dir = scratch / "adapters" / "procedural"
        assert proc_dir.is_dir(), (
            "adapters/procedural/ must exist after restore when procedural is in the bundle"
        )
        pruned_names = [e["name"] for e in result.pruned_orphans]
        assert "procedural" not in pruned_names, (
            "Bundle-included main tier 'procedural' must not appear in pruned_orphans"
        )

    # ----- NEW scenario 4: within-tier stale slot cleanup --------------------

    def test_within_tier_stale_slot_removed(self, tmp_path) -> None:
        """A stale slot inside a tier that IS in the bundle is removed by the clean-slate sweep.

        Scenario: procedural is in the bundle.  Before calling restore, we plant a
        stale old slot dir (STALE_SLOT) inside adapters/procedural/.  After restore:
        - adapters/procedural/ still exists (tier is in the bundle).
        - The freshly-written slot exists.
        - The stale old slot is GONE.
        - The stale slot does NOT appear in pruned_orphans (it is within-tier hygiene,
          not an orphan adapter).
        """
        bundle_slot, src = _build_bundle(tmp_path)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        # First restore: populate scratch so we have a base state.
        restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        # Plant a stale slot in the procedural tier.
        proc_dir = scratch / "adapters" / "procedural"
        assert proc_dir.is_dir(), "procedural tier must exist after first restore"
        stale_slot = proc_dir / "20220101-000000"
        stale_slot.mkdir()
        (stale_slot / "meta.json").write_bytes(b'{"schema_version": 4, "registry_sha256": "old"}')
        (stale_slot / "adapter_model.safetensors").write_bytes(b"stale_weights")
        (stale_slot / "adapter_config.json").write_bytes(b'{"peft_type": "LORA"}')
        assert stale_slot.is_dir(), "Stale slot must exist before second restore"

        # Second restore.
        result = restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        # The stale slot must be gone.
        assert not stale_slot.exists(), (
            "Stale slot in a bundle-included tier must be removed by the clean-slate sweep"
        )
        # procedural tier dir still exists.
        assert proc_dir.is_dir(), "procedural tier dir must survive restore"
        # Not listed in pruned_orphans — within-tier hygiene is not adapter-level.
        pruned_names = [e["name"] for e in result.pruned_orphans]
        assert "procedural" not in pruned_names, (
            "Stale slot cleanup inside a bundle tier must NOT appear in pruned_orphans"
        )

    # ----- NEW scenario 5: episodic-as-interim --------------------------------

    def test_episodic_as_interim_stale_main_registry_removed(self, tmp_path) -> None:
        """Episodic-as-interim: stale main slot + main registry removed; interim family kept.

        The bundle captures episodic_interim_<stamp> but NO finalized episodic main.
        The scratch dir has a stale main-episodic slot + main indexed_key_registry.json
        under adapters/episodic/.

        After restore:
        - adapters/episodic/ still exists (it hosts the bundle's interim family).
        - The restored interim_<stamp>/ family is present.
        - The stale main episodic slot dir is GONE.
        - The stale adapters/episodic/indexed_key_registry.json is GONE.

        This is the critical edge that prevents a stale main-episodic from being
        mounted at boot after an episodic-as-interim restore.
        """
        import json as _json

        # Build a bundle from _make_source_store: captures episodic_interim_20260517T1200
        # (no main episodic) + procedural.
        bundle_slot, src = _build_bundle(tmp_path)

        # Build scratch: plant a stale main-episodic slot + registry.
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        ep_dir = scratch / "adapters" / "episodic"
        ep_dir.mkdir(parents=True)

        # Stale main-episodic slot dir (looks like a finalized slot).
        stale_main_slot = ep_dir / "20200101-000000"
        stale_main_slot.mkdir()
        (stale_main_slot / "meta.json").write_bytes(
            b'{"schema_version": 4, "registry_sha256": "stale_ep"}'
        )
        (stale_main_slot / "adapter_model.safetensors").write_bytes(b"stale_ep_weights")
        (stale_main_slot / "adapter_config.json").write_bytes(b'{"peft_type": "LORA"}')

        # Stale main-episodic tier-level registry.
        stale_main_reg = ep_dir / "indexed_key_registry.json"
        stale_main_reg.write_bytes(
            _json.dumps(
                {
                    "active_keys": ["ep_stale_key"],
                    "fidelity_history": {},
                    "health": None,
                    "stale": {},
                    "simhash": {},
                }
            ).encode("utf-8")
        )

        assert stale_main_slot.is_dir(), "Stale main episodic slot must exist before restore"
        assert stale_main_reg.exists(), "Stale main episodic registry must exist before restore"

        # Restore: bundle has episodic_interim_20260517T1200 (no main episodic).
        result = restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        # adapters/episodic/ must still exist (hosts the interim family).
        assert ep_dir.is_dir(), "adapters/episodic/ must exist after restore (hosts interims)"

        # The interim family must be present.
        interim_dir = ep_dir / "interim_20260517T1200"
        assert interim_dir.is_dir(), "Restored interim family must be present under episodic/"

        # The stale main-episodic slot must be gone.
        assert not stale_main_slot.exists(), (
            "Stale main-episodic slot must be removed when bundle has no finalized episodic"
        )

        # The stale main-episodic tier registry must be gone.
        assert not stale_main_reg.exists(), (
            "Stale adapters/episodic/indexed_key_registry.json must be removed "
            "when bundle has episodic_interim_* but no finalized main episodic — "
            "otherwise boot would mount a stale main-episodic"
        )

        # Episodic is not in pruned_orphans as an orphan main tier (the tier dir survives).
        pruned_names = [e["name"] for e in result.pruned_orphans]
        assert "episodic" not in pruned_names, (
            "episodic tier dir survives (it hosts interims); must not appear in pruned_orphans"
        )

    def test_combined_main_and_interim_same_tier(self, tmp_path) -> None:
        """Tier with both a finalized main slot AND a bundle interim family: both survive.

        This exercises the branch where keep_main=True AND interims_here is non-empty
        simultaneously — the tier-child loop must skip the main slot AND recurse into
        the interim family in the same tier pass.

        Setup (bundle source):
        - episodic: finalized main slot + an interim family (episodic_interim_20260517T1200).
        - procedural: finalized main slot.

        On disk before the second restore (in scratch), also plant:
        - A stale old main slot under episodic/ (e.g. 20200101-000000).
        - A stale orphan interim under episodic/ (interim_99999T0000).
        - A scratch file at the episodic tier root.

        After restore:
        - adapters/episodic/ still contains exactly: the freshly-written main slot +
          indexed_key_registry.json + the restored interim family.
        - The stale old main slot is GONE.
        - The orphan interim is GONE (in pruned_orphans).
        - The scratch file is GONE.
        - The bundle interim family (interim_20260517T1200) survives with only its
          new slot + registry inside (no stale children).
        """
        import json as _json

        # Build a source that has BOTH a finalized episodic main slot AND an interim.
        src = tmp_path / "combined_src"
        src.mkdir()
        config_path_src = src / "server.yaml"
        config_path_src.write_bytes(b"model: mistral\nport: 8420\n")

        registry_dir = src / "registry"
        registry_dir.mkdir()
        registry_path_src = registry_dir / "key_metadata.json"
        registry_path_src.write_bytes(b'{"speakers": {}, "cycles": []}')

        speaker_profiles_src = src / "speaker_profiles.json"
        speaker_profiles_src.write_bytes(b'{"profiles": []}')

        adapters_base_src = src / "adapters"

        # Episodic: finalized main slot
        ep_main_content = b'{"keys": {"ep_main_key": 1}}'
        ep_main_hash = _sha256(ep_main_content)
        ep_dir = adapters_base_src / "episodic"
        ep_dir.mkdir(parents=True)
        _make_adapter_slot(
            ep_dir,
            "20260520-090000",
            ep_main_hash,
            adapter_name="episodic",
            weight_bytes=b"ep_main_weights",
        )
        (ep_dir / "indexed_key_registry.json").write_bytes(ep_main_content)

        # Episodic: also has an interim family
        ep_interim_content = b'{"keys": {"ep_interim_key": "val"}}'
        ep_interim_hash = _sha256(ep_interim_content)
        interim_family_dir = ep_dir / "interim_20260517T1200"
        interim_family_dir.mkdir()
        _make_adapter_slot(
            interim_family_dir,
            "20260517-180430",
            ep_interim_hash,
            adapter_name="episodic_interim_20260517T1200",
            weight_bytes=b"ep_interim_weights",
        )
        (interim_family_dir / "indexed_key_registry.json").write_bytes(ep_interim_content)

        # Procedural: finalized main slot
        proc_content = b'{"keys": {"proc_key": 1}}'
        proc_hash = _sha256(proc_content)
        proc_dir = adapters_base_src / "procedural"
        proc_dir.mkdir(parents=True)
        _make_adapter_slot(proc_dir, "20260520-100000", proc_hash, adapter_name="procedural")
        (proc_dir / "indexed_key_registry.json").write_bytes(proc_content)

        bundle_base = src / "backups" / "snapshot"
        bundle_base.mkdir(parents=True)

        bundle_slot = write_bundle(
            config_path=config_path_src,
            registry_path=registry_path_src,
            adapter_dirs={"episodic": ep_dir, "procedural": proc_dir},
            base_dir=bundle_base,
            meta_fields={"tier": "manual", "label": "combined-test"},
            adapter_scope="live",
            speaker_profiles_path=speaker_profiles_src,
        )

        # Build a scratch dir that has stale content to exercise the sweep.
        scratch = tmp_path / "scratch_combined"
        scratch.mkdir()
        config_path_scratch = scratch / "server.yaml"
        config_path_scratch.write_bytes(b"model: mistral\n")

        ep_dir_scratch = scratch / "adapters" / "episodic"
        ep_dir_scratch.mkdir(parents=True)

        # Plant a stale old main slot.
        stale_main = ep_dir_scratch / "20200101-000000"
        stale_main.mkdir()
        (stale_main / "meta.json").write_bytes(
            b'{"schema_version": 4, "registry_sha256": "old_ep"}'
        )
        (stale_main / "adapter_model.safetensors").write_bytes(b"stale_ep_weights")
        (stale_main / "adapter_config.json").write_bytes(b'{"peft_type": "LORA"}')

        # Plant an orphan interim family (not in the bundle).
        orphan_interim = ep_dir_scratch / "interim_99999T0000"
        orphan_interim.mkdir()
        (orphan_interim / "indexed_key_registry.json").write_bytes(
            _json.dumps(
                {
                    "active_keys": ["orphan_key"],
                    "fidelity_history": {},
                    "health": None,
                    "stale": {},
                    "simhash": {},
                }
            ).encode("utf-8")
        )
        orphan_slot = orphan_interim / "20260614-120000"
        orphan_slot.mkdir()
        (orphan_slot / "adapter_model.safetensors").write_bytes(b"orphan_weights")

        # Plant a stale interim family with an extra stale child inside — the
        # bundle's interim_20260517T1200 family should survive with cleanup inside.
        ep_interim_scratch = ep_dir_scratch / "interim_20260517T1200"
        ep_interim_scratch.mkdir()
        stale_inner_slot = ep_interim_scratch / "20200601-000000"
        stale_inner_slot.mkdir()
        (stale_inner_slot / "meta.json").write_bytes(b'{"schema_version": 4}')

        # Plant a scratch file at the tier root.
        scratch_file = ep_dir_scratch / "scratch_debug.json"
        scratch_file.write_bytes(b'{"debug": true}')

        result = restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path_scratch)

        # The episodic tier dir must survive (both main + interim in bundle).
        assert ep_dir_scratch.is_dir(), "adapters/episodic/ must survive restore"

        # The freshly-written main slot must be present.
        from paramem.adapters.manifest import find_live_slot

        ep_reg = (ep_dir_scratch / "indexed_key_registry.json").read_bytes()
        ep_live = find_live_slot(ep_dir_scratch, _sha256(ep_reg))
        assert ep_live is not None, "find_live_slot must resolve the restored episodic main slot"
        assert ep_live.is_dir()

        # The bundle interim family must be present.
        interim_dir = ep_dir_scratch / "interim_20260517T1200"
        assert interim_dir.is_dir(), "Bundle interim family must survive restore"
        assert (interim_dir / "indexed_key_registry.json").exists(), (
            "Interim family's indexed_key_registry.json must be present after restore"
        )

        # The stale old main slot must be gone.
        assert not stale_main.exists(), "Stale main episodic slot must be removed by sweep"

        # The orphan interim must be gone and appear in pruned_orphans.
        assert not orphan_interim.exists(), "Orphan interim family must be removed by sweep"
        pruned_names = {e["name"]: e for e in result.pruned_orphans}
        assert "episodic_interim_99999T0000" in pruned_names, (
            f"Orphan interim must appear in pruned_orphans; got {list(pruned_names)}"
        )
        assert pruned_names["episodic_interim_99999T0000"]["kind"] == "interim"

        # The scratch file at the tier root must be gone.
        assert not scratch_file.exists(), "Scratch file at episodic tier root must be removed"

        # The stale inner slot inside the bundle interim family must be gone.
        assert not stale_inner_slot.exists(), (
            "Stale inner slot inside bundle interim family must be removed by sweep"
        )

        # episodic must NOT appear in pruned_orphans (the tier dir survived).
        assert "episodic" not in pruned_names, (
            "episodic tier dir survives; must not appear in pruned_orphans"
        )

    def test_total_tree_equals_bundle_set(self, tmp_path) -> None:
        """After restore, the set of top-level entries under adapters/ equals the bundle set.

        _make_source_store produces a bundle with:
        - procedural (finalized main) → adapters/procedural/
        - episodic_interim_20260517T1200 → adapters/episodic/  (interim host)

        So after restore the top-level entries under adapters/ must be exactly
        {"episodic", "procedural"} — no more, no less.

        We plant an extra stale tier ("semantic") and a legacy flat dir
        ("episodic_interim_LEGACY") in scratch before restore to exercise the sweep.
        """
        import json as _json

        bundle_slot, src = _build_bundle(tmp_path)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config_path = scratch / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")

        # Plant orphan semantic tier.
        sem_dir = scratch / "adapters" / "semantic"
        sem_dir.mkdir(parents=True)
        (sem_dir / "indexed_key_registry.json").write_bytes(
            _json.dumps(
                {
                    "active_keys": ["sem_key"],
                    "fidelity_history": {},
                    "health": None,
                    "stale": {},
                    "simhash": {},
                }
            ).encode("utf-8")
        )
        sem_slot = sem_dir / "20200101-000000"
        sem_slot.mkdir()
        _make_adapter_slot(sem_dir, "20200101-000000", _sha256(b"{}"), adapter_name="semantic")

        # Plant a legacy flat dir at the adapters/ root (pre-hierarchy-refactor layout).
        legacy_flat = scratch / "adapters" / "episodic_interim_LEGACY"
        legacy_flat.mkdir(parents=True)
        (legacy_flat / "indexed_key_registry.json").write_bytes(b'{"active_keys": []}')

        assert sem_dir.is_dir()
        assert legacy_flat.is_dir()

        result = restore_bundle(bundle_slot, data_dir=scratch, config_path=config_path)

        adapters_base = scratch / "adapters"
        top_level_names = {p.name for p in adapters_base.iterdir() if not p.name.startswith(".")}
        # Exactly {episodic, procedural} — semantic and legacy flat are gone.
        assert top_level_names == {"episodic", "procedural"}, (
            f"Top-level entries under adapters/ must equal exactly the bundle set "
            f"{{episodic, procedural}}; got {top_level_names}"
        )
        # semantic is in pruned_orphans (orphan main tier).
        pruned_names = {e["name"] for e in result.pruned_orphans}
        assert "semantic" in pruned_names, "Orphan 'semantic' tier must appear in pruned_orphans"
        # Legacy flat dir is NOT reported as a named adapter in pruned_orphans
        # (it is not a recognised adapter family — logged only).
        assert legacy_flat.name not in pruned_names, (
            "Legacy flat top-level dir must NOT appear in pruned_orphans (not a named adapter)"
        )
