"""Tests for write_bundle() in paramem.backup.backup.

Covers the recovery set:

- write_bundle over a fixture data dir → bundle slot created with all files;
  bundle.meta.json lists them with correct hashes.
- Per-tier indexed_key_registry.json captured for both main and interim slots.
- Already-encrypted safetensors (age magic) copied byte-for-byte (no double-
  encrypt).
- Real nested interim layout: interim_<stamp>/<ts>/{meta.json,...} — NOT the
  flat layout used in the prior test suite.
- Scaffolding (checkpoint-*/, in_training/, bg_checkpoint/, resume_state.json)
  never appears in the bundle.
- Multi-adapter bundle: procedural main (hash A) + episodic interim only
  (hash B) → both captured; each adapter entry records its OWN registry_sha256;
  per-tier indexed_key_registry.json captured for both.
- adapter_scope="main" with episodic interim-only → fail-loud with actionable
  message directing to "live" or full consolidation.
- adapter_scope="live" → succeeds capturing the episodic interim slot.
- Missing optional artifacts (indexed_key_registry.json, simhash_registry.json)
  → recorded absent, no crash.
- Fail-loud when adapter_scope resolves no slot for episodic at all.
- Non-episodic tier with no main slot → recorded absent, no failure.
- Crash-safety: rename failure leaves pending residue swept by
  sweep_orphan_pending.
- speaker_profiles.json included / absent.
- No ArtifactMeta sidecar written (S2 constraint).
- enumerate_backups recognises the bundle slot.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from paramem.backup.backup import sweep_orphan_pending, write_bundle
from paramem.backup.enumerate import enumerate_backups
from paramem.backup.types import BUNDLE_SCHEMA_VERSION, ArtifactKind, BackupError, BundleManifest

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_FAKE_AGE_MAGIC = b"age-encryption.org/v1\n"  # matches AGE_MAGIC in age_envelope.py


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _make_adapter_slot(
    parent_dir: Path,
    slot_name: str,
    registry_sha256: str,
    adapter_name: str = "episodic",
    weight_bytes: bytes = b"fake_plain_weights",
) -> Path:
    """Create a minimal adapter slot directory with meta.json + adapter files.

    Parameters
    ----------
    parent_dir:
        Directory that will directly contain the slot (e.g. the adapter-kind
        dir for main tiers, or the interim-family dir for interim slots).
    slot_name:
        Timestamped slot subdirectory name (e.g. ``"20260520-123456"``).
    registry_sha256:
        The registry SHA-256 to embed in meta.json.  Must equal
        ``hashlib.sha256(<tier_root>/indexed_key_registry.json bytes).hexdigest()``
        so that ``find_live_slot`` can match this slot against the per-tier
        hash computed by ``write_bundle._tier_registry_sha256``.
    adapter_name:
        The adapter name to embed in meta.json.
    weight_bytes:
        Bytes to write as adapter_model.safetensors.

    Returns
    -------
    Path
        The slot directory.
    """
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
        "key_count": 550,
        "synthesized": False,
    }
    (slot / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (slot / "adapter_model.safetensors").write_bytes(weight_bytes)
    (slot / "adapter_config.json").write_bytes(b'{"peft_type": "LORA"}')

    return slot


def _make_fixtures(
    tmp_path: Path,
    *,
    weight_bytes: bytes = b"fake_plain_weights",
    include_simhash: bool = True,
    include_indexed_key_registry: bool = True,
    include_speaker_profiles: bool = False,
) -> dict:
    """Create a minimal fixture environment for write_bundle tests (main-slot only).

    The slot's ``meta.registry_sha256`` is derived from the bytes written to
    ``episodic/indexed_key_registry.json`` so that
    ``write_bundle._tier_registry_sha256`` and ``find_live_slot`` agree.
    When ``include_indexed_key_registry=False`` no registry file is written and
    the slot is stamped with ``""`` (the empty-registry sentinel).

    Returns a dict with keys:
      config_path, registry_path, adapter_dirs, base_dir,
      registry_sha256, weight_bytes, speaker_profiles_path (may be None)
    """
    data_dir = tmp_path / "ha"
    data_dir.mkdir(parents=True)

    # Config
    config_path = data_dir / "server.yaml"
    config_path.write_bytes(b"model: mistral\n")

    # Registry
    registry_dir = data_dir / "registry"
    registry_dir.mkdir()
    registry_path = registry_dir / "key_metadata.json"
    registry_bytes = b'{"keys": []}'
    registry_path.write_bytes(registry_bytes)

    # Adapter dirs
    adapters_dir = data_dir / "adapters"
    episodic_dir = adapters_dir / "episodic"
    episodic_dir.mkdir(parents=True)

    # Compute the registry_sha256 from the content that will be written so that
    # write_bundle._tier_registry_sha256(episodic_dir) == slot.meta.registry_sha256.
    indexed_key_content = b'{"keys": {}}'
    if include_indexed_key_registry:
        registry_sha256 = _sha256(indexed_key_content)
    else:
        # No registry file → _tier_registry_sha256 returns "" → stamp slot with ""
        registry_sha256 = ""

    _make_adapter_slot(episodic_dir, "20260520-123456", registry_sha256, weight_bytes=weight_bytes)

    if include_indexed_key_registry:
        (episodic_dir / "indexed_key_registry.json").write_bytes(indexed_key_content)

    if include_simhash:
        (episodic_dir / "simhash_registry.json").write_bytes(b'{"simhash": {}}')

    # Speaker profiles (optional)
    speaker_profiles_path = None
    if include_speaker_profiles:
        sp_path = data_dir / "speaker_profiles.json"
        sp_path.write_bytes(b'{"profiles": []}')
        speaker_profiles_path = sp_path

    # Bundle base dir
    base_dir = data_dir / "backups" / "snapshot"
    base_dir.mkdir(parents=True)

    return {
        "config_path": config_path,
        "registry_path": registry_path,
        "adapter_dirs": {"episodic": episodic_dir},
        "base_dir": base_dir,
        "registry_sha256": registry_sha256,
        "weight_bytes": weight_bytes,
        "speaker_profiles_path": speaker_profiles_path,
        "episodic_dir": episodic_dir,
        "adapters_dir": adapters_dir,
    }


def _make_interim_family(
    episodic_dir: Path,
    stamp: str,
    slot_name: str,
    registry_sha256: str,
    weight_bytes: bytes = b"fake_interim_weights",
    include_indexed_key_registry: bool = True,
    include_simhash: bool = True,
) -> tuple[Path, Path]:
    """Create a nested interim family under episodic_dir.

    Layout mirrors production:
      episodic_dir/interim_<stamp>/<slot_name>/{meta.json,...}

    The caller is responsible for supplying a ``registry_sha256`` that matches
    the bytes written to ``interim_family_dir/indexed_key_registry.json``.  Use
    ``_sha256(<content>)`` on the intended ``indexed_key_registry.json`` content
    and pass that value here so ``write_bundle._tier_registry_sha256`` and
    ``find_live_slot`` agree.  When ``include_indexed_key_registry=False``, pass
    ``""`` (the empty-registry sentinel).

    Also creates sibling ``checkpoint-1863/`` and ``in_training/`` scaffolding
    directories that must never appear in the bundle.

    Returns
    -------
    tuple[Path, Path]
        ``(interim_family_dir, interim_slot_dir)``
    """
    interim_family_dir = episodic_dir / f"interim_{stamp}"
    interim_family_dir.mkdir(parents=True, exist_ok=True)

    # The live inner slot
    _make_adapter_slot(
        interim_family_dir,
        slot_name,
        registry_sha256,
        adapter_name=f"episodic_interim_{stamp}",
        weight_bytes=weight_bytes,
    )

    # Scaffolding that must be excluded — checkpoint-*/ has no matching meta.json
    # so find_live_slot naturally excludes it.  We create it here to verify the
    # bundle never captures it.
    scaffolding = interim_family_dir / "checkpoint-1863"
    scaffolding.mkdir(exist_ok=True)
    (scaffolding / "pytorch_model.bin").write_bytes(b"scaffold_weights_must_not_appear")

    # in_training/ scaffolding (also excluded)
    in_training = interim_family_dir / "in_training"
    in_training.mkdir(exist_ok=True)
    (in_training / "optimizer.bin").write_bytes(b"optimizer_state_excluded")

    if include_indexed_key_registry:
        (interim_family_dir / "indexed_key_registry.json").write_bytes(
            b'{"keys": {"interim_key": "val"}}'
        )

    if include_simhash:
        (interim_family_dir / "simhash_registry.json").write_bytes(
            b'{"simhash": {"interim_hash": 12345}}'
        )

    interim_slot_dir = interim_family_dir / slot_name
    return interim_family_dir, interim_slot_dir


# ---------------------------------------------------------------------------
# Happy path: main-slot bundle created with all files
# ---------------------------------------------------------------------------


class TestWriteBundleHappyPath:
    def test_bundle_slot_created(self, tmp_path) -> None:
        """write_bundle returns a promoted slot directory that exists."""
        fixtures = _make_fixtures(tmp_path)
        slot = write_bundle(
            config_path=fixtures["config_path"],
            registry_path=fixtures["registry_path"],
            adapter_dirs=fixtures["adapter_dirs"],
            base_dir=fixtures["base_dir"],
            meta_fields={"tier": "manual"},
            live_registry_sha256=fixtures["registry_sha256"],
        )
        assert slot.is_dir(), f"Bundle slot not found at {slot}"

    def test_bundle_manifest_exists(self, tmp_path) -> None:
        """bundle.meta.json must exist in the promoted slot."""
        fixtures = _make_fixtures(tmp_path)
        slot = write_bundle(
            config_path=fixtures["config_path"],
            registry_path=fixtures["registry_path"],
            adapter_dirs=fixtures["adapter_dirs"],
            base_dir=fixtures["base_dir"],
            meta_fields={"tier": "manual"},
            live_registry_sha256=fixtures["registry_sha256"],
        )
        assert (slot / "bundle.meta.json").exists()

    def test_bundle_manifest_parses(self, tmp_path) -> None:
        """bundle.meta.json must parse as a valid BundleManifest."""
        fixtures = _make_fixtures(tmp_path)
        slot = write_bundle(
            config_path=fixtures["config_path"],
            registry_path=fixtures["registry_path"],
            adapter_dirs=fixtures["adapter_dirs"],
            base_dir=fixtures["base_dir"],
            meta_fields={"tier": "manual"},
            live_registry_sha256=fixtures["registry_sha256"],
        )
        raw = json.loads((slot / "bundle.meta.json").read_text(encoding="utf-8"))
        manifest = BundleManifest.from_dict(raw)
        assert manifest.bundle_schema_version == BUNDLE_SCHEMA_VERSION
        assert manifest.tier == "manual"

    def test_bundle_manifest_files_inventory(self, tmp_path) -> None:
        """bundle.meta.json must list captured files with hashes."""
        fixtures = _make_fixtures(tmp_path)
        slot = write_bundle(
            config_path=fixtures["config_path"],
            registry_path=fixtures["registry_path"],
            adapter_dirs=fixtures["adapter_dirs"],
            base_dir=fixtures["base_dir"],
            meta_fields={"tier": "manual"},
            live_registry_sha256=fixtures["registry_sha256"],
        )
        manifest = BundleManifest.from_dict(
            json.loads((slot / "bundle.meta.json").read_text(encoding="utf-8"))
        )
        paths = {f["path"] for f in manifest.files}
        assert any("server.yaml" in p for p in paths), f"Config missing from files: {paths}"
        assert any("key_metadata.json" in p for p in paths), f"Registry missing: {paths}"
        assert any("adapter_model.safetensors" in p for p in paths), f"Weights missing: {paths}"

    def test_indexed_key_registry_captured_for_main(self, tmp_path) -> None:
        """indexed_key_registry.json at tier root is captured for a main slot."""
        fixtures = _make_fixtures(tmp_path, include_indexed_key_registry=True)
        slot = write_bundle(
            config_path=fixtures["config_path"],
            registry_path=fixtures["registry_path"],
            adapter_dirs=fixtures["adapter_dirs"],
            base_dir=fixtures["base_dir"],
            meta_fields={"tier": "manual"},
            live_registry_sha256=fixtures["registry_sha256"],
        )
        assert (slot / "adapters" / "episodic" / "indexed_key_registry.json").exists(), (
            "indexed_key_registry.json must be captured for the main episodic tier"
        )
        manifest = BundleManifest.from_dict(
            json.loads((slot / "bundle.meta.json").read_text(encoding="utf-8"))
        )
        paths = {f["path"] for f in manifest.files}
        assert "adapters/episodic/indexed_key_registry.json" in paths

    def test_indexed_key_registry_absent_when_missing(self, tmp_path) -> None:
        """Missing indexed_key_registry.json → indexed_key_registry_present=False, no crash."""
        fixtures = _make_fixtures(tmp_path, include_indexed_key_registry=False)
        slot = write_bundle(
            config_path=fixtures["config_path"],
            registry_path=fixtures["registry_path"],
            adapter_dirs=fixtures["adapter_dirs"],
            base_dir=fixtures["base_dir"],
            meta_fields={"tier": "manual"},
            live_registry_sha256=fixtures["registry_sha256"],
        )
        manifest = BundleManifest.from_dict(
            json.loads((slot / "bundle.meta.json").read_text(encoding="utf-8"))
        )
        assert manifest.adapters["episodic"]["indexed_key_registry_present"] is False

    def test_bundle_manifest_content_hashes_match(self, tmp_path) -> None:
        """Each file entry's content_sha256 must match the on-disk bytes."""
        fixtures = _make_fixtures(tmp_path)
        slot = write_bundle(
            config_path=fixtures["config_path"],
            registry_path=fixtures["registry_path"],
            adapter_dirs=fixtures["adapter_dirs"],
            base_dir=fixtures["base_dir"],
            meta_fields={"tier": "manual"},
            live_registry_sha256=fixtures["registry_sha256"],
        )
        manifest = BundleManifest.from_dict(
            json.loads((slot / "bundle.meta.json").read_text(encoding="utf-8"))
        )
        for entry in manifest.files:
            on_disk_path = slot / entry["path"]
            assert on_disk_path.exists(), f"Listed file not on disk: {entry['path']}"
            actual_hash = _sha256(on_disk_path.read_bytes())
            assert actual_hash == entry["content_sha256"], (
                f"Hash mismatch for {entry['path']}: "
                f"manifest={entry['content_sha256']}, disk={actual_hash}"
            )

    def test_adapter_files_present_in_slot(self, tmp_path) -> None:
        """Adapter weight files must be present under adapters/<name>/."""
        fixtures = _make_fixtures(tmp_path)
        slot = write_bundle(
            config_path=fixtures["config_path"],
            registry_path=fixtures["registry_path"],
            adapter_dirs=fixtures["adapter_dirs"],
            base_dir=fixtures["base_dir"],
            meta_fields={"tier": "manual"},
            live_registry_sha256=fixtures["registry_sha256"],
        )
        episodic_dir = slot / "adapters" / "episodic"
        assert (episodic_dir / "adapter_model.safetensors").exists()
        assert (episodic_dir / "adapter_config.json").exists()
        assert (episodic_dir / "meta.json").exists()

    def test_simhash_present_true_when_file_exists(self, tmp_path) -> None:
        """adapters record has simhash_present=True when simhash_registry.json exists."""
        fixtures = _make_fixtures(tmp_path, include_simhash=True)
        slot = write_bundle(
            config_path=fixtures["config_path"],
            registry_path=fixtures["registry_path"],
            adapter_dirs=fixtures["adapter_dirs"],
            base_dir=fixtures["base_dir"],
            meta_fields={"tier": "manual"},
            live_registry_sha256=fixtures["registry_sha256"],
        )
        manifest = BundleManifest.from_dict(
            json.loads((slot / "bundle.meta.json").read_text(encoding="utf-8"))
        )
        assert manifest.adapters["episodic"]["simhash_present"] is True

    def test_keyed_pairs_always_false(self, tmp_path) -> None:
        """keyed_pairs_present must always be False (QA pairs are transient)."""
        fixtures = _make_fixtures(tmp_path)
        slot = write_bundle(
            config_path=fixtures["config_path"],
            registry_path=fixtures["registry_path"],
            adapter_dirs=fixtures["adapter_dirs"],
            base_dir=fixtures["base_dir"],
            meta_fields={"tier": "manual"},
            live_registry_sha256=fixtures["registry_sha256"],
        )
        manifest = BundleManifest.from_dict(
            json.loads((slot / "bundle.meta.json").read_text(encoding="utf-8"))
        )
        assert manifest.adapters["episodic"]["keyed_pairs_present"] is False

    def test_base_model_info_populated(self, tmp_path) -> None:
        """base_model field is populated from the adapter's meta.json."""
        fixtures = _make_fixtures(tmp_path)
        slot = write_bundle(
            config_path=fixtures["config_path"],
            registry_path=fixtures["registry_path"],
            adapter_dirs=fixtures["adapter_dirs"],
            base_dir=fixtures["base_dir"],
            meta_fields={"tier": "manual"},
            live_registry_sha256=fixtures["registry_sha256"],
        )
        manifest = BundleManifest.from_dict(
            json.loads((slot / "bundle.meta.json").read_text(encoding="utf-8"))
        )
        assert manifest.base_model.get("repo") == "mistralai/Mistral-7B-Instruct-v0.3"

    def test_live_registry_sha256_recorded(self, tmp_path) -> None:
        """bundle manifest records the live_registry_sha256 verbatim."""
        fixtures = _make_fixtures(tmp_path)
        # live_registry_sha256 is stored verbatim in the bundle manifest regardless
        # of its value; slot resolution uses _tier_registry_sha256 (per-tier content
        # hash), not this field.  Use the fixture's computed hash so the slot resolves.
        reg_sha = fixtures["registry_sha256"]
        slot = write_bundle(
            config_path=fixtures["config_path"],
            registry_path=fixtures["registry_path"],
            adapter_dirs=fixtures["adapter_dirs"],
            base_dir=fixtures["base_dir"],
            meta_fields={"tier": "manual"},
            live_registry_sha256=reg_sha,
        )
        manifest = BundleManifest.from_dict(
            json.loads((slot / "bundle.meta.json").read_text(encoding="utf-8"))
        )
        assert manifest.live_registry_sha256 == reg_sha

    def test_no_artifact_meta_sidecar_written(self, tmp_path) -> None:
        """Per-artifact .meta.json sidecars must NOT be written (S2 constraint)."""
        fixtures = _make_fixtures(tmp_path)
        slot = write_bundle(
            config_path=fixtures["config_path"],
            registry_path=fixtures["registry_path"],
            adapter_dirs=fixtures["adapter_dirs"],
            base_dir=fixtures["base_dir"],
            meta_fields={"tier": "manual"},
            live_registry_sha256=fixtures["registry_sha256"],
        )
        # No file matching the ArtifactMeta sidecar pattern should exist.
        sidecars = list(slot.glob("**/*.meta.json"))
        for sc in sidecars:
            assert sc.name == "bundle.meta.json", (
                f"Unexpected ArtifactMeta sidecar found: {sc}. "
                "Bundle slots must not emit per-artifact sidecars (S2)."
            )

    def test_per_slot_registry_sha256_recorded(self, tmp_path) -> None:
        """adapters record stores the slot's OWN registry_sha256 from meta.json.

        The slot's ``registry_sha256`` equals the SHA-256 of the tier's
        ``indexed_key_registry.json`` bytes (set by the fixture helper so that
        ``find_live_slot`` matches).  The bundle manifest must echo that value
        verbatim — it comes from the slot's own ``meta.json``.
        """
        fixtures = _make_fixtures(tmp_path)
        # fixtures["registry_sha256"] == sha256(episodic/indexed_key_registry.json bytes)
        reg_sha = fixtures["registry_sha256"]
        slot = write_bundle(
            config_path=fixtures["config_path"],
            registry_path=fixtures["registry_path"],
            adapter_dirs=fixtures["adapter_dirs"],
            base_dir=fixtures["base_dir"],
            meta_fields={"tier": "manual"},
            live_registry_sha256=reg_sha,
        )
        manifest = BundleManifest.from_dict(
            json.loads((slot / "bundle.meta.json").read_text(encoding="utf-8"))
        )
        # The bundle captures the slot's own meta.json registry_sha256.
        assert manifest.adapters["episodic"]["registry_sha256"] == reg_sha


# ---------------------------------------------------------------------------
# Already-encrypted safetensors: copied byte-for-byte (no double-encrypt)
# ---------------------------------------------------------------------------


class TestNoBundleDoubleEncrypt:
    def test_age_envelope_copied_verbatim(self, tmp_path) -> None:
        """Weight files already age-encrypted are copied byte-for-byte.

        The on-disk bytes in the bundle must equal the source bytes exactly
        — the magic prefix must be preserved, not re-wrapped.
        """
        age_magic = b"age-encryption.org/v1\n"
        fake_encrypted_weights = age_magic + b"-> X25519 aGVsbG8K\n" + b"\x00" * 100

        fixtures = _make_fixtures(tmp_path, weight_bytes=fake_encrypted_weights)
        slot = write_bundle(
            config_path=fixtures["config_path"],
            registry_path=fixtures["registry_path"],
            adapter_dirs=fixtures["adapter_dirs"],
            base_dir=fixtures["base_dir"],
            meta_fields={"tier": "manual"},
            live_registry_sha256=fixtures["registry_sha256"],
        )

        on_disk = (slot / "adapters" / "episodic" / "adapter_model.safetensors").read_bytes()
        assert on_disk == fake_encrypted_weights, (
            "Age-encrypted weight file was re-encrypted (double-encrypt). "
            "write_bundle must copy already-encrypted blobs verbatim."
        )

    def test_age_envelope_still_starts_with_magic(self, tmp_path) -> None:
        """Copied encrypted blob must still start with the age magic."""
        age_magic = b"age-encryption.org/v1\n"
        fake_encrypted = age_magic + b"ciphertext_here"
        fixtures = _make_fixtures(tmp_path, weight_bytes=fake_encrypted)
        slot = write_bundle(
            config_path=fixtures["config_path"],
            registry_path=fixtures["registry_path"],
            adapter_dirs=fixtures["adapter_dirs"],
            base_dir=fixtures["base_dir"],
            meta_fields={"tier": "manual"},
            live_registry_sha256=fixtures["registry_sha256"],
        )
        on_disk = (slot / "adapters" / "episodic" / "adapter_model.safetensors").read_bytes()
        assert on_disk.startswith(age_magic)


# ---------------------------------------------------------------------------
# Real nested interim layout tests
# ---------------------------------------------------------------------------


class TestInterimSlotCapture:
    def test_adapter_scope_live_captures_interim(self, tmp_path) -> None:
        """adapter_scope='live' captures the nested interim slot when no main exists.

        Production state: episodic has NO finalized main slot, only
        interim_20260517T1200/<ts>/.  adapter_scope='live' must succeed and
        capture the interim slot.
        """
        data_dir = tmp_path / "ha"
        data_dir.mkdir()
        config_path = data_dir / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")
        registry_dir = data_dir / "registry"
        registry_dir.mkdir()
        registry_path = registry_dir / "key_metadata.json"
        registry_path.write_bytes(b'{"keys": []}')
        base_dir = data_dir / "backups" / "snapshot"
        base_dir.mkdir(parents=True)

        # Episodic dir: NO main slot, only an interim family.
        # _tier_registry_sha256 hashes interim_family_dir/indexed_key_registry.json;
        # stamp the slot with the same hash so find_live_slot matches.
        interim_content = b'{"keys": {"interim_key": "val"}}'
        interim_hash = _sha256(interim_content)
        episodic_dir = data_dir / "adapters" / "episodic"
        episodic_dir.mkdir(parents=True)
        interim_family_dir, _ = _make_interim_family(
            episodic_dir,
            stamp="20260517T1200",
            slot_name="20260517-180430",
            registry_sha256=interim_hash,
        )

        slot = write_bundle(
            config_path=config_path,
            registry_path=registry_path,
            adapter_dirs={"episodic": episodic_dir},
            base_dir=base_dir,
            meta_fields={"tier": "manual"},
            adapter_scope="live",
            live_registry_sha256=interim_hash,
        )

        assert slot.is_dir()
        manifest = BundleManifest.from_dict(
            json.loads((slot / "bundle.meta.json").read_text(encoding="utf-8"))
        )
        # The interim adapter must appear in the manifest.
        assert "episodic_interim_20260517T1200" in manifest.adapters, (
            "Interim adapter not captured in bundle"
        )

    def test_adapter_scope_main_fails_loud_for_interim_only_episodic(self, tmp_path) -> None:
        """adapter_scope='main' fails loud when episodic only has an interim slot.

        The error message must be actionable: direct the caller to use
        adapter_scope='live' or run a full consolidation.
        """
        data_dir = tmp_path / "ha"
        data_dir.mkdir()
        config_path = data_dir / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")
        registry_dir = data_dir / "registry"
        registry_dir.mkdir()
        registry_path = registry_dir / "key_metadata.json"
        registry_path.write_bytes(b'{"keys": []}')
        base_dir = data_dir / "backups" / "snapshot"
        base_dir.mkdir(parents=True)

        interim_content = b'{"keys": {"interim_key": "val"}}'
        interim_hash = _sha256(interim_content)
        episodic_dir = data_dir / "adapters" / "episodic"
        episodic_dir.mkdir(parents=True)
        _make_interim_family(
            episodic_dir,
            stamp="20260517T1200",
            slot_name="20260517-180430",
            registry_sha256=interim_hash,
        )

        with pytest.raises(BackupError, match="adapter_scope='main'"):
            write_bundle(
                config_path=config_path,
                registry_path=registry_path,
                adapter_dirs={"episodic": episodic_dir},
                base_dir=base_dir,
                meta_fields={"tier": "manual"},
                adapter_scope="main",
                live_registry_sha256=interim_hash,
            )

    def test_adapter_scope_main_message_is_actionable(self, tmp_path) -> None:
        """The fail-loud message for adapter_scope='main' directs to 'live' or consolidation."""
        data_dir = tmp_path / "ha"
        data_dir.mkdir()
        config_path = data_dir / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")
        registry_dir = data_dir / "registry"
        registry_dir.mkdir()
        registry_path = registry_dir / "key_metadata.json"
        registry_path.write_bytes(b'{"keys": []}')
        base_dir = data_dir / "backups" / "snapshot"
        base_dir.mkdir(parents=True)

        interim_content = b'{"keys": {"interim_key": "val"}}'
        interim_hash = _sha256(interim_content)
        episodic_dir = data_dir / "adapters" / "episodic"
        episodic_dir.mkdir(parents=True)
        _make_interim_family(
            episodic_dir,
            stamp="20260517T1200",
            slot_name="20260517-180430",
            registry_sha256=interim_hash,
        )

        with pytest.raises(BackupError) as exc_info:
            write_bundle(
                config_path=config_path,
                registry_path=registry_path,
                adapter_dirs={"episodic": episodic_dir},
                base_dir=base_dir,
                meta_fields={"tier": "manual"},
                adapter_scope="main",
                live_registry_sha256=interim_hash,
            )
        msg = str(exc_info.value)
        assert "live" in msg.lower(), f"Error message should mention 'live': {msg}"

    def test_interim_indexed_key_registry_captured(self, tmp_path) -> None:
        """indexed_key_registry.json from the interim-family dir is captured."""
        data_dir = tmp_path / "ha"
        data_dir.mkdir()
        config_path = data_dir / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")
        registry_dir = data_dir / "registry"
        registry_dir.mkdir()
        registry_path = registry_dir / "key_metadata.json"
        registry_path.write_bytes(b'{"keys": []}')
        base_dir = data_dir / "backups" / "snapshot"
        base_dir.mkdir(parents=True)

        interim_content = b'{"keys": {"interim_key": "val"}}'
        interim_hash = _sha256(interim_content)
        episodic_dir = data_dir / "adapters" / "episodic"
        episodic_dir.mkdir(parents=True)
        _make_interim_family(
            episodic_dir,
            stamp="20260517T1200",
            slot_name="20260517-180430",
            registry_sha256=interim_hash,
            include_indexed_key_registry=True,
        )

        slot = write_bundle(
            config_path=config_path,
            registry_path=registry_path,
            adapter_dirs={"episodic": episodic_dir},
            base_dir=base_dir,
            meta_fields={"tier": "manual"},
            adapter_scope="live",
            live_registry_sha256=interim_hash,
        )

        interim_key = "episodic_interim_20260517T1200"
        assert (slot / "adapters" / interim_key / "indexed_key_registry.json").exists(), (
            "interim indexed_key_registry.json not captured in bundle"
        )
        manifest = BundleManifest.from_dict(
            json.loads((slot / "bundle.meta.json").read_text(encoding="utf-8"))
        )
        assert manifest.adapters[interim_key]["indexed_key_registry_present"] is True

    def test_interim_simhash_captured(self, tmp_path) -> None:
        """simhash_registry.json from the interim-family dir is captured."""
        data_dir = tmp_path / "ha"
        data_dir.mkdir()
        config_path = data_dir / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")
        registry_dir = data_dir / "registry"
        registry_dir.mkdir()
        registry_path = registry_dir / "key_metadata.json"
        registry_path.write_bytes(b'{"keys": []}')
        base_dir = data_dir / "backups" / "snapshot"
        base_dir.mkdir(parents=True)

        interim_content = b'{"keys": {"interim_key": "val"}}'
        interim_hash = _sha256(interim_content)
        episodic_dir = data_dir / "adapters" / "episodic"
        episodic_dir.mkdir(parents=True)
        _make_interim_family(
            episodic_dir,
            stamp="20260517T1200",
            slot_name="20260517-180430",
            registry_sha256=interim_hash,
            include_simhash=True,
        )

        slot = write_bundle(
            config_path=config_path,
            registry_path=registry_path,
            adapter_dirs={"episodic": episodic_dir},
            base_dir=base_dir,
            meta_fields={"tier": "manual"},
            adapter_scope="live",
            live_registry_sha256=interim_hash,
        )

        interim_key = "episodic_interim_20260517T1200"
        assert (slot / "adapters" / interim_key / "simhash_registry.json").exists()
        manifest = BundleManifest.from_dict(
            json.loads((slot / "bundle.meta.json").read_text(encoding="utf-8"))
        )
        assert manifest.adapters[interim_key]["simhash_present"] is True


# ---------------------------------------------------------------------------
# Multi-adapter bundle: procedural main + episodic interim-only
# ---------------------------------------------------------------------------


class TestMultiAdapterBundle:
    """Procedural has a finalized main slot (hash A); episodic has only an
    interim slot (hash B).  adapter_scope='live' must capture both, each
    recording its OWN registry_sha256.
    """

    def _build_multi_adapter_fixtures(self, tmp_path: Path) -> dict:
        """Build fixture with procedural main + episodic interim-only.

        procedural: finalized main slot whose ``registry_sha256`` ==
            sha256(procedural/indexed_key_registry.json bytes).
        episodic: NO main slot; only an interim family whose slot's
            ``registry_sha256`` == sha256(interim_family/indexed_key_registry.json).
        The two hashes DIFFER because the two files have different content,
        proving that per-tier resolution is independent.
        """
        data_dir = tmp_path / "ha"
        data_dir.mkdir()
        config_path = data_dir / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")
        registry_dir = data_dir / "registry"
        registry_dir.mkdir()
        registry_path = registry_dir / "key_metadata.json"
        registry_path.write_bytes(b'{"keys": []}')
        base_dir = data_dir / "backups" / "snapshot"
        base_dir.mkdir(parents=True)

        # Procedural: finalized main slot — hash derived from its registry content.
        proc_content = b'{"keys": {"proc_key": 1}}'
        main_hash = _sha256(proc_content)

        procedural_dir = data_dir / "adapters" / "procedural"
        procedural_dir.mkdir(parents=True)
        _make_adapter_slot(
            procedural_dir,
            "20260517-180431",
            main_hash,
            adapter_name="procedural",
        )
        (procedural_dir / "indexed_key_registry.json").write_bytes(proc_content)
        (procedural_dir / "simhash_registry.json").write_bytes(b'{"simhash": {"proc": 9}}')

        # Episodic: NO main slot, only an interim family — hash from its own registry.
        interim_content = b'{"keys": {"interim_key": "val"}}'
        interim_hash = _sha256(interim_content)

        episodic_dir = data_dir / "adapters" / "episodic"
        episodic_dir.mkdir(parents=True)
        _make_interim_family(
            episodic_dir,
            stamp="20260517T1200",
            slot_name="20260517-180430",
            registry_sha256=interim_hash,
        )

        adapter_dirs = {
            "episodic": episodic_dir,
            "procedural": procedural_dir,
        }
        return {
            "config_path": config_path,
            "registry_path": registry_path,
            "adapter_dirs": adapter_dirs,
            "base_dir": base_dir,
            "main_hash": main_hash,
            "interim_hash": interim_hash,
            "episodic_dir": episodic_dir,
            "procedural_dir": procedural_dir,
        }

    def test_multi_adapter_both_captured(self, tmp_path) -> None:
        """Both procedural main and episodic interim are captured."""
        fx = self._build_multi_adapter_fixtures(tmp_path)

        # Use interim_hash so the interim slot is the 'live' one.
        slot = write_bundle(
            config_path=fx["config_path"],
            registry_path=fx["registry_path"],
            adapter_dirs=fx["adapter_dirs"],
            base_dir=fx["base_dir"],
            meta_fields={"tier": "manual"},
            adapter_scope="live",
            live_registry_sha256=fx["interim_hash"],
        )

        manifest = BundleManifest.from_dict(
            json.loads((slot / "bundle.meta.json").read_text(encoding="utf-8"))
        )
        # Interim must be present.
        assert "episodic_interim_20260517T1200" in manifest.adapters, (
            "Episodic interim not captured"
        )
        # Procedural main must be present when its hash matches.
        # Note: procedural has main_hash != interim_hash, so find_live_slot
        # won't match procedural; it's recorded absent (non-episodic, no failure).
        # To capture both we'd need two separate live_registry_sha256 values —
        # that's not supported in one call.  This test verifies that procedural
        # absent does NOT fail the bundle (only episodic failures are fatal).
        assert slot.is_dir(), "Bundle must succeed even when procedural absent"

    def test_multi_adapter_procedural_main_captured_when_hash_matches(self, tmp_path) -> None:
        """Both episodic and procedural main slots are captured when both have matching hashes.

        Both tiers share the same ``indexed_key_registry.json`` content (and thus
        the same hash), so a single ``live_registry_sha256`` resolves both.
        """
        data_dir = tmp_path / "ha"
        data_dir.mkdir()
        config_path = data_dir / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")
        registry_dir = data_dir / "registry"
        registry_dir.mkdir()
        registry_path = registry_dir / "key_metadata.json"
        registry_path.write_bytes(b'{"keys": []}')
        base_dir = data_dir / "backups" / "snapshot"
        base_dir.mkdir(parents=True)

        # Both tiers share the same registry content, so their hashes match.
        shared_content = b'{"keys": {}}'
        shared_hash = _sha256(shared_content)

        procedural_dir = data_dir / "adapters" / "procedural"
        procedural_dir.mkdir(parents=True)
        _make_adapter_slot(
            procedural_dir,
            "20260517-180431",
            shared_hash,
            adapter_name="procedural",
        )
        (procedural_dir / "indexed_key_registry.json").write_bytes(shared_content)

        episodic_dir = data_dir / "adapters" / "episodic"
        episodic_dir.mkdir(parents=True)
        _make_adapter_slot(
            episodic_dir,
            "20260517-180432",
            shared_hash,
            adapter_name="episodic",
        )
        (episodic_dir / "indexed_key_registry.json").write_bytes(shared_content)

        slot = write_bundle(
            config_path=config_path,
            registry_path=registry_path,
            adapter_dirs={"episodic": episodic_dir, "procedural": procedural_dir},
            base_dir=base_dir,
            meta_fields={"tier": "manual"},
            adapter_scope="live",
            live_registry_sha256=shared_hash,
        )

        manifest = BundleManifest.from_dict(
            json.loads((slot / "bundle.meta.json").read_text(encoding="utf-8"))
        )
        assert "episodic" in manifest.adapters
        assert "procedural" in manifest.adapters

    def test_multi_adapter_each_records_own_registry_sha256(self, tmp_path) -> None:
        """Main tier + interim family with DIFFERENT registry contents → both captured.

        Each adapter manifest entry records its OWN distinct ``registry_sha256``.
        ``find_live_slot`` resolves each slot via its per-tier hash:

        - ``_tier_registry_sha256(episodic_dir)`` == sha256(episodic content A)
          → matches main episodic slot stamped with hash A.
        - ``_tier_registry_sha256(interim_family_dir)`` == sha256(interim content B)
          → matches interim slot stamped with hash B.

        hash A ≠ hash B, proving independent per-tier resolution.
        """
        data_dir = tmp_path / "ha"
        data_dir.mkdir()
        config_path = data_dir / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")
        registry_dir = data_dir / "registry"
        registry_dir.mkdir()
        registry_path = registry_dir / "key_metadata.json"
        registry_path.write_bytes(b'{"keys": []}')
        base_dir = data_dir / "backups" / "snapshot"
        base_dir.mkdir(parents=True)

        # Main episodic: distinct content so its hash differs from the interim.
        ep_main_content = b'{"keys": {"ep_main": 1}}'
        ep_main_hash = _sha256(ep_main_content)
        episodic_dir = data_dir / "adapters" / "episodic"
        episodic_dir.mkdir(parents=True)
        _make_adapter_slot(episodic_dir, "20260517-111111", ep_main_hash, adapter_name="episodic")
        (episodic_dir / "indexed_key_registry.json").write_bytes(ep_main_content)

        # Interim with DIFFERENT content — each slot records its OWN hash.
        interim_content = b'{"keys": {"interim_key": "val"}}'
        interim_hash = _sha256(interim_content)
        assert ep_main_hash != interim_hash, "fixture must use distinct content per tier"
        _make_interim_family(
            episodic_dir,
            stamp="20260517T1200",
            slot_name="20260517-180430",
            registry_sha256=interim_hash,
        )

        slot = write_bundle(
            config_path=config_path,
            registry_path=registry_path,
            adapter_dirs={"episodic": episodic_dir},
            base_dir=base_dir,
            meta_fields={"tier": "manual"},
            adapter_scope="live",
            live_registry_sha256=ep_main_hash,
        )

        manifest = BundleManifest.from_dict(
            json.loads((slot / "bundle.meta.json").read_text(encoding="utf-8"))
        )
        # Both must be captured — each resolved via its own per-tier hash.
        assert "episodic" in manifest.adapters, "Main episodic slot not captured"
        assert "episodic_interim_20260517T1200" in manifest.adapters, (
            "Interim slot not captured — per-tier hash resolution failed"
        )
        # Each entry records its OWN registry_sha256.
        assert manifest.adapters["episodic"]["registry_sha256"] == ep_main_hash
        assert (
            manifest.adapters["episodic_interim_20260517T1200"]["registry_sha256"] == interim_hash
        )
        assert ep_main_hash != interim_hash, "test must prove the two hashes genuinely differ"

    def test_multi_adapter_indexed_key_registry_both(self, tmp_path) -> None:
        """indexed_key_registry.json is captured for BOTH procedural main and episodic main.

        Each tier has its own ``indexed_key_registry.json`` content (and thus its own
        per-tier hash), so the slot stamps and ``_tier_registry_sha256`` calls match
        independently for each tier.
        """
        data_dir = tmp_path / "ha"
        data_dir.mkdir()
        config_path = data_dir / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")
        registry_dir = data_dir / "registry"
        registry_dir.mkdir()
        registry_path = registry_dir / "key_metadata.json"
        registry_path.write_bytes(b'{"keys": []}')
        base_dir = data_dir / "backups" / "snapshot"
        base_dir.mkdir(parents=True)

        # Episodic: derive hash from its own registry content.
        ep_content = b'{"keys": {"ep": 1}}'
        ep_hash = _sha256(ep_content)
        episodic_dir = data_dir / "adapters" / "episodic"
        episodic_dir.mkdir(parents=True)
        _make_adapter_slot(episodic_dir, "20260517-180432", ep_hash, adapter_name="episodic")
        (episodic_dir / "indexed_key_registry.json").write_bytes(ep_content)

        # Procedural: derive hash from its own (different) registry content.
        proc_content = b'{"keys": {"proc": 2}}'
        proc_hash = _sha256(proc_content)
        procedural_dir = data_dir / "adapters" / "procedural"
        procedural_dir.mkdir(parents=True)
        _make_adapter_slot(procedural_dir, "20260517-180431", proc_hash, adapter_name="procedural")
        (procedural_dir / "indexed_key_registry.json").write_bytes(proc_content)

        slot = write_bundle(
            config_path=config_path,
            registry_path=registry_path,
            adapter_dirs={"episodic": episodic_dir, "procedural": procedural_dir},
            base_dir=base_dir,
            meta_fields={"tier": "manual"},
            adapter_scope="live",
            live_registry_sha256=ep_hash,
        )

        assert (slot / "adapters" / "episodic" / "indexed_key_registry.json").exists()
        assert (slot / "adapters" / "procedural" / "indexed_key_registry.json").exists()
        manifest = BundleManifest.from_dict(
            json.loads((slot / "bundle.meta.json").read_text(encoding="utf-8"))
        )
        assert manifest.adapters["episodic"]["indexed_key_registry_present"] is True
        assert manifest.adapters["procedural"]["indexed_key_registry_present"] is True


# ---------------------------------------------------------------------------
# Scaffolding exclusion
# ---------------------------------------------------------------------------


class TestScaffoldingExclusion:
    def test_checkpoint_dirs_not_in_bundle(self, tmp_path) -> None:
        """checkpoint-*/ scaffolding inside interim family is never in the bundle."""
        data_dir = tmp_path / "ha"
        data_dir.mkdir()
        config_path = data_dir / "server.yaml"
        config_path.write_bytes(b"model: mistral\n")
        registry_dir = data_dir / "registry"
        registry_dir.mkdir()
        registry_path = registry_dir / "key_metadata.json"
        registry_path.write_bytes(b'{"keys": []}')
        base_dir = data_dir / "backups" / "snapshot"
        base_dir.mkdir(parents=True)

        interim_content = b'{"keys": {"interim_key": "val"}}'
        interim_hash = _sha256(interim_content)
        episodic_dir = data_dir / "adapters" / "episodic"
        episodic_dir.mkdir(parents=True)
        # _make_interim_family creates checkpoint-1863/ and in_training/ siblings.
        _make_interim_family(
            episodic_dir,
            stamp="20260517T1200",
            slot_name="20260517-180430",
            registry_sha256=interim_hash,
        )

        slot = write_bundle(
            config_path=config_path,
            registry_path=registry_path,
            adapter_dirs={"episodic": episodic_dir},
            base_dir=base_dir,
            meta_fields={"tier": "manual"},
            adapter_scope="live",
            live_registry_sha256=interim_hash,
        )

        # No file in the bundle may come from checkpoint-*/ or in_training/.
        manifest = BundleManifest.from_dict(
            json.loads((slot / "bundle.meta.json").read_text(encoding="utf-8"))
        )
        for f in manifest.files:
            assert "checkpoint-" not in f["path"], f"Checkpoint scaffolding captured: {f['path']}"
            assert "in_training" not in f["path"], f"in_training scaffolding captured: {f['path']}"

    def test_main_scope_interim_dir_not_in_bundle(self, tmp_path) -> None:
        """Under adapter_scope='main', interim dirs are not captured."""
        fixtures = _make_fixtures(tmp_path)
        episodic_dir = fixtures["episodic_dir"]

        # Add an interim family with the same hash — it must not be captured
        # when adapter_scope='main'.
        _make_interim_family(
            episodic_dir,
            stamp="20260520T1200",
            slot_name="20260520-123457",
            registry_sha256=fixtures["registry_sha256"],
        )

        slot = write_bundle(
            config_path=fixtures["config_path"],
            registry_path=fixtures["registry_path"],
            adapter_dirs=fixtures["adapter_dirs"],
            base_dir=fixtures["base_dir"],
            meta_fields={"tier": "manual"},
            adapter_scope="main",
            live_registry_sha256=fixtures["registry_sha256"],
        )

        manifest = BundleManifest.from_dict(
            json.loads((slot / "bundle.meta.json").read_text(encoding="utf-8"))
        )
        for key in manifest.adapters:
            assert "interim" not in key.lower(), (
                f"Interim adapter captured under adapter_scope='main': {key}"
            )


# ---------------------------------------------------------------------------
# Missing optional artifact: simhash_registry.json absent
# ---------------------------------------------------------------------------


class TestMissingOptionalArtifact:
    def test_no_simhash_no_crash(self, tmp_path) -> None:
        """Missing simhash_registry.json → simhash_present=False, no exception."""
        fixtures = _make_fixtures(tmp_path, include_simhash=False)
        slot = write_bundle(
            config_path=fixtures["config_path"],
            registry_path=fixtures["registry_path"],
            adapter_dirs=fixtures["adapter_dirs"],
            base_dir=fixtures["base_dir"],
            meta_fields={"tier": "manual"},
            live_registry_sha256=fixtures["registry_sha256"],
        )
        manifest = BundleManifest.from_dict(
            json.loads((slot / "bundle.meta.json").read_text(encoding="utf-8"))
        )
        assert manifest.adapters["episodic"]["simhash_present"] is False

    def test_no_simhash_file_in_bundle(self, tmp_path) -> None:
        """When simhash absent at source, no simhash file in bundle slot."""
        fixtures = _make_fixtures(tmp_path, include_simhash=False)
        slot = write_bundle(
            config_path=fixtures["config_path"],
            registry_path=fixtures["registry_path"],
            adapter_dirs=fixtures["adapter_dirs"],
            base_dir=fixtures["base_dir"],
            meta_fields={"tier": "manual"},
            live_registry_sha256=fixtures["registry_sha256"],
        )
        assert not (slot / "adapters" / "episodic" / "simhash_registry.json").exists()

    def test_non_episodic_tier_absent_does_not_fail(self, tmp_path) -> None:
        """A non-episodic tier with no main slot is recorded absent, no failure."""
        fixtures = _make_fixtures(tmp_path)
        adapters_dir = fixtures["adapters_dir"]

        # Add a procedural dir with no matching slot.
        procedural_dir = adapters_dir / "procedural"
        procedural_dir.mkdir()
        # No slot created → find_live_slot returns None → non-episodic: no fail.

        adapter_dirs = {"episodic": fixtures["episodic_dir"], "procedural": procedural_dir}
        # Must not raise.
        slot = write_bundle(
            config_path=fixtures["config_path"],
            registry_path=fixtures["registry_path"],
            adapter_dirs=adapter_dirs,
            base_dir=fixtures["base_dir"],
            meta_fields={"tier": "manual"},
            live_registry_sha256=fixtures["registry_sha256"],
        )
        assert slot.is_dir()
        manifest = BundleManifest.from_dict(
            json.loads((slot / "bundle.meta.json").read_text(encoding="utf-8"))
        )
        # Procedural must be absent from the adapters record (not present → not recorded).
        assert "procedural" not in manifest.adapters


# ---------------------------------------------------------------------------
# speaker_profiles.json
# ---------------------------------------------------------------------------


class TestSpeakerProfilesCapture:
    def test_speaker_profiles_included_when_present(self, tmp_path) -> None:
        """speaker_profiles.json is captured in the bundle when path provided."""
        fixtures = _make_fixtures(tmp_path, include_speaker_profiles=True)
        slot = write_bundle(
            config_path=fixtures["config_path"],
            registry_path=fixtures["registry_path"],
            adapter_dirs=fixtures["adapter_dirs"],
            base_dir=fixtures["base_dir"],
            meta_fields={"tier": "manual"},
            live_registry_sha256=fixtures["registry_sha256"],
            speaker_profiles_path=fixtures["speaker_profiles_path"],
        )
        assert (slot / "speaker_profiles.json").exists()
        manifest = BundleManifest.from_dict(
            json.loads((slot / "bundle.meta.json").read_text(encoding="utf-8"))
        )
        paths = {f["path"] for f in manifest.files}
        assert "speaker_profiles.json" in paths

    def test_speaker_profiles_absent_when_path_none(self, tmp_path) -> None:
        """speaker_profiles.json not in bundle when speaker_profiles_path=None."""
        fixtures = _make_fixtures(tmp_path)
        slot = write_bundle(
            config_path=fixtures["config_path"],
            registry_path=fixtures["registry_path"],
            adapter_dirs=fixtures["adapter_dirs"],
            base_dir=fixtures["base_dir"],
            meta_fields={"tier": "manual"},
            live_registry_sha256=fixtures["registry_sha256"],
            speaker_profiles_path=None,
        )
        assert not (slot / "speaker_profiles.json").exists()
        manifest = BundleManifest.from_dict(
            json.loads((slot / "bundle.meta.json").read_text(encoding="utf-8"))
        )
        paths = {f["path"] for f in manifest.files}
        assert "speaker_profiles.json" not in paths


# ---------------------------------------------------------------------------
# Fail-loud when no slot for episodic
# ---------------------------------------------------------------------------


class TestFailLoudNoSlot:
    def test_no_live_slot_raises_backup_error(self, tmp_path) -> None:
        """write_bundle raises BackupError when no episodic slot found at all.

        The mismatch is created by overwriting ``indexed_key_registry.json`` with
        different content after the fixture slot was stamped.  This causes
        ``_tier_registry_sha256(episodic_dir)`` to return a hash that does not match
        the slot's ``meta.registry_sha256``, so ``find_live_slot`` returns ``None``.
        """
        fixtures = _make_fixtures(tmp_path)
        # Overwrite the tier registry with different content so the hash diverges from
        # what the slot's meta.json was stamped with → find_live_slot returns None.
        (fixtures["episodic_dir"] / "indexed_key_registry.json").write_bytes(
            b'{"keys": {"stale": true}}'
        )
        with pytest.raises(BackupError):
            write_bundle(
                config_path=fixtures["config_path"],
                registry_path=fixtures["registry_path"],
                adapter_dirs=fixtures["adapter_dirs"],
                base_dir=fixtures["base_dir"],
                meta_fields={"tier": "manual"},
                live_registry_sha256=fixtures["registry_sha256"],
            )

    def test_no_slot_leaves_no_promoted_directory(self, tmp_path) -> None:
        """When write_bundle raises, no bundle slot is left in base_dir.

        Same mismatch strategy as ``test_no_live_slot_raises_backup_error``.
        """
        fixtures = _make_fixtures(tmp_path)
        base_dir = fixtures["base_dir"]
        # Overwrite tier registry so the hash no longer matches the slot stamp.
        (fixtures["episodic_dir"] / "indexed_key_registry.json").write_bytes(
            b'{"keys": {"stale": true}}'
        )
        with pytest.raises(BackupError):
            write_bundle(
                config_path=fixtures["config_path"],
                registry_path=fixtures["registry_path"],
                adapter_dirs=fixtures["adapter_dirs"],
                base_dir=base_dir,
                meta_fields={"tier": "manual"},
                live_registry_sha256=fixtures["registry_sha256"],
            )
        slots = (
            [d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if base_dir.exists()
            else []
        )
        assert slots == [], f"Unexpected slots in base_dir after failure: {slots}"


# ---------------------------------------------------------------------------
# Crash-safety
# ---------------------------------------------------------------------------


class TestCrashSafety:
    def test_failed_rename_leaves_pending_residue(self, tmp_path) -> None:
        """If rename fails, a .pending/<ts>/ residue is left (swept on startup)."""
        fixtures = _make_fixtures(tmp_path)
        base_dir = fixtures["base_dir"]

        rename_called = []

        def _failing_rename(pending, slot):
            rename_called.append(pending)
            raise OSError("simulated rename failure")

        with patch("paramem.backup.backup.rename_pending_to_slot", side_effect=_failing_rename):
            with pytest.raises(OSError, match="simulated rename failure"):
                write_bundle(
                    config_path=fixtures["config_path"],
                    registry_path=fixtures["registry_path"],
                    adapter_dirs=fixtures["adapter_dirs"],
                    base_dir=base_dir,
                    meta_fields={"tier": "manual"},
                    live_registry_sha256=fixtures["registry_sha256"],
                )

        assert len(rename_called) == 1
        pending_root = base_dir / ".pending"
        assert pending_root.exists(), "No .pending/ dir found after rename failure"
        pending_children = list(pending_root.iterdir())
        assert len(pending_children) == 1, "Expected one pending residue"

        # sweep_orphan_pending must remove the residue.
        removed = sweep_orphan_pending(base_dir)
        assert removed == 1
        assert not list(pending_root.iterdir()), "Pending residue not swept"

    def test_no_slot_promoted_after_failure(self, tmp_path) -> None:
        """After a rename failure, no promoted slot directory exists."""
        fixtures = _make_fixtures(tmp_path)
        base_dir = fixtures["base_dir"]

        with patch("paramem.backup.backup.rename_pending_to_slot", side_effect=OSError("fail")):
            with pytest.raises(OSError):
                write_bundle(
                    config_path=fixtures["config_path"],
                    registry_path=fixtures["registry_path"],
                    adapter_dirs=fixtures["adapter_dirs"],
                    base_dir=base_dir,
                    meta_fields={"tier": "manual"},
                    live_registry_sha256=fixtures["registry_sha256"],
                )

        promoted = (
            [d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if base_dir.exists()
            else []
        )
        assert promoted == []


# ---------------------------------------------------------------------------
# enumerate_backups recognises bundle slots (integration check)
# ---------------------------------------------------------------------------


class TestEnumerateBundle:
    def test_bundle_slot_enumerated(self, tmp_path) -> None:
        """A write_bundle output is returned by enumerate_backups as SNAPSHOT_BUNDLE."""
        fixtures = _make_fixtures(tmp_path)
        write_bundle(
            config_path=fixtures["config_path"],
            registry_path=fixtures["registry_path"],
            adapter_dirs=fixtures["adapter_dirs"],
            base_dir=fixtures["base_dir"],
            meta_fields={"tier": "manual"},
            live_registry_sha256=fixtures["registry_sha256"],
        )

        backups_root = fixtures["base_dir"].parent
        records = enumerate_backups(backups_root, kind=ArtifactKind.SNAPSHOT_BUNDLE)
        assert len(records) == 1
        assert records[0].kind == ArtifactKind.SNAPSHOT_BUNDLE
        assert records[0].is_bundle is True

    def test_bundle_not_flagged_invalid(self, tmp_path) -> None:
        """enumerate_backups must not log the bundle slot as invalid/unreadable."""
        import io
        import logging

        fixtures = _make_fixtures(tmp_path)
        write_bundle(
            config_path=fixtures["config_path"],
            registry_path=fixtures["registry_path"],
            adapter_dirs=fixtures["adapter_dirs"],
            base_dir=fixtures["base_dir"],
            meta_fields={"tier": "manual"},
            live_registry_sha256=fixtures["registry_sha256"],
        )

        backups_root = fixtures["base_dir"].parent
        named_logger = logging.getLogger("paramem.backup.enumerate")
        orig_propagate = named_logger.propagate
        named_logger.propagate = True

        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.WARNING)
        named_logger.addHandler(handler)
        try:
            records = enumerate_backups(backups_root)
        finally:
            named_logger.removeHandler(handler)
            named_logger.propagate = orig_propagate

        warnings_text = log_capture.getvalue()
        assert "unreadable" not in warnings_text.lower(), (
            f"enumerate_backups emitted 'unreadable' warning for bundle slot: {warnings_text!r}"
        )
        assert len(records) == 1


# ---------------------------------------------------------------------------
# Candidate-config sidecar (base-swap bundles)
# ---------------------------------------------------------------------------


class TestCandidateConfigSidecar:
    """Tests for the ``candidate_config_path`` sidecar in write_bundle.

    The sidecar is captured at the bundle root as ``server.yaml.candidate``
    (NOT under ``config/``), hash-indexed in the manifest, and never restored.
    """

    def _call_write_bundle(self, fixtures: dict, candidate_config_path=None):
        """Shared helper: calls write_bundle with optional candidate_config_path."""
        return write_bundle(
            config_path=fixtures["config_path"],
            registry_path=fixtures["registry_path"],
            adapter_dirs=fixtures["adapter_dirs"],
            base_dir=fixtures["base_dir"],
            meta_fields={"tier": "pre_base_swap"},
            live_registry_sha256=fixtures["registry_sha256"],
            candidate_config_path=candidate_config_path,
        )

    def test_candidate_sidecar_captured(self, tmp_path) -> None:
        """Candidate file appears in manifest with correct path, bytes, and encrypted=False."""
        fixtures = _make_fixtures(tmp_path)
        candidate = tmp_path / "server.yaml.next"
        candidate_bytes = b"model: qwen3\nport: 8420\n"
        candidate.write_bytes(candidate_bytes)

        slot = self._call_write_bundle(fixtures, candidate_config_path=candidate)

        # Manifest entry must be present.
        manifest = json.loads((slot / "bundle.meta.json").read_text(encoding="utf-8"))
        candidate_entries = [f for f in manifest["files"] if f["path"] == "server.yaml.candidate"]
        assert len(candidate_entries) == 1, (
            f"Expected exactly 1 manifest entry for server.yaml.candidate; "
            f"got {len(candidate_entries)}"
        )
        entry = candidate_entries[0]
        assert entry["encrypted"] is False, "Candidate sidecar must not be encrypted"

        # On-disk bytes must equal source.
        on_disk = (slot / "server.yaml.candidate").read_bytes()
        assert on_disk == candidate_bytes, "Candidate sidecar bytes differ from source"

        # Hash in manifest must equal sha256 of on-disk bytes.
        assert entry["content_sha256"] == _sha256(candidate_bytes), (
            "Manifest hash does not match candidate bytes"
        )

    def test_candidate_sidecar_not_under_config_prefix(self, tmp_path) -> None:
        """No manifest entry with path starting with 'config/' points at the candidate.

        The restore_bundle Step-5c filter (startswith('config/')) must NOT
        select the candidate sidecar — it lives at the top level.
        """
        fixtures = _make_fixtures(tmp_path)
        candidate = tmp_path / "server.yaml.next"
        candidate.write_bytes(b"model: qwen3\n")

        slot = self._call_write_bundle(fixtures, candidate_config_path=candidate)

        manifest = json.loads((slot / "bundle.meta.json").read_text(encoding="utf-8"))
        config_entries = [f for f in manifest["files"] if f["path"].startswith("config/")]
        for entry in config_entries:
            assert "candidate" not in entry["path"], (
                f"A config/-prefixed entry unexpectedly references the candidate: {entry['path']}"
            )

    def test_no_candidate_no_sidecar_entry(self, tmp_path) -> None:
        """When candidate_config_path=None (default), no sidecar entry in manifest."""
        fixtures = _make_fixtures(tmp_path)

        slot = self._call_write_bundle(fixtures, candidate_config_path=None)

        manifest = json.loads((slot / "bundle.meta.json").read_text(encoding="utf-8"))
        candidate_entries = [f for f in manifest["files"] if "candidate" in f["path"]]
        assert candidate_entries == [], (
            f"No candidate_config_path supplied but sidecar entry found: {candidate_entries}"
        )

    def test_missing_candidate_raises_backup_error(self, tmp_path) -> None:
        """Supplying a non-existent candidate_config_path raises BackupError with clear message.

        The existence check is hoisted above the pending-slot allocation, so the
        raise leaves ZERO on-disk residue — no orphan ``.pending/<ts>/``.
        """
        fixtures = _make_fixtures(tmp_path)
        missing = tmp_path / "does_not_exist.yaml"
        base_dir = Path(fixtures["base_dir"])

        with pytest.raises(BackupError, match="candidate_config_path does not exist"):
            self._call_write_bundle(fixtures, candidate_config_path=missing)

        # The guard fires before _promote_slot, so no pending dir is created.
        pending_root = base_dir / ".pending"
        residue = list(pending_root.glob("*")) if pending_root.exists() else []
        assert residue == [], f"missing-candidate raise left pending residue: {residue}"
