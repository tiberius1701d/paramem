"""Tests for paramem.backup.meta."""

from __future__ import annotations

import json

import pytest

from paramem.backup.meta import read_meta, verify_fingerprint, write_meta
from paramem.backup.types import (
    SCHEMA_VERSION,
    ArtifactKind,
    ArtifactMeta,
    EncryptAtRest,
    FingerprintMismatchError,
    MetaSchemaError,
)


def _make_meta(**overrides) -> ArtifactMeta:
    defaults = dict(
        schema_version=SCHEMA_VERSION,
        kind=ArtifactKind.CONFIG,
        timestamp="20260421-040000" + "00",
        content_sha256="a" * 64,
        size_bytes=128,
        encrypted=False,
        encrypt_at_rest=EncryptAtRest.AUTO,
        key_fingerprint=None,
        tier="scheduled",
        label=None,
    )
    defaults.update(overrides)
    return ArtifactMeta(**defaults)


class TestWriteReadMetaRoundtrip:
    def test_meta_roundtrip(self, tmp_path):
        """Dataclass written to disk and read back must be equal."""
        meta = _make_meta()
        slot_dir = tmp_path / "slot1"
        slot_dir.mkdir()

        write_meta(slot_dir, meta)
        recovered = read_meta(slot_dir)

        assert recovered == meta

    def test_roundtrip_with_label(self, tmp_path):
        """Label field round-trips correctly."""
        meta = _make_meta(label="pre-migration-2026-04-21")
        slot_dir = tmp_path / "slot_label"
        slot_dir.mkdir()
        write_meta(slot_dir, meta)
        assert read_meta(slot_dir).label == "pre-migration-2026-04-21"

    def test_roundtrip_encrypted_with_fingerprint(self, tmp_path):
        """Encrypted meta with key_fingerprint round-trips."""
        meta = _make_meta(
            encrypted=True,
            encrypt_at_rest=EncryptAtRest.ALWAYS,
            key_fingerprint="abcdef0123456789",
        )
        slot_dir = tmp_path / "slot_enc"
        slot_dir.mkdir()
        write_meta(slot_dir, meta)
        recovered = read_meta(slot_dir)
        assert recovered.encrypted is True
        assert recovered.key_fingerprint == "abcdef0123456789"

    def test_meta_rejects_future_schema_version(self, tmp_path):
        """Sidecar with schema_version > SCHEMA_VERSION raises MetaSchemaError. (NIT 2)"""
        meta = _make_meta()
        slot_dir = tmp_path / "future_slot"
        slot_dir.mkdir()
        write_meta(slot_dir, meta)

        # Patch the sidecar on disk to have a future version
        sidecar = list(slot_dir.glob("*.meta.json"))[0]
        raw = json.loads(sidecar.read_text())
        raw["schema_version"] = SCHEMA_VERSION + 1
        sidecar.write_text(json.dumps(raw))

        with pytest.raises(MetaSchemaError, match="forward version"):
            read_meta(slot_dir)

    def test_meta_rejects_legacy_schema_version_documented_unreachable_at_v1(self, tmp_path):
        """Sidecar with schema_version < SCHEMA_VERSION raises MetaSchemaError.

        At SCHEMA_VERSION=1 no legacy sidecar can have been written by this
        codebase, but the path must be reachable for future bump safety.
        """
        meta = _make_meta()
        slot_dir = tmp_path / "legacy_slot"
        slot_dir.mkdir()
        write_meta(slot_dir, meta)

        sidecar = list(slot_dir.glob("*.meta.json"))[0]
        raw = json.loads(sidecar.read_text())
        raw["schema_version"] = max(0, SCHEMA_VERSION - 1)
        sidecar.write_text(json.dumps(raw))

        with pytest.raises(MetaSchemaError, match="legacy"):
            read_meta(slot_dir)

    def test_meta_rejects_missing_required_field(self, tmp_path):
        """Sidecar without content_sha256 raises MetaSchemaError."""
        meta = _make_meta()
        slot_dir = tmp_path / "missing_field"
        slot_dir.mkdir()
        write_meta(slot_dir, meta)

        sidecar = list(slot_dir.glob("*.meta.json"))[0]
        raw = json.loads(sidecar.read_text())
        del raw["content_sha256"]
        sidecar.write_text(json.dumps(raw))

        with pytest.raises(MetaSchemaError):
            read_meta(slot_dir)

    def test_meta_rejects_unknown_kind(self, tmp_path):
        """Sidecar with kind 'banana' raises MetaSchemaError."""
        meta = _make_meta()
        slot_dir = tmp_path / "unknown_kind"
        slot_dir.mkdir()
        write_meta(slot_dir, meta)

        sidecar = list(slot_dir.glob("*.meta.json"))[0]
        raw = json.loads(sidecar.read_text())
        raw["kind"] = "banana"
        sidecar.write_text(json.dumps(raw))

        with pytest.raises(MetaSchemaError, match="unknown kind"):
            read_meta(slot_dir)

    def test_read_meta_missing_sidecar(self, tmp_path):
        """Empty slot directory raises FileNotFoundError."""
        slot_dir = tmp_path / "empty_slot"
        slot_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            read_meta(slot_dir)

    def test_read_meta_rejects_corrupt_json(self, tmp_path):
        """Sidecar containing invalid JSON raises MetaSchemaError("corrupt sidecar:...").

        The ``__cause__`` must be a ``json.JSONDecodeError`` (Fix #2 path A).
        """
        import json

        meta = _make_meta()
        slot_dir = tmp_path / "corrupt_json_slot"
        slot_dir.mkdir()
        write_meta(slot_dir, meta)

        # Overwrite sidecar with invalid JSON content
        sidecar = list(slot_dir.glob("*.meta.json"))[0]
        sidecar.write_bytes(b"not-valid-json{{")

        with pytest.raises(MetaSchemaError) as exc_info:
            read_meta(slot_dir)

        assert "corrupt" in str(exc_info.value).lower(), (
            f"expected 'corrupt' in MetaSchemaError message, got: {exc_info.value!r}"
        )
        assert isinstance(exc_info.value.__cause__, json.JSONDecodeError), (
            f"__cause__ must be JSONDecodeError, got: {type(exc_info.value.__cause__)}"
        )

    def test_read_meta_rejects_binary_garbage_sidecar(self, tmp_path):
        """Sidecar containing binary garbage (non-UTF-8) raises MetaSchemaError.

        The ``__cause__`` must be a ``UnicodeDecodeError`` (Fix #2 path B).
        """
        meta = _make_meta()
        slot_dir = tmp_path / "binary_garbage_slot"
        slot_dir.mkdir()
        write_meta(slot_dir, meta)

        # Overwrite sidecar with non-UTF-8 binary content
        sidecar = list(slot_dir.glob("*.meta.json"))[0]
        sidecar.write_bytes(b"\x00\x01\x02\xff\xfe")

        with pytest.raises(MetaSchemaError) as exc_info:
            read_meta(slot_dir)

        assert "corrupt" in str(exc_info.value).lower(), (
            f"expected 'corrupt' in MetaSchemaError message, got: {exc_info.value!r}"
        )
        assert isinstance(exc_info.value.__cause__, UnicodeDecodeError), (
            f"__cause__ must be UnicodeDecodeError, got: {type(exc_info.value.__cause__)}"
        )


class TestVerifyFingerprint:
    def _slot_with_artifact(self, tmp_path, content: bytes) -> tuple:
        """Write a slot directory with real content and matching meta."""
        from paramem.backup.hashing import content_sha256_bytes

        slot_dir = tmp_path / "vslot"
        slot_dir.mkdir()

        artifact = slot_dir / "config-20260421-04000000.bin"
        artifact.write_bytes(content)

        meta = _make_meta(
            content_sha256=content_sha256_bytes(content),
            size_bytes=len(content),
        )
        write_meta(slot_dir, meta)
        return slot_dir, artifact

    def test_verify_fingerprint_matches_content(self, tmp_path):
        """Verify passes when artifact matches the stored hash."""
        slot_dir, artifact = self._slot_with_artifact(tmp_path, b"correct content")
        verify_fingerprint(slot_dir, artifact)  # should not raise

    def test_verify_fingerprint_rejects_mutation(self, tmp_path):
        """Mutating one byte raises FingerprintMismatchError."""
        slot_dir, artifact = self._slot_with_artifact(tmp_path, b"original content")
        artifact.write_bytes(b"modified content!")
        with pytest.raises(FingerprintMismatchError):
            verify_fingerprint(slot_dir, artifact)
