"""Tests for snapshot-bundle additions to paramem.backup.types.

Covers:
- ``ArtifactKind.SNAPSHOT_BUNDLE`` member present and has correct value.
- ``BUNDLE_SCHEMA_VERSION`` constant present and equals 1.
- ``BundleManifest`` round-trips through ``to_dict`` / ``from_dict``.
- ``BundleManifest.from_dict`` raises ``BundleManifestError`` on schema-version
  mismatch and on missing required fields.
- ``BundleManifestError`` is a subclass of ``BackupError``.
"""

from __future__ import annotations

import pytest

from paramem.backup.types import (
    BUNDLE_SCHEMA_VERSION,
    ArtifactKind,
    BackupError,
    BundleManifest,
    BundleManifestError,
)

# ---------------------------------------------------------------------------
# ArtifactKind.SNAPSHOT_BUNDLE
# ---------------------------------------------------------------------------


class TestSnapshotBundleKind:
    def test_snapshot_bundle_member_exists(self) -> None:
        """ArtifactKind.SNAPSHOT_BUNDLE must be a valid enum member."""
        assert ArtifactKind.SNAPSHOT_BUNDLE is ArtifactKind("snapshot_bundle")

    def test_snapshot_bundle_string_value(self) -> None:
        """The string value must be 'snapshot_bundle' (stored in JSON)."""
        assert ArtifactKind.SNAPSHOT_BUNDLE.value == "snapshot_bundle"

    def test_snapshot_bundle_distinct_from_snapshot(self) -> None:
        """SNAPSHOT_BUNDLE must not collide with the existing SNAPSHOT value."""
        assert ArtifactKind.SNAPSHOT_BUNDLE != ArtifactKind.SNAPSHOT
        assert ArtifactKind.SNAPSHOT.value == "snapshot"


# ---------------------------------------------------------------------------
# BUNDLE_SCHEMA_VERSION
# ---------------------------------------------------------------------------


class TestBundleSchemaVersionConstant:
    def test_bundle_schema_version_is_int(self) -> None:
        """BUNDLE_SCHEMA_VERSION must be an integer."""
        assert isinstance(BUNDLE_SCHEMA_VERSION, int)

    def test_bundle_schema_version_equals_one(self) -> None:
        """BUNDLE_SCHEMA_VERSION must be 1 (initial version)."""
        assert BUNDLE_SCHEMA_VERSION == 1


# ---------------------------------------------------------------------------
# BundleManifestError
# ---------------------------------------------------------------------------


class TestBundleManifestError:
    def test_is_backup_error_subclass(self) -> None:
        """BundleManifestError must be a subclass of BackupError."""
        assert issubclass(BundleManifestError, BackupError)

    def test_can_be_raised_as_backup_error(self) -> None:
        """BundleManifestError instances can be caught as BackupError."""
        with pytest.raises(BackupError):
            raise BundleManifestError("test error")


# ---------------------------------------------------------------------------
# BundleManifest round-trip
# ---------------------------------------------------------------------------


def _make_manifest(**overrides) -> BundleManifest:
    """Build a minimal valid BundleManifest, with optional field overrides."""
    defaults = dict(
        bundle_schema_version=BUNDLE_SCHEMA_VERSION,
        created_at="2026-05-20T20:55:00Z",
        tier="manual",
        label=None,
        live_registry_sha256="a" * 64,
        base_model={
            "repo": "mistralai/Mistral-7B-Instruct-v0.3",
            "sha": "abc123",
            "hash": "sha256:def",
        },
        files=[
            {
                "path": "config/server.yaml",
                "content_sha256": "b" * 64,
                "encrypted": False,
                "size_bytes": 1234,
            }
        ],
        adapters={
            "episodic": {
                "slot_source": "/data/ha/adapters/episodic/20260520-123456",
                "registry_sha256": "c" * 64,
                "key_count": 550,
                "simhash_present": True,
                "keyed_pairs_present": False,
            }
        },
        excluded=["graph (RAM-only by design)"],
    )
    defaults.update(overrides)
    return BundleManifest(**defaults)


class TestBundleManifestRoundTrip:
    def test_to_dict_produces_expected_keys(self) -> None:
        """to_dict() must contain all required manifest fields."""
        bm = _make_manifest()
        d = bm.to_dict()
        expected_keys = {
            "bundle_schema_version",
            "created_at",
            "tier",
            "label",
            "live_registry_sha256",
            "base_model",
            "files",
            "adapters",
            "excluded",
        }
        assert expected_keys == set(d.keys())

    def test_from_dict_round_trip(self) -> None:
        """from_dict(to_dict(m)) must produce an equal manifest."""
        bm = _make_manifest(label="smoke")
        restored = BundleManifest.from_dict(bm.to_dict())
        assert restored == bm

    def test_from_dict_with_label_none(self) -> None:
        """label=None survives the round-trip correctly."""
        bm = _make_manifest(label=None)
        restored = BundleManifest.from_dict(bm.to_dict())
        assert restored.label is None

    def test_from_dict_with_empty_adapters(self) -> None:
        """Empty adapters dict is valid (no enabled adapters)."""
        bm = _make_manifest(adapters={})
        restored = BundleManifest.from_dict(bm.to_dict())
        assert restored.adapters == {}

    def test_from_dict_with_multiple_files(self) -> None:
        """Multiple file entries survive the round-trip."""
        files = [
            {
                "path": "config/server.yaml",
                "content_sha256": "a" * 64,
                "encrypted": False,
                "size_bytes": 100,
            },
            {
                "path": "registry/key_metadata.json",
                "content_sha256": "b" * 64,
                "encrypted": True,
                "size_bytes": 200,
            },
            {
                "path": "adapters/episodic/adapter_model.safetensors",
                "content_sha256": "c" * 64,
                "encrypted": True,
                "size_bytes": 50_000_000,
            },
        ]
        bm = _make_manifest(files=files)
        restored = BundleManifest.from_dict(bm.to_dict())
        assert len(restored.files) == 3

    def test_to_dict_files_is_list_copy(self) -> None:
        """to_dict().files must be a new list (not shared reference)."""
        bm = _make_manifest()
        d = bm.to_dict()
        d["files"].append({"extra": True})
        # The original dataclass must not be affected (frozen=True).
        assert len(bm.files) == 1

    def test_frozen_dataclass_immutable(self) -> None:
        """BundleManifest is frozen — attribute assignment must raise."""
        bm = _make_manifest()
        with pytest.raises((AttributeError, TypeError)):
            bm.tier = "daily"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BundleManifest.from_dict — error paths
# ---------------------------------------------------------------------------


class TestBundleManifestFromDictErrors:
    def test_wrong_schema_version_raises(self) -> None:
        """from_dict with bundle_schema_version != 1 raises BundleManifestError."""
        bm = _make_manifest()
        d = bm.to_dict()
        d["bundle_schema_version"] = 999
        with pytest.raises(BundleManifestError, match="bundle_schema_version"):
            BundleManifest.from_dict(d)

    def test_none_schema_version_raises(self) -> None:
        """from_dict with bundle_schema_version=None raises BundleManifestError."""
        bm = _make_manifest()
        d = bm.to_dict()
        d["bundle_schema_version"] = None
        with pytest.raises(BundleManifestError):
            BundleManifest.from_dict(d)

    def test_missing_created_at_raises(self) -> None:
        """Missing 'created_at' field raises BundleManifestError."""
        bm = _make_manifest()
        d = bm.to_dict()
        del d["created_at"]
        with pytest.raises(BundleManifestError, match="created_at"):
            BundleManifest.from_dict(d)

    def test_missing_tier_raises(self) -> None:
        """Missing 'tier' field raises BundleManifestError."""
        bm = _make_manifest()
        d = bm.to_dict()
        del d["tier"]
        with pytest.raises(BundleManifestError, match="tier"):
            BundleManifest.from_dict(d)

    def test_missing_live_registry_sha256_raises(self) -> None:
        """Missing 'live_registry_sha256' raises BundleManifestError."""
        bm = _make_manifest()
        d = bm.to_dict()
        del d["live_registry_sha256"]
        with pytest.raises(BundleManifestError, match="live_registry_sha256"):
            BundleManifest.from_dict(d)

    def test_missing_files_raises(self) -> None:
        """Missing 'files' raises BundleManifestError."""
        bm = _make_manifest()
        d = bm.to_dict()
        del d["files"]
        with pytest.raises(BundleManifestError, match="files"):
            BundleManifest.from_dict(d)
