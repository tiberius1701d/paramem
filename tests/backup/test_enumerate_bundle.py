"""Tests for Slice 4 — bundle-aware enumerate_backups.

Covers:
- A slot with bundle.meta.json is returned as kind=SNAPSHOT_BUNDLE.
- A slot with bundle.meta.json is NOT flagged as invalid/unreadable.
- enumerate_backups with kind=SNAPSHOT_BUNDLE returns only bundles.
- Mixed store (per-kind + bundle) → all returned when kind=None.
- Bundle slot with corrupt bundle.meta.json → skipped with WARN, not raised.
- prune does not classify bundle slots as invalid (B2 fix).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from paramem.backup.backup import write as backup_write
from paramem.backup.enumerate import enumerate_backups
from paramem.backup.types import BUNDLE_SCHEMA_VERSION, ArtifactKind

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_BUNDLE_MANIFEST: dict[str, Any] = {
    "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
    "created_at": "2026-05-20T20:55:00Z",
    "tier": "manual",
    "label": None,
    "live_registry_sha256": "a" * 64,
    "base_model": {},
    "files": [],
    "adapters": {},
    "excluded": [],
}


def _make_bundle_slot(
    backups_root: Path,
    ts: str = "20260520-20550000",
    manifest: dict | None = None,
) -> Path:
    """Create a minimal bundle slot directory with a bundle.meta.json.

    Parameters
    ----------
    backups_root:
        The backup store root (e.g. ``data/ha/backups/``).
    ts:
        Slot timestamp in ``YYYYMMDD-HHMMSSff`` format.
    manifest:
        Override the default valid manifest dict.

    Returns
    -------
    Path
        The slot directory ``backups_root/snapshot_bundle/<ts>/``.
    """
    slot = backups_root / ArtifactKind.SNAPSHOT_BUNDLE.value / ts
    slot.mkdir(parents=True, exist_ok=True)
    m = manifest if manifest is not None else _VALID_BUNDLE_MANIFEST.copy()
    (slot / "bundle.meta.json").write_text(json.dumps(m), encoding="utf-8")
    return slot


def _make_regular_slot(backups_root: Path) -> Path:
    """Write a regular per-artifact config slot."""
    return backup_write(
        ArtifactKind.CONFIG,
        b"model: mistral\n",
        meta_fields={"tier": "daily"},
        base_dir=backups_root / "config",
    )


# ---------------------------------------------------------------------------
# Basic bundle detection
# ---------------------------------------------------------------------------


class TestEnumerateBundleDetection:
    def test_bundle_slot_returned_as_snapshot_bundle(self, tmp_path) -> None:
        """A slot with bundle.meta.json is enumerated with kind=SNAPSHOT_BUNDLE."""
        base = tmp_path / "backups"
        _make_bundle_slot(base)
        records = enumerate_backups(base, kind=ArtifactKind.SNAPSHOT_BUNDLE)
        assert len(records) == 1
        assert records[0].kind == ArtifactKind.SNAPSHOT_BUNDLE

    def test_bundle_record_is_bundle_flag(self, tmp_path) -> None:
        """BackupRecord.is_bundle is True for bundle slots."""
        base = tmp_path / "backups"
        _make_bundle_slot(base)
        records = enumerate_backups(base, kind=ArtifactKind.SNAPSHOT_BUNDLE)
        assert records[0].is_bundle is True

    def test_regular_slot_is_bundle_false(self, tmp_path) -> None:
        """BackupRecord.is_bundle is False for per-artifact slots."""
        base = tmp_path / "backups"
        _make_regular_slot(base)
        records = enumerate_backups(base, kind=ArtifactKind.CONFIG)
        assert len(records) == 1
        assert records[0].is_bundle is False

    def test_bundle_record_tier_from_manifest(self, tmp_path) -> None:
        """meta.tier is taken from bundle.meta.json (not a per-artifact sidecar)."""
        base = tmp_path / "backups"
        m = _VALID_BUNDLE_MANIFEST.copy()
        m["tier"] = "pre_migration"
        _make_bundle_slot(base, manifest=m)
        records = enumerate_backups(base, kind=ArtifactKind.SNAPSHOT_BUNDLE)
        assert records[0].meta.tier == "pre_migration"

    def test_bundle_record_label_from_manifest(self, tmp_path) -> None:
        """record.label is taken from bundle.meta.json."""
        base = tmp_path / "backups"
        m = _VALID_BUNDLE_MANIFEST.copy()
        m["label"] = "pre-restore-safety"
        _make_bundle_slot(base, manifest=m)
        records = enumerate_backups(base, kind=ArtifactKind.SNAPSHOT_BUNDLE)
        assert records[0].label == "pre-restore-safety"

    def test_bundle_pre_trial_hash_always_none(self, tmp_path) -> None:
        """pre_trial_hash is always None for bundle slots."""
        base = tmp_path / "backups"
        _make_bundle_slot(base)
        records = enumerate_backups(base)
        bundle_records = [r for r in records if r.is_bundle]
        assert len(bundle_records) == 1
        assert bundle_records[0].pre_trial_hash is None

    def test_bundle_slot_dir_is_absolute(self, tmp_path) -> None:
        """record.slot_dir must be an absolute path."""
        base = tmp_path / "backups"
        _make_bundle_slot(base)
        records = enumerate_backups(base, kind=ArtifactKind.SNAPSHOT_BUNDLE)
        assert records[0].slot_dir.is_absolute()


# ---------------------------------------------------------------------------
# Kind filter
# ---------------------------------------------------------------------------


class TestKindFilter:
    def test_filter_snapshot_bundle_returns_only_bundles(self, tmp_path) -> None:
        """kind=SNAPSHOT_BUNDLE → only bundle slots returned."""
        base = tmp_path / "backups"
        _make_bundle_slot(base, ts="20260520-20550000")
        _make_regular_slot(base)  # config slot
        records = enumerate_backups(base, kind=ArtifactKind.SNAPSHOT_BUNDLE)
        assert len(records) == 1
        assert all(r.kind == ArtifactKind.SNAPSHOT_BUNDLE for r in records)

    def test_filter_config_returns_only_regular(self, tmp_path) -> None:
        """kind=CONFIG → only config slots returned, not bundle slots."""
        base = tmp_path / "backups"
        _make_bundle_slot(base)
        _make_regular_slot(base)
        records = enumerate_backups(base, kind=ArtifactKind.CONFIG)
        assert all(r.kind == ArtifactKind.CONFIG for r in records)
        assert not any(r.is_bundle for r in records)

    def test_no_filter_returns_all_kinds(self, tmp_path) -> None:
        """kind=None → all slot kinds returned, including bundles."""
        base = tmp_path / "backups"
        _make_bundle_slot(base)
        _make_regular_slot(base)
        records = enumerate_backups(base)
        kinds = {r.kind for r in records}
        assert ArtifactKind.SNAPSHOT_BUNDLE in kinds
        assert ArtifactKind.CONFIG in kinds


# ---------------------------------------------------------------------------
# Multiple bundle slots — newest-first ordering
# ---------------------------------------------------------------------------


class TestMultipleBundleSlots:
    def test_newest_bundle_first(self, tmp_path) -> None:
        """Multiple bundle slots are returned newest-first."""
        base = tmp_path / "backups"
        _make_bundle_slot(base, ts="20260518-12000000")
        _make_bundle_slot(base, ts="20260519-12000000")
        _make_bundle_slot(base, ts="20260520-12000000")
        records = enumerate_backups(base, kind=ArtifactKind.SNAPSHOT_BUNDLE)
        assert len(records) == 3
        ts_list = [r.timestamp for r in records]
        assert ts_list == sorted(ts_list, reverse=True)


# ---------------------------------------------------------------------------
# Bundle slot with corrupt manifest → skipped with WARN
# ---------------------------------------------------------------------------


class TestCorruptBundleManifest:
    def test_corrupt_manifest_skipped_not_raised(self, tmp_path) -> None:
        """A slot with a corrupt bundle.meta.json is skipped (not returned)."""
        base = tmp_path / "backups"
        slot = base / "snapshot_bundle" / "20260520-20550000"
        slot.mkdir(parents=True)
        (slot / "bundle.meta.json").write_text("not json at all", encoding="utf-8")

        records = enumerate_backups(base, kind=ArtifactKind.SNAPSHOT_BUNDLE)
        assert len(records) == 0

    def test_corrupt_manifest_logs_warning(self, tmp_path, caplog) -> None:
        """Corrupt bundle.meta.json produces a WARNING log."""
        base = tmp_path / "backups"
        slot = base / "snapshot_bundle" / "20260520-20550000"
        slot.mkdir(parents=True)
        (slot / "bundle.meta.json").write_text("{}", encoding="utf-8")  # missing required fields

        named_logger = logging.getLogger("paramem.backup.enumerate")
        orig_propagate = named_logger.propagate
        named_logger.propagate = True
        caplog.set_level(logging.WARNING, logger="paramem.backup.enumerate")
        named_logger.addHandler(caplog.handler)
        try:
            enumerate_backups(base)
        finally:
            named_logger.removeHandler(caplog.handler)
            named_logger.propagate = orig_propagate

        warn_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warn_records, "Expected a WARNING log for corrupt bundle manifest, got none."

    def test_wrong_schema_version_skipped(self, tmp_path) -> None:
        """Bundle manifest with wrong schema version is skipped (not raised)."""
        base = tmp_path / "backups"
        slot = base / "snapshot_bundle" / "20260520-20550000"
        slot.mkdir(parents=True)
        m = _VALID_BUNDLE_MANIFEST.copy()
        m["bundle_schema_version"] = 999  # unknown version
        (slot / "bundle.meta.json").write_text(json.dumps(m), encoding="utf-8")

        records = enumerate_backups(base)
        assert len(records) == 0


# ---------------------------------------------------------------------------
# Prune does NOT classify bundle slots as invalid (B2 fix)
# ---------------------------------------------------------------------------


class TestPruneDoesNotFlagBundleSlotInvalid:
    def test_prune_sees_bundle_slot_as_valid(self, tmp_path) -> None:
        """prune() must not add bundle slots to invalid_slots.

        Before the B2 fix, read_meta raised MetaSchemaError on a bundle slot
        because there was no per-artifact .meta.json, and prune added the slot
        to invalid_slots.  After the fix, enumerate_backups correctly reads
        bundle slots, so prune iterates valid records.
        """
        from paramem.backup.retention import prune
        from paramem.server.config import (
            RetentionConfig,
            RetentionTierConfig,
            ServerBackupsConfig,
        )

        config = ServerBackupsConfig(
            max_total_disk_gb=20.0,
            retention=RetentionConfig(
                manual=RetentionTierConfig(keep="unlimited"),
            ),
        )

        backups_root = tmp_path / "backups"
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        _make_bundle_slot(backups_root)

        result = prune(
            backups_root=backups_root,
            state_dir=state_dir,
            config=config,
            dry_run=True,
        )

        assert result.invalid_slots == [], (
            f"prune classified bundle slot as invalid: {result.invalid_slots}. "
            "B2 regression: bundle slots must not be marked invalid."
        )
