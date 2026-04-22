"""Tests for paramem.backup.enumerate (Slice 3b.2).

Covers:
- enumerate_backups returns newest-first.
- Filter by kind.
- .pending/ directories are skipped.
- Slots with invalid sidecars are skipped with a WARN (not raised).
- pre_trial_hash is extracted from sidecar.
- Empty list when base_dir does not exist.
"""

from __future__ import annotations

from pathlib import Path

from paramem.backup.backup import write as backup_write
from paramem.backup.encryption import SecurityBackupsConfig
from paramem.backup.enumerate import BackupRecord, enumerate_backups
from paramem.backup.types import ArtifactKind

_SEC = SecurityBackupsConfig()


def _write_slot(
    base_dir: Path,
    kind: ArtifactKind,
    data: bytes = b"test",
    tier: str = "manual",
    pre_trial_hash: str | None = None,
    label: str | None = None,
) -> Path:
    """Write a backup slot and return its slot directory path."""
    meta_fields: dict = {"tier": tier}
    if pre_trial_hash is not None:
        meta_fields["pre_trial_hash"] = pre_trial_hash
    if label is not None:
        meta_fields["label"] = label
    return backup_write(
        kind,
        data,
        meta_fields=meta_fields,
        base_dir=base_dir / kind.value,
        security_config=_SEC,
    )


# ---------------------------------------------------------------------------
# Missing base_dir
# ---------------------------------------------------------------------------


class TestMissingBaseDir:
    def test_enumerate_missing_base_dir(self, tmp_path):
        """base_dir does not exist → empty list."""
        result = enumerate_backups(tmp_path / "nonexistent")
        assert result == []


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestEnumerateHappyPath:
    def test_enumerate_returns_newest_first(self, tmp_path):
        """3 slots → returns timestamp-desc (newest first)."""
        base = tmp_path / "backups"
        _write_slot(base, ArtifactKind.CONFIG, b"a")
        _write_slot(base, ArtifactKind.CONFIG, b"b")
        _write_slot(base, ArtifactKind.CONFIG, b"c")
        records = enumerate_backups(base, kind=ArtifactKind.CONFIG)
        assert len(records) == 3
        timestamps = [r.timestamp for r in records]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_enumerate_returns_backup_records(self, tmp_path):
        """Records have expected fields."""
        base = tmp_path / "backups"
        _write_slot(base, ArtifactKind.REGISTRY, b"data", tier="scheduled")
        records = enumerate_backups(base, kind=ArtifactKind.REGISTRY)
        assert len(records) == 1
        rec = records[0]
        assert isinstance(rec, BackupRecord)
        assert rec.kind == ArtifactKind.REGISTRY
        assert rec.slot_dir.is_absolute()
        assert rec.content_sha256  # non-empty hex string
        assert rec.created_at is not None


# ---------------------------------------------------------------------------
# Kind filter
# ---------------------------------------------------------------------------


class TestKindFilter:
    def test_enumerate_filter_by_kind(self, tmp_path):
        """base_dir with config + graph slots; filter=CONFIG → only config."""
        base = tmp_path / "backups"
        _write_slot(base, ArtifactKind.CONFIG, b"cfg")
        _write_slot(base, ArtifactKind.GRAPH, b"graph")
        config_records = enumerate_backups(base, kind=ArtifactKind.CONFIG)
        assert len(config_records) == 1
        assert config_records[0].kind == ArtifactKind.CONFIG

    def test_enumerate_no_filter_returns_all_kinds(self, tmp_path):
        """kind=None → all kinds."""
        base = tmp_path / "backups"
        _write_slot(base, ArtifactKind.CONFIG, b"cfg")
        _write_slot(base, ArtifactKind.GRAPH, b"graph")
        _write_slot(base, ArtifactKind.REGISTRY, b"reg")
        all_records = enumerate_backups(base)
        kinds = {r.kind for r in all_records}
        assert ArtifactKind.CONFIG in kinds
        assert ArtifactKind.GRAPH in kinds
        assert ArtifactKind.REGISTRY in kinds


# ---------------------------------------------------------------------------
# .pending/ skip
# ---------------------------------------------------------------------------


class TestPendingSkip:
    def test_enumerate_skips_pending_dir(self, tmp_path):
        """A .pending/ directory inside a kind directory is ignored."""
        base = tmp_path / "backups"
        _write_slot(base, ArtifactKind.CONFIG, b"real")
        # Create a fake .pending/20260422-010000/ inside the config kind dir
        pending = base / "config" / ".pending" / "20260422-010000"
        pending.mkdir(parents=True)
        (pending / "garbage.bin").write_bytes(b"incomplete")
        records = enumerate_backups(base, kind=ArtifactKind.CONFIG)
        # Only the real slot should appear.
        assert len(records) == 1


# ---------------------------------------------------------------------------
# Invalid sidecar — skipped with WARN
# ---------------------------------------------------------------------------


class TestInvalidSidecar:
    def test_enumerate_skips_invalid_sidecar(self, tmp_path):
        """Slot with broken meta.json → logged WARN, skipped (not raised)."""
        base = tmp_path / "backups"
        _write_slot(base, ArtifactKind.CONFIG, b"good")
        # Create a slot directory with a corrupt sidecar.
        bad_slot = base / "config" / "20260101-000000"
        bad_slot.mkdir(parents=True)
        (bad_slot / "config-20260101-000000.meta.json").write_text(
            "not json at all", encoding="utf-8"
        )
        records = enumerate_backups(base, kind=ArtifactKind.CONFIG)
        # Only the good slot should appear.
        assert len(records) == 1
        assert records[0].kind == ArtifactKind.CONFIG


# ---------------------------------------------------------------------------
# pre_trial_hash
# ---------------------------------------------------------------------------


class TestPreTrialHash:
    def test_enumerate_pre_trial_hash_extracted(self, tmp_path):
        """Slot written with pre_trial_hash → record.pre_trial_hash equals the value."""
        base = tmp_path / "backups"
        expected_hash = "a" * 64
        _write_slot(
            base, ArtifactKind.CONFIG, b"cfg", tier="pre_migration", pre_trial_hash=expected_hash
        )
        records = enumerate_backups(base, kind=ArtifactKind.CONFIG)
        assert len(records) == 1
        assert records[0].pre_trial_hash == expected_hash

    def test_enumerate_pre_trial_hash_none_when_absent(self, tmp_path):
        """Slot without pre_trial_hash → record.pre_trial_hash is None."""
        base = tmp_path / "backups"
        _write_slot(base, ArtifactKind.CONFIG, b"cfg", tier="manual")
        records = enumerate_backups(base, kind=ArtifactKind.CONFIG)
        assert len(records) == 1
        assert records[0].pre_trial_hash is None
