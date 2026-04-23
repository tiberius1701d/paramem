"""Tests for paramem.backup.backup.prune()."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from paramem.backup.backup import _parse_slot_timestamp, prune
from paramem.backup.encryption import _clear_cipher_cache
from paramem.backup.meta import write_meta
from paramem.backup.types import (
    SCHEMA_VERSION,
    ArtifactKind,
    ArtifactMeta,
    EncryptAtRest,
)


def _make_meta(kind: ArtifactKind, timestamp: str, **overrides) -> ArtifactMeta:
    defaults = dict(
        schema_version=SCHEMA_VERSION,
        kind=kind,
        timestamp=timestamp,
        content_sha256="a" * 64,
        size_bytes=16,
        encrypted=False,
        encrypt_at_rest=EncryptAtRest.NEVER,
        key_fingerprint=None,
        tier="scheduled",
        label=None,
    )
    defaults.update(overrides)
    return ArtifactMeta(**defaults)


def _make_slot(base_dir: Path, kind: ArtifactKind, timestamp: str, **meta_overrides) -> Path:
    """Create a minimal valid slot directory for prune testing."""
    slot_dir = base_dir / timestamp
    slot_dir.mkdir(parents=True, exist_ok=True)
    artifact = slot_dir / f"{kind.value}-{timestamp}.bin"
    artifact.write_bytes(b"x" * 16)
    write_meta(slot_dir, _make_meta(kind, timestamp, **meta_overrides))
    return slot_dir


class TestPruneRespect:
    def setup_method(self):
        _clear_cipher_cache()
        os.environ.pop("PARAMEM_MASTER_KEY", None)

    def test_prune_respects_keep_counts(self, tmp_path):
        """Prune with keep=2 retains the 2 newest, deletes the rest."""
        base_dir = tmp_path / "config"
        slot_a = _make_slot(base_dir, ArtifactKind.CONFIG, "20260101-12000000")
        slot_b = _make_slot(base_dir, ArtifactKind.CONFIG, "20260102-12000000")
        slot_c = _make_slot(base_dir, ArtifactKind.CONFIG, "20260103-12000000")

        report = prune(
            ArtifactKind.CONFIG,
            {"keep": 2},
            base_dir=base_dir,
            live_slot=None,
        )

        # Newest two kept
        assert slot_b in report.kept or slot_c in report.kept
        assert slot_c in report.kept

        # Oldest deleted
        assert slot_a in report.deleted
        assert not slot_a.exists()

        # Kept slots still exist
        for s in report.kept:
            assert s.exists()

    def test_prune_unlimited_keeps_all(self, tmp_path):
        """keep='unlimited' means no slots are deleted."""
        base_dir = tmp_path / "graph"
        for i in range(5):
            _make_slot(base_dir, ArtifactKind.GRAPH, f"20260101-1200000{i}")

        report = prune(
            ArtifactKind.GRAPH,
            {"keep": "unlimited"},
            base_dir=base_dir,
            live_slot=None,
        )

        assert report.deleted == []
        assert len(report.kept) == 5

    def test_prune_never_deletes_live_slot(self, tmp_path):
        """live_slot is skipped even when retention would cull it."""
        base_dir = tmp_path / "registry"
        slot_old = _make_slot(base_dir, ArtifactKind.REGISTRY, "20260101-12000000")
        _make_slot(base_dir, ArtifactKind.REGISTRY, "20260102-12000000")

        # keep=1 would delete the older slot, but we mark old as live
        report = prune(
            ArtifactKind.REGISTRY,
            {"keep": 1},
            base_dir=base_dir,
            live_slot=slot_old,
        )

        assert slot_old in report.skipped_live
        assert slot_old in report.kept
        assert slot_old.exists()

    def test_prune_is_idempotent(self, tmp_path):
        """Running prune twice with the same policy produces the same result."""
        base_dir = tmp_path / "config"
        _make_slot(base_dir, ArtifactKind.CONFIG, "20260101-12000000")
        _make_slot(base_dir, ArtifactKind.CONFIG, "20260102-12000000")
        _make_slot(base_dir, ArtifactKind.CONFIG, "20260103-12000000")

        policy = {"keep": 2}

        report1 = prune(ArtifactKind.CONFIG, policy, base_dir=base_dir, live_slot=None)
        report2 = prune(ArtifactKind.CONFIG, policy, base_dir=base_dir, live_slot=None)

        # Second run: nothing to delete (already at keep=2)
        assert report2.deleted == []
        assert len(report2.kept) == len(report1.kept)

    def test_prune_returns_report_with_invalid_slots(self, tmp_path):
        """Slots with corrupt sidecars land in report.invalid with a reason string."""
        base_dir = tmp_path / "config"

        # Valid slot (creates the directory; not directly referenced after creation)
        _make_slot(base_dir, ArtifactKind.CONFIG, "20260103-12000000")

        # Corrupt slot — sidecar is not valid JSON
        corrupt_slot = base_dir / "20260101-12000000"
        corrupt_slot.mkdir()
        (corrupt_slot / "config-20260101-12000000.meta.json").write_text("not json !!!")

        report = prune(
            ArtifactKind.CONFIG,
            {"keep": 10},
            base_dir=base_dir,
            live_slot=None,
        )

        # Invalid slot reported with a reason string
        invalid_paths = [p for p, _reason in report.invalid]
        assert corrupt_slot in invalid_paths

        # Reason must be a non-empty string
        for p, reason in report.invalid:
            assert isinstance(reason, str)
            assert reason

        # Corrupt slot not deleted
        assert corrupt_slot.exists()

    def test_prune_empty_base_dir(self, tmp_path):
        """No slots → empty report, no error."""
        base_dir = tmp_path / "empty"
        base_dir.mkdir()

        report = prune(
            ArtifactKind.CONFIG,
            {"keep": 5},
            base_dir=base_dir,
            live_slot=None,
        )

        assert report.kept == []
        assert report.deleted == []
        assert report.invalid == []

    def test_prune_nonexistent_base_dir(self, tmp_path):
        """Nonexistent base_dir → empty report, no error."""
        report = prune(
            ArtifactKind.CONFIG,
            {"keep": 5},
            base_dir=tmp_path / "does_not_exist",
            live_slot=None,
        )

        assert report.kept == []
        assert report.deleted == []

    def test_prune_live_slot_in_skipped_live(self, tmp_path):
        """live_slot appears in skipped_live and kept, never in deleted."""
        base_dir = tmp_path / "graph"
        slots = [_make_slot(base_dir, ArtifactKind.GRAPH, f"20260101-1200000{i}") for i in range(3)]
        live = slots[0]

        report = prune(
            ArtifactKind.GRAPH,
            {"keep": 1},  # would normally keep only 1
            base_dir=base_dir,
            live_slot=live,
        )

        assert live in report.skipped_live
        assert live in report.kept
        assert live not in report.deleted


def _dt_to_slot_name(dt: datetime) -> str:
    """Convert a UTC datetime to the ``YYYYMMDD-HHMMSSff`` slot-name format."""
    hh = dt.microsecond // 10000
    return dt.strftime("%Y%m%d-%H%M%S") + f"{hh:02d}"


class TestParseSlotTimestamp:
    """Unit tests for the _parse_slot_timestamp private helper (Fix #1)."""

    def test_parse_slot_timestamp_valid_format(self):
        """A well-formed 17-char slot name returns a UTC datetime, not None."""
        result = _parse_slot_timestamp("20260421-04000012")
        assert result is not None
        assert isinstance(result, datetime)
        assert result.tzinfo is timezone.utc
        assert result.year == 2026
        assert result.month == 4
        assert result.day == 21
        assert result.hour == 4
        assert result.minute == 0
        assert result.second == 0

    def test_parse_slot_timestamp_rejects_18_chars(self):
        """A string of 18 chars (the old off-by-one) is rejected — returns None."""
        result = _parse_slot_timestamp("20260421-040000123")  # 18 chars
        assert result is None

    def test_parse_slot_timestamp_rejects_malformed(self):
        """Non-timestamp strings return None without raising."""
        assert _parse_slot_timestamp("") is None
        assert _parse_slot_timestamp("not-a-date") is None
        assert _parse_slot_timestamp("20260421_04000012") is None  # underscore not dash


class TestPruneImmunityDays:
    """Tests that prune() honours immunity_days (Fix #1 — was silently broken)."""

    def setup_method(self):
        _clear_cipher_cache()
        os.environ.pop("PARAMEM_MASTER_KEY", None)

    def test_prune_respects_immunity_days(self, tmp_path):
        """Immune slots are exempt from the tier count; keep budget applies to the tail.

        Spec §Backup retention policy: "pre_migration entries within the last
        30 days are immune from the tier-count rule (rule 3). The keep: 10
        applies only to the >30-day tail."

        Five slots, keep=1, immunity_days=30:
        - Slots A, B: 50 and 40 days ago (outside window) — non-immune tail.
        - Slots C, D, E: inside window (today, 2d, 5d) — immune.

        Expected: all 3 immune slots kept. Of the 2 non-immune slots, the
        newer one (B @ 40d) fits within keep=1 tail budget; the older (A @
        50d) is pruned.
        """
        base_dir = tmp_path / "config"

        now = datetime.now(tz=timezone.utc)
        ts_a = _dt_to_slot_name(now - timedelta(days=50))
        ts_b = _dt_to_slot_name(now - timedelta(days=40))
        ts_c = _dt_to_slot_name(now - timedelta(days=5))
        ts_d = _dt_to_slot_name(now - timedelta(days=2))
        ts_e = _dt_to_slot_name(now)

        slot_a = _make_slot(base_dir, ArtifactKind.CONFIG, ts_a)
        slot_b = _make_slot(base_dir, ArtifactKind.CONFIG, ts_b)
        slot_c = _make_slot(base_dir, ArtifactKind.CONFIG, ts_c)
        slot_d = _make_slot(base_dir, ArtifactKind.CONFIG, ts_d)
        slot_e = _make_slot(base_dir, ArtifactKind.CONFIG, ts_e)

        report = prune(
            ArtifactKind.CONFIG,
            {"keep": 1, "immunity_days": 30},
            base_dir=base_dir,
            live_slot=None,
        )

        # All three immune slots kept regardless of keep budget
        assert slot_c in report.kept, "immune slot (5d old) must be kept"
        assert slot_d in report.kept, "immune slot (2d old) must be kept"
        assert slot_e in report.kept, "immune slot (today) must be kept"
        # Non-immune tail: keep=1 applies — newer non-immune survives, oldest pruned
        assert slot_b in report.kept, "newest non-immune slot must be kept under keep=1"
        assert slot_a in report.deleted, "oldest non-immune slot must be pruned"

    def test_prune_immune_slots_do_not_consume_keep_budget(self, tmp_path):
        """Many immune slots must not starve the non-immune tail.

        Spec: immunity is orthogonal to the tier count. With keep=1 and 4
        immune slots + 2 non-immune slots, all 4 immune kept + 1 newest
        non-immune kept = 5 total; only the oldest non-immune is pruned.
        """
        base_dir = tmp_path / "registry"

        now = datetime.now(tz=timezone.utc)
        immune_ts = [_dt_to_slot_name(now - timedelta(hours=h)) for h in (0, 1, 2, 3)]
        old_ts = [
            _dt_to_slot_name(now - timedelta(days=40)),
            _dt_to_slot_name(now - timedelta(days=50)),
        ]

        immune_slots = [_make_slot(base_dir, ArtifactKind.REGISTRY, ts) for ts in immune_ts]
        old_new = _make_slot(base_dir, ArtifactKind.REGISTRY, old_ts[0])
        old_older = _make_slot(base_dir, ArtifactKind.REGISTRY, old_ts[1])

        report = prune(
            ArtifactKind.REGISTRY,
            {"keep": 1, "immunity_days": 30},
            base_dir=base_dir,
            live_slot=None,
        )

        for s in immune_slots:
            assert s in report.kept, f"immune slot {s.name} must be kept"
        assert old_new in report.kept, "newest non-immune slot fits keep=1 tail"
        assert old_older in report.deleted, "oldest non-immune slot pruned"

    def test_prune_immunity_overrides_keep_for_all_immune_slots(self, tmp_path):
        """Three slots all inside the immunity window with keep=1 — all three kept.

        When the immune set exceeds ``keep``, immunity wins: no slot younger
        than immunity_days is ever deleted.
        """
        base_dir = tmp_path / "graph"

        now = datetime.now(tz=timezone.utc)
        ts_a = _dt_to_slot_name(now - timedelta(hours=2))
        ts_b = _dt_to_slot_name(now - timedelta(hours=1))
        ts_c = _dt_to_slot_name(now)

        slot_a = _make_slot(base_dir, ArtifactKind.GRAPH, ts_a)
        slot_b = _make_slot(base_dir, ArtifactKind.GRAPH, ts_b)
        slot_c = _make_slot(base_dir, ArtifactKind.GRAPH, ts_c)

        report = prune(
            ArtifactKind.GRAPH,
            {"keep": 1, "immunity_days": 30},
            base_dir=base_dir,
            live_slot=None,
        )

        # All three are within the 30-day immunity window — none deleted
        assert slot_a in report.kept
        assert slot_b in report.kept
        assert slot_c in report.kept
        assert report.deleted == []


class TestPruneMissingArtifact:
    """Fix #1 — prune() must flag partial slots (valid sidecar, missing artifact) as invalid."""

    def setup_method(self):
        _clear_cipher_cache()
        os.environ.pop("PARAMEM_MASTER_KEY", None)

    def test_prune_flags_missing_artifact_as_invalid(self, tmp_path):
        """Slot with valid sidecar but missing artifact file lands in report.invalid.

        The slot must NOT appear in report.kept.  The reason string must
        contain both "artifact" and "missing".
        """
        base_dir = tmp_path / "config"

        # Valid slot (will be kept)
        _make_slot(base_dir, ArtifactKind.CONFIG, "20260103-12000000")

        # Partial slot: write sidecar only, no artifact file
        partial_ts = "20260102-12000000"
        partial_slot = base_dir / partial_ts
        partial_slot.mkdir(parents=True, exist_ok=True)
        # Write sidecar without the artifact file
        write_meta(partial_slot, _make_meta(ArtifactKind.CONFIG, partial_ts))
        # Deliberately do NOT create the artifact file

        report = prune(
            ArtifactKind.CONFIG,
            {"keep": 10},
            base_dir=base_dir,
            live_slot=None,
        )

        invalid_paths = [p for p, _reason in report.invalid]
        assert partial_slot in invalid_paths, (
            "partial slot (valid sidecar, missing artifact) must appear in report.invalid"
        )

        # Reason must reference "artifact" and "missing"
        for p, reason in report.invalid:
            if p == partial_slot:
                assert "artifact" in reason.lower(), f"reason must mention 'artifact': {reason!r}"
                assert "missing" in reason.lower(), f"reason must mention 'missing': {reason!r}"
                break

        # Partial slot must NOT appear in kept
        assert partial_slot not in report.kept, "partial slot must not appear in report.kept"

        # Partial slot is not deleted (operator visibility, not auto-remediation)
        assert partial_slot.exists(), "prune must not delete partial slots automatically"


class TestPruneMaxDiskGb:
    """Fix #4 — prune() must enforce per-tier max_disk_gb cap."""

    def setup_method(self):
        _clear_cipher_cache()
        os.environ.pop("PARAMEM_MASTER_KEY", None)

    def _make_sized_slot(
        self, base_dir: Path, kind: ArtifactKind, timestamp: str, size_bytes: int
    ) -> Path:
        """Create a slot whose artifact file is exactly *size_bytes* bytes."""
        slot_dir = base_dir / timestamp
        slot_dir.mkdir(parents=True, exist_ok=True)
        artifact = slot_dir / f"{kind.value}-{timestamp}.bin"
        artifact.write_bytes(b"x" * size_bytes)
        write_meta(slot_dir, _make_meta(kind, timestamp, size_bytes=size_bytes))
        return slot_dir

    def test_prune_enforces_max_disk_gb(self, tmp_path):
        """Oldest slots are pruned until total kept size is within max_disk_gb.

        Three slots of 1 MB each (3 MB total).  Set max_disk_gb so that only
        ~1.5 MB is allowed (1.5 / 1024 GB).  Expected: the oldest slot(s)
        are deleted until total is within cap; newest is always kept.
        """
        base_dir = tmp_path / "config"
        size = 1024 * 1024  # 1 MB per slot

        slot_old = self._make_sized_slot(base_dir, ArtifactKind.CONFIG, "20260101-12000000", size)
        self._make_sized_slot(base_dir, ArtifactKind.CONFIG, "20260102-12000000", size)
        slot_new = self._make_sized_slot(base_dir, ArtifactKind.CONFIG, "20260103-12000000", size)

        # Cap at 1.5 MB — only one 1 MB slot fits comfortably
        max_disk_gb = (size * 1.5) / (1024**3)
        report = prune(
            ArtifactKind.CONFIG,
            {"keep": "unlimited", "max_disk_gb": max_disk_gb},
            base_dir=base_dir,
            live_slot=None,
        )

        # Oldest slot must be deleted to get under cap
        assert slot_old in report.deleted, "oldest slot must be pruned under disk pressure"
        # Newest slot must always be retained
        assert slot_new in report.kept, "newest slot must survive disk pruning"

        # All reported-deleted slots must actually be gone from disk
        for s in report.deleted:
            assert not s.exists(), f"deleted slot {s} must be removed from disk"

    def test_prune_disk_pressure_overrides_immunity(self, tmp_path, monkeypatch):
        """Immune slots are deleted when total size exceeds max_disk_gb.

        Spec: disk pressure (rule 2) overrides immunity (rule 4).
        Two immune slots both exceeding the cap → oldest immune slot deleted,
        and a WARN log is emitted that includes "immune" and the slot path.
        """
        base_dir = tmp_path / "registry"
        now = datetime.now(tz=timezone.utc)
        size = 1024 * 1024  # 1 MB each

        # Both slots are within the immunity window (created "now")
        ts_old = _dt_to_slot_name(now - timedelta(hours=2))
        ts_new = _dt_to_slot_name(now - timedelta(hours=1))

        slot_old = self._make_sized_slot(base_dir, ArtifactKind.REGISTRY, ts_old, size)
        self._make_sized_slot(base_dir, ArtifactKind.REGISTRY, ts_new, size)

        # Cap at 0.5 MB — both 1 MB slots exceed it; oldest must go
        max_disk_gb = (size * 0.5) / (1024**3)

        # Capture WARNING calls on the backup module's logger directly
        warned: list[str] = []
        import paramem.backup.backup as backup_mod

        original_warning = backup_mod.logger.warning

        def _capture_warning(msg, *args):
            warned.append(msg % args if args else msg)
            original_warning(msg, *args)

        monkeypatch.setattr(backup_mod.logger, "warning", _capture_warning)

        report = prune(
            ArtifactKind.REGISTRY,
            {"keep": "unlimited", "immunity_days": 30, "max_disk_gb": max_disk_gb},
            base_dir=base_dir,
            live_slot=None,
        )

        # Oldest immune slot must be deleted under disk pressure
        assert slot_old in report.deleted, (
            "oldest immune slot must be deleted under max_disk_gb pressure"
        )
        # WARN log must mention "immune" and the slot path
        assert any("immune" in m.lower() and str(slot_old) in m for m in warned), (
            f"Expected WARN log for immune slot deletion; got: {warned}"
        )

    def test_prune_disk_pressure_never_deletes_live_slot(self, tmp_path):
        """live_slot survives disk-pressure pruning even if it's the oldest."""
        base_dir = tmp_path / "graph"
        size = 1024 * 1024  # 1 MB each

        slot_live = self._make_sized_slot(base_dir, ArtifactKind.GRAPH, "20260101-12000000", size)
        self._make_sized_slot(base_dir, ArtifactKind.GRAPH, "20260102-12000000", size)
        self._make_sized_slot(base_dir, ArtifactKind.GRAPH, "20260103-12000000", size)

        # Cap below the size of a single slot — extreme pressure
        max_disk_gb = (size * 0.5) / (1024**3)

        report = prune(
            ArtifactKind.GRAPH,
            {"keep": "unlimited", "max_disk_gb": max_disk_gb},
            base_dir=base_dir,
            live_slot=slot_live,
        )

        # live_slot must never be deleted, even under extreme disk pressure
        assert slot_live not in report.deleted, "live_slot must survive disk-pressure pruning"
        assert slot_live.exists(), "live_slot directory must still exist on disk"
