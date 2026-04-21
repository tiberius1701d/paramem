"""Tests for paramem.backup.atomic."""

from __future__ import annotations

import pytest

from paramem.backup.atomic import rename_pending_to_slot, sweep_orphan_pending


class TestRenamePendingToSlot:
    def test_happy_path(self, tmp_path):
        """pending_dir exists → rename → slot_dir exists, pending_dir is gone."""
        pending = tmp_path / ".pending" / "20260421-04000000"
        pending.mkdir(parents=True)
        (pending / "artifact.bin").write_bytes(b"data")

        slot_dir = tmp_path / "20260421-04000000"

        rename_pending_to_slot(pending, slot_dir)

        assert slot_dir.exists()
        assert (slot_dir / "artifact.bin").exists()
        assert not pending.exists()

    def test_rename_pending_to_slot_refuses_existing(self, tmp_path):
        """If slot_dir already exists, raises FileExistsError; pending_dir is untouched. (NIT 1)"""
        pending = tmp_path / ".pending" / "20260421-04000000"
        pending.mkdir(parents=True)
        (pending / "artifact.bin").write_bytes(b"data")

        slot_dir = tmp_path / "20260421-04000000"
        slot_dir.mkdir()  # pre-existing slot

        with pytest.raises(FileExistsError, match=str(slot_dir)):
            rename_pending_to_slot(pending, slot_dir)

        # pending_dir must be untouched
        assert pending.exists()
        assert (pending / "artifact.bin").exists()

    def test_error_message_includes_path(self, tmp_path):
        """FileExistsError message must include the conflicting slot path."""
        pending = tmp_path / ".pending" / "ts"
        pending.mkdir(parents=True)
        slot_dir = tmp_path / "ts"
        slot_dir.mkdir()

        with pytest.raises(FileExistsError) as exc_info:
            rename_pending_to_slot(pending, slot_dir)

        assert str(slot_dir) in str(exc_info.value)

    def test_rename_atomicity_under_injected_crash(self, tmp_path, monkeypatch):
        """Simulate process exit between fsync and rename.

        Injects an OSError from os.rename, then calls sweep_orphan_pending.
        Asserts:
        - No partial slot directory is promoted.
        - sweep_orphan_pending removes the pending residue.
        """
        import os

        kind_root = tmp_path / "config"
        kind_root.mkdir()
        pending_root = kind_root / ".pending"
        pending_root.mkdir()
        pending_slot = pending_root / "20260421-04000000"
        pending_slot.mkdir()
        (pending_slot / "artifact.bin").write_bytes(b"half-written")

        slot_dir = kind_root / "20260421-04000000"

        # Inject crash: os.rename raises before the slot is created
        original_rename = os.rename

        def failing_rename(src, dst):
            raise OSError("injected crash between fsync and rename")

        monkeypatch.setattr(os, "rename", failing_rename)

        with pytest.raises(OSError, match="injected crash"):
            rename_pending_to_slot(pending_slot, slot_dir)

        # Restore rename so sweep can use shutil.rmtree normally
        monkeypatch.setattr(os, "rename", original_rename)

        # No partial slot was created
        assert not slot_dir.exists()

        # sweep_orphan_pending should clean up the pending residue
        removed = sweep_orphan_pending(kind_root)
        assert len(removed) >= 1
        assert not pending_slot.exists()

    def test_rename_atomicity_both_files_under_injected_crash(self, tmp_path, monkeypatch):
        """Crash between fsync and rename with both artifact and sidecar present.

        Plants both ``artifact.bin`` and ``artifact.bin.meta.json`` in
        ``.pending/<ts>/``, injects a crash (OSError from os.rename), then
        asserts:
        - The slot directory (final promotion target) does not exist.
        - ``.pending/<ts>/`` still exists (incomplete pending residue).
        - ``sweep_orphan_pending`` removes ``.pending/<ts>/`` completely.
        - After sweep, ``.pending/`` contains no entries (or does not exist).
        """
        import os

        kind_root = tmp_path / "config"
        kind_root.mkdir()
        pending_root = kind_root / ".pending"
        pending_root.mkdir()
        pending_slot = pending_root / "20260421-04000099"
        pending_slot.mkdir()

        # Plant both artifact and sidecar (the "no artifact without sidecar" invariant)
        (pending_slot / "artifact.bin").write_bytes(b"real artifact data")
        (pending_slot / "artifact.bin.meta.json").write_text('{"schema_version": 1}')

        slot_dir = kind_root / "20260421-04000099"

        # Inject crash at os.rename
        original_rename = os.rename

        def failing_rename(src, dst):
            raise OSError("injected crash — both files present")

        monkeypatch.setattr(os, "rename", failing_rename)

        with pytest.raises(OSError, match="injected crash"):
            rename_pending_to_slot(pending_slot, slot_dir)

        # Restore rename so sweep can proceed
        monkeypatch.setattr(os, "rename", original_rename)

        # Slot directory must NOT exist (no partial promotion)
        assert not slot_dir.exists(), "slot must not exist after injected crash"

        # Pending slot must still exist (residue for sweep to clean)
        assert pending_slot.exists(), ".pending/<ts>/ residue must remain until swept"

        # sweep removes the entire pending slot directory
        removed = sweep_orphan_pending(kind_root)
        assert len(removed) >= 1, "sweep must report at least one removal"
        assert not pending_slot.exists(), ".pending/<ts>/ must be gone after sweep"

        # No orphaned artifact or sidecar survives under .pending/
        if pending_root.exists():
            remaining = list(pending_root.iterdir())
            assert remaining == [], f"No entries should remain in .pending/: {remaining}"


class TestSweepOrphanPending:
    def test_sweep_removes_pending_subdirs(self, tmp_path):
        """Pending entries are removed by sweep."""
        kind_root = tmp_path / "graph"
        kind_root.mkdir()
        pending = kind_root / ".pending"
        pending.mkdir()

        orphan1 = pending / "20260421-04000001"
        orphan1.mkdir()
        (orphan1 / "artifact.bin").write_bytes(b"orphan1")

        orphan2 = pending / "20260421-04000002"
        orphan2.mkdir()
        (orphan2 / "artifact.bin").write_bytes(b"orphan2")

        removed = sweep_orphan_pending(kind_root)

        assert len(removed) == 2
        assert not orphan1.exists()
        assert not orphan2.exists()

    def test_sweep_orphan_pending_removes_only_pending(self, tmp_path):
        """Live slot directories are not touched by sweep."""
        kind_root = tmp_path / "registry"
        kind_root.mkdir()
        pending = kind_root / ".pending"
        pending.mkdir()

        orphan = pending / "20260421-04000000"
        orphan.mkdir()

        # Live slot (should survive)
        live_slot = kind_root / "20260421-03000000"
        live_slot.mkdir()
        (live_slot / "artifact.bin").write_bytes(b"live")

        sweep_orphan_pending(kind_root)

        # Live slot is untouched
        assert live_slot.exists()
        assert not orphan.exists()

    def test_sweep_orphan_pending_empty_root(self, tmp_path):
        """No .pending/ directory → returns [] without error."""
        kind_root = tmp_path / "config"
        kind_root.mkdir()

        result = sweep_orphan_pending(kind_root)

        assert result == []

    def test_sweep_nonexistent_root(self, tmp_path):
        """Nonexistent kind_root → returns [] without error."""
        result = sweep_orphan_pending(tmp_path / "nonexistent")
        assert result == []
