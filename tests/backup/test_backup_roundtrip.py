"""Tests for paramem.backup.backup — write/read round-trip."""

from __future__ import annotations

import os
import stat
from datetime import datetime

import pytest

from paramem.backup.backup import read, write
from paramem.backup.encryption import SecurityBackupsConfig, _clear_cipher_cache
from paramem.backup.types import (
    SCHEMA_VERSION,
    ArtifactKind,
    BackupError,
    EncryptAtRest,
    FatalConfigError,
    FingerprintMismatchError,
)


def _make_fernet_key() -> bytes:
    from cryptography.fernet import Fernet

    return Fernet.generate_key()


class TestWriteReadRoundtripPlain:
    def setup_method(self):
        _clear_cipher_cache()
        os.environ.pop("PARAMEM_MASTER_KEY", None)

    def teardown_method(self):
        _clear_cipher_cache()
        os.environ.pop("PARAMEM_MASTER_KEY", None)

    def test_write_read_roundtrip_plain(self, tmp_path):
        """No key env — bytes in → slot written → read() returns identical bytes + meta."""
        payload = b"plain backup data"
        config = SecurityBackupsConfig(encrypt_at_rest=EncryptAtRest.NEVER)

        slot_dir = write(
            ArtifactKind.CONFIG,
            payload,
            {"tier": "scheduled"},
            base_dir=tmp_path / "backups" / "config",
            security_config=config,
        )

        assert slot_dir.is_dir()
        plaintext, meta = read(slot_dir)

        assert plaintext == payload
        assert meta.encrypted is False
        assert meta.kind == ArtifactKind.CONFIG
        assert meta.tier == "scheduled"
        assert meta.schema_version == SCHEMA_VERSION

    def test_write_read_roundtrip_encrypted(self, tmp_path):
        """Key set → sidecar encrypted=True → read() decrypts and verifies."""
        key = _make_fernet_key()
        os.environ["PARAMEM_MASTER_KEY"] = key.decode()

        payload = b"encrypted backup data"
        config = SecurityBackupsConfig(encrypt_at_rest=EncryptAtRest.ALWAYS)

        slot_dir = write(
            ArtifactKind.GRAPH,
            payload,
            {"tier": "manual"},
            base_dir=tmp_path / "backups" / "graph",
            security_config=config,
        )

        plaintext, meta = read(slot_dir)

        assert plaintext == payload
        assert meta.encrypted is True
        assert meta.key_fingerprint is not None
        assert len(meta.key_fingerprint) == 16

    def test_write_refuses_encrypt_always_without_key(self, tmp_path):
        """encrypt_at_rest=always + missing key → raises before any file is written."""
        os.environ.pop("PARAMEM_MASTER_KEY", None)
        config = SecurityBackupsConfig(encrypt_at_rest=EncryptAtRest.ALWAYS)

        with pytest.raises(FatalConfigError):
            write(
                ArtifactKind.REGISTRY,
                b"data",
                {"tier": "scheduled"},
                base_dir=tmp_path / "backups" / "registry",
                security_config=config,
            )

        # No slot directory should have been created
        slots_root = tmp_path / "backups" / "registry"
        if slots_root.exists():
            slots = [d for d in slots_root.iterdir() if not d.name.startswith(".")]
            assert len(slots) == 0

    def test_read_refuses_on_content_hash_drift(self, tmp_path):
        """Corrupt artifact on disk after write → read() raises FingerprintMismatchError."""
        config = SecurityBackupsConfig(encrypt_at_rest=EncryptAtRest.NEVER)

        slot_dir = write(
            ArtifactKind.CONFIG,
            b"original data",
            {"tier": "scheduled"},
            base_dir=tmp_path / "backups" / "config",
            security_config=config,
        )

        # Corrupt the artifact file
        artifact = [f for f in slot_dir.iterdir() if not f.name.endswith(".meta.json")][0]
        artifact.write_bytes(b"corrupted!")

        with pytest.raises(FingerprintMismatchError):
            read(slot_dir)

    def test_read_refuses_on_missing_sidecar(self, tmp_path):
        """Slot directory with no .meta.json raises FileNotFoundError."""
        slot_dir = tmp_path / "orphan_slot"
        slot_dir.mkdir()
        (slot_dir / "artifact.bin").write_bytes(b"orphan")

        with pytest.raises(FileNotFoundError):
            read(slot_dir)

    def test_write_with_path_source(self, tmp_path):
        """write() accepts a Path source as well as bytes."""
        src_file = tmp_path / "source.bin"
        src_file.write_bytes(b"from file")
        config = SecurityBackupsConfig(encrypt_at_rest=EncryptAtRest.NEVER)

        slot_dir = write(
            ArtifactKind.REGISTRY,
            src_file,
            {"tier": "manual"},
            base_dir=tmp_path / "backups" / "registry",
            security_config=config,
        )

        plaintext, _ = read(slot_dir)
        assert plaintext == b"from file"

    def test_write_creates_unique_slot_per_call(self, tmp_path):
        """Two sequential write() calls produce two distinct slot directories."""
        import time

        config = SecurityBackupsConfig(encrypt_at_rest=EncryptAtRest.NEVER)
        base = tmp_path / "backups" / "config"

        slot1 = write(
            ArtifactKind.CONFIG, b"a", {"tier": "manual"}, base_dir=base, security_config=config
        )
        time.sleep(0.02)  # ensure different hundredths-of-a-second timestamp
        slot2 = write(
            ArtifactKind.CONFIG, b"b", {"tier": "manual"}, base_dir=base, security_config=config
        )

        assert slot1 != slot2
        assert slot1.is_dir()
        assert slot2.is_dir()


class TestWriteFsyncsParentDirAfterRename:
    """Fix #4 — verify os.fsync is called on at least one directory fd after rename."""

    def setup_method(self):
        _clear_cipher_cache()
        os.environ.pop("PARAMEM_MASTER_KEY", None)

    def teardown_method(self):
        _clear_cipher_cache()
        os.environ.pop("PARAMEM_MASTER_KEY", None)

    def test_write_fsyncs_parent_dir_after_rename(self, tmp_path, monkeypatch):
        """write() calls os.fsync on at least one directory fd after the rename.

        Strategy: replace os.fsync with a recording shim that captures whether
        each fd refers to a directory via os.fstat + stat.S_ISDIR.  After
        write() returns assert that:
        1. At least 4 fsync calls occurred (belt-and-braces count check).
        2. At least one call was on a directory fd (proves directory fsync).
        """
        import paramem.backup.backup as backup_mod

        fsync_calls: list[int] = []
        is_dir_per_call: list[bool] = []
        original_fsync = os.fsync

        def recording_fsync(fd: int) -> None:
            fsync_calls.append(fd)
            try:
                st = os.fstat(fd)
                is_dir_per_call.append(stat.S_ISDIR(st.st_mode))
            except OSError:
                is_dir_per_call.append(False)
            original_fsync(fd)

        monkeypatch.setattr(backup_mod.os, "fsync", recording_fsync)

        config = SecurityBackupsConfig(encrypt_at_rest=EncryptAtRest.NEVER)
        write(
            ArtifactKind.CONFIG,
            b"durability test payload",
            {"tier": "manual"},
            base_dir=tmp_path / "backups" / "config",
            security_config=config,
        )

        # Belt-and-braces count: 1 artifact + 1 sidecar + 1 pending-dir + >=1 parent-dir
        assert len(fsync_calls) >= 4, (
            f"Expected at least 4 fsync calls (artifact, sidecar, pending dir, "
            f"parent dir after rename), got {len(fsync_calls)}"
        )
        # Core assertion: at least one fsync call was on a directory fd.
        assert any(is_dir_per_call), (
            "Expected at least one os.fsync call on a directory fd for rename "
            f"durability; recorded is_dir flags: {is_dir_per_call}"
        )


class TestWriteReadRoundtrip:
    """Fix #1 — read() must raise FingerprintMismatchError for partial slots."""

    def setup_method(self):
        _clear_cipher_cache()
        os.environ.pop("PARAMEM_MASTER_KEY", None)

    def teardown_method(self):
        _clear_cipher_cache()
        os.environ.pop("PARAMEM_MASTER_KEY", None)

    def test_read_refuses_on_missing_artifact(self, tmp_path):
        """Slot with valid sidecar but deleted artifact raises FingerprintMismatchError.

        This tests the partial-slot invariant: a sidecar present without its
        paired artifact is an integrity violation, not just a missing file.
        """
        config = SecurityBackupsConfig(encrypt_at_rest=EncryptAtRest.NEVER)
        base = tmp_path / "backups" / "config"

        slot_dir = write(
            ArtifactKind.CONFIG,
            b"payload that will go missing",
            {"tier": "scheduled"},
            base_dir=base,
            security_config=config,
        )

        # Delete the artifact file, leaving the sidecar intact
        artifact = [f for f in slot_dir.iterdir() if not f.name.endswith(".meta.json")][0]
        artifact.unlink()

        with pytest.raises(FingerprintMismatchError, match="artifact file missing"):
            read(slot_dir)


class TestWriteConcurrency:
    """Fix #5 — write() retries on timestamp collision and raises after max attempts."""

    def setup_method(self):
        _clear_cipher_cache()
        os.environ.pop("PARAMEM_MASTER_KEY", None)

    def teardown_method(self):
        _clear_cipher_cache()
        os.environ.pop("PARAMEM_MASTER_KEY", None)

    def test_write_retries_on_timestamp_collision(self, tmp_path, monkeypatch):
        """Two writes with the same initial timestamp succeed and produce distinct slots.

        Monkeypatches datetime.now (via the backup module's datetime) to return
        the same value for the first N calls, then a different value.  Both
        write() calls must succeed and produce distinct slot directories.
        """
        from datetime import timezone

        import paramem.backup.backup as backup_mod

        fixed_dt = datetime(2026, 4, 21, 4, 0, 0, 500_000, tzinfo=timezone.utc)
        call_count = [0]
        original_datetime = backup_mod.datetime

        class _PatchedDatetime:
            @staticmethod
            def now(tz=None):
                call_count[0] += 1
                # Return fixed time for first 3 calls, then a distinct time
                if call_count[0] <= 3:
                    return fixed_dt
                return original_datetime.now(tz=tz)

        monkeypatch.setattr(backup_mod, "datetime", _PatchedDatetime)

        config = SecurityBackupsConfig(encrypt_at_rest=EncryptAtRest.NEVER)
        base = tmp_path / "backups" / "config"

        slot1 = write(
            ArtifactKind.CONFIG, b"first", {"tier": "manual"}, base_dir=base, security_config=config
        )
        # Reset counter so second write also starts from the fixed time
        call_count[0] = 0
        slot2 = write(
            ArtifactKind.CONFIG,
            b"second",
            {"tier": "manual"},
            base_dir=base,
            security_config=config,
        )

        assert slot1.is_dir()
        assert slot2.is_dir()
        assert slot1 != slot2, "Two writes with colliding timestamps must yield distinct slots"

    def test_write_raises_after_max_collision_retries(self, tmp_path, monkeypatch):
        """write() raises BackupError when all 10 collision-retry attempts fail.

        Strategy: monkeypatch datetime.now to return a base time, then
        pre-create the 10 candidate slot directories (base + 10 bumps of
        10ms) so every retry attempt finds its final-slot dir already occupied.
        The write() call must exhaust all 10 attempts and raise BackupError.
        """
        from datetime import timezone

        import paramem.backup.backup as backup_mod

        base_dt = datetime(2026, 4, 21, 4, 0, 0, 0, tzinfo=timezone.utc)

        class _FixedDatetime:
            @staticmethod
            def now(tz=None):
                return base_dt

        monkeypatch.setattr(backup_mod, "datetime", _FixedDatetime)

        config = SecurityBackupsConfig(encrypt_at_rest=EncryptAtRest.NEVER)
        base = tmp_path / "backups" / "config"
        base.mkdir(parents=True, exist_ok=True)

        # Pre-create the 10 final slot directories that the retry loop would try
        # (base_dt hh=00 plus bumps 1..9 → hh=01..09)
        from datetime import timedelta

        for i in range(10):
            bumped = base_dt + timedelta(microseconds=10_000 * i)
            hh = bumped.microsecond // 10000
            slot_name = bumped.strftime("%Y%m%d-%H%M%S") + f"{hh:02d}"
            (base / slot_name).mkdir()

        with pytest.raises(BackupError, match="unique pending slot"):
            write(
                ArtifactKind.CONFIG,
                b"should fail",
                {"tier": "manual"},
                base_dir=base,
                security_config=config,
            )
