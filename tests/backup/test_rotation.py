"""Tests for the rotation manifest + per-file re-encrypt helper."""

from __future__ import annotations

import json
from pathlib import Path

import pyrage
import pytest
from pyrage import x25519

from paramem.backup.age_envelope import age_decrypt_bytes, age_encrypt_bytes, is_age_envelope
from paramem.backup.rotation import (
    RotationManifest,
    delete_manifest,
    finalise_pending_rename,
    read_manifest,
    rotate_file_to_recipients,
    write_manifest_atomic,
)


class TestManifestRoundTrip:
    def test_fresh_manifest_carries_iso_timestamp(self) -> None:
        m = RotationManifest.fresh(
            operation="rotate-daily",
            files=[Path("/x"), Path("/y")],
            new_daily_pub="age1xxx",
        )
        assert m.operation == "rotate-daily"
        assert m.new_daily_pub == "age1xxx"
        assert m.files_pending == ["/x", "/y"]
        assert m.files_done == []
        # ISO Z-suffixed.
        assert m.started_at.endswith("Z")
        assert "T" in m.started_at

    def test_write_then_read(self, tmp_path: Path) -> None:
        m = RotationManifest.fresh(
            operation="rotate-recovery",
            files=[Path("/a")],
            new_recovery_pub="age1yyy",
        )
        p = tmp_path / "manifest.json"
        write_manifest_atomic(p, m)
        loaded = read_manifest(p)
        assert loaded is not None
        assert loaded.operation == "rotate-recovery"
        assert loaded.new_recovery_pub == "age1yyy"
        assert loaded.files_pending == ["/a"]

    def test_read_missing_returns_none(self, tmp_path: Path) -> None:
        assert read_manifest(tmp_path / "absent.json") is None

    def test_write_is_atomic_no_tmp_leak(self, tmp_path: Path) -> None:
        m = RotationManifest.fresh(operation="rotate-daily", files=[])
        p = tmp_path / "manifest.json"
        write_manifest_atomic(p, m)
        assert p.exists()
        # The atomic writer stages under <path>.tmp — that must not linger.
        assert not p.with_suffix(p.suffix + ".tmp").exists()

    def test_delete_missing_is_a_noop(self, tmp_path: Path) -> None:
        delete_manifest(tmp_path / "nope.json")  # must not raise

    def test_delete_removes_file(self, tmp_path: Path) -> None:
        p = tmp_path / "manifest.json"
        p.write_text("{}", encoding="utf-8")
        delete_manifest(p)
        assert not p.exists()

    def test_body_is_deterministic_sorted_json(self, tmp_path: Path) -> None:
        """Stable byte output so a no-op rewrite does not trip file-watch logic."""
        m = RotationManifest.fresh(operation="rotate-daily", files=[Path("/b"), Path("/a")])
        p = tmp_path / "manifest.json"
        write_manifest_atomic(p, m)
        body = json.loads(p.read_text("utf-8"))
        # Sorted keys: files_done, files_pending, new_daily_pub, new_recovery_pub,
        # operation, started_at.
        assert list(body.keys()) == sorted(body.keys())


class TestRotateFileToRecipients:
    def test_basic_re_encrypt(self, tmp_path: Path) -> None:
        old = x25519.Identity.generate()
        new = x25519.Identity.generate()
        p = tmp_path / "file"
        p.write_bytes(age_encrypt_bytes(b"payload", [old.to_public()]))

        rotate_file_to_recipients(
            p,
            decrypt_identities=[old],
            new_recipients=[new.to_public()],
        )
        assert is_age_envelope(p)
        # Old cannot decrypt anymore; new can.
        with pytest.raises(pyrage.DecryptError):
            age_decrypt_bytes(p.read_bytes(), [old])
        assert age_decrypt_bytes(p.read_bytes(), [new]) == b"payload"

    def test_multi_identity_decrypt_handles_resume(self, tmp_path: Path) -> None:
        """On resume, an already-rotated file has the NEW recipient. Supplying
        BOTH OLD and NEW as decrypt identities lets the command skip re-work
        without a DecryptError."""
        old = x25519.Identity.generate()
        new = x25519.Identity.generate()
        recovery = x25519.Identity.generate()
        p = tmp_path / "file"
        # Already rotated: envelope is [new, recovery].
        p.write_bytes(age_encrypt_bytes(b"already", [new.to_public(), recovery.to_public()]))

        rotate_file_to_recipients(
            p,
            decrypt_identities=[old, new],
            new_recipients=[new.to_public(), recovery.to_public()],
        )
        assert age_decrypt_bytes(p.read_bytes(), [new]) == b"already"

    def test_wrong_identity_raises_decrypt_error(self, tmp_path: Path) -> None:
        a = x25519.Identity.generate()
        b = x25519.Identity.generate()
        c = x25519.Identity.generate()
        p = tmp_path / "file"
        p.write_bytes(age_encrypt_bytes(b"x", [a.to_public()]))

        with pytest.raises(pyrage.DecryptError):
            rotate_file_to_recipients(
                p,
                decrypt_identities=[b],  # wrong
                new_recipients=[c.to_public()],
            )

    def test_new_recipients_preserved_on_output(self, tmp_path: Path) -> None:
        """Both the new daily and the recovery recipient must be on the post-
        rotation envelope — a recovery key must still open the file."""
        old = x25519.Identity.generate()
        new = x25519.Identity.generate()
        recovery = x25519.Identity.generate()
        p = tmp_path / "file"
        p.write_bytes(age_encrypt_bytes(b"payload", [old.to_public(), recovery.to_public()]))

        rotate_file_to_recipients(
            p,
            decrypt_identities=[old],
            new_recipients=[new.to_public(), recovery.to_public()],
        )
        assert age_decrypt_bytes(p.read_bytes(), [new]) == b"payload"
        assert age_decrypt_bytes(p.read_bytes(), [recovery]) == b"payload"


class TestFinalisePendingRename:
    def test_atomic_swap(self, tmp_path: Path) -> None:
        canonical = tmp_path / "daily_key.age"
        pending = tmp_path / "daily_key.age.pending"
        canonical.write_bytes(b"OLD")
        pending.write_bytes(b"NEW")

        finalise_pending_rename(pending, canonical)
        assert canonical.read_bytes() == b"NEW"
        assert not pending.exists()

    def test_refuses_cross_directory_rename(self, tmp_path: Path) -> None:
        """Atomicity at the POSIX level requires same parent."""
        pending = tmp_path / "a" / "pending"
        canonical = tmp_path / "b" / "canonical"
        pending.parent.mkdir()
        canonical.parent.mkdir()
        pending.write_bytes(b"x")
        with pytest.raises(ValueError, match="share a parent"):
            finalise_pending_rename(pending, canonical)


class TestCrashSafety:
    def test_per_file_rename_is_atomic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulate an encrypt-raise after the read succeeds. The source must
        remain intact (no partial tmp left, no truncation)."""
        from paramem.backup import rotation as rot_mod

        old = x25519.Identity.generate()
        new = x25519.Identity.generate()
        p = tmp_path / "file"
        original = age_encrypt_bytes(b"secret", [old.to_public()])
        p.write_bytes(original)

        def boom(*args, **kwargs):
            raise RuntimeError("simulated crash")

        monkeypatch.setattr(rot_mod, "age_encrypt_bytes", boom)

        with pytest.raises(RuntimeError, match="simulated"):
            rotate_file_to_recipients(
                p,
                decrypt_identities=[old],
                new_recipients=[new.to_public()],
            )

        assert p.read_bytes() == original, "source must be untouched on encrypt failure"
        tmp = p.with_suffix(p.suffix + ".tmp")
        assert not tmp.exists(), "tmp must be cleaned up"
