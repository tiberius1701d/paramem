"""Tests for ``paramem rotate-recovery``.

Covers the happy path with confirmation, refuse-without-confirm (both
interactive mismatch and non-interactive without ``--yes``), dry-run
cleanup, resume refusal, and the invariant that the daily identity still
decrypts post-rotation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pyrage
import pytest
from pyrage import x25519

from paramem.backup.age_envelope import age_decrypt_bytes, age_encrypt_bytes
from paramem.backup.encryption import (
    MASTER_KEY_ENV_VAR,
    _clear_cipher_cache,
)
from paramem.backup.key_store import (
    DAILY_PASSPHRASE_ENV_VAR,
    _clear_daily_identity_cache,
    load_recovery_recipient,
    mint_daily_identity,
    wrap_daily_identity,
    write_daily_key_file,
    write_recovery_pub_file,
)
from paramem.backup.rotation import RotationManifest, write_manifest_atomic
from paramem.cli import rotate_recovery


@pytest.fixture(autouse=True)
def _isolate_env_and_caches():
    os.environ.pop(MASTER_KEY_ENV_VAR, None)
    os.environ.pop(DAILY_PASSPHRASE_ENV_VAR, None)
    _clear_cipher_cache()
    _clear_daily_identity_cache()
    yield
    os.environ.pop(MASTER_KEY_ENV_VAR, None)
    os.environ.pop(DAILY_PASSPHRASE_ENV_VAR, None)
    _clear_cipher_cache()
    _clear_daily_identity_cache()


def _setup_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passphrase: str = "pw"):
    daily = mint_daily_identity()
    recovery = x25519.Identity.generate()
    daily_path = tmp_path / "daily_key.age"
    recovery_path = tmp_path / "recovery.pub"
    write_daily_key_file(wrap_daily_identity(daily, passphrase), daily_path)
    write_recovery_pub_file(recovery.to_public(), recovery_path)
    monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, passphrase)
    monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", daily_path)
    monkeypatch.setattr("paramem.backup.key_store.RECOVERY_PUB_PATH_DEFAULT", recovery_path)
    return daily, recovery


def _seed_age_files(data_dir: Path, daily: x25519.Identity, recovery: x25519.Identity):
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "adapters").mkdir(parents=True, exist_ok=True)
    recipients = [daily.to_public(), recovery.to_public()]
    seeded = []
    for rel, content in [
        (Path("registry.json"), b'{"a": 1}'),
        (Path("speaker_profiles.json"), b'{"s": 2}'),
    ]:
        p = data_dir / rel
        p.write_bytes(age_encrypt_bytes(content, recipients))
        seeded.append((p, content))
    return seeded


def _default_args(data_dir: Path, *, dry_run=False, verbose=False, yes=True) -> argparse.Namespace:
    return argparse.Namespace(
        data_dir=str(data_dir),
        config="configs/server.yaml",
        dry_run=dry_run,
        verbose=verbose,
        yes=yes,
    )


class TestHappyPath:
    def test_rotates_to_new_recovery_daily_still_decrypts(self, tmp_path, monkeypatch, capsys):
        daily, recovery = _setup_keys(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        seeded = _seed_age_files(data_dir, daily, recovery)

        rc = rotate_recovery.run(_default_args(data_dir, verbose=True))
        assert rc == 0

        # recovery.pub on disk has changed.
        new_recipient = load_recovery_recipient(tmp_path / "recovery.pub")
        assert str(new_recipient) != str(recovery.to_public())

        for path, original in seeded:
            # Daily still decrypts.
            assert age_decrypt_bytes(path.read_bytes(), [daily]) == original
            # Old recovery no longer decrypts.
            with pytest.raises(pyrage.DecryptError):
                age_decrypt_bytes(path.read_bytes(), [recovery])

        err = capsys.readouterr().err
        assert "NEW RECOVERY KEY" in err
        assert "AGE-SECRET-KEY-1" in err

    def test_pending_and_manifest_cleaned_up(self, tmp_path, monkeypatch, capsys):
        daily, recovery = _setup_keys(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        _seed_age_files(data_dir, daily, recovery)

        rc = rotate_recovery.run(_default_args(data_dir))
        assert rc == 0
        assert not (tmp_path / "recovery.pub.pending").exists()
        assert not (tmp_path / "rotation.manifest.json").exists()

    def test_dry_run_emits_secret_but_mutates_nothing(self, tmp_path, monkeypatch, capsys):
        daily, recovery = _setup_keys(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        seeded = _seed_age_files(data_dir, daily, recovery)
        before = {p: p.read_bytes() for p, _ in seeded}
        recovery_before = (tmp_path / "recovery.pub").read_text("utf-8")

        rc = rotate_recovery.run(_default_args(data_dir, dry_run=True, verbose=True))
        assert rc == 0

        for p, _ in seeded:
            assert p.read_bytes() == before[p]
        assert (tmp_path / "recovery.pub").read_text("utf-8") == recovery_before
        assert not (tmp_path / "recovery.pub.pending").exists()
        assert not (tmp_path / "rotation.manifest.json").exists()

        err = capsys.readouterr().err
        assert "AGE-SECRET-KEY-1" in err


class TestConfirmation:
    def test_refuses_non_interactive_without_yes(self, tmp_path, monkeypatch, capsys):
        daily, recovery = _setup_keys(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        seeded = _seed_age_files(data_dir, daily, recovery)
        before = {p: p.read_bytes() for p, _ in seeded}
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        rc = rotate_recovery.run(_default_args(data_dir, yes=False))
        assert rc == 1
        err = capsys.readouterr().err
        assert "non-interactively without --yes" in err
        # Files untouched.
        for p, _ in seeded:
            assert p.read_bytes() == before[p]
        assert not (tmp_path / "recovery.pub.pending").exists()
        assert not (tmp_path / "rotation.manifest.json").exists()

    def test_interactive_happy_path(self, tmp_path, monkeypatch, capsys):
        daily, recovery = _setup_keys(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        _seed_age_files(data_dir, daily, recovery)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda: "I have saved the recovery key")

        rc = rotate_recovery.run(_default_args(data_dir, yes=False))
        assert rc == 0

    def test_interactive_mismatch_refuses(self, tmp_path, monkeypatch, capsys):
        daily, recovery = _setup_keys(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        seeded = _seed_age_files(data_dir, daily, recovery)
        before = {p: p.read_bytes() for p, _ in seeded}
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda: "wrong answer")

        rc = rotate_recovery.run(_default_args(data_dir, yes=False))
        assert rc == 1
        for p, _ in seeded:
            assert p.read_bytes() == before[p]


class TestResumeRefused:
    def test_pre_existing_manifest_refused(self, tmp_path, monkeypatch, capsys):
        """rotate-recovery cannot be resumed because the new secret was
        print-once. Refuse with operator guidance."""
        daily, recovery = _setup_keys(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        _seed_age_files(data_dir, daily, recovery)

        manifest = RotationManifest(
            operation="rotate-recovery",
            started_at="2026-04-24T12:00:00Z",
            new_recovery_pub="age1xxx",
            files_pending=[],
            files_done=[],
        )
        write_manifest_atomic(tmp_path / "rotation.manifest.json", manifest)

        rc = rotate_recovery.run(_default_args(data_dir))
        assert rc == 1
        err = capsys.readouterr().err
        assert "cannot be resumed" in err
        assert "rotation.manifest.json" in err


class TestPreconditions:
    def test_refuses_without_daily_identity(self, tmp_path, monkeypatch, capsys):
        _setup_keys(tmp_path, monkeypatch)
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        rc = rotate_recovery.run(_default_args(data_dir))
        assert rc == 1
        assert "Daily identity is not loadable" in capsys.readouterr().err

    def test_refuses_without_prior_recovery_pub(self, tmp_path, monkeypatch, capsys):
        _setup_keys(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "paramem.backup.key_store.RECOVERY_PUB_PATH_DEFAULT",
            tmp_path / "absent.pub",
        )
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        rc = rotate_recovery.run(_default_args(data_dir))
        assert rc == 1
        err = capsys.readouterr().err
        assert "Recovery public recipient is missing" in err
        assert "`paramem generate-key`" in err
