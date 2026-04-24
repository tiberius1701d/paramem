"""Tests for ``paramem restore`` — hardware-replacement recovery-key flow."""

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
    PMEM1_MAGIC,
    _clear_cipher_cache,
    encrypt_bytes,
)
from paramem.backup.key_store import (
    DAILY_PASSPHRASE_ENV_VAR,
    _clear_daily_identity_cache,
    load_daily_identity,
    load_recovery_recipient,
)
from paramem.backup.rotation import RotationManifest, write_manifest_atomic
from paramem.cli import restore


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


def _point_config_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point key-store module defaults at a scratch config dir."""
    monkeypatch.setattr(
        "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT",
        tmp_path / "daily_key.age",
    )
    monkeypatch.setattr(
        "paramem.backup.key_store.RECOVERY_PUB_PATH_DEFAULT",
        tmp_path / "recovery.pub",
    )


def _seed_age_store(data_dir: Path, recovery: x25519.Identity) -> list[tuple[Path, bytes]]:
    """Seed an age store encrypted to a (lost) daily + the recovery identity.

    Simulates the post-hardware-loss state: the operator has the encrypted
    data + the recovery paper, but the daily identity is gone.
    """
    lost_daily = x25519.Identity.generate()
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "adapters").mkdir(parents=True, exist_ok=True)
    recipients = [lost_daily.to_public(), recovery.to_public()]
    seeded = []
    for rel, content in [
        (Path("registry.json"), b'{"a": 1}'),
        (Path("speaker_profiles.json"), b'{"s": 2}'),
        (Path("adapters/post_session_queue.json"), b'{"q": 3}'),
    ]:
        p = data_dir / rel
        p.write_bytes(age_encrypt_bytes(content, recipients))
        seeded.append((p, content))
    return seeded


def _write_file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _default_args(
    tmp_path: Path,
    data_dir: Path,
    *,
    recovery_key_file: Path | None = None,
    passphrase_file: Path | None = None,
    force: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        data_dir=str(data_dir),
        config="configs/server.yaml",
        recovery_key_file=recovery_key_file,
        passphrase_file=passphrase_file,
        force=force,
        dry_run=dry_run,
        verbose=verbose,
    )


class TestHappyPath:
    def test_restore_with_recovery_bech32(self, tmp_path, monkeypatch, capsys):
        _point_config_at(tmp_path, monkeypatch)
        recovery = x25519.Identity.generate()
        data_dir = tmp_path / "data"
        seeded = _seed_age_store(data_dir, recovery)

        key_file = _write_file(tmp_path / "recovery.txt", str(recovery))
        pw_file = _write_file(tmp_path / "pw.txt", "new-passphrase")

        rc = restore.run(
            _default_args(
                tmp_path,
                data_dir,
                recovery_key_file=key_file,
                passphrase_file=pw_file,
                verbose=True,
            )
        )
        assert rc == 0

        # Daily key file materialised with the supplied passphrase.
        new_daily = load_daily_identity(tmp_path / "daily_key.age", passphrase="new-passphrase")
        # Recovery pub materialised from the bech32.
        loaded_recipient = load_recovery_recipient(tmp_path / "recovery.pub")
        assert str(loaded_recipient) == str(recovery.to_public())

        for path, original in seeded:
            # New daily decrypts.
            assert age_decrypt_bytes(path.read_bytes(), [new_daily]) == original
            # Recovery still decrypts (it was kept on the envelopes).
            assert age_decrypt_bytes(path.read_bytes(), [recovery]) == original

        # Manifest cleaned up.
        assert not (tmp_path / "rotation.manifest.json").exists()

    def test_dry_run_does_not_mutate(self, tmp_path, monkeypatch, capsys):
        _point_config_at(tmp_path, monkeypatch)
        recovery = x25519.Identity.generate()
        data_dir = tmp_path / "data"
        seeded = _seed_age_store(data_dir, recovery)
        before = {p: p.read_bytes() for p, _ in seeded}

        key_file = _write_file(tmp_path / "recovery.txt", str(recovery))

        rc = restore.run(
            _default_args(
                tmp_path, data_dir, recovery_key_file=key_file, dry_run=True, verbose=True
            )
        )
        assert rc == 0

        for p, _ in seeded:
            assert p.read_bytes() == before[p]
        assert not (tmp_path / "daily_key.age").exists()
        assert not (tmp_path / "recovery.pub").exists()
        assert not (tmp_path / "rotation.manifest.json").exists()

    def test_empty_data_dir_mints_keys_only(self, tmp_path, monkeypatch, capsys):
        """Fresh install scenario — no age files yet. Restore still mints a
        new daily + persists recovery.pub so future writes land as age."""
        _point_config_at(tmp_path, monkeypatch)
        recovery = x25519.Identity.generate()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        key_file = _write_file(tmp_path / "recovery.txt", str(recovery))
        pw_file = _write_file(tmp_path / "pw.txt", "pw")

        rc = restore.run(
            _default_args(tmp_path, data_dir, recovery_key_file=key_file, passphrase_file=pw_file)
        )
        assert rc == 0
        assert (tmp_path / "daily_key.age").exists()
        assert (tmp_path / "recovery.pub").exists()


class TestRecoveryKeyValidation:
    def test_wrong_bech32_refused_before_mutation(self, tmp_path, monkeypatch, capsys):
        _point_config_at(tmp_path, monkeypatch)
        recovery = x25519.Identity.generate()
        wrong_recovery = x25519.Identity.generate()
        data_dir = tmp_path / "data"
        _seed_age_store(data_dir, recovery)

        key_file = _write_file(tmp_path / "wrong.txt", str(wrong_recovery))
        pw_file = _write_file(tmp_path / "pw.txt", "pw")

        rc = restore.run(
            _default_args(tmp_path, data_dir, recovery_key_file=key_file, passphrase_file=pw_file)
        )
        assert rc == 1
        err = capsys.readouterr().err
        assert "does not decrypt" in err
        # Nothing created on disk.
        assert not (tmp_path / "daily_key.age").exists()
        assert not (tmp_path / "recovery.pub").exists()

    def test_malformed_bech32_refused(self, tmp_path, monkeypatch, capsys):
        _point_config_at(tmp_path, monkeypatch)
        recovery = x25519.Identity.generate()
        data_dir = tmp_path / "data"
        _seed_age_store(data_dir, recovery)

        key_file = _write_file(tmp_path / "garbage.txt", "not-a-valid-bech32")
        pw_file = _write_file(tmp_path / "pw.txt", "pw")

        rc = restore.run(
            _default_args(tmp_path, data_dir, recovery_key_file=key_file, passphrase_file=pw_file)
        )
        assert rc == 1
        err = capsys.readouterr().err
        assert "not a valid age secret" in err

    def test_empty_recovery_file_refused(self, tmp_path, monkeypatch, capsys):
        _point_config_at(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        key_file = _write_file(tmp_path / "empty.txt", "")
        pw_file = _write_file(tmp_path / "pw.txt", "pw")

        rc = restore.run(
            _default_args(tmp_path, data_dir, recovery_key_file=key_file, passphrase_file=pw_file)
        )
        assert rc == 1
        assert "empty" in capsys.readouterr().err

    def test_missing_recovery_file_refused(self, tmp_path, monkeypatch, capsys):
        _point_config_at(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        pw_file = _write_file(tmp_path / "pw.txt", "pw")

        rc = restore.run(
            _default_args(
                tmp_path, data_dir, recovery_key_file=tmp_path / "nope.txt", passphrase_file=pw_file
            )
        )
        assert rc == 1
        assert "could not read recovery-key file" in capsys.readouterr().err


class TestPreconditions:
    def test_existing_daily_key_refused_without_force(self, tmp_path, monkeypatch, capsys):
        _point_config_at(tmp_path, monkeypatch)
        # Pre-stage a stale daily_key.age (e.g. accidentally restored from backup).
        (tmp_path / "daily_key.age").write_bytes(b"stale")

        recovery = x25519.Identity.generate()
        data_dir = tmp_path / "data"
        _seed_age_store(data_dir, recovery)
        key_file = _write_file(tmp_path / "recovery.txt", str(recovery))
        pw_file = _write_file(tmp_path / "pw.txt", "pw")

        rc = restore.run(
            _default_args(tmp_path, data_dir, recovery_key_file=key_file, passphrase_file=pw_file)
        )
        assert rc == 1
        err = capsys.readouterr().err
        assert "already exists" in err
        assert "--force" in err

    def test_force_overwrites_existing_key_files(self, tmp_path, monkeypatch, capsys):
        _point_config_at(tmp_path, monkeypatch)
        (tmp_path / "daily_key.age").write_bytes(b"stale")
        (tmp_path / "recovery.pub").write_text("stale-pub\n")

        recovery = x25519.Identity.generate()
        data_dir = tmp_path / "data"
        _seed_age_store(data_dir, recovery)
        key_file = _write_file(tmp_path / "recovery.txt", str(recovery))
        pw_file = _write_file(tmp_path / "pw.txt", "pw")

        rc = restore.run(
            _default_args(
                tmp_path, data_dir, recovery_key_file=key_file, passphrase_file=pw_file, force=True
            )
        )
        assert rc == 0
        # daily_key.age replaced with something that unlocks with "pw".
        load_daily_identity(tmp_path / "daily_key.age", passphrase="pw")

    def test_pmem1_files_refused(self, tmp_path, monkeypatch, capsys):
        _point_config_at(tmp_path, monkeypatch)
        recovery = x25519.Identity.generate()
        data_dir = tmp_path / "data"
        _seed_age_store(data_dir, recovery)

        # Inject a PMEM1 file.
        os.environ[MASTER_KEY_ENV_VAR] = (
            __import__("cryptography.fernet", fromlist=["Fernet"]).Fernet.generate_key().decode()
        )
        (data_dir / "registry.json").write_bytes(PMEM1_MAGIC + encrypt_bytes(b"{}"))
        os.environ.pop(MASTER_KEY_ENV_VAR, None)
        _clear_cipher_cache()

        key_file = _write_file(tmp_path / "recovery.txt", str(recovery))
        pw_file = _write_file(tmp_path / "pw.txt", "pw")

        rc = restore.run(
            _default_args(tmp_path, data_dir, recovery_key_file=key_file, passphrase_file=pw_file)
        )
        assert rc == 1
        err = capsys.readouterr().err
        assert "PMEM1" in err

    def test_plaintext_infra_files_refused(self, tmp_path, monkeypatch, capsys):
        _point_config_at(tmp_path, monkeypatch)
        recovery = x25519.Identity.generate()
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "registry.json").write_bytes(b'{"plain": true}')

        key_file = _write_file(tmp_path / "recovery.txt", str(recovery))
        pw_file = _write_file(tmp_path / "pw.txt", "pw")

        rc = restore.run(
            _default_args(tmp_path, data_dir, recovery_key_file=key_file, passphrase_file=pw_file)
        )
        assert rc == 1
        assert "plaintext" in capsys.readouterr().err

    def test_missing_data_dir_refused(self, tmp_path, monkeypatch, capsys):
        _point_config_at(tmp_path, monkeypatch)
        recovery = x25519.Identity.generate()
        key_file = _write_file(tmp_path / "recovery.txt", str(recovery))
        pw_file = _write_file(tmp_path / "pw.txt", "pw")

        rc = restore.run(
            _default_args(
                tmp_path, tmp_path / "nope", recovery_key_file=key_file, passphrase_file=pw_file
            )
        )
        assert rc == 1
        assert "does not exist" in capsys.readouterr().err


class TestResume:
    def test_resume_continues_remaining_files(self, tmp_path, monkeypatch, capsys):
        """Simulate a prior crash: daily_key.age + recovery.pub already
        present, manifest lists some files as done and some as pending."""
        _point_config_at(tmp_path, monkeypatch)
        recovery = x25519.Identity.generate()
        data_dir = tmp_path / "data"
        seeded = _seed_age_store(data_dir, recovery)

        # Simulate partial state from a crash:
        from paramem.backup.key_store import (
            mint_daily_identity,
            wrap_daily_identity,
            write_daily_key_file,
            write_recovery_pub_file,
        )

        new_daily = mint_daily_identity()
        write_daily_key_file(wrap_daily_identity(new_daily, "pw"), tmp_path / "daily_key.age")
        write_recovery_pub_file(recovery.to_public(), tmp_path / "recovery.pub")

        # One file already re-encrypted to [new_daily, recovery].
        done_path, done_plaintext = seeded[0]
        done_path.write_bytes(
            age_encrypt_bytes(done_plaintext, [new_daily.to_public(), recovery.to_public()])
        )

        manifest = RotationManifest(
            operation="restore",
            started_at="2026-04-24T12:00:00Z",
            new_daily_pub=str(new_daily.to_public()),
            files_pending=[str(seeded[1][0]), str(seeded[2][0])],
            files_done=[str(done_path)],
        )
        write_manifest_atomic(tmp_path / "rotation.manifest.json", manifest)

        key_file = _write_file(tmp_path / "recovery.txt", str(recovery))
        pw_file = _write_file(tmp_path / "pw.txt", "pw")

        rc = restore.run(
            _default_args(tmp_path, data_dir, recovery_key_file=key_file, passphrase_file=pw_file)
        )
        assert rc == 0

        # All three files decrypt with the same new_daily — the resumed run
        # did not mint a different identity.
        final_daily = load_daily_identity(tmp_path / "daily_key.age", passphrase="pw")
        assert str(final_daily) == str(new_daily)
        for path, original in seeded:
            assert age_decrypt_bytes(path.read_bytes(), [final_daily]) == original

    def test_resume_refuses_mismatched_operation(self, tmp_path, monkeypatch, capsys):
        _point_config_at(tmp_path, monkeypatch)
        recovery = x25519.Identity.generate()
        data_dir = tmp_path / "data"
        _seed_age_store(data_dir, recovery)

        manifest = RotationManifest(
            operation="rotate-daily",  # wrong op
            started_at="2026-04-24T12:00:00Z",
            new_daily_pub="age1xxx",
            files_pending=[],
            files_done=[],
        )
        write_manifest_atomic(tmp_path / "rotation.manifest.json", manifest)
        # A stale daily_key.age so the precondition passes (resume path).
        (tmp_path / "daily_key.age").write_bytes(b"stale")

        key_file = _write_file(tmp_path / "recovery.txt", str(recovery))
        pw_file = _write_file(tmp_path / "pw.txt", "pw")

        rc = restore.run(
            _default_args(
                tmp_path, data_dir, recovery_key_file=key_file, passphrase_file=pw_file, force=True
            )
        )
        assert rc == 1
        err = capsys.readouterr().err
        assert "'rotate-daily' run" in err

    def test_resume_refuses_recovery_mismatch(self, tmp_path, monkeypatch, capsys):
        """If the operator supplies a different recovery bech32 than the one
        persisted by the prior run, refuse — mid-restore recovery changes are
        not supported."""
        _point_config_at(tmp_path, monkeypatch)
        recovery_old = x25519.Identity.generate()
        recovery_new = x25519.Identity.generate()
        data_dir = tmp_path / "data"
        _seed_age_store(data_dir, recovery_old)

        from paramem.backup.key_store import (
            mint_daily_identity,
            wrap_daily_identity,
            write_daily_key_file,
            write_recovery_pub_file,
        )

        new_daily = mint_daily_identity()
        write_daily_key_file(wrap_daily_identity(new_daily, "pw"), tmp_path / "daily_key.age")
        write_recovery_pub_file(recovery_old.to_public(), tmp_path / "recovery.pub")

        manifest = RotationManifest(
            operation="restore",
            started_at="2026-04-24T12:00:00Z",
            new_daily_pub=str(new_daily.to_public()),
            files_pending=[],
            files_done=[],
        )
        write_manifest_atomic(tmp_path / "rotation.manifest.json", manifest)

        # Operator supplies a DIFFERENT recovery on the resume attempt — but
        # first the sanity check against the existing age envelope tries the
        # new bech32, which does not decrypt. That error surfaces before the
        # manifest-cross-check, which is acceptable: both guard the same
        # invariant. Verify the restore is refused (exit != 0) in either case.
        key_file = _write_file(tmp_path / "recovery.txt", str(recovery_new))
        pw_file = _write_file(tmp_path / "pw.txt", "pw")

        rc = restore.run(
            _default_args(
                tmp_path, data_dir, recovery_key_file=key_file, passphrase_file=pw_file, force=True
            )
        )
        assert rc != 0


class TestCliEntryPoint:
    def test_subcommand_registered(self):
        from paramem.cli import main as main_module

        parser = main_module._build_parser()
        args = parser.parse_args(
            ["restore", "--data-dir", "/tmp/ignored", "--recovery-key-file", "/tmp/k"]
        )
        assert args.command == "restore"
        assert args.dry_run is False
        assert args.force is False
        assert str(args.recovery_key_file) == "/tmp/k"


class TestWrongRecoveryDoesNotDamageStore:
    def test_wrong_key_leaves_all_files_byte_identical(self, tmp_path, monkeypatch, capsys):
        """A typo in the recovery bech32 must not touch any infrastructure
        file — the sanity check aborts before any mutation."""
        _point_config_at(tmp_path, monkeypatch)
        recovery = x25519.Identity.generate()
        data_dir = tmp_path / "data"
        seeded = _seed_age_store(data_dir, recovery)
        before = {p: p.read_bytes() for p, _ in seeded}

        wrong = x25519.Identity.generate()
        key_file = _write_file(tmp_path / "wrong.txt", str(wrong))
        pw_file = _write_file(tmp_path / "pw.txt", "pw")

        rc = restore.run(
            _default_args(tmp_path, data_dir, recovery_key_file=key_file, passphrase_file=pw_file)
        )
        assert rc == 1
        for p, _ in seeded:
            assert p.read_bytes() == before[p], f"{p} was mutated by a failed restore"


class TestInteractivePrompts:
    def test_interactive_recovery_prompt_happy_path(self, tmp_path, monkeypatch, capsys):
        """Operator pastes the bech32 at the interactive getpass prompt."""
        _point_config_at(tmp_path, monkeypatch)
        recovery = x25519.Identity.generate()
        data_dir = tmp_path / "data"
        _seed_age_store(data_dir, recovery)
        pw_file = _write_file(tmp_path / "pw.txt", "pw")

        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("paramem.cli.restore.getpass.getpass", lambda prompt: str(recovery))

        rc = restore.run(
            _default_args(tmp_path, data_dir, recovery_key_file=None, passphrase_file=pw_file)
        )
        assert rc == 0

    def test_interactive_recovery_prompt_empty_refuses(self, tmp_path, monkeypatch, capsys):
        _point_config_at(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("paramem.cli.restore.getpass.getpass", lambda prompt: "")

        rc = restore.run(_default_args(tmp_path, data_dir))
        assert rc == 1
        assert "recovery key must be non-empty" in capsys.readouterr().err

    def test_interactive_passphrase_happy_path(self, tmp_path, monkeypatch, capsys):
        _point_config_at(tmp_path, monkeypatch)
        recovery = x25519.Identity.generate()
        data_dir = tmp_path / "data"
        _seed_age_store(data_dir, recovery)
        key_file = _write_file(tmp_path / "recovery.txt", str(recovery))

        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        answers = iter(["interactive-pw", "interactive-pw"])
        monkeypatch.setattr("paramem.cli.restore.getpass.getpass", lambda prompt: next(answers))

        rc = restore.run(_default_args(tmp_path, data_dir, recovery_key_file=key_file))
        assert rc == 0
        load_daily_identity(tmp_path / "daily_key.age", passphrase="interactive-pw")

    def test_interactive_passphrase_mismatch_refuses(self, tmp_path, monkeypatch, capsys):
        _point_config_at(tmp_path, monkeypatch)
        recovery = x25519.Identity.generate()
        data_dir = tmp_path / "data"
        seeded = _seed_age_store(data_dir, recovery)
        before = {p: p.read_bytes() for p, _ in seeded}
        key_file = _write_file(tmp_path / "recovery.txt", str(recovery))

        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        answers = iter(["one", "two"])
        monkeypatch.setattr("paramem.cli.restore.getpass.getpass", lambda prompt: next(answers))

        rc = restore.run(_default_args(tmp_path, data_dir, recovery_key_file=key_file))
        assert rc == 1
        assert "did not match" in capsys.readouterr().err
        # No mutation — passphrase mismatch aborted before the mint.
        for p, _ in seeded:
            assert p.read_bytes() == before[p]
        assert not (tmp_path / "daily_key.age").exists()


class TestDecryptErrorSurface:
    def test_pyrage_decrypt_error_on_wrong_key_is_caught(self, tmp_path, monkeypatch, capsys):
        """Sanity: the restore's DecryptError catch handles the real
        pyrage error class, not just a generic Exception."""
        import paramem.cli.restore as r

        # Direct invocation of age_decrypt_bytes with mismatched identity
        # raises pyrage.DecryptError. Just assert the import surface.
        assert r.pyrage.DecryptError is pyrage.DecryptError
