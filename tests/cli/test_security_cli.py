"""Tests for the security-track CLI commands.

Covers ``paramem generate-key``, ``paramem encrypt-infra``,
``paramem decrypt-infra``, and ``paramem dump``.  All tests are
pure-Python / filesystem-only — no REST, no GPU.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from paramem.backup.encryption import (
    MASTER_KEY_ENV_VAR,
    PMEM1_MAGIC,
    _clear_cipher_cache,
    is_pmem1_envelope,
    read_maybe_encrypted,
    write_infra_bytes,
)
from paramem.backup.key_store import (
    DAILY_PASSPHRASE_ENV_VAR,
    load_daily_identity,
    load_recovery_recipient,
)
from paramem.cli import decrypt_infra, dump, encrypt_infra, generate_key


def _make_key() -> str:
    return Fernet.generate_key().decode()


def _clean_env() -> None:
    os.environ.pop(MASTER_KEY_ENV_VAR, None)
    _clear_cipher_cache()


@pytest.fixture(autouse=True)
def _env_isolation():
    _clean_env()
    yield
    _clean_env()


# ---------------------------------------------------------------------------
# generate-key
# ---------------------------------------------------------------------------


def _generate_key_args(
    tmp_path: Path,
    *,
    passphrase_file: Path | None = None,
    force: bool = False,
    yes: bool = True,
) -> argparse.Namespace:
    """Namespace matching what argparse would hand to generate_key.run()."""
    return argparse.Namespace(
        daily_key_path=tmp_path / "config" / "paramem" / "daily_key.age",
        recovery_pub_path=tmp_path / "config" / "paramem" / "recovery.pub",
        passphrase_file=passphrase_file,
        force=force,
        yes=yes,
    )


def _write_passphrase_file(tmp_path: Path, passphrase: str) -> Path:
    path = tmp_path / "passphrase.txt"
    path.write_text(passphrase + "\n", encoding="utf-8")
    return path


class TestGenerateKey:
    def test_happy_path_writes_both_files_with_correct_modes(self, tmp_path, capsys):
        import stat

        pw_file = _write_passphrase_file(tmp_path, "live-test-pw")
        args = _generate_key_args(tmp_path, passphrase_file=pw_file)

        rc = generate_key.run(args)
        assert rc == 0

        assert args.daily_key_path.exists()
        assert args.recovery_pub_path.exists()
        assert stat.S_IMODE(args.daily_key_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(args.recovery_pub_path.stat().st_mode) == 0o644

    def test_daily_key_unlocks_with_supplied_passphrase(self, tmp_path, capsys):
        pw_file = _write_passphrase_file(tmp_path, "unlock-me")
        args = _generate_key_args(tmp_path, passphrase_file=pw_file)
        generate_key.run(args)

        ident = load_daily_identity(args.daily_key_path, passphrase="unlock-me")
        assert str(ident).startswith("AGE-SECRET-KEY-1")

    def test_recovery_pub_is_valid_x25519_recipient(self, tmp_path, capsys):
        pw_file = _write_passphrase_file(tmp_path, "x")
        args = _generate_key_args(tmp_path, passphrase_file=pw_file)
        generate_key.run(args)
        recipient = load_recovery_recipient(args.recovery_pub_path)
        assert str(recipient).startswith("age1")

    def test_recovery_secret_printed_to_stderr(self, tmp_path, capsys):
        pw_file = _write_passphrase_file(tmp_path, "x")
        args = _generate_key_args(tmp_path, passphrase_file=pw_file)
        rc = generate_key.run(args)
        assert rc == 0
        err = capsys.readouterr().err
        assert "AGE-SECRET-KEY-1" in err, "recovery bech32 must be printed to stderr"
        assert "WRITE THIS DOWN NOW" in err
        assert "NEVER stored on" in err

    def test_refuses_existing_files_without_force(self, tmp_path, capsys):
        pw_file = _write_passphrase_file(tmp_path, "x")
        args = _generate_key_args(tmp_path, passphrase_file=pw_file)
        # First invocation creates the files.
        assert generate_key.run(args) == 0
        capsys.readouterr()  # drain

        # Second invocation without --force must refuse.
        rc = generate_key.run(args)
        assert rc == 1
        err = capsys.readouterr().err
        assert "already exist" in err
        assert "--force" in err

    def test_force_overwrites_existing_files(self, tmp_path, capsys):
        pw_file = _write_passphrase_file(tmp_path, "x")
        args = _generate_key_args(tmp_path, passphrase_file=pw_file)
        generate_key.run(args)
        first_daily = args.daily_key_path.read_bytes()
        capsys.readouterr()

        args_force = _generate_key_args(tmp_path, passphrase_file=pw_file, force=True)
        rc = generate_key.run(args_force)
        assert rc == 0
        assert args_force.daily_key_path.read_bytes() != first_daily

    def test_passphrase_from_env_var(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "from-env")
        args = _generate_key_args(tmp_path)  # no passphrase_file
        rc = generate_key.run(args)
        assert rc == 0
        ident = load_daily_identity(args.daily_key_path, passphrase="from-env")
        assert str(ident).startswith("AGE-SECRET-KEY-1")

    def test_refuses_no_passphrase_no_tty(self, tmp_path, capsys, monkeypatch):
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        args = _generate_key_args(tmp_path)
        rc = generate_key.run(args)
        assert rc == 1
        err = capsys.readouterr().err
        assert "no passphrase supplied" in err
        assert DAILY_PASSPHRASE_ENV_VAR in err
        assert not args.daily_key_path.exists()

    def test_refuses_missing_passphrase_file(self, tmp_path, capsys):
        args = _generate_key_args(tmp_path, passphrase_file=tmp_path / "nope.txt")
        rc = generate_key.run(args)
        assert rc == 1
        err = capsys.readouterr().err
        assert "could not read passphrase file" in err

    def test_refuses_empty_passphrase_file(self, tmp_path, capsys):
        pw_file = tmp_path / "empty.txt"
        pw_file.write_text("", encoding="utf-8")
        args = _generate_key_args(tmp_path, passphrase_file=pw_file)
        rc = generate_key.run(args)
        assert rc == 1
        assert "passphrase file is empty" in capsys.readouterr().err

    def test_refuses_non_interactive_without_yes(self, tmp_path, capsys, monkeypatch):
        pw_file = _write_passphrase_file(tmp_path, "x")
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        args = _generate_key_args(tmp_path, passphrase_file=pw_file, yes=False)
        rc = generate_key.run(args)
        assert rc == 1
        err = capsys.readouterr().err
        assert "non-interactively without --yes" in err
        assert not args.daily_key_path.exists(), "nothing must be written on aborted confirm"

    def test_two_runs_produce_distinct_identities(self, tmp_path, capsys):
        pw_file = _write_passphrase_file(tmp_path, "x")
        # First invocation.
        args_a = _generate_key_args(tmp_path, passphrase_file=pw_file)
        generate_key.run(args_a)
        daily_a = load_daily_identity(args_a.daily_key_path, passphrase="x")
        recovery_a = load_recovery_recipient(args_a.recovery_pub_path)

        # Second invocation into a different directory.
        tmp_b = tmp_path / "second"
        tmp_b.mkdir()
        args_b = _generate_key_args(tmp_b, passphrase_file=pw_file)
        generate_key.run(args_b)
        daily_b = load_daily_identity(args_b.daily_key_path, passphrase="x")
        recovery_b = load_recovery_recipient(args_b.recovery_pub_path)

        assert str(daily_a) != str(daily_b)
        assert str(recovery_a) != str(recovery_b)

    def test_interactive_passphrase_happy_path(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        pw_answers = iter(["interactive-pw", "interactive-pw"])
        monkeypatch.setattr(
            "paramem.cli.generate_key.getpass.getpass",
            lambda prompt: next(pw_answers),
        )
        args = _generate_key_args(tmp_path)  # no passphrase_file, no env → interactive
        rc = generate_key.run(args)
        assert rc == 0
        ident = load_daily_identity(args.daily_key_path, passphrase="interactive-pw")
        assert str(ident).startswith("AGE-SECRET-KEY-1")

    def test_interactive_passphrase_mismatch_refuses(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        pw_answers = iter(["one", "two"])
        monkeypatch.setattr(
            "paramem.cli.generate_key.getpass.getpass",
            lambda prompt: next(pw_answers),
        )
        args = _generate_key_args(tmp_path)
        rc = generate_key.run(args)
        assert rc == 1
        assert "did not match" in capsys.readouterr().err
        assert not args.daily_key_path.exists()

    def test_interactive_passphrase_empty_refuses(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("paramem.cli.generate_key.getpass.getpass", lambda prompt: "")
        args = _generate_key_args(tmp_path)
        rc = generate_key.run(args)
        assert rc == 1
        assert "non-empty" in capsys.readouterr().err
        assert not args.daily_key_path.exists()

    def test_confirmation_phrase_happy_path(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        pw_file = _write_passphrase_file(tmp_path, "x")
        monkeypatch.setattr("builtins.input", lambda: "I have saved the recovery key")
        args = _generate_key_args(tmp_path, passphrase_file=pw_file, yes=False)
        rc = generate_key.run(args)
        assert rc == 0
        assert args.daily_key_path.exists()
        assert args.recovery_pub_path.exists()

    def test_confirmation_phrase_mismatch_refuses(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        pw_file = _write_passphrase_file(tmp_path, "x")
        monkeypatch.setattr("builtins.input", lambda: "sure whatever")
        args = _generate_key_args(tmp_path, passphrase_file=pw_file, yes=False)
        rc = generate_key.run(args)
        assert rc == 1
        assert "confirmation phrase did not match" in capsys.readouterr().err
        assert not args.daily_key_path.exists()
        assert not args.recovery_pub_path.exists()

    def test_confirmation_phrase_interrupted_refuses_cleanly(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        pw_file = _write_passphrase_file(tmp_path, "x")

        def _raise_keyboard_interrupt() -> str:
            raise KeyboardInterrupt()

        monkeypatch.setattr("builtins.input", _raise_keyboard_interrupt)
        args = _generate_key_args(tmp_path, passphrase_file=pw_file, yes=False)
        rc = generate_key.run(args)
        assert rc == 1
        assert "Aborted" in capsys.readouterr().err
        assert not args.daily_key_path.exists()
        assert not args.recovery_pub_path.exists()

    def test_recovery_secret_is_never_written_to_disk(self, tmp_path, capsys):
        """Regression guard: the printed recovery secret must not appear in
        either daily_key.age (wraps the *daily* secret, not the recovery one)
        or recovery.pub (holds only the public recipient)."""
        import re

        pw_file = _write_passphrase_file(tmp_path, "x")
        args = _generate_key_args(tmp_path, passphrase_file=pw_file)
        rc = generate_key.run(args)
        assert rc == 0

        match = re.search(r"AGE-SECRET-KEY-1[A-Z0-9]+", capsys.readouterr().err)
        assert match, "recovery secret bech32 must appear in stderr"
        recovery_secret = match.group(0)

        daily_bytes = args.daily_key_path.read_bytes()
        pub_text = args.recovery_pub_path.read_text("utf-8")

        assert recovery_secret.encode() not in daily_bytes, (
            "recovery secret must not appear in daily_key.age"
        )
        assert "AGE-SECRET-KEY" not in pub_text, "recovery.pub must hold only the public recipient"
        assert recovery_secret not in pub_text


# ---------------------------------------------------------------------------
# encrypt-infra
# ---------------------------------------------------------------------------


class TestEncryptInfra:
    def _seed_plaintext_files(self, data_dir: Path) -> list[Path]:
        """Write a couple of plaintext infra files under *data_dir* that
        match paths in ``infra_paths``."""
        (data_dir / "registry").mkdir(parents=True, exist_ok=True)
        (data_dir / "adapters").mkdir(parents=True, exist_ok=True)
        seeded = []
        reg = data_dir / "registry.json"
        reg.write_bytes(json.dumps({"graph1": "abc"}).encode("utf-8"))
        seeded.append(reg)
        speakers = data_dir / "speaker_profiles.json"
        speakers.write_bytes(json.dumps({"speakers": {}, "version": 5}).encode("utf-8"))
        seeded.append(speakers)
        queue = data_dir / "adapters" / "post_session_queue.json"
        queue.write_bytes(json.dumps([]).encode("utf-8"))
        seeded.append(queue)
        return seeded

    def test_refuses_without_key(self, tmp_path, capsys):
        data_dir = tmp_path / "data"
        self._seed_plaintext_files(data_dir)
        rc = encrypt_infra.run(
            argparse.Namespace(data_dir=str(data_dir), config="configs/server.yaml")
        )
        assert rc == 1
        assert "PARAMEM_MASTER_KEY must be set" in capsys.readouterr().err

    def test_encrypts_plaintext_files(self, tmp_path, capsys):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        data_dir = tmp_path / "data"
        seeded = self._seed_plaintext_files(data_dir)

        rc = encrypt_infra.run(
            argparse.Namespace(data_dir=str(data_dir), config="configs/server.yaml")
        )
        assert rc == 0

        for path in seeded:
            assert is_pmem1_envelope(path)
            assert path.read_bytes().startswith(PMEM1_MAGIC)

        out = capsys.readouterr().out
        assert "3 converted" in out

    def test_idempotent_on_already_encrypted(self, tmp_path, capsys):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        data_dir = tmp_path / "data"
        self._seed_plaintext_files(data_dir)
        encrypt_infra.run(argparse.Namespace(data_dir=str(data_dir), config=""))
        capsys.readouterr()  # drain first run

        rc = encrypt_infra.run(argparse.Namespace(data_dir=str(data_dir), config=""))
        assert rc == 0
        out = capsys.readouterr().out
        assert "0 converted" in out
        assert "already encrypted" in out

    def test_missing_data_dir_returns_error(self, tmp_path, capsys):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        missing = tmp_path / "does-not-exist"
        rc = encrypt_infra.run(
            argparse.Namespace(data_dir=str(missing), config="configs/server.yaml")
        )
        assert rc == 1
        assert "does not exist" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# decrypt-infra
# ---------------------------------------------------------------------------


class TestDecryptInfra:
    def _seed_ciphertext_files(self, data_dir: Path) -> list[Path]:
        """Seed with files written through write_infra_bytes while the key
        is set, then return them with the key preserved so decrypt-infra can
        unwrap them."""
        (data_dir / "registry").mkdir(parents=True, exist_ok=True)
        (data_dir / "adapters").mkdir(parents=True, exist_ok=True)
        seeded = []
        for rel, content in [
            (Path("registry.json"), b'{"graph1":"abc"}'),
            (Path("speaker_profiles.json"), b'{"speakers":{},"version":5}'),
            (Path("adapters/post_session_queue.json"), b"[]"),
        ]:
            p = data_dir / rel
            write_infra_bytes(p, content)
            seeded.append(p)
        return seeded

    def test_refuses_without_flag(self, tmp_path, capsys):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        data_dir = tmp_path / "data"
        self._seed_ciphertext_files(data_dir)
        rc = decrypt_infra.run(
            argparse.Namespace(i_accept_plaintext=False, data_dir=str(data_dir), config="")
        )
        assert rc == 1
        assert "--i-accept-plaintext is required" in capsys.readouterr().err

    def test_refuses_without_key(self, tmp_path, capsys):
        # Seed ciphertext with a key, then drop the key to simulate the
        # unrecoverable case.
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        data_dir = tmp_path / "data"
        self._seed_ciphertext_files(data_dir)
        _clean_env()

        rc = decrypt_infra.run(
            argparse.Namespace(i_accept_plaintext=True, data_dir=str(data_dir), config="")
        )
        assert rc == 1
        assert "must be set to decrypt" in capsys.readouterr().err

    def test_decrypts_ciphertext_files(self, tmp_path, capsys):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        data_dir = tmp_path / "data"
        seeded = self._seed_ciphertext_files(data_dir)

        rc = decrypt_infra.run(
            argparse.Namespace(i_accept_plaintext=True, data_dir=str(data_dir), config="")
        )
        assert rc == 0

        for path in seeded:
            assert not is_pmem1_envelope(path)
        # Reminder about removing the env var appears on stdout.
        assert "remove PARAMEM_MASTER_KEY" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# dump
# ---------------------------------------------------------------------------


class TestDump:
    def test_plaintext_passthrough(self, tmp_path, capsysbinary):
        p = tmp_path / "plain.json"
        p.write_bytes(b'{"ok":true}')
        rc = dump.run(argparse.Namespace(path=str(p)))
        assert rc == 0
        assert capsysbinary.readouterr().out == b'{"ok":true}'

    def test_encrypted_decrypted_to_stdout(self, tmp_path, capsysbinary):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        p = tmp_path / "encrypted.json"
        write_infra_bytes(p, b'{"secret":"value"}')

        rc = dump.run(argparse.Namespace(path=str(p)))
        assert rc == 0
        assert capsysbinary.readouterr().out == b'{"secret":"value"}'

    def test_missing_path_returns_error(self, tmp_path, capsys):
        rc = dump.run(argparse.Namespace(path=str(tmp_path / "nope.json")))
        assert rc == 1
        assert "does not exist" in capsys.readouterr().err

    def test_encrypted_without_key_returns_error(self, tmp_path, capsys):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        p = tmp_path / "encrypted.json"
        write_infra_bytes(p, b"payload")
        _clean_env()

        rc = dump.run(argparse.Namespace(path=str(p)))
        assert rc == 1
        assert MASTER_KEY_ENV_VAR in capsys.readouterr().err


# ---------------------------------------------------------------------------
# End-to-end: encrypt → decrypt roundtrip
# ---------------------------------------------------------------------------


class TestEncryptDecryptRoundtrip:
    def test_encrypt_then_decrypt_returns_original(self, tmp_path):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)

        original = json.dumps({"graph1": "abc", "graph2": "def"}, indent=2).encode("utf-8")
        reg = data_dir / "registry.json"
        reg.write_bytes(original)

        encrypt_infra.run(argparse.Namespace(data_dir=str(data_dir), config=""))
        assert is_pmem1_envelope(reg)
        # Content still recoverable via the envelope helper.
        assert read_maybe_encrypted(reg) == original

        decrypt_infra.run(
            argparse.Namespace(i_accept_plaintext=True, data_dir=str(data_dir), config="")
        )
        assert not is_pmem1_envelope(reg)
        assert reg.read_bytes() == original
