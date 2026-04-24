"""Tests for the security-track CLI commands.

Covers ``paramem generate-key`` and ``paramem dump``.  All tests are
pure-Python / filesystem-only — no REST, no GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from paramem.backup.encryption import (
    write_infra_bytes,
)
from paramem.backup.key_store import (
    DAILY_PASSPHRASE_ENV_VAR,
    _clear_daily_identity_cache,
    load_daily_identity,
    load_recovery_recipient,
    mint_daily_identity,
    wrap_daily_identity,
    write_daily_key_file,
)
from paramem.cli import dump, generate_key


@pytest.fixture(autouse=True)
def _env_isolation(monkeypatch):
    monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
    _clear_daily_identity_cache()
    yield
    _clear_daily_identity_cache()


# ---------------------------------------------------------------------------
# Helpers shared by multiple suites
# ---------------------------------------------------------------------------


def _setup_daily(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passphrase: str = "pw"):
    """Mint + wrap + write a daily identity; wire env + module default."""
    ident = mint_daily_identity()
    key_path = tmp_path / "daily_key.age"
    write_daily_key_file(wrap_daily_identity(ident, passphrase), key_path)
    monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, passphrase)
    monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path)
    _clear_daily_identity_cache()
    return ident


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
# dump
# ---------------------------------------------------------------------------


class TestDump:
    def test_plaintext_passthrough(self, tmp_path, capsysbinary):
        """Plaintext file passes through dump unchanged."""
        p = tmp_path / "plain.json"
        p.write_bytes(b'{"ok":true}')
        rc = dump.run(argparse.Namespace(path=str(p)))
        assert rc == 0
        assert capsysbinary.readouterr().out == b'{"ok":true}'

    def test_age_encrypted_decrypted_to_stdout(self, tmp_path, capsysbinary, monkeypatch):
        """Age-encrypted file decrypts to stdout when daily identity is loaded."""
        _setup_daily(tmp_path, monkeypatch)
        p = tmp_path / "encrypted.json"
        write_infra_bytes(p, b'{"secret":"value"}')

        rc = dump.run(argparse.Namespace(path=str(p)))
        assert rc == 0
        assert capsysbinary.readouterr().out == b'{"secret":"value"}'

    def test_missing_path_returns_error(self, tmp_path, capsys):
        """Missing file → rc=1, error message mentions 'does not exist'."""
        rc = dump.run(argparse.Namespace(path=str(tmp_path / "nope.json")))
        assert rc == 1
        assert "does not exist" in capsys.readouterr().err

    def test_age_encrypted_without_passphrase_returns_error(self, tmp_path, capsys, monkeypatch):
        """Age-encrypted file + passphrase not set → rc=1, env var named in error."""
        # Write with daily loaded.
        _setup_daily(tmp_path, monkeypatch)
        p = tmp_path / "encrypted.json"
        write_infra_bytes(p, b"payload")

        # Drop the passphrase — dump must fail with a clear message.
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
        _clear_daily_identity_cache()

        rc = dump.run(argparse.Namespace(path=str(p)))
        assert rc == 1
        assert DAILY_PASSPHRASE_ENV_VAR in capsys.readouterr().err
