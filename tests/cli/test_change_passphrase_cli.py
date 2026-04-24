"""Tests for ``paramem change-passphrase``.

Covers happy paths (file-to-file, env-to-file, fully interactive),
precondition refusals (missing daily_key.age, wrong old, empty secrets,
old==new, missing TTY), interactive prompt paths, and the identity-
preservation invariant (same X25519 identity after rewrap).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pyrage
import pytest

from paramem.backup.key_store import (
    DAILY_PASSPHRASE_ENV_VAR,
    _clear_daily_identity_cache,
    load_daily_identity,
    mint_daily_identity,
    wrap_daily_identity,
    write_daily_key_file,
)
from paramem.cli import change_passphrase


@pytest.fixture(autouse=True)
def _isolate_env_and_caches():
    os.environ.pop(DAILY_PASSPHRASE_ENV_VAR, None)
    _clear_daily_identity_cache()
    yield
    os.environ.pop(DAILY_PASSPHRASE_ENV_VAR, None)
    _clear_daily_identity_cache()


def _seed_daily(tmp_path: Path, passphrase: str = "OLD"):
    """Mint + wrap + write a daily identity. Returns (identity, key_path)."""
    ident = mint_daily_identity()
    key_path = tmp_path / "daily_key.age"
    write_daily_key_file(wrap_daily_identity(ident, passphrase), key_path)
    return ident, key_path


def _write_file(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _args(
    tmp_path: Path,
    *,
    daily_key_path: Path | None = None,
    old_passphrase_file: Path | None = None,
    new_passphrase_file: Path | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        daily_key_path=daily_key_path or tmp_path / "daily_key.age",
        old_passphrase_file=old_passphrase_file,
        new_passphrase_file=new_passphrase_file,
    )


class TestHappyPath:
    def test_file_to_file_rewrap(self, tmp_path):
        ident, key_path = _seed_daily(tmp_path, "OLD")
        old_file = _write_file(tmp_path / "old.txt", "OLD\n")
        new_file = _write_file(tmp_path / "new.txt", "NEW\n")

        rc = change_passphrase.run(
            _args(tmp_path, old_passphrase_file=old_file, new_passphrase_file=new_file)
        )
        assert rc == 0

        # Identity unchanged — same X25519 keypair.
        new_ident = load_daily_identity(key_path, passphrase="NEW")
        assert str(new_ident) == str(ident), "identity must survive rewrap"

        # Old passphrase no longer works.
        with pytest.raises(pyrage.DecryptError):
            load_daily_identity(key_path, passphrase="OLD")

    def test_env_old_file_new(self, tmp_path, monkeypatch):
        ident, key_path = _seed_daily(tmp_path, "env-pw")
        monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "env-pw")
        new_file = _write_file(tmp_path / "new.txt", "new-pw\n")

        rc = change_passphrase.run(_args(tmp_path, new_passphrase_file=new_file))
        assert rc == 0

        new_ident = load_daily_identity(key_path, passphrase="new-pw")
        assert str(new_ident) == str(ident)

    def test_success_prints_operator_reminder(self, tmp_path, capsys):
        _seed_daily(tmp_path, "OLD")
        old_file = _write_file(tmp_path / "old.txt", "OLD\n")
        new_file = _write_file(tmp_path / "new.txt", "NEW\n")
        change_passphrase.run(
            _args(tmp_path, old_passphrase_file=old_file, new_passphrase_file=new_file)
        )

        out = capsys.readouterr().out
        assert "Passphrase changed" in out
        assert DAILY_PASSPHRASE_ENV_VAR in out
        assert "restart" in out.lower()


class TestRefusals:
    def test_missing_daily_key_file(self, tmp_path, capsys):
        # No daily_key.age seeded.
        old_file = _write_file(tmp_path / "old.txt", "x\n")
        new_file = _write_file(tmp_path / "new.txt", "y\n")
        rc = change_passphrase.run(
            _args(tmp_path, old_passphrase_file=old_file, new_passphrase_file=new_file)
        )
        assert rc == 1
        err = capsys.readouterr().err
        assert "daily key file not found" in err
        assert "paramem generate-key" in err

    def test_wrong_old_passphrase(self, tmp_path, capsys):
        _, key_path = _seed_daily(tmp_path, "CORRECT")
        before = key_path.read_bytes()
        old_file = _write_file(tmp_path / "old.txt", "WRONG\n")
        new_file = _write_file(tmp_path / "new.txt", "new-pw\n")

        rc = change_passphrase.run(
            _args(tmp_path, old_passphrase_file=old_file, new_passphrase_file=new_file)
        )
        assert rc == 1
        err = capsys.readouterr().err
        assert "does not unwrap" in err
        # daily_key.age bytes unchanged — no partial mutation on wrong old.
        assert key_path.read_bytes() == before
        load_daily_identity(key_path, passphrase="CORRECT")

    def test_empty_old_passphrase_file(self, tmp_path, capsys):
        _seed_daily(tmp_path, "CORRECT")
        old_file = _write_file(tmp_path / "old.txt", "")
        new_file = _write_file(tmp_path / "new.txt", "new-pw\n")

        rc = change_passphrase.run(
            _args(tmp_path, old_passphrase_file=old_file, new_passphrase_file=new_file)
        )
        assert rc == 1
        assert "old passphrase file is empty" in capsys.readouterr().err

    def test_empty_new_passphrase_file(self, tmp_path, capsys):
        _seed_daily(tmp_path, "OLD")
        old_file = _write_file(tmp_path / "old.txt", "OLD\n")
        new_file = _write_file(tmp_path / "new.txt", "")

        rc = change_passphrase.run(
            _args(tmp_path, old_passphrase_file=old_file, new_passphrase_file=new_file)
        )
        assert rc == 1
        assert "new passphrase file is empty" in capsys.readouterr().err

    def test_old_equals_new_refuses(self, tmp_path, capsys):
        _, key_path = _seed_daily(tmp_path, "SAME")
        before = key_path.read_bytes()
        old_file = _write_file(tmp_path / "old.txt", "SAME\n")
        new_file = _write_file(tmp_path / "new.txt", "SAME\n")

        rc = change_passphrase.run(
            _args(tmp_path, old_passphrase_file=old_file, new_passphrase_file=new_file)
        )
        assert rc == 1
        assert "identical" in capsys.readouterr().err
        # File untouched.
        assert key_path.read_bytes() == before

    def test_missing_old_passphrase_file(self, tmp_path, capsys):
        _seed_daily(tmp_path, "OLD")
        new_file = _write_file(tmp_path / "new.txt", "NEW\n")

        rc = change_passphrase.run(
            _args(
                tmp_path,
                old_passphrase_file=tmp_path / "nope.txt",
                new_passphrase_file=new_file,
            )
        )
        assert rc == 1
        assert "could not read old passphrase file" in capsys.readouterr().err

    def test_no_old_passphrase_no_env_no_tty(self, tmp_path, capsys, monkeypatch):
        _seed_daily(tmp_path, "OLD")
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)

        rc = change_passphrase.run(_args(tmp_path))
        assert rc == 1
        err = capsys.readouterr().err
        assert "no old passphrase supplied" in err
        assert DAILY_PASSPHRASE_ENV_VAR in err

    def test_no_new_passphrase_no_tty(self, tmp_path, capsys, monkeypatch):
        _seed_daily(tmp_path, "OLD")
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        old_file = _write_file(tmp_path / "old.txt", "OLD\n")

        rc = change_passphrase.run(_args(tmp_path, old_passphrase_file=old_file))
        assert rc == 1
        assert "no new passphrase supplied" in capsys.readouterr().err


class TestInteractivePrompts:
    def test_fully_interactive(self, tmp_path, monkeypatch):
        _, key_path = _seed_daily(tmp_path, "OLD_INT")
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        answers = iter(["OLD_INT", "NEW_INT", "NEW_INT"])
        monkeypatch.setattr(
            "paramem.cli.change_passphrase.getpass.getpass", lambda prompt: next(answers)
        )

        rc = change_passphrase.run(_args(tmp_path))
        assert rc == 0
        load_daily_identity(key_path, passphrase="NEW_INT")

    def test_interactive_new_mismatch_refuses(self, tmp_path, monkeypatch, capsys):
        _, key_path = _seed_daily(tmp_path, "OLD")
        before = key_path.read_bytes()
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        answers = iter(["OLD", "new-a", "new-b"])
        monkeypatch.setattr(
            "paramem.cli.change_passphrase.getpass.getpass", lambda prompt: next(answers)
        )

        rc = change_passphrase.run(_args(tmp_path))
        assert rc == 1
        assert "did not match" in capsys.readouterr().err
        # daily_key.age unchanged — OLD still works.
        assert key_path.read_bytes() == before
        load_daily_identity(key_path, passphrase="OLD")

    def test_interactive_empty_old_refuses(self, tmp_path, monkeypatch, capsys):
        _seed_daily(tmp_path, "OLD")
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("paramem.cli.change_passphrase.getpass.getpass", lambda prompt: "")

        rc = change_passphrase.run(_args(tmp_path))
        assert rc == 1
        assert "old passphrase must be non-empty" in capsys.readouterr().err

    def test_interactive_empty_new_refuses(self, tmp_path, monkeypatch, capsys):
        _seed_daily(tmp_path, "OLD")
        monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "OLD")  # old from env
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        # New prompt returns empty.
        monkeypatch.setattr("paramem.cli.change_passphrase.getpass.getpass", lambda prompt: "")

        rc = change_passphrase.run(_args(tmp_path))
        assert rc == 1
        assert "new passphrase must be non-empty" in capsys.readouterr().err


class TestAtomicity:
    def test_write_failure_leaves_old_file_intact(self, tmp_path, monkeypatch, capsys):
        """If write_daily_key_file raises mid-write, the on-disk daily_key.age
        stays on its OLD wrapping. The primitive's O_EXCL + atomic-rename
        contract guarantees this; re-test here against the change-passphrase
        code path specifically."""
        _, key_path = _seed_daily(tmp_path, "OLD")
        before = key_path.read_bytes()

        old_file = _write_file(tmp_path / "old.txt", "OLD\n")
        new_file = _write_file(tmp_path / "new.txt", "NEW\n")

        def raising_write(*args, **kwargs):
            raise OSError("disk full (simulated)")

        monkeypatch.setattr("paramem.cli.change_passphrase.write_daily_key_file", raising_write)

        with pytest.raises(OSError, match="simulated"):
            change_passphrase.run(
                _args(
                    tmp_path,
                    old_passphrase_file=old_file,
                    new_passphrase_file=new_file,
                )
            )

        # File byte-identical to pre-run state.
        assert key_path.read_bytes() == before
        load_daily_identity(key_path, passphrase="OLD")


class TestCliEntryPoint:
    def test_subcommand_registered(self):
        from paramem.cli import main as main_module

        parser = main_module._build_parser()
        args = parser.parse_args(["change-passphrase", "--daily-key-path", "/tmp/x.age"])
        assert args.command == "change-passphrase"
        assert str(args.daily_key_path) == "/tmp/x.age"

    def test_help_includes_change_passphrase(self):
        """Regression guard — paramem --help must list the command so
        operators discovering the CLI see it."""
        from paramem.cli import main as main_module

        parser = main_module._build_parser()
        help_text = parser.format_help()
        assert "change-passphrase" in help_text
