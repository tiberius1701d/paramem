"""Tests for ``paramem encrypt-infra``.

Covers:
- Refuses when PARAMEM_DAILY_PASSPHRASE is unset (exit 1, message names the env var).
- --dry-run lists plaintext files but does not mutate them.
- Mixed store (some age, some plaintext) → exit 0, plaintext become age envelopes,
  age files unchanged.
- Pure-age store → no-op, exit 0, no writes (mtimes unchanged).
- --continue-on-error: one read failure → rest still encrypted, exit 2.
- Idempotency: two consecutive runs leave identical bytes on disk.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pytest

from paramem.backup.age_envelope import is_age_envelope
from paramem.backup.encryption import envelope_encrypt_bytes
from paramem.backup.key_store import (
    DAILY_PASSPHRASE_ENV_VAR,
    _clear_daily_identity_cache,
    mint_daily_identity,
    wrap_daily_identity,
    write_daily_key_file,
)
from paramem.cli import encrypt_infra

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _env_and_cache_isolation(monkeypatch):
    """Remove the passphrase env var and clear key cache around every test."""
    monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
    _clear_daily_identity_cache()
    yield
    _clear_daily_identity_cache()


def _setup_daily(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passphrase: str = "test-pw",
) -> Path:
    """Mint a daily identity, write it, wire env var and module default.

    Returns the path to the daily key file.
    """
    ident = mint_daily_identity()
    key_path = tmp_path / "daily_key.age"
    write_daily_key_file(wrap_daily_identity(ident, passphrase), key_path)
    monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, passphrase)
    monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path)
    _clear_daily_identity_cache()
    return key_path


def _make_args(
    config: Path,
    *,
    dry_run: bool = False,
    verbose: bool = False,
    continue_on_error: bool = False,
) -> argparse.Namespace:
    """Build a Namespace matching the parser's output."""
    return argparse.Namespace(
        config=config,
        dry_run=dry_run,
        verbose=verbose,
        continue_on_error=continue_on_error,
    )


def _write_plaintext(path: Path, content: bytes = b'{"key": "value"}') -> Path:
    """Write a plaintext file, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _write_age(path: Path, content: bytes = b'{"a": 1}') -> Path:
    """Write an age-encrypted file to *path* using the currently wired daily identity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encrypted = envelope_encrypt_bytes(content)
    path.write_bytes(encrypted)
    return path


def _make_config_yaml(tmp_path: Path, data_dir: Path) -> Path:
    """Write a minimal server.yaml pointing at *data_dir*."""
    cfg_path = tmp_path / "server.yaml"
    cfg_path.write_text(
        f"model: mistral\npaths:\n  data: {data_dir}\n  simulate: {data_dir}/simulate\n",
        encoding="utf-8",
    )
    return cfg_path


# ---------------------------------------------------------------------------
# Refusal when no daily identity
# ---------------------------------------------------------------------------


class TestRefuseWithoutDailyIdentity:
    def test_refuses_when_env_unset(self, tmp_path, capsys):
        """Exit 1 and message names PARAMEM_DAILY_PASSPHRASE when env is unset."""
        # Do NOT set up a daily identity — env var stays unset (autouse fixture).
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)

        rc = encrypt_infra.run(_make_args(cfg))

        assert rc == 1
        err = capsys.readouterr().err
        assert DAILY_PASSPHRASE_ENV_VAR in err

    def test_dry_run_does_not_require_identity(self, tmp_path, capsys):
        """--dry-run must not gate on daily identity; it only lists files."""
        data_dir = tmp_path / "data"
        _write_plaintext(data_dir / "graph.json")
        cfg = _make_config_yaml(tmp_path, data_dir)

        # No daily identity wired — but dry_run=True.
        rc = encrypt_infra.run(_make_args(cfg, dry_run=True))

        assert rc == 0
        out = capsys.readouterr().out
        assert "graph.json" in out or "would encrypt" in out


# ---------------------------------------------------------------------------
# Dry-run: lists without mutating
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_lists_plaintext_no_mutation(self, tmp_path, monkeypatch, capsys):
        """--dry-run lists plaintext files; bytes and mtime are unchanged."""
        _setup_daily(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        target = _write_plaintext(data_dir / "graph.json", b"PLAIN")

        cfg = _make_config_yaml(tmp_path, data_dir)
        before_bytes = target.read_bytes()
        before_mtime = target.stat().st_mtime

        # Tiny sleep so mtime would differ if a write happened.
        time.sleep(0.01)

        rc = encrypt_infra.run(_make_args(cfg, dry_run=True))

        assert rc == 0
        assert target.read_bytes() == before_bytes, "dry-run must not mutate bytes"
        assert target.stat().st_mtime == before_mtime, "dry-run must not touch mtime"
        out = capsys.readouterr().out
        assert "graph.json" in out

    def test_dry_run_no_plaintext_summary_says_zero(self, tmp_path, monkeypatch, capsys):
        """When store is already encrypted, dry-run summary says 0 to encrypt."""
        _setup_daily(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        # Write an age-encrypted file.
        _write_age(data_dir / "graph.json")

        cfg = _make_config_yaml(tmp_path, data_dir)

        rc = encrypt_infra.run(_make_args(cfg, dry_run=True))

        assert rc == 0
        out = capsys.readouterr().out
        assert "would encrypt 0" in out


# ---------------------------------------------------------------------------
# Mixed store: plaintext alongside age
# ---------------------------------------------------------------------------


class TestMixedStore:
    def test_mixed_store_encrypts_plaintext_leaves_age_unchanged(
        self, tmp_path, monkeypatch, capsys
    ):
        """Mixed store → exit 0; plaintext become age envelopes; age files unchanged."""
        _setup_daily(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"

        plaintext_path = _write_plaintext(data_dir / "graph.json", b'{"plain": true}')
        age_path = _write_age(data_dir / "speaker_profiles.json")
        age_before = age_path.read_bytes()

        cfg = _make_config_yaml(tmp_path, data_dir)

        rc = encrypt_infra.run(_make_args(cfg))

        assert rc == 0

        # Previously-plaintext file is now an age envelope.
        assert is_age_envelope(plaintext_path), "plaintext file must be encrypted after run"

        # Previously-age file is byte-identical.
        assert age_path.read_bytes() == age_before, "age file must be unchanged"

        out = capsys.readouterr().out
        assert "encrypted 1" in out

    def test_plaintext_content_survives_round_trip(self, tmp_path, monkeypatch):
        """Encrypted content must decrypt back to original plaintext."""
        from paramem.backup.encryption import read_maybe_encrypted

        _setup_daily(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        original = b'{"round_trip": "ok"}'
        target = _write_plaintext(data_dir / "graph.json", original)

        cfg = _make_config_yaml(tmp_path, data_dir)
        encrypt_infra.run(_make_args(cfg))

        assert is_age_envelope(target)
        recovered = read_maybe_encrypted(target)
        assert recovered == original


# ---------------------------------------------------------------------------
# Pure-age store: no-op
# ---------------------------------------------------------------------------


class TestPureAgeStore:
    def test_pure_age_store_is_noop(self, tmp_path, monkeypatch, capsys):
        """When every file is already an age envelope, nothing is written."""
        _setup_daily(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        age_path = _write_age(data_dir / "graph.json")
        mtime_before = age_path.stat().st_mtime
        bytes_before = age_path.read_bytes()

        cfg = _make_config_yaml(tmp_path, data_dir)

        time.sleep(0.01)
        rc = encrypt_infra.run(_make_args(cfg))

        assert rc == 0
        assert age_path.read_bytes() == bytes_before, "age file bytes must not change"
        assert age_path.stat().st_mtime == mtime_before, "age file mtime must not change"

        out = capsys.readouterr().out
        assert "encrypted 0" in out


# ---------------------------------------------------------------------------
# --continue-on-error: partial failure
# ---------------------------------------------------------------------------


class TestContinueOnError:
    def test_continue_on_error_partial_success_exits_2(self, tmp_path, monkeypatch, capsys):
        """One read failure → rest still encrypted; exit code is 2."""
        _setup_daily(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"

        good_path = _write_plaintext(data_dir / "graph.json", b'{"good": true}')
        bad_path = _write_plaintext(data_dir / "registry.json", b'{"bad": true}')

        cfg = _make_config_yaml(tmp_path, data_dir)

        # Make bad_path unreadable.
        bad_path.chmod(0o000)

        try:
            rc = encrypt_infra.run(_make_args(cfg, continue_on_error=True))
        finally:
            # Restore so tmp_path cleanup can delete it.
            bad_path.chmod(0o644)

        assert rc == 2

        # The good file should be encrypted.
        assert is_age_envelope(good_path), "good file must be encrypted"

        err = capsys.readouterr().err
        assert "registry.json" in err or "failed" in err

    def test_stop_on_first_error_by_default(self, tmp_path, monkeypatch, capsys):
        """Without --continue-on-error, first failure returns exit 1."""
        _setup_daily(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"

        bad_path = _write_plaintext(data_dir / "graph.json", b'{"x": 1}')

        cfg = _make_config_yaml(tmp_path, data_dir)

        bad_path.chmod(0o000)
        try:
            rc = encrypt_infra.run(_make_args(cfg, continue_on_error=False))
        finally:
            bad_path.chmod(0o644)

        assert rc == 1


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_two_runs_leave_identical_bytes(self, tmp_path, monkeypatch):
        """Running encrypt-infra twice on the same store leaves identical bytes."""
        _setup_daily(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        target = _write_plaintext(data_dir / "graph.json", b'{"idempotent": true}')

        cfg = _make_config_yaml(tmp_path, data_dir)

        encrypt_infra.run(_make_args(cfg))
        bytes_after_first = target.read_bytes()
        assert is_age_envelope(target)

        encrypt_infra.run(_make_args(cfg))
        bytes_after_second = target.read_bytes()

        assert bytes_after_first == bytes_after_second, "second run must not change the file"


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------


class TestCliRegistration:
    def test_subcommand_registered(self):
        """encrypt-infra must appear in the top-level parser."""
        from paramem.cli import main as main_module

        parser = main_module._build_parser()
        args = parser.parse_args(["encrypt-infra", "--dry-run"])
        assert args.command == "encrypt-infra"
        assert args.dry_run is True

    def test_help_includes_encrypt_infra(self):
        """paramem --help must list encrypt-infra."""
        from paramem.cli import main as main_module

        parser = main_module._build_parser()
        help_text = parser.format_help()
        assert "encrypt-infra" in help_text

    def test_all_flags_present_in_help(self):
        """--dry-run, --verbose, --continue-on-error must be accepted by the parser."""
        from paramem.cli import main as main_module

        parser = main_module._build_parser()

        # Parse with the subcommand to exercise it.
        args = parser.parse_args(
            [
                "encrypt-infra",
                "--dry-run",
                "--verbose",
                "--continue-on-error",
                "--config",
                "/tmp/cfg.yaml",
            ]
        )
        assert args.dry_run is True
        assert args.verbose is True
        assert args.continue_on_error is True
        assert str(args.config) == "/tmp/cfg.yaml"


# ---------------------------------------------------------------------------
# Verbose flag
# ---------------------------------------------------------------------------


class TestVerboseFlag:
    def test_verbose_logs_skipped_files(self, tmp_path, monkeypatch, capsys):
        """--verbose must log already-encrypted files as skipped."""
        _setup_daily(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        _write_age(data_dir / "graph.json")

        cfg = _make_config_yaml(tmp_path, data_dir)

        encrypt_infra.run(_make_args(cfg, verbose=True))

        out = capsys.readouterr().out
        assert "skip" in out.lower() or "graph.json" in out
