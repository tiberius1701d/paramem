"""Tests for ``paramem migrate-to-age``.

Covers the happy path, idempotency, dry-run, precondition refusals, mixed
plaintext refusal, and per-file atomic-rename safety.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from pyrage import x25519

from paramem.backup.age_envelope import age_encrypt_bytes, is_age_envelope
from paramem.backup.encryption import (
    MASTER_KEY_ENV_VAR,
    _clear_cipher_cache,
    is_pmem1_envelope,
    read_maybe_encrypted,
    write_infra_bytes,
)
from paramem.backup.key_store import (
    DAILY_PASSPHRASE_ENV_VAR,
    _clear_daily_identity_cache,
    mint_daily_identity,
    wrap_daily_identity,
    write_daily_key_file,
    write_recovery_pub_file,
)
from paramem.cli import migrate_to_age


def _make_fernet_key() -> str:
    return Fernet.generate_key().decode()


@pytest.fixture(autouse=True)
def _env_and_cache_isolation():
    os.environ.pop(MASTER_KEY_ENV_VAR, None)
    os.environ.pop(DAILY_PASSPHRASE_ENV_VAR, None)
    _clear_cipher_cache()
    _clear_daily_identity_cache()
    yield
    os.environ.pop(MASTER_KEY_ENV_VAR, None)
    os.environ.pop(DAILY_PASSPHRASE_ENV_VAR, None)
    _clear_cipher_cache()
    _clear_daily_identity_cache()


def _setup_fernet_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Load just the Fernet master key — the pre-migration posture. The daily
    identity is NOT loaded, so write_infra_bytes produces PMEM1."""
    os.environ[MASTER_KEY_ENV_VAR] = _make_fernet_key()
    # Point the daily path defaults at non-existent files so the probe returns
    # False even if a leftover env var lingers.
    monkeypatch.setattr(
        "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT",
        tmp_path / "absent-daily.age",
    )
    monkeypatch.setattr(
        "paramem.backup.key_store.RECOVERY_PUB_PATH_DEFAULT",
        tmp_path / "absent-recovery.pub",
    )


def _load_age_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Mint + load daily + recovery identities. Mirrors the real-world flow:
    after PMEM1 data is on disk, the operator runs `paramem generate-key` and
    then invokes migrate-to-age."""
    daily = mint_daily_identity()
    daily_path = tmp_path / "daily_key.age"
    write_daily_key_file(wrap_daily_identity(daily, "pw"), daily_path)
    monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "pw")
    monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", daily_path)
    _clear_daily_identity_cache()

    recovery = x25519.Identity.generate()
    recovery_path = tmp_path / "recovery.pub"
    write_recovery_pub_file(recovery.to_public(), recovery_path)
    monkeypatch.setattr("paramem.backup.key_store.RECOVERY_PUB_PATH_DEFAULT", recovery_path)
    return daily, recovery


def _setup_all_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Convenience wrapper for tests that don't care about seeding order."""
    _setup_fernet_only(tmp_path, monkeypatch)
    return _load_age_keys(tmp_path, monkeypatch)


def _seed_pmem1_files(data_dir: Path) -> list[tuple[Path, bytes]]:
    """Write a handful of PMEM1 infrastructure files.

    The caller must have loaded the Fernet master key and kept the daily
    identity OUT of the writer's view (see :func:`_setup_fernet_only`) so
    ``write_infra_bytes`` produces PMEM1 instead of age.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "registry").mkdir(parents=True, exist_ok=True)
    (data_dir / "adapters").mkdir(parents=True, exist_ok=True)

    seeded: list[tuple[Path, bytes]] = []
    for rel, content in [
        (Path("registry.json"), b'{"graph1": "abc"}'),
        (Path("speaker_profiles.json"), b'{"speakers": {}, "version": 5}'),
        (Path("adapters/post_session_queue.json"), b"[]"),
    ]:
        p = data_dir / rel
        write_infra_bytes(p, content)
        assert is_pmem1_envelope(p), f"seed precondition: {p} must be PMEM1"
        seeded.append((p, content))
    return seeded


def _default_args(
    data_dir: Path,
    *,
    dry_run: bool = False,
    verbose: bool = False,
    continue_on_error: bool = False,
    allow_daily_only: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        data_dir=str(data_dir),
        config="configs/server.yaml",
        dry_run=dry_run,
        verbose=verbose,
        continue_on_error=continue_on_error,
        allow_daily_only=allow_daily_only,
    )


class TestHappyPath:
    def test_migrates_all_pmem1_to_age(self, tmp_path, monkeypatch, capsys):
        # Real operational flow: PMEM1 store exists first, then operator runs
        # generate-key, then migrate-to-age.
        _setup_fernet_only(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        seeded = _seed_pmem1_files(data_dir)
        _load_age_keys(tmp_path, monkeypatch)

        rc = migrate_to_age.run(_default_args(data_dir))
        assert rc == 0

        for path, original in seeded:
            assert is_age_envelope(path), f"{path} must be age after migration"
            assert not is_pmem1_envelope(path)
            assert read_maybe_encrypted(path) == original, (
                f"{path} plaintext must survive the format flip bitwise"
            )

        out = capsys.readouterr().out
        assert f"{len(seeded)} migrated" in out

    def test_dry_run_does_not_modify_files(self, tmp_path, monkeypatch, capsys):
        _setup_fernet_only(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        seeded = _seed_pmem1_files(data_dir)
        _load_age_keys(tmp_path, monkeypatch)
        before = {path: path.read_bytes() for path, _ in seeded}

        rc = migrate_to_age.run(_default_args(data_dir, dry_run=True))
        assert rc == 0

        for path, _ in seeded:
            assert path.read_bytes() == before[path], "dry-run must not mutate"
            assert is_pmem1_envelope(path)

        out = capsys.readouterr().out
        assert "(dry-run)" in out

    def test_idempotent_re_run_skips_age_files(self, tmp_path, monkeypatch, capsys):
        _setup_fernet_only(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        seeded = _seed_pmem1_files(data_dir)
        _load_age_keys(tmp_path, monkeypatch)

        assert migrate_to_age.run(_default_args(data_dir)) == 0
        capsys.readouterr()  # drain

        # Snapshot the migrated bytes — a second run should leave them verbatim.
        first_run = {path: path.read_bytes() for path, _ in seeded}

        rc = migrate_to_age.run(_default_args(data_dir))
        assert rc == 0

        for path, _ in seeded:
            assert path.read_bytes() == first_run[path], (
                "second run must not re-encrypt already-age files"
            )

        out = capsys.readouterr().out
        assert "0 migrated" in out
        assert f"{len(seeded)} already age" in out

    def test_empty_data_dir_succeeds(self, tmp_path, monkeypatch, capsys):
        _setup_all_keys(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        rc = migrate_to_age.run(_default_args(data_dir))
        assert rc == 0
        assert "0 migrated" in capsys.readouterr().out

    def test_verbose_logs_per_file(self, tmp_path, monkeypatch, capsys):
        _setup_fernet_only(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        seeded = _seed_pmem1_files(data_dir)
        _load_age_keys(tmp_path, monkeypatch)

        assert migrate_to_age.run(_default_args(data_dir, verbose=True)) == 0
        out = capsys.readouterr().out
        for path, _ in seeded:
            assert str(path) in out, f"verbose mode must log {path}"


class TestPreconditionRefusals:
    def test_refuses_without_fernet_key(self, tmp_path, monkeypatch, capsys):
        _setup_all_keys(tmp_path, monkeypatch)
        # Drop the Fernet key after seeding preconditions.
        os.environ.pop(MASTER_KEY_ENV_VAR, None)
        _clear_cipher_cache()

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        rc = migrate_to_age.run(_default_args(data_dir))
        assert rc == 1
        assert MASTER_KEY_ENV_VAR in capsys.readouterr().err

    def test_refuses_without_daily_identity(self, tmp_path, monkeypatch, capsys):
        _setup_all_keys(tmp_path, monkeypatch)
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        rc = migrate_to_age.run(_default_args(data_dir))
        assert rc == 1
        err = capsys.readouterr().err
        assert "daily identity is not loadable" in err.lower() or (
            "daily identity is not loadable" in err.casefold()
        )
        assert DAILY_PASSPHRASE_ENV_VAR in err

    def test_refuses_without_recovery_pub(self, tmp_path, monkeypatch, capsys):
        _setup_all_keys(tmp_path, monkeypatch)
        # Point recovery at a missing path.
        monkeypatch.setattr(
            "paramem.backup.key_store.RECOVERY_PUB_PATH_DEFAULT",
            tmp_path / "absent.pub",
        )

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        rc = migrate_to_age.run(_default_args(data_dir))
        assert rc == 1
        err = capsys.readouterr().err
        assert "Recovery public recipient is missing" in err
        assert "allow-daily-only" in err

    def test_allow_daily_only_bypasses_recovery_refusal(self, tmp_path, monkeypatch, capsys):
        _setup_fernet_only(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        seeded = _seed_pmem1_files(data_dir)
        _load_age_keys(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "paramem.backup.key_store.RECOVERY_PUB_PATH_DEFAULT",
            tmp_path / "absent.pub",
        )

        rc = migrate_to_age.run(_default_args(data_dir, allow_daily_only=True))
        assert rc == 0

        err = capsys.readouterr().err
        assert "--allow-daily-only" in err  # WARN banner
        for path, _ in seeded:
            assert is_age_envelope(path)

    def test_missing_data_dir_returns_error(self, tmp_path, monkeypatch, capsys):
        _setup_all_keys(tmp_path, monkeypatch)
        rc = migrate_to_age.run(_default_args(tmp_path / "does-not-exist"))
        assert rc == 1
        assert "does not exist" in capsys.readouterr().err


class TestPlaintextRefusal:
    def test_refuses_when_plaintext_infra_present(self, tmp_path, monkeypatch, capsys):
        _setup_all_keys(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # Plaintext infrastructure file — migrate-to-age should refuse since
        # encrypt-infra needs to run first.
        (data_dir / "registry.json").write_bytes(b'{"plain": true}')

        rc = migrate_to_age.run(_default_args(data_dir))
        assert rc == 1
        err = capsys.readouterr().err
        assert "plaintext" in err
        assert "encrypt-infra" in err


class TestMixedState:
    def test_pmem1_plus_already_age_handled_correctly(self, tmp_path, monkeypatch, capsys):
        """Partial-migration resume: some files are already age, others still
        PMEM1. The command migrates the PMEM1 tail and leaves the age files
        alone."""
        _setup_fernet_only(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # One PMEM1 file — seeded while only Fernet is loaded.
        pmem1_path = data_dir / "registry.json"
        write_infra_bytes(pmem1_path, b'{"tail": 1}')
        assert is_pmem1_envelope(pmem1_path)

        daily, recovery = _load_age_keys(tmp_path, monkeypatch)

        # One already-age file — placed directly to simulate prior migration progress.
        age_path = data_dir / "speaker_profiles.json"
        age_path.write_bytes(
            age_encrypt_bytes(b'{"already": true}', [daily.to_public(), recovery.to_public()])
        )
        before_age_bytes = age_path.read_bytes()

        rc = migrate_to_age.run(_default_args(data_dir))
        assert rc == 0

        assert is_age_envelope(pmem1_path), "PMEM1 tail must be migrated"
        assert age_path.read_bytes() == before_age_bytes, (
            "already-age file must remain byte-identical"
        )

        out = capsys.readouterr().out
        assert "1 migrated" in out
        assert "1 already age" in out


class TestMigratedEnvelopeShape:
    def test_migrated_envelopes_are_multi_recipient(self, tmp_path, monkeypatch, capsys):
        """Migrated files must list BOTH daily and recovery as recipients, so
        the recovery identity still opens them on hardware replacement."""
        from paramem.backup.age_envelope import age_decrypt_bytes

        _setup_fernet_only(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        seeded = _seed_pmem1_files(data_dir)
        daily, recovery = _load_age_keys(tmp_path, monkeypatch)

        assert migrate_to_age.run(_default_args(data_dir)) == 0

        for path, original in seeded:
            envelope = path.read_bytes()
            # Daily decrypts (common path).
            assert age_decrypt_bytes(envelope, [daily]) == original
            # Recovery also decrypts (hardware-replacement path).
            assert age_decrypt_bytes(envelope, [recovery]) == original


class TestAtomicPerFile:
    def test_nothing_persists_if_a_single_file_fails(self, tmp_path, monkeypatch, capsys):
        """If the age-encrypt step raises for one file, the tmp is cleaned and
        the source PMEM1 envelope is left untouched. Other files in the sweep
        that had already succeeded remain migrated (the command is per-file
        atomic, not batch-atomic)."""
        _setup_fernet_only(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        seeded = _seed_pmem1_files(data_dir)
        _load_age_keys(tmp_path, monkeypatch)

        # Fail on the second file only.
        call_count = {"n": 0}
        original_encrypt = migrate_to_age.age_encrypt_bytes

        def flaky_encrypt(plaintext, recipients):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("simulated encrypt failure")
            return original_encrypt(plaintext, recipients)

        monkeypatch.setattr("paramem.cli.migrate_to_age.age_encrypt_bytes", flaky_encrypt)

        rc = migrate_to_age.run(_default_args(data_dir, continue_on_error=True))
        assert rc == 2, "exit code reflects at least one failure"

        # First seeded file migrated (succeeded), second left PMEM1 (failed),
        # third migrated (succeeded after continue_on_error).
        assert is_age_envelope(seeded[0][0])
        assert is_pmem1_envelope(seeded[1][0]), "failed file must keep its PMEM1 body"
        # No leftover tmp for the failed file.
        failed_tmp = seeded[1][0].with_suffix(seeded[1][0].suffix + ".tmp")
        assert not failed_tmp.exists(), "tmp must be cleaned up on failure"


class TestCliEntryPoint:
    def test_subcommand_registered(self):
        """Smoke check that argparse accepts `migrate-to-age --help`."""
        from paramem.cli import main as main_module

        parser = main_module._build_parser()
        args = parser.parse_args(["migrate-to-age", "--data-dir", "/tmp/ignored", "--dry-run"])
        assert args.command == "migrate-to-age"
        assert args.dry_run is True
        assert args.data_dir == "/tmp/ignored"
