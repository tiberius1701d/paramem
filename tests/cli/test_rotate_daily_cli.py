"""Tests for ``paramem rotate-daily``.

Covers the happy path, dry-run, precondition refusals, resume after a
simulated mid-sweep crash, and the invariant that recovery continues to
decrypt post-rotation.
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
    PMEM1_MAGIC,
    _clear_cipher_cache,
    encrypt_bytes,
)
from paramem.backup.key_store import (
    DAILY_PASSPHRASE_ENV_VAR,
    _clear_daily_identity_cache,
    load_daily_identity,
    mint_daily_identity,
    wrap_daily_identity,
    write_daily_key_file,
    write_recovery_pub_file,
)
from paramem.backup.rotation import write_manifest_atomic
from paramem.cli import rotate_daily


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
    """Mint daily + recovery, point module defaults at tmp_path, set env."""
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
        (Path("adapters/post_session_queue.json"), b'{"q": 3}'),
    ]:
        p = data_dir / rel
        p.write_bytes(age_encrypt_bytes(content, recipients))
        seeded.append((p, content))
    return seeded


def _default_args(data_dir: Path, *, dry_run=False, verbose=False) -> argparse.Namespace:
    return argparse.Namespace(
        data_dir=str(data_dir),
        config="configs/server.yaml",
        dry_run=dry_run,
        verbose=verbose,
    )


class TestHappyPath:
    def test_rotates_all_age_files_to_new_daily(self, tmp_path, monkeypatch, capsys):
        daily, recovery = _setup_keys(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        seeded = _seed_age_files(data_dir, daily, recovery)

        rc = rotate_daily.run(_default_args(data_dir, verbose=True))
        assert rc == 0

        # daily_key.age has been swapped — load NEW daily from disk.
        new_daily = load_daily_identity(tmp_path / "daily_key.age", passphrase="pw")
        assert str(new_daily) != str(daily), "new daily must differ from old"

        for path, original in seeded:
            # OLD daily must no longer decrypt.
            with pytest.raises(pyrage.DecryptError):
                age_decrypt_bytes(path.read_bytes(), [daily])
            # NEW daily decrypts.
            assert age_decrypt_bytes(path.read_bytes(), [new_daily]) == original
            # Recovery still decrypts — hardware replacement still works.
            assert age_decrypt_bytes(path.read_bytes(), [recovery]) == original

        out = capsys.readouterr().out
        assert f"{len(seeded)} rotated" in out

    def test_pending_file_and_manifest_cleaned_up(self, tmp_path, monkeypatch, capsys):
        daily, recovery = _setup_keys(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        _seed_age_files(data_dir, daily, recovery)

        rc = rotate_daily.run(_default_args(data_dir))
        assert rc == 0
        assert not (tmp_path / "daily_key.age.pending").exists()
        assert not (tmp_path / "rotation.manifest.json").exists()

    def test_dry_run_mutates_nothing(self, tmp_path, monkeypatch, capsys):
        daily, recovery = _setup_keys(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        seeded = _seed_age_files(data_dir, daily, recovery)
        before = {p: p.read_bytes() for p, _ in seeded}

        rc = rotate_daily.run(_default_args(data_dir, dry_run=True, verbose=True))
        assert rc == 0
        for p, _ in seeded:
            assert p.read_bytes() == before[p]
        # OLD daily still decrypts every file — rotation is a no-op.
        for p, original in seeded:
            assert age_decrypt_bytes(p.read_bytes(), [daily]) == original
        # Dry-run must not leave rotation artefacts behind on a fresh start;
        # otherwise a subsequent non-dry-run would misinterpret the pending
        # key file as crash-resume state.
        assert not (tmp_path / "daily_key.age.pending").exists()
        assert not (tmp_path / "rotation.manifest.json").exists()
        out = capsys.readouterr().out
        assert "[dry-run]" in out

    def test_dry_run_preserves_prior_resume_state(self, tmp_path, monkeypatch, capsys):
        """If a resume is in progress, dry-run reports against the existing
        pending state without clobbering it — the operator can inspect the
        crash-resume situation without destroying recovery info."""
        daily, recovery = _setup_keys(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        _seed_age_files(data_dir, daily, recovery)

        # Pre-stage a crash state: pending key + manifest.
        new_daily = mint_daily_identity()
        pending_path = tmp_path / "daily_key.age.pending"
        write_daily_key_file(wrap_daily_identity(new_daily, "pw"), pending_path)

        from paramem.backup.rotation import RotationManifest

        manifest_path = tmp_path / "rotation.manifest.json"
        manifest = RotationManifest(
            operation="rotate-daily",
            started_at="2026-04-24T12:00:00Z",
            new_daily_pub=str(new_daily.to_public()),
            files_pending=["/nonexistent/x"],
            files_done=[],
        )
        write_manifest_atomic(manifest_path, manifest)

        pending_before = pending_path.read_bytes()
        manifest_before = manifest_path.read_bytes()

        rc = rotate_daily.run(_default_args(data_dir, dry_run=True))
        assert rc == 0

        # Pending + manifest preserved — dry-run on a resume path leaves the
        # crash state alone.
        assert pending_path.read_bytes() == pending_before
        assert manifest_path.read_bytes() == manifest_before


class TestPreconditions:
    def test_refuses_without_daily_identity(self, tmp_path, monkeypatch, capsys):
        _setup_keys(tmp_path, monkeypatch)
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        rc = rotate_daily.run(_default_args(data_dir))
        assert rc == 1
        assert "Daily identity is not loadable" in capsys.readouterr().err

    def test_refuses_without_recovery_pub(self, tmp_path, monkeypatch, capsys):
        _setup_keys(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "paramem.backup.key_store.RECOVERY_PUB_PATH_DEFAULT",
            tmp_path / "absent.pub",
        )
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        rc = rotate_daily.run(_default_args(data_dir))
        assert rc == 1
        assert "Recovery public recipient is missing" in capsys.readouterr().err

    def test_refuses_when_pmem1_files_present(self, tmp_path, monkeypatch, capsys):
        daily, recovery = _setup_keys(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # Drop a PMEM1 file among the infra paths.
        os.environ[MASTER_KEY_ENV_VAR] = (
            __import__("cryptography.fernet", fromlist=["Fernet"]).Fernet.generate_key().decode()
        )
        pmem1_path = data_dir / "registry.json"
        pmem1_path.write_bytes(PMEM1_MAGIC + encrypt_bytes(b"{}"))

        rc = rotate_daily.run(_default_args(data_dir))
        assert rc == 1
        assert "PMEM1" in capsys.readouterr().err

    def test_refuses_missing_data_dir(self, tmp_path, monkeypatch, capsys):
        _setup_keys(tmp_path, monkeypatch)
        rc = rotate_daily.run(_default_args(tmp_path / "does-not-exist"))
        assert rc == 1
        assert "does not exist" in capsys.readouterr().err


class TestResumeAfterCrash:
    def test_resume_with_pending_key_file_and_manifest(self, tmp_path, monkeypatch, capsys):
        """Simulate a prior crash: daily_key.age.pending already written, one
        file already rotated, manifest says the other two are still pending."""
        daily, recovery = _setup_keys(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        seeded = _seed_age_files(data_dir, daily, recovery)

        # Pre-stage: mint a NEW daily, wrap it to the pending path, rotate ONE
        # file by hand, mark that one as done in the manifest.
        new_daily = mint_daily_identity()
        pending_path = tmp_path / "daily_key.age.pending"
        write_daily_key_file(wrap_daily_identity(new_daily, "pw"), pending_path)

        done_path, done_plaintext = seeded[0]
        done_path.write_bytes(
            age_encrypt_bytes(done_plaintext, [new_daily.to_public(), recovery.to_public()])
        )

        from paramem.backup.rotation import RotationManifest

        manifest = RotationManifest(
            operation="rotate-daily",
            started_at="2026-04-24T12:00:00Z",
            new_daily_pub=str(new_daily.to_public()),
            files_pending=[str(seeded[1][0]), str(seeded[2][0])],
            files_done=[str(done_path)],
        )
        write_manifest_atomic(tmp_path / "rotation.manifest.json", manifest)

        # Resume.
        rc = rotate_daily.run(_default_args(data_dir))
        assert rc == 0

        # All files decrypt with the NEW daily now.
        final_daily = load_daily_identity(tmp_path / "daily_key.age", passphrase="pw")
        assert str(final_daily) == str(new_daily), (
            "finalisation must promote the pre-existing pending identity"
        )
        for path, original in seeded:
            assert age_decrypt_bytes(path.read_bytes(), [final_daily]) == original

    def test_resume_refuses_mismatched_operation(self, tmp_path, monkeypatch, capsys):
        daily, recovery = _setup_keys(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        _seed_age_files(data_dir, daily, recovery)

        # Pending key + manifest with WRONG operation.
        new_daily = mint_daily_identity()
        write_daily_key_file(
            wrap_daily_identity(new_daily, "pw"),
            tmp_path / "daily_key.age.pending",
        )
        from paramem.backup.rotation import RotationManifest

        manifest = RotationManifest(
            operation="rotate-recovery",  # mismatch
            started_at="2026-04-24T12:00:00Z",
            new_recovery_pub="age1xxx",
            files_pending=[],
            files_done=[],
        )
        write_manifest_atomic(tmp_path / "rotation.manifest.json", manifest)

        rc = rotate_daily.run(_default_args(data_dir))
        assert rc == 1
        assert "belongs to a 'rotate-recovery' run" in capsys.readouterr().err


class TestEmptyDataDir:
    def test_empty_store_is_not_an_error(self, tmp_path, monkeypatch, capsys):
        _setup_keys(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        rc = rotate_daily.run(_default_args(data_dir))
        assert rc == 0
        # No manifest, no pending file left behind.
        assert not (tmp_path / "rotation.manifest.json").exists()


class TestManifestInvariants:
    def test_manifest_rewritten_after_each_file(self, tmp_path, monkeypatch, capsys):
        """After a successful rotation the manifest is gone, but the intermediate
        states must have reflected progress. Capture them via a write hook."""
        daily, recovery = _setup_keys(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        seeded = _seed_age_files(data_dir, daily, recovery)

        from paramem.backup import rotation as rot_mod

        snapshots: list[tuple[int, int]] = []
        original_write = rot_mod.write_manifest_atomic

        def capturing_write(path, m):
            snapshots.append((len(m.files_pending), len(m.files_done)))
            return original_write(path, m)

        monkeypatch.setattr(rot_mod, "write_manifest_atomic", capturing_write)
        monkeypatch.setattr("paramem.cli.rotate_daily._rot.write_manifest_atomic", capturing_write)

        rc = rotate_daily.run(_default_args(data_dir))
        assert rc == 0

        # Fresh write: (3, 0). Then one per rotated file: (2, 1), (1, 2), (0, 3).
        assert snapshots[0] == (len(seeded), 0), f"first snapshot: {snapshots}"
        assert snapshots[-1] == (0, len(seeded)), f"last snapshot: {snapshots}"
        assert len(snapshots) == len(seeded) + 1
