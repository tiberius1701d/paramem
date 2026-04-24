"""Tests for the master-key + infrastructure-envelope primitives.

Covers:

1. ``master_key_env_value`` / ``master_key_loaded`` — env probing.
2. ``current_key_fingerprint`` — stable 16-hex string derived from the key.
3. ``write_infra_bytes`` / ``read_maybe_encrypted`` — PMEM1 envelope.
4. ``assert_mode_consistency`` — SECURITY.md §4 four-case startup refuse.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from cryptography.fernet import Fernet, InvalidToken

from paramem.backup.encryption import (
    MASTER_KEY_ENV_VAR,
    PMEM1_MAGIC,
    _clear_cipher_cache,
    assert_mode_consistency,
    current_key_fingerprint,
    is_pmem1_envelope,
    master_key_env_value,
    master_key_loaded,
    read_maybe_encrypted,
    write_infra_bytes,
)
from paramem.backup.types import FatalConfigError


def _make_key() -> str:
    return Fernet.generate_key().decode()


def _clean_env() -> None:
    os.environ.pop(MASTER_KEY_ENV_VAR, None)
    _clear_cipher_cache()


@pytest.fixture(autouse=True)
def _env_isolation():
    """Isolate key-related env vars and cipher cache per test."""
    _clean_env()
    yield
    _clean_env()


# ---------------------------------------------------------------------------
# master_key_env_value / master_key_loaded
# ---------------------------------------------------------------------------


class TestMasterKeyEnv:
    def test_returns_value_when_set(self):
        key = _make_key()
        os.environ[MASTER_KEY_ENV_VAR] = key
        assert master_key_env_value() == key

    def test_returns_none_when_unset(self):
        assert master_key_env_value() is None
        assert master_key_loaded() is False

    def test_master_key_loaded_true_when_set(self):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        assert master_key_loaded() is True

    def test_empty_value_treated_as_unset(self):
        """Empty-string env var is indistinguishable from unset — important so
        an operator who wipes the value in .env doesn't leave a broken cipher
        build attempt (Fernet would crash on empty bytes)."""
        os.environ[MASTER_KEY_ENV_VAR] = ""
        assert master_key_env_value() is None
        assert master_key_loaded() is False


# ---------------------------------------------------------------------------
# current_key_fingerprint
# ---------------------------------------------------------------------------


class TestCurrentKeyFingerprint:
    def test_none_when_no_key(self):
        assert current_key_fingerprint() is None

    def test_stable_16_hex(self):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        fp = current_key_fingerprint()
        assert fp is not None
        assert len(fp) == 16
        assert all(c in "0123456789abcdef" for c in fp)
        # Stable: repeated calls with same key yield same fingerprint.
        assert fp == current_key_fingerprint()

    def test_different_keys_yield_different_fingerprints(self):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        fp_a = current_key_fingerprint()
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        fp_b = current_key_fingerprint()
        assert fp_a != fp_b


# ---------------------------------------------------------------------------
# PMEM1 envelope: write_infra_bytes / read_maybe_encrypted
# ---------------------------------------------------------------------------


class TestPmem1Envelope:
    def test_plaintext_roundtrip_when_no_key(self, tmp_path):
        target = tmp_path / "infra.json"
        payload = b'{"hello": "world"}'
        write_infra_bytes(target, payload)
        # No key → on-disk is plaintext.
        assert target.read_bytes() == payload
        assert not is_pmem1_envelope(target)
        assert read_maybe_encrypted(target) == payload

    def test_encrypted_roundtrip_when_key_set(self, tmp_path):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        target = tmp_path / "infra.json"
        payload = b'{"secret": "value"}'
        write_infra_bytes(target, payload)
        # On-disk starts with PMEM1 magic.
        on_disk = target.read_bytes()
        assert on_disk.startswith(PMEM1_MAGIC)
        assert on_disk != payload
        assert is_pmem1_envelope(target)
        # Read round-trips to plaintext.
        assert read_maybe_encrypted(target) == payload

    def test_read_rejects_ciphertext_when_key_absent(self, tmp_path):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        target = tmp_path / "infra.json"
        write_infra_bytes(target, b"payload")
        _clean_env()  # key gone; ciphertext remains
        with pytest.raises(RuntimeError, match=MASTER_KEY_ENV_VAR):
            read_maybe_encrypted(target)

    def test_read_rejects_wrong_key(self, tmp_path):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        target = tmp_path / "infra.json"
        write_infra_bytes(target, b"payload")
        # Swap key.
        _clear_cipher_cache()
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        with pytest.raises(InvalidToken):
            read_maybe_encrypted(target)

    def test_write_is_atomic(self, tmp_path):
        """Temp file is cleaned up; no partial write visible."""
        target = tmp_path / "infra.json"
        write_infra_bytes(target, b"payload")
        assert target.exists()
        assert not (tmp_path / "infra.json.tmp").exists()


# ---------------------------------------------------------------------------
# assert_mode_consistency — SECURITY.md §4 four-case matrix
# ---------------------------------------------------------------------------


class TestAssertModeConsistency:
    def _write_plaintext(self, path: Path, body: bytes = b'{"ok":true}') -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)

    def _write_encrypted(self, tmp_path: Path, path: Path) -> None:
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        path.parent.mkdir(parents=True, exist_ok=True)
        write_infra_bytes(path, b'{"ok":true}')
        _clean_env()

    def test_case_set_encrypted_is_ok(self, tmp_path):
        """Key set + on-disk encrypted → proceed."""
        self._write_encrypted(tmp_path, tmp_path / "graph.json")
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()  # any key — sniff only
        # Must not raise.
        assert_mode_consistency(tmp_path, key_loaded=True)

    def test_case_unset_plaintext_is_ok(self, tmp_path):
        """Key unset + on-disk plaintext → Security OFF, proceed."""
        self._write_plaintext(tmp_path / "graph.json")
        assert_mode_consistency(tmp_path, key_loaded=False)

    def test_case_set_plaintext_refuses(self, tmp_path):
        """Key set + plaintext on disk → refuse, point at encrypt-infra."""
        self._write_plaintext(tmp_path / "graph.json")
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        with pytest.raises(FatalConfigError, match="encrypt-infra"):
            assert_mode_consistency(tmp_path, key_loaded=True)

    def test_case_unset_encrypted_refuses(self, tmp_path):
        """Key unset + ciphertext on disk → refuse, point at decrypt-infra."""
        self._write_encrypted(tmp_path, tmp_path / "graph.json")
        with pytest.raises(FatalConfigError, match="decrypt-infra"):
            assert_mode_consistency(tmp_path, key_loaded=False)

    def test_mixed_on_disk_refuses_regardless_of_key(self, tmp_path):
        """Mixed state → refuse regardless of key presence."""
        self._write_plaintext(tmp_path / "graph.json")
        self._write_encrypted(tmp_path, tmp_path / "registry.json")
        # Without key.
        with pytest.raises(FatalConfigError, match="Mixed encryption state"):
            assert_mode_consistency(tmp_path, key_loaded=False)
        # With key.
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        with pytest.raises(FatalConfigError, match="Mixed encryption state"):
            assert_mode_consistency(tmp_path, key_loaded=True)

    def test_empty_data_dir_is_consistent_with_either_mode(self, tmp_path):
        """Fresh deployment with no infra files → OK in both modes."""
        assert_mode_consistency(tmp_path, key_loaded=False)
        assert_mode_consistency(tmp_path, key_loaded=True)

    def test_missing_data_dir_is_consistent(self, tmp_path):
        """Non-existent path probe is neutral."""
        ghost = tmp_path / "does-not-exist"
        assert_mode_consistency(ghost, key_loaded=False)
        assert_mode_consistency(ghost, key_loaded=True)

    def test_carve_out_plaintext_state_files_ignored(self, tmp_path):
        """state/trial.json and state/backup.json are plaintext-by-design and
        must not trigger a mismatch when the rest of the store is encrypted."""
        self._write_encrypted(tmp_path, tmp_path / "graph.json")
        # Write carve-out files as plaintext.
        self._write_plaintext(tmp_path / "state" / "trial.json")
        self._write_plaintext(tmp_path / "state" / "backup.json")
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        # Must not raise — the carve-out files are not in the probed set.
        assert_mode_consistency(tmp_path, key_loaded=True)


# ---------------------------------------------------------------------------
# Integration: backup.write populates ArtifactMeta.key_fingerprint via
# current_key_fingerprint().  Locks the migration from the prior inline
# fingerprint_key_bytes(key_env.encode()) call so a future refactor can't
# bypass the helper.
# ---------------------------------------------------------------------------


class TestBackupWriteUsesCurrentKeyFingerprint:
    def test_encrypted_slot_fingerprint_matches_current_key(self, tmp_path):
        """Writing an encrypted artifact stamps the sidecar with the value
        returned by ``current_key_fingerprint()`` — not a different helper."""
        from paramem.backup.backup import write as backup_write
        from paramem.backup.meta import read_meta
        from paramem.backup.types import ArtifactKind

        os.environ[MASTER_KEY_ENV_VAR] = _make_key()

        expected_fp = current_key_fingerprint()
        assert expected_fp is not None

        slot_dir = backup_write(
            ArtifactKind.CONFIG,
            b"model: mistral\n",
            meta_fields={"tier": "scheduled"},
            base_dir=tmp_path / "config",
        )
        meta = read_meta(slot_dir)

        assert meta.encrypted is True
        assert meta.key_fingerprint == expected_fp

    def test_plaintext_slot_has_no_fingerprint(self, tmp_path):
        """No key set → plaintext slot with key_fingerprint=None."""
        from paramem.backup.backup import write as backup_write
        from paramem.backup.meta import read_meta
        from paramem.backup.types import ArtifactKind

        slot_dir = backup_write(
            ArtifactKind.CONFIG,
            b"model: mistral\n",
            meta_fields={"tier": "scheduled"},
            base_dir=tmp_path / "config",
        )
        meta = read_meta(slot_dir)

        assert meta.encrypted is False
        assert meta.key_fingerprint is None


class TestEnvelopeEncryptBytesHelper:
    """``envelope_encrypt_bytes`` + ``envelope_decrypt_bytes`` are the primitives
    that power both ``write_infra_bytes`` (on-disk writes) and the backup
    subsystem (bytes-in-hand). Verifying them in isolation pins the format
    selection logic independent of any atomic-write or sidecar concerns.
    """

    def test_plaintext_when_no_keys_loaded(self):
        from paramem.backup.encryption import envelope_encrypt_bytes

        result = envelope_encrypt_bytes(b"payload")
        assert result == b"payload"

    def test_pmem1_when_only_fernet_loaded(self):
        from paramem.backup.encryption import envelope_encrypt_bytes

        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        result = envelope_encrypt_bytes(b"payload")
        assert result.startswith(PMEM1_MAGIC)
        assert result != b"payload"

    def test_age_when_daily_loaded(self, tmp_path, monkeypatch):
        from paramem.backup.age_envelope import AGE_MAGIC
        from paramem.backup.encryption import envelope_encrypt_bytes
        from paramem.backup.key_store import (
            _clear_daily_identity_cache,
            mint_daily_identity,
            wrap_daily_identity,
            write_daily_key_file,
        )

        daily = mint_daily_identity()
        key_path = tmp_path / "daily_key.age"
        write_daily_key_file(wrap_daily_identity(daily, "pw"), key_path)
        monkeypatch.setenv("PARAMEM_DAILY_PASSPHRASE", "pw")
        monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path)
        _clear_daily_identity_cache()

        result = envelope_encrypt_bytes(b"payload")
        assert result.startswith(AGE_MAGIC), f"expected age magic; got {result[:30]!r}"


class TestEnvelopeDecryptBytesHelper:
    def test_dispatches_age_via_loaded_daily(self, tmp_path, monkeypatch):
        from paramem.backup.age_envelope import age_encrypt_bytes
        from paramem.backup.encryption import envelope_decrypt_bytes
        from paramem.backup.key_store import (
            _clear_daily_identity_cache,
            mint_daily_identity,
            wrap_daily_identity,
            write_daily_key_file,
        )

        daily = mint_daily_identity()
        key_path = tmp_path / "daily_key.age"
        write_daily_key_file(wrap_daily_identity(daily, "pw"), key_path)
        monkeypatch.setenv("PARAMEM_DAILY_PASSPHRASE", "pw")
        monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path)
        _clear_daily_identity_cache()

        envelope = age_encrypt_bytes(b"payload", [daily.to_public()])
        assert envelope_decrypt_bytes(envelope) == b"payload"

    def test_dispatches_pmem1_envelope(self):
        from paramem.backup.encryption import envelope_decrypt_bytes

        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        from paramem.backup.encryption import encrypt_bytes as _encrypt_bytes

        envelope = PMEM1_MAGIC + _encrypt_bytes(b"payload")
        assert envelope_decrypt_bytes(envelope) == b"payload"

    def test_dispatches_bare_fernet_token_legacy_backup_format(self):
        """Pre-envelope backups stored raw Fernet tokens (no PMEM1 magic).
        ``envelope_decrypt_bytes`` recognises the non-magic case as legacy
        bare Fernet so existing backups still restore after the D3 flip."""
        from paramem.backup.encryption import encrypt_bytes as _encrypt_bytes
        from paramem.backup.encryption import envelope_decrypt_bytes

        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        bare_token = _encrypt_bytes(b"legacy payload")
        # Sanity: the token does not carry either magic.
        assert not bare_token.startswith(PMEM1_MAGIC)
        assert not bare_token.startswith(b"age-encryption.org")

        assert envelope_decrypt_bytes(bare_token) == b"legacy payload"

    def test_age_without_daily_raises_actionable_runtime_error(self, tmp_path, monkeypatch):
        from pyrage import x25519

        from paramem.backup.age_envelope import age_encrypt_bytes
        from paramem.backup.encryption import envelope_decrypt_bytes
        from paramem.backup.key_store import _clear_daily_identity_cache

        # Point the default at a non-existent path; daily_identity_loadable
        # will return False, and envelope_decrypt_bytes should raise with a
        # message naming the env var + expected path.
        monkeypatch.setattr(
            "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT",
            tmp_path / "absent.age",
        )
        _clear_daily_identity_cache()

        envelope = age_encrypt_bytes(b"x", [x25519.Identity.generate().to_public()])
        with pytest.raises(RuntimeError, match="PARAMEM_DAILY_PASSPHRASE"):
            envelope_decrypt_bytes(envelope)


class TestBackupWritePathProducesMagicWrappedEnvelope:
    """Regression guard: the backup subsystem's encrypted payload must be a
    magic-wrapped envelope (PMEM1 or age), not a bare Fernet token. Prior to
    the D3 fix, backup.py called ``encrypt_bytes`` directly and produced
    naked Fernet tokens that diverged from the rest of the infra store."""

    def test_fernet_posture_writes_pmem1_magic(self, tmp_path):
        """key loaded → encrypted artifact uses PMEM1 magic envelope."""
        from paramem.backup import backup as backup_mod
        from paramem.backup.meta import read_meta
        from paramem.backup.types import ArtifactKind

        os.environ[MASTER_KEY_ENV_VAR] = _make_key()

        slot_dir = backup_mod.write(
            ArtifactKind.CONFIG,
            b"payload",
            meta_fields={"tier": "manual"},
            base_dir=tmp_path / "config",
        )
        meta = read_meta(slot_dir)
        artifact = next(p for p in slot_dir.iterdir() if not p.name.endswith(".meta.json"))
        head = artifact.read_bytes()[: len(PMEM1_MAGIC)]
        assert head == PMEM1_MAGIC, f"expected PMEM1 magic, got {head!r}"
        assert meta.encrypted is True
        assert meta.key_fingerprint is not None, "Fernet backup must record fingerprint"

    def test_age_posture_writes_age_magic_and_null_fingerprint(self, tmp_path, monkeypatch):
        """age daily loaded → encrypted artifact uses age magic; fingerprint is None."""
        from pyrage import x25519

        from paramem.backup import backup as backup_mod
        from paramem.backup.age_envelope import AGE_MAGIC
        from paramem.backup.key_store import (
            _clear_daily_identity_cache,
            mint_daily_identity,
            wrap_daily_identity,
            write_daily_key_file,
            write_recovery_pub_file,
        )
        from paramem.backup.meta import read_meta
        from paramem.backup.types import ArtifactKind

        # Load the age identities.
        daily = mint_daily_identity()
        recovery = x25519.Identity.generate()
        key_path = tmp_path / "daily_key.age"
        recovery_path = tmp_path / "recovery.pub"
        write_daily_key_file(wrap_daily_identity(daily, "pw"), key_path)
        write_recovery_pub_file(recovery.to_public(), recovery_path)
        monkeypatch.setenv("PARAMEM_DAILY_PASSPHRASE", "pw")
        monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path)
        monkeypatch.setattr("paramem.backup.key_store.RECOVERY_PUB_PATH_DEFAULT", recovery_path)
        _clear_daily_identity_cache()
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()  # Fernet also set (transitional)

        slot_dir = backup_mod.write(
            ArtifactKind.CONFIG,
            b"payload",
            meta_fields={"tier": "manual"},
            base_dir=tmp_path / "config",
        )
        meta = read_meta(slot_dir)
        artifact = next(p for p in slot_dir.iterdir() if not p.name.endswith(".meta.json"))
        head = artifact.read_bytes()[: len(AGE_MAGIC)]
        assert head == AGE_MAGIC, f"expected age magic, got {head!r}"
        assert meta.encrypted is True
        assert meta.key_fingerprint is None, (
            "age backup must record null fingerprint (Fernet-style check does not apply); "
            "the None-path in the restore endpoint's fingerprint check is what carries the "
            "operator through"
        )

        # Round-trip: backup_read must return the original plaintext via the
        # age → envelope_decrypt_bytes path.
        plaintext, _ = backup_mod.read(slot_dir)
        assert plaintext == b"payload"


class TestLegacyFernetBackupStillRestores:
    """Old backups written with the pre-envelope ``encrypt_bytes`` call (bare
    Fernet token, no magic) must continue to round-trip through backup.read
    after the envelope refactor."""

    def test_bare_fernet_round_trip_via_read(self, tmp_path):
        """Hand-built pre-envelope slot (bare Fernet token) still round-trips."""
        from paramem.backup import backup as backup_mod
        from paramem.backup.encryption import encrypt_bytes
        from paramem.backup.meta import ArtifactMeta, write_meta
        from paramem.backup.types import ArtifactKind

        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        slot_dir = tmp_path / "config" / "20260424-12000000"
        slot_dir.mkdir(parents=True)

        # Hand-build a pre-envelope backup: bare Fernet token + sidecar.
        payload = b"legacy payload"
        bare = encrypt_bytes(payload)
        artifact_path = slot_dir / "config-20260424-12000000.bin.enc"
        artifact_path.write_bytes(bare)

        from paramem.backup.encryption import current_key_fingerprint
        from paramem.backup.hashing import content_sha256_bytes

        meta = ArtifactMeta(
            schema_version=1,
            kind=ArtifactKind.CONFIG,
            timestamp="20260424-12000000",
            content_sha256=content_sha256_bytes(bare),
            size_bytes=len(bare),
            encrypted=True,
            key_fingerprint=current_key_fingerprint(),
            tier="manual",
            label=None,
            pre_trial_hash=None,
        )
        write_meta(slot_dir, meta)

        plaintext, meta_read = backup_mod.read(slot_dir)
        assert plaintext == payload
        assert meta_read.encrypted is True
