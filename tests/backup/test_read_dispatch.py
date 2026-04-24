"""Tests for the age-aware ``read_maybe_encrypted`` dispatch + mode consistency.

Covers the Slice-D2 surface:

- ``read_maybe_encrypted`` sniffs both PMEM1 and age magics and routes to the
  appropriate decryptor (existing PMEM1 behaviour regresses; new age branch
  unwraps via the cached daily identity).
- Clear error when an age envelope is present but the daily identity is not
  loadable.
- ``ModeProbe`` / ``_probe_data_dir`` classify age-encrypted files as a third
  category alongside PMEM1 + plaintext.
- ``assert_mode_consistency`` permits age-only, PMEM1-only, age+PMEM1 mixed
  (transitional) with the right keys loaded; refuses any combination that
  pairs plaintext with an encryption magic, or an encryption magic whose
  key is not loaded.
"""

from __future__ import annotations

import os
from pathlib import Path

import pyrage
import pytest
from cryptography.fernet import Fernet
from pyrage import x25519

from paramem.backup.age_envelope import AGE_MAGIC, age_encrypt_bytes, is_age_envelope
from paramem.backup.encryption import (
    MASTER_KEY_ENV_VAR,
    PMEM1_MAGIC,
    _clear_cipher_cache,
    _probe_data_dir,
    assert_mode_consistency,
    encrypt_bytes,
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
from paramem.backup.types import FatalConfigError


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


def _setup_daily(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passphrase: str = "pw"):
    """Mint + wrap + write a daily identity; point the env + module default at it.

    Uses ``monkeypatch.setattr`` on the module attribute so the reader's
    late-lookup resolves to this test's key file without leaking state across
    tests. Returns the minted identity so callers can encrypt to its public
    recipient.
    """
    ident = mint_daily_identity()
    key_path = tmp_path / "daily_key.age"
    write_daily_key_file(wrap_daily_identity(ident, passphrase), key_path)
    monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, passphrase)
    monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path)
    return ident


def _setup_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> x25519.Recipient:
    """Mint + persist a recovery identity; point RECOVERY_PUB_PATH_DEFAULT at it."""
    recovery = x25519.Identity.generate()
    recovery_path = tmp_path / "recovery.pub"
    write_recovery_pub_file(recovery.to_public(), recovery_path)
    monkeypatch.setattr("paramem.backup.key_store.RECOVERY_PUB_PATH_DEFAULT", recovery_path)
    return recovery.to_public()


def _write_pmem1_bypass_dispatch(path: Path, plaintext: bytes) -> None:
    """Write a PMEM1 envelope directly, bypassing write_infra_bytes' routing.

    The production ``write_infra_bytes`` prefers age over PMEM1 when both keys
    are loaded. Tests that need a mixed on-disk state (e.g. to verify the
    mode-consistency classifier or a refuse case) must build the PMEM1 body
    directly instead of relying on the smart writer.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PMEM1_MAGIC + encrypt_bytes(plaintext))


class TestReadDispatchPlaintext:
    def test_plaintext_passthrough(self, tmp_path: Path) -> None:
        p = tmp_path / "plain.json"
        p.write_bytes(b'{"ok": true}')
        assert read_maybe_encrypted(p) == b'{"ok": true}'


class TestReadDispatchPmem1:
    def test_pmem1_round_trip_regression(self, tmp_path: Path) -> None:
        os.environ[MASTER_KEY_ENV_VAR] = _make_fernet_key()
        p = tmp_path / "payload.json"
        write_infra_bytes(p, b'{"k": "v"}')
        assert read_maybe_encrypted(p) == b'{"k": "v"}'


class TestReadDispatchAge:
    def test_age_envelope_decrypts_via_cached_daily(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ident = _setup_daily(tmp_path, monkeypatch)
        p = tmp_path / "payload.json"
        p.write_bytes(age_encrypt_bytes(b'{"k": "v"}', [ident.to_public()]))
        assert read_maybe_encrypted(p) == b'{"k": "v"}'

    def test_age_envelope_without_daily_passphrase_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ident = _setup_daily(tmp_path, monkeypatch)
        p = tmp_path / "payload.json"
        p.write_bytes(age_encrypt_bytes(b"x", [ident.to_public()]))

        # Drop the passphrase — loader raises RuntimeError, dispatch should
        # surface an actionable wrapped error naming the env var.
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
        _clear_daily_identity_cache()

        with pytest.raises(RuntimeError, match=DAILY_PASSPHRASE_ENV_VAR):
            read_maybe_encrypted(p)

    def test_age_envelope_tampered_raises_decrypt_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ident = _setup_daily(tmp_path, monkeypatch)
        p = tmp_path / "payload.json"
        ct = age_encrypt_bytes(b"payload" * 100, [ident.to_public()])
        # Zero out bytes past the age header.
        p.write_bytes(ct[:80] + bytes(len(ct) - 80))
        with pytest.raises(pyrage.DecryptError):
            read_maybe_encrypted(p)


class TestProbeDataDirClassification:
    def test_classifies_age_files_separately(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ident = _setup_daily(tmp_path, monkeypatch)
        data = tmp_path / "data"
        data.mkdir()
        # One plaintext, one PMEM1, one age — using the registry path which is
        # in infra_paths().
        os.environ[MASTER_KEY_ENV_VAR] = _make_fernet_key()
        pt_path = data / "registry.json"
        pt_path.write_bytes(b'{"plain": true}')
        probe_pt = _probe_data_dir(data)
        assert len(probe_pt.plaintext_paths) == 1
        assert not probe_pt.encrypted_paths
        assert not probe_pt.age_paths

        # Rewrite as PMEM1 directly — write_infra_bytes would prefer age
        # because _setup_daily loaded the daily identity in this test.
        _write_pmem1_bypass_dispatch(pt_path, b'{"pmem1": true}')
        probe_pm = _probe_data_dir(data)
        assert len(probe_pm.encrypted_paths) == 1
        assert not probe_pm.plaintext_paths
        assert not probe_pm.age_paths

        # Overwrite with an age envelope.
        pt_path.write_bytes(age_encrypt_bytes(b'{"age": true}', [ident.to_public()]))
        probe_age = _probe_data_dir(data)
        assert len(probe_age.age_paths) == 1
        assert not probe_age.plaintext_paths
        assert not probe_age.encrypted_paths


class TestAssertModeConsistencyAge:
    def test_age_only_with_daily_loadable_ok(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ident = _setup_daily(tmp_path, monkeypatch)
        data = tmp_path / "data"
        data.mkdir()
        (data / "registry.json").write_bytes(age_encrypt_bytes(b"{}", [ident.to_public()]))

        # Must not raise — age file + daily identity loadable.
        assert_mode_consistency(data, key_loaded=False, daily_identity_loadable=True)

    def test_age_without_daily_loadable_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ident = _setup_daily(tmp_path, monkeypatch)
        data = tmp_path / "data"
        data.mkdir()
        (data / "registry.json").write_bytes(age_encrypt_bytes(b"{}", [ident.to_public()]))

        with pytest.raises(FatalConfigError, match="daily identity is not loadable"):
            assert_mode_consistency(data, key_loaded=False, daily_identity_loadable=False)

    def test_mixed_pmem1_and_age_with_both_keys_ok_with_warn(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        ident = _setup_daily(tmp_path, monkeypatch)
        os.environ[MASTER_KEY_ENV_VAR] = _make_fernet_key()
        data = tmp_path / "data"
        data.mkdir()

        # PMEM1 file — write directly; write_infra_bytes would prefer age
        # now that the daily identity is loaded alongside the Fernet key.
        _write_pmem1_bypass_dispatch(data / "registry.json", b"{}")
        # age file directly.
        (data / "speaker_profiles.json").write_bytes(age_encrypt_bytes(b"{}", [ident.to_public()]))

        import logging

        with caplog.at_level(logging.WARNING, logger="paramem.backup.encryption"):
            assert_mode_consistency(data, key_loaded=True, daily_identity_loadable=True)
        # caplog + capfd union so the assertion survives either capture style
        # (some CI runners route logger output through stderr; others deliver
        # it via caplog's own handler).
        captured = capfd.readouterr()
        log_text = "\n".join(rec.message for rec in caplog.records)
        combined = captured.err + captured.out + log_text
        assert "Transitional" in combined, (
            f"expected 'Transitional' WARN, got caplog={log_text!r} stderr={captured.err!r}"
        )

    def test_mixed_age_and_plaintext_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ident = _setup_daily(tmp_path, monkeypatch)
        data = tmp_path / "data"
        data.mkdir()
        (data / "registry.json").write_bytes(age_encrypt_bytes(b"{}", [ident.to_public()]))
        (data / "speaker_profiles.json").write_bytes(b'{"plain": true}')

        with pytest.raises(FatalConfigError, match="Mixed encryption state"):
            assert_mode_consistency(data, key_loaded=False, daily_identity_loadable=True)

    def test_plaintext_with_daily_loaded_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _setup_daily(tmp_path, monkeypatch)
        data = tmp_path / "data"
        data.mkdir()
        (data / "registry.json").write_bytes(b'{"plain": true}')

        with pytest.raises(FatalConfigError, match="key is loaded but"):
            assert_mode_consistency(data, key_loaded=False, daily_identity_loadable=True)

    def test_mixed_pmem1_and_age_with_only_fernet_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial migration: operator has age envelopes on disk but no
        daily identity loaded (e.g. resumed from snapshot, passphrase env
        not yet set). Refuse because the age file cannot be decrypted."""
        ident = _setup_daily(tmp_path, monkeypatch)
        os.environ[MASTER_KEY_ENV_VAR] = _make_fernet_key()
        data = tmp_path / "data"
        data.mkdir()
        _write_pmem1_bypass_dispatch(data / "registry.json", b"{}")
        (data / "speaker_profiles.json").write_bytes(age_encrypt_bytes(b"{}", [ident.to_public()]))

        with pytest.raises(FatalConfigError, match="daily identity is not loadable"):
            assert_mode_consistency(data, key_loaded=True, daily_identity_loadable=False)

    def test_mixed_pmem1_and_age_with_only_daily_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Symmetric partial state: the operator retired the Fernet key
        before `migrate-to-age` finished the last PMEM1 tail. PMEM1 files
        are unreadable until the Fernet key is restored or the tail is
        re-encrypted."""
        ident = _setup_daily(tmp_path, monkeypatch)
        # Write PMEM1 directly — write_infra_bytes would prefer age now that
        # the daily identity is loaded.
        os.environ[MASTER_KEY_ENV_VAR] = _make_fernet_key()
        data = tmp_path / "data"
        data.mkdir()
        _write_pmem1_bypass_dispatch(data / "registry.json", b"{}")
        (data / "speaker_profiles.json").write_bytes(age_encrypt_bytes(b"{}", [ident.to_public()]))
        os.environ.pop(MASTER_KEY_ENV_VAR, None)
        _clear_cipher_cache()

        with pytest.raises(FatalConfigError, match=MASTER_KEY_ENV_VAR):
            assert_mode_consistency(data, key_loaded=False, daily_identity_loadable=True)


class TestAssertModeConsistencyRegression:
    """Pre-D2 Fernet-only behavior must continue to hold when daily_identity_loadable is False."""

    def test_empty_store_ok(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        data.mkdir()
        assert_mode_consistency(data, key_loaded=False)
        assert_mode_consistency(data, key_loaded=True)

    def test_pmem1_only_with_fernet_key_ok(self, tmp_path: Path) -> None:
        os.environ[MASTER_KEY_ENV_VAR] = _make_fernet_key()
        data = tmp_path / "data"
        data.mkdir()
        write_infra_bytes(data / "registry.json", b"{}")
        assert_mode_consistency(data, key_loaded=True)

    def test_pmem1_without_key_refuses(self, tmp_path: Path) -> None:
        os.environ[MASTER_KEY_ENV_VAR] = _make_fernet_key()
        data = tmp_path / "data"
        data.mkdir()
        write_infra_bytes(data / "registry.json", b"{}")
        os.environ.pop(MASTER_KEY_ENV_VAR, None)
        _clear_cipher_cache()
        with pytest.raises(FatalConfigError, match=MASTER_KEY_ENV_VAR):
            assert_mode_consistency(data, key_loaded=False)

    def test_plaintext_only_no_keys_ok(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        data.mkdir()
        (data / "registry.json").write_bytes(b'{"plain": true}')
        assert_mode_consistency(data, key_loaded=False)


class TestWriteInfraBytesFlip:
    """write_infra_bytes chooses envelope format by loaded-key posture.

    Priority: age-multi > age-single > PMEM1 > plaintext.
    """

    def test_writes_age_multi_recipient_when_both_keys_loaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _setup_daily(tmp_path, monkeypatch)
        _setup_recovery(tmp_path, monkeypatch)

        target = tmp_path / "registry.json"
        write_infra_bytes(target, b'{"payload": 1}')
        assert is_age_envelope(target)
        assert not is_pmem1_envelope(target)
        assert read_maybe_encrypted(target) == b'{"payload": 1}'

    def test_writes_age_single_recipient_when_recovery_pub_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _setup_daily(tmp_path, monkeypatch)
        # Deliberately do NOT set up recovery — point the default at a missing path.
        monkeypatch.setattr(
            "paramem.backup.key_store.RECOVERY_PUB_PATH_DEFAULT",
            tmp_path / "nope.pub",
        )

        target = tmp_path / "registry.json"
        write_infra_bytes(target, b'{"daily-only": true}')
        assert is_age_envelope(target), "daily loadable → age write even without recovery"
        assert read_maybe_encrypted(target) == b'{"daily-only": true}'

    def test_writes_pmem1_when_only_fernet_loaded(self, tmp_path: Path) -> None:
        os.environ[MASTER_KEY_ENV_VAR] = _make_fernet_key()

        target = tmp_path / "registry.json"
        write_infra_bytes(target, b'{"pmem1": 1}')
        assert is_pmem1_envelope(target)
        assert not is_age_envelope(target)
        assert target.read_bytes().startswith(PMEM1_MAGIC)
        assert read_maybe_encrypted(target) == b'{"pmem1": 1}'

    def test_writes_plaintext_when_no_keys_loaded(self, tmp_path: Path) -> None:
        target = tmp_path / "registry.json"
        write_infra_bytes(target, b'{"plain": true}')
        assert target.read_bytes() == b'{"plain": true}'
        assert not is_age_envelope(target)
        assert not is_pmem1_envelope(target)

    def test_age_priority_wins_over_fernet_when_both_loaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Transitional state during migration — both keys loaded. New writes
        must land as age so the migration converges."""
        _setup_daily(tmp_path, monkeypatch)
        _setup_recovery(tmp_path, monkeypatch)
        os.environ[MASTER_KEY_ENV_VAR] = _make_fernet_key()

        target = tmp_path / "registry.json"
        write_infra_bytes(target, b'{"both": 1}')
        assert is_age_envelope(target), "age takes precedence over Fernet"
        assert not is_pmem1_envelope(target)

    def test_wrong_passphrase_gracefully_falls_back_to_pmem1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Loadable probe passes (file + env present) but the env passphrase
        is wrong → unwrap fails. Writer must not crash; fall through to the
        next available format (PMEM1 here)."""
        _setup_daily(tmp_path, monkeypatch, passphrase="correct")
        # Override the env with a wrong passphrase; daily_identity_loadable still
        # returns True because it only probes presence, not correctness.
        monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "wrong-passphrase")
        _clear_daily_identity_cache()
        os.environ[MASTER_KEY_ENV_VAR] = _make_fernet_key()

        target = tmp_path / "registry.json"
        write_infra_bytes(target, b'{"fallback": 1}')
        assert is_pmem1_envelope(target), (
            "failed unwrap must gracefully fall through to the Fernet path"
        )

    def test_multi_recipient_envelope_decryptable_by_either_identity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Operational guarantee: a file written under [daily, recovery] must
        remain readable via the recovery identity on hardware replacement."""
        from pyrage import x25519 as _x25519

        from paramem.backup.age_envelope import age_decrypt_bytes

        _setup_daily(tmp_path, monkeypatch)
        # We need the recovery *identity*, not just the recipient, to simulate
        # the restore-with-recovery-key flow. Generate here and wire both.
        recovery = _x25519.Identity.generate()
        recovery_path = tmp_path / "recovery.pub"
        write_recovery_pub_file(recovery.to_public(), recovery_path)
        monkeypatch.setattr("paramem.backup.key_store.RECOVERY_PUB_PATH_DEFAULT", recovery_path)

        target = tmp_path / "registry.json"
        payload = b'{"critical": "data"}'
        write_infra_bytes(target, payload)

        assert is_age_envelope(target)
        # Daily identity decrypts (via the cached loader / universal reader).
        assert read_maybe_encrypted(target) == payload
        # Recovery identity decrypts the raw envelope bytes directly — this is
        # the hardware-replacement path (no daily key on the new device).
        assert age_decrypt_bytes(target.read_bytes(), [recovery]) == payload


class TestWriterPriorityEdgeCases:
    def test_env_set_but_daily_file_missing_falls_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """daily_identity_loadable probe is False when the file is missing
        even if the env is set. Writer must skip the age branch cleanly."""
        monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "pw")
        monkeypatch.setattr(
            "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT",
            tmp_path / "absent.age",
        )
        os.environ[MASTER_KEY_ENV_VAR] = _make_fernet_key()

        target = tmp_path / "registry.json"
        write_infra_bytes(target, b'{"x": 1}')
        assert is_pmem1_envelope(target)
        assert not is_age_envelope(target)

    def test_round_trip_under_every_posture(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All four postures produce a file that round-trips through
        read_maybe_encrypted."""
        # 1. plaintext
        t1 = tmp_path / "plain.json"
        write_infra_bytes(t1, b"plain")
        assert read_maybe_encrypted(t1) == b"plain"

        # 2. PMEM1
        os.environ[MASTER_KEY_ENV_VAR] = _make_fernet_key()
        t2 = tmp_path / "pmem1.json"
        write_infra_bytes(t2, b"pmem1")
        assert read_maybe_encrypted(t2) == b"pmem1"

        # 3. age single (daily only)
        _setup_daily(tmp_path, monkeypatch)
        t3 = tmp_path / "age-single.json"
        write_infra_bytes(t3, b"age-single")
        assert read_maybe_encrypted(t3) == b"age-single"

        # 4. age multi
        _setup_recovery(tmp_path, monkeypatch)
        t4 = tmp_path / "age-multi.json"
        write_infra_bytes(t4, b"age-multi")
        assert read_maybe_encrypted(t4) == b"age-multi"

        # Both envelopes live under the tmp_path; reader dispatches by magic.
        assert t2.read_bytes().startswith(PMEM1_MAGIC)
        assert t3.read_bytes().startswith(AGE_MAGIC)
        assert t4.read_bytes().startswith(AGE_MAGIC)
