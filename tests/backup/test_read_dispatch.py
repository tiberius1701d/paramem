"""Tests for the age-aware ``read_maybe_encrypted`` dispatch + mode consistency.

Covers the Slice-D2 surface:

- ``read_maybe_encrypted`` sniffs the age magic and routes to the appropriate
  decryptor (age branch unwraps via the cached daily identity).
- Clear error when an age envelope is present but the daily identity is not
  loadable.
- ``ModeProbe`` / ``_probe_data_dir`` classify age-encrypted files alongside
  plaintext.
- ``assert_mode_consistency`` permits age-only (daily loaded) and plaintext-only
  (no key) postures; refuses mixed and key-loaded-but-plaintext states.
"""

from __future__ import annotations

import os
from pathlib import Path

import pyrage
import pytest
from pyrage import x25519

from paramem.backup.age_envelope import AGE_MAGIC, age_encrypt_bytes, is_age_envelope
from paramem.backup.encryption import (
    _probe_data_dir,
    assert_mode_consistency,
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


@pytest.fixture(autouse=True)
def _env_and_cache_isolation():
    os.environ.pop(DAILY_PASSPHRASE_ENV_VAR, None)
    _clear_daily_identity_cache()
    yield
    os.environ.pop(DAILY_PASSPHRASE_ENV_VAR, None)
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


class TestReadDispatchPlaintext:
    def test_plaintext_passthrough(self, tmp_path: Path) -> None:
        """Plain bytes (no magic) pass through read_maybe_encrypted unchanged."""
        p = tmp_path / "plain.json"
        p.write_bytes(b'{"ok": true}')
        assert read_maybe_encrypted(p) == b'{"ok": true}'


class TestReadDispatchAge:
    def test_age_envelope_decrypts_via_cached_daily(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Age envelope written under a daily identity decrypts transparently."""
        ident = _setup_daily(tmp_path, monkeypatch)
        p = tmp_path / "payload.json"
        p.write_bytes(age_encrypt_bytes(b'{"k": "v"}', [ident.to_public()]))
        assert read_maybe_encrypted(p) == b'{"k": "v"}'

    def test_age_envelope_without_daily_passphrase_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Age envelope present, but passphrase env unset → RuntimeError naming the var."""
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
        """Tampered age envelope → pyrage.DecryptError."""
        ident = _setup_daily(tmp_path, monkeypatch)
        p = tmp_path / "payload.json"
        ct = age_encrypt_bytes(b"payload" * 100, [ident.to_public()])
        # Zero out bytes past the age header.
        p.write_bytes(ct[:80] + bytes(len(ct) - 80))
        with pytest.raises(pyrage.DecryptError):
            read_maybe_encrypted(p)


class TestProbeDataDirClassification:
    def test_classifies_age_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """_probe_data_dir correctly classifies age vs plaintext files."""
        ident = _setup_daily(tmp_path, monkeypatch)
        data = tmp_path / "data"
        data.mkdir()

        # Plaintext infra file — uses registry.json which is in infra_paths().
        pt_path = data / "registry.json"
        pt_path.write_bytes(b'{"plain": true}')
        probe_pt = _probe_data_dir(data)
        assert len(probe_pt.plaintext_paths) == 1
        assert not probe_pt.age_paths

        # Overwrite with an age envelope.
        pt_path.write_bytes(age_encrypt_bytes(b'{"age": true}', [ident.to_public()]))
        probe_age = _probe_data_dir(data)
        assert len(probe_age.age_paths) == 1
        assert not probe_age.plaintext_paths

    def test_empty_data_dir_returns_empty_probe(self, tmp_path: Path) -> None:
        """Empty data dir → both lists empty."""
        data = tmp_path / "data"
        data.mkdir()
        probe = _probe_data_dir(data)
        assert not probe.age_paths
        assert not probe.plaintext_paths

    def test_missing_data_dir_returns_empty_probe(self, tmp_path: Path) -> None:
        """Nonexistent data dir → empty probe (not an error)."""
        probe = _probe_data_dir(tmp_path / "does-not-exist")
        assert not probe.age_paths
        assert not probe.plaintext_paths


class TestAssertModeConsistency:
    def test_age_only_with_daily_loadable_ok(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """age file + daily identity loadable → no error."""
        ident = _setup_daily(tmp_path, monkeypatch)
        data = tmp_path / "data"
        data.mkdir()
        (data / "registry.json").write_bytes(age_encrypt_bytes(b"{}", [ident.to_public()]))

        # Must not raise — age file + daily identity loadable.
        assert_mode_consistency(data, daily_identity_loadable=True)

    def test_age_without_daily_loadable_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """age file on disk but daily identity not loadable → FatalConfigError."""
        ident = _setup_daily(tmp_path, monkeypatch)
        data = tmp_path / "data"
        data.mkdir()
        (data / "registry.json").write_bytes(age_encrypt_bytes(b"{}", [ident.to_public()]))

        with pytest.raises(FatalConfigError, match="daily identity is not loadable"):
            assert_mode_consistency(data, daily_identity_loadable=False)

    def test_mixed_age_and_plaintext_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mixed age + plaintext → FatalConfigError(Mixed encryption state)."""
        ident = _setup_daily(tmp_path, monkeypatch)
        data = tmp_path / "data"
        data.mkdir()
        (data / "registry.json").write_bytes(age_encrypt_bytes(b"{}", [ident.to_public()]))
        (data / "speaker_profiles.json").write_bytes(b'{"plain": true}')

        with pytest.raises(FatalConfigError, match="Mixed encryption state"):
            assert_mode_consistency(data, daily_identity_loadable=True)

    def test_plaintext_with_daily_loaded_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plaintext files + daily identity loaded → FatalConfigError."""
        _setup_daily(tmp_path, monkeypatch)
        data = tmp_path / "data"
        data.mkdir()
        (data / "registry.json").write_bytes(b'{"plain": true}')

        with pytest.raises(FatalConfigError, match="daily identity is loaded but"):
            assert_mode_consistency(data, daily_identity_loadable=True)

    def test_empty_store_ok_no_key(self, tmp_path: Path) -> None:
        """Empty data dir passes regardless of key state."""
        data = tmp_path / "data"
        data.mkdir()
        assert_mode_consistency(data, daily_identity_loadable=False)

    def test_empty_store_ok_with_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty data dir + daily loadable → no error (nothing to migrate)."""
        _setup_daily(tmp_path, monkeypatch)
        data = tmp_path / "data"
        data.mkdir()
        assert_mode_consistency(data, daily_identity_loadable=True)

    def test_plaintext_only_no_key_ok(self, tmp_path: Path) -> None:
        """Plaintext files + no key loaded → Security OFF, passes."""
        data = tmp_path / "data"
        data.mkdir()
        (data / "registry.json").write_bytes(b'{"plain": true}')
        assert_mode_consistency(data, daily_identity_loadable=False)


class TestWriteInfraBytesFlip:
    """write_infra_bytes chooses envelope format by loaded-key posture.

    Priority: age-multi > age-single > plaintext.
    """

    def test_writes_age_multi_recipient_when_both_keys_loaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Daily + recovery loaded → age multi-recipient envelope."""
        _setup_daily(tmp_path, monkeypatch)
        _setup_recovery(tmp_path, monkeypatch)

        target = tmp_path / "registry.json"
        write_infra_bytes(target, b'{"payload": 1}')
        assert is_age_envelope(target)
        assert read_maybe_encrypted(target) == b'{"payload": 1}'

    def test_writes_age_single_recipient_when_recovery_pub_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Daily loaded but recovery.pub missing → age single-recipient envelope."""
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

    def test_writes_plaintext_when_no_keys_loaded(self, tmp_path: Path) -> None:
        """No key loaded → plaintext on disk."""
        target = tmp_path / "registry.json"
        write_infra_bytes(target, b'{"plain": true}')
        assert target.read_bytes() == b'{"plain": true}'
        assert not is_age_envelope(target)

    def test_multi_recipient_envelope_decryptable_by_either_identity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Multi-recipient envelope must be readable via the recovery identity."""
        from pyrage import x25519 as _x25519

        from paramem.backup.age_envelope import age_decrypt_bytes

        _setup_daily(tmp_path, monkeypatch)
        # We need the recovery *identity*, not just the recipient, to simulate
        # the restore-with-recovery-key flow.
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
        even if the env is set. Writer must write plaintext cleanly."""
        monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "pw")
        monkeypatch.setattr(
            "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT",
            tmp_path / "absent.age",
        )

        target = tmp_path / "registry.json"
        write_infra_bytes(target, b'{"x": 1}')
        assert target.read_bytes() == b'{"x": 1}'
        assert not is_age_envelope(target)

    def test_round_trip_under_every_posture(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All postures produce a file that round-trips through read_maybe_encrypted."""
        # 1. plaintext
        t1 = tmp_path / "plain.json"
        write_infra_bytes(t1, b"plain")
        assert read_maybe_encrypted(t1) == b"plain"

        # 2. age single (daily only)
        _setup_daily(tmp_path, monkeypatch)
        t2 = tmp_path / "age-single.json"
        write_infra_bytes(t2, b"age-single")
        assert read_maybe_encrypted(t2) == b"age-single"

        # 3. age multi
        _setup_recovery(tmp_path, monkeypatch)
        t3 = tmp_path / "age-multi.json"
        write_infra_bytes(t3, b"age-multi")
        assert read_maybe_encrypted(t3) == b"age-multi"

        # Both envelopes carry the age magic.
        assert t2.read_bytes().startswith(AGE_MAGIC)
        assert t3.read_bytes().startswith(AGE_MAGIC)
