"""Tests for paramem.backup.encryption."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from paramem.backup.encryption import (
    SecurityBackupsConfig,
    _clear_cipher_cache,
    assert_encryption_feasible,
    decrypt_bytes,
    encrypt_bytes,
    resolve_policy,
    should_encrypt,
)
from paramem.backup.types import ArtifactKind, EncryptAtRest, FatalConfigError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_FERNET_KEY = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="


def _make_valid_fernet_key() -> bytes:
    """Return a valid 32-byte Fernet key (base64url-encoded)."""
    from cryptography.fernet import Fernet

    return Fernet.generate_key()


# ---------------------------------------------------------------------------
# resolve_policy
# ---------------------------------------------------------------------------


class TestResolvePolicy:
    def test_returns_global_policy_when_no_per_kind(self):
        config = SecurityBackupsConfig(encrypt_at_rest=EncryptAtRest.ALWAYS)
        assert resolve_policy(ArtifactKind.CONFIG, config) is EncryptAtRest.ALWAYS

    def test_per_kind_overrides_global(self):
        config = SecurityBackupsConfig(
            encrypt_at_rest=EncryptAtRest.AUTO,
            per_kind={ArtifactKind.CONFIG: EncryptAtRest.NEVER},
        )
        assert resolve_policy(ArtifactKind.CONFIG, config) is EncryptAtRest.NEVER
        # Other kinds still get global
        assert resolve_policy(ArtifactKind.GRAPH, config) is EncryptAtRest.AUTO


# ---------------------------------------------------------------------------
# should_encrypt
# ---------------------------------------------------------------------------


class TestShouldEncrypt:
    def test_always_encrypts_regardless_of_key(self):
        assert should_encrypt(EncryptAtRest.ALWAYS, key_loaded=False) is True
        assert should_encrypt(EncryptAtRest.ALWAYS, key_loaded=True) is True

    def test_never_does_not_encrypt(self):
        assert should_encrypt(EncryptAtRest.NEVER, key_loaded=False) is False
        assert should_encrypt(EncryptAtRest.NEVER, key_loaded=True) is False

    def test_auto_follows_key_presence(self):
        assert should_encrypt(EncryptAtRest.AUTO, key_loaded=True) is True
        assert should_encrypt(EncryptAtRest.AUTO, key_loaded=False) is False


# ---------------------------------------------------------------------------
# assert_encryption_feasible
# ---------------------------------------------------------------------------


class TestAssertEncryptionFeasible:
    def test_always_with_key_does_not_raise(self):
        config = SecurityBackupsConfig(encrypt_at_rest=EncryptAtRest.ALWAYS)
        assert_encryption_feasible(config, key_loaded=True)  # no exception

    def test_always_without_key_raises_fatal(self):
        config = SecurityBackupsConfig(encrypt_at_rest=EncryptAtRest.ALWAYS)
        with pytest.raises(FatalConfigError):
            assert_encryption_feasible(config, key_loaded=False)

    def test_per_kind_always_without_key_raises(self):
        config = SecurityBackupsConfig(
            encrypt_at_rest=EncryptAtRest.AUTO,
            per_kind={ArtifactKind.REGISTRY: EncryptAtRest.ALWAYS},
        )
        with pytest.raises(FatalConfigError):
            assert_encryption_feasible(config, key_loaded=False)

    def test_auto_without_key_does_not_raise(self):
        config = SecurityBackupsConfig(encrypt_at_rest=EncryptAtRest.AUTO)
        assert_encryption_feasible(config, key_loaded=False)  # no exception

    def test_never_without_key_does_not_raise(self):
        config = SecurityBackupsConfig(encrypt_at_rest=EncryptAtRest.NEVER)
        assert_encryption_feasible(config, key_loaded=False)  # no exception

    def test_encrypt_without_key_when_always_raises(self):
        """Alias: same as test_always_without_key_raises_fatal — explicit name from task."""
        config = SecurityBackupsConfig(encrypt_at_rest=EncryptAtRest.ALWAYS)
        with pytest.raises(FatalConfigError):
            assert_encryption_feasible(config, key_loaded=False)


# ---------------------------------------------------------------------------
# encrypt / decrypt round-trip
# ---------------------------------------------------------------------------


class TestEncryptDecrypt:
    def setup_method(self):
        _clear_cipher_cache()
        # Defensive: ensure a clean env for every test in this class.
        os.environ.pop("PARAMEM_MASTER_KEY", None)
        os.environ.pop("PARAMEM_SNAPSHOT_KEY", None)

    def teardown_method(self):
        _clear_cipher_cache()
        os.environ.pop("PARAMEM_MASTER_KEY", None)
        os.environ.pop("PARAMEM_SNAPSHOT_KEY", None)

    def _set_key(self):
        key = _make_valid_fernet_key()
        os.environ["PARAMEM_MASTER_KEY"] = key.decode()
        return key

    def test_roundtrip(self):
        """Encrypt then decrypt returns original plaintext."""
        self._set_key()
        plaintext = b"the quick brown fox"
        ciphertext = encrypt_bytes(plaintext)
        assert ciphertext != plaintext
        assert decrypt_bytes(ciphertext) == plaintext

    def test_encrypt_produces_different_ciphertext_each_call(self):
        """Fernet includes a random IV — same plaintext produces different ciphertext."""
        self._set_key()
        plaintext = b"same input"
        ct1 = encrypt_bytes(plaintext)
        ct2 = encrypt_bytes(plaintext)
        assert ct1 != ct2

    def test_encrypt_optional_when_auto_and_no_key_passthrough(self):
        """When no key is set and policy is AUTO, should_encrypt returns False
        (no encryption attempted).  encrypt_bytes would raise if called without
        a key — this test verifies should_encrypt(AUTO, False) → False."""
        # No key in environment
        os.environ.pop("PARAMEM_MASTER_KEY", None)
        result = should_encrypt(EncryptAtRest.AUTO, key_loaded=False)
        assert result is False  # passthrough — no encryption, no key required

    def test_encrypt_without_key_raises_runtime_error(self):
        """Calling encrypt_bytes without PARAMEM_MASTER_KEY set raises RuntimeError."""
        os.environ.pop("PARAMEM_MASTER_KEY", None)
        with pytest.raises(RuntimeError, match="PARAMEM_MASTER_KEY"):
            encrypt_bytes(b"data")

    def test_clear_cipher_cache_forces_reload(self):
        """After _clear_cipher_cache(), the cipher is rebuilt on next call.

        Verifies operationally: set key A, encrypt, clear cache, set key B,
        decrypt with key B fails (was encrypted with key A).
        """
        from cryptography.fernet import InvalidToken

        key_a = _make_valid_fernet_key()
        key_b = _make_valid_fernet_key()
        assert key_a != key_b

        os.environ["PARAMEM_MASTER_KEY"] = key_a.decode()
        ciphertext = encrypt_bytes(b"secret")

        _clear_cipher_cache()
        os.environ["PARAMEM_MASTER_KEY"] = key_b.decode()

        with pytest.raises(InvalidToken):
            decrypt_bytes(ciphertext)

    def test_cipher_cache_is_lazy(self):
        """The cipher is built on first call, not at import time.

        Monkeypatches the Fernet constructor to count calls.
        """
        from paramem.backup import encryption as enc_module

        key = _make_valid_fernet_key()
        os.environ["PARAMEM_MASTER_KEY"] = key.decode()
        _clear_cipher_cache()

        call_count = 0
        original_fernet = enc_module.Fernet

        class _CountingFernet:
            def __init__(self, k):
                nonlocal call_count
                call_count += 1
                self._inner = original_fernet(k)

            def encrypt(self, data):
                return self._inner.encrypt(data)

            def decrypt(self, data):
                return self._inner.decrypt(data)

        with patch.object(enc_module, "Fernet", _CountingFernet):
            _clear_cipher_cache()
            assert call_count == 0  # not yet called
            encrypt_bytes(b"a")
            assert call_count == 1  # built on first use
            encrypt_bytes(b"b")
            assert call_count == 1  # reused from cache

    def test_cache_clear_supported_operationally(self):
        """_clear_cipher_cache() is a supported operational call; must not raise."""
        _clear_cipher_cache()  # safe even when empty
        self._set_key()
        encrypt_bytes(b"populate cache")
        _clear_cipher_cache()  # safe when populated
        _clear_cipher_cache()  # idempotent

    def test_invalid_fernet_key_raises_fatal_config(self):
        """PARAMEM_MASTER_KEY set to an invalid Fernet key raises FatalConfigError.

        Fix #3: verifies that:
        1. FatalConfigError is raised (not ValueError or anything else).
        2. The message mentions "valid Fernet key" to identify the root cause.
        3. The raw key value is NOT included in the message (security: no
           credential leakage to logs).
        """
        bad_key = "not-a-valid-key"
        os.environ["PARAMEM_MASTER_KEY"] = bad_key
        _clear_cipher_cache()

        with pytest.raises(FatalConfigError) as exc_info:
            encrypt_bytes(b"payload")

        message = str(exc_info.value)
        assert "valid Fernet key" in message, (
            f"FatalConfigError message must mention 'valid Fernet key': {message!r}"
        )
        # Security: the raw key value must NOT appear in the error message
        assert bad_key not in message, (
            f"FatalConfigError must not leak the raw key value in the message: {message!r}"
        )
