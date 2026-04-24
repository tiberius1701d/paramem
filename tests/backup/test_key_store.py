"""Tests for the daily-key file primitives (Slice C of WP2b).

Covers:

1. Mint → wrap → write → read → unlock round-trip reconstructs the same identity.
2. Wrong passphrase on unlock surfaces :class:`pyrage.DecryptError`.
3. On-disk file mode is enforced at ``0o600``; parent directory at ``0o700``.
4. Tamper on the wrapped envelope surfaces :class:`pyrage.DecryptError`.
5. Missing daily-key file raises :class:`FileNotFoundError` verbatim.
6. Env-var passphrase lookup; unset env raises :class:`RuntimeError` with the
   actionable env-var name.
7. Written envelope carries the age v1 magic (``is_age_envelope`` recognises it).
8. Stale ``<path>.tmp`` from a crash does not sabotage the 0o600 mode guarantee.
9. ``wrap_daily_identity`` rejects empty passphrase and non-X25519 identities.
10. Default path points under ``~/.config/paramem`` as the handover specifies.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pyrage
import pytest
from pyrage import x25519

from paramem.backup.age_envelope import AGE_MAGIC, PluginRecipientRejected, is_age_envelope
from paramem.backup.key_store import (
    DAILY_KEY_PATH_DEFAULT,
    DAILY_PASSPHRASE_ENV_VAR,
    RECOVERY_PUB_PATH_DEFAULT,
    _clear_daily_identity_cache,
    daily_identity_loadable,
    daily_passphrase_env_value,
    load_daily_identity,
    load_daily_identity_cached,
    load_recovery_recipient,
    mint_daily_identity,
    recovery_pub_available,
    wrap_daily_identity,
    write_daily_key_file,
    write_recovery_pub_file,
)


def _key_path(tmp_path: Path) -> Path:
    return tmp_path / "paramem" / "daily_key.age"


class TestRoundTrip:
    def test_mint_wrap_write_load_reconstructs_identity(self, tmp_path: Path) -> None:
        ident = mint_daily_identity()
        path = _key_path(tmp_path)
        wrapped = wrap_daily_identity(ident, "correct horse battery staple")
        write_daily_key_file(wrapped, path)
        loaded = load_daily_identity(path, passphrase="correct horse battery staple")
        assert str(loaded) == str(ident), "unlocked identity must match the one that was wrapped"

    def test_loaded_identity_can_decrypt_its_own_recipient_envelope(self, tmp_path: Path) -> None:
        """End-to-end sanity: the unlocked identity still works as a decryption key."""
        from paramem.backup.age_envelope import age_decrypt_bytes, age_encrypt_bytes

        ident = mint_daily_identity()
        path = _key_path(tmp_path)
        write_daily_key_file(wrap_daily_identity(ident, "pw"), path)
        loaded = load_daily_identity(path, passphrase="pw")

        ct = age_encrypt_bytes(b"payload", [ident.to_public()])
        assert age_decrypt_bytes(ct, [loaded]) == b"payload"


class TestPassphraseFailures:
    def test_wrong_passphrase_raises(self, tmp_path: Path) -> None:
        path = _key_path(tmp_path)
        write_daily_key_file(wrap_daily_identity(mint_daily_identity(), "right"), path)
        with pytest.raises(pyrage.DecryptError):
            load_daily_identity(path, passphrase="wrong")

    def test_empty_passphrase_on_wrap_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            wrap_daily_identity(mint_daily_identity(), "")

    def test_wrap_rejects_non_x25519_identity(self) -> None:
        with pytest.raises(TypeError):
            wrap_daily_identity(object(), "pw")  # type: ignore[arg-type]


class TestTamperDetection:
    def test_tampered_envelope_raises(self, tmp_path: Path) -> None:
        path = _key_path(tmp_path)
        wrapped = wrap_daily_identity(mint_daily_identity(), "pw")
        write_daily_key_file(wrapped, path)

        raw = path.read_bytes()
        # Zero out bytes deep inside the envelope, past the age header.
        tampered = raw[:80] + bytes(len(raw) - 80)
        path.write_bytes(tampered)

        with pytest.raises(pyrage.DecryptError):
            load_daily_identity(path, passphrase="pw")


class TestFileMode:
    def test_key_file_is_0600(self, tmp_path: Path) -> None:
        path = _key_path(tmp_path)
        write_daily_key_file(wrap_daily_identity(mint_daily_identity(), "pw"), path)
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600, f"daily key file mode must be 0600, got {oct(mode)}"

    def test_parent_dir_created_with_0700(self, tmp_path: Path) -> None:
        path = tmp_path / "new-config-root" / "daily_key.age"
        assert not path.parent.exists()
        write_daily_key_file(wrap_daily_identity(mint_daily_identity(), "pw"), path)
        mode = stat.S_IMODE(path.parent.stat().st_mode)
        assert mode == 0o700, f"parent dir mode must be 0700, got {oct(mode)}"

    def test_existing_parent_dir_mode_preserved(self, tmp_path: Path) -> None:
        """We never rewrite the mode of a pre-existing parent directory."""
        parent = tmp_path / "preexisting"
        parent.mkdir(mode=0o755)
        # A subsequent chmod is required because mkdir honours umask.
        os.chmod(parent, 0o755)
        path = parent / "daily_key.age"

        write_daily_key_file(wrap_daily_identity(mint_daily_identity(), "pw"), path)

        assert stat.S_IMODE(parent.stat().st_mode) == 0o755
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_stale_tmp_does_not_leak_bad_mode(self, tmp_path: Path) -> None:
        """A leftover .tmp with weak mode must not sabotage the 0o600 guarantee."""
        path = _key_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stale_tmp = path.with_suffix(path.suffix + ".tmp")
        stale_tmp.write_bytes(b"leftover from a crash")
        os.chmod(stale_tmp, 0o644)

        write_daily_key_file(wrap_daily_identity(mint_daily_identity(), "pw"), path)

        assert not stale_tmp.exists(), "stale tmp must be consumed or removed"
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


class TestMissingFile:
    def test_load_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_daily_identity(tmp_path / "does-not-exist.age", passphrase="pw")


class TestEnvVarPassphrase:
    def test_env_passphrase_used_when_not_passed_explicitly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ident = mint_daily_identity()
        path = _key_path(tmp_path)
        write_daily_key_file(wrap_daily_identity(ident, "from-env"), path)

        monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "from-env")
        loaded = load_daily_identity(path)
        assert str(loaded) == str(ident)

    def test_unset_env_raises_runtime_error_with_env_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _key_path(tmp_path)
        write_daily_key_file(wrap_daily_identity(mint_daily_identity(), "pw"), path)

        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
        with pytest.raises(RuntimeError, match=DAILY_PASSPHRASE_ENV_VAR):
            load_daily_identity(path)

    def test_empty_env_treated_as_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _key_path(tmp_path)
        write_daily_key_file(wrap_daily_identity(mint_daily_identity(), "pw"), path)

        monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "")
        assert daily_passphrase_env_value() is None
        with pytest.raises(RuntimeError, match=DAILY_PASSPHRASE_ENV_VAR):
            load_daily_identity(path)

    def test_explicit_passphrase_overrides_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ident = mint_daily_identity()
        path = _key_path(tmp_path)
        write_daily_key_file(wrap_daily_identity(ident, "explicit"), path)

        monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "would-be-wrong")
        loaded = load_daily_identity(path, passphrase="explicit")
        assert str(loaded) == str(ident)


class TestEnvelopeFormat:
    def test_wrapped_envelope_carries_age_magic(self, tmp_path: Path) -> None:
        path = _key_path(tmp_path)
        write_daily_key_file(wrap_daily_identity(mint_daily_identity(), "pw"), path)
        assert path.read_bytes().startswith(AGE_MAGIC)
        assert is_age_envelope(path) is True


class TestDefaults:
    def test_default_path_under_user_config(self) -> None:
        """Default path is operator-visible contract — pin it."""
        expected_tail = Path(".config") / "paramem" / "daily_key.age"
        assert DAILY_KEY_PATH_DEFAULT.parts[-3:] == expected_tail.parts
        assert DAILY_KEY_PATH_DEFAULT.is_absolute()

    def test_recovery_pub_default_path_under_user_config(self) -> None:
        expected_tail = Path(".config") / "paramem" / "recovery.pub"
        assert RECOVERY_PUB_PATH_DEFAULT.parts[-3:] == expected_tail.parts
        assert RECOVERY_PUB_PATH_DEFAULT.is_absolute()

    def test_default_env_var_name(self) -> None:
        assert DAILY_PASSPHRASE_ENV_VAR == "PARAMEM_DAILY_PASSPHRASE"


class TestRecoveryPub:
    def test_recovery_pub_round_trip(self, tmp_path: Path) -> None:
        ident = x25519.Identity.generate()
        path = tmp_path / "paramem" / "recovery.pub"
        write_recovery_pub_file(ident.to_public(), path)
        loaded = load_recovery_recipient(path)
        assert str(loaded) == str(ident.to_public())

    def test_recovery_pub_file_mode_is_0644(self, tmp_path: Path) -> None:
        ident = x25519.Identity.generate()
        path = tmp_path / "paramem" / "recovery.pub"
        write_recovery_pub_file(ident.to_public(), path)
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o644, f"recovery pub file mode must be 0644, got {oct(mode)}"

    def test_recovery_pub_parent_dir_created_at_0700(self, tmp_path: Path) -> None:
        ident = x25519.Identity.generate()
        parent = tmp_path / "fresh-config-root"
        assert not parent.exists()
        write_recovery_pub_file(ident.to_public(), parent / "recovery.pub")
        assert stat.S_IMODE(parent.stat().st_mode) == 0o700

    def test_recovery_pub_body_is_bech32_plus_newline(self, tmp_path: Path) -> None:
        ident = x25519.Identity.generate()
        path = tmp_path / "recovery.pub"
        write_recovery_pub_file(ident.to_public(), path)
        body = path.read_text(encoding="utf-8")
        assert body.endswith("\n")
        assert body.rstrip("\n") == str(ident.to_public())
        assert body.rstrip("\n").startswith("age1")

    def test_load_recovery_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_recovery_recipient(tmp_path / "does-not-exist.pub")

    def test_load_recovery_empty_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "recovery.pub"
        path.write_text("   \n\n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_recovery_recipient(path)

    def test_load_recovery_rejects_non_age_prefix(self, tmp_path: Path) -> None:
        path = tmp_path / "recovery.pub"
        path.write_text("AGE-SECRET-KEY-1notapublickey\n", encoding="utf-8")
        with pytest.raises(ValueError, match="not an age recipient"):
            load_recovery_recipient(path)

    def test_load_recovery_rejects_garbage(self, tmp_path: Path) -> None:
        path = tmp_path / "recovery.pub"
        path.write_text("age1garbage-not-a-real-key\n", encoding="utf-8")
        with pytest.raises(PluginRecipientRejected):
            load_recovery_recipient(path)

    def test_write_recovery_pub_rejects_non_x25519(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError):
            write_recovery_pub_file(object(), tmp_path / "recovery.pub")  # type: ignore[arg-type]

    def test_recovery_pub_tolerates_surrounding_whitespace(self, tmp_path: Path) -> None:
        ident = x25519.Identity.generate()
        path = tmp_path / "recovery.pub"
        path.write_text(f"\n  {ident.to_public()}  \n", encoding="utf-8")
        loaded = load_recovery_recipient(path)
        assert str(loaded) == str(ident.to_public())


class TestIdentityMinting:
    def test_mint_returns_native_x25519_identity(self) -> None:
        ident = mint_daily_identity()
        assert isinstance(ident, x25519.Identity)
        assert str(ident).startswith("AGE-SECRET-KEY-1")

    def test_two_mints_yield_distinct_identities(self) -> None:
        assert str(mint_daily_identity()) != str(mint_daily_identity())


@pytest.fixture
def _clear_cache_between_tests():
    """Ensure the module-level identity cache does not leak across tests."""
    _clear_daily_identity_cache()
    yield
    _clear_daily_identity_cache()


class TestLoadableProbes:
    def test_daily_identity_loadable_true_when_env_and_file_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "daily_key.age"
        write_daily_key_file(wrap_daily_identity(mint_daily_identity(), "pw"), path)
        monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "pw")
        assert daily_identity_loadable(path) is True

    def test_daily_identity_loadable_false_when_env_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "daily_key.age"
        write_daily_key_file(wrap_daily_identity(mint_daily_identity(), "pw"), path)
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
        assert daily_identity_loadable(path) is False

    def test_daily_identity_loadable_false_when_empty_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "daily_key.age"
        write_daily_key_file(wrap_daily_identity(mint_daily_identity(), "pw"), path)
        monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "")
        assert daily_identity_loadable(path) is False

    def test_daily_identity_loadable_false_when_file_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "pw")
        assert daily_identity_loadable(tmp_path / "nope.age") is False

    def test_recovery_pub_available_true_when_file_exists(self, tmp_path: Path) -> None:
        path = tmp_path / "recovery.pub"
        write_recovery_pub_file(x25519.Identity.generate().to_public(), path)
        assert recovery_pub_available(path) is True

    def test_recovery_pub_available_false_when_file_missing(self, tmp_path: Path) -> None:
        assert recovery_pub_available(tmp_path / "nope.pub") is False


class TestCachedIdentity:
    def test_cached_load_returns_same_object(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        _clear_cache_between_tests: None,
    ) -> None:
        path = tmp_path / "daily_key.age"
        ident = mint_daily_identity()
        write_daily_key_file(wrap_daily_identity(ident, "pw"), path)
        monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "pw")

        first = load_daily_identity_cached(path)
        second = load_daily_identity_cached(path)
        assert first is second, "second call must return the cached instance"
        assert str(first) == str(ident)

    def test_clear_cache_forces_reload(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        _clear_cache_between_tests: None,
    ) -> None:
        path = tmp_path / "daily_key.age"
        write_daily_key_file(wrap_daily_identity(mint_daily_identity(), "pw"), path)
        monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "pw")

        first = load_daily_identity_cached(path)
        _clear_daily_identity_cache()
        second = load_daily_identity_cached(path)
        assert first is not second, "clearing the cache must force a fresh unwrap"
        assert str(first) == str(second), "same underlying identity after rewrap"

    def test_cached_load_raises_without_passphrase(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        _clear_cache_between_tests: None,
    ) -> None:
        path = tmp_path / "daily_key.age"
        write_daily_key_file(wrap_daily_identity(mint_daily_identity(), "pw"), path)
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)

        with pytest.raises(RuntimeError, match=DAILY_PASSPHRASE_ENV_VAR):
            load_daily_identity_cached(path)

    def test_failed_load_does_not_poison_cache(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        _clear_cache_between_tests: None,
    ) -> None:
        """A failed first load must leave the cache empty so the next attempt
        (after the operator fixes the env) succeeds cleanly."""
        path = tmp_path / "daily_key.age"
        write_daily_key_file(wrap_daily_identity(mint_daily_identity(), "pw"), path)

        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
        with pytest.raises(RuntimeError):
            load_daily_identity_cached(path)

        monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "pw")
        ident = load_daily_identity_cached(path)
        assert str(ident).startswith("AGE-SECRET-KEY-1")
