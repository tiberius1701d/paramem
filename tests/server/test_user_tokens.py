"""Tests for `paramem.server.user_tokens` — per-user bearer-token store."""

from __future__ import annotations

from pathlib import Path

import pytest

from paramem.backup.key_store import (
    DAILY_PASSPHRASE_ENV_VAR,
    _clear_daily_identity_cache,
    mint_daily_identity,
    wrap_daily_identity,
    write_daily_key_file,
)
from paramem.server.user_tokens import UserTokenStore

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _setup_daily(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passphrase: str = "pw"):
    """Mint + wrap + write a daily identity; point the env + module default at it.

    Mirrors the helper in ``tests/backup/test_encryption_primitives.py``.
    """
    ident = mint_daily_identity()
    key_path = tmp_path / "daily_key.age"
    write_daily_key_file(wrap_daily_identity(ident, passphrase), key_path)
    monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, passphrase)
    monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path)
    _clear_daily_identity_cache()
    return ident


@pytest.fixture(autouse=True)
def _env_isolation(monkeypatch: pytest.MonkeyPatch):
    """Isolate daily identity cache per test."""
    _clear_daily_identity_cache()
    yield
    _clear_daily_identity_cache()


@pytest.fixture()
def no_daily_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Fixture: ensure no daily key is loadable."""
    monkeypatch.setattr(
        "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT",
        tmp_path / "absent.age",
    )
    monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
    _clear_daily_identity_cache()


@pytest.fixture()
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "user_tokens.json"


# ---------------------------------------------------------------------------
# Basic store operations
# ---------------------------------------------------------------------------


class TestMintLookupRoundtrip:
    def test_mint_returns_plaintext_and_lookup_finds_speaker(
        self, tmp_path, monkeypatch, store_path
    ):
        """mint returns a token that lookup resolves to the correct speaker_id."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)

        token = store.mint("Speaker0", "Test Device")

        assert isinstance(token, str)
        assert len(token) > 20
        assert store.lookup(token) == "Speaker0"

    def test_unknown_token_returns_none(self, tmp_path, monkeypatch, store_path):
        """Lookup of a token that was never minted returns None."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        store.mint("Speaker0", "Device A")

        result = store.lookup("completely-unknown-token-value")

        assert result is None

    def test_multiple_tokens_resolve_independently(self, tmp_path, monkeypatch, store_path):
        """Two tokens for different speakers each resolve to their own speaker_id."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)

        t1 = store.mint("Speaker0", "iPad")
        t2 = store.mint("Speaker1", "Desktop")

        assert store.lookup(t1) == "Speaker0"
        assert store.lookup(t2) == "Speaker1"


# ---------------------------------------------------------------------------
# Revocation
# ---------------------------------------------------------------------------


class TestRevoke:
    def test_revoke_token_makes_lookup_return_none(self, tmp_path, monkeypatch, store_path):
        """revoke_token() on a valid token makes lookup return None."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        token = store.mint("Speaker0", "Device A")

        found = store.revoke_token(token)

        assert found is True
        assert store.lookup(token) is None

    def test_revoke_speaker_revokes_all_tokens(self, tmp_path, monkeypatch, store_path):
        """revoke_speaker() revokes ALL tokens for that speaker."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)

        t1 = store.mint("Speaker0", "iPad")
        t2 = store.mint("Speaker0", "Phone")
        t_other = store.mint("Speaker1", "Other device")

        count = store.revoke_speaker("Speaker0")

        assert count == 2
        assert store.lookup(t1) is None
        assert store.lookup(t2) is None
        # Other speaker's token unaffected.
        assert store.lookup(t_other) == "Speaker1"

    def test_revoke_token_unknown_returns_false(self, tmp_path, monkeypatch, store_path):
        """revoke_token() with an unknown token returns False."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        store.mint("Speaker0", "Device")

        found = store.revoke_token("no-such-token-value")

        assert found is False

    def test_revoke_speaker_unknown_returns_zero(self, tmp_path, monkeypatch, store_path):
        """revoke_speaker() with an unknown speaker_id returns 0."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        store.mint("Speaker0", "Device")

        count = store.revoke_speaker("Speaker99")

        assert count == 0

    def test_revoke_token_already_revoked_returns_false(self, tmp_path, monkeypatch, store_path):
        """revoke_token() on an already-revoked token returns False (idempotent)."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        token = store.mint("Speaker0", "Device")
        store.revoke_token(token)

        found = store.revoke_token(token)

        assert found is False

    def test_revoke_speaker_already_all_revoked_returns_zero(
        self, tmp_path, monkeypatch, store_path
    ):
        """revoke_speaker() when all tokens already revoked returns 0 (idempotent)."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        store.mint("Speaker0", "Device")
        store.revoke_speaker("Speaker0")

        count = store.revoke_speaker("Speaker0")

        assert count == 0


# ---------------------------------------------------------------------------
# list() — never exposes hashes or plaintext tokens
# ---------------------------------------------------------------------------


class TestList:
    def test_list_contains_expected_fields(self, tmp_path, monkeypatch, store_path):
        """list() returns entries with the four public fields."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        store.mint("Speaker0", "My iPad")

        entries = store.list()

        assert len(entries) == 1
        entry = entries[0]
        assert entry["speaker_id"] == "Speaker0"
        assert entry["label"] == "My iPad"
        assert "created" in entry
        assert entry["revoked"] is False

    def test_list_never_contains_hash_or_plaintext(self, tmp_path, monkeypatch, store_path):
        """list() output must not contain the sha256 key or plaintext token."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        token = store.mint("Speaker0", "Device")

        entries = store.list()

        for entry in entries:
            # No field should be the plaintext token.
            assert token not in entry.values()
            # None of the dict keys represent a sha256 hex digest
            # (64-char lowercase hex string).
            for key in entry:
                assert not (isinstance(key, str) and len(key) == 64 and key.islower())

    def test_list_shows_revoked_flag(self, tmp_path, monkeypatch, store_path):
        """list() marks revoked entries with revoked=True."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        token = store.mint("Speaker0", "Device")
        store.revoke_token(token)

        entries = store.list()

        assert len(entries) == 1
        assert entries[0]["revoked"] is True


# ---------------------------------------------------------------------------
# has_active_tokens
# ---------------------------------------------------------------------------


class TestHasActiveTokens:
    def test_false_when_empty(self, tmp_path, monkeypatch, store_path):
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        assert store.has_active_tokens() is False

    def test_true_after_mint(self, tmp_path, monkeypatch, store_path):
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        store.mint("Speaker0", "Device")
        assert store.has_active_tokens() is True

    def test_false_after_all_revoked(self, tmp_path, monkeypatch, store_path):
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        store.mint("Speaker0", "Device")
        store.revoke_speaker("Speaker0")
        assert store.has_active_tokens() is False


# ---------------------------------------------------------------------------
# On-disk security guarantees
# ---------------------------------------------------------------------------


class TestOnDiskSecurity:
    def test_on_disk_bytes_do_not_contain_plaintext_token(self, tmp_path, monkeypatch, store_path):
        """The plaintext token must never appear in the on-disk file bytes."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        token = store.mint("Speaker0", "Device")

        disk_bytes = store_path.read_bytes()

        assert token.encode("utf-8") not in disk_bytes

    def test_on_disk_is_age_envelope_when_daily_loaded(self, tmp_path, monkeypatch, store_path):
        """When the daily key is loaded, the store is written as an age envelope."""
        from paramem.backup.age_envelope import AGE_MAGIC

        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        store.mint("Speaker0", "Device")

        disk_bytes = store_path.read_bytes()

        assert disk_bytes.startswith(AGE_MAGIC), (
            f"expected age envelope on disk; got {disk_bytes[:40]!r}"
        )

    def test_persist_without_daily_key_succeeds_plaintext(
        self, tmp_path, monkeypatch, store_path, no_daily_key
    ):
        """Security OFF: mint + persist succeeds without a daily key (AUTO plaintext).

        The store follows the deployment-wide AUTO encryption mode — plaintext
        when no daily key is loaded.  The on-disk file must be parseable JSON
        containing the tokens map but must NOT contain the plaintext token value
        (only the sha256 hash is stored).
        """
        import json

        store = UserTokenStore(store_path)
        # mint() must not raise in Security OFF mode.
        token = store.mint("Speaker0", "TestDevice")

        assert store_path.exists(), "store file must exist after mint"

        disk_bytes = store_path.read_bytes()

        # Must not begin with the age magic — plaintext in Security OFF.
        assert not disk_bytes.startswith(b"age-encryption.org"), (
            "expected plaintext JSON in Security OFF mode; got age envelope"
        )

        # Must be valid JSON and contain the tokens map.
        data = json.loads(disk_bytes.decode("utf-8"))
        assert "tokens" in data, "on-disk JSON must contain 'tokens' key"
        assert len(data["tokens"]) == 1, "expected exactly one token entry"

        # Plaintext token value must not appear anywhere on disk.
        assert token.encode("utf-8") not in disk_bytes, (
            "plaintext token must never be written to disk"
        )

    def test_empty_store_can_persist_without_daily_key(
        self, tmp_path, monkeypatch, store_path, no_daily_key
    ):
        """An empty store writes successfully even without a daily key (no-op)."""
        store = UserTokenStore(store_path)
        # Should not raise.
        store._save()

    def test_store_roundtrips_across_reload(self, tmp_path, monkeypatch, store_path):
        """A minted token survives store teardown and reload."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        token = store.mint("Speaker0", "Device")

        store2 = UserTokenStore(store_path)

        assert store2.lookup(token) == "Speaker0"


# ---------------------------------------------------------------------------
# TOCTOU guard clears in-memory state
# ---------------------------------------------------------------------------


class TestToctouClearsMemory:
    def test_toctou_guard_clears_in_memory_tokens(self, tmp_path, monkeypatch, store_path):
        """When the TOCTOU guard fires, in-memory tokens are reset.

        After the RuntimeError is raised, lookup() and list() must reflect
        cleared state — not the stale in-memory token that was inserted into
        _tokens before _save() raised.
        """
        from unittest.mock import patch

        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)

        # Capture the token value before mint() raises so we can verify lookup.
        known_token = "fixed-token-value-for-toctou-test"
        monkeypatch.setattr(
            "paramem.server.user_tokens.secrets.token_urlsafe", lambda n: known_token
        )

        def _plaintext_write(path: Path, payload: bytes) -> None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(payload)

        with patch("paramem.server.user_tokens.write_infra_bytes", side_effect=_plaintext_write):
            with pytest.raises(RuntimeError, match="written in plaintext"):
                store.mint("Speaker0", "TestDevice")

        # After the guard fires, in-memory state must be cleared — the minted
        # token must not authenticate even though it was inserted into _tokens
        # before _save() raised.
        assert store.lookup(known_token) is None, (
            "lookup() must return None after TOCTOU guard clears in-memory state"
        )
        assert store.list() == [], "list() must return [] after TOCTOU guard clears in-memory state"
