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

        token = store.mint("speaker0", "Test Device")

        assert isinstance(token, str)
        assert len(token) > 20
        assert store.lookup(token) == "speaker0"

    def test_unknown_token_returns_none(self, tmp_path, monkeypatch, store_path):
        """Lookup of a token that was never minted returns None."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        store.mint("speaker0", "Device A")

        result = store.lookup("completely-unknown-token-value")

        assert result is None

    def test_multiple_tokens_resolve_independently(self, tmp_path, monkeypatch, store_path):
        """Two tokens for different speakers each resolve to their own speaker_id."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)

        t1 = store.mint("speaker0", "iPad")
        t2 = store.mint("speaker1", "Desktop")

        assert store.lookup(t1) == "speaker0"
        assert store.lookup(t2) == "speaker1"


# ---------------------------------------------------------------------------
# Revocation
# ---------------------------------------------------------------------------


class TestRevoke:
    def test_revoke_token_makes_lookup_return_none(self, tmp_path, monkeypatch, store_path):
        """revoke_token() on a valid token makes lookup return None."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        token = store.mint("speaker0", "Device A")

        found = store.revoke_token(token)

        assert found is True
        assert store.lookup(token) is None

    def test_revoke_speaker_revokes_all_tokens(self, tmp_path, monkeypatch, store_path):
        """revoke_speaker() revokes ALL tokens for that speaker."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)

        t1 = store.mint("speaker0", "iPad")
        t2 = store.mint("speaker0", "Phone")
        t_other = store.mint("speaker1", "Other device")

        count = store.revoke_speaker("speaker0")

        assert count == 2
        assert store.lookup(t1) is None
        assert store.lookup(t2) is None
        # Other speaker's token unaffected.
        assert store.lookup(t_other) == "speaker1"

    def test_revoke_token_unknown_returns_false(self, tmp_path, monkeypatch, store_path):
        """revoke_token() with an unknown token returns False."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        store.mint("speaker0", "Device")

        found = store.revoke_token("no-such-token-value")

        assert found is False

    def test_revoke_speaker_unknown_returns_zero(self, tmp_path, monkeypatch, store_path):
        """revoke_speaker() with an unknown speaker_id returns 0."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        store.mint("speaker0", "Device")

        count = store.revoke_speaker("speaker99")

        assert count == 0

    def test_revoke_token_already_revoked_returns_false(self, tmp_path, monkeypatch, store_path):
        """revoke_token() on an already-revoked token returns False (idempotent)."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        token = store.mint("speaker0", "Device")
        store.revoke_token(token)

        found = store.revoke_token(token)

        assert found is False

    def test_revoke_speaker_already_all_revoked_returns_zero(
        self, tmp_path, monkeypatch, store_path
    ):
        """revoke_speaker() when all tokens already revoked returns 0 (idempotent)."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        store.mint("speaker0", "Device")
        store.revoke_speaker("speaker0")

        count = store.revoke_speaker("speaker0")

        assert count == 0

    def test_revoke_label_revokes_matching_entries(self, tmp_path, monkeypatch, store_path):
        """revoke_label() revokes all non-revoked tokens with that exact label."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        t_phone0 = store.mint("speaker0", "phone")
        t_phone1 = store.mint("speaker1", "phone")
        t_tablet = store.mint("speaker0", "tablet")

        count = store.revoke_label("phone")

        assert count == 2
        assert store.lookup(t_phone0) is None
        assert store.lookup(t_phone1) is None
        assert store.lookup(t_tablet) == "speaker0"

    def test_revoke_label_unknown_returns_zero(self, tmp_path, monkeypatch, store_path):
        """revoke_label() with an unknown label returns 0."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        store.mint("speaker0", "tablet")

        count = store.revoke_label("no-such-label")

        assert count == 0

    def test_revoke_label_already_revoked_returns_zero(self, tmp_path, monkeypatch, store_path):
        """revoke_label() on an already-revoked label returns 0 (idempotent)."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        store.mint("speaker0", "phone")
        store.revoke_label("phone")

        count = store.revoke_label("phone")

        assert count == 0


# ---------------------------------------------------------------------------
# list() — never exposes hashes or plaintext tokens
# ---------------------------------------------------------------------------


class TestList:
    def test_list_contains_expected_fields(self, tmp_path, monkeypatch, store_path):
        """list() returns entries with the four public fields."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        store.mint("speaker0", "My iPad")

        entries = store.list()

        assert len(entries) == 1
        entry = entries[0]
        assert entry["speaker_id"] == "speaker0"
        assert entry["label"] == "My iPad"
        assert "created" in entry
        assert entry["revoked"] is False

    def test_list_never_contains_hash_or_plaintext(self, tmp_path, monkeypatch, store_path):
        """list() output must not contain the sha256 key or plaintext token."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        token = store.mint("speaker0", "Device")

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
        token = store.mint("speaker0", "Device")
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
        store.mint("speaker0", "Device")
        assert store.has_active_tokens() is True

    def test_false_after_all_revoked(self, tmp_path, monkeypatch, store_path):
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        store.mint("speaker0", "Device")
        store.revoke_speaker("speaker0")
        assert store.has_active_tokens() is False


# ---------------------------------------------------------------------------
# count_active
# ---------------------------------------------------------------------------


class TestCountActive:
    """count_active() returns only non-revoked token count (startup posture fix).

    Regression: the startup log previously used ``len(store.list())`` which
    counted revoked entries, inflating the "N active per-user token(s)" message.
    """

    def test_zero_when_empty(self, tmp_path, monkeypatch, store_path):
        """count_active() returns 0 for an empty store."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        assert store.count_active() == 0

    def test_counts_only_non_revoked(self, tmp_path, monkeypatch, store_path):
        """count_active() excludes revoked entries from the total.

        Mint 3 tokens, revoke 2 — count_active() must return 1, not 3.
        """
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        t1 = store.mint("speaker0", "Device A")
        t2 = store.mint("speaker0", "Device B")
        store.mint("speaker1", "Device C")

        # Revoke two of the three.
        store.revoke_token(t1)
        store.revoke_token(t2)

        assert store.count_active() == 1

    def test_equals_total_when_none_revoked(self, tmp_path, monkeypatch, store_path):
        """count_active() equals total token count when none are revoked."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        store.mint("speaker0", "Device A")
        store.mint("speaker1", "Device B")

        assert store.count_active() == 2

    def test_zero_after_all_revoked(self, tmp_path, monkeypatch, store_path):
        """count_active() returns 0 after all tokens are revoked."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        store.mint("speaker0", "Device A")
        store.revoke_speaker("speaker0")

        assert store.count_active() == 0


# ---------------------------------------------------------------------------
# On-disk security guarantees
# ---------------------------------------------------------------------------


class TestOnDiskSecurity:
    def test_on_disk_bytes_do_not_contain_plaintext_token(self, tmp_path, monkeypatch, store_path):
        """The plaintext token must never appear in the on-disk file bytes."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        token = store.mint("speaker0", "Device")

        disk_bytes = store_path.read_bytes()

        assert token.encode("utf-8") not in disk_bytes

    def test_on_disk_is_age_envelope_when_daily_loaded(self, tmp_path, monkeypatch, store_path):
        """When the daily key is loaded, the store is written as an age envelope."""
        from paramem.backup.age_envelope import AGE_MAGIC

        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        store.mint("speaker0", "Device")

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
        token = store.mint("speaker0", "TestDevice")

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
        token = store.mint("speaker0", "Device")

        store2 = UserTokenStore(store_path)

        assert store2.lookup(token) == "speaker0"


# ---------------------------------------------------------------------------
# TOCTOU guard clears in-memory state
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Scope field
# ---------------------------------------------------------------------------


class TestScope:
    def test_mint_admin_scope_resolve_returns_admin(self, tmp_path, monkeypatch, store_path):
        """mint(..., scope='admin') → resolve() returns scope 'admin'."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        token = store.mint("speaker0", "Admin Device", scope="admin")

        result = store.resolve(token)
        assert result is not None
        _auth, _sid, scope = result
        assert scope == "admin"

    def test_mint_default_scope_resolve_returns_chat(self, tmp_path, monkeypatch, store_path):
        """mint() with default scope → resolve() returns scope 'chat'."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        token = store.mint("speaker0", "Phone")

        result = store.resolve(token)
        assert result is not None
        _auth, _sid, scope = result
        assert scope == "chat"

    def test_missing_scope_on_disk_defaults_to_chat(self, tmp_path, monkeypatch, store_path):
        """A Phase-1 record without a 'scope' field reads as 'chat' (secure default)."""
        import json

        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        token = store.mint("speaker0", "OldDevice")

        # Directly strip the scope field from the on-disk JSON to simulate a
        # Phase-1 record minted before the scope field existed.
        from paramem.backup.encryption import read_maybe_encrypted, write_infra_bytes

        raw = json.loads(read_maybe_encrypted(store_path).decode("utf-8"))
        for entry in raw["tokens"].values():
            entry.pop("scope", None)
        write_infra_bytes(store_path, json.dumps(raw).encode("utf-8"))

        # Load a fresh store (simulates a server restart reading an old file).
        store2 = UserTokenStore(store_path)
        result = store2.resolve(token)
        assert result is not None
        _auth, _sid, scope = result
        assert scope == "chat", f"Expected 'chat' for Phase-1 record, got {scope!r}"

    def test_unattributed_token_resolve_returns_none_speaker(
        self, tmp_path, monkeypatch, store_path
    ):
        """Unattributed token (speaker_id=None) → resolve() returns (True, None, scope)."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        token = store.mint(None, "Shared Device", scope="chat")

        result = store.resolve(token)
        assert result is not None
        authenticated, speaker_id, scope = result
        assert authenticated is True
        assert speaker_id is None
        assert scope == "chat"

    def test_unattributed_token_distinguishable_from_unknown(
        self, tmp_path, monkeypatch, store_path
    ):
        """An unattributed token is distinguishable from an unknown token via resolve()."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        _token = store.mint(None, "Shared Device", scope="chat")

        # An unknown token returns None, not (True, None, ...).
        result = store.resolve("completely-bogus-token-value")
        assert result is None

    def test_invalid_scope_raises_value_error(self, tmp_path, monkeypatch, store_path):
        """mint(..., scope='bogus') raises ValueError."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)

        with pytest.raises(ValueError, match="Invalid scope"):
            store.mint("speaker0", "Device", scope="bogus")

    def test_list_includes_scope(self, tmp_path, monkeypatch, store_path):
        """list() entries include a 'scope' key."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        store.mint("speaker0", "Admin iPad", scope="admin")
        store.mint("speaker1", "Chat Phone", scope="chat")

        entries = store.list()
        assert len(entries) == 2
        scopes = {e["scope"] for e in entries}
        assert scopes == {"admin", "chat"}

    def test_list_phase1_record_surfaces_chat(self, tmp_path, monkeypatch, store_path):
        """list() surfaces 'chat' scope for Phase-1 records without a scope field."""
        import json

        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        store.mint("speaker0", "OldDevice")

        from paramem.backup.encryption import read_maybe_encrypted, write_infra_bytes

        raw = json.loads(read_maybe_encrypted(store_path).decode("utf-8"))
        for entry in raw["tokens"].values():
            entry.pop("scope", None)
        write_infra_bytes(store_path, json.dumps(raw).encode("utf-8"))

        store2 = UserTokenStore(store_path)
        entries = store2.list()
        assert len(entries) == 1
        assert entries[0]["scope"] == "chat"

    def test_lookup_unchanged_contract_unattributed_returns_none(
        self, tmp_path, monkeypatch, store_path
    ):
        """lookup() returns None for an unattributed token (preserves str|None contract)."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        token = store.mint(None, "Shared Device")

        # lookup() returns None for unattributed (speaker_id is None).
        assert store.lookup(token) is None

    def test_revoke_speaker_none_raises(self, tmp_path, monkeypatch, store_path):
        """revoke_speaker(None) raises ValueError (prevents bulk-revoke of unattributed tokens)."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)

        with pytest.raises(ValueError, match="revoke_speaker.*None"):
            store.revoke_speaker(None)  # type: ignore[arg-type]

    def test_revoke_speaker_skips_unattributed_entries(self, tmp_path, monkeypatch, store_path):
        """revoke_speaker() skips entries whose speaker_id is None."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        unattributed_token = store.mint(None, "Shared Device")
        attributed_token = store.mint("speaker0", "Personal Phone")

        # Revoke speaker0 — should NOT revoke the unattributed token.
        count = store.revoke_speaker("speaker0")
        assert count == 1
        # The unattributed token remains valid.
        result = store.resolve(unattributed_token)
        assert result is not None, "Unattributed token should not be revoked by revoke_speaker"
        # The attributed token is revoked.
        assert store.lookup(attributed_token) is None

    def test_revoke_label_revokes_unattributed_token(self, tmp_path, monkeypatch, store_path):
        """revoke_label() is the correct revocation path for unattributed tokens."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        token = store.mint(None, "Shared Kitchen Tablet")

        count = store.revoke_label("Shared Kitchen Tablet")
        assert count == 1
        assert store.resolve(token) is None


# ---------------------------------------------------------------------------
# Live-reload (mtime-triggered)
# ---------------------------------------------------------------------------


class TestLiveReload:
    def test_second_store_sees_new_token_after_mtime_change(
        self, tmp_path, monkeypatch, store_path
    ):
        """store_b loaded before mint; after mtime advances, store_b.resolve() sees the new token.

        Two UserTokenStore instances on the same file path (simulating the
        running server's store vs. a CLI mint out of process).
        """
        import os
        import time

        _setup_daily(tmp_path, monkeypatch)
        store_a = UserTokenStore(store_path)
        # store_b is loaded before any mint — empty.
        store_b = UserTokenStore(store_path)

        token = store_a.mint("speaker0", "New Device")

        # Ensure the mtime actually advanced by at least 1 ns.
        # On most filesystems write_infra_bytes does an atomic replace so
        # mtime updates are guaranteed; still bump if same second.
        new_mtime = store_a._current_mtime()
        if new_mtime == store_b._mtime:
            # Force a distinct mtime via os.utime.
            time.sleep(0.01)
            os.utime(str(store_path), None)

        # store_b's resolve() should trigger mtime reload and find the token.
        result = store_b.resolve(token)
        assert result is not None, (
            "store_b.resolve() must find the token after mtime-triggered reload "
            "(live-reload not working)"
        )

    def test_second_store_sees_revocation_after_mtime_change(
        self, tmp_path, monkeypatch, store_path
    ):
        """store_b loaded before revoke; resolve() returns None after mtime reload."""
        import os
        import time

        _setup_daily(tmp_path, monkeypatch)
        store_a = UserTokenStore(store_path)
        token = store_a.mint("speaker0", "Device")

        store_b = UserTokenStore(store_path)
        # Both stores currently see the token.
        assert store_b.resolve(token) is not None

        store_a.revoke_token(token)

        # Ensure mtime advanced.
        new_mtime = store_a._current_mtime()
        if new_mtime == store_b._mtime:
            time.sleep(0.01)
            os.utime(str(store_path), None)

        # store_b must now reflect the revocation.
        result = store_b.resolve(token)
        assert result is None, "store_b.resolve() must return None after revocation + reload"

    def test_self_write_does_not_trigger_spurious_reload(self, tmp_path, monkeypatch, store_path):
        """After an in-process mint, the mtime is stamped so the next resolve
        is NOT a reload (verify _mtime matches disk after _save)."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        token = store.mint("speaker0", "Device")

        # _mtime should equal the current disk mtime (no spurious reload).
        disk_mtime = store._current_mtime()
        assert store._mtime == disk_mtime, (
            "_mtime not stamped after _save — would cause spurious reload on next resolve()"
        )

        # Calling resolve again must not error.
        result = store.resolve(token)
        assert result is not None

    def test_missing_file_midlife_is_noop(self, tmp_path, monkeypatch, store_path):
        """If the store file disappears, _maybe_reload is a no-op and RAM state is kept."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)
        token = store.mint("speaker0", "Device")

        # Delete the file.
        store_path.unlink()

        # resolve() should not crash and should still find the token in RAM.
        result = store.resolve(token)
        assert result is not None, (
            "resolve() must not crash or lose RAM state when the file disappears"
        )


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
                store.mint("speaker0", "TestDevice")

        # After the guard fires, in-memory state must be cleared — the minted
        # token must not authenticate even though it was inserted into _tokens
        # before _save() raised.
        assert store.lookup(known_token) is None, (
            "lookup() must return None after TOCTOU guard clears in-memory state"
        )
        assert store.list() == [], "list() must return [] after TOCTOU guard clears in-memory state"


# ---------------------------------------------------------------------------
# Speaker-id canonicality enforcement at mint boundary (strict REJECT)
# ---------------------------------------------------------------------------


class TestMintSpeakerIdValidation:
    """mint() raises ValueError on a non-canonical speaker_id (strict enforcement).

    The invariant is absolute: every attributed token MUST reference a
    canonical ``speaker{N}`` id.  A non-conforming non-None id raises
    ``ValueError`` — no warn-and-allow path exists.
    """

    def test_mint_raises_on_non_speaker_id(self, tmp_path, monkeypatch, store_path):
        """mint() raises ValueError when speaker_id is a non-speaker{N} string."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)

        with pytest.raises(ValueError, match="speaker{N}"):
            store.mint("legacyuuid12", "Legacy Device")

        # No token must have been stored — the store stays clean.
        assert store.list() == []

    def test_mint_none_speaker_id_succeeds(self, tmp_path, monkeypatch, store_path):
        """mint(None, ...) succeeds — None is a valid sentinel for unattributed tokens."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)

        token = store.mint(None, "Shared Device")
        assert token is not None
        result = store.resolve(token)
        assert result is not None
        _authenticated, speaker_id, _scope = result
        assert speaker_id is None

    def test_mint_canonical_speaker_id_succeeds(self, tmp_path, monkeypatch, store_path):
        """mint('speaker0', ...) succeeds — speaker{N} is the canonical form."""
        _setup_daily(tmp_path, monkeypatch)
        store = UserTokenStore(store_path)

        token = store.mint("speaker0", "My Device")
        assert token is not None
        assert store.lookup(token) == "speaker0"


# ---------------------------------------------------------------------------
# v1 store raises; v2 store loads normally
# ---------------------------------------------------------------------------


class TestTokenStoreMigrationV1ToV2:
    """v1 store raises ValueError (migration retired); v2 store loads normally."""

    def test_v1_store_raises(self, tmp_path):
        """A v1 store raises ValueError — migration rung has been removed."""
        import json

        store_path = tmp_path / "tokens.json"
        fake_hash = "a" * 64
        v1_payload = {
            "version": 1,
            "tokens": {
                fake_hash: {
                    "speaker_id": "Speaker0",
                    "label": "Test Device",
                    "created": "2026-01-01T00:00:00+00:00",
                    "revoked": False,
                    "scope": "chat",
                }
            },
        }
        store_path.write_text(json.dumps(v1_payload))

        import pytest

        with pytest.raises(ValueError, match="Unsupported user-token store version"):
            UserTokenStore(store_path)

    def test_v2_store_loads_normally(self, tmp_path):
        """A v2 store loads without raising."""
        import json

        store_path = tmp_path / "tokens.json"
        fake_hash = "c" * 64
        v2_payload = {
            "version": 2,
            "tokens": {
                fake_hash: {
                    "speaker_id": "speaker0",
                    "label": "Device",
                    "created": "2026-01-01T00:00:00+00:00",
                    "revoked": False,
                    "scope": "chat",
                }
            },
        }
        store_path.write_text(json.dumps(v2_payload))

        store = UserTokenStore(store_path)

        entry = store._tokens.get(fake_hash)
        assert entry is not None
        assert entry["speaker_id"] == "speaker0"


# ---------------------------------------------------------------------------
# Dynamic binding regression — _save must consult key_store at call time
# ---------------------------------------------------------------------------


class TestSaveConsultsDynamicKeyStorePath:
    """_save() must read key_store.DAILY_KEY_PATH_DEFAULT at call time (dynamic binding).

    Regression guard: before the fix, user_tokens.py imported
    ``DAILY_KEY_PATH_DEFAULT`` and ``daily_identity_loadable`` at module level,
    freezing the values at import time.  A runtime monkeypatch of
    ``paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT`` was invisible to _save.

    The spy form used here is deterministic regardless of whether a real
    ``~/.config/paramem/daily_key.age`` exists on the host.

    ``write_infra_bytes`` is patched to a no-op at the ``user_tokens`` module
    boundary so that its own internal dynamic call to ``daily_identity_loadable``
    (in ``encryption.py``) does not reach the spy.  Only the call originating
    inside ``_save`` itself is visible to the spy — which is the bug site.
    """

    def test_save_calls_daily_identity_loadable_with_current_key_store_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, store_path: Path, no_daily_key
    ):
        """_save() passes the sentinel path from key_store to daily_identity_loadable.

        Procedure:
        1. Monkeypatch ``paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT`` to a
           sentinel Path — tests that the path is read at call time.
        2. Monkeypatch ``paramem.backup.key_store.daily_identity_loadable`` to a
           spy that records all Path arguments it is called with.
        3. Monkeypatch ``paramem.server.user_tokens.write_infra_bytes`` to a
           no-op so that ``encryption.py``'s own dynamic call to
           ``daily_identity_loadable`` inside ``write_infra_bytes`` is
           suppressed — isolating only the call inside ``_save``.
        4. Trigger ``_save()`` via ``mint()``.
        5. Assert the spy was called exactly once and received the sentinel path.
           With the old frozen-import code the spy would not be called (the
           frozen reference bypasses the patched module attribute), so the first
           assertion would fail → RED before fix, GREEN after.
        """
        import paramem.backup.key_store as _ks

        sentinel_path = tmp_path / "sentinel_key.age"
        monkeypatch.setattr(_ks, "DAILY_KEY_PATH_DEFAULT", sentinel_path)

        received_paths: list[Path] = []

        def _spy(path: Path) -> bool:
            received_paths.append(path)
            return False  # Security OFF — no encryption, no TOCTOU guard

        monkeypatch.setattr(_ks, "daily_identity_loadable", _spy)

        # Suppress write_infra_bytes at the user_tokens call site so that
        # encryption.py's internal dynamic call to daily_identity_loadable
        # does not pollute received_paths.
        monkeypatch.setattr(
            "paramem.server.user_tokens.write_infra_bytes",
            lambda path, data: None,
        )

        store = UserTokenStore(store_path)
        store.mint("speaker0", "Regression Device")

        assert received_paths, (
            "_save() did not call daily_identity_loadable via key_store — "
            "the dynamic binding (from paramem.backup import key_store as _ks) is absent"
        )
        assert received_paths[0] == sentinel_path, (
            f"_save() called daily_identity_loadable with {received_paths[0]!r} "
            f"instead of the sentinel {sentinel_path!r} — "
            "frozen import (_save reads a copy captured at import time) is not fixed"
        )
