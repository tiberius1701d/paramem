"""Unit tests for the ``auth_speaker_id`` parameter of ``_resolve_speaker``.

Verifies:
- When *auth_speaker_id* is set and the speaker store knows that ID, the
  authenticated identity is returned immediately (authoritative priority 0).
- When *auth_speaker_id* is set but the speaker store is absent (None), the
  authenticated ID is still returned (no store required).
- When *auth_speaker_id* is set and the store does not know the ID (e.g. a
  newly-minted token for a speaker not yet profiled), the ID is still returned
  without a name (no store override of the authenticated identity).
- When *auth_speaker_id* is None, the existing voice-embedding / session /
  anonymous resolution path is used unchanged.

CPU-only — no model load.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from paramem.server.app import _resolve_speaker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_buffer(speaker_id: str | None = None, speaker_name: str | None = None):
    """Return a minimal mock SessionBuffer."""
    buf = MagicMock()
    buf.get_speaker_id.return_value = speaker_id
    buf.get_speaker.return_value = speaker_name
    return buf


def _make_store(known_ids: dict[str, str | None] | None = None):
    """Return a minimal mock SpeakerStore.

    *known_ids* maps speaker_id → display_name (None when not profiled).
    ``match`` always returns a tentative/no-match result so it never fires.

    Both ``get_name`` and ``resolve_speaker_name`` are wired to the same
    lookup so tests that check the auth-speaker-id path (which calls
    ``resolve_speaker_name``) and tests that check other paths (which may
    call ``get_name``) both work correctly.
    """
    store = MagicMock()

    def _get_name(sid):
        if known_ids is None:
            return None
        return known_ids.get(sid)

    store.get_name.side_effect = _get_name
    # _resolve_speaker now calls resolve_speaker_name (P3) for the
    # auth-speaker-id path (app.py:3549). Wire the same lookup so mocks
    # that rely on known_ids still work.
    store.resolve_speaker_name.side_effect = _get_name

    # match returns a tentative non-match (voice path should not fire)
    match_result = MagicMock()
    match_result.speaker_id = None
    match_result.tentative = True
    store.match.return_value = match_result

    return store


def _make_request(embedding=None, conversation_id="default"):
    req = MagicMock()
    req.speaker_embedding = embedding
    req.conversation_id = conversation_id
    return req


# ---------------------------------------------------------------------------
# auth_speaker_id authoritative path (priority 0)
# ---------------------------------------------------------------------------


class TestAuthSpeakerIdAuthoritative:
    def test_returns_auth_id_and_name_when_store_knows_id(self):
        """auth_speaker_id set + store has profile → returns (id, name)."""
        store = _make_store({"Speaker0": "Mara"})
        buf = _make_buffer()
        req = _make_request()

        sid, name = _resolve_speaker(req, buf, store, auth_speaker_id="Speaker0")

        assert sid == "Speaker0"
        assert name == "Mara"

    def test_returns_auth_id_with_none_name_when_store_has_no_profile(self):
        """auth_speaker_id set + store has no profile for that ID → (id, None)."""
        store = _make_store({})  # empty — no profiles
        buf = _make_buffer()
        req = _make_request()

        sid, name = _resolve_speaker(req, buf, store, auth_speaker_id="Speaker0")

        assert sid == "Speaker0"
        assert name is None

    def test_returns_auth_id_with_none_name_when_store_is_absent(self):
        """auth_speaker_id set + speaker_store=None → (id, None) without crashing."""
        buf = _make_buffer()
        req = _make_request()

        sid, name = _resolve_speaker(req, buf, speaker_store=None, auth_speaker_id="Speaker0")

        assert sid == "Speaker0"
        assert name is None

    def test_auth_id_overrides_voice_embedding(self):
        """auth_speaker_id must not be overridden by a voice match.

        Even when the request carries a speaker embedding that would match a
        different speaker via voice, the authenticated token identity wins.
        """
        # Store: auth ID "Speaker0" (name "Mara"); voice match would return
        # "Speaker1" if the embedding branch ran — but it must not run.
        store = MagicMock()
        store.get_name.return_value = "Mara"
        # _resolve_speaker now calls resolve_speaker_name (P3) for the auth path.
        store.resolve_speaker_name.return_value = "Mara"

        voice_match = MagicMock()
        voice_match.speaker_id = "Speaker1"
        voice_match.tentative = False
        store.match.return_value = voice_match

        buf = _make_buffer()
        req = _make_request(embedding=[0.1, 0.2, 0.3])

        sid, name = _resolve_speaker(req, buf, store, auth_speaker_id="Speaker0")

        assert sid == "Speaker0"
        assert name == "Mara"
        # store.match must never have been called (voice path skipped)
        store.match.assert_not_called()

    def test_auth_id_overrides_session_history(self):
        """auth_speaker_id overrides a previously identified session speaker."""
        store = _make_store({"Speaker0": "Mara"})
        # Buffer has a different speaker from an earlier turn.
        buf = _make_buffer(speaker_id="Speaker1", speaker_name="Alice")
        req = _make_request()

        sid, name = _resolve_speaker(req, buf, store, auth_speaker_id="Speaker0")

        assert sid == "Speaker0"
        assert name == "Mara"


# ---------------------------------------------------------------------------
# auth_speaker_id=None falls through to existing logic
# ---------------------------------------------------------------------------


class TestNoAuthSpeakerIdFallthrough:
    def test_none_auth_uses_voice_embedding(self):
        """auth_speaker_id=None → voice embedding branch runs normally."""
        store = MagicMock()
        voice_match = MagicMock()
        voice_match.speaker_id = "Speaker1"
        voice_match.name = "Alice"
        voice_match.tentative = False
        store.match.return_value = voice_match

        buf = _make_buffer()
        req = _make_request(embedding=[0.1, 0.2])

        sid, name = _resolve_speaker(req, buf, store, auth_speaker_id=None)

        assert sid == "Speaker1"
        assert name == "Alice"

    def test_none_auth_uses_session_history(self):
        """auth_speaker_id=None, no embedding → session history used."""
        store = _make_store()
        buf = _make_buffer(speaker_id="Speaker2", speaker_name="Bob")
        req = _make_request(embedding=None)

        sid, name = _resolve_speaker(req, buf, store, auth_speaker_id=None)

        assert sid == "Speaker2"
        assert name == "Bob"

    def test_none_auth_returns_anonymous_when_no_signals(self):
        """auth_speaker_id=None, no embedding, no session → (None, None)."""
        store = _make_store()
        buf = _make_buffer(speaker_id=None, speaker_name=None)
        req = _make_request(embedding=None)

        sid, name = _resolve_speaker(req, buf, store, auth_speaker_id=None)

        assert sid is None
        assert name is None

    def test_default_auth_speaker_id_is_none(self):
        """Calling _resolve_speaker without auth_speaker_id defaults to None (fallthrough)."""
        store = _make_store()
        buf = _make_buffer(speaker_id=None)
        req = _make_request(embedding=None)

        # Omit auth_speaker_id entirely — must behave as if None.
        sid, name = _resolve_speaker(req, buf, store)

        assert sid is None
        assert name is None
