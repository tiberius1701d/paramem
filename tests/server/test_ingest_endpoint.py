"""Tests for POST /ingest-sessions and POST /ingest-sessions/cancel endpoints.

Pattern: monkeypatch ``paramem.server.app._state`` with a minimal state dict,
use TestClient without lifespan (no model load, no GPU required).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import paramem.server.app as app_module
from paramem.server.session_buffer import SessionBuffer

# ---------------------------------------------------------------------------
# State factory
# ---------------------------------------------------------------------------


def _make_state(tmp_path: Path) -> dict:
    """Build a minimal _state dict for ingest endpoint tests."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    config = MagicMock()
    config.paths.sessions = sessions_dir
    config.debug = False

    buffer = SessionBuffer(session_dir=sessions_dir, debug=False)

    # Build a real SpeakerStore with one known speaker.
    from paramem.server.speaker import SpeakerStore

    store = SpeakerStore(tmp_path / "profiles.json")
    known_speaker_id = store.enroll("Alice", [0.1, 0.2, 0.3])

    return {
        "model": None,
        "config": config,
        "consolidating": False,
        "migration": {},  # no TRIAL
        "server_started_at": "2026-04-26T00:00:00+00:00",
        "session_buffer": buffer,
        "speaker_store": store,
        # Store the known speaker id for use in tests.
        "_test_known_speaker_id": known_speaker_id,
    }


@pytest.fixture()
def state(tmp_path, monkeypatch):
    fresh = _make_state(tmp_path)
    monkeypatch.setattr(app_module, "_state", fresh)
    return fresh


@pytest.fixture()
def client(state):
    return TestClient(app_module.app, raise_server_exceptions=False)


def _ingest_payload(speaker_id: str, text: str = "Hello world chunk.", n: int = 1) -> dict:
    return {
        "speaker_id": speaker_id,
        "sessions": [
            {
                "source": "/docs/notes.md",
                "chunk": f"{text} (chunk {i})" if n > 1 else text,
                "chunk_index": i,
                "source_type": "document",
                "doc_title": "notes",
            }
            for i in range(n)
        ],
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestKnownSpeakerAppends:
    def test_two_chunks_queued(self, client, state):
        """Two chunks for a known speaker → both queued in SessionBuffer."""
        spk = state["_test_known_speaker_id"]
        resp = client.post("/ingest-sessions", json=_ingest_payload(spk, n=2))
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_chunks"] == 2
        assert body["registry_skipped"] == 0
        assert len(body["queued"]) == 2
        assert not body["rejected_unknown_speaker"]
        assert not body["rejected_no_speaker_id"]

    def test_source_type_document_in_pending(self, client, state):
        """Queued chunks appear in get_pending() with source_type='document'."""
        spk = state["_test_known_speaker_id"]
        client.post("/ingest-sessions", json=_ingest_payload(spk, n=1))

        buf: SessionBuffer = state["session_buffer"]
        pending = buf.get_pending()
        assert len(pending) == 1
        assert pending[0]["source_type"] == "document"
        assert pending[0]["doc_title"] == "notes"

    def test_session_ids_prefixed_doc(self, client, state):
        """Returned session ids start with 'doc-'."""
        spk = state["_test_known_speaker_id"]
        resp = client.post("/ingest-sessions", json=_ingest_payload(spk, n=1))
        queued = resp.json()["queued"]
        assert all(sid.startswith("doc-") for sid in queued)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestUnknownSpeaker404:
    def test_unknown_speaker_returns_404(self, client, state):
        resp = client.post("/ingest-sessions", json=_ingest_payload("unknown-xyz"))
        assert resp.status_code == 404
        body = resp.json()
        assert body["rejected_unknown_speaker"] is True
        assert body["queued"] == []

    def test_unknown_speaker_nothing_queued(self, client, state):
        client.post("/ingest-sessions", json=_ingest_payload("unknown-xyz"))
        buf: SessionBuffer = state["session_buffer"]
        assert buf.get_pending() == []


class TestEmptySpeakerId400:
    def test_empty_speaker_id_returns_400(self, client, state):
        resp = client.post("/ingest-sessions", json=_ingest_payload(""))
        assert resp.status_code == 400
        body = resp.json()
        assert body["rejected_no_speaker_id"] is True
        assert body["queued"] == []

    def test_empty_speaker_id_nothing_queued(self, client, state):
        client.post("/ingest-sessions", json=_ingest_payload(""))
        buf: SessionBuffer = state["session_buffer"]
        assert buf.get_pending() == []


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestRegistryIdempotent:
    def test_second_post_skips_all_chunks(self, client, state):
        """Posting the same payload twice: second call skips all chunks."""
        spk = state["_test_known_speaker_id"]
        payload = _ingest_payload(spk, n=2)

        r1 = client.post("/ingest-sessions", json=payload)
        r2 = client.post("/ingest-sessions", json=payload)

        assert r1.status_code == 200
        assert r2.status_code == 200

        b1 = r1.json()
        b2 = r2.json()

        assert b1["registry_skipped"] == 0
        assert b2["registry_skipped"] == 2
        assert b2["queued"] == []

    def test_queue_size_unchanged_after_second_post(self, client, state):
        """Queue size does not grow on the second identical post."""
        spk = state["_test_known_speaker_id"]
        payload = _ingest_payload(spk, n=2)

        client.post("/ingest-sessions", json=payload)
        buf: SessionBuffer = state["session_buffer"]
        size_after_first = len(buf.get_pending())

        client.post("/ingest-sessions", json=payload)
        size_after_second = len(buf.get_pending())

        assert size_after_second == size_after_first


# ---------------------------------------------------------------------------
# /status pending_documents counter
# ---------------------------------------------------------------------------


class TestStatusPendingDocumentsCounter:
    def test_status_pending_documents_after_ingest(self, client, state):
        """POST /ingest-sessions with 2 chunks → pending_documents=2 in buffer summary.

        The full /status endpoint requires heavy mocking of many unrelated subsystems
        (systemd_timer, TTS, STT, backup, …) which is out of scope for this test.
        Instead we verify the data at the layer the /status handler reads from:
        ``session_buffer.get_summary()["per_source_type"]``.
        """
        spk = state["_test_known_speaker_id"]
        client.post("/ingest-sessions", json=_ingest_payload(spk, n=2))

        buf: SessionBuffer = state["session_buffer"]
        per_source = buf.get_summary()["per_source_type"]
        assert per_source.get("document", 0) == 2
        assert per_source.get("transcript", 0) == 0


# ---------------------------------------------------------------------------
# Cancel endpoint
# ---------------------------------------------------------------------------


class TestCancelClearsQueueViaDiscardSessions:
    def test_cancel_removes_queued_sessions(self, client, state):
        """POST /ingest-sessions/cancel removes the given session ids."""
        spk = state["_test_known_speaker_id"]
        resp = client.post("/ingest-sessions", json=_ingest_payload(spk, n=2))
        queued = resp.json()["queued"]
        assert len(queued) == 2

        cancel_resp = client.post("/ingest-sessions/cancel", json={"session_ids": queued})
        assert cancel_resp.status_code == 200
        body = cancel_resp.json()
        assert sorted(body["cancelled"]) == sorted(queued)
        assert body["not_found"] == []

        buf: SessionBuffer = state["session_buffer"]
        assert buf.get_pending() == []

    def test_cancel_calls_discard_sessions_not_mark_consolidated(self, client, state, monkeypatch):
        """cancel calls discard_sessions, NOT mark_consolidated."""
        spk = state["_test_known_speaker_id"]
        resp = client.post("/ingest-sessions", json=_ingest_payload(spk, n=1))
        queued = resp.json()["queued"]

        buf: SessionBuffer = state["session_buffer"]
        discard_calls: list = []
        mark_calls: list = []

        original_discard = buf.discard_sessions
        original_mark = buf.mark_consolidated

        def spy_discard(session_ids):
            discard_calls.append(session_ids)
            return original_discard(session_ids)

        def spy_mark(session_ids):
            mark_calls.append(session_ids)
            return original_mark(session_ids)

        buf.discard_sessions = spy_discard
        buf.mark_consolidated = spy_mark

        client.post("/ingest-sessions/cancel", json={"session_ids": queued})

        assert len(discard_calls) == 1
        assert mark_calls == []

    def test_cancel_not_found_returns_in_response(self, client, state):
        """Session ids not in the queue appear in not_found list."""
        cancel_resp = client.post(
            "/ingest-sessions/cancel",
            json={"session_ids": ["doc-aabbccdd"]},
        )
        assert cancel_resp.status_code == 200
        body = cancel_resp.json()
        assert body["cancelled"] == []
        assert "doc-aabbccdd" in body["not_found"]

    def test_cancel_partial_found(self, client, state):
        """Some ids found, some not — correctly split into cancelled / not_found."""
        spk = state["_test_known_speaker_id"]
        resp = client.post("/ingest-sessions", json=_ingest_payload(spk, n=1))
        real_id = resp.json()["queued"][0]

        cancel_resp = client.post(
            "/ingest-sessions/cancel",
            json={"session_ids": [real_id, "doc-not-real"]},
        )
        body = cancel_resp.json()
        assert real_id in body["cancelled"]
        assert "doc-not-real" in body["not_found"]


# ---------------------------------------------------------------------------
# 409 trial-active gate
# ---------------------------------------------------------------------------


class TestTrialActive409:
    def test_ingest_sessions_409_when_trial_active(self, state, monkeypatch):
        """POST /ingest-sessions returns 409 trial_active when migration TRIAL is active.

        The gate is in the endpoint body after speaker validation.  The request
        uses a known speaker so the gate is reached; we then assert that the
        TRIAL state is enforced before any chunk processing occurs.
        """
        state["migration"] = {"state": "TRIAL"}
        client = TestClient(app_module.app, raise_server_exceptions=False)
        spk = state["_test_known_speaker_id"]

        resp = client.post("/ingest-sessions", json=_ingest_payload(spk, n=1))

        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "trial_active"
