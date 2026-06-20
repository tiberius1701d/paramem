"""Tests for POST /ingest-sessions and POST /ingest-sessions/cancel endpoints.

Pattern: monkeypatch ``paramem.server.app._state`` with a minimal state dict,
use TestClient without lifespan (no model load, no GPU required).
"""

from __future__ import annotations

import base64
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

    buffer = SessionBuffer(
        session_dir=sessions_dir, state_dir=sessions_dir.parent / "state", debug=False
    )

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


_SAMPLE_CONTENT = b"This is a sample document for testing."
_SAMPLE_FILENAME = "notes.md"
_SAMPLE_B64 = base64.b64encode(_SAMPLE_CONTENT).decode("ascii")


def _ingest_payload(
    speaker_id: str,
    text: str = "Hello world chunk.",
    n: int = 1,
    document_filename: str = _SAMPLE_FILENAME,
    document_b64: str = _SAMPLE_B64,
) -> dict:
    return {
        "speaker_id": speaker_id,
        "document_filename": document_filename,
        "document_b64": document_b64,
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

    def test_doc_id_returned(self, client, state):
        """Response carries a non-empty doc_id."""
        spk = state["_test_known_speaker_id"]
        resp = client.post("/ingest-sessions", json=_ingest_payload(spk, n=2))
        body = resp.json()
        assert body["doc_id"]
        assert body["doc_id"].startswith("doc-")

    def test_chunk_session_ids_match_doc_id(self, client, state):
        """Each queued session id is '<doc_id>-c<chunk_index:03d>'."""
        spk = state["_test_known_speaker_id"]
        resp = client.post("/ingest-sessions", json=_ingest_payload(spk, n=3))
        body = resp.json()
        doc_id = body["doc_id"]
        expected = [f"{doc_id}-c000", f"{doc_id}-c001", f"{doc_id}-c002"]
        assert sorted(body["queued"]) == sorted(expected)

    def test_two_requests_get_distinct_doc_ids(self, client, state):
        """Two separate ingest requests produce different doc_ids."""
        spk = state["_test_known_speaker_id"]
        r1 = client.post("/ingest-sessions", json=_ingest_payload(spk, n=1))
        r2 = client.post("/ingest-sessions", json=_ingest_payload(spk, n=1))
        assert r1.json()["doc_id"] != r2.json()["doc_id"]

    def test_origdoc_stored_in_session_dir(self, client, state):
        """After ingest, a <doc_id>.origdoc file exists in session_dir."""
        spk = state["_test_known_speaker_id"]
        resp = client.post("/ingest-sessions", json=_ingest_payload(spk, n=1))
        doc_id = resp.json()["doc_id"]
        buf: SessionBuffer = state["session_buffer"]
        origdoc = buf.session_dir / f"{doc_id}.origdoc"
        assert origdoc.exists()
        assert origdoc.read_bytes() == _SAMPLE_CONTENT

    def test_get_pending_glob_skips_origdoc(self, client, state):
        """get_pending() does not pick up .origdoc files as sessions."""
        spk = state["_test_known_speaker_id"]
        resp = client.post("/ingest-sessions", json=_ingest_payload(spk, n=1))
        buf: SessionBuffer = state["session_buffer"]
        # Verify the .origdoc file is present
        doc_id = resp.json()["doc_id"]
        assert (buf.session_dir / f"{doc_id}.origdoc").exists()
        # get_pending only returns the one chunk session, not the origdoc
        assert len(buf.get_pending()) == 1

    def test_chunk_metadata_carries_doc_id_and_chunk_count(self, client, state):
        """Each chunk turn's metadata includes doc_id and chunk_count."""
        spk = state["_test_known_speaker_id"]
        resp = client.post("/ingest-sessions", json=_ingest_payload(spk, n=2))
        doc_id = resp.json()["doc_id"]
        buf: SessionBuffer = state["session_buffer"]
        for sid in resp.json()["queued"]:
            turns = buf._turns[sid]
            meta = turns[0]["metadata"]
            assert meta["doc_id"] == doc_id
            assert meta["chunk_count"] == 2
            assert meta["doc_filename"] == _SAMPLE_FILENAME

    def test_reingest_same_content_queues_fresh(self, client, state):
        """Re-ingesting the same content creates new sessions (no dedup)."""
        spk = state["_test_known_speaker_id"]
        payload = _ingest_payload(spk, n=2)
        r1 = client.post("/ingest-sessions", json=payload)
        r2 = client.post("/ingest-sessions", json=payload)
        assert r1.status_code == 200
        assert r2.status_code == 200
        # Both should return queued sessions (not skipped)
        assert len(r1.json()["queued"]) == 2
        assert len(r2.json()["queued"]) == 2
        # Different doc_ids for each request
        assert r1.json()["doc_id"] != r2.json()["doc_id"]


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


class TestOversizedDocument400:
    def test_oversized_document_b64_returns_400(self, client, state):
        """document_b64 whose decoded size exceeds 25 MiB → HTTP 400."""
        spk = state["_test_known_speaker_id"]
        # Build a payload that decodes to 25 MiB + 1 byte.
        big_bytes = b"x" * (25 * 1024 * 1024 + 1)
        big_b64 = base64.b64encode(big_bytes).decode("ascii")
        payload = _ingest_payload(spk, n=1, document_b64=big_b64)
        resp = client.post("/ingest-sessions", json=payload)
        assert resp.status_code == 400
        assert resp.json()["error"] == "document_too_large"

    def test_oversized_document_nothing_queued(self, client, state):
        """No sessions are queued when the document is too large."""
        spk = state["_test_known_speaker_id"]
        big_bytes = b"x" * (25 * 1024 * 1024 + 1)
        big_b64 = base64.b64encode(big_bytes).decode("ascii")
        payload = _ingest_payload(spk, n=1, document_b64=big_b64)
        client.post("/ingest-sessions", json=payload)
        buf: SessionBuffer = state["session_buffer"]
        assert buf.get_pending() == []


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

    def test_cancel_also_removes_origdoc(self, client, state):
        """Cancelling all chunks for a doc removes the origdoc blob."""
        spk = state["_test_known_speaker_id"]
        resp = client.post("/ingest-sessions", json=_ingest_payload(spk, n=2))
        doc_id = resp.json()["doc_id"]
        queued = resp.json()["queued"]

        buf: SessionBuffer = state["session_buffer"]
        assert (buf.session_dir / f"{doc_id}.origdoc").exists()

        client.post("/ingest-sessions/cancel", json={"session_ids": queued})

        assert not (buf.session_dir / f"{doc_id}.origdoc").exists()

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

        def spy_mark(session_ids, **kwargs):
            mark_calls.append(session_ids)
            return original_mark(session_ids, **kwargs)

        buf.discard_sessions = spy_discard
        buf.mark_consolidated = spy_mark

        client.post("/ingest-sessions/cancel", json={"session_ids": queued})

        assert len(discard_calls) == 1
        assert mark_calls == []

    def test_cancel_not_found_returns_in_response(self, client, state):
        """Session ids not in the queue appear in not_found list."""
        cancel_resp = client.post(
            "/ingest-sessions/cancel",
            json={"session_ids": ["doc-aabbccdd-c000"]},
        )
        assert cancel_resp.status_code == 200
        body = cancel_resp.json()
        assert body["cancelled"] == []
        assert "doc-aabbccdd-c000" in body["not_found"]

    def test_cancel_partial_found(self, client, state):
        """Some ids found, some not — correctly split into cancelled / not_found."""
        spk = state["_test_known_speaker_id"]
        resp = client.post("/ingest-sessions", json=_ingest_payload(spk, n=1))
        real_id = resp.json()["queued"][0]

        cancel_resp = client.post(
            "/ingest-sessions/cancel",
            json={"session_ids": [real_id, "doc-not-real-c000"]},
        )
        body = cancel_resp.json()
        assert real_id in body["cancelled"]
        assert "doc-not-real-c000" in body["not_found"]


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
