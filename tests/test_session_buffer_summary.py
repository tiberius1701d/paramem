"""Unit tests for SessionBuffer.get_summary, metadata propagation, and discard_sessions."""

import pytest

from paramem.server.session_buffer import SessionBuffer


@pytest.fixture
def buf(tmp_path):
    return SessionBuffer(session_dir=tmp_path / "sessions", debug=False)


@pytest.fixture
def buf_debug(tmp_path):
    return SessionBuffer(session_dir=tmp_path / "sessions", debug=True)


def test_summary_empty(buf):
    s = buf.get_summary()
    assert s == {
        "total": 0,
        "orphaned": 0,
        "oldest_age_seconds": None,
        "per_speaker": {},
        "per_source_type": {},
    }


def test_summary_orphaned_session(buf):
    buf.append("conv-1", "user", "hello")
    s = buf.get_summary()
    assert s["total"] == 1
    assert s["orphaned"] == 1
    assert s["per_speaker"] == {}
    assert s["oldest_age_seconds"] is not None
    assert s["oldest_age_seconds"] >= 0


def test_summary_attributed_session(buf):
    buf.set_speaker("conv-1", "spk-abc", "Alice")
    buf.append("conv-1", "user", "hello")
    s = buf.get_summary()
    assert s["total"] == 1
    assert s["orphaned"] == 0
    assert s["per_speaker"] == {"spk-abc": 1}


def test_retro_claim_attributes_matching_orphan(buf, tmp_path):
    """Orphan sessions with matching voice embeddings get claimed by existing profiles."""
    import math

    from paramem.server.speaker import SpeakerStore

    v = [0.5, 0.3, 0.7, 0.1, 0.4, 0.6, 0.2, 0.8]
    norm = math.sqrt(sum(x * x for x in v))
    embedding = [x / norm for x in v]

    store = SpeakerStore(tmp_path / "profiles.json")
    speaker_id = store.enroll("Alex", embedding)

    # Orphan session with matching voice
    buf.append("conv-orphan", "user", "hello there", embedding=embedding)
    assert buf.get_summary()["orphaned"] == 1

    claimed = buf.claim_sessions_for_speaker(speaker_id, "Alex", store)
    assert claimed == 1

    s = buf.get_summary()
    assert s["orphaned"] == 0
    assert s["per_speaker"] == {speaker_id: 1}


def test_summary_mixed_orphaned_and_attributed(buf):
    buf.set_speaker("conv-a", "spk-1", "Alice")
    buf.append("conv-a", "user", "hi")
    buf.append("conv-b", "user", "anonymous")  # no speaker
    buf.set_speaker("conv-c", "spk-1", "Alice")
    buf.append("conv-c", "user", "hi again")
    s = buf.get_summary()
    assert s["total"] == 3
    assert s["orphaned"] == 1
    assert s["per_speaker"] == {"spk-1": 2}


# ---------------------------------------------------------------------------
# metadata= kwarg propagation
# ---------------------------------------------------------------------------


class TestMetadataPropagation:
    def test_append_without_metadata_schema_compatible(self, buf):
        """Turns without metadata stay schema-compatible."""
        buf.append("conv-1", "user", "hello")
        turns = buf._turns["conv-1"]
        assert len(turns) == 1
        assert "metadata" not in turns[0]

    def test_append_with_metadata_stored(self, buf):
        """metadata= kwarg is stored in the turn entry."""
        meta = {"source_type": "document", "doc_title": "notes", "chunk_index": 0}
        buf.append("conv-1", "user", "hello", metadata=meta)
        turns = buf._turns["conv-1"]
        assert turns[0]["metadata"] == meta

    def test_get_pending_source_type_transcript_default(self, buf):
        """Sessions without metadata return source_type='transcript'."""
        buf.append("conv-1", "user", "regular chat turn")
        pending = buf.get_pending()
        assert len(pending) == 1
        assert pending[0]["source_type"] == "transcript"
        assert pending[0]["doc_title"] is None

    def test_get_pending_source_type_document(self, buf):
        """Sessions with document metadata return correct source_type and doc_title."""
        meta = {"source_type": "document", "doc_title": "my_notes", "chunk_index": 2}
        buf.set_speaker("doc-1", "spk-a", "Alice")
        buf.append("doc-1", "user", "document chunk text", metadata=meta)
        pending = buf.get_pending()
        assert len(pending) == 1
        assert pending[0]["source_type"] == "document"
        assert pending[0]["doc_title"] == "my_notes"
        assert pending[0]["session_id"] == "doc-1"


# ---------------------------------------------------------------------------
# get_summary per_source_type counts
# ---------------------------------------------------------------------------


class TestGetSummaryPerSourceType:
    def test_per_source_type_empty(self, buf):
        assert buf.get_summary()["per_source_type"] == {}

    def test_per_source_type_transcript_only(self, buf):
        buf.append("conv-1", "user", "hello")
        buf.append("conv-2", "user", "world")
        s = buf.get_summary()
        assert s["per_source_type"] == {"transcript": 2}

    def test_per_source_type_document_only(self, buf):
        meta = {"source_type": "document", "doc_title": "t"}
        buf.set_speaker("doc-1", "spk-a", "Alice")
        buf.append("doc-1", "user", "chunk one", metadata=meta)
        buf.set_speaker("doc-2", "spk-a", "Alice")
        buf.append("doc-2", "user", "chunk two", metadata=meta)
        s = buf.get_summary()
        assert s["per_source_type"] == {"document": 2}

    def test_per_source_type_mixed(self, buf):
        buf.append("conv-1", "user", "chat turn")
        meta = {"source_type": "document", "doc_title": "t"}
        buf.set_speaker("doc-1", "spk-a", "Alice")
        buf.append("doc-1", "user", "doc chunk", metadata=meta)
        s = buf.get_summary()
        assert s["per_source_type"] == {"transcript": 1, "document": 1}


# ---------------------------------------------------------------------------
# discard_sessions
# ---------------------------------------------------------------------------


class TestDiscardSessions:
    def test_discard_removes_from_memory(self, buf):
        buf.append("conv-1", "user", "hello")
        buf.append("conv-2", "user", "world")
        assert len(buf.get_pending()) == 2

        buf.discard_sessions(["conv-1"])

        pending = buf.get_pending()
        assert len(pending) == 1
        assert pending[0]["session_id"] == "conv-2"

    def test_discard_unknown_is_noop(self, buf):
        buf.append("conv-1", "user", "hello")
        buf.discard_sessions(["no-such-id"])
        assert len(buf.get_pending()) == 1

    def test_discard_empty_list_is_noop(self, buf):
        buf.append("conv-1", "user", "hello")
        buf.discard_sessions([])
        assert len(buf.get_pending()) == 1

    def test_discard_does_not_archive(self, buf_debug):
        """discard_sessions deletes the JSONL file — it does not archive it."""
        buf_debug.append("conv-x", "user", "hello")
        jsonl_path = buf_debug.session_dir / "conv-x.jsonl"
        assert jsonl_path.exists()

        buf_debug.discard_sessions(["conv-x"])

        assert not jsonl_path.exists()
        # Archive should be empty.
        archive_dir = buf_debug.session_dir / "archive"
        assert not list(archive_dir.glob("*.jsonl"))

    def test_discard_debug_deletes_disk_file(self, buf_debug):
        """With debug=True, the JSONL file is deleted after discard_sessions."""
        buf_debug.set_speaker("doc-1", "spk-a", "Alice")
        buf_debug.append("doc-1", "user", "chunk text")
        jsonl_path = buf_debug.session_dir / "doc-1.jsonl"
        assert jsonl_path.exists()

        buf_debug.discard_sessions(["doc-1"])

        assert not jsonl_path.exists()
        assert "doc-1" not in buf_debug._turns

    def test_discard_no_disk_file_when_debug_false(self, buf):
        """With debug=False no disk I/O is attempted."""
        buf.append("conv-1", "user", "hello")
        # Should not raise even though no JSONL exists.
        buf.discard_sessions(["conv-1"])
        assert len(buf.get_pending()) == 0
