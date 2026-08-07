"""Unit tests for SessionBuffer.get_summary, metadata propagation, and discard_sessions."""

import pytest

from paramem.server.session_buffer import _TRANSCRIPT_MAX_TOKENS, SessionBuffer
from paramem.utils.tokens import MEASURED_TOKENS_PER_WORD, estimate_tokens


@pytest.fixture
def buf(tmp_path):
    return SessionBuffer(
        session_dir=tmp_path / "sessions", state_dir=tmp_path / "state", debug=False
    )


@pytest.fixture
def buf_debug(tmp_path):
    return SessionBuffer(
        session_dir=tmp_path / "sessions", state_dir=tmp_path / "state", debug=True
    )


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


def test_append_explicit_speaker_id_attributes_without_set_speaker(buf):
    """Caller-supplied speaker_id is authoritative and needs no prior set_speaker.

    Regression for the token-auth gap: a per-user-token /chat resolved the
    speaker but never called set_speaker, so append() read empty session state
    and persisted speaker_id=None — consolidation then skipped the session and
    dropped the user's facts. append() now takes the resolved id explicitly.
    """
    buf.append("conv-tok", "user", "I live in Freiburg", speaker_id="spk-tok", speaker="Mara")
    session_id = buf.get_pending()[0]["session_id"]
    turns = buf._turns[session_id]
    assert turns[0]["speaker_id"] == "spk-tok"
    assert turns[0]["speaker"] == "Mara"
    s = buf.get_summary()
    assert s["orphaned"] == 0
    assert s["per_speaker"] == {"spk-tok": 1}


def test_append_explicit_speaker_id_overrides_unset_session_state(buf):
    """Explicit speaker_id wins even when session state was never populated."""
    # No set_speaker for this conversation_id at all. per_speaker counts
    # sessions, not turns, so one conversation → count 1.
    buf.append("conv-x", "user", "hi", speaker_id="spk-1", speaker="Alice")
    buf.append("conv-x", "assistant", "hello", speaker_id="spk-1", speaker="Alice")
    assert buf.get_summary()["per_speaker"] == {"spk-1": 1}
    assert buf.get_summary()["orphaned"] == 0


def test_append_without_explicit_speaker_id_falls_back_to_session_state(buf):
    """Omitting speaker_id preserves the legacy set_speaker→append contract."""
    buf.set_speaker("conv-vs", "spk-voice", "Bob")
    buf.append("conv-vs", "user", "hi")  # no explicit speaker_id
    session_id = buf.get_pending()[0]["session_id"]
    assert buf._turns[session_id][0]["speaker_id"] == "spk-voice"
    assert buf.get_summary()["per_speaker"] == {"spk-voice": 1}


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
        session_id = buf.get_pending()[0]["session_id"]
        turns = buf._turns[session_id]
        assert len(turns) == 1
        assert "metadata" not in turns[0]

    def test_append_with_metadata_stored(self, buf):
        """metadata= kwarg is stored in the turn entry."""
        meta = {"source_type": "document", "doc_title": "notes", "chunk_index": 0}
        buf.append("conv-1", "user", "hello", metadata=meta)
        session_id = buf.get_pending()[0]["session_id"]
        turns = buf._turns[session_id]
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
        assert pending[0]["session_id"].startswith("doc-1-")


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
        session_id_1 = next(
            p["session_id"] for p in buf.get_pending() if p["session_id"].startswith("conv-1-")
        )
        session_id_2 = next(
            p["session_id"] for p in buf.get_pending() if p["session_id"].startswith("conv-2-")
        )

        buf.discard_sessions([session_id_1])

        pending = buf.get_pending()
        assert len(pending) == 1
        assert pending[0]["session_id"] == session_id_2

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
        session_id = buf_debug.get_pending()[0]["session_id"]
        jsonl_path = buf_debug.session_dir / f"{session_id}.jsonl"
        assert jsonl_path.exists()

        buf_debug.discard_sessions([session_id])

        assert not jsonl_path.exists()
        # Archive should be empty.
        archive_dir = buf_debug.session_dir / "archive"
        assert not list(archive_dir.glob("*.jsonl"))

    def test_discard_debug_deletes_disk_file(self, buf_debug):
        """With debug=True, the JSONL file is deleted after discard_sessions."""
        buf_debug.set_speaker("doc-1", "spk-a", "Alice")
        buf_debug.append_document_chunk("doc-1", "user", "chunk text")
        jsonl_path = buf_debug.session_dir / "doc-1.jsonl"
        assert jsonl_path.exists()

        buf_debug.discard_sessions(["doc-1"])

        assert not jsonl_path.exists()
        assert "doc-1" not in buf_debug._turns

    def test_discard_no_disk_file_when_debug_false(self, buf):
        """With debug=False no disk I/O is attempted."""
        buf.append("conv-1", "user", "hello")
        session_id = buf.get_pending()[0]["session_id"]
        # Should not raise even though no JSONL exists.
        buf.discard_sessions([session_id])
        assert len(buf.get_pending()) == 0


# ---------------------------------------------------------------------------
# retirable() — document-atomic retirement filter
# ---------------------------------------------------------------------------


class TestRetirable:
    def _add_doc_chunk(
        self,
        buf: SessionBuffer,
        session_id: str,
        doc_id: str,
        chunk_count: int,
        text: str = "chunk text",
    ) -> None:
        """Helper: add a document chunk session to the buffer."""
        buf.set_speaker(session_id, "spk-1", "Alice")
        buf.set_document_metadata(session_id, doc_id=doc_id, chunk_count=chunk_count)
        buf.append_document_chunk(
            session_id,
            "user",
            text,
            metadata={
                "source_type": "document",
                "doc_id": doc_id,
                "chunk_count": chunk_count,
                "doc_filename": "test.md",
            },
        )

    def test_transcript_sessions_always_retire(self, buf):
        """Transcript sessions (no doc_id) always appear in retirable output."""
        buf.append("conv-1", "user", "chat")
        session_id = buf.get_pending()[0]["session_id"]
        result = buf.retirable({session_id})
        assert result == [session_id]

    def test_empty_completed_returns_empty(self, buf):
        """Empty completed set returns an empty list."""
        assert buf.retirable(set()) == []

    def test_complete_document_retires(self, buf):
        """All chunks of a document completed → entire document retires."""
        self._add_doc_chunk(buf, "doc-1-c000", "doc-1", chunk_count=2)
        self._add_doc_chunk(buf, "doc-1-c001", "doc-1", chunk_count=2)
        result = buf.retirable({"doc-1-c000", "doc-1-c001"})
        assert sorted(result) == ["doc-1-c000", "doc-1-c001"]

    def test_partial_document_held_back(self, buf):
        """Only one of two chunks completed → entire document held back."""
        self._add_doc_chunk(buf, "doc-2-c000", "doc-2", chunk_count=2)
        self._add_doc_chunk(buf, "doc-2-c001", "doc-2", chunk_count=2)
        # Only chunk 0 completed; chunk 1 failed (not in completed set).
        result = buf.retirable({"doc-2-c000"})
        assert result == []

    def test_mixed_transcript_and_doc_partial(self, buf):
        """Transcript retires; partially-complete doc does not."""
        buf.append("conv-x", "user", "chat")
        conv_x_session_id = buf.get_pending()[0]["session_id"]
        self._add_doc_chunk(buf, "doc-3-c000", "doc-3", chunk_count=2)
        self._add_doc_chunk(buf, "doc-3-c001", "doc-3", chunk_count=2)
        # Transcript completes; doc is only 1/2 done.
        result = buf.retirable({conv_x_session_id, "doc-3-c000"})
        assert result == [conv_x_session_id]

    def test_mixed_transcript_and_doc_complete(self, buf):
        """Transcript and complete doc both retire together."""
        buf.append("conv-y", "user", "chat")
        conv_y_session_id = buf.get_pending()[0]["session_id"]
        self._add_doc_chunk(buf, "doc-4-c000", "doc-4", chunk_count=1)
        result = buf.retirable({conv_y_session_id, "doc-4-c000"})
        assert sorted(result) == sorted([conv_y_session_id, "doc-4-c000"])

    def test_two_docs_one_complete_one_partial(self, buf):
        """Complete doc retires; partial doc stays pending."""
        # Doc A: 1 chunk, complete
        self._add_doc_chunk(buf, "docA-c000", "docA", chunk_count=1)
        # Doc B: 2 chunks, only one complete
        self._add_doc_chunk(buf, "docB-c000", "docB", chunk_count=2)
        self._add_doc_chunk(buf, "docB-c001", "docB", chunk_count=2)

        result = buf.retirable({"docA-c000", "docB-c000"})
        assert result == ["docA-c000"]

    def test_retirable_does_not_mutate_state(self, buf):
        """retirable() is a pure filter — does not remove sessions from memory."""
        self._add_doc_chunk(buf, "doc-5-c000", "doc-5", chunk_count=1)
        buf.retirable({"doc-5-c000"})
        # Session still in _turns after the call
        assert "doc-5-c000" in buf._turns


# ---------------------------------------------------------------------------
# mark_consolidated with doc archival / deletion
# ---------------------------------------------------------------------------


class TestMarkConsolidatedDocGroups:
    def _add_doc_chunk(
        self,
        buf: SessionBuffer,
        session_id: str,
        doc_id: str,
        chunk_count: int,
        doc_filename: str = "test.md",
    ) -> None:
        buf.set_speaker(session_id, "spk-1", "Alice")
        buf.set_document_metadata(session_id, doc_id=doc_id, chunk_count=chunk_count)
        buf.append_document_chunk(
            session_id,
            "user",
            "chunk text",
            metadata={
                "source_type": "document",
                "doc_id": doc_id,
                "chunk_count": chunk_count,
                "doc_filename": doc_filename,
            },
        )

    def test_retain_archives_chunks_and_origdoc(self, tmp_path):
        """When retaining, chunk JSONLs and origdoc are co-located under retention_dir/<doc_id>/.

        Regression: chunk JSONLs were archived flat (retention_dir/<session_id>.jsonl)
        while the origdoc was archived under retention_dir/<doc_id>/.  Both must
        land under the same doc_id subdirectory.
        """
        sessions_dir = tmp_path / "sessions"
        buf = SessionBuffer(
            session_dir=sessions_dir,
            state_dir=sessions_dir.parent / "state",
            retain_sessions=True,
            debug=False,
        )

        doc_id = "doc-retain1"
        chunk_sid = f"{doc_id}-c000"
        self._add_doc_chunk(buf, chunk_sid, doc_id, chunk_count=1, doc_filename="notes.md")
        buf.write_origdoc(doc_id, b"original content")

        retention_dir = tmp_path / "retention"
        buf.mark_consolidated([chunk_sid], retention_dir=retention_dir)

        # Chunk JSONL must be co-located with the origdoc under retention_dir/<doc_id>/.
        assert (retention_dir / doc_id / f"{chunk_sid}.jsonl").exists(), (
            f"Chunk JSONL must be under retention_dir/{doc_id}/, not flat"
        )
        # Chunk JSONL must NOT be at the flat (buggy) path.
        assert not (retention_dir / f"{chunk_sid}.jsonl").exists(), (
            "Chunk JSONL must not be archived flat; it belongs under the doc_id subdirectory"
        )
        # origdoc archived under retention_dir/<doc_id>/notes.md
        assert (retention_dir / doc_id / "notes.md").exists()
        assert (retention_dir / doc_id / "notes.md").read_bytes() == b"original content"
        # origdoc removed from session_dir
        assert not (sessions_dir / f"{doc_id}.origdoc").exists()

    def test_delete_removes_chunks_and_origdoc(self, tmp_path):
        """In privacy mode (retain=False, debug=False), both chunk JSONLs and origdoc deleted."""
        sessions_dir = tmp_path / "sessions"
        buf = SessionBuffer(
            session_dir=sessions_dir,
            state_dir=sessions_dir.parent / "state",
            retain_sessions=False,
            debug=False,
        )

        doc_id = "doc-delete1"
        self._add_doc_chunk(buf, f"{doc_id}-c000", doc_id, chunk_count=1)
        buf.write_origdoc(doc_id, b"some bytes")

        buf.mark_consolidated([f"{doc_id}-c000"], retention_dir=None)

        assert not (sessions_dir / f"{doc_id}-c000.jsonl").exists()
        assert not (sessions_dir / f"{doc_id}.origdoc").exists()

    def test_transcript_sessions_use_flat_layout(self, tmp_path):
        """Transcript sessions are archived flat under retention_dir (unchanged)."""
        sessions_dir = tmp_path / "sessions"
        buf = SessionBuffer(
            session_dir=sessions_dir,
            state_dir=sessions_dir.parent / "state",
            retain_sessions=True,
            debug=False,
        )
        buf.append("conv-t1", "user", "hello")
        session_id = buf.get_pending()[0]["session_id"]

        retention_dir = tmp_path / "retention"
        buf.mark_consolidated([session_id], retention_dir=retention_dir)

        assert (retention_dir / f"{session_id}.jsonl").exists()

    def test_retired_doc_chunk_archives_group_under_retired_doc_subdir(self, tmp_path):
        """A retired chunk pulls the whole doc_id group + origdoc into
        retention_dir/retired_recall_failed/<doc_id>/ — doc-atomicity extends
        to the retired distinction; a document is never split across the two
        destinations.
        """
        sessions_dir = tmp_path / "sessions"
        buf = SessionBuffer(
            session_dir=sessions_dir,
            state_dir=sessions_dir.parent / "state",
            retain_sessions=True,
            debug=False,
        )

        doc_id = "doc-retired1"
        chunk_sid = f"{doc_id}-c000"
        self._add_doc_chunk(buf, chunk_sid, doc_id, chunk_count=1, doc_filename="retired.md")
        buf.write_origdoc(doc_id, b"retired original content")

        retention_dir = tmp_path / "retention"
        buf.mark_consolidated(
            [chunk_sid],
            retention_dir=retention_dir,
            retired_session_ids={chunk_sid},
        )

        retired_doc_dir = retention_dir / "retired_recall_failed" / doc_id
        assert (retired_doc_dir / f"{chunk_sid}.jsonl").exists()
        assert (retired_doc_dir / "retired.md").exists()
        assert (retired_doc_dir / "retired.md").read_bytes() == b"retired original content"
        # Not archived under the plain (non-retired) doc location.
        assert not (retention_dir / doc_id).exists()

    def test_retired_doc_chunk_pulls_sibling_chunks_and_origdoc_too(self, tmp_path):
        """Two-chunk document, only ONE chunk in retired_session_ids: BOTH chunk
        JSONLs and the origdoc still land under the retired subdir together —
        this is the case that would pass even if ``retired_doc_ids`` (the
        doc-atomicity mechanism, session_buffer.py) were deleted, since a
        single-chunk document can't distinguish "this chunk's own retired flag"
        from "the group's retired flag". Two chunks force the distinction.
        """
        sessions_dir = tmp_path / "sessions"
        buf = SessionBuffer(
            session_dir=sessions_dir,
            state_dir=sessions_dir.parent / "state",
            retain_sessions=True,
            debug=False,
        )

        doc_id = "doc-retired2"
        chunk0_sid = f"{doc_id}-c000"
        chunk1_sid = f"{doc_id}-c001"
        self._add_doc_chunk(buf, chunk0_sid, doc_id, chunk_count=2, doc_filename="retired2.md")
        self._add_doc_chunk(buf, chunk1_sid, doc_id, chunk_count=2, doc_filename="retired2.md")
        buf.write_origdoc(doc_id, b"two-chunk retired content")

        retention_dir = tmp_path / "retention"
        # Only chunk0 is retired; chunk1 consolidated cleanly.
        buf.mark_consolidated(
            [chunk0_sid, chunk1_sid],
            retention_dir=retention_dir,
            retired_session_ids={chunk0_sid},
        )

        retired_doc_dir = retention_dir / "retired_recall_failed" / doc_id
        # BOTH chunks — including the non-retired sibling — land under the
        # retired subdir, since a document is never split across destinations.
        assert (retired_doc_dir / f"{chunk0_sid}.jsonl").exists()
        assert (retired_doc_dir / f"{chunk1_sid}.jsonl").exists()
        assert (retired_doc_dir / "retired2.md").exists()
        assert (retired_doc_dir / "retired2.md").read_bytes() == b"two-chunk retired content"
        # The plain (non-retired) doc location must not exist at all.
        assert not (retention_dir / doc_id).exists()


class TestMarkConsolidatedRetiredSessions:
    def test_retired_transcript_lands_under_retired_subdir(self, tmp_path):
        """A retired (retry-capped) transcript session is archived under
        retention_dir/retired_recall_failed/, not the flat layout.
        """
        sessions_dir = tmp_path / "sessions"
        buf = SessionBuffer(
            session_dir=sessions_dir,
            state_dir=sessions_dir.parent / "state",
            retain_sessions=True,
            debug=False,
        )
        buf.append("conv-retired1", "user", "hello")
        session_id = buf.get_pending()[0]["session_id"]

        retention_dir = tmp_path / "retention"
        buf.mark_consolidated(
            [session_id],
            retention_dir=retention_dir,
            retired_session_ids={session_id},
        )

        assert (retention_dir / "retired_recall_failed" / f"{session_id}.jsonl").exists()
        assert not (retention_dir / f"{session_id}.jsonl").exists()

    def test_retired_transcript_unlinked_when_retain_false(self, tmp_path):
        """retain_sessions=False (and debug=False) still means unlink for
        retired sessions too — retention is the operator's choice, not a
        second policy for the retired subset.
        """
        sessions_dir = tmp_path / "sessions"
        buf = SessionBuffer(
            session_dir=sessions_dir,
            state_dir=sessions_dir.parent / "state",
            retain_sessions=False,
            debug=False,
        )
        buf.append("conv-retired2", "user", "hello")
        session_id = buf.get_pending()[0]["session_id"]

        buf.mark_consolidated(
            [session_id],
            retention_dir=None,
            retired_session_ids={session_id},
        )

        assert not (sessions_dir / f"{session_id}.jsonl").exists()

    def test_non_retired_sessions_in_mixed_batch_keep_flat_layout(self, tmp_path):
        """A mixed mark_consolidated call (one retired, one clean) leaves the
        clean session on today's flat layout — retired_session_ids does not
        change behaviour for sessions outside it.
        """
        sessions_dir = tmp_path / "sessions"
        buf = SessionBuffer(
            session_dir=sessions_dir,
            state_dir=sessions_dir.parent / "state",
            retain_sessions=True,
            debug=False,
        )
        buf.append("conv-retired3", "user", "hello")
        retired_sid = buf.get_pending()[0]["session_id"]
        buf.append("conv-clean3", "user", "hi")
        clean_sid = [s["session_id"] for s in buf.get_pending() if s["session_id"] != retired_sid][
            0
        ]

        retention_dir = tmp_path / "retention"
        buf.mark_consolidated(
            [retired_sid, clean_sid],
            retention_dir=retention_dir,
            retired_session_ids={retired_sid},
        )

        assert (retention_dir / "retired_recall_failed" / f"{retired_sid}.jsonl").exists()
        assert (retention_dir / f"{clean_sid}.jsonl").exists()
        assert not (retention_dir / f"{retired_sid}.jsonl").exists()


# ---------------------------------------------------------------------------
# discard_sessions with origdoc cleanup
# ---------------------------------------------------------------------------


class TestDiscardSessionsOrigdoc:
    def _add_doc_chunk(self, buf: SessionBuffer, session_id: str, doc_id: str) -> None:
        buf.set_speaker(session_id, "spk-1", "Alice")
        buf.set_document_metadata(session_id, doc_id=doc_id, chunk_count=1)
        buf.append_document_chunk(
            session_id,
            "user",
            "chunk text",
            metadata={"source_type": "document", "doc_id": doc_id, "chunk_count": 1},
        )

    def test_discard_removes_origdoc_when_all_chunks_discarded(self, buf):
        """Discarding all chunks removes the origdoc blob."""
        doc_id = "doc-discard1"
        self._add_doc_chunk(buf, f"{doc_id}-c000", doc_id)
        buf.write_origdoc(doc_id, b"content")
        assert (buf.session_dir / f"{doc_id}.origdoc").exists()

        buf.discard_sessions([f"{doc_id}-c000"])

        assert not (buf.session_dir / f"{doc_id}.origdoc").exists()

    def test_discard_leaves_origdoc_when_sibling_chunk_remains(self, buf):
        """Partial discard (one chunk of two) leaves the origdoc in place."""
        doc_id = "doc-partial"
        buf.set_speaker(f"{doc_id}-c000", "spk-1", "Alice")
        buf.set_document_metadata(f"{doc_id}-c000", doc_id=doc_id, chunk_count=2)
        buf.append_document_chunk(
            f"{doc_id}-c000",
            "user",
            "chunk 0",
            metadata={"source_type": "document", "doc_id": doc_id, "chunk_count": 2},
        )

        buf.set_speaker(f"{doc_id}-c001", "spk-1", "Alice")
        buf.set_document_metadata(f"{doc_id}-c001", doc_id=doc_id, chunk_count=2)
        buf.append_document_chunk(
            f"{doc_id}-c001",
            "user",
            "chunk 1",
            metadata={"source_type": "document", "doc_id": doc_id, "chunk_count": 2},
        )

        buf.write_origdoc(doc_id, b"content")

        # Discard only chunk 0; chunk 1 still in buffer.
        buf.discard_sessions([f"{doc_id}-c000"])

        # origdoc must remain because chunk 1 is still pending.
        assert (buf.session_dir / f"{doc_id}.origdoc").exists()

    def test_discard_origdoc_noop_when_no_origdoc_file(self, buf):
        """Discarding a doc chunk without an origdoc file does not raise."""
        doc_id = "doc-no-origdoc"
        self._add_doc_chunk(buf, f"{doc_id}-c000", doc_id)
        # No write_origdoc call — file absent.
        buf.discard_sessions([f"{doc_id}-c000"])  # must not raise
        assert len(buf.get_pending()) == 0


# ---------------------------------------------------------------------------
# pending_facts / get_pending population parity
# ---------------------------------------------------------------------------


class TestPendingFactsPopulationParity:
    """pending_facts() and get_pending() must expose the same session ids.

    Covers three cases: a normal session with turns, an empty in-memory
    session (turns registered but list is empty), and an empty disk-only
    JSONL (file on disk with no content).
    """

    def _session_ids_from_pending_facts(self, buf: SessionBuffer) -> set[str]:
        return {f["session_id"] for f in buf.pending_facts()}

    def _session_ids_from_get_pending(self, buf: SessionBuffer) -> set[str]:
        return {p["session_id"] for p in buf.get_pending()}

    def test_normal_session_both_see_it(self, buf):
        """A session with actual turns appears in both pending_facts and get_pending."""
        buf.append("conv-normal", "user", "hello world")
        assert self._session_ids_from_pending_facts(buf) == self._session_ids_from_get_pending(buf)
        assert any(
            sid.startswith("conv-normal-") for sid in self._session_ids_from_pending_facts(buf)
        )

    def test_empty_in_memory_session_excluded_by_both(self, buf, tmp_path):
        """An in-memory session with no turns is invisible to both methods."""
        # Register a session in _sessions and _turns but with an empty turns list.
        # _turns is a defaultdict(list), so accessing the key is enough to register it
        # without appending any turns.
        buf._turns["conv-empty"]  # creates the key; leaves list empty
        buf._sessions["conv-empty"] = {"speaker": None, "state": "new"}
        # Also write an empty JSONL to ensure the disk branch sees the same thing.
        (buf.session_dir / "conv-empty.jsonl").write_text("")

        assert self._session_ids_from_pending_facts(buf) == self._session_ids_from_get_pending(buf)
        assert "conv-empty" not in self._session_ids_from_pending_facts(buf)

    def test_empty_disk_only_jsonl_excluded_by_both(self, buf):
        """A disk-only JSONL with no content is invisible to both methods."""
        # Write an empty JSONL file that exists on disk but was never loaded into RAM.
        (buf.session_dir / "conv-disk-empty.jsonl").write_text("")

        assert self._session_ids_from_pending_facts(buf) == self._session_ids_from_get_pending(buf)
        assert "conv-disk-empty" not in self._session_ids_from_pending_facts(buf)

    def test_parity_with_mixed_population(self, buf):
        """Normal session and empty session together: only normal appears in both."""
        buf.append("conv-good", "user", "some text")
        conv_good_session_id = buf.get_pending()[0]["session_id"]
        # Empty in-memory entry
        buf._turns["conv-bad"]  # noqa: B018 — defaultdict side-effect: registers empty list
        (buf.session_dir / "conv-bad.jsonl").write_text("")

        pf_ids = self._session_ids_from_pending_facts(buf)
        gp_ids = self._session_ids_from_get_pending(buf)
        assert pf_ids == gp_ids
        assert pf_ids == {conv_good_session_id}


# ---------------------------------------------------------------------------
# Session-boundary rotation: idle timeout, restart split, doc no-rotate,
# collision-free minting, durable-unit methods after the two-map split.
# ---------------------------------------------------------------------------


class TestSessionRotation:
    def test_within_window_stays_one_session(self, tmp_path):
        """Two turns on the same conversation_id inside the idle window join
        one session."""
        buf = SessionBuffer(
            session_dir=tmp_path / "sessions",
            state_dir=tmp_path / "state",
            idle_timeout_minutes=10,
        )
        buf.append("conv-1", "user", "hello")
        buf.append("conv-1", "assistant", "hi")

        pending = buf.get_pending()
        assert len(pending) == 1
        assert "[user] hello" in pending[0]["transcript"]
        assert "[assistant] hi" in pending[0]["transcript"]

    def test_idle_rollover_mints_two_sessions_with_distinct_started_at(self, tmp_path):
        """A gap exceeding idle_timeout mints a fresh session_id with a new
        started_at; the two sessions carry distinct started_at values.

        Backdates ``_open["conv-1"]["last_turn_at"]`` to simulate elapsed
        idle time rather than mocking ``datetime`` module-wide (which would
        also have to account for ``datetime.fromisoformat`` inside
        ``_resolve_session_id``).
        """
        from datetime import datetime, timedelta, timezone

        buf = SessionBuffer(
            session_dir=tmp_path / "sessions",
            state_dir=tmp_path / "state",
            idle_timeout_minutes=10,
        )
        buf.append("conv-1", "user", "first turn")

        # Simulate 11 minutes of inactivity (> the 10-minute idle timeout).
        stale = datetime.now(timezone.utc) - timedelta(minutes=11)
        buf._open["conv-1"]["last_turn_at"] = stale.isoformat()

        buf.append("conv-1", "user", "second turn, after idle timeout")

        pending = buf.get_pending()
        assert len(pending) == 2
        session_ids = {p["session_id"] for p in pending}
        assert len(session_ids) == 2
        started_ats = {p["started_at"] for p in pending}
        assert len(started_ats) == 2

    def test_restart_split_new_session_after_rehydrate(self, tmp_path):
        """rehydrate_from_disk (cold start) does not restore _open — the next
        append on the same conversation_id opens a fresh session."""
        buf1 = SessionBuffer(session_dir=tmp_path / "sessions", state_dir=tmp_path / "state")
        buf1.append("conv-1", "user", "before restart")
        session_id_before = buf1.get_pending()[0]["session_id"]

        # Cold restart: fresh buffer, rehydrate pending JSONL (no snapshot).
        buf2 = SessionBuffer(session_dir=tmp_path / "sessions", state_dir=tmp_path / "state")
        buf2.rehydrate_from_disk()
        buf2.append("conv-1", "user", "after restart")

        pending = buf2.get_pending()
        session_ids = {p["session_id"] for p in pending}
        assert session_id_before in session_ids, "rehydrated pre-restart session must survive"
        assert len(session_ids) == 2, f"restart must split into a new session, got {session_ids}"

    def test_doc_chunks_never_rotate_even_past_idle_timeout(self, tmp_path):
        """Two chunks of the same document, appended with an idle-timeout-
        exceeding gap between them (simulated via a stale ``_open`` entry
        under the chunk's own session_id key), still keep their exact
        deterministic session_ids — ``append_document_chunk`` never
        consults ``_open``/idle rotation at all (slow-ingest regime)."""
        from datetime import datetime, timedelta, timezone

        buf = SessionBuffer(
            session_dir=tmp_path / "sessions",
            state_dir=tmp_path / "state",
            idle_timeout_minutes=10,
        )
        buf.set_speaker("doc-1-c000", "spk-a", "Alice")
        buf.set_document_metadata("doc-1-c000", doc_id="doc-1", chunk_count=2)
        buf.append_document_chunk("doc-1-c000", "user", "chunk zero")

        # Simulate 30 minutes elapsing before the second chunk arrives —
        # if append_document_chunk routed through _resolve_session_id like
        # append() does, this would trigger a rotation. It must not.
        stale = datetime.now(timezone.utc) - timedelta(minutes=30)
        buf._open["doc-1-c000"] = {"last_turn_at": stale.isoformat()}

        buf.set_speaker("doc-1-c001", "spk-a", "Alice")
        buf.set_document_metadata("doc-1-c001", doc_id="doc-1", chunk_count=2)
        buf.append_document_chunk("doc-1-c001", "user", "chunk one, much later")

        session_ids = {p["session_id"] for p in buf.get_pending()}
        assert session_ids == {"doc-1-c000", "doc-1-c001"}

    def test_session_id_collision_free_under_same_second_opens(self):
        """Two rotations for the SAME conversation_id, minted within the same
        wall-clock second, produce distinct session_ids — the mandatory
        8-char random suffix (not "only on collision") makes the jsonl
        filename race-free."""
        from paramem.server.session_buffer import _mint_session_id

        ids = {_mint_session_id("conv-1") for _ in range(50)}
        # 50 mints of the same conversation_id within the same test-run
        # second (the timestamp component is identical for all of them
        # in practice) must still be all-distinct thanks to the rand
        # suffix.
        assert len(ids) == 50

    def test_durable_unit_methods_resolve_after_two_map_split(self, tmp_path):
        """retirable / mark_consolidated / discard_sessions / hydrate_retry_counts
        / bump_retry_and_release / reset_retry_count_for all still operate on
        minted session_ids — the highest-risk regression surface from the
        _sessions/_open split."""
        buf = SessionBuffer(
            session_dir=tmp_path / "sessions",
            state_dir=tmp_path / "state",
            consolidation_retry_cap=2,
        )
        buf.append("conv-1", "user", "hello")
        session_id = buf.get_pending()[0]["session_id"]

        # retirable(): transcript sessions (no doc_id) always retire.
        assert buf.retirable({session_id}) == [session_id]

        # bump_retry_and_release / reset_retry_count_for operate on _sessions,
        # keyed by the minted session_id (durable, unaffected by the _open split).
        buf.hydrate_retry_counts()  # no durable file yet — non-fatal no-op
        released = buf.bump_retry_and_release({session_id})
        assert released == []  # cap is 2, first bump doesn't release
        assert buf._sessions[session_id]["recall_retry_count"] == 1

        buf.reset_retry_count_for(session_id)
        assert "recall_retry_count" not in buf._sessions[session_id]

        # discard_sessions / mark_consolidated resolve the minted id and clear it.
        buf.append("conv-2", "user", "hello again")
        sid2 = next(
            p["session_id"] for p in buf.get_pending() if p["session_id"].startswith("conv-2-")
        )
        buf.mark_consolidated([sid2])
        assert sid2 not in {p["session_id"] for p in buf.get_pending()}

    def test_mark_consolidated_evicts_open_and_next_append_mints_fresh(self, tmp_path):
        """After mark_consolidated retires a conversation's session, the next
        append on the SAME conversation_id mints a NEW session_id — it does
        not reuse the retired one, and _open no longer references it."""
        buf = SessionBuffer(session_dir=tmp_path / "sessions", state_dir=tmp_path / "state")
        buf.append("conv-1", "user", "hello")
        retired_session_id = buf.get_pending()[0]["session_id"]
        assert buf._open["conv-1"]["session_id"] == retired_session_id

        buf.mark_consolidated([retired_session_id])
        assert "conv-1" not in buf._open, "_open must be pruned for the retired session_id"

        buf.append("conv-1", "user", "hello again, new session")
        new_session_id = buf.get_pending()[0]["session_id"]
        assert new_session_id != retired_session_id
        assert buf._open["conv-1"]["session_id"] == new_session_id

    def test_discard_sessions_evicts_open_and_next_append_mints_fresh(self, tmp_path):
        """Same coherence guarantee as mark_consolidated, for discard_sessions
        (the cancel path)."""
        buf = SessionBuffer(session_dir=tmp_path / "sessions", state_dir=tmp_path / "state")
        buf.append("conv-1", "user", "hello")
        discarded_session_id = buf.get_pending()[0]["session_id"]

        buf.discard_sessions([discarded_session_id])
        assert "conv-1" not in buf._open

        buf.append("conv-1", "user", "hello again, new session")
        new_session_id = buf.get_pending()[0]["session_id"]
        assert new_session_id != discarded_session_id

    def test_doc_chunk_open_entry_evicted_on_retirement(self, tmp_path):
        """Same orphan-leak class as the voice transport-id bug: a doc
        chunk's _open entry (created by the ingest-style set_speaker call)
        must be pruned on retirement, not left behind forever.

        append_document_chunk backlinks _open[session_id]["session_id"] =
        session_id (mirroring _resolve_session_id) so
        _prune_open_for_retired_sessions' existing session_id-match rule
        evicts it uniformly with conversational sessions — no doc special
        case in the prune itself.
        """
        buf = SessionBuffer(session_dir=tmp_path / "sessions", state_dir=tmp_path / "state")
        chunk_id = "doc-1-c000"
        buf.set_speaker(chunk_id, "spk-a", "Alice")
        buf.set_document_metadata(chunk_id, doc_id="doc-1", chunk_count=1)
        buf.append_document_chunk(chunk_id, "user", "chunk text")

        assert chunk_id in buf._open
        assert buf._open[chunk_id]["session_id"] == chunk_id

        buf.mark_consolidated([chunk_id])

        assert chunk_id not in buf._open, "doc chunk _open entry must be pruned on retirement"

    # -----------------------------------------------------------------
    # Size rotation (deliberate) + conversation-scoped chain read
    # -----------------------------------------------------------------
    #
    # Note (plan-review verification): the four idle-rotation tests above
    # (test_within_window_stays_one_session, test_idle_rollover_mints_
    # two_sessions_with_distinct_started_at, test_restart_split_new_
    # session_after_rehydrate, test_doc_chunks_never_rotate_even_past_
    # idle_timeout) call through the public append()/append_document_chunk()
    # API and are unaffected by _resolve_session_id's added role/text
    # parameters — they never call it directly.

    @staticmethod
    def _turn_tokens(text: str, role: str = "user") -> int:
        """The exact estimated-token cost _resolve_session_id assigns one
        turn — via the same _format_turns + estimate_tokens path."""
        formatted, _ = SessionBuffer._format_turns([{"role": role, "text": text}])
        return estimate_tokens(formatted[0])

    @classmethod
    def _drive_to_size_rotation(cls, buf: SessionBuffer, conv_id: str, turn_text: str) -> int:
        """Append *turn_text* repeatedly until a size rotation fires.

        Detects rotation via the concrete ``session_id`` changing (not via
        ``get_pending()`` count, which trivially rises 0->1 on a fresh
        conversation's very first turn — a false positive for "rotated").

        Returns the number of turns appended (including the rotating one).
        Safety-capped so a broken cap constant fails the test loudly
        instead of hanging.
        """
        buf.append(conv_id, "user", turn_text)
        session_id = buf._open[conv_id]["session_id"]
        count = 1
        while True:
            buf.append(conv_id, "user", turn_text)
            count += 1
            new_session_id = buf._open[conv_id]["session_id"]
            if new_session_id != session_id:
                return count
            assert count < 10_000, "size rotation never fired — cap constant likely broken"

    def test_size_rotation_fires_across_the_cap(self, tmp_path):
        """Turns whose accumulated estimate crosses _TRANSCRIPT_MAX_TOKENS
        land in two sessions."""
        buf = SessionBuffer(
            session_dir=tmp_path / "sessions",
            state_dir=tmp_path / "state",
            idle_timeout_minutes=10,
        )
        turn_text = ("word " * 100).strip()
        count = self._drive_to_size_rotation(buf, "conv-size", turn_text)
        assert count > 1, "a single turn should never itself exceed the cap here"
        assert len(buf.get_pending()) == 2

    def test_lone_oversize_turn_admitted_whole(self, tmp_path):
        """A single turn well over the cap produces ONE session containing
        it, no split (the fail-closed accumulated-token residual behaviour:
        accumulated > 0 is what makes rotation possible, so a lone turn is
        always admitted)."""
        buf = SessionBuffer(
            session_dir=tmp_path / "sessions",
            state_dir=tmp_path / "state",
            idle_timeout_minutes=10,
        )
        huge_text = ("word " * 5000).strip()
        assert self._turn_tokens(huge_text) > _TRANSCRIPT_MAX_TOKENS
        buf.append("conv-lone", "user", huge_text)
        pending = buf.get_pending()
        assert len(pending) == 1
        assert huge_text in pending[0]["transcript"]

    def test_accumulator_resets_on_size_rotation(self, tmp_path):
        """After a size rotation the new session's accumulator holds only
        the rotating turn's own cost — not the pre-rotation cumulative
        total — so it accepts a full cap's worth again."""
        buf = SessionBuffer(
            session_dir=tmp_path / "sessions",
            state_dir=tmp_path / "state",
            idle_timeout_minutes=10,
        )
        turn_text = ("word " * 100).strip()
        self._drive_to_size_rotation(buf, "conv-reset", turn_text)
        assert buf._open["conv-reset"]["session_tokens"] == self._turn_tokens(turn_text)

    def test_idle_rotation_also_resets_accumulator(self, tmp_path):
        """An idle-timeout rotation resets the accumulator exactly like a
        size rotation — the new session's count reflects only the turn
        that opened it."""
        from datetime import datetime, timedelta, timezone

        buf = SessionBuffer(
            session_dir=tmp_path / "sessions",
            state_dir=tmp_path / "state",
            idle_timeout_minutes=10,
        )
        buf.append("conv-idle-acc", "user", "first turn with several words in it")
        assert buf._open["conv-idle-acc"]["session_tokens"] > 0

        stale = datetime.now(timezone.utc) - timedelta(minutes=20)
        buf._open["conv-idle-acc"]["last_turn_at"] = stale.isoformat()

        second_text = "second turn after the idle gap"
        buf.append("conv-idle-acc", "user", second_text)
        assert buf._open["conv-idle-acc"]["session_tokens"] == self._turn_tokens(second_text)

    def test_idle_rotation_clears_chain_size_rotation_appends(self, tmp_path):
        """Size rotation appends the retiring session onto
        prior_session_ids; an idle rotation clears the chain instead
        (context reset on idle is deliberate)."""
        from datetime import datetime, timedelta, timezone

        buf = SessionBuffer(
            session_dir=tmp_path / "sessions",
            state_dir=tmp_path / "state",
            idle_timeout_minutes=10,
        )
        turn_text = ("word " * 100).strip()
        self._drive_to_size_rotation(buf, "conv-chain", turn_text)
        assert buf._open["conv-chain"]["prior_session_ids"] != []

        stale = datetime.now(timezone.utc) - timedelta(minutes=20)
        buf._open["conv-chain"]["last_turn_at"] = stale.isoformat()
        buf.append("conv-chain", "user", "after the idle gap")

        assert buf._open["conv-chain"]["prior_session_ids"] == []

    def test_idle_expired_wins_over_size_exceeded_for_chain_continuity(self, tmp_path):
        """When BOTH idle-expired and size-exceeded are true on the same
        append, idle wins: the chain resets (prior_session_ids stays
        empty) rather than carrying the retiring session forward. Pins
        the `size_exceeded and not idle_expired` term in
        _resolve_session_id's `context_continues` — dropping the
        `and not idle_expired` clause would carry the chain forward here
        and fail this test."""
        from datetime import datetime, timedelta, timezone

        buf = SessionBuffer(
            session_dir=tmp_path / "sessions",
            state_dir=tmp_path / "state",
            idle_timeout_minutes=10,
        )
        turn_text = ("word " * 100).strip()
        turn_tokens = self._turn_tokens(turn_text)

        buf.append("conv-both", "user", turn_text)
        count = 1
        # Build up to exactly one turn short of a size rotation: after the
        # loop, the NEXT identical turn is guaranteed to size-rotate on
        # its own merits (accumulated + turn_tokens > cap).
        while buf._open["conv-both"]["session_tokens"] + turn_tokens <= _TRANSCRIPT_MAX_TOKENS:
            buf.append("conv-both", "user", turn_text)
            count += 1
            assert count < 10_000, "never reached the size-rotation boundary"
        session_id_before = buf._open["conv-both"]["session_id"]

        # Force idle expiry too, on top of a turn that would independently
        # size-rotate.
        stale = datetime.now(timezone.utc) - timedelta(minutes=20)
        buf._open["conv-both"]["last_turn_at"] = stale.isoformat()

        buf.append("conv-both", "user", turn_text)

        assert buf._open["conv-both"]["session_id"] != session_id_before, (
            "a rotation must have fired (either trigger alone would cause one)"
        )
        assert buf._open["conv-both"]["prior_session_ids"] == [], (
            "idle must win: the chain must reset, not carry the retiring "
            "session forward, when idle-expired is also true"
        )

    def test_retiring_session_transcript_fits_within_the_transcript_cap(self, tmp_path):
        """End-to-end pin mirroring the document path's
        TestDocumentPathBudget (tests/test_document_chunker.py): build a
        session up to the rotation boundary, then verify the RETIRED
        session's actual formatted transcript — the same shape
        get_pending() produces and the same shape the anonymize call is
        sized against — fits within _TRANSCRIPT_MAX_TOKENS. Not the
        internal per-turn accumulator (already pinned above), the real
        end-to-end quantity."""
        buf = SessionBuffer(
            session_dir=tmp_path / "sessions",
            state_dir=tmp_path / "state",
            idle_timeout_minutes=10,
        )
        turn_text = ("word " * 100).strip()
        self._drive_to_size_rotation(buf, "conv-budget", turn_text)

        current_session_id = buf._open["conv-budget"]["session_id"]
        pending = buf.get_pending()
        retired = [p for p in pending if p["session_id"] != current_session_id]
        assert len(retired) == 1, f"expected exactly one retired session, got {len(retired)}"
        retired_transcript = retired[0]["transcript"]
        assert estimate_tokens(retired_transcript) <= _TRANSCRIPT_MAX_TOKENS

    def test_chain_read_spans_size_rotation_chronologically(self, tmp_path):
        """get_conversation_turns returns turns across a size rotation, in
        chronological (append) order — a mid-dialogue size rotation is
        invisible to a caller reading the conversation's served context."""
        buf = SessionBuffer(
            session_dir=tmp_path / "sessions",
            state_dir=tmp_path / "state",
            idle_timeout_minutes=10,
        )
        turn_text_base = ("word " * 100).strip()
        appended: list[str] = []
        i = 0
        while len(buf.get_pending()) < 2:
            text = f"turn{i} {turn_text_base}"
            buf.append("conv-chrono", "user", text)
            appended.append(text)
            i += 1
            assert i < 10_000, "size rotation never fired — cap constant likely broken"

        turns = buf.get_conversation_turns("conv-chrono")
        assert [t["text"] for t in turns] == appended

    def test_chain_read_after_idle_rotation_returns_only_current_session(self, tmp_path):
        """The idle-vs-size rotation boundary: an idle rotation DOES reset
        served context — the chain read after one returns only the new
        session's turns, not the pre-idle-gap ones."""
        from datetime import datetime, timedelta, timezone

        buf = SessionBuffer(
            session_dir=tmp_path / "sessions",
            state_dir=tmp_path / "state",
            idle_timeout_minutes=10,
        )
        buf.append("conv-idle-chain", "user", "before the idle gap")

        stale = datetime.now(timezone.utc) - timedelta(minutes=20)
        buf._open["conv-idle-chain"]["last_turn_at"] = stale.isoformat()
        buf.append("conv-idle-chain", "user", "after the idle gap")

        turns = buf.get_conversation_turns("conv-idle-chain")
        assert [t["text"] for t in turns] == ["after the idle gap"]

    def _setup_daily(self, tmp_path, monkeypatch, passphrase="pw"):
        """Install a daily age identity so the envelope-encrypt snapshot
        path engages (mirrors tests/test_server.py's TestSessionBuffer
        helper of the same name)."""
        from paramem.backup.key_store import (
            _clear_daily_identity_cache,
            mint_daily_identity,
            wrap_daily_identity,
            write_daily_key_file,
        )

        ident = mint_daily_identity()
        key_path = tmp_path / "daily_key.age"
        write_daily_key_file(wrap_daily_identity(ident, passphrase), key_path)
        monkeypatch.setenv("PARAMEM_DAILY_PASSPHRASE", passphrase)
        monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path)
        _clear_daily_identity_cache()
        return ident

    def test_snapshot_round_trip_preserves_rotation_state(self, tmp_path, monkeypatch):
        """save_snapshot -> fresh buffer -> load_snapshot round-trips both
        session_tokens and prior_session_ids (new keys inside the existing
        "open" dict — no snapshot schema version bump needed)."""
        self._setup_daily(tmp_path, monkeypatch)

        buf1 = SessionBuffer(
            session_dir=tmp_path / "sessions",
            state_dir=tmp_path / "state",
            idle_timeout_minutes=10,
        )
        turn_text = ("word " * 100).strip()
        self._drive_to_size_rotation(buf1, "conv-snap", turn_text)
        prior_before = list(buf1._open["conv-snap"]["prior_session_ids"])
        tokens_before = buf1._open["conv-snap"]["session_tokens"]
        assert prior_before != []

        assert buf1.save_snapshot()

        buf2 = SessionBuffer(
            session_dir=tmp_path / "sessions",
            state_dir=tmp_path / "state",
            idle_timeout_minutes=10,
        )
        assert buf2.load_snapshot()

        assert buf2._open["conv-snap"]["prior_session_ids"] == prior_before
        assert buf2._open["conv-snap"]["session_tokens"] == tokens_before

    def test_snapshot_schema_tolerance_missing_rotation_keys(self, tmp_path, monkeypatch):
        """An _open payload predating this change (neither session_tokens
        nor prior_session_ids) loads cleanly and behaves as a fresh
        accumulator (0, empty chain) — both fields are read with
        .get(..., default) everywhere."""
        self._setup_daily(tmp_path, monkeypatch)

        buf1 = SessionBuffer(
            session_dir=tmp_path / "sessions",
            state_dir=tmp_path / "state",
            idle_timeout_minutes=10,
        )
        buf1.append("conv-legacy", "user", "hello")
        # Simulate a pre-upgrade snapshot: strip the two new keys.
        buf1._open["conv-legacy"].pop("session_tokens", None)
        buf1._open["conv-legacy"].pop("prior_session_ids", None)
        assert buf1.save_snapshot()

        buf2 = SessionBuffer(
            session_dir=tmp_path / "sessions",
            state_dir=tmp_path / "state",
            idle_timeout_minutes=10,
        )
        assert buf2.load_snapshot()
        assert "session_tokens" not in buf2._open["conv-legacy"]
        assert "prior_session_ids" not in buf2._open["conv-legacy"]

        # The next turn resumes cleanly: accumulator starts at 0 (bounded
        # one-time over-run of at most one cap) and the chain starts empty.
        buf2.append("conv-legacy", "user", "resumed turn")
        assert buf2._open["conv-legacy"]["session_tokens"] == self._turn_tokens("resumed turn")
        assert buf2._open["conv-legacy"].get("prior_session_ids", []) == []

    def test_prune_drops_retired_prior_chain_member_leaves_conversation_open(self, tmp_path):
        """Retiring a PRIOR chain member (a size-rotated-out session that is
        now consolidated) drops it from prior_session_ids but leaves the
        conversation's _open entry live; retiring the CURRENT session still
        evicts the whole entry (existing behaviour)."""
        buf = SessionBuffer(
            session_dir=tmp_path / "sessions",
            state_dir=tmp_path / "state",
            idle_timeout_minutes=10,
        )
        turn_text = ("word " * 100).strip()
        self._drive_to_size_rotation(buf, "conv-prune", turn_text)
        prior_ids = list(buf._open["conv-prune"]["prior_session_ids"])
        assert len(prior_ids) >= 1
        current_id = buf._open["conv-prune"]["session_id"]

        # Retire the PRIOR chain member only.
        buf._prune_open_for_retired_sessions([prior_ids[0]])
        assert "conv-prune" in buf._open, "conversation must stay open"
        assert prior_ids[0] not in buf._open["conv-prune"]["prior_session_ids"]
        assert buf._open["conv-prune"]["session_id"] == current_id

        # Retiring the CURRENT session still evicts the whole entry.
        buf._prune_open_for_retired_sessions([current_id])
        assert "conv-prune" not in buf._open

    def test_document_chunk_path_never_rotates_chain_read_unaffected(self, tmp_path):
        """append_document_chunk never rotates (no _resolve_session_id
        call at all) and get_conversation_turns still resolves it by
        treating the routing handle as the session id directly — no
        prior_session_ids chain ever forms for a doc-chunk session."""
        buf = SessionBuffer(
            session_dir=tmp_path / "sessions",
            state_dir=tmp_path / "state",
            idle_timeout_minutes=10,
        )
        chunk_id = "doc-2-c000"
        buf.set_speaker(chunk_id, "spk-a", "Alice")
        buf.set_document_metadata(chunk_id, doc_id="doc-2", chunk_count=1)
        # Text well over the size cap — must NOT rotate for a doc chunk.
        huge_text = ("word " * 5000).strip()
        buf.append_document_chunk(chunk_id, "user", huge_text)

        assert len(buf.get_pending()) == 1
        assert buf._open[chunk_id].get("prior_session_ids") in (None, [])
        turns = buf.get_conversation_turns(chunk_id)
        assert [t["text"] for t in turns] == [huge_text]

    def test_rotation_decision_margin_inside_vs_outside_the_cap(self, tmp_path):
        """A turn sequence well INSIDE the cap never rotates; a sequence
        whose second turn pushes the total comfortably (50+ words) PAST
        the cap does. The cap constant itself is ratio-invariant by
        construction (it is built through
        paramem.utils.tokens.envelope_derived_cap_tokens /
        words_to_estimator_tokens, whose ratio-cancellation property is
        pinned generically in tests/test_tokens.py and, for the document
        shape specifically, in
        TestDocMaxTokensDerivation::test_ratio_cancellation_invariant) —
        this test pins the session_buffer-specific consumption of that
        cap, not the cancellation property itself, which is not
        re-derived here (no duplicate invariant)."""
        buf = SessionBuffer(
            session_dir=tmp_path / "sessions",
            state_dir=tmp_path / "state",
            idle_timeout_minutes=10,
        )
        cap_words = _TRANSCRIPT_MAX_TOKENS / MEASURED_TOKENS_PER_WORD

        # Well inside the cap: a handful of short turns.
        for i in range(3):
            buf.append("conv-inside", "user", f"short turn {i}")
        assert len(buf.get_pending()) == 1

        # Well outside the cap: two turns whose combined word count
        # exceeds cap_words by a comfortable margin (50+ words).
        turn_words = int(cap_words) + 200
        buf.append("conv-outside", "user", ("word " * turn_words).strip())
        buf.append("conv-outside", "user", ("word " * turn_words).strip())
        assert len(buf.get_pending()) == 3  # 1 (conv-inside) + 2 (conv-outside rotated)


# ---------------------------------------------------------------------------
# claim_sessions_for_speaker on an open (unretired) session
# ---------------------------------------------------------------------------


class TestClaimNotRevertedByNextAppend:
    def test_claim_survives_subsequent_append(self, tmp_path):
        """Retro-claiming an orphan session's speaker tags is not reverted by
        a later append to the same (still-open) conversation."""
        import math

        from paramem.server.speaker import SpeakerStore

        buf = SessionBuffer(session_dir=tmp_path / "sessions", state_dir=tmp_path / "state")
        v = [0.5, 0.3, 0.7, 0.1, 0.4, 0.6, 0.2, 0.8]
        norm = math.sqrt(sum(x * x for x in v))
        embedding = [x / norm for x in v]

        store = SpeakerStore(tmp_path / "profiles.json")
        speaker_id = store.enroll("Alex", embedding)

        buf.append("conv-orphan", "user", "hello there", embedding=embedding)
        claimed = buf.claim_sessions_for_speaker(speaker_id, "Alex", store)
        assert claimed == 1

        session_id = buf.get_pending()[0]["session_id"]
        assert buf._turns[session_id][0]["speaker_id"] == speaker_id

        # A later turn on the SAME conversation_id (still within the idle
        # window, so it joins the same open session) must not revert the
        # claim: append()'s fallback reads _open, which was never told
        # about the claim (claim_sessions_for_speaker rewrites _turns
        # directly) — the already-written turn's speaker_id must survive.
        buf.append("conv-orphan", "assistant", "nice to meet you")
        turns = buf._turns[session_id]
        assert turns[0]["speaker_id"] == speaker_id, (
            "claim on turn 0 must not be reverted by a later append"
        )


# ---------------------------------------------------------------------------
# Timestamp propagation: session started_at threads through get_pending
# ---------------------------------------------------------------------------


class TestStartedAtEndedAt:
    def test_get_pending_exposes_started_at_and_ended_at(self, tmp_path):
        buf = SessionBuffer(session_dir=tmp_path / "sessions", state_dir=tmp_path / "state")
        buf.append("conv-1", "user", "first")
        buf.append("conv-1", "assistant", "second")

        pending = buf.get_pending()
        assert len(pending) == 1
        session = pending[0]
        assert "started_at" in session
        assert "ended_at" in session
        # started_at == turns[0].timestamp; ended_at == turns[-1].timestamp
        turns = buf._turns[session["session_id"]]
        assert session["started_at"] == turns[0]["timestamp"]
        assert session["ended_at"] == turns[-1]["timestamp"]


# ---------------------------------------------------------------------------
# Durability: fsync on per-turn append, atomic rewrite in
# claim_sessions_for_speaker, torn-tail tolerance in _read_jsonl.
# ---------------------------------------------------------------------------


class TestAppendTurnFsync:
    def test_append_calls_fsync(self, tmp_path, monkeypatch):
        """_append_turn's JSONL write is followed by an explicit fsync, not
        just a buffered write — a served turn must survive a host power
        loss, not just an ordinary process exit."""
        import os

        buf = SessionBuffer(session_dir=tmp_path / "sessions", state_dir=tmp_path / "state")

        calls = []
        real_fsync = os.fsync
        monkeypatch.setattr(
            "paramem.server.session_buffer.os.fsync",
            lambda fd: (calls.append(fd), real_fsync(fd))[1],
        )

        buf.append("conv-1", "user", "hello")

        assert calls, "os.fsync must be called after writing a turn"

    def test_turn_readable_after_reopen(self, tmp_path):
        """A behavioral proxy for durability: a turn appended by one
        SessionBuffer instance is readable by a fresh instance that
        rehydrates from the same session_dir (simulating a restart after
        the writing process exited)."""
        buf1 = SessionBuffer(session_dir=tmp_path / "sessions", state_dir=tmp_path / "state")
        buf1.append("conv-1", "user", "hello durable world")

        buf2 = SessionBuffer(session_dir=tmp_path / "sessions", state_dir=tmp_path / "state")
        buf2.rehydrate_from_disk()

        pending = buf2.get_pending()
        assert len(pending) == 1
        assert "hello durable world" in pending[0]["transcript"]


class TestClaimSessionsAtomicRewrite:
    def _enroll_and_embed(self, tmp_path):
        import math

        from paramem.server.speaker import SpeakerStore

        v = [0.5, 0.3, 0.7, 0.1, 0.4, 0.6, 0.2, 0.8]
        norm = math.sqrt(sum(x * x for x in v))
        embedding = [x / norm for x in v]
        store = SpeakerStore(tmp_path / "profiles.json")
        speaker_id = store.enroll("Alex", embedding)
        return store, speaker_id, embedding

    def test_in_memory_claim_rewrite_leaves_no_tmp_file(self, buf, tmp_path):
        """claim_sessions_for_speaker's in-memory rewrite site writes via the
        atomic tmp + fsync + rename helper — no ``.tmp`` sibling survives a
        successful call, and the rewritten file holds every turn."""
        store, speaker_id, embedding = self._enroll_and_embed(tmp_path)

        buf.append("conv-orphan", "user", "hello there", embedding=embedding)
        buf.append("conv-orphan", "assistant", "hi back")
        session_id = buf.get_pending()[0]["session_id"]

        claimed = buf.claim_sessions_for_speaker(speaker_id, "Alex", store)
        assert claimed == 1

        jsonl_path = buf.session_dir / f"{session_id}.jsonl"
        tmp_files = list(buf.session_dir.glob("*.tmp"))
        assert not tmp_files, f"leftover tmp files after claim rewrite: {tmp_files}"

        turns = buf._read_jsonl(jsonl_path)
        assert len(turns) == 2
        assert all(t["speaker_id"] == speaker_id for t in turns)

    def test_disk_only_claim_rewrite_leaves_no_tmp_file(self, tmp_path):
        """Same atomicity guarantee for the disk-only-session branch of
        claim_sessions_for_speaker (session not yet loaded into RAM)."""
        store, speaker_id, embedding = self._enroll_and_embed(tmp_path)

        buf1 = SessionBuffer(session_dir=tmp_path / "sessions", state_dir=tmp_path / "state")
        buf1.append("conv-orphan", "user", "hello there", embedding=embedding)
        session_id = buf1.get_pending()[0]["session_id"]

        # Fresh buffer that has NOT loaded this session into RAM — forces
        # the disk-only branch.
        buf2 = SessionBuffer(session_dir=tmp_path / "sessions", state_dir=tmp_path / "state")
        claimed = buf2.claim_sessions_for_speaker(speaker_id, "Alex", store)
        assert claimed == 1

        jsonl_path = buf2.session_dir / f"{session_id}.jsonl"
        tmp_files = list(buf2.session_dir.glob("*.tmp"))
        assert not tmp_files, f"leftover tmp files after claim rewrite: {tmp_files}"

        turns = buf2._read_jsonl(jsonl_path)
        assert len(turns) == 1
        assert turns[0]["speaker_id"] == speaker_id


class TestReadJsonlTornTail:
    def test_tolerates_torn_final_line(self, tmp_path, caplog):
        """A torn (partially-written) final line does not blow up the read —
        the well-formed turns before it are still returned, and the
        malformed line is logged, not silently dropped without a trace."""
        import json
        import logging

        path = tmp_path / "conv-1.jsonl"
        good_turn = {"role": "user", "text": "hello", "speaker": None, "speaker_id": None}
        with open(path, "w") as f:
            f.write(json.dumps(good_turn) + "\n")
            f.write('{"role": "user", "text": "cut off mid-wri')  # torn, no closing brace

        caplog.set_level(logging.WARNING, logger="paramem.server.session_buffer")
        turns = SessionBuffer._read_jsonl(path)

        assert turns == [good_turn]
        assert any("malformed" in r.message for r in caplog.records), (
            f"Expected WARNING about malformed line; got: {[r.message for r in caplog.records]}"
        )

    def test_well_formed_file_unaffected(self, tmp_path):
        """No regression for the common case: a clean file with no torn
        lines round-trips exactly."""
        import json

        path = tmp_path / "conv-1.jsonl"
        turn_a = {"role": "user", "text": "a"}
        turn_b = {"role": "assistant", "text": "b"}
        with open(path, "w") as f:
            f.write(json.dumps(turn_a) + "\n")
            f.write(json.dumps(turn_b) + "\n")

        turns = SessionBuffer._read_jsonl(path)
        assert turns == [turn_a, turn_b]
