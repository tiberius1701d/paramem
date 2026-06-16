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


def test_append_explicit_speaker_id_attributes_without_set_speaker(buf):
    """Caller-supplied speaker_id is authoritative and needs no prior set_speaker.

    Regression for the token-auth gap: a per-user-token /chat resolved the
    speaker but never called set_speaker, so append() read empty session state
    and persisted speaker_id=None — consolidation then skipped the session and
    dropped the user's facts. append() now takes the resolved id explicitly.
    """
    buf.append("conv-tok", "user", "I live in Kelkheim", speaker_id="spk-tok", speaker="Tobias")
    turns = buf._turns["conv-tok"]
    assert turns[0]["speaker_id"] == "spk-tok"
    assert turns[0]["speaker"] == "Tobias"
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
    assert buf._turns["conv-vs"][0]["speaker_id"] == "spk-voice"
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
        buf.append(
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
        result = buf.retirable({"conv-1"})
        assert result == ["conv-1"]

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
        self._add_doc_chunk(buf, "doc-3-c000", "doc-3", chunk_count=2)
        self._add_doc_chunk(buf, "doc-3-c001", "doc-3", chunk_count=2)
        # Transcript completes; doc is only 1/2 done.
        result = buf.retirable({"conv-x", "doc-3-c000"})
        assert result == ["conv-x"]

    def test_mixed_transcript_and_doc_complete(self, buf):
        """Transcript and complete doc both retire together."""
        buf.append("conv-y", "user", "chat")
        self._add_doc_chunk(buf, "doc-4-c000", "doc-4", chunk_count=1)
        result = buf.retirable({"conv-y", "doc-4-c000"})
        assert sorted(result) == ["conv-y", "doc-4-c000"]

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
        buf.append(
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
        buf = SessionBuffer(session_dir=sessions_dir, retain_sessions=True, debug=False)

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
        buf = SessionBuffer(session_dir=sessions_dir, retain_sessions=False, debug=False)

        doc_id = "doc-delete1"
        self._add_doc_chunk(buf, f"{doc_id}-c000", doc_id, chunk_count=1)
        buf.write_origdoc(doc_id, b"some bytes")

        buf.mark_consolidated([f"{doc_id}-c000"], retention_dir=None)

        assert not (sessions_dir / f"{doc_id}-c000.jsonl").exists()
        assert not (sessions_dir / f"{doc_id}.origdoc").exists()

    def test_transcript_sessions_use_flat_layout(self, tmp_path):
        """Transcript sessions are archived flat under retention_dir (unchanged)."""
        sessions_dir = tmp_path / "sessions"
        buf = SessionBuffer(session_dir=sessions_dir, retain_sessions=True, debug=False)
        buf.append("conv-t1", "user", "hello")

        retention_dir = tmp_path / "retention"
        buf.mark_consolidated(["conv-t1"], retention_dir=retention_dir)

        assert (retention_dir / "conv-t1.jsonl").exists()


# ---------------------------------------------------------------------------
# discard_sessions with origdoc cleanup
# ---------------------------------------------------------------------------


class TestDiscardSessionsOrigdoc:
    def _add_doc_chunk(self, buf: SessionBuffer, session_id: str, doc_id: str) -> None:
        buf.set_speaker(session_id, "spk-1", "Alice")
        buf.set_document_metadata(session_id, doc_id=doc_id, chunk_count=1)
        buf.append(
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
        buf.append(
            f"{doc_id}-c000",
            "user",
            "chunk 0",
            metadata={"source_type": "document", "doc_id": doc_id, "chunk_count": 2},
        )

        buf.set_speaker(f"{doc_id}-c001", "spk-1", "Alice")
        buf.set_document_metadata(f"{doc_id}-c001", doc_id=doc_id, chunk_count=2)
        buf.append(
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
        assert "conv-normal" in self._session_ids_from_pending_facts(buf)

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
        # Empty in-memory entry
        buf._turns["conv-bad"]  # noqa: B018 — defaultdict side-effect: registers empty list
        (buf.session_dir / "conv-bad.jsonl").write_text("")

        pf_ids = self._session_ids_from_pending_facts(buf)
        gp_ids = self._session_ids_from_get_pending(buf)
        assert pf_ids == gp_ids
        assert pf_ids == {"conv-good"}
