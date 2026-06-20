"""Session transcript buffering — accumulates conversation turns.

Each conversation tracks speaker identity and conversation state.
Speaker names are prefixed into transcript lines so the graph extractor
attributes facts to the correct entity.

Every user turn stores a voice embedding fingerprint alongside the text.
When a speaker enrolls, pending sessions are retroactively claimed by
matching stored embeddings against the new profile.

Privacy mode (persist=False): transcripts live only in RAM. After
consolidation the knowledge is in the adapter weights — no textual
traces remain on disk. Set persist=True (debug mode) to write JSONL
files for inspection.

Document ingest: each chunk becomes a separate session whose id is
``<doc_id>-c<chunk_index:03d>``.  All chunk sessions for the same
document must retire together (document-atomic retirement) — a partial
success leaves all of that document's chunks pending for the next cycle.
The original document bytes are stored as ``<doc_id>.origdoc`` in
``session_dir`` until the document is retired.
"""

import json
import logging
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from paramem.backup.encryption import (
    envelope_decrypt_bytes,
    envelope_encrypt_bytes,
)
from paramem.server import retry_state as _retry_state

logger = logging.getLogger(__name__)


def _snapshots_enabled() -> bool:
    """Return True when the daily age identity is loadable.

    Snapshots need a key to encrypt with; without one, the
    ``save_snapshot`` / ``load_snapshot`` pair no-ops.

    Late-binds ``key_store`` attrs so tests that monkeypatch
    ``paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT`` see their override.
    """
    from paramem.backup import key_store as _ks

    return _ks.daily_identity_loadable(_ks.DAILY_KEY_PATH_DEFAULT)


# Session conversation states
STATE_NEW = "new"
STATE_IDENTIFIED = "identified"


class SessionBuffer:
    """Buffers conversation transcripts for later consolidation.

    When persist=True (debug mode), each conversation gets a JSONL file
    on disk. When persist=False (production), transcripts live only in
    RAM and are discarded after consolidation.

    Speaker identity is tracked per conversation_id in memory. When a
    speaker is identified, their name is prefixed into transcript lines
    so the graph extractor can attribute facts to the correct entity.
    """

    def __init__(
        self,
        session_dir: Path,
        state_dir: Path,
        retain_sessions: bool = True,
        debug: bool = False,
        consolidation_retry_cap: int = 3,
    ):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._state_dir = Path(state_dir)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self.retain_sessions = retain_sessions
        self.debug = debug
        self._consolidation_retry_cap = consolidation_retry_cap
        self._snapshot_path = self.session_dir / "session_snapshot.enc"

        if _snapshots_enabled():
            logger.info(
                "Session snapshots enabled — mid-turn state will persist across "
                "graceful restarts via envelope_encrypt_bytes"
            )
        else:
            logger.info(
                "Session snapshots disabled — no key material loaded; mid-turn "
                "state is ephemeral across restarts"
            )

        # In-memory session state: conversation_id → {speaker, speaker_id, state}
        self._sessions: dict[str, dict] = {}
        # In-memory turn storage (always populated, sole source when debug=False)
        self._turns: dict[str, list[dict]] = defaultdict(list)

    def get_session_state(self, conversation_id: str) -> str:
        """Return the conversation state for this session."""
        return self._sessions.get(conversation_id, {}).get("state", STATE_NEW)

    def get_speaker(self, conversation_id: str) -> str | None:
        """Return the identified speaker name for this session, or None."""
        return self._sessions.get(conversation_id, {}).get("speaker")

    def get_speaker_id(self, conversation_id: str) -> str | None:
        """Return the speaker ID for this session, or None."""
        return self._sessions.get(conversation_id, {}).get("speaker_id")

    def set_state(self, conversation_id: str, state: str) -> None:
        """Update the conversation state."""
        if conversation_id not in self._sessions:
            self._sessions[conversation_id] = {"speaker": None, "state": STATE_NEW}
        self._sessions[conversation_id]["state"] = state

    def set_speaker(self, conversation_id: str, speaker_id: str, speaker_name: str) -> None:
        """Store the identified speaker and mark as identified."""
        if conversation_id not in self._sessions:
            self._sessions[conversation_id] = {"speaker": None, "state": STATE_NEW}
        self._sessions[conversation_id]["speaker"] = speaker_name
        self._sessions[conversation_id]["speaker_id"] = speaker_id
        self._sessions[conversation_id]["state"] = STATE_IDENTIFIED

    def set_document_metadata(
        self,
        conversation_id: str,
        *,
        doc_id: str,
        chunk_count: int,
    ) -> None:
        """Record document-group metadata for a chunk session.

        Called immediately after :meth:`set_speaker` by the ingest handler
        for every chunk session.  Stores ``doc_id`` and ``chunk_count`` in
        the per-session state dict so :meth:`retirable` can group chunks by
        document and apply document-atomic retirement.

        Args:
            conversation_id: Chunk session identifier
                (``<doc_id>-c<chunk_index:03d>``).
            doc_id: Document group identifier shared by all chunks from the
                same ingest request (``"doc-" + secrets.token_hex(4)``).
            chunk_count: Total number of chunks in this document (i.e.
                ``len(request.sessions)``).  Used as a cross-check that
                no chunk sessions are missing before the group retires.
        """
        if conversation_id not in self._sessions:
            self._sessions[conversation_id] = {"speaker": None, "state": STATE_NEW}
        self._sessions[conversation_id]["doc_id"] = doc_id
        self._sessions[conversation_id]["chunk_count"] = chunk_count

    def set_pending_embedding(self, conversation_id: str, embedding: list[float]) -> None:
        """Store a voice embedding for deferred enrollment (awaiting name)."""
        if conversation_id not in self._sessions:
            self._sessions[conversation_id] = {"speaker": None, "state": STATE_NEW}
        self._sessions[conversation_id]["pending_embedding"] = embedding

    def get_pending_embedding(self, conversation_id: str) -> list[float] | None:
        """Retrieve and clear the pending embedding for enrollment."""
        session = self._sessions.get(conversation_id, {})
        return session.pop("pending_embedding", None)

    def append(
        self,
        conversation_id: str,
        role: str,
        text: str,
        embedding: list[float] | None = None,
        metadata: dict | None = None,
        speaker_id: str | None = None,
        speaker: str | None = None,
    ) -> None:
        """Append a turn to the conversation transcript.

        For user turns from voice, embedding is the voice fingerprint
        from that utterance. Stored for retroactive speaker attribution.

        Args:
            conversation_id: Unique session identifier.
            role: Turn role (``"user"`` or ``"assistant"``).
            text: Turn text content.
            embedding: Optional voice fingerprint for speaker matching.
            metadata: Optional dict attached to the entry as ``"metadata"``.
                Used by the document ingest path to carry
                ``{"source_type": "document", "doc_title": str,
                "chunk_index": int, "source_path": str}``.
                Existing transcript turns without metadata stay
                schema-compatible — the field is simply absent.
            speaker_id: Resolved speaker ID for this turn. When provided, it is
                authoritative and written directly — the caller's resolved
                identity is the single source of truth, so the write path no
                longer depends on a prior ``set_speaker`` having populated
                session state (the gap that silently dropped token-authenticated
                text sessions). When ``None`` (default), falls back to session
                state via ``get_speaker_id`` — preserving voice / anon-promotion
                / document-ingest callers that ``set_speaker`` before appending.
            speaker: Display name companion to *speaker_id*, same precedence.
        """
        if speaker_id is None:
            speaker_id = self.get_speaker_id(conversation_id)
            speaker = self.get_speaker(conversation_id)
        entry = {
            "role": role,
            "text": text,
            "speaker": speaker,
            "speaker_id": speaker_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if embedding and role == "user":
            entry["embedding"] = embedding
        if metadata is not None:
            entry["metadata"] = metadata

        self._turns[conversation_id].append(entry)

        # Pending sessions ALWAYS persist on disk until consumed by a
        # consolidation (2026-05-14 invariant — independent of debug / mode).
        path = self.session_dir / f"{conversation_id}.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def claim_sessions_for_speaker(self, speaker_id: str, speaker_name: str, speaker_store) -> int:
        """Retroactively claim pending sessions whose embeddings match.

        Scans all pending sessions for user turns with stored embeddings.
        If the embedding matches the given speaker at high confidence,
        rewrites all turns with speaker_id/speaker tags.

        Returns the number of sessions claimed.
        """
        claimed = 0

        # Claim from in-memory turns
        for conv_id, turns in self._turns.items():
            already_claimed = any(t.get("speaker_id") for t in turns)
            if already_claimed:
                continue

            has_match = False
            for turn in turns:
                emb = turn.get("embedding")
                if emb and turn.get("role") == "user":
                    match = speaker_store.match(emb)
                    if match.speaker_id == speaker_id and not match.tentative:
                        has_match = True
                        break

            if not has_match:
                continue

            for turn in turns:
                turn["speaker"] = speaker_name
                turn["speaker_id"] = speaker_id
            claimed += 1
            logger.info("Claimed session %s for speaker %s", conv_id, speaker_name)

            # Sync to disk — JSONL is the durable representation of pending
            # sessions (2026-05-14 invariant, always-on).
            path = self.session_dir / f"{conv_id}.jsonl"
            if path.exists():
                with open(path, "w") as f:
                    for turn in turns:
                        f.write(json.dumps(turn) + "\n")

        # Also check disk-only sessions (pending JSONL files not yet loaded
        # into RAM, e.g. after a graceful exit before consolidation).
        for path in sorted(self.session_dir.glob("*.jsonl")):
            conv_id = path.stem
            if conv_id in self._turns:
                continue  # already handled above

            lines = self._read_jsonl(path)
            if not lines:
                continue

            already_claimed = any(t.get("speaker_id") for t in lines)
            if already_claimed:
                continue

            has_match = False
            for turn in lines:
                emb = turn.get("embedding")
                if emb and turn.get("role") == "user":
                    match = speaker_store.match(emb)
                    if match.speaker_id == speaker_id and not match.tentative:
                        has_match = True
                        break

            if not has_match:
                continue

            with open(path, "w") as f:
                for turn in lines:
                    turn["speaker"] = speaker_name
                    turn["speaker_id"] = speaker_id
                    f.write(json.dumps(turn) + "\n")

            claimed += 1
            logger.info("Claimed session %s for speaker %s (disk)", path.stem, speaker_name)

        return claimed

    def get_pending(self) -> list[dict]:
        """Return all pending (non-archived) session transcripts.

        Returns list of dicts with keys:
        - ``"session_id"`` — unique session identifier.
        - ``"transcript"`` — formatted conversation text.
        - ``"speaker_id"`` — dominant speaker id (None if none identified).
        - ``"source_type"`` — ``"transcript"`` or ``"document"``, read from
          the first turn's ``metadata`` dict.  Defaults to ``"transcript"``
          for existing turns without metadata.
        - ``"doc_title"`` — document title from the first turn's ``metadata``
          dict, or ``None`` for transcript sessions.

        ``_format_turns`` return shape is unchanged.  ``source_type`` and
        ``doc_title`` are read directly from the first turn's ``metadata``
        after the ``_format_turns`` call — they are not threaded through
        the tuple.
        """
        pending = []
        seen_ids = set()

        # In-memory sessions
        for conv_id, turns in self._turns.items():
            seen_ids.add(conv_id)
            formatted, session_speaker_id = self._format_turns(turns)
            if formatted:
                first_meta = turns[0].get("metadata", {}) if turns else {}
                pending.append(
                    {
                        "session_id": conv_id,
                        "transcript": "\n".join(formatted),
                        "speaker_id": session_speaker_id,
                        "source_type": first_meta.get("source_type", "transcript"),
                        "doc_title": first_meta.get("doc_title"),
                    }
                )

        # Disk-only sessions (e.g. pending JSONL on cold start after a
        # graceful exit or unclean restart).  Per the 2026-05-14 invariant,
        # JSONL is the durable representation of pending state and is read
        # back unconditionally.
        for path in sorted(self.session_dir.glob("*.jsonl")):
            conv_id = path.stem
            if conv_id in seen_ids:
                continue
            turns = self._read_jsonl(path)
            formatted, session_speaker_id = self._format_turns(turns)
            if formatted:
                first_meta = turns[0].get("metadata", {}) if turns else {}
                pending.append(
                    {
                        "session_id": conv_id,
                        "transcript": "\n".join(formatted),
                        "speaker_id": session_speaker_id,
                        "source_type": first_meta.get("source_type", "transcript"),
                        "doc_title": first_meta.get("doc_title"),
                    }
                )

        return pending

    def rehydrate_from_disk(self) -> int:
        """Load pending JSONL files into ``self._turns`` and ``self._sessions``.

        Called at lifespan startup so the in-memory state matches what's on
        disk before any chat handler can append.  Per the 2026-05-14
        invariant, pending JSONL is the durable source of truth for
        unconsolidated sessions; cold-start must restore it.

        Returns the number of sessions rehydrated.  Idempotent: already-loaded
        conversation IDs are skipped.
        """
        loaded = 0
        for path in sorted(self.session_dir.glob("*.jsonl")):
            conv_id = path.stem
            if conv_id in self._turns:
                continue
            turns = self._read_jsonl(path)
            if not turns:
                continue
            self._turns[conv_id] = turns
            # Reconstruct minimal _sessions state for any turn carrying a
            # speaker — speaker name/id are derived from the most recent
            # tagged turn.
            for t in reversed(turns):
                sid = t.get("speaker_id")
                if sid:
                    self._sessions.setdefault(
                        conv_id,
                        {
                            "speaker": t.get("speaker"),
                            "speaker_id": sid,
                            "state": STATE_IDENTIFIED,
                        },
                    )
                    break
            # Restore document-group metadata from the first turn's metadata
            # field so retirable() works correctly after a server restart.
            if turns:
                first_meta = turns[0].get("metadata", {}) or {}
                doc_id = first_meta.get("doc_id")
                chunk_count = first_meta.get("chunk_count")
                if doc_id is not None and chunk_count is not None:
                    session_state = self._sessions.setdefault(conv_id, {})
                    session_state.setdefault("doc_id", doc_id)
                    session_state.setdefault("chunk_count", chunk_count)
            loaded += 1
        if loaded:
            logger.info(
                "SessionBuffer.rehydrate_from_disk: restored %d pending session(s) from %s",
                loaded,
                self.session_dir,
            )
        return loaded

    def write_origdoc(self, doc_id: str, raw_bytes: bytes) -> None:
        """Write the original document bytes to ``session_dir/<doc_id>.origdoc``.

        Called once per ingest request after the chunk sessions are queued.
        The file persists until :meth:`mark_consolidated` (retaining branch:
        moved under ``retention_dir/<doc_id>/``) or :meth:`discard_sessions`
        (always deleted) finishes for the document group.

        Args:
            doc_id: Document group identifier (``"doc-" + secrets.token_hex(4)``).
            raw_bytes: Raw bytes of the original document file.
        """
        dest = self.locate_origdoc(doc_id)
        dest.write_bytes(raw_bytes)
        logger.debug("Stored origdoc: %s (%d bytes)", dest, len(raw_bytes))

    def locate_origdoc(self, doc_id: str) -> Path:
        """Return the canonical path for ``<doc_id>.origdoc`` in *session_dir*.

        This is the single source of truth for the origdoc path literal.
        Callers that need an existence check should call ``.exists()`` on the
        returned value; this method does not perform that check.

        Args:
            doc_id: Document group identifier.

        Returns:
            :class:`Path` to ``session_dir/<doc_id>.origdoc`` (may or may not
            exist on disk).
        """
        return self.session_dir / f"{doc_id}.origdoc"

    def retirable(self, completed_ids: set[str]) -> list[str]:
        """Filter *completed_ids* by document-atomic retirement rules.

        Transcript sessions (sessions without a ``doc_id`` in their state)
        are returned unconditionally for every id in *completed_ids*.

        Document chunk sessions retire ONLY when ALL chunks for the same
        ``doc_id`` are present in *completed_ids* AND the number of completed
        chunks matches the stored ``chunk_count``.  If any chunk is missing,
        the entire document group is withheld from the result — all chunks
        stay pending until the next consolidation cycle picks them up.

        This method does not mutate any state; it is a pure filter.

        Args:
            completed_ids: Set of session ids whose extraction succeeded.

        Returns:
            List of session ids that are safe to retire this cycle.
        """
        if not completed_ids:
            return []

        # Group document chunk sessions by doc_id.
        # Keys: all doc_id values seen among completed_ids.
        doc_chunks_completed: dict[str, list[str]] = {}
        transcript_ids: list[str] = []

        for sid in completed_ids:
            session_meta = self._sessions.get(sid, {})
            doc_id = session_meta.get("doc_id")
            if doc_id is None:
                transcript_ids.append(sid)
            else:
                doc_chunks_completed.setdefault(doc_id, []).append(sid)

        result = list(transcript_ids)

        for doc_id, chunk_sids in doc_chunks_completed.items():
            # Find the expected chunk_count for this doc_id by scanning the
            # completed chunks (all should agree on the same value).
            expected = None
            for sid in chunk_sids:
                cc = self._sessions.get(sid, {}).get("chunk_count")
                if cc is not None:
                    expected = cc
                    break

            if expected is None:
                # Metadata missing — retire as-is rather than block forever.
                logger.warning(
                    "retirable: doc_id=%s missing chunk_count metadata; retiring %d chunk(s)",
                    doc_id,
                    len(chunk_sids),
                )
                result.extend(chunk_sids)
                continue

            if len(chunk_sids) < expected:
                # Not all chunks completed — hold the entire document back.
                logger.info(
                    "retirable: doc_id=%s holding back (%d/%d chunks completed)",
                    doc_id,
                    len(chunk_sids),
                    expected,
                )
                continue

            # All expected chunks completed.
            result.extend(chunk_sids)

        return result

    def mark_consolidated(
        self,
        session_ids: list[str],
        *,
        retention_dir: Path | None = None,
    ) -> None:
        """Consume consolidated sessions and dispose of their JSONL.

        Disposition (2026-05-14 user spec):

        - ``retain_sessions=True`` OR ``debug=True``: move JSONL to
          *retention_dir* (caller-supplied, typically
          ``loop.snapshot_dir/cycle_<N>/sessions/``).
        - both flags False: unlink.

        Always clears the in-memory state.  Callers MUST pass
        *retention_dir* when at least one of the two flags is True;
        otherwise retained transcripts are silently dropped.

        Document-atomic archival: chunk sessions that share a ``doc_id``
        are retired together.  When retaining, their chunk JSONLs AND the
        ``<doc_id>.origdoc`` blob are moved under
        ``retention_dir/<doc_id>/``, with the origdoc renamed to the
        original ``doc_filename`` recorded in the first turn's metadata.
        When deleting (privacy mode), both the chunk JSONLs and the
        origdoc are unlinked.  Transcript sessions use the existing flat
        layout (``retention_dir/<session_id>.jsonl``).
        """
        retain = self.retain_sessions or self.debug
        if retain and retention_dir is not None:
            retention_dir.mkdir(parents=True, exist_ok=True)

        # Collect doc_id → (list[session_id], doc_filename) for document chunk
        # sessions so we can archive origdoc alongside the chunk JSONLs.
        # doc_filename is resolved NOW, before turns are popped below.
        doc_id_to_chunk_sids: dict[str, list[str]] = {}
        doc_id_to_filename: dict[str, str] = {}
        for session_id in session_ids:
            session_meta = self._sessions.get(session_id, {})
            doc_id = session_meta.get("doc_id")
            if doc_id is not None:
                doc_id_to_chunk_sids.setdefault(doc_id, []).append(session_id)
                if doc_id not in doc_id_to_filename:
                    turns = self._turns.get(session_id) or []
                    fname = (turns[0].get("metadata") or {}).get("doc_filename") if turns else None
                    doc_id_to_filename[doc_id] = fname or f"{doc_id}.bin"

        # Reverse map: session_id → doc_id, built before any state is popped.
        sid_to_doc_id: dict[str, str] = {
            sid: doc_id for doc_id, sids in doc_id_to_chunk_sids.items() for sid in sids
        }

        for session_id in session_ids:
            self._turns.pop(session_id, None)
            self._sessions.pop(session_id, None)
            source = self.session_dir / f"{session_id}.jsonl"
            if not source.exists():
                continue
            if retain and retention_dir is not None:
                doc_id = sid_to_doc_id.get(session_id)
                if doc_id is not None:
                    # Chunk JSONL co-located with the origdoc under
                    # retention_dir/<doc_id>/<session_id>.jsonl.
                    doc_ret_dir = retention_dir / doc_id
                    doc_ret_dir.mkdir(parents=True, exist_ok=True)
                    dest = doc_ret_dir / f"{session_id}.jsonl"
                else:
                    # Transcript sessions keep the flat layout.
                    dest = retention_dir / f"{session_id}.jsonl"
                shutil.move(str(source), str(dest))
                logger.info("Retained session %s → %s", session_id, dest)
            else:
                source.unlink()
                logger.info("Deleted session transcript: %s", session_id)

        # Handle origdoc archival or deletion for each retiring document group.
        for doc_id in doc_id_to_chunk_sids:
            origdoc = self.locate_origdoc(doc_id)
            if not origdoc.exists():
                continue
            if retain and retention_dir is not None:
                # Archive origdoc under retention_dir/<doc_id>/ renamed to
                # the original filename collected in the first pass above.
                doc_filename = doc_id_to_filename.get(doc_id, f"{doc_id}.bin")
                doc_ret_dir = retention_dir / doc_id
                doc_ret_dir.mkdir(parents=True, exist_ok=True)
                dest = doc_ret_dir / doc_filename
                shutil.move(str(origdoc), str(dest))
                logger.info("Archived origdoc %s → %s", doc_id, dest)
            else:
                origdoc.unlink()
                logger.info("Deleted origdoc: %s", doc_id)

        # Clear durable retry-count rows for all retired sessions so stale
        # entries do not accumulate and do not mis-count a future session that
        # reuses an id.  Non-fatal: a failed clear is logged but does not roll
        # back the retirement (the session is gone from _sessions/_turns already).
        if session_ids:
            try:
                _retry_state.clear_retry_counts(self._state_dir, list(session_ids))
            except Exception:
                logger.exception(
                    "SessionBuffer.mark_consolidated: failed to clear durable retry counts "
                    "for %d session(s) (non-fatal)",
                    len(session_ids),
                )

    def discard_sessions(self, session_ids: list[str]) -> None:
        """Drop named sessions from the in-memory queue and disk state.

        Unlike :meth:`mark_consolidated`, this method does NOT retain
        the JSONL — it is always deleted outright.  Designed for the
        cancel path (``POST /ingest-sessions/cancel``), where the
        operator wants to remove queued document chunks without
        running consolidation.

        For document chunk sessions, also unlinks the ``<doc_id>.origdoc``
        blob when all known chunk sessions for a document are being
        discarded (or the blob exists without any remaining sessions).
        This prevents an orphaned original-doc file when the operator
        cancels a full ingest request.

        Silent no-op for unknown session ids (mirrors
        :meth:`mark_consolidated`'s tolerance).

        Args:
            session_ids: Session identifiers to discard.
        """
        # Collect doc_ids for document chunk sessions before popping state.
        doc_ids_touched: set[str] = set()
        for session_id in session_ids:
            doc_id = self._sessions.get(session_id, {}).get("doc_id")
            if doc_id is not None:
                doc_ids_touched.add(doc_id)

        for session_id in session_ids:
            self._turns.pop(session_id, None)
            self._sessions.pop(session_id, None)
            path = self.session_dir / f"{session_id}.jsonl"
            if path.exists():
                path.unlink()
                logger.info("Discarded session file: %s", session_id)

        # Unlink origdoc for any document group that no longer has chunk
        # sessions in the buffer.  If any chunk session for a doc_id still
        # exists in _sessions (e.g. only a partial cancel), leave it.
        for doc_id in doc_ids_touched:
            origdoc = self.locate_origdoc(doc_id)
            if not origdoc.exists():
                continue
            remaining = any(s.get("doc_id") == doc_id for s in self._sessions.values())
            if not remaining:
                origdoc.unlink()
                logger.info("Deleted origdoc for discarded doc: %s", doc_id)

        # Clear durable retry-count rows for discarded sessions (same R3-guard
        # as mark_consolidated; stale rows must not accumulate or mis-count).
        if session_ids:
            try:
                _retry_state.clear_retry_counts(self._state_dir, list(session_ids))
            except Exception:
                logger.exception(
                    "SessionBuffer.discard_sessions: failed to clear durable retry counts "
                    "for %d session(s) (non-fatal)",
                    len(session_ids),
                )

    def hydrate_retry_counts(self) -> None:
        """Seed in-memory retry counts from the durable ``consolidation_retry.json``.

        Called at boot after :meth:`load_snapshot` so the durable store is the
        authoritative source of truth.  Overwrites any ``recall_retry_count``
        values that :meth:`load_snapshot` restored from the encrypted snapshot
        — the durable file wins because it survives ungraceful restarts whereas
        the encrypted snapshot does not.

        Sessions that are in the durable file but not (yet) in ``_sessions`` are
        silently skipped; they will be reconciled when their JSONL is rehydrated.

        Non-fatal: a schema or I/O error is logged and hydration is skipped so
        the server still starts.  A subsequent :meth:`bump_retry_and_release` call
        on those sessions will re-read from disk atomically.
        """
        try:
            durable = _retry_state.read_retry_counts(self._state_dir)
        except Exception:
            logger.exception(
                "SessionBuffer.hydrate_retry_counts: failed to read durable retry counts "
                "(non-fatal; counts will be re-read on next bump)"
            )
            return
        for sid, count in durable.items():
            if sid in self._sessions:
                self._sessions[sid]["recall_retry_count"] = count

    def bump_retry_and_release(self, failed_ids: set[str]) -> list[str]:
        """Increment the durable retry counter for each failed session; release capped ones.

        Called from ``_run_interim_training`` (``app.py``) after
        ``run_consolidation_cycle`` returns for sessions whose facts were NOT
        successfully encoded into adapter weights this cycle.  Each call covers
        exactly the sessions from one cycle, so one-per-cycle cadence is
        structural.

        For each session id in *failed_ids* that is currently buffered:

        - Atomically increment the durable ``recall_retry_count`` via
          ``retry_state.bump_retry_count`` (crash-safe: count is on disk before
          this method returns).
        - Mirror the returned count into ``self._sessions[sid]["recall_retry_count"]``
          so in-memory state stays consistent with the durable store.
        - If the new count reaches :attr:`_consolidation_retry_cap`, add the
          session id to the returned release list.

        Released sessions are removed from the caller's ``failed_session_ids``
        set so ``_completed_session_ids()`` retires them on the current cycle
        (they are "un-pinned").  A WARNING is logged for each released session.
        The corresponding incident is recorded by the caller (not here) so the
        caller owns the per-session ``key`` for dedup.

        Session ids absent from :attr:`_sessions` are silently skipped (R3
        guard: synthetic-id leakage protection — synthetic ids are never present
        in the buffer).

        Reset-on-recall-success: when a previously-counted session passes recall
        in a cycle, the caller calls :meth:`reset_retry_count_for` before
        marking it consolidated, clearing the durable count so only consecutive
        failures accrue toward the cap.

        Restart semantics: the counter is durable — written atomically to
        ``data/state/consolidation_retry.json`` on every increment via
        ``fcntl.flock``-guarded RMW.  An ungraceful restart (TDR / host crash)
        does NOT reset the budget.  On boot, :meth:`hydrate_retry_counts` seeds
        in-memory counts from this durable file.

        Raises:
            retry_state.RetryStateCapacityError: when disk is full (ENOSPC /
                EDQUOT) during the durable write.  The caller must record a
                ``storage_capacity_reached`` incident and stop — do NOT retry or
                spin.

        Args:
            failed_ids: Session ids whose facts were not encoded this cycle.

        Returns:
            List of session ids whose retry count reached the cap.  The caller
            must remove them from its pending-failure set to un-pin them.
        """
        released: list[str] = []
        for sid in failed_ids:
            if sid not in self._sessions:
                # R3: not a buffered session (already retired or a synthetic id
                # that slipped through).  Skip silently — do not mutate state.
                continue
            new_count = _retry_state.bump_retry_count(self._state_dir, sid)
            self._sessions[sid]["recall_retry_count"] = new_count
            if new_count >= self._consolidation_retry_cap:
                logger.warning(
                    "SessionBuffer.bump_retry_and_release: session %s hit "
                    "consolidation-retry cap (%d) — releasing; facts could not be encoded",
                    sid,
                    self._consolidation_retry_cap,
                )
                released.append(sid)
        return released

    def reset_retry_count_for(self, session_id: str) -> None:
        """Clear the durable retry count for a session that passed recall.

        Called when a previously-counted session produces a clean recall result
        in a cycle (reset-on-recall-success).  Clears both the durable store
        entry and the in-memory cache so the session re-enters the retry budget
        at 0 — only consecutive failures accrue toward the cap.

        Idempotent: a no-op when the session has no durable entry.

        Non-fatal: a failed reset is logged but does not block the caller.

        Args:
            session_id: Session identifier whose retry count should be cleared.
        """
        try:
            _retry_state.reset_retry_count(self._state_dir, session_id)
        except Exception:
            logger.exception(
                "SessionBuffer.reset_retry_count_for: failed to reset durable retry count "
                "for session %s (non-fatal)",
                session_id,
            )
        if session_id in self._sessions:
            self._sessions[session_id].pop("recall_retry_count", None)

    def get_session_turns(self, conversation_id: str) -> list[dict]:
        """Read all turns from a session."""
        # In-memory first
        if conversation_id in self._turns:
            return list(self._turns[conversation_id])

        # Fall back to disk
        if self.debug:
            path = self.session_dir / f"{conversation_id}.jsonl"
            if path.exists():
                return self._read_jsonl(path)

        return []

    def get_summary(self) -> dict:
        """Per-speaker pending-session attribution summary.

        Returns:
            {
                "total": int,
                "orphaned": int,                        # pending with no speaker_id
                "oldest_age_seconds": int | None,       # age of oldest pending session
                "per_speaker": {speaker_id: count},     # matched pending per speaker
                "per_source_type": {source_type: count}, # sessions by source type
            }

        ``per_source_type`` counts sessions (not turns) by their
        ``source_type`` value.  Keys are ``"transcript"`` and ``"document"``.
        """
        per_speaker: dict[str, int] = {}
        per_source_type: dict[str, int] = {}
        orphaned = 0
        total = 0

        for session in self.get_pending():
            total += 1
            sid = session.get("speaker_id")
            if sid:
                per_speaker[sid] = per_speaker.get(sid, 0) + 1
            else:
                orphaned += 1
            st = session.get("source_type", "transcript")
            per_source_type[st] = per_source_type.get(st, 0) + 1

        # Oldest pending: largest age across in-memory sessions
        # (the oldest session has the highest age_seconds value).
        oldest_age: int | None = None
        for turns in self._turns.values():
            age = self._first_turn_age_seconds(turns)
            if age is None:
                continue
            if oldest_age is None or age > oldest_age:
                oldest_age = age

        return {
            "total": total,
            "orphaned": orphaned,
            "oldest_age_seconds": oldest_age,
            "per_speaker": per_speaker,
            "per_source_type": per_source_type,
        }

    def pending_facts(self) -> list[dict]:
        """Return attribution facts for every pending session — no policy, data only.

        Mirrors the dual-branch walk of :meth:`get_pending` exactly, including
        the ``if formatted:`` empty-turns guard in both branches, so the caller
        sees the same session population: sessions with no usable turns are
        excluded, just as :meth:`get_pending` excludes them.

        Returns a list of dicts, one per session::

            {
                "session_id":          str,
                "speaker_id":          str | None,  # dominant speaker id
                "has_voice_embedding": bool,         # any user turn carries embedding
                "age_seconds":         int | None,   # now − first-turn timestamp
            }

        ``has_voice_embedding`` is ``True`` iff at least one user-role turn in
        the session carries a non-``None`` ``"embedding"`` field.

        This method exposes facts for the caller to classify via
        :func:`~paramem.server.consolidation.classify_session`; it does NOT
        make a policy decision itself.
        """
        result = []
        seen_ids: set[str] = set()

        # In-memory sessions first
        for conv_id, turns in self._turns.items():
            seen_ids.add(conv_id)
            formatted, session_speaker_id = self._format_turns(turns)
            if not formatted:
                continue
            has_emb = any(t.get("embedding") is not None and t.get("role") == "user" for t in turns)
            result.append(
                {
                    "session_id": conv_id,
                    "speaker_id": session_speaker_id,
                    "has_voice_embedding": has_emb,
                    "age_seconds": self._first_turn_age_seconds(turns),
                }
            )

        # Disk-only sessions
        for path in sorted(self.session_dir.glob("*.jsonl")):
            conv_id = path.stem
            if conv_id in seen_ids:
                continue
            turns = self._read_jsonl(path)
            formatted, session_speaker_id = self._format_turns(turns)
            if not formatted:
                continue
            has_emb = any(t.get("embedding") is not None and t.get("role") == "user" for t in turns)
            result.append(
                {
                    "session_id": conv_id,
                    "speaker_id": session_speaker_id,
                    "has_voice_embedding": has_emb,
                    "age_seconds": self._first_turn_age_seconds(turns),
                }
            )

        return result

    @property
    def pending_count(self) -> int:
        count = len(self._turns)
        if self.debug:
            # Add disk-only sessions not already in RAM
            for path in self.session_dir.glob("*.jsonl"):
                if path.stem not in self._turns:
                    count += 1
        return count

    @staticmethod
    def _first_turn_age_seconds(turns: list[dict]) -> int | None:
        """Return seconds since the first turn's timestamp, or ``None``.

        Shared helper used by :meth:`get_summary` and :meth:`pending_facts`
        so timestamp math lives in one place.

        Coerces naive timestamps (legacy snapshots) to UTC before computing
        the delta — same defensive logic ``get_summary`` previously inlined.

        Args:
            turns: List of turn dicts.  The first element's ``"timestamp"``
                field (ISO-format string) is used.  Returns ``None`` when
                ``turns`` is empty or the field is absent / unparseable.

        Returns:
            Non-negative integer seconds, or ``None``.
        """
        if not turns:
            return None
        try:
            first_ts = datetime.fromisoformat(turns[0]["timestamp"])
        except (KeyError, ValueError, TypeError):
            return None
        if first_ts.tzinfo is None:
            first_ts = first_ts.replace(tzinfo=timezone.utc)
        return int((datetime.now(timezone.utc) - first_ts).total_seconds())

    @staticmethod
    def _format_turns(turns: list[dict]) -> tuple[list[str], str | None]:
        """Format turns into ``[user]`` / ``[assistant]`` marker lines.

        ``[user]`` / ``[assistant]`` is the single transcript marker
        format the extraction prompt's few-shots use.  Document chunks
        (which arrive at ``buffer.append`` with ``role="user"`` and no
        turn structure of their own) come out as a single
        ``[user] <chunk text>`` line — same surface form as the leading
        user turn of a transcript.

        Speaker identity is bound via the ``{speaker_context}`` slot in
        the user template (see ``paramem/graph/extractor.py:437``
        ``build_speaker_context``), not by inlining the speaker's name
        here.  ``speaker_id`` continues to track per-turn for downstream
        provenance.
        """
        formatted = []
        speaker_ids = []
        for turn in turns:
            role = turn.get("role", "unknown")
            text = turn.get("text", "")
            sid = turn.get("speaker_id")

            if sid:
                speaker_ids.append(sid)

            marker = (
                "[user]"
                if role == "user"
                else "[assistant]"
                if role == "assistant"
                else f"[{role}]"
            )
            formatted.append(f"{marker} {text}")

        session_speaker_id = None
        if speaker_ids:
            session_speaker_id = max(set(speaker_ids), key=speaker_ids.count)

        return formatted, session_speaker_id

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        """Read all turns from a JSONL file."""
        turns = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                turns.append(json.loads(line))
        return turns

    def save_snapshot(self) -> bool:
        """Write age-encrypted snapshot of in-memory state to disk.

        Called on graceful shutdown (SIGUSR1, SIGTERM). Returns True on
        success. When the daily identity is not loaded, returns False
        without writing — snapshot persistence requires a key so a restart
        on a fresh host with the key restored can read it back.
        """
        if not _snapshots_enabled():
            return False
        if not self._turns and not self._sessions:
            logger.info("No session data to snapshot")
            return True

        payload = {
            "turns": dict(self._turns),
            "sessions": self._sessions,
        }
        try:
            plaintext = json.dumps(payload).encode()
            envelope = envelope_encrypt_bytes(plaintext)
            from paramem.backup.encryption import _atomic_write_bytes

            _atomic_write_bytes(self._snapshot_path, envelope)
            logger.info(
                "Session snapshot saved: %d conversations, %d turns",
                len(self._turns),
                sum(len(v) for v in self._turns.values()),
            )
            return True
        except Exception:
            logger.exception("Failed to save session snapshot")
            return False

    def load_snapshot(self) -> bool:
        """Restore in-memory state from an age-encrypted snapshot.

        Called on startup. Decrypts via :func:`envelope_decrypt_bytes`,
        loads the payload, then deletes the file. Returns True iff a
        snapshot was actually restored.

        When the snapshot file exists but the daily identity is not
        loadable, logs a warning and returns False without unlinking —
        the file cannot be decrypted without the retired identity, but
        the operator may want to inspect / recover it manually. A
        deliberate ``paramem restore`` workflow is the right way to bring
        it back.
        """
        if not self._snapshot_path.exists():
            return False
        if not _snapshots_enabled():
            logger.warning(
                "Session snapshot present at %s but no key material is loaded "
                "— leaving in place; remove it manually or restore the key to "
                "recover its contents",
                self._snapshot_path,
            )
            return False

        try:
            raw = self._snapshot_path.read_bytes()
            plaintext = envelope_decrypt_bytes(raw)
            payload = json.loads(plaintext.decode())

            restored_turns = payload.get("turns", {})
            restored_sessions = payload.get("sessions", {})

            for conv_id, turns in restored_turns.items():
                self._turns[conv_id] = turns
            self._sessions.update(restored_sessions)

            self._snapshot_path.unlink()
            logger.info(
                "Session snapshot restored: %d conversations, %d turns",
                len(restored_turns),
                sum(len(v) for v in restored_turns.values()),
            )
            return True
        except Exception:
            logger.exception("Failed to restore session snapshot — discarding")
            self._snapshot_path.unlink(missing_ok=True)
            return False
