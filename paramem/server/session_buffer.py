"""Session transcript buffering — accumulates conversation turns on disk.

Each conversation tracks speaker identity and conversation state.
Speaker names are prefixed into transcript lines so the graph extractor
attributes facts to the correct entity.

Every user turn stores a voice embedding fingerprint alongside the text.
When a speaker enrolls, pending sessions are retroactively claimed by
matching stored embeddings against the new profile.
"""

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Session conversation states
STATE_NEW = "new"
STATE_IDENTIFIED = "identified"


class SessionBuffer:
    """Buffers conversation transcripts to disk for later consolidation.

    Each conversation gets a JSONL file. After consolidation, files are
    moved to an archive subdirectory.

    Speaker identity is tracked per conversation_id in memory. When a
    speaker is identified, their name is prefixed into transcript lines
    so the graph extractor can attribute facts to the correct entity.
    """

    def __init__(self, session_dir: Path, retain_sessions: bool = True):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.retain_sessions = retain_sessions
        self.archive_dir = self.session_dir / "archive"
        if self.retain_sessions:
            self.archive_dir.mkdir(exist_ok=True)
        # In-memory session state: conversation_id → {speaker, speaker_id, state}
        self._sessions: dict[str, dict] = {}

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
    ) -> None:
        """Append a turn to the conversation transcript.

        For user turns from voice, embedding is the voice fingerprint
        from that utterance. Stored for retroactive speaker attribution.
        """
        path = self.session_dir / f"{conversation_id}.jsonl"
        speaker = self.get_speaker(conversation_id)
        speaker_id = self.get_speaker_id(conversation_id)
        entry = {
            "role": role,
            "text": text,
            "speaker": speaker,
            "speaker_id": speaker_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if embedding and role == "user":
            entry["embedding"] = embedding
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def claim_sessions_for_speaker(self, speaker_id: str, speaker_name: str, speaker_store) -> int:
        """Retroactively claim pending sessions whose embeddings match.

        Scans all pending JSONL files for user turns with stored embeddings.
        If the embedding matches the given speaker at high confidence,
        rewrites the file with speaker_id/speaker tags on all turns.

        Returns the number of sessions claimed.
        """
        claimed = 0
        for path in sorted(self.session_dir.glob("*.jsonl")):
            lines = []
            has_match = False
            already_claimed = False

            with open(path) as f:
                for raw_line in f:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    turn = json.loads(raw_line)
                    lines.append(turn)

                    # Skip if already has a speaker_id
                    if turn.get("speaker_id"):
                        already_claimed = True
                        break

                    # Check embedding on user turns
                    emb = turn.get("embedding")
                    if emb and turn.get("role") == "user":
                        match = speaker_store.match(emb)
                        if match.speaker_id == speaker_id and not match.tentative:
                            has_match = True

            if already_claimed or not has_match:
                continue

            # Rewrite file with speaker attribution
            with open(path, "w") as f:
                for turn in lines:
                    turn["speaker"] = speaker_name
                    turn["speaker_id"] = speaker_id
                    f.write(json.dumps(turn) + "\n")

            claimed += 1
            logger.info("Claimed session %s for speaker %s", path.stem, speaker_name)

        return claimed

    def get_pending(self) -> list[dict]:
        """Return all pending (non-archived) session transcripts.

        Returns list of {"session_id", "transcript", "speaker_id"} dicts.
        The transcript is formatted as a readable conversation with
        speaker names prefixed for user turns. speaker_id is the most
        frequent speaker_id across turns (None if no speaker identified).
        """
        pending = []
        for path in sorted(self.session_dir.glob("*.jsonl")):
            turns = []
            speaker_ids = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    turn = json.loads(line)
                    role = turn.get("role", "unknown")
                    text = turn.get("text", "")
                    speaker = turn.get("speaker")
                    sid = turn.get("speaker_id")

                    if sid:
                        speaker_ids.append(sid)

                    if role == "user" and speaker:
                        turns.append(f"{speaker}: {text}")
                    else:
                        role_label = role.capitalize()
                        turns.append(f"{role_label}: {text}")

            if turns:
                session_speaker_id = None
                if speaker_ids:
                    session_speaker_id = max(set(speaker_ids), key=speaker_ids.count)
                pending.append(
                    {
                        "session_id": path.stem,
                        "transcript": "\n".join(turns),
                        "speaker_id": session_speaker_id,
                    }
                )
        return pending

    def mark_consolidated(self, session_ids: list[str]) -> None:
        """Remove or archive consolidated session files."""
        for session_id in session_ids:
            source = self.session_dir / f"{session_id}.jsonl"
            if source.exists():
                if self.retain_sessions:
                    dest = self.archive_dir / f"{session_id}.jsonl"
                    shutil.move(str(source), str(dest))
                    logger.info("Archived session: %s", session_id)
                else:
                    source.unlink()
                    logger.info("Deleted session transcript: %s", session_id)

    def get_session_turns(self, conversation_id: str) -> list[dict]:
        """Read all turns from a session's JSONL file."""
        path = self.session_dir / f"{conversation_id}.jsonl"
        if not path.exists():
            return []
        turns = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                turns.append(json.loads(line))
        return turns

    @property
    def pending_count(self) -> int:
        return len(list(self.session_dir.glob("*.jsonl")))
