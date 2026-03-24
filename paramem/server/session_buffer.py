"""Session transcript buffering — accumulates conversation turns on disk.

Each conversation tracks speaker identity and conversation state.
Speaker names are prefixed into transcript lines so the graph extractor
attributes facts to the correct entity.
"""

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Session conversation states
STATE_NEW = "new"
STATE_GREETING_SENT = "greeting_sent"
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
        # In-memory session state: conversation_id → {speaker, state}
        self._sessions: dict[str, dict] = {}

    def get_session_state(self, conversation_id: str) -> str:
        """Return the conversation state for this session."""
        return self._sessions.get(conversation_id, {}).get("state", STATE_NEW)

    def get_speaker(self, conversation_id: str) -> str | None:
        """Return the identified speaker for this session, or None."""
        return self._sessions.get(conversation_id, {}).get("speaker")

    def set_state(self, conversation_id: str, state: str) -> None:
        """Update the conversation state."""
        if conversation_id not in self._sessions:
            self._sessions[conversation_id] = {"speaker": None, "state": STATE_NEW}
        self._sessions[conversation_id]["state"] = state

    def set_speaker(self, conversation_id: str, speaker: str) -> None:
        """Store the identified speaker and mark as identified."""
        if conversation_id not in self._sessions:
            self._sessions[conversation_id] = {"speaker": None, "state": STATE_NEW}
        self._sessions[conversation_id]["speaker"] = speaker
        self._sessions[conversation_id]["state"] = STATE_IDENTIFIED

    def append(self, conversation_id: str, role: str, text: str) -> None:
        """Append a turn to the conversation transcript."""
        path = self.session_dir / f"{conversation_id}.jsonl"
        speaker = self.get_speaker(conversation_id)
        entry = {
            "role": role,
            "text": text,
            "speaker": speaker,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_pending(self) -> list[dict]:
        """Return all pending (non-archived) session transcripts.

        Returns list of {"session_id": str, "transcript": str} dicts.
        The transcript is formatted as a readable conversation with
        speaker names prefixed for user turns.
        """
        pending = []
        for path in sorted(self.session_dir.glob("*.jsonl")):
            turns = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    turn = json.loads(line)
                    role = turn.get("role", "unknown")
                    text = turn.get("text", "")
                    speaker = turn.get("speaker")

                    if role == "user" and speaker:
                        turns.append(f"{speaker}: {text}")
                    else:
                        role_label = role.capitalize()
                        turns.append(f"{role_label}: {text}")

            if turns:
                pending.append(
                    {
                        "session_id": path.stem,
                        "transcript": "\n".join(turns),
                    }
                )
        return pending

    def mark_consolidated(self, session_ids: list[str]) -> None:
        """Remove or archive consolidated session files.

        When retain_sessions is True, files are archived (safety net).
        When False, files are deleted (transcripts are transient).
        """
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

    @property
    def pending_count(self) -> int:
        return len(list(self.session_dir.glob("*.jsonl")))
