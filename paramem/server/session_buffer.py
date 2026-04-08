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
"""

import json
import logging
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

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
        retain_sessions: bool = True,
        debug: bool = False,
        snapshot_key: str = "",
    ):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.retain_sessions = retain_sessions
        self.debug = debug
        self.archive_dir = self.session_dir / "archive"
        if self.retain_sessions and self.debug:
            self.archive_dir.mkdir(exist_ok=True)
        self._snapshot_path = self.session_dir / "session_snapshot.enc"

        if snapshot_key:
            self._fernet = Fernet(snapshot_key.encode())
        else:
            self._fernet = None
            logger.info("No snapshot_key configured — session snapshots disabled")

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

        self._turns[conversation_id].append(entry)

        if self.debug:
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

            # Sync to disk if persisting
            if self.debug:
                path = self.session_dir / f"{conv_id}.jsonl"
                if path.exists():
                    with open(path, "w") as f:
                        for turn in turns:
                            f.write(json.dumps(turn) + "\n")

        # Also check disk-only sessions (from previous runs in debug mode)
        if self.debug:
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

        Returns list of {"session_id", "transcript", "speaker_id"} dicts.
        The transcript is formatted as a readable conversation with
        speaker names prefixed for user turns. speaker_id is the most
        frequent speaker_id across turns (None if no speaker identified).
        """
        pending = []
        seen_ids = set()

        # In-memory sessions
        for conv_id, turns in self._turns.items():
            seen_ids.add(conv_id)
            formatted, session_speaker_id = self._format_turns(turns)
            if formatted:
                pending.append(
                    {
                        "session_id": conv_id,
                        "transcript": "\n".join(formatted),
                        "speaker_id": session_speaker_id,
                    }
                )

        # Disk-only sessions (from previous runs in debug mode)
        if self.debug:
            for path in sorted(self.session_dir.glob("*.jsonl")):
                conv_id = path.stem
                if conv_id in seen_ids:
                    continue
                turns = self._read_jsonl(path)
                formatted, session_speaker_id = self._format_turns(turns)
                if formatted:
                    pending.append(
                        {
                            "session_id": conv_id,
                            "transcript": "\n".join(formatted),
                            "speaker_id": session_speaker_id,
                        }
                    )

        return pending

    def mark_consolidated(self, session_ids: list[str]) -> None:
        """Remove or archive consolidated sessions."""
        for session_id in session_ids:
            # Clear from RAM
            self._turns.pop(session_id, None)
            self._sessions.pop(session_id, None)

            # Handle disk file if persisting
            if self.debug:
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
    def _format_turns(turns: list[dict]) -> tuple[list[str], str | None]:
        """Format turns into readable lines and extract dominant speaker_id."""
        formatted = []
        speaker_ids = []
        for turn in turns:
            role = turn.get("role", "unknown")
            text = turn.get("text", "")
            speaker = turn.get("speaker")
            sid = turn.get("speaker_id")

            if sid:
                speaker_ids.append(sid)

            if role == "user" and speaker:
                formatted.append(f"{speaker}: {text}")
            else:
                role_label = role.capitalize()
                formatted.append(f"{role_label}: {text}")

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
        """Write encrypted snapshot of in-memory state to disk.

        Called on graceful shutdown (SIGUSR1, SIGTERM). Returns True on success.
        """
        if not self._fernet:
            return False
        if not self._turns and not self._sessions:
            logger.info("No session data to snapshot")
            return True

        payload = {
            "turns": {k: v for k, v in self._turns.items()},
            "sessions": self._sessions,
        }
        try:
            plaintext = json.dumps(payload).encode()
            ciphertext = self._fernet.encrypt(plaintext)
            tmp = self._snapshot_path.with_suffix(".tmp")
            tmp.write_bytes(ciphertext)
            tmp.rename(self._snapshot_path)
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
        """Restore in-memory state from encrypted snapshot.

        Called on startup. Decrypts and loads, then deletes the file.
        Returns True if a snapshot was restored.
        """
        if not self._fernet or not self._snapshot_path.exists():
            return False

        try:
            ciphertext = self._snapshot_path.read_bytes()
            plaintext = self._fernet.decrypt(ciphertext)
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
