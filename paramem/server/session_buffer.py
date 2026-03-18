"""Session transcript buffering — accumulates conversation turns on disk."""

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class SessionBuffer:
    """Buffers conversation transcripts to disk for later consolidation.

    Each conversation gets a JSONL file. After consolidation, files are
    moved to an archive subdirectory.
    """

    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir = self.session_dir / "archive"
        self.archive_dir.mkdir(exist_ok=True)

    def append(self, conversation_id: str, role: str, text: str) -> None:
        """Append a turn to the conversation transcript."""
        path = self.session_dir / f"{conversation_id}.jsonl"
        entry = {
            "role": role,
            "text": text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_pending(self) -> list[dict]:
        """Return all pending (non-archived) session transcripts.

        Returns list of {"session_id": str, "transcript": str} dicts.
        The transcript is formatted as a readable conversation.
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
                    role = turn.get("role", "unknown").capitalize()
                    turns.append(f"{role}: {turn.get('text', '')}")

            if turns:
                pending.append(
                    {
                        "session_id": path.stem,
                        "transcript": "\n".join(turns),
                    }
                )
        return pending

    def mark_consolidated(self, session_ids: list[str]) -> None:
        """Move consolidated session files to the archive directory."""
        for session_id in session_ids:
            source = self.session_dir / f"{session_id}.jsonl"
            if source.exists():
                dest = self.archive_dir / f"{session_id}.jsonl"
                shutil.move(str(source), str(dest))
                logger.info("Archived session: %s", session_id)

    @property
    def pending_count(self) -> int:
        return len(list(self.session_dir.glob("*.jsonl")))
