"""Persistent, atomic post-session training queue.

Stores pending post-session training jobs in a JSON file so they survive
server restarts.  The queue is consumed at startup: any entries present when
the server starts are replayed through the same code path that handles
"session finished → enqueue interim training".

Atomicity guarantee
-------------------
Every write goes through a temp-file + ``os.replace`` sequence.  ``os.replace``
is atomic on POSIX and on Windows (Python 3.3+, same filesystem).  A crash
during the temp-file write leaves the original file untouched; a crash after
the rename leaves the new file in place.  No partial-write corruption is
possible regardless of where the process is killed.

Thread safety
-------------
All mutations (enqueue, drain, remove) acquire a ``threading.Lock`` before
reading state and before writing to disk.  Concurrent enqueuers from multiple
``/chat`` turns therefore serialize safely without deadlock.

Schema
------
The file contains a JSON array of entry objects.  Each entry has at minimum:

``session_id``
    The conversation identifier (used for deduplication and removal).

``transcript``
    The full session transcript (joined turns) needed to extract facts.

``speaker_id``
    Speaker identifier for key ownership scoping.

``speaker_name``
    Human-readable speaker name (may be ``null``).

``enqueued_at``
    ISO-8601 UTC timestamp when the entry was enqueued.

Callers may include additional metadata fields; they are stored and returned
verbatim.

Usage example
-------------
>>> queue = PostSessionQueue(Path("data/ha/adapters/post_session_queue.json"))
>>> queue.enqueue({
...     "session_id": "conv-001",
...     "transcript": "user: Hello\\nassistant: Hi",
...     "speaker_id": "spk-abc",
...     "speaker_name": "Alice",
... })
>>> entries = queue.drain()   # returns and clears
>>> for entry in entries:
...     process(entry)
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from paramem.backup.encryption import read_maybe_encrypted, write_infra_bytes

logger = logging.getLogger(__name__)


class PostSessionQueue:
    """Atomic persistent queue for post-session training jobs.

    Args:
        path: Absolute path to the JSON backing file.  The parent directory
              must already exist when the first write is attempted (i.e. the
              adapter directory must be created before constructing the queue).
    """

    def __init__(self, path: Path) -> None:
        """Load existing queue from *path*, or start empty if the file is absent.

        Args:
            path: Path to the JSON backing file.
        """
        self._path = Path(path)
        self._lock = threading.Lock()
        self._entries: list[dict] = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, entry: dict) -> None:
        """Append *entry* to the queue and atomically persist to disk.

        A copy of *entry* is stored; the caller's dict is not mutated.  An
        ``enqueued_at`` field (ISO-8601 UTC) is added to the copy when absent
        so entries can be audited after a recovery replay.

        Args:
            entry: Mapping that must include ``session_id``.  Any additional
                   keys are preserved verbatim.

        Raises:
            ValueError: When *entry* does not contain ``session_id``.
        """
        if "session_id" not in entry:
            raise ValueError("PostSessionQueue.enqueue: entry must include 'session_id'")
        item = dict(entry)
        if "enqueued_at" not in item:
            item["enqueued_at"] = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._entries.append(item)
            self._save_locked()

    def drain(self) -> list[dict]:
        """Snapshot and atomically clear the queue.

        Returns:
            A list of all entries that were present before the clear.  The
            returned list is a copy; subsequent mutations to it do not affect
            the queue.
        """
        with self._lock:
            snapshot = list(self._entries)
            self._entries = []
            self._save_locked()
        return snapshot

    def remove(self, session_id: str) -> None:
        """Remove the entry with *session_id* and atomically persist.

        A no-op when *session_id* is not present.

        Args:
            session_id: The session identifier to remove.
        """
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if e.get("session_id") != session_id]
            if len(self._entries) != before:
                self._save_locked()

    def peek(self) -> list[dict]:
        """Return a snapshot of current entries without modifying the queue.

        Returns:
            A copy of the current entry list.
        """
        with self._lock:
            return list(self._entries)

    def __len__(self) -> int:
        """Return the number of entries currently in the queue."""
        with self._lock:
            return len(self._entries)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> list[dict]:
        """Read and parse the queue file; return empty list on any error.

        Logs a warning (not error) when the file exists but is unreadable /
        corrupt, because a corrupt queue file is a recoverable condition
        (entries are lost but the server can continue).

        Returns:
            Parsed list of entry dicts, or ``[]`` when absent / unreadable.
        """
        if not self._path.exists():
            return []
        try:
            data = json.loads(read_maybe_encrypted(self._path).decode("utf-8"))
            if not isinstance(data, list):
                logger.warning(
                    "post_session_queue: expected JSON array in %s, got %s — starting empty",
                    self._path,
                    type(data).__name__,
                )
                return []
            logger.info(
                "post_session_queue: loaded %d pending entry(s) from %s",
                len(data),
                self._path,
            )
            return data
        except Exception:
            logger.warning(
                "post_session_queue: could not read %s — starting empty",
                self._path,
                exc_info=True,
            )
            return []

    def _save_locked(self) -> None:
        """Write current entries to disk atomically (caller must hold ``_lock``).

        Routes through ``write_infra_bytes`` which handles the temp-file +
        rename sequence and envelope-encrypts when a master key is set.
        """
        try:
            payload = json.dumps(self._entries, ensure_ascii=False, indent=2).encode("utf-8")
            write_infra_bytes(self._path, payload)
        except Exception:
            logger.exception("post_session_queue: failed to write %s", self._path)
