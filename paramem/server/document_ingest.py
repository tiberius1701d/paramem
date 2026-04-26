"""Server-side helpers for the document ingestion pipeline.

Provides:
- ``IngestRegistry`` — JSON-backed idempotency index for ingest chunks.
  Persists via :func:`paramem.backup.encryption.write_infra_bytes` /
  :func:`paramem.backup.encryption.read_maybe_encrypted` so the file
  is age-encrypted at rest whenever the daily identity is loaded,
  and plaintext otherwise.  No conditional branch in caller code.
- ``normalize_chunk_text`` — NFC-normalize and collapse whitespace runs.

The ``extract_only`` preview helper and chunkers live in Phase 3 / Phase 4
respectively.
"""

from __future__ import annotations

import hashlib
import json
import logging
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paramem.backup.encryption import read_maybe_encrypted, write_infra_bytes

logger = logging.getLogger(__name__)

EXPECTED_REGISTRY_VERSION = 1
"""Registry schema version this code understands.

Used on read (to reject future-version files) and on write (single source
of truth so the constant never drifts between the two paths).
"""


def normalize_chunk_text(text: str) -> str:
    """NFC-normalize *text* then collapse runs of whitespace.

    Applies Unicode NFC normalization followed by whitespace collapsing
    (any whitespace run including newlines and tabs becomes a single
    ASCII space; leading and trailing whitespace is stripped).  No case
    folding — two chunks that differ only in case produce distinct hashes,
    which is the intended behaviour: the operator may have edited a document
    and changed only capitalisation, and the system should ingest the updated
    version.

    Args:
        text: Raw chunk text.

    Returns:
        Normalized text string.
    """
    normalized = unicodedata.normalize("NFC", text)
    return " ".join(normalized.split())


class IngestRegistry:
    """JSON-backed idempotency index for document ingest chunks.

    Stores a ``sha256`` fingerprint per chunk so re-running the ingest CLI
    on the same files is a no-op.  The registry does NOT store chunk text —
    only the hash, metadata, and the session_id that was enqueued.

    Persistence uses :func:`paramem.backup.encryption.write_infra_bytes`
    unconditionally.  That helper encrypts when the daily age identity is
    loaded and writes plaintext otherwise.  There is no
    ``if require_encryption`` branch in this class — the caller simply calls
    :meth:`flush` after a batch.

    Schema on disk::

        {
          "version": 1,
          "entries": {
            "<sha256_hex_64>": {
              "speaker_id": "...",
              "source_path": "...",
              "chunk_index": <int>,
              "source_type": "document",
              "doc_title": "...",
              "ingested_at": "<iso8601>",
              "session_id": "doc-<hex8>"
            }
          }
        }

    Args:
        path: Filesystem path for the registry JSON file.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._entries: dict[str, dict] = {}
        self._dirty = False

        if self._path.exists():
            try:
                raw = read_maybe_encrypted(self._path)
                data = json.loads(raw.decode("utf-8"))
                version = data.get("version", 1)
                if version != EXPECTED_REGISTRY_VERSION:
                    logger.warning(
                        "IngestRegistry version mismatch at %s: expected %d, got %r"
                        " — starting empty",
                        self._path,
                        EXPECTED_REGISTRY_VERSION,
                        version,
                    )
                    self._entries = {}
                    self._dirty = False
                    return
                self._entries = data.get("entries", {})
                logger.debug(
                    "IngestRegistry loaded %d entries from %s",
                    len(self._entries),
                    self._path,
                )
            except Exception:
                logger.exception("IngestRegistry failed to load %s — starting empty", self._path)
                self._entries = {}

    def chunk_hash(
        self,
        *,
        speaker_id: str,
        source_path: str,
        chunk_index: int,
        normalized_text: str,
        source_type: str,
    ) -> str:
        """Compute a stable SHA-256 fingerprint for a chunk.

        Hash key composition:
        ``sha256(speaker_id || "\\0" || source_path || "\\0" ||
        str(chunk_index) || "\\0" || normalized_text || "\\0" ||
        source_type)``

        The null byte ``"\\0"`` is used as a field separator to prevent
        prefix-extension collisions (e.g. ``speaker="a\\0b"`` vs
        ``speaker="a", source_path="b"``).

        Args:
            speaker_id: Speaker identifier (determines ownership).
            source_path: Original file path (display/tracing only).
            chunk_index: Zero-based index of this chunk within the source.
            normalized_text: Whitespace-collapsed NFC text from
                :func:`normalize_chunk_text`.
            source_type: Source category (``"document"``).

        Returns:
            Lowercase hex SHA-256 digest (64 characters).
        """
        payload = "\0".join(
            [
                speaker_id,
                source_path,
                str(chunk_index),
                normalized_text,
                source_type,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def is_known(self, chunk_hash: str) -> bool:
        """Return ``True`` when *chunk_hash* is already recorded.

        Args:
            chunk_hash: SHA-256 hex digest from :meth:`chunk_hash`.

        Returns:
            ``True`` if this hash has been recorded; ``False`` otherwise.
        """
        return chunk_hash in self._entries

    def record(self, chunk_hash: str, *, session_id: str, **meta: Any) -> None:
        """Record a chunk hash with associated metadata in memory.

        Does not write to disk.  Call :meth:`flush` after processing a
        batch to persist the registry.

        Args:
            chunk_hash: SHA-256 hex digest from :meth:`chunk_hash`.
            session_id: Session id that was enqueued for this chunk.
            **meta: Additional fields stored verbatim.  Expected keys:
                ``speaker_id``, ``source_path``, ``chunk_index``,
                ``source_type``, ``doc_title``, ``ingested_at``.
        """
        self._entries[chunk_hash] = {"session_id": session_id, **meta}
        self._dirty = True

    def flush(self) -> None:
        """Persist the in-memory registry to disk via ``write_infra_bytes``.

        No-op when no entries have been added since the last flush (i.e.
        :attr:`_dirty` is ``False``).  This avoids redundant disk writes when
        every chunk in a request was already known to the registry.

        Writes unconditionally when dirty — no ``if require_encryption``
        check.  :func:`paramem.backup.encryption.write_infra_bytes` handles
        both the encrypted and plaintext branches internally.

        Creates the parent directory if it does not exist.
        """
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": EXPECTED_REGISTRY_VERSION, "entries": self._entries}
        write_infra_bytes(self._path, json.dumps(data).encode("utf-8"))
        self._dirty = False
        logger.debug("IngestRegistry flushed %d entries to %s", len(self._entries), self._path)


def _now_iso8601() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
