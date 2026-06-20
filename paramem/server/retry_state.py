"""Durable per-session consolidation retry counter for the ParaMem server.

Persists the count of consecutive failed consolidation attempts per session
to ``data/state/consolidation_retry.json`` (plaintext control-plane — survives
an ungraceful restart / host TDR so a deterministically-failing fact cannot pin
its session forever across crash loops).  Not registered in ``infra_paths()``.

Schema
------
``{"version": 1, "counts": {session_id: int}}``

Each entry is a non-negative integer counting how many consecutive consolidation
cycles the session contributed a recall-gate failure or a DEGENERATE outcome.
ABORT cycles do NOT increment the counter (abort = yield-to-inference, not a
failed encoding attempt).

Lifecycle
---------
- **Increment**: ``bump_retry_count(state_dir, session_id) -> int``
  Atomically increments the count for one session and returns the new value.
  Raises ``RetryStateCapacityError`` on ENOSPC/EDQUOT; propagates all other
  ``OSError`` unchanged.

- **Reset**: ``reset_retry_count(state_dir, session_id)``
  Clears the count for one session (reset-on-recall-success: when a previously-
  counted session passes recall, its counter is cleared so only consecutive
  failures accrue toward the cap).

- **Clear (bulk)**: ``clear_retry_counts(state_dir, session_ids)``
  Removes the durable row for each retired session.  Called by
  ``SessionBuffer.mark_consolidated`` and ``SessionBuffer.discard_sessions``
  so stale rows do not accumulate and do not mis-count a future session that
  reuses an id.

- **Read**: ``read_retry_counts(state_dir) -> dict[str, int]``
  Returns the full ``session_id → count`` mapping.  Used during boot hydration.

Concurrency
-----------
All write functions use ``flock_rmw`` (``fcntl.flock`` on
``consolidation_retry.lock``) to serialise concurrent callers.
"""

from __future__ import annotations

import errno
import logging
from pathlib import Path

from paramem.server.atomic_json import flock_rmw, read_json_or_none

logger = logging.getLogger(__name__)

RETRY_STATE_FILENAME: str = "consolidation_retry.json"
RETRY_STATE_SCHEMA_VERSION: int = 1
RETRY_STATE_LOCKFILE: str = "consolidation_retry.lock"

# errno values that indicate disk-capacity exhaustion (not a code bug).
_CAPACITY_ERRNOS: frozenset[int] = frozenset({errno.ENOSPC, errno.EDQUOT})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RetryStateError(Exception):
    """Base class for retry-state I/O errors."""


class RetryStateSchemaError(RetryStateError):
    """Retry-state file fails schema validation (version mismatch, bad JSON)."""


class RetryStateCapacityError(RetryStateError):
    """Disk full or quota exceeded while writing the retry-state file.

    Raised by ``bump_retry_count`` when ``atomic_write_json`` raises an
    ``OSError`` with ``errno`` in ``{ENOSPC, EDQUOT}``.  The caller must
    record a ``storage_capacity_reached`` incident and stop — no retry spin.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_store(raw: dict | None) -> dict[str, int]:
    """Parse the raw store dict into ``session_id → count`` mapping.

    Returns an empty dict when the file is absent (``raw is None``).

    Raises ``RetryStateSchemaError`` on version mismatch or structural
    problems.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RetryStateSchemaError("consolidation_retry store root is not a JSON object")
    version = raw.get("version")
    if version is None:
        raise RetryStateSchemaError("consolidation_retry store missing required field: version")
    if version != RETRY_STATE_SCHEMA_VERSION:
        raise RetryStateSchemaError(
            f"consolidation_retry store version={version} does not match "
            f"current RETRY_STATE_SCHEMA_VERSION={RETRY_STATE_SCHEMA_VERSION}"
        )
    counts = raw.get("counts", {})
    if not isinstance(counts, dict):
        raise RetryStateSchemaError("consolidation_retry store 'counts' field is not a dict")
    return dict(counts)


def _build_store(counts: dict[str, int]) -> dict:
    """Build the on-disk store dict from a ``session_id → count`` mapping."""
    return {"version": RETRY_STATE_SCHEMA_VERSION, "counts": counts}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def bump_retry_count(state_dir: Path, session_id: str) -> int:
    """Atomically increment the retry counter for ``session_id``.

    Reads the current count (or 0 when absent), increments by 1, writes back
    atomically, and returns the new count.

    Parameters
    ----------
    state_dir:
        Directory containing ``consolidation_retry.json`` (e.g. ``data/state/``).
        Created if absent.
    session_id:
        Session identifier (control-plane key; verified non-personal).

    Returns
    -------
    int
        The new retry count after incrementing.

    Raises
    ------
    RetryStateCapacityError
        When the disk write fails with ``ENOSPC`` or ``EDQUOT``.  The caller
        must record a ``storage_capacity_reached`` incident and stop — no
        retry spin.
    RetryStateSchemaError
        When the file exists but is structurally invalid.
    OSError
        Any non-capacity ``OSError`` propagates unchanged — it is a real
        infrastructure error, not a capacity event.
    """
    new_count_holder: list[int] = [0]

    def _mutate(current: dict | None) -> dict:
        counts = _parse_store(current)
        new_count = counts.get(session_id, 0) + 1
        new_count_holder[0] = new_count
        counts[session_id] = new_count
        return _build_store(counts)

    try:
        flock_rmw(Path(state_dir), RETRY_STATE_LOCKFILE, _mutate, RETRY_STATE_FILENAME)
    except OSError as exc:
        if exc.errno in _CAPACITY_ERRNOS:
            raise RetryStateCapacityError(
                f"Disk full (errno={exc.errno}) — consolidation retry state cannot "
                f"persist for session {session_id!r}: {exc}"
            ) from exc
        raise

    return new_count_holder[0]


def reset_retry_count(state_dir: Path, session_id: str) -> None:
    """Clear the durable retry count for ``session_id`` (reset-on-recall-success).

    Called when a previously-counted session passes recall in a cycle.  Clearing
    on success ensures that only consecutive failures accrue toward the cap —
    a session that eventually encodes its facts cleanly re-enters the retry budget
    at 0, not at its historical failure count.

    Idempotent: a no-op when the session has no durable entry.

    Parameters
    ----------
    state_dir:
        Directory containing ``consolidation_retry.json``.
    session_id:
        Session identifier to reset.
    """

    def _mutate(current: dict | None) -> dict:
        counts = _parse_store(current)
        counts.pop(session_id, None)
        return _build_store(counts)

    flock_rmw(Path(state_dir), RETRY_STATE_LOCKFILE, _mutate, RETRY_STATE_FILENAME)


def clear_retry_counts(state_dir: Path, session_ids: list[str]) -> None:
    """Remove durable retry-count rows for all ``session_ids`` in one atomic write.

    Called by ``SessionBuffer.mark_consolidated`` and
    ``SessionBuffer.discard_sessions`` when sessions are retired so stale rows
    do not accumulate and do not mis-count a future session that reuses an id.

    Idempotent: session ids with no durable entry are silently skipped.

    Parameters
    ----------
    state_dir:
        Directory containing ``consolidation_retry.json``.
    session_ids:
        Session identifiers to clear.  May be empty (no-op).
    """
    if not session_ids:
        return

    to_clear = set(session_ids)

    def _mutate(current: dict | None) -> dict:
        counts = _parse_store(current)
        for sid in to_clear:
            counts.pop(sid, None)
        return _build_store(counts)

    flock_rmw(Path(state_dir), RETRY_STATE_LOCKFILE, _mutate, RETRY_STATE_FILENAME)


def read_retry_counts(state_dir: Path) -> dict[str, int]:
    """Read the full ``session_id → count`` mapping from disk.

    Returns an empty dict when the file is absent.  Used during boot hydration
    (``SessionBuffer.hydrate_retry_counts``) to seed in-memory caches from the
    durable store.

    Parameters
    ----------
    state_dir:
        Directory containing ``consolidation_retry.json``.

    Returns
    -------
    dict[str, int]
        Mapping of session id to retry count for all entries present on disk.

    Raises
    ------
    RetryStateSchemaError
        When the file exists but is malformed JSON or has a version mismatch.
    """
    import json

    try:
        raw = read_json_or_none(Path(state_dir), RETRY_STATE_FILENAME)
    except json.JSONDecodeError as exc:
        raise RetryStateSchemaError(f"consolidation_retry store is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise RetryStateSchemaError(f"cannot read consolidation_retry store: {exc}") from exc

    return _parse_store(raw)
