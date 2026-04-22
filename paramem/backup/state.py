"""Backup state file I/O for the ParaMem backup subsystem.

Implements ``state/backup.json`` read/write, following the same atomic
``.pending/<file>`` + ``os.rename`` pattern as ``trial_state.py``.

Schema version contract
-----------------------
``BACKUP_STATE_SCHEMA_VERSION`` is bumped on every breaking change.  Adding
optional fields with defaults is non-breaking (no bump).  Renaming or removing
fields requires a bump.

``read_backup_state`` raises ``BackupStateSchemaError`` on parse failure or
schema-version mismatch.  Returns ``None`` when the file is absent (server
never ran a scheduled backup).

Concurrency
-----------
``update_backup_state`` acquires an exclusive ``fcntl.flock`` on a lock file
before the read-modify-write to serialise concurrent callers (manual CLI vs
timer runner).  The lock is released when the lock-file ``with`` block exits.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paramem.backup.runner import ScheduledBackupResult

logger = logging.getLogger(__name__)

BACKUP_STATE_SCHEMA_VERSION: int = 1
BACKUP_STATE_FILENAME: str = "backup.json"

_PENDING_DIRNAME: str = ".pending"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BackupStateError(Exception):
    """Base class for backup-state I/O errors."""


class BackupStateSchemaError(BackupStateError):
    """State file fails schema validation (version mismatch, bad JSON, missing field)."""


# ---------------------------------------------------------------------------
# BackupStateRecord
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackupStateRecord:
    """In-memory mirror of ``state/backup.json``.

    Attributes
    ----------
    schema_version:
        Always ``BACKUP_STATE_SCHEMA_VERSION`` at write time.
    last_run:
        Serialised ``ScheduledBackupResult`` dict from the most recent run
        (success or failure).  ``None`` when no run has completed yet.
    last_success_at:
        ISO-8601 UTC timestamp of the most recent successful run.  Monotonic:
        set only when ``last_run.success == True``.  ``None`` when no
        successful run has occurred.
    last_failure_at:
        ISO-8601 UTC timestamp of the most recent failed run.  Not cleared on
        subsequent success — reflects "the last time this failed" for operator
        visibility (spec L487–489).  ``None`` when no failure has occurred.
    last_failure_reason:
        Short error string from the most recent failure.  Mirrors
        ``ScheduledBackupResult.error``.  ``None`` when no failure.
    """

    schema_version: int
    last_run: dict | None
    last_success_at: str | None
    last_failure_at: str | None
    last_failure_reason: str | None

    @classmethod
    def from_dict(cls, d: dict) -> "BackupStateRecord":
        """Deserialise from a JSON dict.

        Raises ``BackupStateSchemaError`` on missing fields or wrong schema
        version.
        """
        if not isinstance(d, dict):
            raise BackupStateSchemaError("state root is not a JSON object")

        version = d.get("schema_version")
        if version is None:
            raise BackupStateSchemaError("state missing required field: schema_version")
        if version != BACKUP_STATE_SCHEMA_VERSION:
            raise BackupStateSchemaError(
                f"state schema_version={version} does not match "
                f"current BACKUP_STATE_SCHEMA_VERSION={BACKUP_STATE_SCHEMA_VERSION}"
            )

        return cls(
            schema_version=version,
            last_run=d.get("last_run"),
            last_success_at=d.get("last_success_at"),
            last_failure_at=d.get("last_failure_at"),
            last_failure_reason=d.get("last_failure_reason"),
        )

    def to_dict(self) -> dict:
        """Serialise to a JSON-ready dict."""
        return {
            "schema_version": self.schema_version,
            "last_run": self.last_run,
            "last_success_at": self.last_success_at,
            "last_failure_at": self.last_failure_at,
            "last_failure_reason": self.last_failure_reason,
        }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def write_backup_state(state_dir: Path, record: BackupStateRecord) -> Path:
    """Atomically write ``state_dir/backup.json`` via a ``.pending`` rename.

    Mirrors ``paramem.server.trial_state.write_trial_marker``:

    1. Create ``state_dir/.pending/`` if absent.
    2. Write ``backup.json`` into the pending directory.
    3. ``fsync`` the file fd.
    4. ``fsync`` the pending directory entry.
    5. ``os.rename(.pending/backup.json, state_dir/backup.json)`` — atomic.
    6. ``fsync(state_dir)`` for rename durability.

    Parameters
    ----------
    state_dir:
        Directory that will contain ``backup.json``
        (e.g. ``data/ha/state/``).  Created with parents if absent.
    record:
        ``BackupStateRecord`` to serialise.

    Returns
    -------
    Path
        The final path of the written file (``state_dir/backup.json``).
    """
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    pending_dir = state_dir / _PENDING_DIRNAME
    pending_dir.mkdir(parents=True, exist_ok=True)

    pending_file = pending_dir / BACKUP_STATE_FILENAME
    final_path = state_dir / BACKUP_STATE_FILENAME

    data = json.dumps(record.to_dict(), indent=2, ensure_ascii=False)

    with open(pending_file, "w", encoding="utf-8") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())

    # fsync pending directory entry.
    try:
        pd_fd = os.open(str(pending_dir), os.O_RDONLY)
        try:
            os.fsync(pd_fd)
        except OSError as exc:
            logger.warning("write_backup_state: pending dir fsync failed: %s", exc)
        finally:
            os.close(pd_fd)
    except OSError as exc:
        logger.warning("write_backup_state: could not open pending dir for fsync: %s", exc)

    os.rename(pending_file, final_path)

    # fsync state_dir for rename durability.
    try:
        sd_fd = os.open(str(state_dir), os.O_RDONLY)
        try:
            os.fsync(sd_fd)
        except OSError as exc:
            logger.warning("write_backup_state: state_dir fsync failed: %s", exc)
        finally:
            os.close(sd_fd)
    except OSError as exc:
        logger.warning("write_backup_state: could not open state_dir for fsync: %s", exc)

    return final_path


def read_backup_state(state_dir: Path) -> BackupStateRecord | None:
    """Return ``None`` when ``state_dir/backup.json`` is absent (LIVE / never run).

    Raises ``BackupStateSchemaError`` on parse / schema failure.

    Parameters
    ----------
    state_dir:
        Directory containing ``backup.json``.

    Returns
    -------
    BackupStateRecord or None
    """
    state_path = Path(state_dir) / BACKUP_STATE_FILENAME
    if not state_path.exists():
        return None

    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BackupStateSchemaError(f"backup state is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise BackupStateSchemaError(f"cannot read backup state: {exc}") from exc

    return BackupStateRecord.from_dict(raw)


def update_backup_state(
    state_dir: Path,
    new_run: "ScheduledBackupResult",
) -> BackupStateRecord:
    """Read-modify-write the state file with the result of a new backup run.

    Acquires an exclusive ``fcntl.flock`` on ``state_dir/backup.lock`` before
    the read-modify-write to serialise concurrent callers (Fix 3).

    Promotion rules:

    - ``last_success_at``: set to ``new_run.completed_at`` when
      ``new_run.success == True``; otherwise left unchanged (monotonic).
    - ``last_failure_at`` / ``last_failure_reason``: set to the new values
      when ``new_run.success == False``; left unchanged on success so that
      the last-failure timestamp remains for operator visibility (spec §9.4).

    Parameters
    ----------
    state_dir:
        Directory containing ``backup.json``.  Created if absent.
    new_run:
        The ``ScheduledBackupResult`` from the most recent invocation.

    Returns
    -------
    BackupStateRecord
        The new record as written to disk.
    """
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "backup.lock"

    with open(lock_path, "w") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)

        # Read existing record (or start fresh).
        existing = read_backup_state(state_dir)
        if existing is None:
            last_success_at = None
            last_failure_at = None
            last_failure_reason = None
        else:
            last_success_at = existing.last_success_at
            last_failure_at = existing.last_failure_at
            last_failure_reason = existing.last_failure_reason

        # Update fields based on the new run.
        if new_run.success:
            last_success_at = new_run.completed_at
        else:
            last_failure_at = new_run.completed_at
            last_failure_reason = new_run.error

        # Build the serialised last_run dict.
        last_run_dict = {
            "started_at": new_run.started_at,
            "completed_at": new_run.completed_at,
            "success": new_run.success,
            "tier": new_run.tier,
            "label": new_run.label,
            "written_slots": new_run.written_slots,
            "skipped_artifacts": new_run.skipped_artifacts,
            "error": new_run.error,
            "prune_result_summary": new_run.prune_result_summary,
        }

        new_record = BackupStateRecord(
            schema_version=BACKUP_STATE_SCHEMA_VERSION,
            last_run=last_run_dict,
            last_success_at=last_success_at,
            last_failure_at=last_failure_at,
            last_failure_reason=last_failure_reason,
        )

        write_backup_state(state_dir, new_record)
        # Lock released on file close (exiting the with block).

    return new_record
