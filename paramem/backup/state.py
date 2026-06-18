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

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from paramem.server.atomic_json import atomic_write_json, flock_rmw, read_json_or_none

if TYPE_CHECKING:
    from paramem.backup.runner import ScheduledBackupResult

logger = logging.getLogger(__name__)

BACKUP_STATE_SCHEMA_VERSION: int = 1
BACKUP_STATE_FILENAME: str = "backup.json"
_BACKUP_LOCKFILE: str = "backup.lock"


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


# Plaintext-by-design: control-plane state only (paths, timestamps, counts —
# no secrets). Carve-out parallels state/trial.json — encrypting this file
# would risk bricking backup recovery on key loss without protecting anything
# sensitive.
def write_backup_state(state_dir: Path, record: BackupStateRecord) -> Path:
    """Atomically write ``state_dir/backup.json`` via a ``.pending`` rename.

    Delegates to :func:`paramem.server.atomic_json.atomic_write_json` for the
    atomic-write mechanics.

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
    return atomic_write_json(Path(state_dir), BACKUP_STATE_FILENAME, record.to_dict())


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
    try:
        raw = read_json_or_none(Path(state_dir), BACKUP_STATE_FILENAME)
    except json.JSONDecodeError as exc:
        raise BackupStateSchemaError(f"backup state is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise BackupStateSchemaError(f"cannot read backup state: {exc}") from exc

    if raw is None:
        return None
    return BackupStateRecord.from_dict(raw)


def update_backup_state(
    state_dir: Path,
    new_run: "ScheduledBackupResult",
) -> BackupStateRecord:
    """Read-modify-write the state file with the result of a new backup run.

    Acquires an exclusive ``fcntl.flock`` on ``state_dir/backup.lock`` before
    the read-modify-write to serialise concurrent callers (Fix 3).

    Delegates locking and atomic-write mechanics to
    :func:`paramem.server.atomic_json.flock_rmw`.

    Promotion rules:

    - ``last_success_at``: set to ``new_run.completed_at`` when
      ``new_run.success == True``; otherwise left unchanged (monotonic).
    - ``last_failure_at`` / ``last_failure_reason``: set to the new values
      when ``new_run.success == False``; left unchanged on success so that
      the last-failure timestamp remains for operator visibility (updated on
      failure only, never cleared on success).

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

    def _mutate(current: dict | None) -> dict:
        if current is None:
            last_success_at = None
            last_failure_at = None
            last_failure_reason = None
        else:
            # Re-parse through from_dict to get typed access; errors propagate.
            existing = BackupStateRecord.from_dict(current)
            last_success_at = existing.last_success_at
            last_failure_at = existing.last_failure_at
            last_failure_reason = existing.last_failure_reason

        if new_run.success:
            last_success_at = new_run.completed_at
        else:
            last_failure_at = new_run.completed_at
            last_failure_reason = new_run.error

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
        return new_record.to_dict()

    result_dict = flock_rmw(state_dir, _BACKUP_LOCKFILE, _mutate, BACKUP_STATE_FILENAME)
    return BackupStateRecord.from_dict(result_dict)
