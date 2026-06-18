"""Shared atomic-JSON I/O primitive for control-plane state files.

Provides three generic helpers that every domain-specific state module
(``backup/state.py``, ``incidents.py``, ``run_status.py``) builds on:

- :func:`atomic_write_json` — ``.pending`` + double-fsync + ``os.rename``.
- :func:`read_json_or_none` — absence → ``None``; raw JSON parse; I/O errors
  propagate as ``json.JSONDecodeError`` / ``OSError`` for the domain caller to
  wrap in its own typed schema error.
- :func:`flock_rmw` — ``fcntl.flock``-guarded read-modify-write.

Per-domain schema-version validation is intentionally NOT in this module.
Each domain's ``from_dict`` owns schema-version checking and raises its own
typed schema error (e.g. ``BackupStateSchemaError``, ``IncidentStoreSchemaError``).

Concurrency
-----------
``flock_rmw`` acquires an exclusive ``fcntl.flock`` on a named lock file
before reading, mutating, and writing.  The lock is released when the lock-file
``with`` block exits.  This serialises concurrent callers (background-trainer
thread, event-loop thread, migration coroutine).
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_PENDING_DIRNAME: str = ".pending"


def atomic_write_json(state_dir: Path, filename: str, payload: dict) -> Path:
    """Atomically write ``state_dir/<filename>`` via a ``.pending`` rename.

    Write sequence (mirrors ``paramem.backup.state.write_backup_state``):

    1. Create ``state_dir/.pending/`` if absent.
    2. Write ``<filename>`` into the pending directory.
    3. ``fsync`` the file fd.
    4. ``fsync`` the pending directory entry.
    5. ``os.rename(.pending/<filename>, state_dir/<filename>)`` — atomic.
    6. ``fsync(state_dir)`` for rename durability.

    Parameters
    ----------
    state_dir:
        Directory that will contain ``<filename>``.  Created with parents
        if absent.
    filename:
        Bare filename (e.g. ``"backup.json"``).
    payload:
        JSON-serialisable dict to write.

    Returns
    -------
    Path
        The final path of the written file (``state_dir/<filename>``).
    """
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    pending_dir = state_dir / _PENDING_DIRNAME
    pending_dir.mkdir(parents=True, exist_ok=True)

    pending_file = pending_dir / filename
    final_path = state_dir / filename

    data = json.dumps(payload, indent=2, ensure_ascii=False)

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
            logger.warning("atomic_write_json: pending dir fsync failed: %s", exc)
        finally:
            os.close(pd_fd)
    except OSError as exc:
        logger.warning("atomic_write_json: could not open pending dir for fsync: %s", exc)

    os.rename(pending_file, final_path)

    # fsync state_dir for rename durability.
    try:
        sd_fd = os.open(str(state_dir), os.O_RDONLY)
        try:
            os.fsync(sd_fd)
        except OSError as exc:
            logger.warning("atomic_write_json: state_dir fsync failed: %s", exc)
        finally:
            os.close(sd_fd)
    except OSError as exc:
        logger.warning("atomic_write_json: could not open state_dir for fsync: %s", exc)

    return final_path


def read_json_or_none(state_dir: Path, filename: str) -> dict | None:
    """Read ``state_dir/<filename>`` and return the parsed dict, or ``None`` if absent.

    Does NOT validate schema version — that is the domain caller's responsibility.

    Parameters
    ----------
    state_dir:
        Directory containing ``<filename>``.
    filename:
        Bare filename (e.g. ``"backup.json"``).

    Returns
    -------
    dict or None
        Parsed JSON object, or ``None`` when the file does not exist.

    Raises
    ------
    json.JSONDecodeError
        When the file exists but is not valid JSON.
    OSError
        When the file exists but cannot be read.
    """
    path = Path(state_dir) / filename
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw


def flock_rmw(
    state_dir: Path,
    lockfile: str,
    mutate: Callable[[dict | None], dict],
    filename: str,
) -> dict:
    """``fcntl.flock``-guarded read-modify-write for a JSON state file.

    Acquires an exclusive lock on ``state_dir/<lockfile>``, reads the current
    JSON dict (or ``None`` when absent), calls ``mutate(current)`` to produce
    the new dict, atomically writes it, and returns the new dict.

    Parameters
    ----------
    state_dir:
        Directory containing the state file and lock file.  Created if absent.
    lockfile:
        Bare filename for the lock file (e.g. ``"incidents.lock"``).
    mutate:
        Callable that receives the current dict (or ``None`` when the file
        does not yet exist) and returns the updated dict to write.
    filename:
        Bare filename of the JSON state file to read-modify-write (e.g.
        ``"incidents.json"``).

    Returns
    -------
    dict
        The new dict as written to disk (the return value of ``mutate``).
    """
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / lockfile

    with open(lock_path, "w") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        current = read_json_or_none(state_dir, filename)
        new_data = mutate(current)
        atomic_write_json(state_dir, filename, new_data)
        # Lock released on file close (exiting the with block).

    return new_data
