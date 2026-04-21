"""Directory-level atomic rename and orphan-pending sweep.

The "no artifact without sidecar" invariant is enforced by writing both files
into a ``.pending/<timestamp>/`` directory, fsyncing the directory, and then
promoting it to the live slot via a single ``os.rename()``.  Because ``rename``
is atomic on POSIX filesystems (including WSL2 ext4), readers either see both
files or neither.

**WSL2 filesystem requirement** — ``base_dir`` (and therefore ``.pending/``)
must live on the Linux-native ext4 filesystem, *not* on a 9P-mounted Windows
path (``/mnt/c/…``).  ``os.rename()`` on 9P-mounted paths has known
atomicity edge cases.  This matches the project-wide WSL2 rule in ``CLAUDE.md``.

Crash safety
------------
If the process dies between fsync and rename, a ``.pending/<timestamp>/``
directory is left on disk.  ``sweep_orphan_pending`` removes all such residue
on startup — it is the only recovery action required.

Slot collision
--------------
If ``slot_dir`` already exists when ``rename_pending_to_slot`` is called, the
function raises ``FileExistsError`` immediately and leaves ``pending_dir``
intact.  This is a non-fatal condition:

- In normal operation it indicates a clock-skew or duplicate-timestamp event
  (two rapid backups within the same hundredths-of-a-second window).
- Retry is safe once the conflicting slot is removed.
- Incomplete writes from *this* process are impossible by construction — the
  slot is created atomically.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_PENDING_DIR_NAME = ".pending"


def rename_pending_to_slot(pending_dir: Path, slot_dir: Path) -> None:
    """Promote a fully-written ``.pending/<ts>/`` into the live *slot_dir*.

    Contract
    --------
    - *pending_dir* must exist, be on the same filesystem as
      ``slot_dir.parent``, and be fsync'd by the caller before this
      function runs.
    - *slot_dir* must NOT exist.  If it does, raises ``FileExistsError``
      immediately — operator-visible as
      "slot <path> already exists; refusing to overwrite".
      Retry is safe once the conflicting slot is removed; a slot collision
      indicates clock skew or a duplicate timestamp, never a partial-write
      from this process.
    - On success, *pending_dir* no longer exists and *slot_dir* contains the
      exact artifact + sidecar pair that was fsync'd in pending.
    - On any ``OSError`` other than ``FileExistsError``, the caller is
      responsible for leaving *pending_dir* in place so the startup sweep can
      remove it later.

    Parameters
    ----------
    pending_dir:
        Source directory (the ``.pending/<ts>/`` path).
    slot_dir:
        Destination directory (the live ``<kind>/<ts>/`` path).

    Raises
    ------
    FileExistsError
        If *slot_dir* already exists.
    OSError
        On any other rename failure.

    (NIT 1)
    """
    if slot_dir.exists():
        raise FileExistsError(
            f"slot {slot_dir} already exists; refusing to overwrite — "
            "retry is safe once the conflicting slot is removed"
        )
    os.rename(pending_dir, slot_dir)


def sweep_orphan_pending(kind_root: Path) -> list[Path]:
    """Remove every ``.pending/*`` residue under *kind_root* on startup.

    Looks for a ``.pending/`` subdirectory inside *kind_root* and removes
    every entry inside it (the timestamped sub-directories left by a crashed
    write).  Non-fatal — logs a warning for each removed path and continues.

    *kind_root* is the per-kind backup directory, e.g.
    ``data/ha/backups/config/``.  The ``.pending/`` directory lives at
    ``<kind_root>/.pending/``.

    Parameters
    ----------
    kind_root:
        The per-kind backup directory to sweep.

    Returns
    -------
    list[Path]
        Paths that were removed (for caller logging).  Empty when there was
        nothing to sweep.
    """
    pending_root = kind_root / _PENDING_DIR_NAME
    removed: list[Path] = []

    if not pending_root.exists():
        return removed

    for entry in sorted(pending_root.iterdir()):
        try:
            _remove_tree(entry)
            removed.append(entry)
            logger.warning("sweep_orphan_pending: removed incomplete pending slot %s", entry)
        except OSError as exc:
            logger.warning("sweep_orphan_pending: could not remove %s: %s", entry, exc)

    return removed


def _remove_tree(path: Path) -> None:
    """Recursively remove *path* (file or directory tree).

    Prefers ``shutil.rmtree`` for directories and ``path.unlink`` for files.
    """
    import shutil

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)
