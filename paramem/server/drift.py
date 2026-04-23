"""Config drift detection for the ParaMem server.

Hashes ``configs/server.yaml`` at startup, then re-hashes it on a background
polling loop.  Any change to the file's raw bytes is surfaced as
``detected=True`` in the state slot so the ``/status`` endpoint and the Slice 3
Attention block can alert the operator.

Design notes:

- Content-hash only (WSL2 mtime is unreliable across filesystem events).
- Routing through ``paramem.backup.hashing.content_sha256_path`` so Slice 1's
  hash primitive stays the single source of truth (Resolved Decision 29).
- Env-var references in the yaml (e.g. ``${PARAMEM_MASTER_KEY}``) are hashed
  as the literal template string, not the resolved value.  Key rotation is
  detected separately via ``.meta.json`` fingerprints, not via config drift.
- The poll loop never raises — on ``OSError`` (missing file, permission error,
  etc.) ``disk_hash`` is set to ``""`` and ``detected`` to ``True`` so the
  operator is alerted that the file has gone away.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from paramem.backup.hashing import content_sha256_path

logger = logging.getLogger(__name__)


class ConfigDriftState(TypedDict):
    """Shape of the config-drift block stored on ``_state["config_drift"]``."""

    detected: bool
    loaded_hash: str  # full 64-hex sha256, truncated at render time if needed
    disk_hash: str  # "" if the file disappeared between polls
    last_checked_at: str  # ISO 8601 UTC timestamp; useful for debugging


def compute_config_hash(path: Path) -> str:
    """Return the SHA-256 digest of the file at *path*.

    Thin wrapper over :func:`paramem.backup.hashing.content_sha256_path` so
    Slice 1's hash primitive remains the single source of truth.

    Parameters
    ----------
    path:
        Filesystem path to ``server.yaml`` (or any file).

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest (64 characters).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    OSError
        If the file cannot be read.
    """
    return content_sha256_path(path)


def initial_drift_state(config_path: Path) -> ConfigDriftState:
    """Compute the load-time hash and seed an initial drift-state dict.

    Called once from the lifespan bootstrap.  The returned dict is stored on
    ``_state["config_drift"]`` so the ``/status`` endpoint can surface it
    immediately, before the first poll fires.

    Parameters
    ----------
    config_path:
        Filesystem path to the loaded ``server.yaml``.

    Returns
    -------
    ConfigDriftState
        ``detected=False``, ``loaded_hash`` set to the file's current digest,
        ``disk_hash`` equal to ``loaded_hash`` (no drift yet), and
        ``last_checked_at`` set to the current UTC time.

    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.  The lifespan must never call this
        before ``load_server_config`` confirms the file is readable; fail loud.
    """
    digest = compute_config_hash(config_path)
    now = datetime.now(timezone.utc).isoformat()
    return ConfigDriftState(
        detected=False,
        loaded_hash=digest,
        disk_hash=digest,
        last_checked_at=now,
    )


async def drift_poll_loop(
    config_path: Path,
    state_slot: dict,
    *,
    interval_seconds: float = 30.0,
) -> None:
    """Background coroutine that polls ``config_path`` for content changes.

    Re-hashes the file every ``interval_seconds`` and updates
    ``state_slot["config_drift"]`` in place.  The coroutine is designed to run
    indefinitely until cancelled by the lifespan shutdown block.

    Error handling: on ``OSError`` (file deleted, permissions changed, etc.)
    ``disk_hash`` is set to ``""`` and ``detected`` is set to ``True``.  The
    ``loaded_hash`` is never cleared — it remains the value captured at startup
    so callers can always identify which config version was loaded.

    Parameters
    ----------
    config_path:
        Filesystem path to watch.
    state_slot:
        The shared ``_state`` dict from ``app.py``.  Updated in place under
        the ``"config_drift"`` key.
    interval_seconds:
        Seconds between consecutive polls.  Defaults to 30 s in production;
        pass a smaller value in unit tests to drive the loop faster.
    """
    while True:
        await asyncio.sleep(interval_seconds)
        drift: ConfigDriftState = state_slot.get("config_drift", {})
        loaded_hash = drift.get("loaded_hash", "")
        now = datetime.now(timezone.utc).isoformat()
        try:
            disk_hash = compute_config_hash(config_path)
        except OSError:
            state_slot["config_drift"] = ConfigDriftState(
                detected=True,
                loaded_hash=loaded_hash,
                disk_hash="",
                last_checked_at=now,
            )
            logger.warning(
                "config drift poller: could not read %s — file may have been moved or deleted",
                config_path,
            )
            continue

        detected = disk_hash != loaded_hash
        state_slot["config_drift"] = ConfigDriftState(
            detected=detected,
            loaded_hash=loaded_hash,
            disk_hash=disk_hash,
            last_checked_at=now,
        )
        if detected:
            logger.warning(
                "config drift detected: server.yaml on disk differs from the "
                "version loaded at startup (loaded=%s… disk=%s…)",
                loaded_hash[:12],
                disk_hash[:12],
            )
