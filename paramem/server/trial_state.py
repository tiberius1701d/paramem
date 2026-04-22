"""Trial-state marker I/O for the ParaMem migration subsystem.

Implements the ``state/trial.json`` plaintext marker that bridges a process
crash between /migration/confirm steps 3–4 and the next server startup.  The
marker is always plaintext (never encrypted) so the CLI fallback and crash
recovery can read it without key material.

Schema contract
---------------
``TRIAL_MARKER_SCHEMA_VERSION`` is bumped on every breaking change.  The
version is checked on read; forward or wrong-version markers raise
``TrialMarkerSchemaError`` rather than being silently tolerated or silently
deleted.  An unreadable marker is treated as AMBIGUOUS by crash recovery.

Atomic write pattern
--------------------
``write_trial_marker`` follows the same ``.pending/<file>`` + ``os.rename``
pattern as ``paramem.backup.backup.write()`` to guarantee that a crash
mid-write leaves either no marker or a fully-written marker, never a
partially-written file.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

TRIAL_MARKER_SCHEMA_VERSION: int = 1
TRIAL_MARKER_FILENAME: str = "trial.json"

_PENDING_DIRNAME: str = ".pending"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TrialMarkerError(Exception):
    """Base class for trial-marker I/O errors."""


class TrialMarkerSchemaError(TrialMarkerError):
    """Marker fails schema validation (version mismatch, missing field, bad JSON)."""


# ---------------------------------------------------------------------------
# TrialMarker dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialMarker:
    """Contents of state/trial.json.

    Plaintext per spec L607 — contains no secrets, so key rotation never
    bricks recovery.  The CLI fallback in migrate_status.py reads this file
    directly when the server is offline.

    Paths stored in ``backup_paths``, ``trial_adapter_dir``, and
    ``trial_graph_dir`` are absolute (Correction 5) so the marker is
    portable across working-directory changes (e.g. systemd units).

    Attributes
    ----------
    schema_version:
        Always ``TRIAL_MARKER_SCHEMA_VERSION`` at write time.
    started_at:
        ISO-8601 UTC timestamp when TRIAL was entered.
    pre_trial_config_sha256:
        SHA-256 of the live ``server.yaml`` bytes before the atomic rename
        (step 4 of /migration/confirm).
    candidate_config_sha256:
        SHA-256 of the candidate bytes (matches ``stash["candidate_hash"]``).
    backup_paths:
        Dict ``{"config": "<abs_path>", "graph": "<abs_path>",
        "registry": "<abs_path>"}`` pointing at the three pre-migration
        backup slot directories created in step 2.  Paths are absolute.
    trial_adapter_dir:
        Absolute path to the directory where the trial consolidation will
        write adapter weights (``data/ha/state/trial_adapter/``).
    trial_graph_dir:
        Absolute path to the directory where the trial consolidation will
        persist the trial graph (``data/ha/state/trial_graph/``).
    config_artifact_filename:
        Filename of the config artifact file inside the config backup slot
        (e.g. ``"config-20260421-040000.bin"``).  Used by the rollback handler
        to resolve the exact A-config file without directory listing.
        Defaults to ``""`` for backward compatibility with markers written
        before Slice 3b.3; the rollback handler asserts non-empty.
    """

    schema_version: int
    started_at: str
    pre_trial_config_sha256: str
    candidate_config_sha256: str
    backup_paths: dict[str, str]
    trial_adapter_dir: str
    trial_graph_dir: str
    config_artifact_filename: str

    def to_dict(self) -> dict:
        """Serialize the marker to a JSON-ready dict."""
        return {
            "schema_version": self.schema_version,
            "started_at": self.started_at,
            "pre_trial_config_sha256": self.pre_trial_config_sha256,
            "candidate_config_sha256": self.candidate_config_sha256,
            "backup_paths": dict(self.backup_paths),
            "trial_adapter_dir": self.trial_adapter_dir,
            "trial_graph_dir": self.trial_graph_dir,
            "config_artifact_filename": self.config_artifact_filename,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrialMarker":
        """Deserialize a marker from a JSON dict.

        Raises ``TrialMarkerSchemaError`` on missing fields or wrong
        schema version.
        """
        if not isinstance(d, dict):
            raise TrialMarkerSchemaError("marker root is not a JSON object")

        version = d.get("schema_version")
        if version is None:
            raise TrialMarkerSchemaError("marker missing required field: schema_version")
        if version != TRIAL_MARKER_SCHEMA_VERSION:
            raise TrialMarkerSchemaError(
                f"marker schema_version={version} does not match "
                f"current TRIAL_MARKER_SCHEMA_VERSION={TRIAL_MARKER_SCHEMA_VERSION}"
            )

        required = (
            "started_at",
            "pre_trial_config_sha256",
            "candidate_config_sha256",
            "backup_paths",
            "trial_adapter_dir",
            "trial_graph_dir",
        )
        missing = [f for f in required if f not in d]
        if missing:
            raise TrialMarkerSchemaError(f"marker missing required fields: {sorted(missing)}")

        return cls(
            schema_version=version,
            started_at=d["started_at"],
            pre_trial_config_sha256=d["pre_trial_config_sha256"],
            candidate_config_sha256=d["candidate_config_sha256"],
            backup_paths=dict(d["backup_paths"]),
            trial_adapter_dir=d["trial_adapter_dir"],
            trial_graph_dir=d["trial_graph_dir"],
            # Default to "" for backward compatibility with markers written
            # before Slice 3b.3. The rollback handler asserts non-empty.
            config_artifact_filename=d.get("config_artifact_filename", ""),
        )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def write_trial_marker(state_dir: Path, marker: TrialMarker) -> Path:
    """Atomically write ``state_dir/trial.json`` via a ``.pending`` rename.

    Mirrors the ``paramem.backup.backup.write()`` crash-safe pattern:

    1. Create ``state_dir/.pending/`` if absent.
    2. Write ``trial.json`` into the pending directory (with paths resolved).
    3. ``fsync`` the file fd.
    4. ``fsync`` the pending directory entry.
    5. ``os.rename(.pending/trial.json, state_dir/trial.json)`` — atomic.
    6. ``fsync(state_dir)`` for rename durability.

    Path fields (``backup_paths``, ``trial_adapter_dir``, ``trial_graph_dir``)
    are resolved to absolute paths at write time (``Path.resolve()``) so the
    marker is portable across working-directory changes (e.g. systemd units).
    Relative inputs are resolved against the process cwd when this function is
    called; absolute inputs are unchanged.

    Parameters
    ----------
    state_dir:
        Directory that will contain ``trial.json``
        (e.g. ``data/ha/state/``).  Created with parents if absent.
    marker:
        Fully-populated ``TrialMarker`` to serialise.  Path strings may be
        relative; they will be resolved before writing.

    Returns
    -------
    Path
        The written ``state_dir/trial.json`` path.

    Raises
    ------
    OSError
        On any filesystem error.
    """
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    pending_dir = state_dir / _PENDING_DIRNAME
    pending_dir.mkdir(exist_ok=True)

    pending_file = pending_dir / TRIAL_MARKER_FILENAME
    final_file = state_dir / TRIAL_MARKER_FILENAME

    # Resolve backup_paths, trial_adapter_dir, and trial_graph_dir to absolute
    # strings so the marker is portable across working-directory changes
    # (e.g. systemd units that start in a different cwd).  Relative inputs
    # are resolved against the process cwd at write time.
    data = marker.to_dict()
    data["backup_paths"] = {k: str(Path(v).resolve()) for k, v in data["backup_paths"].items()}
    data["trial_adapter_dir"] = str(Path(data["trial_adapter_dir"]).resolve())
    data["trial_graph_dir"] = str(Path(data["trial_graph_dir"]).resolve())

    payload = json.dumps(data, indent=2).encode("utf-8")
    pending_file.write_bytes(payload)

    # fsync the file
    with open(pending_file, "rb") as fh:
        os.fsync(fh.fileno())

    # fsync the pending directory
    dir_fd = os.open(str(pending_dir), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    except OSError as exc:
        logger.warning("write_trial_marker: pending dir fsync failed: %s", exc)
    finally:
        os.close(dir_fd)

    # Atomic rename into the final location
    os.rename(pending_file, final_file)

    # fsync state_dir for rename durability
    dir_fd = os.open(str(state_dir), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    except OSError as exc:
        logger.warning("write_trial_marker: state_dir fsync failed: %s", exc)
    finally:
        os.close(dir_fd)

    logger.info("write_trial_marker: written to %s", final_file)
    return final_file


def read_trial_marker(state_dir: Path) -> TrialMarker | None:
    """Read and parse ``state_dir/trial.json``.

    Returns ``None`` when the file does not exist (normal LIVE case — not an
    error).  Raises ``TrialMarkerSchemaError`` when the file exists but fails
    schema validation.  Raises ``TrialMarkerSchemaError`` on JSON parse
    errors.

    Parameters
    ----------
    state_dir:
        Directory containing ``trial.json``.

    Returns
    -------
    TrialMarker | None
        Parsed marker, or ``None`` when the file is absent.

    Raises
    ------
    TrialMarkerSchemaError
        On parse failure or schema mismatch.
    """
    path = Path(state_dir) / TRIAL_MARKER_FILENAME
    if not path.exists():
        return None

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrialMarkerSchemaError(f"trial.json is not valid JSON: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise TrialMarkerSchemaError(f"trial.json is not valid UTF-8: {exc}") from exc

    return TrialMarker.from_dict(raw)


def clear_trial_marker(state_dir: Path) -> None:
    """Remove ``state_dir/trial.json`` and fsync the parent directory.

    No-op when the file does not exist.

    Parameters
    ----------
    state_dir:
        Directory containing ``trial.json``.
    """
    path = Path(state_dir) / TRIAL_MARKER_FILENAME
    if not path.exists():
        return

    path.unlink()

    # fsync the directory so the unlink is durable.
    dir_fd = os.open(str(state_dir), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    except OSError as exc:
        logger.warning("clear_trial_marker: state_dir fsync failed: %s", exc)
    finally:
        os.close(dir_fd)

    logger.info("clear_trial_marker: removed %s", path)
