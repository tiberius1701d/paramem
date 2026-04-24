"""Shared types for the backup/restore subsystem.

Defines the public dataclasses, enums, TypedDicts, and exceptions used across
all backup modules.  No I/O, no side effects — import is always safe.

Schema version contract
-----------------------
``SCHEMA_VERSION`` is bumped on every breaking change to ``ArtifactMeta``.
Adding an *Optional* field with a default value is non-breaking (no bump).
Renaming a field, changing its semantics, or removing it **requires** a bump
plus a migration helper (Slice 6 concern).

``read_meta`` rejects sidecars whose ``schema_version`` does not equal
``SCHEMA_VERSION`` — forward versions (written by a newer release) raise
``MetaSchemaError("forward version")``; legacy versions (written by an older
release that was already bumped) raise ``MetaSchemaError("legacy; migration
required")``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# ---------------------------------------------------------------------------
# Schema version — increment + migration required for every breaking change.
# ---------------------------------------------------------------------------

SCHEMA_VERSION: int = 1


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ArtifactKind(str, Enum):
    """Identifies which subsystem produced an artifact.

    The string value is stored verbatim in the ``.meta.json`` sidecar; it
    must not be changed without bumping ``SCHEMA_VERSION``.
    """

    CONFIG = "config"
    GRAPH = "graph"
    REGISTRY = "registry"
    RESUME = "resume"  # BG-trainer per-file artifacts
    SNAPSHOT = "snapshot"  # session snapshot, per-file


# ---------------------------------------------------------------------------
# ArtifactMeta — the .meta.json sidecar schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactMeta:
    """Immutable sidecar record stored alongside every backup artifact.

    Written by ``backup.write()``; read and validated by ``backup.read()``.
    ``frozen=True`` prevents accidental mutation after construction.

    Fields
    ------
    schema_version : int
        Must equal ``SCHEMA_VERSION`` at write time.  ``read_meta`` refuses
        sidecars with any other value (see module docstring).
    kind : ArtifactKind
        Which artifact this sidecar belongs to.
    timestamp : str
        ISO 8601 UTC timestamp aligned with the slot directory name
        (``YYYYMMDD-HHMMSSff`` format, hundredths of a second).
    content_sha256 : str
        Hex digest of the raw bytes *as written to disk* — ciphertext when
        ``encrypted=True``, plaintext otherwise.  Whitespace and key-order
        changes in the source data are visible as hash changes (Resolved
        Decision 29 — no YAML canonicalization).
    size_bytes : int
        Byte count of the artifact file on disk (after encryption if applied).
    encrypted : bool
        ``True`` when the paired file is an age envelope.
    tier : str
        Backup tier tag — one of ``"scheduled"``, ``"pre-migration"``,
        ``"manual"``, ``"trial_adapter"``.
    label : str | None
        Optional user-supplied annotation.

    Note: there is no ``registry_sha256`` field.  That field belongs to the
    per-adapter manifest (``paramem.adapters.manifest.AdapterManifest``).
    See Resolved Decision 17.
    """

    schema_version: int
    kind: ArtifactKind
    timestamp: str
    content_sha256: str
    size_bytes: int
    encrypted: bool
    tier: str
    label: str | None = None
    pre_trial_hash: str | None = None
    """SHA-256 of the live config at the moment /migration/confirm ran step 2.

    Written into every pre-migration backup's sidecar by the confirm handler.
    Used by crash recovery (case 3/4) to correlate an orphan backup with the
    live config hash.  Optional (default None) — absent in all non-migration
    backups; adding it is non-breaking per the schema-version contract (see
    module docstring).
    """


# ---------------------------------------------------------------------------
# PruneReport
# ---------------------------------------------------------------------------


@dataclass
class PruneReport:
    """Result of a single prune() call.

    ``invalid`` entries are slots where the sidecar is missing, unreadable,
    or schema-mismatched.  They are *not* deleted by ``prune()`` — operator
    visibility and remediation are Slice 5/6 responsibilities.

    Fields
    ------
    kept : list[Path]
        Slot directories retained after applying the retention policy.
    deleted : list[Path]
        Slot directories removed.
    skipped_live : list[Path]
        Slots that would have been pruned but were protected by the
        ``live_slot`` parameter.  Typically zero or one entry.
    invalid : list[tuple[Path, str]]
        ``(slot_dir, reason)`` pairs for corrupt/missing/schema-mismatch
        sidecars.  Surfaced by the operator-visibility layer in Slice 5/6.
    """

    kept: list[Path] = field(default_factory=list)
    deleted: list[Path] = field(default_factory=list)
    skipped_live: list[Path] = field(default_factory=list)
    invalid: list[tuple[Path, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RetentionPolicy — shape only; defaults/enforcement engine land in Slice 6
# ---------------------------------------------------------------------------

RetentionPolicy = dict  # typed alias for {keep, max_disk_gb, immunity_days}
# Full TypedDict definition deferred to Slice 6 where server.yaml wiring lands.
# Slice 1 callers may pass a plain dict with the keys documented below.
#
#   keep          : int | "unlimited"  — max slots to retain in the tier
#   max_disk_gb   : float | None       — tier-level disk cap (None = no cap)
#   immunity_days : int | None         — slots younger than this are immune
#                                        from keep-count pruning

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BackupError(Exception):
    """Base class for all backup-subsystem errors."""


class FingerprintMismatchError(BackupError):
    """Content hash of an artifact does not match its sidecar record.

    Raised by ``backup.read()`` after reading the artifact file.  Indicates
    either disk corruption or tampering.
    """


class MetaSchemaError(BackupError):
    """Sidecar fails schema validation.

    Raised on:
    - ``schema_version`` mismatch (forward or legacy)
    - Missing required fields
    - Unknown ``kind`` value

    Callers must not proceed with artifact data when this is raised.
    """


class FatalConfigError(BackupError):
    """A fatal configuration problem was detected.

    Raised by ``security_posture.assert_startup_posture()`` when
    ``security.require_encryption=true`` is set but no key is loadable,
    and by ``encryption.assert_mode_consistency()`` on key × on-disk format
    mismatches.  The server refuses to start.
    """
