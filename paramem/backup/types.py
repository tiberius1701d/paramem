"""Shared types for the backup/restore subsystem.

Defines the public dataclasses, enums, TypedDicts, and exceptions used across
all backup modules.  No I/O, no side effects — import is always safe.

Schema version contract (ArtifactMeta)
---------------------------------------
``SCHEMA_VERSION`` is bumped on every breaking change to ``ArtifactMeta``.
Adding an *Optional* field with a default value is non-breaking (no bump).
Renaming a field, changing its semantics, or removing it **requires** a bump
plus a migration helper.

``read_meta`` rejects sidecars whose ``schema_version`` does not equal
``SCHEMA_VERSION`` — forward versions (written by a newer release) raise
``MetaSchemaError("forward version")``; legacy versions (written by an older
release that was already bumped) raise ``MetaSchemaError("legacy; migration
required")``.

Bundle manifest schema (BundleManifest)
----------------------------------------
``BUNDLE_SCHEMA_VERSION`` is versioned independently from ``SCHEMA_VERSION``.
A bundle slot stores a ``bundle.meta.json`` in place of the per-artifact
``.meta.json`` sidecar.  The bundle manifest indexes every captured file with
its content hash, records the per-adapter slot source information, and lists
explicitly-excluded artifacts.  Incrementing ``BUNDLE_SCHEMA_VERSION`` does
NOT require a ``SCHEMA_VERSION`` bump and vice versa.
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
# Bundle manifest schema version — independent of SCHEMA_VERSION.
# ---------------------------------------------------------------------------

BUNDLE_SCHEMA_VERSION: int = 1


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
    SNAPSHOT_BUNDLE = "snapshot_bundle"  # self-contained recovery-set bundle


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
# BundleManifest — the bundle.meta.json top-level manifest schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BundleManifest:
    """Immutable top-level manifest written as ``bundle.meta.json`` inside
    every self-contained recovery-set bundle slot.

    A bundle slot aggregates the full recovery set (config, registry, adapter
    weights, speaker profiles) into one timestamped directory.  The bundle
    manifest is the sole index for that directory — no per-artifact
    ``ArtifactMeta`` sidecars are written inside a bundle slot.

    Schema versioning: ``bundle_schema_version`` is independent of the
    per-artifact ``SCHEMA_VERSION``.  Additions of optional fields (with
    defaults) are non-breaking.  Rename or removal requires a
    ``BUNDLE_SCHEMA_VERSION`` bump and a migration helper.

    Fields
    ------
    bundle_schema_version : int
        Must equal ``BUNDLE_SCHEMA_VERSION`` at write time.  Readers should
        refuse manifests with an unrecognised version.
    created_at : str
        ISO-8601 UTC timestamp (e.g. ``"2026-05-20T20:55:00Z"``).
    tier : str
        Backup tier — one of ``"scheduled"``, ``"manual"``,
        ``"pre_migration"``, etc.
    label : str | None
        Optional operator-supplied annotation.
    live_registry_sha256 : str
        SHA-256 hex of the registry (``key_metadata.json``) bytes at bundle
        creation time.  This hash ties the captured adapter weights to the
        captured registry: ``find_live_slot(adapter_kind_dir,
        live_registry_sha256)`` selects exactly the weights stored in this
        bundle.
    base_model : dict
        Base-model identity copied from the first enabled adapter's
        ``meta.json``.  Expected keys: ``repo`` (str), ``sha`` (str),
        ``hash`` (str).  Empty dict when no adapter meta was available.
    files : list[dict]
        One entry per captured file.  Each entry is a dict with keys:
        ``path`` (str, relative to bundle slot root), ``content_sha256``
        (str, hex), ``encrypted`` (bool), ``size_bytes`` (int).
    adapters : dict
        Per-tier adapter capture record.  Keys are adapter names (e.g.
        ``"episodic"``).  Each value is a dict with:
        ``slot_source`` (str, original slot path),
        ``registry_sha256`` (str),
        ``key_count`` (int | str),
        ``simhash_present`` (bool),
        ``keyed_pairs_present`` (bool, always False — QA pairs are
        transient and regenerated from the graph on every cycle).
    excluded : list[str]
        Human-readable list of artifact categories intentionally excluded
        from this bundle (e.g. ``"graph (RAM-only by design)"``).
    """

    bundle_schema_version: int
    created_at: str
    tier: str
    label: str | None
    live_registry_sha256: str
    base_model: dict
    files: list[dict]
    adapters: dict
    excluded: list[str]

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for ``json.dumps``.

        Returns
        -------
        dict
            All fields as JSON-serialisable types.
        """
        return {
            "bundle_schema_version": self.bundle_schema_version,
            "created_at": self.created_at,
            "tier": self.tier,
            "label": self.label,
            "live_registry_sha256": self.live_registry_sha256,
            "base_model": self.base_model,
            "files": list(self.files),
            "adapters": dict(self.adapters),
            "excluded": list(self.excluded),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BundleManifest":
        """Deserialise from a plain dict (as loaded from ``bundle.meta.json``).

        Parameters
        ----------
        data:
            Dict loaded from the bundle manifest file.

        Returns
        -------
        BundleManifest

        Raises
        ------
        BundleManifestError
            If required fields are missing or ``bundle_schema_version`` does
            not equal ``BUNDLE_SCHEMA_VERSION``.
        """
        version = data.get("bundle_schema_version")
        if version != BUNDLE_SCHEMA_VERSION:
            raise BundleManifestError(
                f"bundle_schema_version mismatch: expected {BUNDLE_SCHEMA_VERSION}, got {version!r}"
            )
        required = (
            "created_at",
            "tier",
            "live_registry_sha256",
            "base_model",
            "files",
            "adapters",
            "excluded",
        )
        for key in required:
            if key not in data:
                raise BundleManifestError(f"bundle manifest missing required field: {key!r}")
        return cls(
            bundle_schema_version=version,
            created_at=data["created_at"],
            tier=data["tier"],
            label=data.get("label"),
            live_registry_sha256=data["live_registry_sha256"],
            base_model=data["base_model"],
            files=data["files"],
            adapters=data["adapters"],
            excluded=data["excluded"],
        )


# ---------------------------------------------------------------------------
# PruneReport
# ---------------------------------------------------------------------------


@dataclass
class PruneReport:
    """Result of a single prune() call.

    ``invalid`` entries are slots where the sidecar is missing, unreadable,
    or schema-mismatched.  They are *not* deleted by ``prune()`` — operator
    visibility and remediation are the responsibility of the operator-attention
    layer.

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
        sidecars.  Surfaced by the operator-attention layer.
    """

    kept: list[Path] = field(default_factory=list)
    deleted: list[Path] = field(default_factory=list)
    skipped_live: list[Path] = field(default_factory=list)
    invalid: list[tuple[Path, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RetentionPolicy — typed alias for the retention policy dict
# ---------------------------------------------------------------------------

RetentionPolicy = dict  # typed alias for {keep, max_disk_gb, immunity_days}
# Callers may pass a plain dict with the keys documented below.
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


class BundleManifestError(BackupError):
    """Bundle manifest fails schema validation.

    Raised by ``BundleManifest.from_dict()`` on:

    - ``bundle_schema_version`` mismatch (forward or unknown version).
    - Missing required fields in the manifest dict.

    Callers must not treat bundle content as trustworthy when this is raised.
    """


class FatalConfigError(BackupError):
    """A fatal configuration problem was detected.

    Raised by ``security_posture.assert_startup_posture()`` when
    ``security.require_encryption=true`` is set but no key is loadable,
    and by ``encryption.assert_mode_consistency()`` on key × on-disk format
    mismatches.  The server refuses to start.
    """


class RestoreAbortedError(BackupError):
    """Raised by ``restore_bundle()`` when a filesystem error occurs in the
    atomic restore phase (step 5), after the safety bundle has already been
    captured (step 4).

    Carries ``safety_slot`` so callers (e.g. the ``/backup/restore`` HTTP
    handler) can surface the recovery path in the error response without
    requiring the operator to search server logs.

    Attributes
    ----------
    safety_slot : Path | None
        Path to the pre-restore safety bundle captured before any live
        mutation.  ``None`` when the safety bundle was skipped (fresh/empty
        target).
    cause : BaseException
        The original exception that triggered the abort.
    """

    def __init__(self, message: str, safety_slot: "Path | None", cause: BaseException) -> None:
        super().__init__(message)
        self.safety_slot = safety_slot
        self.cause = cause


# ---------------------------------------------------------------------------
# RestoreResult — returned by restore_bundle()
# ---------------------------------------------------------------------------


@dataclass
class RestoreResult:
    """Result of a successful ``restore_bundle()`` call.

    Fields
    ------
    restored_adapters : list[str]
        Names of the adapter entries restored from the bundle (e.g.
        ``["episodic", "episodic_interim_20260517T1200", "procedural"]``).
        Each entry corresponds to a key in the bundle manifest's ``adapters``
        dict for which a new slot directory was written under ``data_dir``.
    safety_slot : Path | None
        Path to the pre-restore safety bundle that was captured before any
        mutation occurred.  ``None`` when the target ``data_dir`` had no
        episodic slot (fresh / empty target) and the safety bundle write
        was skipped gracefully.
    restart_required : bool
        Always ``True`` — the in-VRAM adapters are stale after a restore.
        A server restart re-mounts adapters from the freshly restored slots
        via ``find_live_slot``.  No hot-swap is performed (8 GB VRAM
        constraint; restart is the clean boundary).
    restored_config : bool
        ``True`` when the bundle's ``server.yaml`` was atomically written to
        ``config_path`` (only when ``restore_config=True`` was requested).
        ``False`` otherwise — the live config is left untouched.
    """

    restored_adapters: list[str] = field(default_factory=list)
    safety_slot: Path | None = None
    restart_required: bool = True
    restored_config: bool = False
