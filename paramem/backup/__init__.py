"""paramem.backup — artifact write/read/prune primitives.

Public surface
--------------
- ``write()``  — write an artifact + sidecar into a new slot directory.
- ``read()``   — read an artifact from a slot, validate, and decrypt.
- ``prune()``  — apply a retention policy to a kind directory.
- ``sweep_orphan_pending()`` — startup sweep of incomplete ``.pending/`` dirs.
- ``enforce_disk_cap()`` — the write-door disk-cap predicate, callable ahead
  of a write that hasn't started yet (e.g. the pre-base-swap gate).

Types
-----
- ``ArtifactMeta``       — immutable sidecar schema dataclass.
- ``ArtifactKind``       — enum: config, graph, registry, resume, snapshot.
- ``PruneReport``        — result of a prune() call.
- ``RetentionPolicy``    — dict alias for {keep, max_disk_gb, immunity_days}.

Errors
------
- ``BackupError``              — base class.
- ``DiskCapExceeded``          — write refused, backup store at/over its cap.
- ``FingerprintMismatchError`` — content hash mismatch on read.
- ``MetaSchemaError``          — sidecar schema validation failure.
- ``FatalConfigError``         — startup refused (require_encryption / mode mismatch).
"""

from paramem.backup.backup import (
    enforce_disk_cap,
    prune,
    read,
    sweep_orphan_pending,
    write,
)
from paramem.backup.types import (
    SCHEMA_VERSION,
    ArtifactKind,
    ArtifactMeta,
    BackupError,
    DiskCapExceeded,
    FatalConfigError,
    FingerprintMismatchError,
    MetaSchemaError,
    PruneReport,
    RetentionPolicy,
)

__all__ = [
    # Functions
    "write",
    "read",
    "prune",
    "sweep_orphan_pending",
    "enforce_disk_cap",
    # Types
    "ArtifactMeta",
    "ArtifactKind",
    "PruneReport",
    "RetentionPolicy",
    "SCHEMA_VERSION",
    # Errors
    "BackupError",
    "DiskCapExceeded",
    "FingerprintMismatchError",
    "MetaSchemaError",
    "FatalConfigError",
]
