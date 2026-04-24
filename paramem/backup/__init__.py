"""paramem.backup — artifact write/read/prune primitives.

Public surface (Slice 1)
------------------------
- ``write()``  — write an artifact + sidecar into a new slot directory.
- ``read()``   — read an artifact from a slot, validate, and decrypt.
- ``prune()``  — apply a retention policy to a kind directory.
- ``sweep_orphan_pending()`` — startup sweep of incomplete ``.pending/`` dirs.

Types
-----
- ``ArtifactMeta``       — immutable sidecar schema dataclass.
- ``ArtifactKind``       — enum: config, graph, registry, resume, snapshot.
- ``PruneReport``        — result of a prune() call.
- ``RetentionPolicy``    — dict alias for {keep, max_disk_gb, immunity_days}.

Errors
------
- ``BackupError``              — base class.
- ``FingerprintMismatchError`` — content hash mismatch on read.
- ``MetaSchemaError``          — sidecar schema validation failure.
- ``FatalConfigError``         — startup refused (require_encryption / mode mismatch).
"""

from paramem.backup.backup import (
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
    # Types
    "ArtifactMeta",
    "ArtifactKind",
    "PruneReport",
    "RetentionPolicy",
    "SCHEMA_VERSION",
    # Errors
    "BackupError",
    "FingerprintMismatchError",
    "MetaSchemaError",
    "FatalConfigError",
]
