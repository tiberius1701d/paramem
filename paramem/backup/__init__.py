"""paramem.backup — artifact write/read primitives.

This package has no re-export surface: it carries no code of its own, and
importing ``paramem.backup`` (bare) executes nothing beyond this docstring.
Every submodule is imported directly, e.g. ``from paramem.backup.backup
import write, read`` or ``from paramem.backup.types import ArtifactMeta``.
This keeps a purely-typed import (``paramem.backup.types``, pure stdlib) free
of the heavy transitive graph that :mod:`paramem.backup.backup` pulls in
(:mod:`paramem.memory.interim_adapter` -> PEFT -> torch/transformers).

Submodules of note
-------------------
- :mod:`paramem.backup.backup` — ``write()``, ``read()``,
  ``sweep_orphan_pending()``, ``enforce_disk_cap()``.
- :mod:`paramem.backup.types` — ``ArtifactMeta``, ``ArtifactKind``,
  ``SCHEMA_VERSION``, and the error hierarchy (``BackupError``,
  ``DiskCapExceeded``, ``FingerprintMismatchError``, ``MetaSchemaError``,
  ``FatalConfigError``).
- :mod:`paramem.backup.hashing` — pure-stdlib content-hash helpers, safe to
  import at module scope from anywhere.
"""
