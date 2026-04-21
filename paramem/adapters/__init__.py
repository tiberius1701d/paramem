"""paramem.adapters — adapter management subpackage.

Public surface (Slice 3a)
--------------------------
All manifest types, functions, errors, and sentinels from
:mod:`paramem.adapters.manifest`.
"""

from paramem.adapters.manifest import (
    MANIFEST_SCHEMA_VERSION,
    UNKNOWN,
    AdapterManifest,
    BaseModelFingerprint,
    LoRAShape,
    ManifestError,
    ManifestFingerprintMismatchError,
    ManifestNotFoundError,
    ManifestSchemaError,
    TokenizerFingerprint,
    build_manifest_for,
    find_live_slot,
    read_manifest,
    resolve_adapter_slot,
    write_manifest,
)

__all__ = [
    # Sentinels
    "MANIFEST_SCHEMA_VERSION",
    "UNKNOWN",
    # Dataclasses
    "AdapterManifest",
    "BaseModelFingerprint",
    "TokenizerFingerprint",
    "LoRAShape",
    # Functions
    "build_manifest_for",
    "find_live_slot",
    "read_manifest",
    "resolve_adapter_slot",
    "write_manifest",
    # Errors
    "ManifestError",
    "ManifestNotFoundError",
    "ManifestSchemaError",
    "ManifestFingerprintMismatchError",
]
