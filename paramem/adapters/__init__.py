"""paramem.adapters — adapter management subpackage.

Public surface
--------------
Manifest types, functions, and errors from :mod:`paramem.adapters.manifest`,
plus the slot promotion envelope from :mod:`paramem.adapters.slot`.
"""

from paramem.adapters.manifest import (
    MANIFEST_SCHEMA_VERSION,
    UNKNOWN,
    AdapterManifest,
    BaseModelFingerprint,
    LoRAShape,
    ManifestError,
    ManifestNotFoundError,
    ManifestSchemaError,
    PayloadFingerprint,
    TokenizerFingerprint,
    build_manifest_for,
    find_live_slot,
    graph_payload_manifest,
    iter_slot_candidates,
    read_manifest,
    resolve_adapter_slot,
    write_manifest,
)
from paramem.adapters.slot import (
    payload_filename,
    required_slot_files,
    write_slot,
)

__all__ = [
    # Sentinels
    "MANIFEST_SCHEMA_VERSION",
    "UNKNOWN",
    # Dataclasses
    "AdapterManifest",
    "BaseModelFingerprint",
    "PayloadFingerprint",
    "TokenizerFingerprint",
    "LoRAShape",
    # Functions
    "build_manifest_for",
    "find_live_slot",
    "graph_payload_manifest",
    "iter_slot_candidates",
    "payload_filename",
    "read_manifest",
    "required_slot_files",
    "resolve_adapter_slot",
    "write_slot",
    "write_manifest",
    # Errors
    "ManifestError",
    "ManifestNotFoundError",
    "ManifestSchemaError",
]
