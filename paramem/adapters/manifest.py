"""Per-adapter meta.json schema and live-slot resolver.

STUB — full implementation lands in Slice 3.

This module owns the per-adapter meta.json schema (distinct from
paramem.backup.types.ArtifactMeta — see Resolved Decision 17) and the
live-slot resolver that matches meta.registry_sha256 against the hash
of the live registry file (Resolved Decision 31).

The per-adapter manifest is written by the training pipeline after every
successful training pass into:

    data/ha/adapters/<name>/<timestamp>/

It is consumed by:
- The server startup validator (refuses to mount a mismatched adapter).
- The migration preview (/migration/preview cross-checks candidate
  server.yaml against each adapter's meta.json).
- The trial state machine (Slice 3) for promotion and crash-recovery.

Planned Slice-3 surface (DO NOT implement in Slice 1):

# @dataclass(frozen=True)
# class AdapterManifest:
#     name: str
#     trained_at: str
#     base_model: BaseModelFingerprint
#     tokenizer: TokenizerFingerprint
#     lora: LoRAShape
#     registry_sha256: str
#     keyed_pairs_sha256: str
#     key_count: int
#
# def write_manifest(adapter_slot: Path, manifest: AdapterManifest) -> None: ...
# def read_manifest(adapter_slot: Path) -> AdapterManifest: ...
# def find_live_slot(adapter_dir: Path, live_registry_sha256: str) -> Path | None: ...
#     # Scan <adapter_dir>/<name>/<timestamp>/ for a manifest whose registry_sha256 matches.
#     # None → startup validator raises FINGERPRINT MISMATCH (primary: red; secondary: yellow).

References
----------
- Resolved Decision 17: per-adapter manifest schema (distinct from ArtifactMeta).
- Resolved Decision 31: live-slot determination by registry_sha256 match, not
  by a current-pointer file.
- Plan §6: Slice 3 is the first caller of find_live_slot (trial state machine).

TODO(slice3): implement AdapterManifest, write_manifest, read_manifest,
find_live_slot in this module.
"""
