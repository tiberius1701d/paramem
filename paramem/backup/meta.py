"""Sidecar (.meta.json) read/write and fingerprint verification.

Each artifact in a slot directory has a paired ``.meta.json`` file.  This
module owns the serialisation / deserialisation of ``ArtifactMeta`` and the
content-hash fingerprint check that guards against silent on-disk corruption.

Schema version policy (NIT 2)
-------------------------------
``read_meta`` enforces a strict schema-version gate:

- ``schema_version == SCHEMA_VERSION`` → accepted.
- ``schema_version > SCHEMA_VERSION`` → raises ``MetaSchemaError("forward version")``
  and logs ERROR.  A future release wrote this sidecar; the current code
  cannot know whether the schema is compatible.
- ``schema_version < SCHEMA_VERSION`` → raises
  ``MetaSchemaError("legacy; migration required")``.  A past release wrote
  this sidecar; a migration helper is required before the artifact can be
  used.  In practice Slice 1 starts at version 1, so this path is never
  taken by code that ships this version — it is documented here as defensive
  coverage for future bumps.

File naming convention
-----------------------
The sidecar is named ``<artifact_stem>.meta.json`` where ``<artifact_stem>``
is the stem of the artifact filename (e.g. ``config-20260421-040000``) so
that artifact + sidecar share a common prefix and are visually associated
in directory listings.

The single-sidecar-per-slot invariant is enforced by ``write_meta``
(overwrite) and ``read_meta`` (reads the sole ``.meta.json`` in the slot).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from paramem.backup.encryption import read_maybe_encrypted
from paramem.backup.hashing import content_sha256_path
from paramem.backup.types import (
    SCHEMA_VERSION,
    ArtifactKind,
    ArtifactMeta,
    FingerprintMismatchError,
    MetaSchemaError,
)

logger = logging.getLogger(__name__)

_META_SUFFIX = ".meta.json"


def _meta_path(slot_dir: Path) -> Path:
    """Return the path to the sole ``.meta.json`` file in *slot_dir*.

    Used internally by ``write_meta`` and ``read_meta``.  Does *not* verify
    that the file exists — callers handle ``FileNotFoundError``.

    The slot-naming convention is ``<artifact_stem>.meta.json``.
    ``write_meta`` derives the stem from ``meta.kind`` and ``meta.timestamp``;
    ``read_meta`` locates the file by glob.
    """
    matches = list(slot_dir.glob("*" + _META_SUFFIX))
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        # Return a deterministic "expected" path for error messages
        return slot_dir / ("artifact" + _META_SUFFIX)
    # Multiple sidecars — slot is corrupt; pick the first for the error message
    raise MetaSchemaError(
        f"slot {slot_dir} contains multiple .meta.json files: {[m.name for m in matches]}"
    )


def write_meta(slot_dir: Path, meta: ArtifactMeta) -> Path:
    """Serialise *meta* into ``<slot_dir>/<kind>-<timestamp>.meta.json``.

    Called inside the ``.pending/`` write sequence before the directory is
    fsync'd and renamed into the live slot.

    Parameters
    ----------
    slot_dir:
        Directory that will contain the artifact + sidecar pair.  Must exist.
    meta:
        Fully populated ``ArtifactMeta`` instance.

    Returns
    -------
    Path
        Path to the written sidecar file.

    Raises
    ------
    OSError
        If the directory does not exist or the file cannot be written.
    """
    stem = f"{meta.kind.value}-{meta.timestamp}"
    sidecar_path = slot_dir / (stem + _META_SUFFIX)

    payload: dict = {
        "schema_version": meta.schema_version,
        "kind": meta.kind.value,
        "timestamp": meta.timestamp,
        "content_sha256": meta.content_sha256,
        "size_bytes": meta.size_bytes,
        "encrypted": meta.encrypted,
        "key_fingerprint": meta.key_fingerprint,
        "tier": meta.tier,
        "label": meta.label,
        "pre_trial_hash": meta.pre_trial_hash,
    }

    # Plaintext-by-design (SECURITY.md §4 carve-out): backup sidecars are
    # control-plane metadata only (timestamp, content_sha256 of the already-
    # encrypted payload, size, tier, label, key_fingerprint).  Encrypting them
    # would turn every wrong-key restore into a silent "backup not found"
    # instead of a clear decrypt_invalid_token error.  The user-facts live in
    # the paired `.bin.enc` artifact, which stays encrypted.  ``read_meta``
    # still tolerates PMEM1-wrapped sidecars so any pre-existing ciphertext
    # sidecar remains readable.
    sidecar_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return sidecar_path


def read_meta(slot_dir: Path) -> ArtifactMeta:
    """Read and validate the ``.meta.json`` sidecar in *slot_dir*.

    Schema-version gate (NIT 2):
    - Forward version (``schema_version > SCHEMA_VERSION``) → logs ERROR,
      raises ``MetaSchemaError("forward version")``.
    - Legacy version (``schema_version < SCHEMA_VERSION``) → raises
      ``MetaSchemaError("legacy; migration required")``.

    Parameters
    ----------
    slot_dir:
        Slot directory containing exactly one ``.meta.json`` sidecar.

    Returns
    -------
    ArtifactMeta
        Validated, fully populated dataclass instance.

    Raises
    ------
    MetaSchemaError
        On schema_version mismatch, missing required fields, unknown kind, or
        if the sidecar file is not valid UTF-8 JSON (binary garbage, truncated
        file, or any other parse failure).  The message begins with
        ``"corrupt sidecar:"`` for parse errors.
    FileNotFoundError
        If no ``.meta.json`` file exists in *slot_dir*.
    """
    try:
        meta_file = _meta_path(slot_dir)
    except MetaSchemaError:
        raise

    if not meta_file.exists():
        raise FileNotFoundError(f"No .meta.json sidecar found in {slot_dir}")

    # Sidecars may be PMEM1-wrapped (Security ON) or plaintext (Security OFF).
    # read_maybe_encrypted handles both; InvalidToken surfaces as a corrupt
    # sidecar for the caller.
    from cryptography.fernet import InvalidToken

    try:
        plaintext = read_maybe_encrypted(meta_file)
        raw = json.loads(plaintext.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MetaSchemaError(f"corrupt sidecar: {meta_file} — {exc}") from exc
    except InvalidToken as exc:
        raise MetaSchemaError(
            f"corrupt sidecar: {meta_file} — ciphertext failed authentication "
            "(key mismatch or tampering)"
        ) from exc
    except RuntimeError as exc:
        raise MetaSchemaError(
            f"corrupt sidecar: {meta_file} — encrypted sidecar requires a master key to be set"
        ) from exc

    # --- schema_version gate (NIT 2) ---
    raw_version = raw.get("schema_version")
    if raw_version is None:
        raise MetaSchemaError(f"sidecar {meta_file} missing required field: schema_version")
    if not isinstance(raw_version, int):
        raise MetaSchemaError(f"sidecar {meta_file} schema_version is not an int: {raw_version!r}")
    if raw_version > SCHEMA_VERSION:
        logger.error(
            "Sidecar %s has forward schema_version %d (current: %d); "
            "this artifact was written by a newer release and cannot be read safely",
            meta_file,
            raw_version,
            SCHEMA_VERSION,
        )
        raise MetaSchemaError(
            f"forward version: sidecar schema_version={raw_version} "
            f"but current SCHEMA_VERSION={SCHEMA_VERSION}"
        )
    if raw_version < SCHEMA_VERSION:
        raise MetaSchemaError(
            f"legacy; migration required: sidecar schema_version={raw_version} "
            f"but current SCHEMA_VERSION={SCHEMA_VERSION}"
        )

    # --- required field presence ---
    required = {
        "kind",
        "timestamp",
        "content_sha256",
        "size_bytes",
        "encrypted",
        "tier",
    }
    missing = required - set(raw.keys())
    if missing:
        raise MetaSchemaError(f"sidecar {meta_file} missing required fields: {sorted(missing)}")

    # --- kind validation ---
    raw_kind = raw["kind"]
    try:
        kind = ArtifactKind(raw_kind)
    except ValueError:
        raise MetaSchemaError(
            f"sidecar {meta_file} has unknown kind: {raw_kind!r}; "
            f"valid kinds: {[k.value for k in ArtifactKind]}"
        )

    return ArtifactMeta(
        schema_version=raw_version,
        kind=kind,
        timestamp=raw["timestamp"],
        content_sha256=raw["content_sha256"],
        size_bytes=raw["size_bytes"],
        encrypted=raw["encrypted"],
        key_fingerprint=raw.get("key_fingerprint"),
        tier=raw["tier"],
        label=raw.get("label"),
        pre_trial_hash=raw.get("pre_trial_hash"),
    )


def verify_fingerprint(slot_dir: Path, artifact_path: Path) -> None:
    """Assert that *artifact_path* matches the content hash in the sidecar.

    Recomputes the SHA-256 of *artifact_path* (as raw bytes on disk) and
    compares it to ``meta.content_sha256`` from the slot's sidecar.

    Called by ``backup.read()`` *after* any decryption, passing the
    decrypted plaintext written to a temporary path, or *before* decryption
    against the raw on-disk bytes.

    Parameters
    ----------
    slot_dir:
        Slot directory containing the sidecar.
    artifact_path:
        Path to the artifact file whose content should be verified.

    Raises
    ------
    FingerprintMismatchError
        If the computed hash does not match the stored hash.
    MetaSchemaError
        If the sidecar cannot be read or validated.
    FileNotFoundError
        If *artifact_path* or the sidecar does not exist.
    """
    meta = read_meta(slot_dir)
    actual = content_sha256_path(artifact_path)
    if actual != meta.content_sha256:
        raise FingerprintMismatchError(
            f"content hash mismatch for {artifact_path.name}: "
            f"stored={meta.content_sha256!r}, actual={actual!r}"
        )
