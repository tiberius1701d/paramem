"""The slot promotion envelope — one write sequence for both venues.

A consolidation tier's payload — LoRA weights (``payload.kind == "train"``)
or a projected knowledge graph (``payload.kind == "simulate"``) — is written
the same way regardless of which one it is: into a fresh
``<tier_root>/.pending/<ts>/`` staging directory, the payload bytes are
written there by a caller-supplied closure, the plaintext SHA-256 of the
payload file is computed and stamped into the manifest, ``meta.json`` is
written, and the whole directory is fsync'd and atomically renamed to
``<tier_root>/<ts>/``.

This module owns the slot's *lifecycle* — naming, promotion, the
payload-file-per-kind rule — not the manifest's schema (owned by
:mod:`paramem.adapters.manifest`) or registry-binding verification (owned
by :mod:`paramem.adapters.registry_binding`). Splitting it out here is what
lets :mod:`paramem.models.loader` (the weight-payload writer) and
:mod:`paramem.memory.persistence` (both venues) share one promotion
sequence instead of two.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Final

from paramem.backup.hashing import plaintext_sha256

if TYPE_CHECKING:
    from paramem.adapters.manifest import AdapterManifest

_PAYLOAD_FILENAME: Final[dict[str, str]] = {
    "train": "adapter_model.safetensors",
    "simulate": "graph.json",
}

_REQUIRED_SLOT_FILES: Final[dict[str, tuple[str, ...]]] = {
    "train": ("meta.json", "adapter_config.json", "adapter_model.safetensors"),
    "simulate": ("meta.json", "graph.json"),
}


def payload_filename(kind: str) -> str:
    """Return the one payload filename for *kind*.

    The single filename rule for the whole package — the write envelope, the
    publish preflight, ``cleanup_partial_slots`` and the boot payload check
    all consume this rather than re-deriving it.

    Args:
        kind: ``"train"`` or ``"simulate"`` — matches
            :attr:`~paramem.adapters.manifest.PayloadFingerprint.kind`.

    Returns:
        ``"adapter_model.safetensors"`` for ``"train"``, ``"graph.json"``
        for ``"simulate"``.

    Raises:
        ValueError: *kind* is neither ``"train"`` nor ``"simulate"``.
    """
    try:
        return _PAYLOAD_FILENAME[kind]
    except KeyError:
        raise ValueError(f"payload_filename: unknown payload kind {kind!r}") from None


def required_slot_files(kind: str) -> tuple[str, ...]:
    """Return the complete-slot file set for *kind*.

    A subdirectory missing any of these is partial-trained scratch, not a
    committed slot — ``adapter_config.json`` is shape metadata (already
    fingerprinted field-by-field in the manifest's ``lora``) and is part of
    the ``"train"`` set only; it has no ``"simulate"`` counterpart.

    Args:
        kind: ``"train"`` or ``"simulate"``.

    Returns:
        The filenames a complete slot of that kind must all carry.

    Raises:
        ValueError: *kind* is neither ``"train"`` nor ``"simulate"``.
    """
    try:
        return _REQUIRED_SLOT_FILES[kind]
    except KeyError:
        raise ValueError(f"required_slot_files: unknown payload kind {kind!r}") from None


def make_slot_ts() -> str:
    """Return a ``YYYYMMDD-HHMMSS`` UTC timestamp string for slot naming."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def write_slot(
    tier_root: Path,
    *,
    manifest: "AdapterManifest | None",
    write_payload: "Callable[[Path], None]",
) -> Path:
    """Write one payload into a fresh slot and promote it atomically.

    The one promotion sequence both venues share:

    1. Ensure *tier_root* and ``tier_root/.pending/`` exist; pick a
       collision-safe ``<ts>`` stamp (retried against both the pending path
       and an already-promoted slot of the same stamp) and create
       ``pending_slot = tier_root/.pending/<ts>/``.
    2. Call ``write_payload(pending_slot)`` — the caller's own payload
       write (PEFT's ``save_pretrained`` plus flatten plus in-place
       encryption for a weight payload; a plain ``graph.json`` write for a
       graph payload). This function has no opinion on *how* the payload
       gets written, only on what happens before and after.
    3. When *manifest* is not ``None``: compute the plaintext SHA-256 of
       ``pending_slot / payload_filename(manifest.payload.kind)`` (via
       :func:`~paramem.backup.hashing.plaintext_sha256` — the payload may
       already be age-encrypted by step 2), substitute that digest into
       ``manifest.payload.sha256``, and write the stamped manifest to
       ``pending_slot/meta.json``.
    4. fsync ``pending_slot`` (best-effort; ``OSError`` tolerated on
       filesystems that don't support directory fsync).
    5. Atomic rename ``pending_slot -> tier_root/<ts>/`` (the final slot).
       fsync ``tier_root`` for durability.

    ``manifest=None`` skips steps 3's digest computation and manifest write
    entirely — the debug weight-shadow shape
    (:func:`~paramem.utils.artifacts.on_main_adapters_saved`), which never
    lives under a tier root and is never a slot candidate.

    Args:
        tier_root: Adapter-kind or tier-root directory.  Slots are created
            as ``tier_root/<ts>/``.
        manifest: The manifest to stamp with the payload digest and write
            into the slot, or ``None`` to write no manifest.
        write_payload: Called with the pending slot directory; must write
            the slot's payload file (and, for a weight payload, its sibling
            ``adapter_config.json``) into it before returning.

    Returns:
        Path to the final (promoted) slot directory.
    """
    tier_root = Path(tier_root)
    tier_root.mkdir(parents=True, exist_ok=True)

    pending_root = tier_root / ".pending"
    pending_root.mkdir(exist_ok=True)

    ts = make_slot_ts()
    pending_slot = pending_root / ts
    for _attempt in range(10):
        # Both paths must be clear: pending_slot (mid-write) AND
        # tier_root/ts (the FINAL promoted slot this same ts already named
        # — a prior write's pending_slot can be renamed away by the time this
        # check runs, so pending_slot.exists() alone misses a same-second
        # collision against an already-promoted slot, and the rename below
        # would OSError on the existing destination).
        if not pending_slot.exists() and not (tier_root / ts).exists():
            break
        time.sleep(1)
        ts = make_slot_ts()
        pending_slot = pending_root / ts
    pending_slot.mkdir(parents=True, exist_ok=False)

    write_payload(pending_slot)

    if manifest is not None:
        from paramem.adapters.manifest import write_manifest

        payload_path = pending_slot / payload_filename(manifest.payload.kind)
        digest = plaintext_sha256(payload_path)
        stamped_manifest = replace(manifest, payload=replace(manifest.payload, sha256=digest))
        write_manifest(pending_slot, stamped_manifest)

    # fsync pending_slot
    try:
        fd = os.open(str(pending_slot), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass

    # Atomic promotion: .pending/<ts> -> tier_root/<ts>
    final_slot = tier_root / ts
    pending_slot.rename(final_slot)
    try:
        fd = os.open(str(tier_root), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass

    return final_slot
