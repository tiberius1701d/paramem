"""Rotation manifest + per-file re-encrypt helper for ``rotate-daily`` / ``rotate-recovery``.

Rotation walks ``infra_paths(data_dir)`` re-encrypting each age envelope to a
fresh recipient set and atomically swapping it in place. A crash mid-sweep
would otherwise leave the store in an ambiguous state where some files are
keyed to the new identity and some to the old; this module provides:

- :class:`RotationManifest` — on-disk JSON tracking operation, files pending,
  files completed, and metadata.
- :func:`write_manifest_atomic` — ``tmp -> fsync -> rename`` so the manifest
  is always consistent between crashes.
- :func:`rotate_file_to_recipients` — read an age envelope, decrypt with *any*
  of the supplied identities (robust to already-rotated files on resume),
  re-encrypt to the new recipient list, atomic-rename in place.

The manifest is stored next to the daily key, e.g.
``~/.config/paramem/rotation.manifest.json``. Payload:

.. code-block:: json

    {
      "operation": "rotate-daily",
      "started_at": "2026-04-24T11:30:00Z",
      "new_daily_pub": "age1…",
      "files_pending": ["/path/a", "/path/b"],
      "files_done": ["/path/c"]
    }

The manifest never contains secret material. The new daily secret lives
passphrase-wrapped at ``daily_key.age.pending``; the new recovery secret is
printed once and is never persisted. Both pending files are atomically
renamed over their canonical counterparts only after every infrastructure
file has been re-encrypted.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from pyrage import x25519

from paramem.backup.age_envelope import age_decrypt_bytes, age_encrypt_bytes
from paramem.backup.encryption import _atomic_write_bytes


@dataclass
class RotationManifest:
    """On-disk representation of an in-flight rotation.

    ``operation`` is ``"rotate-daily"`` or ``"rotate-recovery"`` so a resuming
    process can cross-check it is continuing the same work the prior run
    started. ``new_daily_pub`` (or ``new_recovery_pub``) pins the expected
    recipient of the output envelopes — mismatch means the pending key file
    was tampered with or swapped, and the operator must discard the rotation.
    """

    operation: str
    started_at: str
    new_daily_pub: str | None = None
    new_recovery_pub: str | None = None
    files_pending: list[str] = field(default_factory=list)
    files_done: list[str] = field(default_factory=list)

    @classmethod
    def fresh(
        cls,
        *,
        operation: str,
        files: Iterable[Path],
        new_daily_pub: str | None = None,
        new_recovery_pub: str | None = None,
    ) -> "RotationManifest":
        return cls(
            operation=operation,
            started_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            new_daily_pub=new_daily_pub,
            new_recovery_pub=new_recovery_pub,
            files_pending=[str(p) for p in files],
            files_done=[],
        )


def manifest_path_default() -> Path:
    """Default on-disk location for the rotation manifest.

    Lives next to the daily key — a rotation is always operator-initiated so
    the path follows ``DAILY_KEY_PATH_DEFAULT``'s parent directory.
    """
    from paramem.backup.key_store import DAILY_KEY_PATH_DEFAULT

    return DAILY_KEY_PATH_DEFAULT.parent / "rotation.manifest.json"


def read_manifest(path: Path) -> RotationManifest | None:
    """Return the parsed manifest at *path*, or ``None`` when it does not exist."""
    if not Path(path).is_file():
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return RotationManifest(**data)


def write_manifest_atomic(path: Path, manifest: RotationManifest) -> None:
    """Atomically write *manifest* to *path*.

    The manifest is not age-wrapped — it contains no secret material, only
    file lists + public recipients + ISO timestamps. Using
    :func:`_atomic_write_bytes` gives the same ``tmp -> fsync -> rename`` dance
    the infra writer uses, so the file is always parseable between crashes.
    """
    body = json.dumps(asdict(manifest), indent=2, sort_keys=True).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(path, body)


def delete_manifest(path: Path) -> None:
    """Best-effort manifest removal. Missing is not an error."""
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def rotate_file_to_recipients(
    path: Path,
    *,
    decrypt_identities: list[x25519.Identity],
    new_recipients: list[x25519.Recipient],
) -> bool:
    """Re-encrypt the age envelope at *path* to *new_recipients*.

    Tries each of *decrypt_identities* in order (pyrage internally picks the
    matching one), so resuming after a crash works regardless of whether the
    file still has the old recipient set or the rotation wrote the new set
    before the manifest update landed.

    Returns ``True`` when the file was re-encrypted (or no-op re-encrypt
    because the envelope already targeted the new recipients). Raises
    :class:`pyrage.DecryptError` if none of *decrypt_identities* match — in
    practice this means the operator's pre-rotation state is inconsistent
    with the rotation inputs (e.g. resuming with the wrong OLD daily).

    The write uses :func:`_atomic_write_bytes`, so a crash between the
    encrypt and rename leaves either the prior envelope at *path* or the new
    one, never a partial.
    """
    envelope = Path(path).read_bytes()
    plaintext = age_decrypt_bytes(envelope, decrypt_identities)
    _atomic_write_bytes(Path(path), age_encrypt_bytes(plaintext, new_recipients))
    return True


def finalise_pending_rename(pending: Path, canonical: Path) -> None:
    """Atomically swap *pending* over *canonical* and fsync the parent dir.

    Used at the end of a successful rotation to promote
    ``daily_key.age.pending`` to ``daily_key.age`` (or the symmetric
    ``recovery.pub.pending`` case). Both paths must live in the same
    directory so the rename is atomic at the POSIX level.
    """
    pending = Path(pending)
    canonical = Path(canonical)
    if pending.parent != canonical.parent:
        raise ValueError(
            f"pending and canonical paths must share a parent directory: {pending} vs {canonical}"
        )
    os.rename(pending, canonical)
    # fsync parent for rename durability (power-loss safety).
    try:
        dir_fd = os.open(str(canonical.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


__all__ = [
    "RotationManifest",
    "delete_manifest",
    "finalise_pending_rename",
    "manifest_path_default",
    "read_manifest",
    "rotate_file_to_recipients",
    "write_manifest_atomic",
]
