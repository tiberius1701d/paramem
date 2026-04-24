"""Per-file envelope encryption for HF Trainer checkpoint shards.

HF Trainer owns the atomic-save flow for ``bg_checkpoint/checkpoint-<step>/``
directories — ``adapter_model.safetensors``, ``optimizer.pt``, ``scheduler.pt``,
``trainer_state.json``, and siblings. The server-wide envelope writer does
not fit cleanly: HF's atomic-checkpoint code writes a temp directory then
renames, and ``resume_from_checkpoint`` reads files through HF-internal
loaders that expect native formats.

This module adapts the envelope to that constraint:

- :func:`encrypt_checkpoint_dir` is called from a ``TrainerCallback.on_save``
  hook. After HF finishes writing a checkpoint directory, every plaintext
  file in the tree is rewritten atomically through
  :func:`paramem.backup.encryption.write_infra_bytes` as an age envelope.
  Idempotent and no-op when Security is OFF.
- :func:`materialize_checkpoint_to_shm` is called before
  ``Trainer.train(resume_from_checkpoint=...)``. It copies the on-disk
  checkpoint into a ``/dev/shm``-backed tempdir, decrypting any age envelope
  via the universal reader en route. The tempdir path is handed to HF; the
  caller owns cleanup.

``/dev/shm`` is tmpfs (RAM-backed) so decrypted plaintext never lands on
persistent storage during a resume. When the mount is unavailable, a loud
WARN is logged and the system tempdir is used as a fallback — the resume
still works but briefly exposes plaintext to disk.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from paramem.backup.age_envelope import is_age_envelope
from paramem.backup.encryption import (
    read_maybe_encrypted,
    write_infra_bytes,
)


def _security_on() -> bool:
    """Return True when the daily age identity is loadable.

    Late-binds ``key_store`` attrs (no ``from … import`` at module top) so
    tests that monkeypatch ``paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT``
    see their override — ``from`` imports would freeze a stale Path reference
    at import time.
    """
    from paramem.backup import key_store as _ks

    return _ks.daily_identity_loadable(_ks.DAILY_KEY_PATH_DEFAULT)


def _is_encrypted_envelope(path: Path) -> bool:
    """Return True when *path* carries the age v1 magic."""
    return is_age_envelope(path)


logger = logging.getLogger(__name__)

_SHM_ROOT = Path("/dev/shm")
_SHM_PREFIX = "paramem-ckpt-"


def encrypt_checkpoint_dir(checkpoint_dir: Path) -> int:
    """Encrypt every plaintext file in *checkpoint_dir* in place.

    Walks the directory recursively; for each file not already age-wrapped,
    rewrites it via :func:`write_infra_bytes` (atomic temp+rename + age
    envelope). Idempotent. Returns the count of files encrypted on this call
    (zero when Security is OFF or every file was already wrapped).

    A partial failure (e.g. disk-full mid-iteration) leaves the directory in a
    mixed state — some files age-wrapped, others plaintext. The re-encrypt
    path is idempotent, and :func:`materialize_checkpoint_to_shm` tolerates
    mixed state on the read side.

    Parameters
    ----------
    checkpoint_dir:
        HF Trainer ``checkpoint-<step>/`` directory to encrypt.

    Returns
    -------
    int
        Number of files newly encrypted.
    """
    if not _security_on():
        return 0
    checkpoint_dir = Path(checkpoint_dir)
    encrypted = 0
    for entry in sorted(checkpoint_dir.rglob("*")):
        if not entry.is_file():
            continue
        # Skip files already in either envelope format — without the age
        # check, the D3-era writer would re-encrypt age files and produce
        # nested (double-wrapped) ciphertext.
        if _is_encrypted_envelope(entry):
            continue
        plaintext = entry.read_bytes()
        write_infra_bytes(entry, plaintext)
        encrypted += 1
    return encrypted


def _shm_tempdir() -> Path:
    """Return a fresh tempdir for materialized plaintext.

    Prefers ``/dev/shm`` (tmpfs, RAM-backed) so decrypted checkpoint plaintext
    never touches persistent storage during a resume. Falls back to the system
    tempdir with a WARN when ``/dev/shm`` is unavailable.
    """
    if _SHM_ROOT.is_dir():
        try:
            return Path(tempfile.mkdtemp(dir=str(_SHM_ROOT), prefix=_SHM_PREFIX))
        except OSError as exc:
            logger.warning("Could not create tempdir under /dev/shm (%s) — falling back", exc)
    logger.warning(
        "/dev/shm unavailable — decrypted checkpoint plaintext will land in "
        "the system tempdir, likely disk-backed. Host shutdown clears it, "
        "but a crash during training may leak plaintext to persistent storage."
    )
    return Path(tempfile.mkdtemp(prefix=_SHM_PREFIX))


def materialize_checkpoint_to_shm(checkpoint_dir: Path) -> Path:
    """Copy *checkpoint_dir* into a ``/dev/shm``-backed tempdir, decrypting en route.

    For each file under the source:

    - age-wrapped → decrypted via :func:`read_maybe_encrypted` and the
      plaintext is written into the tempdir at the same relative path.
    - Plaintext → byte-for-byte copy.

    The resulting tempdir has the same layout as the source, suitable for
    ``Trainer.train(resume_from_checkpoint=<tempdir>)``. Caller owns cleanup
    via ``shutil.rmtree(tempdir)`` in a ``finally`` block.

    Parameters
    ----------
    checkpoint_dir:
        On-disk checkpoint directory to materialize. Can be mixed-state.

    Returns
    -------
    Path
        Fresh tempdir containing plaintext files ready for HF Trainer load.
    """
    checkpoint_dir = Path(checkpoint_dir)
    tempdir = _shm_tempdir()
    try:
        for src in checkpoint_dir.rglob("*"):
            if src.is_dir():
                continue
            rel = src.relative_to(checkpoint_dir)
            dest = tempdir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Envelope-format-agnostic dispatch: age files
            # need decrypting; plaintext files byte-copy. Without the age
            # check, a post-D3 age envelope would fall through to the
            # copyfile branch and HF Trainer would load ciphertext.
            if _is_encrypted_envelope(src):
                dest.write_bytes(read_maybe_encrypted(src))
            else:
                shutil.copyfile(src, dest)
    except BaseException:
        shutil.rmtree(tempdir, ignore_errors=True)
        raise
    return tempdir
