"""Per-file PMEM1 envelope encryption for HF Trainer checkpoint shards.

HF Trainer owns the atomic-save flow for ``bg_checkpoint/checkpoint-<step>/``
directories — ``adapter_model.safetensors``, ``optimizer.pt``, ``scheduler.pt``,
``trainer_state.json``, and siblings. The PMEM1 envelope applied to arbitrary
infrastructure JSON does not fit cleanly: HF's atomic-checkpoint code writes a
temp directory then renames, and ``resume_from_checkpoint`` reads files through
HF-internal loaders that expect native formats.

This module adapts the envelope to that constraint:

- :func:`encrypt_checkpoint_dir` is called from a ``TrainerCallback.on_save``
  hook. After HF finishes writing a checkpoint directory, every plaintext file
  in the tree is rewritten atomically with the PMEM1 envelope. Idempotent and
  no-op when Security is OFF.
- :func:`materialize_checkpoint_to_shm` is called before
  ``Trainer.train(resume_from_checkpoint=...)``. It copies the on-disk
  checkpoint into a ``/dev/shm``-backed tempdir, decrypting PMEM1-wrapped
  files en route. The tempdir path is handed to HF; the caller owns cleanup.

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

from paramem.backup.encryption import (
    is_pmem1_envelope,
    master_key_loaded,
    read_maybe_encrypted,
    write_infra_bytes,
)

logger = logging.getLogger(__name__)

_SHM_ROOT = Path("/dev/shm")
_SHM_PREFIX = "paramem-ckpt-"


def encrypt_checkpoint_dir(checkpoint_dir: Path) -> int:
    """Encrypt every plaintext file in *checkpoint_dir* in place.

    Walks the directory recursively; for each file not already PMEM1-wrapped,
    rewrites it via :func:`write_infra_bytes` (atomic temp+rename + Fernet
    envelope). Idempotent. Returns the count of files encrypted on this call
    (zero when Security is OFF or every file was already wrapped).

    A partial failure (e.g. disk-full mid-iteration) leaves the directory in a
    mixed state — some files PMEM1, others plaintext. The re-encrypt path is
    idempotent, and :func:`materialize_checkpoint_to_shm` tolerates mixed state
    on the read side.

    Parameters
    ----------
    checkpoint_dir:
        HF Trainer ``checkpoint-<step>/`` directory to encrypt.

    Returns
    -------
    int
        Number of files newly encrypted.
    """
    if not master_key_loaded():
        return 0
    checkpoint_dir = Path(checkpoint_dir)
    encrypted = 0
    for entry in sorted(checkpoint_dir.rglob("*")):
        if not entry.is_file():
            continue
        if is_pmem1_envelope(entry):
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

    - PMEM1-wrapped → decrypted via :func:`read_maybe_encrypted` and the
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
            if is_pmem1_envelope(src):
                dest.write_bytes(read_maybe_encrypted(src))
            else:
                shutil.copyfile(src, dest)
    except BaseException:
        shutil.rmtree(tempdir, ignore_errors=True)
        raise
    return tempdir
