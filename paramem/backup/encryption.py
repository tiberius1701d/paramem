"""Encryption wrapper for the backup subsystem + infrastructure store.

Single on-disk envelope format:

- **age v1** — the two-identity envelope (``age-encryption.org/v1\\n``
  magic). Decrypted with the cached daily identity from
  :mod:`paramem.backup.key_store`. Primitives live in
  :mod:`paramem.backup.age_envelope`.

Services provided:

- Uniform AUTO semantics: :func:`envelope_encrypt_bytes` encrypts when
  the daily identity is loadable and returns plaintext otherwise; no
  per-artifact policy knob exists. Operators opt into a fail-loud
  posture via the single uniform ``security.require_encryption`` flag
  enforced at server startup by
  :func:`paramem.server.security_posture.assert_startup_posture`.
- Mode-mismatch startup refuse (:func:`assert_mode_consistency`):
  classifies infrastructure files as age / plaintext and refuses any
  combination that would be silently unreadable or that mixes
  plaintext with age envelopes.
- The universal read path :func:`read_maybe_encrypted` dispatches by
  envelope magic — age ciphertext unwraps with the cached daily
  identity, non-magic bytes pass through as plaintext.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from paramem.backup.age_envelope import (
    AGE_MAGIC,
    age_decrypt_bytes,
    age_encrypt_bytes,
    is_age_envelope,
)
from paramem.backup.types import FatalConfigError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Universal read path
# ---------------------------------------------------------------------------


def read_maybe_encrypted(path: Path) -> bytes:
    """Return plaintext bytes from *path*, dispatching by envelope magic.

    The universal read path for any infrastructure file that may have been
    written via :func:`write_infra_bytes`. Two on-disk shapes are handled
    transparently so callers never branch on key-loaded state:

    - **age v1** envelope (``age-encryption.org/v1\\n`` magic) — decrypted
      with the cached daily identity loaded from
      :func:`paramem.backup.key_store.load_daily_identity_cached`. The
      identity is unwrapped once on first read via the scrypt KDF and
      cached module-side; rotation handlers call
      :func:`paramem.backup.key_store._clear_daily_identity_cache`.
    - No recognised magic — returned verbatim (plaintext pass-through).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    RuntimeError
        If the file carries an age magic but the daily identity is not
        loaded — actionable message names the env var the operator needs
        to set.
    pyrage.DecryptError
        age ciphertext is corrupt, tampered with, or cannot be decrypted by
        the loaded daily identity (neither daily nor recovery recipient
        match).
    """
    raw = Path(path).read_bytes()
    if raw.startswith(AGE_MAGIC):
        # Late-lookup the key_store module so tests (and a future operator
        # override) can monkeypatch DAILY_KEY_PATH_DEFAULT without having the
        # value frozen into this function's defaults at import time.
        from paramem.backup import key_store as _ks

        try:
            identity = _ks.load_daily_identity_cached(_ks.DAILY_KEY_PATH_DEFAULT)
        except RuntimeError as exc:
            raise RuntimeError(
                f"{path} is an age envelope but the daily identity is not loaded: "
                f"{exc}. Set {_ks.DAILY_PASSPHRASE_ENV_VAR} and ensure "
                f"{_ks.DAILY_KEY_PATH_DEFAULT} exists."
            ) from exc
        return age_decrypt_bytes(raw, [identity])
    return raw


def _atomic_write_bytes(path: Path, body: bytes) -> None:
    """Shared atomic-write core: ``<path>.tmp`` → fsync → rename → fsync parent.

    On any failure before the rename completes, the temp file is removed so
    no partial content is left on disk.  Callers that need to make an
    encryption decision layer it on top of this helper.
    """
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with open(tmp_path, "wb") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        os.rename(tmp_path, path)
    except Exception:
        # Clean up the temp file if anything went wrong before/at rename.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    # fsync the parent directory so the rename is durable across crashes.
    parent = path.parent
    try:
        dir_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        except OSError as exc:
            logger.warning("_atomic_write_bytes: parent dir fsync failed: %s", exc)
        finally:
            os.close(dir_fd)
    except OSError as exc:
        logger.warning("_atomic_write_bytes: could not open parent for fsync: %s", exc)


def envelope_encrypt_bytes(plaintext: bytes) -> bytes:
    """Return encrypted envelope bytes based on the loaded-key posture.

    Priority (highest first):

    1. **age multi-recipient** ``[daily, recovery]`` — when the daily
       identity is loadable AND ``recovery.pub`` is on disk.
    2. **age single-recipient** ``[daily]`` — daily loadable but
       ``recovery.pub`` missing. Degraded mode; the startup log warns.
    3. **Plaintext** — no key material loaded; caller should gate on this
       case explicitly if they require encryption.

    Always returns a magic-prefixed age envelope when encryption happens,
    or raw plaintext bytes when no key is loaded. Used by both
    :func:`write_infra_bytes` (writes to disk) and the backup subsystem
    (holds the bytes in hand for sidecar construction).
    """
    from paramem.backup import key_store as _ks

    if _ks.daily_identity_loadable(_ks.DAILY_KEY_PATH_DEFAULT):
        try:
            daily = _ks.load_daily_identity_cached(_ks.DAILY_KEY_PATH_DEFAULT)
        except Exception:  # noqa: BLE001 — graceful fallback on any unwrap failure
            daily = None
        if daily is not None:
            recipients = [daily.to_public()]
            if _ks.recovery_pub_available(_ks.RECOVERY_PUB_PATH_DEFAULT):
                recipients.append(_ks.load_recovery_recipient(_ks.RECOVERY_PUB_PATH_DEFAULT))
            return age_encrypt_bytes(plaintext, recipients)
    return plaintext


def envelope_decrypt_bytes(raw: bytes) -> bytes:
    """Return plaintext from age envelope bytes.

    Unlike :func:`read_maybe_encrypted`, this does NOT treat non-magic
    bytes as plaintext — the caller is expected to know the bytes are
    encrypted (e.g. via a sidecar ``meta.encrypted`` field).
    Pass-through plaintext is the :func:`read_maybe_encrypted` behaviour.

    Raises
    ------
    RuntimeError
        When the daily identity is not loaded.
    pyrage.DecryptError
        When the ciphertext cannot be decrypted by the loaded identity.
    """
    if not raw.startswith(AGE_MAGIC):
        raise RuntimeError(
            "envelope_decrypt_bytes expected an age envelope but the bytes "
            "do not carry the age magic prefix"
        )
    from paramem.backup import key_store as _ks

    try:
        identity = _ks.load_daily_identity_cached(_ks.DAILY_KEY_PATH_DEFAULT)
    except RuntimeError as exc:
        raise RuntimeError(
            f"age envelope encountered but the daily identity is not loaded: {exc}. "
            f"Set {_ks.DAILY_PASSPHRASE_ENV_VAR} and ensure "
            f"{_ks.DAILY_KEY_PATH_DEFAULT} exists."
        ) from exc
    return age_decrypt_bytes(raw, [identity])


def write_infra_bytes(path: Path, plaintext: bytes) -> None:
    """Atomically write *plaintext* to *path*, encrypting when a key is loaded.

    Delegates format selection to :func:`envelope_encrypt_bytes` — age
    multi-recipient when ``recovery.pub`` is present, age single-recipient
    otherwise, plaintext when no key is loaded. The universal reader
    :func:`read_maybe_encrypted` unwraps either shape.

    Parameters
    ----------
    path:
        Destination path.  Parent directory must exist.
    plaintext:
        Raw content to write.

    Raises
    ------
    OSError
        On any filesystem error.
    """
    _atomic_write_bytes(Path(path), envelope_encrypt_bytes(plaintext))


def write_plaintext_atomic(path: Path, plaintext: bytes) -> None:
    """Atomically write *plaintext* to *path*, never encrypting.

    Escape hatch for operators intentionally writing plaintext (e.g. the
    ``paramem dump`` redirect pattern or a one-off debug dump). Normal
    infrastructure writers must use :func:`write_infra_bytes`.
    """
    _atomic_write_bytes(Path(path), plaintext)


# ---------------------------------------------------------------------------
# Mode-mismatch startup refuse (SECURITY.md §4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeProbe:
    """Result of scanning the data directory for encryption-state evidence.

    Attributes
    ----------
    age_paths:
        Files carrying the age v1 envelope magic. Decryptable with the
        loaded daily identity.
    plaintext_paths:
        Files that carry no encryption magic but fall under the §4
        encrypted-infrastructure list.
    """

    age_paths: list[Path] = field(default_factory=list)
    plaintext_paths: list[Path] = field(default_factory=list)


def infra_paths(data_dir: Path, simulate_dir: Path | None = None) -> list[Path]:
    """Return the list of infrastructure files subject to envelope encryption.

    Single source of truth for the startup mode-consistency scan
    (:func:`_probe_data_dir`).

    Paths that do not currently exist on disk are still returned — the
    caller filters as needed. This keeps the "what counts as infra
    metadata" definition in one place regardless of whether the operator
    has populated every path yet.

    Excluded (plaintext-by-design per SECURITY.md §4 carve-out):
    - ``state/trial.json`` and ``state/backup.json`` — control-plane only.
    - Backup ``*.meta.json`` sidecars — operator visibility on wrong-key
      restore trumps the marginal info hiding.

    Parameters
    ----------
    data_dir:
        Root of the ParaMem data directory (typically
        ``configs/server.yaml``'s ``paths.data``).
    simulate_dir:
        Optional path to the simulate-mode peer-storage root
        (``configs/server.yaml``'s ``paths.simulate``). When provided,
        the per-tier ``keyed_pairs.json`` files under it are appended
        to the candidate set so rotation, restore, and the startup
        mode-consistency scan all cover the simulate store. Callers
        that do not have a config (e.g. legacy callers) may pass
        ``None`` to keep the historical data-dir-only behaviour.

    Returns
    -------
    list[Path]
        Ordered list of candidate paths.  Neither filtered by existence
        nor classified by on-disk state.
    """
    data_dir = Path(data_dir)
    paths: list[Path] = [
        data_dir / "graph.json",
        data_dir / "registry.json",
        data_dir / "indexed_key_registry.json",
        data_dir / "registry" / "key_metadata.json",
        data_dir / "speaker_profiles.json",
        data_dir / "adapters" / "post_session_queue.json",
        data_dir / "adapters" / "keyed_pairs.json",
        data_dir / "adapters" / "episodic" / "keyed_pairs.json",
        data_dir / "adapters" / "semantic" / "keyed_pairs.json",
        data_dir / "adapters" / "procedural" / "keyed_pairs.json",
    ]
    # BG-trainer resume states live under per-job in_training directories.
    adapters_root = data_dir / "adapters"
    if adapters_root.exists():
        for resume in adapters_root.rglob("in_training/resume_state.json"):
            paths.append(resume)
    # Simulate-mode peer-storage keyed_pairs (canonical per-tier layout).
    # Encryption posture matches train: respects the master switch via the
    # same encrypted helpers. Inclusion here ensures rotation/restore/scan
    # cover the simulate store as a first-class production artifact.
    if simulate_dir is not None:
        simulate_dir = Path(simulate_dir)
        paths.extend(
            [
                simulate_dir / "episodic" / "keyed_pairs.json",
                simulate_dir / "semantic" / "keyed_pairs.json",
                simulate_dir / "procedural" / "keyed_pairs.json",
            ]
        )
    return paths


def _probe_data_dir(data_dir: Path, simulate_dir: Path | None = None) -> ModeProbe:
    """Scan *data_dir* (and *simulate_dir* if provided) for infrastructure
    files and classify each as age / plaintext.

    Uses ``infra_paths`` as the authoritative candidate set.  Missing files
    do NOT contribute to either classification — only files that actually
    exist on disk steer the mode verdict.
    """
    probe = ModeProbe()
    data_dir = Path(data_dir)
    # data_dir is the primary root; simulate_dir is optionally a sibling root.
    # Either may be missing on first start; that's fine — the candidate set
    # is filtered by existence below.
    if not data_dir.exists() and (simulate_dir is None or not Path(simulate_dir).exists()):
        return probe

    for path in infra_paths(data_dir, simulate_dir=simulate_dir):
        if not path.exists() or not path.is_file():
            continue
        if is_age_envelope(path):
            probe.age_paths.append(path)
        else:
            probe.plaintext_paths.append(path)

    return probe


def assert_mode_consistency(
    data_dir: Path,
    *,
    daily_identity_loadable: bool = False,
    simulate_dir: Path | None = None,
) -> None:
    """Refuse startup when on-disk encryption state conflicts with the loaded keys.

    Acceptable combinations:

    - Empty store                               → OK regardless of keys.
    - Plaintext only, no daily identity         → OK (Security OFF).
    - age only, daily identity loadable         → OK.

    Refusal cases:

    - Plaintext alongside age envelopes         → refuse (mixed state).
    - Plaintext present while the daily identity is loaded → operator has
      a key but the store is unencrypted; refuse before writing anything.
    - age present without the daily identity    → unreadable; set
      ``PARAMEM_DAILY_PASSPHRASE`` and ensure the daily key file exists.

    Parameters
    ----------
    data_dir:
        Root of the data directory.
    daily_identity_loadable:
        Whether the daily age identity is loadable (passphrase env var set
        + daily key file exists). Does not force an unwrap; a stale
        passphrase surfaces on first actual read.

    Raises
    ------
    FatalConfigError
        On any refusal case above, with an operator-actionable message.
    """
    from paramem.backup.key_store import DAILY_PASSPHRASE_ENV_VAR

    probe = _probe_data_dir(Path(data_dir), simulate_dir=simulate_dir)

    has_age = bool(probe.age_paths)
    has_pt = bool(probe.plaintext_paths)

    # Mixing plaintext with age envelopes is a fatal mismatch regardless of
    # which keys are loaded — the store is inconsistent with itself.
    if has_pt and has_age:
        sample_enc = probe.age_paths[0]
        sample_pt = probe.plaintext_paths[0]
        raise FatalConfigError(
            "Mixed encryption state on disk: "
            f"{sample_enc} is age-encrypted but {sample_pt} is plaintext "
            f"({len(probe.age_paths)} age, {len(probe.plaintext_paths)} "
            "plaintext in total). Reconcile the store before startup."
        )

    # Plaintext files while the daily identity is loaded → encryption enabled
    # but the store is not migrated. Refuse before writing anything.
    if has_pt and daily_identity_loadable:
        raise FatalConfigError(
            f"The daily identity is loaded but {len(probe.plaintext_paths)} "
            f"infrastructure file(s) on disk are plaintext "
            f"(e.g. {probe.plaintext_paths[0]}). Migrate the store to age "
            "before startup, or unset the daily passphrase to run in the "
            "Security-OFF posture."
        )

    # age files present without the daily identity → unreadable.
    if has_age and not daily_identity_loadable:
        raise FatalConfigError(
            f"{len(probe.age_paths)} infrastructure file(s) on disk are age-"
            f"encrypted (e.g. {probe.age_paths[0]}) but the daily identity "
            f"is not loadable. Set {DAILY_PASSPHRASE_ENV_VAR} and ensure "
            "~/.config/paramem/daily_key.age exists before startup."
        )

    # Otherwise the store is consistent with the loaded keys — proceed.
    return
