"""Encryption wrapper for the backup subsystem + infrastructure store.

Single on-disk envelope format:

- **age v1** — the two-identity envelope (``age-encryption.org/v1\\n``
  magic). Decrypted with the cached daily identity from
  :mod:`paramem.backup.key_store`. Primitives live in
  :mod:`paramem.backup.age_envelope`.

Services provided:

- Uniform AUTO semantics: :func:`envelope_encrypt_bytes` encrypts when a
  daily identity is available, returns plaintext when no key is
  configured at all, and raises when a configured key cannot be
  unwrapped — an unwrap failure can never silently fall back to
  plaintext. No per-artifact policy knob exists. Operators opt into a
  fail-loud posture at boot via the single uniform
  ``security.require_encryption`` flag enforced at server startup by
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

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import pyrage

from paramem.backup.age_envelope import (
    AGE_MAGIC,
    age_decrypt_bytes,
    age_encrypt_bytes,
    is_age_envelope,
)
from paramem.backup.types import FatalConfigError
from paramem.training.stage_ledger import data_state_dir

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
       identity is available AND ``recovery.pub`` is on disk.
    2. **age single-recipient** ``[daily]`` — daily available but
       ``recovery.pub`` missing. Degraded mode; the startup log warns.
    3. **Plaintext** — no key material configured at all; caller should
       gate on this case explicitly if they require encryption.

    Contract: no key configured → plaintext (the AUTO opt-out).
    A key that IS configured but cannot be unwrapped (wrong passphrase,
    corrupt or tampered envelope, key file vanished after the
    availability check) raises instead of degrading to plaintext — an
    unwrap failure can never silently produce a plaintext write.

    Always returns a magic-prefixed age envelope when encryption
    happens, or raw plaintext bytes when no key is configured. Used by
    both :func:`write_infra_bytes` (writes to disk) and the backup
    subsystem (holds the bytes in hand for sidecar construction).

    Raises
    ------
    RuntimeError
        A daily identity is configured (:func:`paramem.backup.key_store.daily_identity_available`
        is True) but could not be unwrapped.
    """
    from paramem.backup import key_store as _ks

    if not _ks.daily_identity_available(_ks.DAILY_KEY_PATH_DEFAULT):
        return plaintext
    try:
        daily = _ks.load_daily_identity_cached(_ks.DAILY_KEY_PATH_DEFAULT)
    except (RuntimeError, pyrage.DecryptError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"a daily encryption key is configured but could not be unwrapped: "
            f"{exc}. Refusing to write plaintext. Confirm "
            f"{_ks.DAILY_PASSPHRASE_ENV_VAR} matches the passphrase used to "
            f"create {_ks.DAILY_KEY_PATH_DEFAULT} and that the file is not "
            f"corrupt or tampered with."
        ) from exc
    recipients = [daily.to_public()]
    if _ks.recovery_pub_available(_ks.RECOVERY_PUB_PATH_DEFAULT):
        recipients.append(_ks.load_recovery_recipient(_ks.RECOVERY_PUB_PATH_DEFAULT))
    return age_encrypt_bytes(plaintext, recipients)


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
    """Atomically write *plaintext* to *path*, encrypting when a key is configured.

    Delegates format selection to :func:`envelope_encrypt_bytes` — age
    multi-recipient when ``recovery.pub`` is present, age single-recipient
    otherwise, plaintext when no key is configured. The universal reader
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
    RuntimeError
        A daily identity is configured but could not be unwrapped (see
        :func:`envelope_encrypt_bytes`) — no plaintext is written in this
        case.
    """
    _atomic_write_bytes(Path(path), envelope_encrypt_bytes(plaintext))


def write_infra_json(path: Path, data: dict | list) -> None:
    """Atomically write *data* as indented JSON to *path* via :func:`write_infra_bytes`.

    The one chokepoint for JSON-shaped infrastructure files, so every one of
    them respects the operator's ``security.require_encryption`` posture
    identically — age-encrypted when a daily identity is configured,
    plaintext when none is, and raising rather than degrading when a
    configured identity cannot be unwrapped.  Creates the parent directory
    (:func:`write_infra_bytes` requires it to already exist) and serializes *data* with
    ``json.dumps(..., indent=2)`` before handing the bytes off.  Inspection
    output is not written here: that is an artifact, and goes through
    :func:`paramem.utils.artifacts.write_artifact`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_infra_bytes(path, json.dumps(data, indent=2).encode("utf-8"))


def write_plaintext_atomic(path: Path, plaintext: bytes) -> None:
    """Atomically write *plaintext* to *path*, never encrypting.

    Escape hatch for operators intentionally writing plaintext (e.g. the
    ``paramem dump`` redirect pattern or a one-off debug dump). Normal
    infrastructure writers must use :func:`write_infra_bytes`.
    """
    _atomic_write_bytes(Path(path), plaintext)


# ---------------------------------------------------------------------------
# Mode-mismatch startup refuse
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
        Files that carry no encryption magic but are listed in the
        encrypted-infrastructure set returned by :func:`infra_paths`.
    """

    age_paths: list[Path] = field(default_factory=list)
    plaintext_paths: list[Path] = field(default_factory=list)


def _under_hidden_dir(path: Path, root: Path) -> bool:
    """True when any path segment between *root* and *path* is dot-prefixed.

    ``rglob`` descends into every subdirectory unconditionally, including
    dot-prefixed ones (e.g. ``.pending-delete/`` — the reap tombstone
    directory, :mod:`paramem.memory.persistence`) that condemned-but-not-yet-
    deleted debris can transiently sit in. Mirrors the dot-entry skip already
    applied at every ``iterdir()`` walker in this codebase (e.g.
    ``paramem.backup.backup``'s "skip .pending and other hidden entries").

    Args:
        path: Candidate file path, expected to be under *root*.
        root: Root directory the ``rglob`` walk started from.

    Returns:
        ``True`` if *path* sits under a dot-prefixed directory relative to
        *root*; ``False`` otherwise (including when *path* is not under
        *root* at all).
    """
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        return False
    return any(part.startswith(".") for part in relative_parts[:-1])


# The exact per-tier shadow inventory `ConsolidationLoop.stage_event` writes
# under `extraction/<event>/shadow/<tier>/` (`_write_shadow_tier`) — filename-
# scoped so `_extraction_shadow_paths` never has to glob "*" and pick up a
# crash-orphaned `<name>.tmp` staging file left beside one of these three
# (atomic writes stage `.tmp` beside the target) and misclassify it as
# plaintext, refusing the next boot.
_EXTRACTION_SHADOW_FILENAMES: tuple[str, ...] = (
    "key_metadata.json",
    "keyed.json",
    "indexed_key_registry.json",
)


_TIER_SLOT_FILENAMES: tuple[str, ...] = (
    "indexed_key_registry.json",
    "key_metadata.json",
    "graph.json",
)


def _tier_slot_paths(tier_root: Path) -> list[Path]:
    """Return one tier's main-slot file paths plus every nested match.

    The per-tier counterpart of :func:`_extraction_shadow_paths`:
    :data:`_TIER_SLOT_FILENAMES` listed unconditionally at the tier root
    (``indexed_key_registry.json`` and ``key_metadata.json`` live there in
    either venue; ``graph.json`` never does — it lives inside a timestamped
    slot, same as a train payload) plus a recursive ``rglob`` for all
    three filenames — this is what actually finds ``graph.json`` and any
    other nested match, wherever it lives: inside a main-tier timestamped
    slot, inside an ``interim_<stamp>/`` family's own timestamped slot, or
    (legacy) directly under an ``interim_<stamp>/`` root — excluding a
    dot-prefixed directory (:func:`_under_hidden_dir`, which also skips a
    slot's own ``.pending/`` scratch). Called once per tier per tree (live,
    trial) by :func:`infra_paths`, so the walk itself is written once rather
    than repeated per tier per tree.

    Args:
        tier_root: A tier's root directory, e.g.
            ``<data_dir>/adapters/episodic`` (live) or
            ``<data_dir>/state/trial/adapters/episodic`` (trial).

    Returns:
        The three tier-root candidate paths (listed unconditionally, not
        filtered by existence — ``tier_root / "graph.json"`` never resolves
        on disk under the current layout, but a stale caller filtering by
        existence tolerates that harmlessly) followed by every nested match
        found on disk, wherever it actually lives.
    """
    paths: list[Path] = [tier_root / fname for fname in _TIER_SLOT_FILENAMES]
    if tier_root.exists():
        for fname in _TIER_SLOT_FILENAMES:
            for path in tier_root.rglob(fname):
                if path != tier_root / fname and not _under_hidden_dir(path, tier_root):
                    paths.append(path)
    return paths


def _extraction_shadow_paths(root: Path) -> list[Path]:
    """Return the stage-ledger record plus its per-event shadow inventory for *root*.

    *root* is either the live data dir or the trial tree's own root
    (the parent of its adapters dir) — :func:`~paramem.training.stage_ledger.data_state_dir`
    derives ``<root>/state`` for either, so this one helper serves both the
    live and trial extraction trees; :func:`infra_paths` calls it once per
    tree rather than repeating the walk.

    Args:
        root: Data-tree root whose ``state/extraction/`` tree is walked.

    Returns:
        ``[<root>/state/stage_ledger.json, *shadow files]`` — the shadow
        files are every :data:`_EXTRACTION_SHADOW_FILENAMES` match under
        ``<root>/state/extraction/``, excluding any match under a
        dot-prefixed directory (:func:`_under_hidden_dir`). Neither filtered
        by existence beyond the ``extraction/`` root itself.
    """
    state_dir = data_state_dir(root)
    found: list[Path] = [state_dir / "stage_ledger.json"]
    extraction_root = state_dir / "extraction"
    if extraction_root.exists():
        for fname in _EXTRACTION_SHADOW_FILENAMES:
            for path in extraction_root.rglob(fname):
                if not _under_hidden_dir(path, extraction_root):
                    found.append(path)
    return found


def infra_paths(data_dir: Path) -> list[Path]:
    """Return the list of infrastructure files subject to envelope encryption.

    Single source of truth for the startup mode-consistency scan
    (:func:`_probe_data_dir`).

    Paths that do not currently exist on disk are still returned — the
    caller filters as needed. This keeps the "what counts as infra
    metadata" definition in one place regardless of whether the operator
    has populated every path yet.

    Excluded (plaintext-by-design carve-out):
    - ``state/trial.json`` and ``state/backup.json`` — control-plane only.
    - Backup ``*.meta.json`` sidecars — operator visibility on wrong-key
      restore trumps the marginal info hiding.

    Included (full-file encryption):
    - ``user_tokens.json`` — per-user bearer-token store (SHA-256-hashed
      credentials; must never exist in plaintext).
    - ``adapters/<tier>/<slot>/graph.json`` — the simulate venue's projected
      payload, written inside a timestamped slot by ``write_tier_slot`` on the
      fold path (main tiers and interim slots alike); ``commit_tier_slot``
      writes it only on the base-swap migration and trial-tree main-tier
      copy paths — through the shared slot envelope
      (:func:`~paramem.adapters.slot.write_slot`) exactly like a train
      payload — never a tier-root file.
    - ``adapters/<tier>/key_metadata.json`` — per-tier bookkeeping rows,
      written by ``publish_tier_registry`` beside the tier's own
      ``indexed_key_registry.json`` on every fold (main tiers and interim
      slots alike); ``commit_tier_slot`` writes the same pair only on the
      base-swap migration and trial-tree main-tier copy paths.
    - ``adapters/<tier>/<slot>/adapter_model.safetensors`` (and any
      ``episodic/interim_*`` siblings) — LoRA weight tensors encrypted
      in-place by :func:`~paramem.models.loader._encrypt_adapter_safetensors`
      at save time; decrypted into anonymous RAM at load time via
      :func:`~paramem.models.loader._adapter_slot_for_load`.
    - ``state/trial/adapters/<kind>/{indexed_key_registry,key_metadata,graph}.json``,
      including its own ``interim_<stamp>/`` slots — the trial tree's
      counterparts of the live per-tier files. The main-tier copies are
      written by ``commit_tier_slot`` (``commit_main_tiers``'s copy-forward
      into the trial loop's ``output_dir``); the trial fold's own interim
      slots are written by the same ``write_tier_slot`` /
      ``publish_tier_registry`` path as the live tree.
    - ``state/stage_ledger.json`` — the two-phase training event's progress
      record (:mod:`paramem.training.stage_ledger`) — plus, filename-scoped,
      every ``key_metadata.json`` / ``keyed.json`` / ``indexed_key_registry.json``
      under ``state/extraction/`` (the per-event shadow tree
      ``ConsolidationLoop.stage_event`` writes per tier it builds; see
      ``_EXTRACTION_SHADOW_FILENAMES``), plus their trial-tree counterparts
      at ``state/trial/state/stage_ledger.json`` and the same three
      filenames under ``state/trial/state/extraction/`` (the trial loop's
      own ``output_dir`` is ``state/trial/adapters``, so its own fold state
      dir — the same :func:`~paramem.training.stage_ledger.data_state_dir`
      formula every other caller uses, applied to the trial root — is
      ``state/trial/state``).

    Parameters
    ----------
    data_dir:
        Root of the ParaMem data directory (typically
        ``configs/server.yaml``'s ``paths.data``).

    Returns
    -------
    list[Path]
        Ordered list of candidate paths.  Neither filtered by existence
        nor classified by on-disk state.
    """
    data_dir = Path(data_dir)
    paths: list[Path] = [
        data_dir / "registry.json",
        data_dir / "indexed_key_registry.json",
        data_dir / "speaker_profiles.json",
        data_dir / "user_tokens.json",
        # Web Push Tier-2 infra files — same posture as user_tokens.json.
        # vapid_keys.json holds the VAPID EC private key PEM; effectively
        # immutable once browsers have subscribed (rotation invalidates all
        # subscriptions).  push_subscriptions.json holds per-speaker push
        # endpoint registrations.
        data_dir / "vapid_keys.json",
        data_dir / "push_subscriptions.json",
    ]
    # Per-tier adapter registry + bookkeeping + graph:
    # indexed_key_registry.json and key_metadata.json are tier-root files —
    # <adapters>/<tier>/<file> for main tiers,
    # <adapters>/<tier>/interim_<stamp>/<file> for interim families —
    # written by publish_tier_registry on the fold path, in both simulate
    # and train modes (commit_tier_slot writes the same pair only on the
    # base-swap migration and trial-tree main-tier copy paths). graph.json
    # is a SLOT payload, written inside a timestamped slot under either
    # root exactly like adapter_model.safetensors is for a train payload —
    # never at the tier root — which is why _tier_slot_paths' rglob (not a
    # flat tier-root join) is what actually finds it. Simhashes live inside
    # indexed_key_registry.json under the "simhash" key.
    adapters_root = data_dir / "adapters"
    for _tier in ("episodic", "semantic", "procedural"):
        paths.extend(_tier_slot_paths(adapters_root / _tier))
    # Trial tree counterparts — the trial-migration path's isolated adapter
    # tree (paramem.server.app._build_trial_loop's output_dir). Main-tier
    # files are written by the same commit_tier_slot primitive as the live
    # tree's migration path (commit_main_tiers's copy-forward). The trial
    # fold writes interim slots the same way the live tree does
    # (episodic/interim_<stamp>/) — via write_tier_slot / publish_tier_registry,
    # not commit_tier_slot — so the same per-tier walk
    # (:func:`_tier_slot_paths`) applies without special-casing — a rotation
    # never leaves a trial-tree interim file permanently undecryptable.
    trial_adapters_root = data_state_dir(data_dir) / "trial" / "adapters"
    for _tier in ("episodic", "semantic", "procedural"):
        paths.extend(_tier_slot_paths(trial_adapters_root / _tier))
    # Stage ledger + per-event extraction tree — the two-phase training
    # event's progress record and its phase-1 artifacts (per tier it builds,
    # a shadow registry/key_metadata/keyed list — see
    # `_EXTRACTION_SHADOW_FILENAMES`), written via
    # write_infra_bytes/write_infra_json.  A file outside this enumeration
    # becomes permanently undecryptable after a key rotation.  `data_dir` is
    # the live tree's root; the trial tree's own root
    # (`trial_adapters_root.parent`, i.e. `output_dir.parent` for the trial
    # loop — `ConsolidationLoop._fold_state_dir`) is enumerated separately so
    # a rotation between two base-swap trials never fails gate 4 on an
    # unreadable trial-tree registry.  `_extraction_shadow_paths` derives
    # `data_state_dir(root)` for either, so the live/trial split below is one
    # call per tree, not a duplicated walk.
    paths.extend(_extraction_shadow_paths(data_dir))
    paths.extend(_extraction_shadow_paths(trial_adapters_root.parent))
    # Training scratch state — staging_resume.json holds the per-job fingerprint
    # + checkpoint pointer used by `paramem.training.trainer.train_adapter` for
    # crash resume.  Written via write_infra_bytes (encrypted under Security ON);
    # globbed here so the boot-time mode-consistency probe covers it alongside
    # the other infra files.
    if adapters_root.exists():
        for resume in adapters_root.rglob("staging_resume.json"):
            if _under_hidden_dir(resume, adapters_root):
                continue  # skip .pending-delete and other hidden dirs
            paths.append(resume)
    # Adapter safetensors — full-file encrypted when daily identity is loaded.
    # Each tier's slot directories (and episodic/interim_* siblings) may hold
    # one or more adapter_model.safetensors files.  Enumerate them so
    # rotation, restore, and the startup mode-consistency scan all cover the
    # adapter weight blobs alongside the JSON metadata.
    if adapters_root.exists():
        for safetensors in adapters_root.rglob("adapter_model.safetensors"):
            if _under_hidden_dir(safetensors, adapters_root):
                continue  # skip .pending-delete and other hidden dirs
            paths.append(safetensors)
    return paths


def _probe_data_dir(data_dir: Path) -> ModeProbe:
    """Scan *data_dir* for infrastructure files and classify each as age / plaintext.

    Uses ``infra_paths`` as the authoritative candidate set.  Missing files
    do NOT contribute to either classification — only files that actually
    exist on disk steer the mode verdict.
    """
    probe = ModeProbe()
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return probe

    for path in infra_paths(data_dir):
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

    probe = _probe_data_dir(Path(data_dir))

    has_age = bool(probe.age_paths)
    has_pt = bool(probe.plaintext_paths)

    # Mixing plaintext with age envelopes is a fatal mismatch regardless of
    # which keys are loaded — the store is inconsistent with itself.
    if has_pt and has_age:
        sample_enc = probe.age_paths[0]
        sample_pt = probe.plaintext_paths[0]
        raise FatalConfigError(
            f"Mixed encryption state on disk: "
            f"{sample_enc} is age-encrypted but {sample_pt} is plaintext "
            f"({len(probe.age_paths)} age, {len(probe.plaintext_paths)} "
            f"plaintext in total).\n"
            f"\n"
            f"Likely cause:\n"
            f"  Encryption was enabled after some files were written plaintext,\n"
            f"  or a writer skipped the encrypted helper.\n"
            f"\n"
            f"Remediation:\n"
            f"  - paramem encrypt-infra --dry-run  # preview what would change\n"
            f"  - paramem encrypt-infra            # migrate plaintext files in-place\n"
            f"\n"
            f"See SECURITY.md for a full reset procedure that\n"
            f"preserves speaker_profiles.json."
        )

    # Plaintext files while the daily identity is loaded → encryption enabled
    # but the store is not migrated. Refuse before writing anything.
    if has_pt and daily_identity_loadable:
        raise FatalConfigError(
            f"The daily identity is loaded but {len(probe.plaintext_paths)} "
            f"infrastructure file(s) on disk are plaintext "
            f"(e.g. {probe.plaintext_paths[0]}).\n"
            f"\n"
            f"Likely cause:\n"
            f"  The daily identity is loaded but legacy plaintext files predate\n"
            f"  the encryption rollout (or a prior migration was incomplete).\n"
            f"\n"
            f"Remediation:\n"
            f"  - paramem encrypt-infra  # migrate plaintext files in-place\n"
            f"  - OR unset {DAILY_PASSPHRASE_ENV_VAR} to run in the Security-OFF posture.\n"
            f"\n"
            f"See SECURITY.md for a full reset procedure that\n"
            f"preserves speaker_profiles.json."
        )

    # age files present without the daily identity → unreadable.
    if has_age and not daily_identity_loadable:
        raise FatalConfigError(
            f"{len(probe.age_paths)} infrastructure file(s) on disk are age-"
            f"encrypted (e.g. {probe.age_paths[0]}) but the daily identity "
            f"is not loadable.\n"
            f"\n"
            f"Likely cause:\n"
            f"  {DAILY_PASSPHRASE_ENV_VAR} is unset, the daily key file moved,\n"
            f"  or the passphrase changed since the files were written.\n"
            f"\n"
            f"Remediation:\n"
            f"  - export {DAILY_PASSPHRASE_ENV_VAR}=<your daily passphrase>\n"
            f"  - Confirm ~/.config/paramem/daily_key.age exists and is readable.\n"
            f"  - If you've lost the passphrase, see: paramem restore --help\n"
            f"\n"
            f"See SECURITY.md for a full reset procedure that\n"
            f"preserves speaker_profiles.json."
        )

    # Otherwise the store is consistent with the loaded keys — proceed.
    return
