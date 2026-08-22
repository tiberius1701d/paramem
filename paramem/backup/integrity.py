"""Infrastructure integrity checker for ParaMem.

Verifies the on-disk state of every tier's registry, simhash, manifest,
graph, key_metadata (per tier), and common files (speaker_profiles,
observed_languages, state/backup.json).

Public API
----------
- :class:`FileCheck` — result of checking one file.
- :class:`IntegrityReport` — aggregated result for the whole store.
- :func:`verify_infrastructure_integrity` — run the full suite and return
  an :class:`IntegrityReport`.

Exception→status mapping
-------------------------
The boundary ``try/except`` blocks in this module wrap ONLY the loader call
and name specific exception types.  They record a status and propagate the
detail as a human-readable string.  They do NOT silently swallow errors —
the ``ok``/``failures`` fields on :class:`IntegrityReport` expose every
non-ok non-skipped result to the caller.  Unexpected exceptions (loader bugs,
``OSError``, ``AttributeError``, etc.) propagate so the caller sees them
rather than a silent ``parse_error`` fallback.

Encryption handling
-------------------
``daily_loadable=False`` (default): age-encrypted files whose daily key is
absent report ``undecryptable`` with the distinct "daily identity not loaded"
detail.  These are NOT counted as corruption failures.  The caller decides
(see boot wiring).

``daily_loadable=True``: a decrypt failure IS corruption and IS counted as
a failure.

Interim slot enumeration
------------------------
Tiers to check come from :func:`paramem.memory.interim_adapter.iter_tier_roots`:
the three main tiers (root ``adapter_dir / tier``), then every interim dir
found under ``episodic/interim_*`` — interim slots are episodic-only, never
scanned under other tier roots. :func:`paramem.memory.interim_adapter.adapter_slot_root_for_name`
is used only in the store-only addendum, to resolve the slot root for a tier
the live store carries that has no on-disk match yet.

Required-vs-optional matrix
----------------------------
- Registry required when a tier has keys; simhash required when registry
  non-empty. Both are venue-blind.
- Manifest and payload rows are venue-blind: a tier with active keys gets
  exactly one ``"manifest"``-category row and one ``"payload"``-category row,
  both derived from the SAME :func:`~paramem.adapters.registry_binding.verify_tier_binding`
  resolution — no venue fork. Both venues' payload row consult the binding
  FIRST, identically: a non-``VERIFIED`` binding is ``"inconsistent"``
  regardless of payload kind. Once ``VERIFIED``, a ``"train"``-kind payload
  row reports that SAME verdict — the byte digest was already verified once,
  inside :func:`~paramem.adapters.registry_binding.verify_tier_binding`'s own
  step 7 — while a ``"simulate"``-kind payload row ALSO reads and parses the
  BOUND slot's ``graph.json`` (nothing writes a tier-root ``graph.json`` any
  more), the one structural check the binding's own byte-digest check cannot
  perform (see :func:`_check_payload_row`).
- key_metadata (per tier), speaker_profiles, observed_languages,
  state/backup.json: always optional (skipped when absent — fresh installs
  lack them).
- Empty/absent semantic, absent interim, and partial interim slots (dir present
  but registry absent) → skipped, NOT a failure.

Registry↔slot binding
----------------------
Every keyed tier's live slot is resolved through
:func:`~paramem.adapters.registry_binding.verify_tier_binding` — the one
oracle for "does this tier's on-disk registry bind to a slot manifest",
venue-blind, shared with the boot mount loop and post-fold revalidation. This
module never re-derives hash/slot resolution itself and never re-parses a
manifest the binding already parsed. Each such tier gets exactly ONE
``"manifest"``-category row (:func:`_binding_check`) and one
``"payload"``-category row (:func:`_check_payload_row`) reporting the SAME
binding verdict — so a keyed tier with no slot at all, a hash-mismatched
slot, or a key-count-mismatched slot is each a single visible non-ok check,
never a second row that could assert something the binding already knows is
false (e.g. "no slot" for a tier whose slot exists but doesn't bind).
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pyrage

from paramem.training.stage_ledger import data_state_dir

if TYPE_CHECKING:
    from paramem.adapters.registry_binding import TierBinding

logger = logging.getLogger(__name__)

# ---- Status vocabulary ----
_OK = "ok"
_SKIPPED = "skipped"
_MISSING = "missing"
_UNDECRYPTABLE = "undecryptable"
_PARSE_ERROR = "parse_error"
_SCHEMA_ERROR = "schema_error"
_INCONSISTENT = "inconsistent"

# Detail string used for the "no daily key" undecryptable case — the boot
# wiring uses it to distinguish no-key from corruption.
_DETAIL_NO_KEY = "daily identity not loaded"
# Detail string for a ciphertext that fails decryption (wrong key or corrupt).
_DETAIL_BAD_KEY = "ciphertext corrupt or wrong key"


def _no_key_detail(exc: Exception) -> str:
    """Return ``_DETAIL_NO_KEY`` when *exc* indicates a missing daily identity.

    Used to normalise the detail string from RuntimeError raised by
    ``read_maybe_encrypted`` when the daily age identity is not loaded.

    Args:
        exc: The caught exception.

    Returns:
        ``_DETAIL_NO_KEY`` when the exception text matches a missing-identity
        pattern; otherwise ``str(exc)``.
    """
    msg = str(exc).lower()
    if "age envelope" in msg or "daily identity" in msg:
        return _DETAIL_NO_KEY
    return str(exc)


@dataclass(frozen=True)
class FileCheck:
    """Result of checking one infrastructure file.

    Attributes:
        path: String path of the checked file (str(Path), JSON-serializable).
        category: Logical category — one of ``"registry"``, ``"simhash"``,
            ``"manifest"``, ``"payload"``, ``"key_metadata"``, or ``"common"``.
        tier: Tier name (e.g. ``"episodic"``) or ``"common"`` for cross-tier
            files.
        status: One of ``"ok"``, ``"skipped"``, ``"missing"``,
            ``"undecryptable"``, ``"parse_error"``, ``"schema_error"``, or
            ``"inconsistent"``.
        detail: Human-readable explanation; empty string when status is ``"ok"``.
    """

    path: str
    category: str
    tier: str
    status: str
    detail: str

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict representation."""
        return asdict(self)


@dataclass(frozen=True)
class IntegrityReport:
    """Aggregated result of a full infrastructure integrity check.

    Attributes:
        ok: ``True`` when no failures are present (skipped entries are not
            failures).
        checks: All :class:`FileCheck` results including ok and skipped.
        failures: Subset of *checks* whose ``status`` is not ``"ok"`` or
            ``"skipped"``.
    """

    ok: bool
    checks: list[FileCheck]
    failures: list[FileCheck]

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict suitable for the ``GET /integrity`` endpoint."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_registry(path: Path, tier: str) -> tuple[FileCheck, list[str] | None, list[str] | None]:
    """Check an ``indexed_key_registry.json`` via :class:`KeyRegistry.load`.

    Returns a ``(FileCheck, active_keys, known_keys)`` triple.

    - ``active_keys`` is the list of active key names on success, ``None``
      otherwise.  Used for ``has_keys`` (required-vs-optional logic) and the
      ``missing_from_sh`` direction — every SERVED key must have a fingerprint.
    - ``known_keys`` is the union of active ∪ stale key names on success,
      ``None`` otherwise.  Used for the ``orphan_in_sh`` direction — a
      fingerprint is legitimate iff the key is active (the registry's
      one-map shape refuses a file whose ``"simhash"`` section names a
      stale/withheld id at parse time, so ``known_keys`` and ``active_keys``
      are equivalent populations for this check in any loadable file).

    Callers reuse both lists for cross-consistency checks to avoid reading the
    file twice.  Unexpected exceptions propagate so the caller sees them rather
    than a silent fallback status.
    """
    from paramem.training.key_registry import KeyRegistry

    path_str = str(path)
    if not path.exists():
        return FileCheck(path_str, "registry", tier, _SKIPPED, ""), None, None

    try:
        reg = KeyRegistry.load(path)
        active_keys = reg.list_active()
        known_keys = reg.list_known()
        return FileCheck(path_str, "registry", tier, _OK, ""), active_keys, known_keys
    except RuntimeError as exc:
        # RuntimeError from read_maybe_encrypted means no daily identity loaded.
        check = FileCheck(path_str, "registry", tier, _UNDECRYPTABLE, _no_key_detail(exc))
        return check, None, None
    except pyrage.DecryptError:
        check = FileCheck(path_str, "registry", tier, _UNDECRYPTABLE, _DETAIL_BAD_KEY)
        return check, None, None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return FileCheck(path_str, "registry", tier, _PARSE_ERROR, str(exc)), None, None
    except (KeyError, ValueError) as exc:
        return FileCheck(path_str, "registry", tier, _SCHEMA_ERROR, str(exc)), None, None


def _check_simhash(path: Path, tier: str) -> tuple[FileCheck, dict | None]:
    """Extract the simhash fingerprint map from an ``indexed_key_registry.json``.

    The simhash map is now co-located with the registry in a single file
    (``"simhash"`` key in the registry payload) rather than in a separate
    ``simhash_registry.json`` sidecar.  *path* MUST be the
    ``indexed_key_registry.json`` path (the same file passed to
    :func:`_check_registry`).

    Returns a ``(FileCheck, simhash_dict)`` pair.  ``simhash_dict`` is the
    ``{key: int}`` mapping when the file parsed successfully with a ``"simhash"``
    entry, ``None`` otherwise.  Callers reuse the returned dict for
    cross-consistency checks to avoid a second read of the same file.

    Unexpected exceptions propagate so the caller sees them rather than a
    silent fallback status.

    Returns:
        ``(FileCheck, simhash_dict | None)`` where ``simhash_dict`` is the
        loaded simhash map on success, or ``None`` on any non-ok status.
    """
    from paramem.training.key_registry import KeyRegistry

    path_str = str(path)
    if not path.exists():
        return FileCheck(path_str, "simhash", tier, _SKIPPED, ""), None

    try:
        sh_dict = KeyRegistry.load_simhashes(path)
        return FileCheck(path_str, "simhash", tier, _OK, ""), sh_dict
    except RuntimeError as exc:
        return FileCheck(path_str, "simhash", tier, _UNDECRYPTABLE, _no_key_detail(exc)), None
    except pyrage.DecryptError:
        return FileCheck(path_str, "simhash", tier, _UNDECRYPTABLE, _DETAIL_BAD_KEY), None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return FileCheck(path_str, "simhash", tier, _PARSE_ERROR, str(exc)), None
    except (KeyError, ValueError) as exc:
        return FileCheck(path_str, "simhash", tier, _SCHEMA_ERROR, str(exc)), None


def _check_graph(path: Path, tier: str) -> FileCheck:
    """Check a written slot's ``graph.json`` via
    :func:`paramem.memory.persistence.load_memory_from_disk`.

    *path* must be the BOUND slot's payload file (``<slot>/graph.json``) —
    resolved by the caller from a :class:`~paramem.adapters.registry_binding.TierBinding`,
    never a tier-root path (nothing writes a tier-root ``graph.json``).

    Graph checks do not contribute a parsed payload to cross-consistency checks
    so this helper returns a plain :class:`FileCheck` (not a tuple).

    Unexpected exceptions propagate so the caller sees them rather than a
    silent fallback status.

    Returns:
        :class:`FileCheck` with category ``"payload"`` and the resolved status.
    """
    from paramem.memory.persistence import load_memory_from_disk

    path_str = str(path)
    if not path.exists():
        return FileCheck(path_str, "payload", tier, _MISSING, "graph.json not found")

    try:
        load_memory_from_disk(path)
        return FileCheck(path_str, "payload", tier, _OK, "")
    except RuntimeError as exc:
        return FileCheck(path_str, "payload", tier, _UNDECRYPTABLE, _no_key_detail(exc))
    except pyrage.DecryptError:
        return FileCheck(path_str, "payload", tier, _UNDECRYPTABLE, _DETAIL_BAD_KEY)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return FileCheck(path_str, "payload", tier, _PARSE_ERROR, str(exc))


def _check_common_file(
    path: Path, category: str, tier: str = "common"
) -> tuple[FileCheck, dict | None]:
    """Check a common JSON file via :func:`read_maybe_encrypted` + ``json.loads``.

    Returns a ``(FileCheck, parsed_dict)`` pair.  ``parsed_dict`` is the
    deserialized JSON object when the file loaded successfully, ``None``
    otherwise.  Callers use the returned dict for cross-consistency checks
    (e.g. a tier's key_metadata orphan check) to avoid a second read of the
    same file.

    Unexpected exceptions propagate so the caller sees them rather than a
    silent fallback status.

    Args:
        path: Path to check.
        category: One of ``"key_metadata"`` or ``"common"``.
        tier: The :class:`FileCheck` row's tier label.  Defaults to
            ``"common"`` for genuinely tier-independent files
            (speaker_profiles, observed_languages, state/backup.json); a
            per-tier ``key_metadata.json`` check passes its own tier name so
            the row is attributed to that tier rather than lumped under
            ``"common"``.

    Returns:
        ``(FileCheck, parsed_dict | None)`` where ``parsed_dict`` is the
        deserialized JSON on success, or ``None`` on any non-ok status.
        Missing files return ``(_SKIPPED, None)`` (optional files — fresh
        installs lack them).
    """
    from paramem.backup.encryption import read_maybe_encrypted

    path_str = str(path)
    if not path.exists():
        return FileCheck(path_str, category, tier, _SKIPPED, ""), None

    try:
        raw = read_maybe_encrypted(path)
        parsed = json.loads(raw.decode("utf-8"))
        return FileCheck(path_str, category, tier, _OK, ""), parsed
    except RuntimeError as exc:
        return FileCheck(path_str, category, tier, _UNDECRYPTABLE, _no_key_detail(exc)), None
    except pyrage.DecryptError:
        return FileCheck(path_str, category, tier, _UNDECRYPTABLE, _DETAIL_BAD_KEY), None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return FileCheck(path_str, category, tier, _PARSE_ERROR, str(exc)), None


def _binding_check(binding: "TierBinding", tier: str) -> FileCheck:
    """Map one tier's :class:`~paramem.adapters.registry_binding.TierBinding`
    verdict onto this module's status vocabulary.

    THE single manifest-category row for a keyed tier, either venue — never
    paired with a second, independently-derived row. Deliberately does not
    re-read ``meta.json``: :func:`~paramem.adapters.registry_binding.verify_tier_binding`
    already parsed it (``binding.manifest``) when it resolved a slot, and
    ``find_live_slot`` (the module docstring's "consumers must read the
    manifest from the binding, never re-read" contract) already skips
    candidates that fail to parse — so a second read could only ever repeat
    the same verdict, except across the exact TOCTOU race that contract
    guards against.

    :data:`~paramem.adapters.registry_binding.VERIFIED` is ``"ok"``.
    :data:`~paramem.adapters.registry_binding.NO_CANDIDATES` is ``"missing"``
    — no written payload slot candidate exists and the registry carries no
    active key (fresh tier, or one whose slot was fully removed and never
    held content).
    Every other verdict
    (:data:`~paramem.adapters.registry_binding.KEYS_WITHOUT_SLOT`,
    :data:`~paramem.adapters.registry_binding.NO_MATCHING_SLOT`,
    :data:`~paramem.adapters.registry_binding.KEY_COUNT_MISMATCH`,
    :data:`~paramem.adapters.registry_binding.PAYLOAD_MISMATCH`,
    :data:`~paramem.adapters.registry_binding.REGISTRY_UNREADABLE`,
    :data:`~paramem.adapters.registry_binding.REGISTRY_ABSENT_WITH_SLOTS`) is
    a registry↔slot disagreement and maps to ``"inconsistent"`` — the same
    vocabulary entry already used elsewhere in this module for cross-artifact
    disagreement (registry vs simhash).

    Args:
        binding: The tier's already-resolved
            :class:`~paramem.adapters.registry_binding.TierBinding`.
        tier: Tier name, for the :class:`FileCheck` row.

    Returns:
        :class:`FileCheck` with category ``"manifest"`` reporting the
        binding verdict. ``path`` is ``binding.slot / "meta.json"`` when the
        binding resolved a slot (:data:`VERIFIED`,
        :data:`KEY_COUNT_MISMATCH`, :data:`PAYLOAD_MISMATCH`) — the manifest
        a written payload slot actually does exist at, even when it doesn't bind.
        Otherwise ``path`` is the tier's registry file, since no slot was
        resolved for this verdict.
    """
    from paramem.adapters.registry_binding import NO_CANDIDATES, VERIFIED

    if binding.slot is not None:
        path_str = str(binding.slot / "meta.json")
    else:
        path_str = str(binding.tier_root / "indexed_key_registry.json")

    if binding.status == VERIFIED:
        return FileCheck(path_str, "manifest", tier, _OK, "")
    if binding.status == NO_CANDIDATES:
        return FileCheck(path_str, "manifest", tier, _MISSING, binding.detail)
    return FileCheck(path_str, "manifest", tier, _INCONSISTENT, binding.detail)


def _check_payload_row(binding: "TierBinding", tier: str) -> FileCheck:
    """Derive the one ``"payload"``-category row for *tier* from its already-
    resolved :class:`~paramem.adapters.registry_binding.TierBinding` —
    venue-blind, no second slot resolution.

    Both venues consult the binding FIRST, identically: a non-``VERIFIED``
    binding — ``PAYLOAD_MISMATCH``, ``KEY_COUNT_MISMATCH``, or any other
    non-publishable verdict — reports ``"inconsistent"`` with the binding's
    own ``detail``, regardless of payload kind. Only once the binding is
    :data:`~paramem.adapters.registry_binding.VERIFIED` does the two venues'
    treatment diverge: a ``train`` payload's byte-level digest was already
    verified once, inside
    :func:`~paramem.adapters.registry_binding.verify_tier_binding`'s own
    step 7 — this row surfaces that SAME verdict (``"ok"``) rather than
    re-deriving a second, independent digest comparison. A ``simulate``
    payload has no independent digest check inside ``verify_tier_binding``
    beyond the same step-7 byte digest, so a VERIFIED simulate binding gets
    an ADDITIONAL structural check here: the bound slot's ``graph.json`` is
    read and PARSED via :func:`_check_graph` (never a tier-root path —
    nothing writes one) — the one check ``verify_tier_binding`` cannot
    perform itself (it hashes bytes, never parses graph structure).
    :func:`_check_graph` only ever runs when *binding.status* is
    :data:`~paramem.adapters.registry_binding.VERIFIED`: both venues agree
    that a non-VERIFIED binding never gets an ``"ok"`` payload row.

    A binding that resolved no slot at all (``NO_CANDIDATES``,
    ``KEYS_WITHOUT_SLOT``, ``NO_MATCHING_SLOT``, etc.) has no payload to
    check; the row is ``"skipped"`` with the binding's own detail so this
    row never independently re-derives what the manifest row already says.

    Args:
        binding: The tier's already-resolved
            :class:`~paramem.adapters.registry_binding.TierBinding`.
        tier: Tier name, for the :class:`FileCheck` row.

    Returns:
        :class:`FileCheck` with category ``"payload"``.
    """
    from paramem.adapters.registry_binding import VERIFIED
    from paramem.adapters.slot import payload_filename

    if binding.slot is None or binding.manifest is None:
        return FileCheck(
            str(binding.tier_root),
            "payload",
            tier,
            _SKIPPED,
            binding.detail or "no bound slot",
        )

    kind = binding.manifest.payload.kind
    payload_path = binding.slot / payload_filename(kind)

    # Both venues consult the binding first: a non-VERIFIED binding is
    # "inconsistent" regardless of payload kind — see this function's
    # docstring for why the simulate arm can no longer skip this check.
    if binding.status != VERIFIED:
        return FileCheck(str(payload_path), "payload", tier, _INCONSISTENT, binding.detail)

    if kind == "simulate":
        # VERIFIED only proves the byte digest matches (verify_tier_binding
        # step 7) -- a simulate payload gets the additional structural parse
        # check a train payload's digest-only verification doesn't need.
        return _check_graph(payload_path, tier)

    # train: the payload digest is verified inside verify_tier_binding's own
    # step 7 whenever the binding reaches VERIFIED — this row surfaces that
    # SAME verdict rather than re-deriving a second, independent comparison.
    return FileCheck(str(payload_path), "payload", tier, _OK, "")


def _is_no_key_check(check: FileCheck) -> bool:
    """Return ``True`` when *check* is an undecryptable-with-no-key result.

    Args:
        check: The file check to inspect.

    Returns:
        ``True`` when the status is ``"undecryptable"`` and the detail
        indicates the daily key was not loaded.
    """
    return check.status == _UNDECRYPTABLE and check.detail == _DETAIL_NO_KEY


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cleanup_partial_slots(adapter_dir: Path) -> list[dict]:
    """Delete partial-trained/partial-written adapter slot directories under each main tier.

    Walks ``<adapter_dir>/<tier>/`` for every tier in
    :data:`~paramem.utils.tiers.MAIN_TIERS` and removes any
    subdirectory that is NOT a complete slot.  A "complete slot" carries
    every file :func:`~paramem.adapters.slot.required_slot_files` names for
    the slot's OWN payload kind — read from its ``meta.json`` — so a train
    slot's completeness set differs from a simulate slot's.  A subdirectory
    whose ``meta.json`` is ABSENT is scratch by definition — there is no
    payload kind to derive a requirement from — and is removed exactly like
    one whose declared kind is short a file, via ``shutil.rmtree``.

    A subdirectory whose ``meta.json`` is PRESENT but fails to parse (e.g.
    prior-schema) is NOT scratch and is NEVER deleted here — this function
    repairs nothing; it only reaps debris that never had a manifest to
    begin with. Such a slot is left on disk untouched and logged at
    WARNING with the parse error; registry↔slot binding verification
    (:mod:`paramem.adapters.registry_binding`) will leave the tier
    unpublishable loudly if it holds active keys, for the operator to
    investigate via ``POST /backup/restore`` — never an automatic repair
    or delete.

    Called from :func:`paramem.server.app._sweep_keyless_tier_artifacts`
    (itself called from ``_mount_adapters_from_slots``), pre-mount: before
    any tier's :func:`~paramem.adapters.registry_binding.verify_tier_binding`
    is computed, on every boot/reload that loads a local model. A torn
    scratch slot left in place could otherwise be counted as a weight-slot
    candidate (``count_slot_candidates`` only requires a readable
    ``meta.json``, not a complete slot) and produce a binding verdict that
    disagrees with what a later pass — the mount loop itself, the
    memory-store publish, or the integrity report, all reached later in the
    same boot — would compute for the identical tier once the scratch is
    gone. Running this before all three means they see one already-cleaned
    tree instead of independently timed snapshots.

    Skipped (never touched):
    - Dotted entries (``.quarantine``, ``.tmp``).
    - Interim container directories (``interim_*`` under episodic/).  These
      are nested containers whose integrity is fully owned by
      ``find_live_slot`` (manifest.py), the boot-time keyless-tier sweep
      (``_sweep_keyless_tier_artifacts``, app.py — reaps a tier's on-disk
      artifacts, pre-mount, based on
      :func:`~paramem.adapters.registry_binding.verify_tier_binding`'s
      registry↔slot binding verdict for the tier, gated by the
      erase-in-flight marker for any shape the binding does not
      independently corroborate as empty; see that function's docstring for
      the full decision table), and the post-consolidation teardown
      (``unload_interim_adapters``).  Applying the flat 3-file completeness
      check to an interim container is wrong because weights live in the
      inner ``<ts>/`` slot, not at the container root.  Passing judgment
      here would be a parallel-topology drift bug.
    - The staging slot conventions are in-memory PEFT keys, not on disk —
      this function cannot affect them.
    - The ``bg_checkpoint_epoch`` and ``checkpoint-*`` scratch dirs written
      by HF Trainer live UNDER the caller's ``output_dir`` (training-side),
      not under ``adapter_dir/<tier>/`` — they are out of scope.

    Args:
        adapter_dir: Root adapter directory (``config.adapter_dir`` /
            ``paths.adapters``).

    Returns:
        One dict per deleted slot describing what was removed, suitable for
        embedding in ``_state["integrity_cleanup"]`` and rendering by the
        attention populator::

            {"tier": "episodic",
             "slot_name": "20260526T1200",
             "path": "/.../adapters/episodic/20260526T1200",
             "missing": ["meta.json"]}

        Returns an empty list when no partial slots are found.
    """
    from paramem.adapters.manifest import (
        ManifestNotFoundError,
        ManifestSchemaError,
        read_manifest,
    )
    from paramem.adapters.slot import required_slot_files
    from paramem.memory.interim_adapter import INTERIM_DIR_PREFIX
    from paramem.utils.tiers import MAIN_TIERS

    removed: list[dict] = []
    for tier_name in MAIN_TIERS:
        tier_root = adapter_dir / tier_name
        if not tier_root.is_dir():
            continue
        for entry in tier_root.iterdir():
            if entry.name.startswith("."):
                continue
            if not entry.is_dir():
                continue
            # Interim containers are owned by find_live_slot +
            # _sweep_keyless_tier_artifacts's boot-time keyless-tier sweep
            # (reaps a tier's on-disk artifacts pre-mount only when its
            # registry file exists and affirmatively reads zero known keys;
            # preserves an unreadable/foreign-shaped registry or one that
            # still lists a key), not by flat-slot cleanup.  Skip
            # unconditionally.
            if entry.name.startswith(INTERIM_DIR_PREFIX):
                continue
            try:
                manifest = read_manifest(entry)
            except ManifestNotFoundError:
                missing = ["meta.json"]
            except ManifestSchemaError as exc:
                # meta.json is PRESENT but unreadable (e.g. prior-schema).
                # This is NOT scratch — a manifest that exists but fails to
                # parse is a store the operator must investigate, not debris
                # this boot-time sweep may delete. Binding verification will
                # leave the tier unpublishable loudly (KEYS_WITHOUT_SLOT or
                # similar); nothing here repairs or removes it. Leave it in
                # place.
                logger.warning(
                    "integrity-cleanup: leaving slot %s in place — meta.json "
                    "present but unparseable (%s); binding verification will "
                    "leave this tier unpublishable if it holds active keys",
                    entry,
                    exc,
                )
                continue
            else:
                missing = [
                    f
                    for f in required_slot_files(manifest.payload.kind)
                    if not (entry / f).exists()
                ]
            if not missing:
                continue  # complete slot — retained
            logger.warning(
                "integrity-cleanup: removing partial slot %s (missing: %s)",
                entry,
                ", ".join(missing),
            )
            shutil.rmtree(entry, ignore_errors=False)
            removed.append(
                {
                    "tier": tier_name,
                    "slot_name": entry.name,
                    "path": str(entry),
                    "missing": missing,
                }
            )
    return removed


def verify_infrastructure_integrity(
    config,
    *,
    store=None,
    daily_loadable: bool = False,
) -> IntegrityReport:
    """Run the full infrastructure integrity check and return an :class:`IntegrityReport`.

    Checks every tier's ``indexed_key_registry.json`` (which now carries the
    unified simhash map) and its own ``key_metadata.json``.  Also checks
    ``speaker_profiles.json``, ``observed_languages.json``, and
    ``state/backup.json`` (tier-independent, category ``"common"``).

    A keyed tier's live slot and its registry↔slot binding verdict are both
    resolved venue-blind via a single call to
    :func:`~paramem.adapters.registry_binding.verify_tier_binding` (see the
    module docstring's "Registry↔slot binding" section) — every such tier
    gets exactly one ``"manifest"``-category binding row (:func:`_binding_check`)
    and one ``"payload"``-category row (:func:`_check_payload_row`, reading
    the bound slot's ``graph.json`` for a ``simulate`` payload); no second,
    independently re-derived row is ever added for either category.

    Runs cross-consistency checks on tiers whose registry loaded ``"ok"``:
    registry keys vs simhash keys, and — per tier, like the prune — that
    tier's own key_metadata orphans.

    Args:
        config: Live :class:`paramem.server.config.ServerConfig` (or mock with
            ``adapter_dir`` and ``paths.data``).
        store: Optional live :class:`paramem.memory.store.MemoryStore`.  When
            supplied, augments tier enumeration with in-memory registry data.
        daily_loadable: Whether the daily age identity is loadable.  When
            ``False``, ``undecryptable`` entries with the "daily identity not
            loaded" detail are not counted as corruption failures.  When
            ``True``, any decrypt failure is a real failure.

    Returns:
        :class:`IntegrityReport` with all check results and the aggregated
        ``ok`` flag.
    """
    adapter_dir = Path(config.adapter_dir)
    data_dir = Path(config.paths.data)

    checks: list[FileCheck] = []

    # -----------------------------------------------------------------------
    # Build the set of tiers to check.
    # Main tiers + any interim dirs discovered on disk or in the store.
    # -----------------------------------------------------------------------
    tiers_to_check: list[tuple[str, Path | None, str]] = []
    # (tier_name, slot_root_for_manifest, "main"|"interim")
    #
    # Interim adapters are episodic-only: day-by-day session slots that
    # collapse into the main tiers on consolidation. (procedural/semantic
    # interim_* adapter slots do not exist; any procedural/interim_* dir holds
    # training debris — epoch_log/progress — not an adapter, and is ignored.)
    from paramem.adapters.registry_binding import verify_tier_binding
    from paramem.memory.interim_adapter import INTERIM_NAME_PREFIX, iter_tier_roots

    for tier_name, tier_root in iter_tier_roots(adapter_dir):
        kind = "interim" if tier_name.startswith(INTERIM_NAME_PREFIX) else "main"
        tiers_to_check.append((tier_name, tier_root, kind))

    # Also add any tiers from the live store not yet on disk
    if store is not None:
        for store_tier in store.tiers_with_registry():
            if not any(t == store_tier for t, _, _ in tiers_to_check):
                from paramem.memory.interim_adapter import adapter_slot_root_for_name

                slot_root = adapter_slot_root_for_name(adapter_dir, store_tier)
                kind = "interim" if store_tier.startswith(INTERIM_NAME_PREFIX) else "main"
                tiers_to_check.append((store_tier, slot_root, kind))

    # -----------------------------------------------------------------------
    # Per-tier checks
    # -----------------------------------------------------------------------
    # Track which registry loads succeeded (for cross-consistency checks).
    registry_ok_keys: dict[str, list[str]] = {}  # tier -> active_keys list (SERVE)
    registry_known_keys: dict[str, list[str]] = {}  # tier -> known (active∪stale) list
    simhash_ok_keys: dict[str, list[str]] = {}  # tier -> simhash keys list
    tier_root_by_name: dict[str, Path] = {}  # tier -> resolved slot root (for display labels)

    for tier_name, tier_root, tier_kind in tiers_to_check:
        tier_root = Path(tier_root)
        tier_root_by_name[tier_name] = tier_root

        # --- Determine if this tier is "committed" (has any data) ---
        # A partial interim slot (dir present but registry absent) is skipped.
        reg_path = tier_root / "indexed_key_registry.json"
        # The simhash map now lives inside indexed_key_registry.json under the
        # "simhash" key.  simhash_path is kept as a display string in
        # FileCheck records so the API output is human-readable.
        simhash_path = reg_path  # same file — display label only

        # Skip entirely-absent tiers (no dir at all or no registry signal).
        if not tier_root.exists():
            # Absent interim → skipped (not a failure)
            # Absent main tier (e.g. semantic on a fresh install) → skipped
            checks.append(FileCheck(str(reg_path), "registry", tier_name, _SKIPPED, ""))
            continue

        # Partial interim slot: dir present but no registry file → skipped
        if tier_kind == "interim" and not reg_path.exists():
            checks.append(
                FileCheck(str(reg_path), "registry", tier_name, _SKIPPED, "partial interim slot")
            )
            continue

        # --- Registry check ---
        # _check_registry returns (FileCheck, active_keys | None, known_keys | None);
        # reuse both payloads for cross-consistency so the file is read only once.
        reg_check, reg_active_keys, reg_known_keys = _check_registry(reg_path, tier_name)
        checks.append(reg_check)

        # Determine whether this tier has active keys (drives required/optional logic).
        # has_keys = ACTIVE keys only — a stale-only tier serves nothing; its
        # simhash/manifest remains optional.
        has_keys = False
        if reg_check.status == _OK and reg_active_keys is not None:
            has_keys = len(reg_active_keys) > 0
            registry_ok_keys[tier_name] = reg_active_keys
        if reg_check.status == _OK and reg_known_keys is not None:
            registry_known_keys[tier_name] = reg_known_keys

        # Empty registry (zero active keys) → simhash + manifest optional for this tier.
        # Non-existent registry for a main tier → skipped (fresh install or cleared tier).
        if not reg_path.exists():
            # Fresh tier — skip simhash and graph too
            checks.append(FileCheck(str(simhash_path), "simhash", tier_name, _SKIPPED, ""))
            continue

        # --- SimHash check ---
        # Simhashes live in the same indexed_key_registry.json file under the
        # "simhash" key.  _check_simhash reads from that file and extracts the
        # fingerprint map; no separate simhash_registry.json exists.
        # reuse the parsed dict for cross-consistency so the file is read only once.
        if not has_keys:
            # Registry loaded ok but is empty → simhash is optional
            simhash_check: FileCheck = FileCheck(
                str(simhash_path), "simhash", tier_name, _SKIPPED, "empty registry"
            )
            simhash_payload: dict | None = None
        else:
            # Both train and simulate: simhash is in the registry file,
            # which was already confirmed to exist above.
            simhash_check, simhash_payload = _check_simhash(reg_path, tier_name)
        checks.append(simhash_check)

        if simhash_check.status == _OK and simhash_payload is not None:
            simhash_ok_keys[tier_name] = list(simhash_payload.keys())

        # --- key_metadata check (per-tier; orphan question is local to this
        # tier, like the prune) ---
        km_path = tier_root / "key_metadata.json"
        km_check, km_parsed = _check_common_file(km_path, "key_metadata", tier_name)
        checks.append(km_check)
        if km_check.status == _OK and km_parsed is not None and reg_known_keys is not None:
            km_known_set = set(reg_known_keys)
            km_orphan_keys = sorted(k for k in km_parsed.get("keys", {}) if k not in km_known_set)
            if km_orphan_keys:
                sample = km_orphan_keys[:10]
                checks.append(
                    FileCheck(
                        str(km_path),
                        "key_metadata",
                        tier_name,
                        _INCONSISTENT,
                        f"key_metadata keys not in {tier_name} registry: {sample}",
                    )
                )

        # --- Manifest + payload check (venue-blind, one binding resolution) ---
        # verify_tier_binding is the one oracle for "does this tier's
        # on-disk registry bind to a slot manifest" — venue-blind: it
        # resolves the live slot (hash-matched, not newest-mtime) by reading
        # meta.json regardless of whether the slot's payload is weights or a
        # graph, and reports whether the registry and the slot corroborate
        # each other, in a single verdict. _binding_check is THE one
        # manifest-category row for this tier — never paired with a second,
        # independently resolved row. _check_payload_row derives the
        # "payload"-category row from that SAME binding — no second slot
        # resolution — reading the bound slot's graph.json when the payload
        # kind is "simulate" (nothing writes a tier-root graph.json any
        # more); a train-kind payload's byte-level digest was already
        # verified once inside verify_tier_binding itself, and this row
        # surfaces that SAME verdict (see _check_payload_row).
        # tier_root is already resolved for both main and interim tiers:
        # main  → <adapter_dir>/<tier>/
        # interim → <adapter_dir>/episodic/interim_<stamp>/   (nested)
        if has_keys:
            binding = verify_tier_binding(tier_name, tier_root)
            checks.append(_binding_check(binding, tier_name))
            checks.append(_check_payload_row(binding, tier_name))
        else:
            # Empty registry: nothing to bind, nothing to check.
            checks.append(
                FileCheck(str(reg_path), "manifest", tier_name, _SKIPPED, "empty registry")
            )
            checks.append(
                FileCheck(str(reg_path), "payload", tier_name, _SKIPPED, "empty registry")
            )

    # -----------------------------------------------------------------------
    # Common files (always optional — fresh installs lack them).
    # key_metadata is now checked per-tier, inside the per-tier loop above —
    # there is no global file left to check here.
    # -----------------------------------------------------------------------
    speaker_profiles_path = data_dir / "speaker_profiles.json"
    observed_languages_path = data_dir / "observed_languages.json"
    backup_state_path = data_state_dir(data_dir) / "backup.json"

    checks.append(_check_common_file(speaker_profiles_path, "common")[0])
    checks.append(_check_common_file(observed_languages_path, "common")[0])
    checks.append(_check_common_file(backup_state_path, "common")[0])

    # -----------------------------------------------------------------------
    # Cross-consistency checks
    # -----------------------------------------------------------------------
    # Only for tiers whose registry loaded "ok".
    for tier_name, reg_keys in registry_ok_keys.items():
        sh_keys = simhash_ok_keys.get(tier_name, [])
        reg_set = set(reg_keys)
        sh_set = set(sh_keys)

        # Keys in registry but not in simhash (in-payload self-consistency).
        # Since simhashes and registry keys are now in the same file, this
        # detects an in-file invariant violation rather than cross-file desync.
        # Uses the tier's already-resolved slot root (interim tiers live at
        # <adapter_dir>/episodic/interim_<stamp>/, not a flat
        # <adapter_dir>/<tier_name>/ join) — the resolver is not called again.
        _reg_file = str(tier_root_by_name[tier_name] / "indexed_key_registry.json")
        missing_from_sh = sorted(reg_set - sh_set)
        if missing_from_sh:
            sample = missing_from_sh[:10]
            detail = f"registry keys without simhash fingerprint: {sample}"
            checks.append(
                FileCheck(
                    _reg_file,
                    "simhash",
                    tier_name,
                    _INCONSISTENT,
                    detail,
                )
            )

        # Keys in simhash but not known to registry (orphan fingerprints).
        # sh_set is parsed by KeyRegistry.load_simhashes (via
        # _check_simhash, above) -> load -> _from_payload. The parse itself
        # REFUSES a file whose "simhash" section names a withheld id
        # (KeyRegistry.load_from_bytes, key_registry.py) rather than
        # dropping the entry — a withheld id's fingerprint is instead
        # dropped once, at write time, by the one-time
        # scripts/migrate/stamp_slot_manifests_v5.py pass that upgrades a
        # pre-migration file. Either way a withheld id never reaches sh_set
        # in a loadable file. In every loadable state, ``sh_set - known_set`` and
        # ``sh_set - reg_set`` (active only) are therefore the same set:
        # comparing against known_set (active ∪ withheld) below is a
        # parse-level equivalence, not a design choice this check depends
        # on — a fingerprint is an orphan only when the key is absent from
        # both the registry file's populations.
        known_set = set(registry_known_keys.get(tier_name, reg_keys))
        orphan_in_sh = sorted(sh_set - known_set)
        if orphan_in_sh:
            sample = orphan_in_sh[:10]
            detail = f"simhash keys absent from registry: {sample}"
            checks.append(
                FileCheck(
                    _reg_file,
                    "simhash",
                    tier_name,
                    _INCONSISTENT,
                    detail,
                )
            )

    # -----------------------------------------------------------------------
    # Build report
    # -----------------------------------------------------------------------
    failures: list[FileCheck] = []
    for check in checks:
        if check.status not in (_OK, _SKIPPED):
            # Distinguish no-key undecryptable from real corruption.
            if _is_no_key_check(check) and not daily_loadable:
                # No-key undecryptable when daily identity not loaded — not
                # a corruption failure; caller decides what to do with it.
                continue
            failures.append(check)

    return IntegrityReport(ok=not failures, checks=checks, failures=failures)
