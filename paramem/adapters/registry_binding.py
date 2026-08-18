"""Registry↔slot-manifest binding verification for one adapter tier.

Answers a single question for a tier root — "is the on-disk registry
corroborated by its slot manifests" — as one verdict per tier, so every
consumer (boot mount, post-fold manifest revalidation, store publish,
config-apply shape-change preview) reads the same decision instead of
re-deriving it from :mod:`paramem.adapters.manifest`'s lower-level
primitives.

Split out of :mod:`paramem.adapters.manifest` — that module already owns
manifest schema, slot resolution, and base-model hashing; registry↔slot
binding is a distinct concern layered on top of it.

What verification does NOT cover
---------------------------------
* **Slot completeness beyond the payload file itself.** A candidate slot is
  ``meta.json`` only (see
  :func:`~paramem.adapters.manifest.count_slot_candidates`); a train slot's
  sibling ``adapter_config.json`` is not independently verified here — that
  is :func:`~paramem.backup.integrity.cleanup_partial_slots`'s job. The
  payload file itself (``payload_filename(manifest.payload.kind)``) IS read
  and hashed, for both venues alike — see the payload check below.
* **A payload without a manifest.** Weights (or a graph) on disk with no
  ``meta.json`` are invisible here — the tier's ``candidate_count`` stays
  0 in that shape, and it publishes as :data:`NO_CANDIDATES` when the
  registry carries no active key, or is unpublishable as
  :data:`KEYS_WITHOUT_SLOT` when it does.
* **Which keys the count refers to** — only the active-key *count* is
  compared against the manifest's ``key_count`` stamp; individual key
  identity is never checked.
* **Bookkeeping** (``key_metadata.json``) — entirely out of scope.

:func:`verify_tier_binding` is a pure function of the tree as it stands at
call time. Boot deletes slots between stages (e.g.
:func:`~paramem.backup.integrity.cleanup_partial_slots` runs before the
mount loop) — callers that must agree on one verdict for a tier need to
call this after the same housekeeping point; two calls straddling a
mutation can legitimately disagree.

Totality
--------
:func:`verify_tier_binding` raises only for the donor guard (a caller
error — a donor store is never a memory tier). Every filesystem failure on
a *tier_root* it is legitimately asked to verify — an unreadable registry,
an unreadable slot candidate, a directory that cannot be enumerated —
resolves to a verdict rather than propagating, because
:func:`verify_adapter_tree` walks every tier unconditionally and one
tier's broken permissions must never crash boot for the others. See each
resolution-order step below for exactly which verdict a given failure
mode maps to.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from paramem.adapters.manifest import (
    AdapterManifest,
    ManifestNotFoundError,
    ManifestSchemaError,
    count_slot_candidates,
    find_live_slot,
    read_manifest,
    tier_registry_sha256,
)
from paramem.adapters.slot import payload_filename
from paramem.backup.hashing import plaintext_sha256
from paramem.training.donor import DONOR_STORE_PREFIX
from paramem.training.key_registry import KeyRegistry

# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

VERIFIED = "verified"
NO_CANDIDATES = "no_candidates"
KEYS_WITHOUT_SLOT = "keys_without_slot"
NO_MATCHING_SLOT = "no_matching_slot"
KEY_COUNT_MISMATCH = "key_count_mismatch"
PAYLOAD_MISMATCH = "payload_mismatch"
REGISTRY_UNREADABLE = "registry_unreadable"
REGISTRY_ABSENT_WITH_SLOTS = "registry_absent_with_slots"

ALL_VERDICTS: frozenset[str] = frozenset(
    {
        VERIFIED,
        NO_CANDIDATES,
        KEYS_WITHOUT_SLOT,
        NO_MATCHING_SLOT,
        KEY_COUNT_MISMATCH,
        PAYLOAD_MISMATCH,
        REGISTRY_UNREADABLE,
        REGISTRY_ABSENT_WITH_SLOTS,
    }
)
"""The complete, closed set of verdict strings :func:`verify_tier_binding`
can return. Single definition — every module that needs "the whole verdict
vocabulary" (rather than one named verdict) imports this instead of
hand-listing the constants again, so a verdict added here is a type error
at every consuming totality check, never a silent drift."""

_PUBLISHABLE = frozenset({VERIFIED, NO_CANDIDATES})
"""Verdicts under which a tier's registry is safe to publish into the live
store: :data:`VERIFIED` (registry corroborated by a matching slot whose
payload digest still verifies) and :data:`NO_CANDIDATES` (no written payload
slot candidate exists AND the registry carries no active key — a fresh install,
or a tier that has never held content). Every other verdict — including
:data:`KEYS_WITHOUT_SLOT` (active keys, no candidate at all) and
:data:`PAYLOAD_MISMATCH` (a bound slot whose payload bytes no longer hash to
the manifest's stamped digest) — leaves the tier unpublishable."""


# ---------------------------------------------------------------------------
# TierBinding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TierBinding:
    """One tier's registry↔slot-manifest verification result.

    Attributes:
        tier: PEFT adapter / tier name (e.g. ``"episodic"``,
            ``"episodic_interim_20260421T0400"``).
        tier_root: Resolved directory holding this tier's
            ``indexed_key_registry.json`` at its root.
        status: One of the module-level verdict constants.
        registry: The loaded :class:`~paramem.training.key_registry.KeyRegistry`,
            or ``None`` — ``None`` only when ``status == REGISTRY_UNREADABLE``;
            every other verdict carries a real (possibly empty) registry.
            Active/known key counts are derivable from this object
            (``list_active()`` / ``list_known()``) — not duplicated as
            separate fields here.
        registry_present: ``False`` when ``indexed_key_registry.json`` does
            not exist on disk for this tier.
        slot: The hash-matched live slot directory, when one was resolved
            (:data:`VERIFIED`, :data:`KEY_COUNT_MISMATCH`, and
            :data:`PAYLOAD_MISMATCH` only).
        manifest: The matched slot's already-parsed
            :class:`~paramem.adapters.manifest.AdapterManifest` — set
            whenever ``slot`` is set (:data:`VERIFIED`,
            :data:`KEY_COUNT_MISMATCH`, and :data:`PAYLOAD_MISMATCH`),
            ``None`` otherwise.
            :class:`~paramem.adapters.manifest.AdapterManifest` is a frozen
            dataclass, so holding onto this parsed object carries no
            mutability/lifetime hazard. Consumers that need the manifest
            (fingerprint checks, LoRA-shape comparison) read it from here —
            re-reading ``slot`` a second time would risk observing a
            DIFFERENT file than the one this verdict was computed from.
        candidate_count: Written payload slot candidate count from
            :func:`~paramem.adapters.manifest.count_slot_candidates` —
            informational; does not gate the verdict when a slot already
            matched.
        detail: Human-readable explanation, primarily populated for
            non-:data:`VERIFIED` verdicts.
    """

    tier: str
    tier_root: Path
    status: str
    registry: "KeyRegistry | None"
    registry_present: bool
    slot: "Path | None"
    manifest: "AdapterManifest | None"
    candidate_count: int
    detail: str

    @property
    def publishable(self) -> bool:
        """``True`` when this tier's registry is safe to publish (see
        :data:`_PUBLISHABLE`)."""
        return self.status in _PUBLISHABLE


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _candidate_count_tolerant(tier_root: Path) -> int:
    """``count_slot_candidates``, tolerating an OSError from an unreadable
    directory. Used only at step 1's :data:`REGISTRY_UNREADABLE` return,
    where the verdict has ALREADY been decided and this count is purely
    informational — a secondary I/O failure computing it must not crash a
    verdict that is otherwise fully formed. Step 3's own
    :data:`REGISTRY_UNREADABLE` return does NOT call this — by that point
    step 2 has already succeeded, so it reuses that already-computed count
    directly instead of re-deriving it."""
    try:
        return count_slot_candidates(tier_root)
    except OSError:
        return 0


def verify_tier_binding(tier: str, tier_root: Path) -> TierBinding:
    """Verify one tier's on-disk registry against its slot manifests.

    Resolution order (deliberate — the registry is read FIRST):

    0. A *tier_root* whose directory name starts with
       :data:`~paramem.training.donor.DONOR_STORE_PREFIX` raises
       ``ValueError`` — a donor store carries no registry by design and
       stamps ``key_count`` with a triple count against
       ``registry_sha256=""``; every verdict here would misclassify it.
    1. :meth:`~paramem.training.key_registry.KeyRegistry.load` the tier's
       ``indexed_key_registry.json``. Any raise →
       :data:`REGISTRY_UNREADABLE`. The registry is never inferred empty
       from a read failure; an absent file loads as a legitimately empty
       registry with ``registry_present=False``.
    2. :func:`~paramem.adapters.manifest.count_slot_candidates` on
       *tier_root*. An ``OSError`` here (e.g. the tier directory itself is
       unreadable) → :data:`NO_MATCHING_SLOT` — honest in the sense that
       "no confirmed match" is exactly what a caller can rely on when slot
       enumeration itself failed; the ``OSError`` text lands in ``detail``.
    3. :func:`~paramem.adapters.manifest.tier_registry_sha256` on
       *tier_root* → :data:`REGISTRY_UNREADABLE` on raise (an empty string
       for an absent file is not a failure — it is
       :func:`~paramem.adapters.manifest.find_live_slot`'s fresh-install
       match convention).
    4. **Registry-absent-with-candidates resolves BEFORE any slot-match
       attempt.** An absent registry's live hash (step 3) is always ``""``,
       and :func:`~paramem.adapters.manifest.find_live_slot` matches an
       empty hash against ANY candidate whose own ``meta.registry_sha256``
       is also ``""`` — the donor / fresh-install convention documented on
       ``find_live_slot`` itself. On a keyless boot (daily identity not
       loaded) that convention is indistinguishable from "no registry has
       ever existed here": both leave ``registry_present=False`` without
       needing any decrypt (an absent file never reads its own bytes; see
       steps 1 and 3). Binding via that empty-digest match here would carry
       the verdict through to step 7's payload read — the boot's FIRST
       decrypt — and a decrypt failure there resolves
       :data:`PAYLOAD_MISMATCH`, misdiagnosing a missing passphrase as
       corruption. So: ``registry_present is False`` AND
       ``candidate_count > 0`` → :data:`REGISTRY_ABSENT_WITH_SLOTS`,
       unconditionally, without ever calling ``find_live_slot``. This means
       ``find_live_slot``'s empty-digest convention can never actually bind
       a slot from INSIDE this function any more (see the "empty-digest
       convention" note below) — it remains live for callers that resolve
       their own slot directly (donor stores via the step-0 guard's
       exclusion, and other production call sites — ``backup.py``,
       ``active_store_migration.py``, ``memory/source.py``, ``app.py`` —
       that call ``find_live_slot``/``tier_registry_sha256`` themselves,
       outside this function's ambiguity).
    5. :func:`~paramem.adapters.manifest.find_live_slot` with the step-3
       hash — reached only when ``registry_present`` is ``True`` or
       ``candidate_count == 0`` (the empty-hash match trivially returns
       ``None`` with zero candidates, so step 4 does not need to special-case
       it). An ``OSError`` reading a candidate's ``meta.json`` during this
       scan (e.g. a mode-000 file) is not caught internally by
       ``find_live_slot`` and propagates here → :data:`NO_MATCHING_SLOT`,
       same reasoning as step 2. Otherwise: a match proceeds to step 6. No
       match with zero candidates and at least one active key in the
       registry → :data:`KEYS_WITHOUT_SLOT` (a tier that should have a slot
       has none). No match with zero candidates and no active key →
       :data:`NO_CANDIDATES` (genuinely nothing here yet). No match with
       candidates → :data:`NO_MATCHING_SLOT` (``registry_present`` is always
       ``True`` here — the absent+candidates shape already resolved at
       step 4).
    6. :func:`~paramem.adapters.manifest.read_manifest` on the matched slot;
       compare its ``key_count`` against ``len(registry.list_active())``
       when the stamp is an ``int``. Disagreement →
       :data:`KEY_COUNT_MISMATCH` (short-circuits — step 7 does not run).
       Agreement proceeds to step 7. An ``UNKNOWN`` stamp (or any
       non-``int`` stamp) is never treated as a mismatch — it is the
       migrated-slot state (:data:`~paramem.adapters.manifest.UNKNOWN`).
       Every MEMORY-TIER manifest writer stamps ``registry_sha256`` and
       ``key_count`` together from the same registry snapshot
       (:func:`~paramem.adapters.manifest.build_manifest_for`) — so an
       empty-hash match (the absent-registry / fresh-install convention)
       never carries a positive-int ``key_count`` in practice for a tier
       this function is legitimately asked to verify; only a hand-crafted
       manifest could disagree there, which is exactly what this comparison
       still catches. (The donor writer is the one production caller that
       stamps ``registry_sha256=""`` alongside a positive, non-registry-
       derived ``key_count`` — a triple count, not an active-key count —
       but a donor store never reaches this step: it is excluded by the
       step-0 guard.) ``ManifestNotFoundError``,
       ``ManifestSchemaError``, or a bare ``OSError`` reading the matched
       slot (a race — ``find_live_slot`` already skips unreadable
       manifests internally, so reaching one at this step means the file
       changed between its scan and this read, or is unreadable for a
       reason ``find_live_slot``'s narrower catch didn't cover) resolves
       to :data:`NO_MATCHING_SLOT`.
    7. The payload check: ``plaintext_sha256(slot / payload_filename(manifest.
       payload.kind))`` — :mod:`paramem.backup.hashing` is pure stdlib and
       imported at module scope, the same as
       :func:`~paramem.adapters.manifest.tier_registry_sha256`'s own
       module-scope import — against ``manifest.payload.sha256``.
       Runs for both venues alike: a ``train`` payload hashes
       ``adapter_model.safetensors``, a ``simulate`` payload hashes
       ``graph.json``. Always — every manifest that parses carries a digest
       computed from real bytes, by the write envelope or by the migration, so
       there is no skip arm. Agreement → :data:`VERIFIED`. Disagreement, a
       missing payload file, or any read failure (including a decrypt
       failure) → :data:`PAYLOAD_MISMATCH`, with the cause in ``detail``;
       both carry the parsed manifest on ``TierBinding.manifest`` and the
       resolved slot on ``TierBinding.slot``, same as
       :data:`KEY_COUNT_MISMATCH`. This step's decrypt is never the FIRST
       one a keyless boot attempts: a registry file that EXISTS but is
       undecryptable already fails at steps 1/3 → :data:`REGISTRY_UNREADABLE`;
       a registry that is genuinely ABSENT with candidates present is
       intercepted at step 4 → :data:`REGISTRY_ABSENT_WITH_SLOTS`, before
       any candidate is even considered a match. Only a tier whose registry
       reads (plaintext or decrypted) AND whose slot's own payload file is
       independently undecryptable reaches this step's decrypt first — a
       shape distinct from "no key loaded at all".

    See the module docstring for what this function does NOT verify, and
    for the totality guarantee (only the donor guard raises).

    Args:
        tier: PEFT adapter / tier name.
        tier_root: Resolved directory holding this tier's
            ``indexed_key_registry.json`` at its root (main tier:
            ``<adapter_dir>/<tier>/``; interim tier: the exact path
            :func:`~paramem.memory.interim_adapter.iter_interim_dirs`
            yielded).

    Returns:
        A :class:`TierBinding` describing the verdict.

    Raises:
        ValueError: *tier_root*'s directory name starts with
            :data:`~paramem.training.donor.DONOR_STORE_PREFIX`. No other
            exception propagates — see "Totality" in the module docstring.
    """
    tier_root = Path(tier_root)
    if tier_root.name.startswith(DONOR_STORE_PREFIX):
        raise ValueError(
            f"verify_tier_binding: {tier_root} is a donor store (name starts "
            f"with {DONOR_STORE_PREFIX!r}) — donor stores carry no registry "
            "by design and must never be verified as a memory tier"
        )

    registry_path = tier_root / "indexed_key_registry.json"
    registry_present = registry_path.exists()

    # Step 1: read the registry FIRST — deliberately, before any candidate
    # scan (see module docstring for why).
    try:
        registry = KeyRegistry.load(registry_path)
    except Exception as exc:  # noqa: BLE001
        return TierBinding(
            tier=tier,
            tier_root=tier_root,
            status=REGISTRY_UNREADABLE,
            registry=None,
            registry_present=registry_present,
            slot=None,
            manifest=None,
            candidate_count=_candidate_count_tolerant(tier_root),
            detail=f"registry load failed: {exc}",
        )

    # Step 2.
    try:
        candidate_count = count_slot_candidates(tier_root)
    except OSError as exc:
        return TierBinding(
            tier=tier,
            tier_root=tier_root,
            status=NO_MATCHING_SLOT,
            registry=registry,
            registry_present=registry_present,
            slot=None,
            manifest=None,
            candidate_count=0,
            detail=f"slot candidate enumeration failed: {exc}",
        )

    # Step 3.
    try:
        live_hash = tier_registry_sha256(tier_root)
    except Exception as exc:  # noqa: BLE001
        return TierBinding(
            tier=tier,
            tier_root=tier_root,
            status=REGISTRY_UNREADABLE,
            registry=None,
            registry_present=registry_present,
            slot=None,
            manifest=None,
            candidate_count=candidate_count,
            detail=f"registry hash failed: {exc}",
        )

    # Step 4: registry-absent-with-candidates resolves BEFORE any slot-match
    # attempt — see this function's docstring, step 4, for why: an absent
    # registry's live_hash is always "", and find_live_slot("") would match
    # ANY candidate stamped with the same empty-registry convention without
    # distinguishing "no registry has ever existed here" from "keyless boot,
    # registry undecryptable" — binding via that match would carry the
    # verdict to step 7's payload read, the boot's FIRST decrypt, and
    # misdiagnose a missing passphrase as PAYLOAD_MISMATCH corruption.
    if not registry_present and candidate_count > 0:
        return TierBinding(
            tier=tier,
            tier_root=tier_root,
            status=REGISTRY_ABSENT_WITH_SLOTS,
            registry=registry,
            registry_present=registry_present,
            slot=None,
            manifest=None,
            candidate_count=candidate_count,
            detail=(
                f"{candidate_count} candidate slot(s) present but no "
                "indexed_key_registry.json exists for this tier"
            ),
        )

    # Step 5.
    try:
        slot = find_live_slot(tier_root, live_hash)
    except OSError as exc:
        return TierBinding(
            tier=tier,
            tier_root=tier_root,
            status=NO_MATCHING_SLOT,
            registry=registry,
            registry_present=registry_present,
            slot=None,
            manifest=None,
            candidate_count=candidate_count,
            detail=f"slot resolution failed: {exc}",
        )

    if slot is None:
        if candidate_count == 0:
            active_count = len(registry.list_active())
            if active_count > 0:
                status = KEYS_WITHOUT_SLOT
                detail = (
                    f"registry holds {active_count} active key(s) but no "
                    "written payload slot candidate exists"
                )
            else:
                status = NO_CANDIDATES
                detail = "no written payload slot candidates on disk"
        else:
            # registry_present is always True here — the absent+candidates
            # shape already resolved REGISTRY_ABSENT_WITH_SLOTS at step 4,
            # above, before find_live_slot was ever called.
            status = NO_MATCHING_SLOT
            detail = (
                f"{candidate_count} candidate slot(s) present, none "
                "readable/matching the live registry hash"
            )
        return TierBinding(
            tier=tier,
            tier_root=tier_root,
            status=status,
            registry=registry,
            registry_present=registry_present,
            slot=None,
            manifest=None,
            candidate_count=candidate_count,
            detail=detail,
        )

    # Step 6.
    try:
        manifest = read_manifest(slot)
    except (ManifestNotFoundError, ManifestSchemaError, OSError) as exc:
        return TierBinding(
            tier=tier,
            tier_root=tier_root,
            status=NO_MATCHING_SLOT,
            registry=registry,
            registry_present=registry_present,
            slot=None,
            manifest=None,
            candidate_count=candidate_count,
            detail=f"live slot manifest failed to parse (race): {exc}",
        )

    active_count = len(registry.list_active())
    if isinstance(manifest.key_count, int) and manifest.key_count != active_count:
        return TierBinding(
            tier=tier,
            tier_root=tier_root,
            status=KEY_COUNT_MISMATCH,
            registry=registry,
            registry_present=registry_present,
            slot=slot,
            manifest=manifest,
            candidate_count=candidate_count,
            detail=(
                f"manifest key_count={manifest.key_count} disagrees with "
                f"registry active count={active_count}"
            ),
        )

    # Step 7: payload check — see this function's docstring, step 7.
    payload_path = slot / payload_filename(manifest.payload.kind)
    try:
        actual_digest = plaintext_sha256(payload_path)
    except Exception as exc:  # noqa: BLE001
        return TierBinding(
            tier=tier,
            tier_root=tier_root,
            status=PAYLOAD_MISMATCH,
            registry=registry,
            registry_present=registry_present,
            slot=slot,
            manifest=manifest,
            candidate_count=candidate_count,
            detail=f"payload read failed: {exc}",
        )

    if actual_digest != manifest.payload.sha256:
        return TierBinding(
            tier=tier,
            tier_root=tier_root,
            status=PAYLOAD_MISMATCH,
            registry=registry,
            registry_present=registry_present,
            slot=slot,
            manifest=manifest,
            candidate_count=candidate_count,
            detail=(
                f"payload digest {actual_digest} disagrees with manifest "
                f"payload.sha256={manifest.payload.sha256}"
            ),
        )

    return TierBinding(
        tier=tier,
        tier_root=tier_root,
        status=VERIFIED,
        registry=registry,
        registry_present=registry_present,
        slot=slot,
        manifest=manifest,
        candidate_count=candidate_count,
        detail="",
    )


def verify_adapter_tree(adapter_dir: Path) -> dict[str, TierBinding]:
    """Verify every memory tier under *adapter_dir* — main tiers + interims.

    Enumerates via
    :func:`~paramem.memory.interim_adapter.iter_tier_roots`, imported here
    inside the function body rather than at module scope: that module
    imports ``peft`` directly at its own module scope, and a module-scope
    import here would make ``paramem.adapters`` torch-dependent for every
    caller of this module. ``iter_tier_roots`` never yields a donor store,
    so :func:`verify_tier_binding`'s donor guard is never triggered by this
    walk.

    Args:
        adapter_dir: Adapter store root (``config.adapter_dir``).

    Returns:
        ``{tier_name: TierBinding}`` for every main tier (whether or not its
        directory exists) and every interim slot currently on disk.
    """
    from paramem.memory.interim_adapter import iter_tier_roots

    adapter_dir = Path(adapter_dir)
    return {
        tier: verify_tier_binding(tier, tier_root)
        for tier, tier_root in iter_tier_roots(adapter_dir)
    }


# ---------------------------------------------------------------------------
# Publish-time invariant: every tier must verify before the store publishes
# ---------------------------------------------------------------------------


class TierBindingUnpublishable(RuntimeError):
    """One or more tiers' registry↔slot-manifest bindings failed
    verification and cannot be published into the live memory store.

    There is no per-tier half-service at the store-publish boundary: a
    store is published in full or not at all, so a single unpublishable
    tier (:attr:`TierBinding.publishable` is ``False`` — every status
    except :data:`VERIFIED` and :data:`NO_CANDIDATES`) blocks the whole
    publish. The one raise site is :func:`raise_tier_binding_unpublishable`;
    every boundary that enforces this invariant raises through it rather
    than constructing this exception directly."""


def raise_tier_binding_unpublishable(tier_bindings: "dict[str, TierBinding]") -> NoReturn:
    """Raise :class:`TierBindingUnpublishable` naming every unpublishable tier.

    THE one formatting/raise helper for the store-publish binding
    invariant, mirroring
    :func:`~paramem.memory.store.raise_bookkeeping_invariant_violation`'s
    role for the bookkeeping-completeness invariant. The message names
    every tier that failed verification, together with its status and
    detail, so a caller that only logs ``str(exc)`` (or persists it as an
    incident cause) still records which tier and why.

    Args:
        tier_bindings: ``{tier: TierBinding}`` — the full walk from
            :func:`verify_adapter_tree`. Callers pass the whole map
            (not a pre-filtered subset); only the entries where
            ``binding.publishable`` is ``False`` are named in the raised
            message. Calling this with no unpublishable entry is a caller
            error — there is nothing to raise.

    Raises:
        TierBindingUnpublishable: Always — this function never returns.
    """
    unpublishable = {
        tier: binding for tier, binding in tier_bindings.items() if not binding.publishable
    }
    detail = ", ".join(
        f"{tier!r} ({binding.status}: {binding.detail})"
        for tier, binding in sorted(unpublishable.items())
    )
    raise TierBindingUnpublishable(f"tier registry binding unverified for publish: {detail}")
