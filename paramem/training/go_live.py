"""The joint go-live sequencer: takes a written bundle live, once.

:func:`publish_bundle` is the single publisher for both training events.
Every increment it is handed is already built and written; this module
computes nothing new — it orders the writes, mounts written weights, converges the
live store, reloads the router, reaps the absorbed ring, and reclaims disk
space, and nothing else.

Members never go live alone: every durable commit signal for a bundle
happens inside one call to this function, in the order publish -> mount ->
adopt -> reload -> reap -> ONE ledger write (the ``tier_live`` record) ->
prune.  The record is deliberately the LAST durable act before the prune:
it exists to record a go-live that already fully happened, on disk, in VRAM
and in RAM, never to promise one that is still in flight.  A crash before
the record leaves no ``tier_live`` entry at all, so a resume treats the
whole bundle as not-yet-live and re-publishes, re-mounts, re-adopts,
re-reloads and re-reaps — every one of those acts is individually
idempotent, so repeating the whole sequence is always safe.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:
    from peft import PeftModel

    from paramem.memory.increment import TierIncrement, TierWriteContext
    from paramem.server.router import QueryRouter
    from paramem.training.stage_ledger import StageLedger

logger = logging.getLogger(__name__)


def publish_bundle(
    bundle: "Sequence[TierIncrement]",
    *,
    ctx: "TierWriteContext",
    ledger: "StageLedger",
    written_slots: "Mapping[str, Path | None]",
    router: "QueryRouter | None" = None,
    absorbed_interim_tiers: "Sequence[str]" = (),
) -> "PeftModel":
    """Take a written bundle live.  Computes nothing new; every input is already built.

    Order: publish each member (destination tier first — the caller orders
    *bundle*) -> mount each written slot into its tier, one member at a time
    (each inside its own ``staged_weights`` scope) -> ONE ``adopt_increments``
    (converge; the absorbed interim tiers, if any, are dropped inside the
    same call) -> ONE ``router.reload()`` -> reap the absorbed interim slots
    whole (disk half) -> ONE atomic ledger write carrying the bundle's
    ``tier_live`` entries -> ``prune_old_slots`` per written member.

    Before any of that: :func:`~paramem.memory.persistence.assert_publish_preconditions`
    validates every bundle member's publish preconditions (pure reads only)
    and refuses the WHOLE bundle, with zero bytes written, when any member
    fails — see that function's own docstring for the precondition sets and
    the zero-bytes-on-refusal proof.

    The ledger write is deliberately placed AFTER the mount, the adoption,
    the reload and the reap: a ``tier_live`` entry records a go-live that has
    already fully landed — on disk, in VRAM (the mount) and in RAM (the
    adoption) — never one still in flight.  A crash at any point before the
    write leaves no ``tier_live`` entry for this bundle, so a resume
    classifies every member not-yet-live and repeats the whole sequence;
    every act above is individually idempotent (the preflight is a pure
    read and refuses identically on a repeat, ``publish_tier_registry``
    restamps identical bytes, the mount re-promotes the same written slot,
    ``adopt_increments`` re-converges to the same state, a second
    ``router.reload()`` costs nothing, and the ring reap is a no-op once the
    ring is already gone), so repeating it is always safe.

    Args:
        bundle: Membership derives from interdependency — a promotion
            couples ``episodic`` and ``semantic`` into one bundle; a
            self-contained tier goes live on its own. Every member of a
            bundle goes live together: at no observation point is one
            member live while another is not, since the ONE ledger write in
            step 6 is what makes any member observable at all, and it
            covers the whole bundle atomically. Order destination-first
            (``semantic`` before ``episodic``) regardless: this only
            matters during the unobservable in-flight window before that
            write, where a duplicate probe (the less severe intermediate)
            is preferable to a briefly unreachable fact — it never
            describes an outcome any observer may see, since a crash before
            the write leaves no ``tier_live`` entry and the whole bundle
            resumes as not-yet-live (see this function's own docstring).
        ctx: The named collaborator record — ``model`` and the per-tier
            ``AdapterConfig`` map for the mount, ``store`` for the
            adoption, ``output_dir`` and ``keep_prior_slots`` for the
            prune.
        ledger: The event's stage ledger (head fields already filled by
            phase 1). Its ``tier_live`` entries for this bundle are
            appended in one atomic write.
        written_slots: ``tier -> write_tier_slot's return``. A tier absent
            from this mapping (or mapped to ``None``) written no payload —
            it is published (rows + registry) but never mounted, and no
            slot is pruned for it.
        router: The live ``QueryRouter`` for the full event (threaded from
            ``consolidate(router=...)``); ``None`` for the interim event,
            whose app-layer finalizer owns the one reload it already
            performs.
        absorbed_interim_tiers: Interim tier names this bundle's go-live
            reaps whole. ``run_build_and_publish`` groups every not-yet-live
            tier of the event into ONE ordered bundle and makes exactly one
            ``publish_bundle`` call for it, passing ``ledger.absorbed_interim_tiers``
            on that call whenever it fires — non-empty for a full-topology
            event (``full`` or ``reconcile``) that has ring content to
            absorb, empty for an interim event (which never absorbs a
            ring) and empty on a resumed call that finds every tier
            already live (nothing left to publish, so ``publish_bundle``
            is not called at all). The RAM half (the absorbed tier's
            registry and entries, dropped inside the same
            ``adopt_increments`` window -- the former
            ``MemoryStore.drop_registry_and_entries`` primitive, now
            inlined there) and the disk half (``unload_interim_adapters``,
            unfiltered) both key off this same list.

    Returns:
        The model, possibly reassigned by the mount loop's
        ``ensure_adapter_matching`` call (an unwrapped-base cold birth) —
        ``ctx.model`` otherwise, since ``TierWriteContext`` is frozen and
        cannot carry the reassignment itself.  Callers that built ``ctx.model``
        from their own live reference must adopt this return value.
    """
    from paramem.adapters.manifest import read_manifest
    from paramem.memory.interim_adapter import adapter_slot_root_for_name
    from paramem.memory.persistence import (
        assert_publish_preconditions,
        prune_old_slots,
        publish_tier_registry,
    )
    from paramem.models.loader import _adapter_slot_for_load, ensure_adapter_matching
    from paramem.training.stage_ledger import (
        build_artifact_list,
        data_state_dir,
        tier_live_stage,
        write_stages,
    )
    from paramem.training.trainer import (
        STAGING_ADAPTER,
        assert_staging_absent,
        promote_staging_adapter,
        staged_weights,
    )

    # --- 0. Preflight -- validate every member's publish preconditions
    # BEFORE the first durable write; refuses the whole bundle, zero bytes
    # written, naming every failing member (see the function's own
    # docstring for the zero-bytes proof). ---
    assert_publish_preconditions(bundle=bundle, ctx=ctx, written_slots=written_slots)

    # --- 1. Publish (destination-first per the caller's ordering) ---
    for increment in bundle:
        publish_tier_registry(
            increment=increment, ctx=ctx, written_slot=written_slots.get(increment.tier)
        )

    # --- 2. Mount every written slot whose payload is PEFT weights.  Read
    # each member's own written-slot manifest (`payload.kind`) rather than
    # the ledger's event-level venue: the bundle already carries the answer
    # at the member level, so a "simulate" member (no weights to mount) is
    # skipped on its own account, never via an event-wide fork. ---
    # `ctx` is frozen (TierWriteContext takes no live-state reassignment) and
    # ensure_adapter_matching may reassign the model (an unwrapped-base cold
    # birth) — track the current model in a local across the loop rather
    # than writing back onto ctx.
    model = ctx.model
    for increment in bundle:
        written_slot = written_slots.get(increment.tier)
        if written_slot is None:
            continue  # rows-only member — nothing to mount
        if read_manifest(written_slot).payload.kind != "train":
            continue  # simulate-payload slot — no PEFT weights to mount
        # ctx.tier_configs is a total map over every tier this bundle
        # names — the driver (ConsolidationLoop.run_build_and_publish)
        # resolves each member's config, via the one rule home
        # (_tier_adapter_config), into ctx before calling here.
        tier_config = ctx.tier_configs[increment.tier]
        assert_staging_absent(model)
        with staged_weights(model, fallback_adapter=increment.tier):
            with _adapter_slot_for_load(written_slot) as load_path:
                model.load_adapter(str(load_path), adapter_name=STAGING_ADAPTER)
            model = ensure_adapter_matching(model, tier_config, increment.tier)
            promote_staging_adapter(model, increment.tier)
        logger.info("publish_bundle: mounted written slot for tier %s", increment.tier)

    # --- 3. ONE adopt_increments (converge the bundle AND drop the
    # absorbed ring's registry/entries/bookkeeping, all inside its one
    # lock -- no reader-visible window where a key is active in both the
    # bundle's destination tier and the not-yet-reaped interim tier) ---
    ctx.store.adopt_increments(bundle, absorbed_tiers=absorbed_interim_tiers)

    # --- 4. ONE router reload ---
    if router is not None:
        router.reload()

    # --- 5. Reap the absorbed interim slots whole (disk half) ---
    if absorbed_interim_tiers:
        from paramem.memory.interim_adapter import unload_interim_adapters

        unload_interim_adapters(model, ctx.output_dir)

    # --- 6. ONE atomic ledger write for the bundle's tier_live entries --
    # deliberately AFTER publish/mount/adopt/reload/reap: this write records
    # a go-live that has already fully happened (see this function's own
    # docstring for the crash-safety argument).
    completed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    live_entries = []
    for increment in bundle:
        tier_root = adapter_slot_root_for_name(ctx.output_dir, increment.adapter_name)
        artifact_paths = [
            tier_root / "key_metadata.json",
            tier_root / "indexed_key_registry.json",
        ]
        written_slot = written_slots.get(increment.tier)
        if written_slot is not None:
            # The write boundary (persistence.write_tier_slot, either venue)
            # always writes this manifest -- read directly, no existence
            # guard; an absent file here is a write defect, not a case to
            # tolerate. The condition is this member's own written_slots
            # entry, not the event's venue -- a member writes a slot in
            # both venues now, so the same rule covers both.
            artifact_paths.append(written_slot / "meta.json")
        live_entries.append(
            tier_live_stage(
                tier=increment.tier,
                completed_at=completed_at,
                artifacts=build_artifact_list(artifact_paths),
            )
        )
    write_stages(data_state_dir(ctx.output_dir.parent), ledger, live_entries)

    # --- 7. Prune prior slots, last — after the publish that bound the new
    # one, so find_live_slot never observes a pruned pair ---
    for increment in bundle:
        written_slot = written_slots.get(increment.tier)
        if written_slot is None:
            continue
        tier_root = adapter_slot_root_for_name(ctx.output_dir, increment.adapter_name)
        prune_old_slots(tier_root, written_slot, keep=ctx.keep_prior_slots)

    return model
