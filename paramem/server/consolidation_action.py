"""The dispatch's pure decision layer: what an action may do, and whether
there is anything to do.

Split out of ``paramem.server.app`` so the vocabulary, the ``stages_event``
property, and the content gate are same-arguments-same-answer functions with
no access to ``_state``, no buffer mutation, and no disk touch beyond the
interim-slot listing the gate already does.  ``paramem.server.app`` imports
every name from here, so ``paramem.server.app.ConsolidationAction`` keeps
resolving for existing callers.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paramem.server.config import ServerConfig

logger = logging.getLogger(__name__)


class ConsolidationAction(str, Enum):
    """What a dispatch is asked to do — the internal vocabulary.

    Every run the server executes against the model is one of these,
    whether it was asked for by the schedule, by an operator's
    consolidation door, or by a calibration probe.  Nothing below the
    arbitrator knows who asked.

    ``AUTO`` is requested by ``/scheduled-tick`` (the timer's own door) and
    by the boot-completion catch-up task, dispatching in-process rather than
    through a REST call — nothing else requests it.  It resolves to ``FULL``
    or ``INTERIM`` in the arbitrator according to the schedule's own deadline
    math, and it alone carries the suspend/power-off catch-up gate and the
    cadence stamp.

    ``FULL`` and ``INTERIM`` are each requestable two ways — resolved from
    ``AUTO``, or requested directly by ``/consolidate`` and
    ``/consolidate/interim`` respectively — and the content gate applies to
    both paths identically: naming the action manually drops only the TIME
    condition (is a full/interim cycle due), never the CONTENT condition (is
    there anything to consume).

    - ``FULL`` — collapse the interim slots into the main tiers now.
    - ``INTERIM`` — absorb the pending conversations into a new interim slot.
    - ``RECONCILE`` — a full consolidation whose input excludes pending
      sessions.
    - ``CALIBRATE`` — a probe whose artifact the operator supplies (a
      transcript, a graph, a fact list, a turn list, an utterance).  The
      nine existing ``/calibrate/*`` routes request it.
    - ``CALIBRATE_PENDING`` — a probe whose artifact is the pending NAMED
      session set, i.e. exactly what a fold would take.  ``POST
      /calibrate/extract_pending`` requests it.
    """

    AUTO = "auto"
    FULL = "full"
    INTERIM = "interim"
    RECONCILE = "reconcile"
    CALIBRATE = "calibrate"
    CALIBRATE_PENDING = "calibrate_pending"

    @property
    def stages_event(self) -> bool:
        """Whether a run of this action stages a consolidation event.

        A staging run writes a stage-ledger record, mutates the memory
        store, and retires the pending sessions it consumed.  That
        authority is what the arbitrator's resume-first step, its
        tier-binding gate, its retiring triage pre-stage, its cadence stamp
        and its overdue incident are conditioned on, and what gives a run
        the right to call a retirement primitive or to write an
        extraction-time incident or attention row.
        """
        return self in _STAGING_ACTIONS


_STAGING_ACTIONS = frozenset(
    {
        ConsolidationAction.AUTO,
        ConsolidationAction.FULL,
        ConsolidationAction.INTERIM,
        ConsolidationAction.RECONCILE,
    }
)


def consolidation_content_gate(
    action: ConsolidationAction,
    config: "ServerConfig",
    *,
    pending_count: int,
    named_count: int,
    memory_store,
) -> "str | None":
    """The ONE "is there anything to run?" check for every action.

    Each action has its own input set, and dispatching one with an empty
    input seizes the GPU (or, for a calibration probe, holds the
    consolidation mutex) to do nothing:

    - ``INTERIM`` / ``CALIBRATE_PENDING`` — input is pending NAMED
      sessions.  With none there is nothing to extract and nothing to
      train or probe.
    - ``FULL`` — input is any payload-bearing interim slot ON DISK, checked
      regardless of the CURRENT ``max_interim_count``; only when no such
      slot exists does ``max_interim_count`` matter — at ``> 0`` there is
      nothing left to check and the gate noops; at ``== 0`` pending NAMED
      sessions are the fold's own content and the gate falls through to
      the shared check below.
    - ``RECONCILE`` — content is any active key in any registered tier,
      main or interim.  No live store yet is unprovable rather than empty,
      so the gate lets the dispatch proceed.
    - ``CALIBRATE`` — the operator supplied the artifact and validation
      already accepted it; there is nothing left to check, so this action
      can never noop.

    Args:
        action: ``FULL``, ``INTERIM``, ``RECONCILE``, ``CALIBRATE``, or
            ``CALIBRATE_PENDING`` — never ``AUTO`` (resolved before this is
            called).
        config: Live server config.
        pending_count: Pending sessions seen by the triage pre-stage,
            counted BEFORE retirement.
        named_count: How many of those classified NAMED (attributable).
        memory_store: The live ``MemoryStore``, or ``None`` when no store
            has been constructed yet.  Read only for ``RECONCILE``.

    Returns:
        A terminal ``"noop_*"`` string when there is nothing to run,
        ``None`` when the dispatch may proceed.
    """
    from paramem.memory.interim_adapter import iter_interim_dirs

    if action is ConsolidationAction.CALIBRATE:
        return None

    if action is ConsolidationAction.RECONCILE:
        if memory_store is None:
            return None
        if any(
            memory_store.active_keys_in_tier(tier) for tier in memory_store.tiers_with_registry()
        ):
            return None
        logger.info("Consolidation dispatch: no active keys in any tier — noop")
        return "noop_no_stored_keys"

    if action is ConsolidationAction.FULL:
        if any(iter_interim_dirs(config.adapter_dir, payload_only=True)):
            return None
        if config.consolidation.max_interim_count > 0:
            logger.info("Consolidation dispatch: no content-bearing interim slots — noop")
            return "noop_no_interim_slots"
        # max_interim_count == 0: no interim tier ever exists, so pending
        # NAMED sessions are this fold's own content -- fall through.

    # FULL at max_interim_count == 0, INTERIM (any count), and
    # CALIBRATE_PENDING (the pending NAMED set, exactly what a fold would
    # take) share the same input: pending sessions.
    if named_count > 0:
        logger.info("Consolidation dispatch: %d NAMED session(s) pending", named_count)
        return None

    if pending_count == 0:
        logger.info("Consolidation dispatch: no pending sessions — noop")
        return "noop_no_pending"
    logger.info("Consolidation dispatch: no NAMED sessions remain — noop")
    return "noop_no_named"
