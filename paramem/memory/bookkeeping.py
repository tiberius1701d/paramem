"""The one implementation of the reinforcement-credit rule.

:func:`credit_reinforcement` is extracted whole out of
:meth:`~paramem.memory.store.MemoryStore.reinforce` so the rule has exactly
one implementation, usable both on the live store's rows (via
``MemoryStore.reinforce``, a locked delegation to this function) and on a
phase-1 working row set built off-store while a consolidation event is
staging.
"""

from __future__ import annotations

from collections.abc import Iterable

from paramem.graph.merger import min_nonempty


def credit_reinforcement(
    rows: "dict[str, dict]",
    key: str,
    *,
    cycle: int,
    first_seen: str,
    timestamp: str = "",
    absorbed_counts: "Iterable[int]" = (),
    reobserved: bool = False,
) -> None:
    """Apply the reinforcement-credit rule to *rows* in place.

    The ONE implementation of ``max(own, *inherited) + earned`` and the
    timestamp earn condition.  Operates on any ``key -> row`` mapping — the
    live store's bookkeeping dict (through
    :meth:`~paramem.memory.store.MemoryStore.reinforce`) or a phase-1
    working set.  Holds no lock and knows no store: the caller owns
    concurrency control and persistence.

    The new count is::

        max(own_count, *absorbed_counts) + (1 if an independent sighting)

    **Inheritance** (*absorbed_counts*) is how a fold-time collapse preserves
    maturity: when several keys carrying the same fact are merged into one
    survivor, the survivor inherits the highest count of the group.  ``max``
    rather than a sum — the counter persists across folds, so summing would
    compound on every fold and manufacture promotions.  *absorbed_counts*
    are the durable counts of the keys being merged into *key*, resolved by
    the caller from whichever row set owns each of them — this function
    reads only *rows*, which holds the survivor.  A key merged from a
    DIFFERENT row set than *rows* (a cross-tier absorption) cannot be
    resolved from *rows* alone, which is why the caller resolves the counts
    rather than this function looking them up by name.

    **Earning** (*reobserved*) is how a key's count grows at all.  It adds
    exactly 1, and only when *timestamp* is strictly newer than the row's
    stored ``last_seen``: two sightings bearing the same session timestamp
    are the same transcript, and being mentioned twice in one conversation
    is multiplicity, not reinforcement.  An empty *timestamp* is "unknown",
    never evidence of a temporal gap, so it never earns.  An
    out-of-order *older* re-observation (*timestamp* older than the row's
    stored ``last_seen``) also never earns: the row already reflects a
    newer sighting, and ``last_seen`` only ever moves forward (the
    max-not-sum rule), so an older timestamp cannot move it and must not
    earn either.  This makes the rule idempotent unconditionally:
    re-processing an unchanged ledger earns zero regardless of call order.

    Every known key carries a bookkeeping row from the moment it is minted
    (the mint sites in :meth:`~paramem.training.consolidation.ConsolidationLoop`
    write the registry entry and the full seven-field row in the same
    block).  A reinforcement credit therefore always finds an existing row
    in *rows* — ``key`` absent from *rows* is a violation of that invariant,
    never a legitimate race to paper over, and raises via
    :func:`~paramem.memory.store.raise_bookkeeping_invariant_violation`
    rather than fabricating a placeholder row.

    Args:
        rows: The ``key -> row`` mapping to mutate in place.
        key: The indexed-key string (e.g. ``"graph42"``) — the survivor.
        cycle: Current consolidation cycle number.  Written to
            ``last_reinforced_cycle`` unconditionally.
        first_seen: ISO 8601 timestamp of the earliest session that
            (re-)contributed this fact.  When non-empty, sets
            ``first_seen = min_nonempty(existing_first_seen, first_seen)``.
        timestamp: ISO 8601 timestamp of the session that triggered this
            credit.  When non-empty, sets
            ``last_seen = max(existing_last_seen, timestamp)``.  Also the
            temporal-order evidence for *reobserved*.
        absorbed_counts: The durable ``reinforcement_count`` of every key
            being merged into *key*, already resolved by the caller.  An
            empty iterable contributes nothing.
        reobserved: ``True`` when this credit accompanies an independent
            sighting of the fact.  ``False`` for a normalization merge that
            rewrites how a fact is spelled without observing it again.

    Raises:
        ~paramem.memory.store.BookkeepingInvariantViolation: *key* has no
            row in *rows* — every known key already carries one by the time
            reinforcement credit runs.
    """
    existing = rows.get(key)
    if existing is None:
        from paramem.memory.store import raise_bookkeeping_invariant_violation

        raise_bookkeeping_invariant_violation(None, [key], "reinforcement credit")
    prior = existing.get("reinforcement_count", 1)
    inherited = max(absorbed_counts, default=0)
    prior_last_seen = existing.get("last_seen", "")
    earned = 1 if (reobserved and timestamp and timestamp > prior_last_seen) else 0
    count = max(prior, inherited) + earned

    existing["reinforcement_count"] = count
    existing["last_reinforced_cycle"] = cycle
    if timestamp:
        existing["last_seen"] = max(existing.get("last_seen", ""), timestamp)
    existing["first_seen"] = min_nonempty(existing.get("first_seen", ""), first_seen)
