"""Bookkeeping-row rules: the one constructor and the one credit function.

:func:`bookkeeping_row` is the one seven-field bookkeeping-row constructor.
Every write of a key's row — store-side (``MemoryStore.set_bookkeeping``,
``MemoryStore.adopt_increments``) or fold-side (a working tier's mint or
replay) — shapes the row through it.  It canonicalises the speaker id and
refuses an empty one, making the no-unattributed-keys invariant executable
at the one place a row's shape is decided, rather than asserted at each
caller.

:func:`credit_reinforcement` is extracted whole out of
:meth:`~paramem.memory.store.MemoryStore.reinforce` so the rule has exactly
one implementation, usable both on the live store's rows (via
``MemoryStore.reinforce``, a locked delegation to this function) and on a
phase-1 working row set built off-store while a consolidation event is
staging.  It is a distinct concern from :func:`bookkeeping_row`: it
reconciles an EXISTING row against a scalar timestamp and carries the
earning rule (``reobserved and timestamp > prior_last_seen``), neither of
which applies to shaping a fresh row.

:func:`credit_reinforcement` is also distinct from
:func:`~paramem.graph.merger.reconcile_provenance`, which lives in
``paramem.graph.merger`` rather than here.  The two are separate rules over
separate artifacts and must not be collapsed into one: this module's
:func:`credit_reinforcement` reconciles a bookkeeping ROW against a scalar
timestamp and carries the earning rule that grows ``reinforcement_count``;
``reconcile_provenance`` reconciles a RELATION onto a graph target (a
merged edge or a node's attribute record) — first-non-empty-wins on
``speaker_id``/``edge_source``, ``max``/``min_nonempty`` on
``last_seen``/``first_seen`` — and never touches reinforcement count or
promotion state.  A bookkeeping row's own ``last_seen``/``first_seen`` are
populated FROM the graph target ``reconcile_provenance`` already
reconciled, not recomputed by it.
"""

from __future__ import annotations

from collections.abc import Iterable

from paramem.graph.merger import min_nonempty
from paramem.utils.identity import canonical, is_speaker_id


def bookkeeping_row(
    key: str,
    *,
    speaker_id: str,
    relation_type: str,
    first_seen: str,
    promoted: bool,
    reinforcement_count: int = 1,
    last_reinforced_cycle: int = 0,
    last_seen: str = "",
) -> dict:
    """Build the one canonical seven-field bookkeeping row for *key*.

    All seven fields are mandatory in the persisted schema (one mandatory
    tier, zero optional buckets).  ``first_seen`` and ``promoted`` carry no
    Python default — every caller must pass them explicitly.  The
    ``reinforcement_count``, ``last_reinforced_cycle``, and ``last_seen``
    params carry Python defaults solely as a legacy-fill convenience for
    new-key sites and boot-reload callers that do not yet know the values.
    The returned dict always contains all seven keys.

    ``speaker_id``: the speaker who first introduced this key.
    ``relation_type``: the model-assigned relation type from extraction
    (e.g. ``"factual"``, ``"preference"``, ``"temporal"``, ``"social"``).
    ``first_seen``: ISO 8601 wall-clock timestamp of the earliest session
    that contained this fact.  Paired with ``last_seen`` to give each fact
    its true assertion window ``[first_seen, last_seen]``.  Mandatory — no
    runtime fallback; callers must supply the real value.
    ``reinforcement_count``: the fact's durable maturity — how many
    separately-timed sightings it has accumulated, plus whatever it
    inherited from keys merged into it.  Only :func:`credit_reinforcement`
    may change it after minting; see there for the exact rule.  Default 1
    (a new key has been seen once).
    ``last_reinforced_cycle``: the most recent consolidation cycle at which
    this key's fact was reinforced (cycle counter; drives promotion/decay).
    Default 0 (unknown).
    ``last_seen``: ISO 8601 wall-clock timestamp of the most recent session
    that contained this fact.  Drives contradiction detection and temporal
    reasoning.  Default ``""`` (unknown).
    ``promoted``: whether this key has already been promoted from episodic
    to semantic.  Mandatory, no default — the flag sits on the key's own
    row, so the promotion cannot exist without the registry move it
    describes having been published with the same tier state.  Every new-key call
    site passes ``promoted=False``; the live writer that flips it to
    ``True`` is ``ConsolidationLoop._promote_working_keys`` (via the
    working row it hands to
    :meth:`~paramem.memory.store.MemoryStore.adopt_increments` at go-live)
    at the point it moves a key from episodic to semantic.

    ``speaker_id`` is canonicalized via
    :func:`~paramem.utils.identity.canonical` when it matches the
    ``speaker{N}`` pattern (``is_speaker_id``) so legacy-loaded and
    runtime-set data both match the casing the router's
    ``_speaker_key_index`` (:meth:`~paramem.server.router.QueryRouter.reload`,
    the sole privacy boundary) is built from: legacy cased ``Speaker0`` from
    a tier's ``key_metadata.json`` is silently coerced to ``speaker0`` at
    boot via the
    :meth:`~paramem.memory.store.MemoryStore.load_bookkeeping_from_disk` ->
    :meth:`~paramem.memory.store.MemoryStore.set_bookkeeping` path,
    self-healing on the next save.  Empty strings and non-speaker values
    pass through unchanged.

    Mutates nothing; returns a fresh dict on every call.

    Raises:
        ValueError: *speaker_id* is empty — unattributed keys are not
            recallable by speaker (no-unattributed-keys invariant).
    """
    if is_speaker_id(speaker_id):
        speaker_id = canonical(speaker_id)
    if not speaker_id:
        raise ValueError(
            f"bookkeeping_row: empty speaker_id for key {key!r} — unattributed "
            f"keys are not recallable by speaker (no-unattributed-keys invariant)."
        )
    return {
        "speaker_id": speaker_id,
        "relation_type": relation_type,
        "reinforcement_count": reinforcement_count,
        "last_reinforced_cycle": last_reinforced_cycle,
        "last_seen": last_seen,
        "first_seen": first_seen,
        "promoted": promoted,
    }


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
