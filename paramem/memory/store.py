"""Per-tier indexed-key memory store.

Public API: :class:`MemoryStore`.

Single source of truth for the answer content, integrity fingerprints, and
lifecycle registries of every indexed key the system holds in RAM.  Replaces
the previous mixed-shape state on :class:`ConsolidationLoop`:

* ``indexed_key_cache: dict[str, dict]`` — flat, all tiers in one bucket; tier
  was recovered indirectly by scanning every registry.
* ``episodic_simhash``, ``semantic_simhash``, ``procedural_simhash`` — three
  separate flat dicts; now folded into the per-tier :class:`KeyRegistry` (each
  registry carries ``_simhash: dict[str, int]`` — the tier's one fingerprint
  map, active keys only; a withheld id carries no fingerprint).
* ``indexed_key_registry: Optional[dict[str, KeyRegistry]]`` — the only
  structure that was already per-tier; folded in here for symmetry.

The unified shape is ``tier → key → value`` for all three concerns:

* :attr:`MemoryStore.entries_in_tier` returns ``dict[key, entry_payload]``.
* :meth:`MemoryStore.tier_simhashes` returns ``dict[key, int]`` — the tier's
  one fingerprint map, active keys only — the **only** public accessor for a
  fingerprint set.
* :meth:`MemoryStore.registry` returns the tier's :class:`KeyRegistry`.

Tier ownership of a key is the single source of truth; an indexed key
belongs to exactly one tier.  Cross-tier lookups (``get``, ``has``,
``tier_of``) scan tier-first then key — O(tier_count); the tier count is
small (3 main + N interim slots, typically ≤ 10).

The ``_registry`` dict is ALWAYS present (never ``None``) and lifecycle
recording is unconditional — the memory-key lifecycle registry has no
disabled state; a trained tier must always be provable.  ``registry()``
never returns ``None``.

**Content cache vs. bookkeeping — read before editing:**

:attr:`MemoryStore._entries` IS A NON-AUTHORITATIVE MIRROR of what the
venue serves — never validated for completeness, never a veto over go-live
or a turn.  On the SERVING path it has exactly two writers — the boot fill
(``app._build_store_contents``, gated on ``inference.preload_cache=True``)
and go-live adoption (:meth:`adopt_increments`, which installs a rebuilt
member's entries regardless of the setting) — and one reader,
:meth:`probe_cache`.  Outside serving, one sanctioned non-serving writer
survives on its own evidence — the active-store migration's own
``loop.store.put`` (``paramem.server.active_store_migration``), which
projects a converted tier's content straight into the mirror it is
migrating — and two read-only, non-serving consumers:
``GET /debug/dump`` (a zero-GPU operator inspection of whatever the mirror
currently holds) and
:func:`~paramem.memory.persistence.build_tier_graph_from_store` (re-projects
a tier's mirror content into a graph for persistence, not for a turn).
Neither adds a THIRD serving-path writer or reader; the two-writer,
one-reader count above is scoped to the doors.  A missing entry answers
``None`` at the cache door, the same no-fact shape as a live-door miss;
nothing treats a gap in it as a fault.  ``inference.preload_cache=False``
means the cache is plain off at serving — every serving read goes through
:meth:`probe_source` instead, and the mirror is neither read nor written on
that path.  The serving boundary
(:func:`paramem.server.inference._probe_and_reason`) forks once on
``inference.preload_cache`` and calls exactly one of :meth:`probe_cache` /
:meth:`probe_source` — never both, never layered.

The per-key bookkeeping fields live in :attr:`MemoryStore._bookkeeping` — a flat
``{key → {speaker_id, relation_type, reinforcement_count, last_reinforced_cycle,
last_seen, first_seen, promoted}}`` dict SEPARATE from ``_entries``.  Populated by
:meth:`load_bookkeeping_from_disk` at boot (unconditionally; entry-independent),
which merges each tier's own ``key_metadata.json`` — bookkeeping rows are a
per-tier file now, not a single global one.  Never enters
:meth:`KeyRegistry.save_bytes` or any hash path.

**Content-only invariant:**
Every ``_entries`` slot — whether written by the boot fill or by go-live
adoption (:meth:`adopt_increments`) — carries exactly
``{key, subject, predicate, object}``, projected via
:func:`~paramem.memory.entry.content_only_entry` by every writer before the
entry reaches the store — the boot fill, the increment build feeding go-live
adoption (``paramem.memory.increment.build_tier_increment``), and the
migration's ``store.put`` site.  Per-key provenance (``speaker_id``, ``relation_type``, ...) lives
exclusively in :attr:`MemoryStore._bookkeeping`:
:func:`~paramem.memory.persistence.build_tier_graph_from_store` reads
``speaker_id`` from ``store.bookkeeping_for_key``, never from the entry.

All ``store.get`` readers in ``consolidation.py`` are SPO-only readers, not
bookkeeping sites; a fresh fold and a resumed fold write the identical entry
shape.

**SimHash storage:**
SimHash fingerprints live exclusively in :class:`KeyRegistry` (one per tier),
one fingerprint map (``registry._simhash``) for the tier's active keys — a
withheld id carries no fingerprint.  Serialised to
``indexed_key_registry.json`` under the ``"simhash"`` key so the on-disk file
is the single source of truth.  The separate ``simhash_registry.json``
sidecar has been eliminated.  Use :meth:`tier_simhashes` as the only public
path to a fingerprint set.

**Thread-safety concurrency contract:**

:class:`MemoryStore` is shared between the asyncio event-loop thread, FastAPI
handler threads (ThreadPoolExecutor), and the BG-trainer worker thread.  A
single :class:`threading.RLock` (``self._lock``) guards all accesses to the
three mutable structures: ``_entries``, ``_registry``, and ``_bookkeeping``.

A second, separate lock (``_SIMHASH_REGISTRY_CACHE_LOCK``, module-level, a
plain :class:`threading.Lock`) guards the module-level opt-in registry
cache described just above the class (``_SIMHASH_REGISTRY_CACHE`` and its
generation counter ``_SIMHASH_REGISTRY_CACHE_GENERATION``). This is process
state, not instance state — it is not covered by ``self._lock`` and does
not participate in any :class:`MemoryStore` instance's locking. See rule 7.

Rules for callers and maintainers:

1. Every method that reads or writes any of the three structures holds
   ``self._lock`` for the duration of its in-RAM access.  RLock (not plain Lock)
   is used because compound mutators (``discard_keys``, compound ``put``)
   call other wrapped leaf methods reentrantly.

2. ``iter_entries()`` and ``iter_bookkeeping()`` materialise a snapshot list
   under the lock, then yield from the snapshot outside the lock.  Callers
   iterate lock-free without risk of observing a concurrent structural mutation.

3. Compound mutators (``discard_keys``,
   ``put`` when writing entry + simhash + registry) hold the lock ONCE around
   the whole compound so a reader never sees a half-updated multi-structure state.
   Nested leaf-method calls succeed via RLock reentrancy.

4. ``entries_in_tier()`` returns a shallow copy of the internal tier dict so
   callers that iterate the result outside the lock cannot observe a concurrent
   structural mutation.  Callers must not write to the returned dict; use
   :meth:`put` for writes.

5. :meth:`probe_source` is deliberately NOT wrapped in the lock.  It calls
   ``source.probe(...)``, which is a GPU ``model.generate()`` call that may
   block for seconds.  Holding the store lock across GPU work would deadlock
   the event loop.  Its one read inside the loop — the confidence gate's
   fingerprint lookup — goes through individually locked leaf methods
   (``self.simhash``, ``self._tier_for_simhash``).  Each read is a single
   locked acquisition; no lock is held across the GPU call, and the door
   writes nothing (the cache is neither read nor written on this path).
   :meth:`probe_cache` is a registry-scoped lookup: one locked,
   NON-CREATING ``self._registry.get(tier)`` read per requested tier (never
   :meth:`registry`'s ``setdefault`` — a door must not phantom-register an
   unknown tier), then the already-locked :meth:`get` per key.  It holds no
   lock across multiple keys, matching :meth:`entries_in_tier`'s per-key
   locking rather than a single compound acquisition.

6. :meth:`swap` is the atomic whole-store rebind used by boot hydration
   (and its quarantine-lift retry, the identical re-runnable primitive):
   the caller builds the replacement ``_entries``/``_registry``/
   ``_bookkeeping`` fully off-store, then publishes them in one locked
   rebind so no reader ever sees a torn/half-rebuilt store.

7. :meth:`adopt_increments` is the atomic whole-bundle go-live: ONE lock
   acquisition covers the written bundle's increments and any absorbed
   interim tiers together, across its own three internal passes (check,
   drop, install) — no reader ever observes a fact live in both a
   bundle's destination tier and the not-yet-reaped interim tier it
   moved out of.

8. :meth:`read_simhash_registry_from_disk` (a ``@staticmethod``, no
   ``self._lock`` involved) reads ``_SIMHASH_REGISTRY_CACHE`` under
   ``_SIMHASH_REGISTRY_CACHE_LOCK`` for its cache-hit check, then performs
   its disk walk lock-free (the walk is not an in-RAM structure access).
   Publishing the walked result back into the cache re-acquires the lock
   and re-checks ``_SIMHASH_REGISTRY_CACHE_GENERATION`` against the value
   snapshotted before the walk: an :func:`invalidate_simhash_registry_cache`
   call that lands mid-walk bumps the generation, so the walking caller's
   publish is a no-op and the walked (now-known-stale) result is still
   returned to that one caller but never enters the cache. This is
   structural, not a serialization convention callers must uphold.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, NoReturn

from paramem.memory.bookkeeping import bookkeeping_row, credit_reinforcement
from paramem.training.key_registry import KeyRegistry

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paramem.memory.increment import TierIncrement

logger = logging.getLogger(__name__)


class BookkeepingInvariantViolation(RuntimeError):
    """A key known to a tier's registry — active or stale — has no
    bookkeeping row, two registries contradict each other about which one
    owns a key, or a withheld (stale) id is asked to re-enter a registry's
    active set or fingerprint map.

    Every known key carries a full bookkeeping row from the moment it is
    formed, through every artifact (working copy, shadow files, increment,
    live store, persisted per-tier files, bundles) and through retirement.
    A registry-known key without a row, or a key two registries both claim,
    is never tolerated and never repaired in code — it is raised here, at
    the boundary that meets it.  A withheld id is likewise never tolerated
    back into active standing: :meth:`~paramem.training.key_registry.KeyRegistry.add`
    and :meth:`~paramem.training.key_registry.KeyRegistry.set_simhash` refuse
    a withheld id through the shared
    :meth:`~paramem.training.key_registry.KeyRegistry._refuse_withheld`
    guard, which raises here too.
    :meth:`~paramem.training.key_registry.KeyRegistry.replace_simhashes`
    refuses a whole map naming any withheld id through this same raise
    helper directly — a set-level check rather than a per-key
    ``_refuse_withheld`` call, so one raise names every offending id. The
    one raise site is
    :func:`raise_bookkeeping_invariant_violation`; every boundary that
    enforces this invariant raises through it rather than constructing this
    exception directly — including the pre-write parity gate
    (:meth:`~paramem.training.consolidation.ConsolidationLoop._assert_increment_registry_bookkeeping_parity`),
    which is this same invariant, checked against an increment's shadow
    artifacts rather than the live store.

    Attributes:
        divergent_keys: The offending tier → key-list map, as raised.  A
            crash-envelope caller (e.g.
            ``paramem.server.app._run_stage_b_cycle``) folds this into a
            recorded incident's detail so the incident names exactly what
            diverged, not just the log traceback.
    """

    def __init__(self, message: str, *, divergent_keys: "dict[str, list[str]]"):
        super().__init__(message)
        self.divergent_keys = divergent_keys


def raise_bookkeeping_invariant_violation(
    tier: "str | None", keys: "Iterable[str]", context: str
) -> NoReturn:
    """Raise :class:`BookkeepingInvariantViolation` naming *tier* and *keys*.

    THE one formatting/raise helper for the every-known-key-has-a-row
    invariant.  Every boundary that establishes or preserves the invariant
    — mint, promotion, fold recall, fold write, bundle capture, boot load,
    reinforcement credit, registry key adoption, withheld-id refusal (a
    withheld id asked to re-enter a registry's active set or fingerprint
    map), pre-write parity — raises
    through this helper instead of hand-rolling its own message, so every
    violation names the same three things: which boundary caught it, which
    tier (when one applies), and which key(s); and carries the same
    structured ``divergent_keys`` payload (``{tier_desc: key_list}``) on the
    raised exception for a crash-envelope caller to fold into an incident.

    Args:
        tier: The tier the violation was found under, or ``None`` when the
            boundary has no single owning tier to name (e.g. a
            cross-registry key-adoption contradiction, or a reinforcement
            credit applied to a row set with no tier of its own) — the
            exception's ``divergent_keys`` then keys on the literal string
            ``"an unnamed tier"`` rather than a real tier name.
        keys: The offending key or keys — always rendered as a list in the
            message and in ``divergent_keys``, even when there is exactly
            one.
        context: A short phrase naming the boundary/check that raised
            (e.g. ``"fold recall"``, ``"boot bookkeeping load"``,
            ``"pre-write parity"``).

    Raises:
        BookkeepingInvariantViolation: Always — this function never returns.
    """
    key_list = list(keys)
    tier_desc = f"tier {tier!r}" if tier is not None else "an unnamed tier"
    raise BookkeepingInvariantViolation(
        f"{context}: {tier_desc}, key(s) {key_list!r}",
        divergent_keys={(tier if tier is not None else "an unnamed tier"): key_list},
    )


class EntryCacheInvariantViolation(RuntimeError):
    """A written increment's own registry calls a key active but its own
    ``entries`` (the materialized keyed-list projection) carries no payload
    for it.

    Increment-internal only: :func:`~paramem.memory.increment.build_tier_increment`
    reads a rebuilt member's ``indexed_key_registry.json`` and ``keyed.json``
    as two independent files, so this catches a genuine divergence between
    them before that increment's entries would install into the live
    mirror — never a claim about the live store's current ``_entries``,
    which :meth:`~paramem.memory.store.MemoryStore.adopt_increments` never
    reads.

    A SEPARATE invariant from :class:`BookkeepingInvariantViolation` — an
    active key without a bookkeeping row and an active key without a
    materialized entry are two different completeness checks, and can fail
    independently.  The one raise site is
    :func:`raise_entry_cache_invariant_violation`; every boundary that
    enforces this invariant raises through it rather than constructing this
    exception directly.

    Attributes:
        tier: The tier the violation was found under.
        missing_keys: The active key(s) with no entry-cache payload.
    """

    def __init__(self, message: str, *, tier: str, missing_keys: "list[str]"):
        super().__init__(message)
        self.tier = tier
        self.missing_keys = missing_keys


def raise_entry_cache_invariant_violation(
    tier: str, keys: "Iterable[str]", context: str
) -> NoReturn:
    """Raise :class:`EntryCacheInvariantViolation` naming *tier* and *keys*.

    THE one formatting/raise helper for the active-key-has-an-entry
    invariant — the entry-cache counterpart to
    :func:`raise_bookkeeping_invariant_violation`.

    Args:
        tier: The tier the violation was found under.
        keys: The active key(s) missing an entry-cache payload — always
            rendered as a list in the message, even when there is exactly
            one.
        context: A short phrase naming the boundary/check that raised
            (e.g. ``"adopt_increments"``).

    Raises:
        EntryCacheInvariantViolation: Always — this function never returns.
    """
    key_list = list(keys)
    raise EntryCacheInvariantViolation(
        f"{context}: tier {tier!r} would violate entry-cache completeness — "
        f"active key(s) with no entry: {key_list!r}",
        tier=tier,
        missing_keys=key_list,
    )


# Process-wide cache for the merged simhash registry produced by
# ``MemoryStore.read_simhash_registry_from_disk(..., cached=True)``.  Keyed
# by the adapter_dir's resolved path.  Guarded by
# ``_SIMHASH_REGISTRY_CACHE_LOCK`` so concurrent request-handler threads
# never observe a half-written entry.  Opt-in only: every existing caller
# keeps reading disk-truth on every call (``cached`` defaults to False);
# only the per-turn inference probe (``paramem.server.inference``) opts in.
#
# ``_SIMHASH_REGISTRY_CACHE_GENERATION`` is bumped by every
# ``invalidate_simhash_registry_cache`` call.  ``read_simhash_registry_from_disk``
# snapshots the generation before its disk walk and only publishes the
# walked result if the generation is still unchanged at publish time — an
# invalidation landing mid-walk is then structurally discarded rather than
# depending on the caller to serialize against a concurrent invalidator.
#
# ``QueryRouter.reload`` (``paramem.server.router``) is the caller that
# keeps this cache fresh for fold finalize, erase-keys, speaker-forget, and
# interim-discard — every one of those ends in a router reload. The
# active-store-migration finalizer is the one registry-mutating path that
# does NOT reload the router; see its own comment
# (``paramem.server.router.QueryRouter.reload``) for why that is safe.
_SIMHASH_REGISTRY_CACHE_LOCK = threading.Lock()
_SIMHASH_REGISTRY_CACHE: dict[str, dict[str, int]] = {}
_SIMHASH_REGISTRY_CACHE_GENERATION = 0


def invalidate_simhash_registry_cache() -> None:
    """Drop every cached merged simhash registry populated by ``cached=True`` reads.

    Clears the whole process-wide cache and bumps
    ``_SIMHASH_REGISTRY_CACHE_GENERATION`` — the sole caller,
    :meth:`paramem.server.router.QueryRouter.reload`, has no single
    ``adapter_dir`` of its own to target, so there is no per-adapter_dir
    invalidation arm to expose.  The generation bump is what lets
    :meth:`MemoryStore.read_simhash_registry_from_disk` detect and discard
    a walk that raced this call (see its docstring).
    """
    global _SIMHASH_REGISTRY_CACHE_GENERATION
    with _SIMHASH_REGISTRY_CACHE_LOCK:
        _SIMHASH_REGISTRY_CACHE.clear()
        _SIMHASH_REGISTRY_CACHE_GENERATION += 1


class MemoryStore:
    """Per-tier {entries, simhash, registry} for the indexed-key memory layer.

    ``_registry`` is always a ``dict[str, KeyRegistry]`` (never ``None``);
    lifecycle recording is unconditional.

    Thread-safety: a single ``threading.RLock`` (``self._lock``) guards all
    in-RAM access to ``_entries``, ``_registry``, and ``_bookkeeping``.  See
    the module-level concurrency contract in the module docstring for the
    complete rules including the deliberately unwrapped :meth:`probe_source`
    and the atomic :meth:`swap` publish primitive.
    """

    def __init__(self) -> None:
        # Single RLock guards _entries, _registry, and _bookkeeping.  RLock
        # (not plain Lock) is required because compound mutators call wrapped
        # leaf methods reentrantly (discard_keys→tiers_with_registry,
        # adopt_increments's own check/drop/install passes, etc.).
        self._lock = threading.RLock()
        # tier -> key -> entry payload dict.  PURE CONTENT CACHE — an entry
        # slot exists only when SPO is materialised, and it is always
        # content-only ({key, subject, predicate, object}).  See the module
        # docstring's "Content-only invariant" section.
        self._entries: dict[str, dict[str, dict]] = {}
        # tier -> KeyRegistry — ALWAYS present (never None); key lifecycle
        # recording is unconditional.  SimHash fingerprints live ON the
        # registry (not on MemoryStore).
        self._registry: dict[str, KeyRegistry] = {}
        # Per-key provenance bookkeeping — SEPARATE from _entries.
        # key -> {"speaker_id": str, "relation_type": str,
        #         "reinforcement_count": int, "last_reinforced_cycle": int,
        #         "last_seen": str, "first_seen": str}
        # Populated by load_bookkeeping_from_disk at boot.  Never enters
        # KeyRegistry.save_bytes — stays out of the hash-frozen slot-identity
        # path.
        self._bookkeeping: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Entry payload — read
    # ------------------------------------------------------------------
    def get(self, key: str) -> dict | None:
        """Return the entry payload for *key* (any tier), or ``None`` on miss.

        Scan order is the insertion order of tiers; for cross-tier ambiguity
        the first matching tier wins.  Per the single-tier-ownership
        invariant, ambiguity should not arise — surface it as a bug if it
        does (do not silently coalesce)."""
        with self._lock:
            for tier_entries in self._entries.values():
                if key in tier_entries:
                    return tier_entries[key]
            return None

    def has(self, key: str) -> bool:
        """Membership check across all tiers."""
        with self._lock:
            for tier_entries in self._entries.values():
                if key in tier_entries:
                    return True
            return False

    def tier_of(self, key: str) -> str | None:
        """Return the tier that owns *key* in the entry store, or ``None``."""
        with self._lock:
            for tier, tier_entries in self._entries.items():
                if key in tier_entries:
                    return tier
            return None

    def entries_in_tier(self, tier: str) -> dict[str, dict]:
        """Return a shallow copy of the ``key -> entry`` map for *tier*.

        Returns an empty dict when *tier* is absent.  The returned dict is a
        snapshot taken under the lock — it is safe to iterate after the method
        returns even if concurrent mutations occur.  Use :meth:`put` for
        writes; do not mutate the returned dict."""
        with self._lock:
            return dict(self._entries.get(tier, {}))

    def iter_entries(self) -> Iterator[tuple[str, str, dict]]:
        """Yield ``(tier, key, entry)`` for every entry.

        A snapshot of the full entry set is taken under the lock before the
        first yield.  The caller iterates the snapshot lock-free and will not
        observe concurrent structural mutations (insertions or deletions by
        another thread after this method returns)."""
        with self._lock:
            snap = [
                (tier, key, entry)
                for tier, tier_entries in self._entries.items()
                for key, entry in tier_entries.items()
            ]
        yield from snap

    def __len__(self) -> int:
        """Total number of keys held across all tiers."""
        with self._lock:
            return sum(len(tq) for tq in self._entries.values())

    def __contains__(self, key: str) -> bool:
        return self.has(key)

    # ------------------------------------------------------------------
    # Entry payload — write
    # ------------------------------------------------------------------
    def put(
        self,
        tier: str,
        key: str,
        entry: dict,
        *,
        simhash: int | None = None,
        register: bool = True,
    ) -> None:
        """Store *entry* for *key* under *tier*.

        If *simhash* is supplied, the per-tier fingerprint is written to the
        tier's :class:`KeyRegistry` in the same call.
        When *register* is True, the key is added to the tier's lifecycle
        registry.

        Caller is responsible for ensuring *key* is unique across tiers —
        cross-tier key movement (e.g. promotion) goes through
        :meth:`adopt_increments`, which converges a whole bundle atomically,
        not through calling ``put`` on the new tier while the key is still
        registered on the old one.

        All three writes (entry, simhash, registry) are performed under a
        single lock acquisition so a reader never observes a half-updated state.
        """
        with self._lock:
            self._entries.setdefault(tier, {})[key] = entry
            if simhash is not None:
                self._registry.setdefault(tier, KeyRegistry()).set_simhash(key, simhash)
            if register:
                self._registry.setdefault(tier, KeyRegistry()).add(key)

    # ------------------------------------------------------------------
    # Per-key bookkeeping — speaker_id / relation_type / reinforcement_count
    #                       / last_reinforced_cycle / last_seen / first_seen
    # SEPARATE from _entries; never in KeyRegistry.save_bytes.
    # ------------------------------------------------------------------
    def set_bookkeeping(
        self,
        key: str,
        *,
        speaker_id: str,
        relation_type: str,
        first_seen: str,
        promoted: bool,
        reinforcement_count: int = 1,
        last_reinforced_cycle: int = 0,
        last_seen: str = "",
    ) -> None:
        """Store or replace the bookkeeping record for *key*.

        Builds the row via :func:`~paramem.memory.bookkeeping.bookkeeping_row`
        — see there for the seven-field contract, the speaker
        canonicalisation, and the empty-speaker ``ValueError`` (there is no
        carve-out here: every call site, including boot reload via
        :meth:`load_bookkeeping_from_disk`, must supply a real speaker_id —
        the no-unattributed-keys invariant).

        **Callers that need to update ONE field on an existing key must use
        :meth:`reinforce` (for reinforcement/last_seen/first_seen updates)
        rather than calling this method, which overwrites ALL seven fields
        and would silently reset the counters the caller did not supply.
        The promotion flag has no store-level setter of its own — the live
        promotion path (``ConsolidationLoop._promote_working_keys``) sets
        ``row["promoted"] = True`` directly on the working tier's row dict,
        and the flag reaches the live store's own bookkeeping only via the
        increment's install at go-live (:meth:`adopt_increments`).**

        Does NOT touch ``_entries`` — bookkeeping presence MUST NOT
        manufacture a content cache hit.

        Raises:
            ValueError: propagated from :func:`~paramem.memory.bookkeeping.bookkeeping_row`
                when ``speaker_id`` is empty.
        """
        row = bookkeeping_row(
            key,
            speaker_id=speaker_id,
            relation_type=relation_type,
            first_seen=first_seen,
            promoted=promoted,
            reinforcement_count=reinforcement_count,
            last_reinforced_cycle=last_reinforced_cycle,
            last_seen=last_seen,
        )
        with self._lock:
            self._bookkeeping[key] = row

    def reinforce(
        self,
        key: str,
        *,
        cycle: int,
        first_seen: str,
        timestamp: str = "",
        absorbing: "Iterable[str]" = (),
        reobserved: bool = False,
    ) -> None:
        """Update *key*'s ``reinforcement_count`` and refresh cycle/timestamps.

        This is the ONLY correct way to update the reinforcement counter on an
        existing bookkeeping record without resetting unrelated fields.  Callers
        that use :meth:`set_bookkeeping` to update a single field would silently
        overwrite ``speaker_id`` / ``relation_type`` with defaults.

        The new count is::

            max(own_count, *absorbed_counts) + (1 if an independent sighting)

        **Inheritance** (``absorbing``) is how a fold-time collapse preserves
        maturity.  When several keys carrying the same fact are merged into one
        survivor, the survivor inherits the highest count of the group; the
        keys left behind are staled and their counts would otherwise be
        discarded, silently demoting a promoted fact back to episodic.  ``max``
        rather than a sum: the counter persists in the registry across folds, so
        summing would compound on every fold and manufacture promotions.
        Under-counting only delays a promotion; over-counting corrupts the tier.

        **Earning** (``reobserved``) is how a key's count grows at all.  It
        adds exactly 1, and only when *timestamp* differs from the record's
        stored ``last_seen``: two sightings bearing the same session timestamp
        are the same transcript, and being mentioned twice in one conversation
        is multiplicity, not reinforcement.  An empty *timestamp* is "unknown",
        never evidence of a temporal gap, so it never earns.

        Every known key already carries a bookkeeping record by the time
        this is called — :meth:`set_bookkeeping` (or a mint site's direct
        row write) always runs first.  ``key`` absent from
        ``self._bookkeeping`` is a violation of that invariant and raises
        :class:`BookkeepingInvariantViolation` (via
        :func:`credit_reinforcement`), never a fabricated placeholder.

        Args:
            key: The indexed-key string (e.g. ``"graph42"``) — the survivor.
            cycle: Current consolidation cycle number.  Written to
                ``last_reinforced_cycle`` unconditionally (the fact was
                re-seen this cycle).
            first_seen: ISO 8601 wall-clock timestamp of the earliest session
                that (re-)contributed this fact.  Mandatory — no runtime
                fallback.  When non-empty, sets ``first_seen =
                min_nonempty(existing_first_seen, first_seen)`` so the window
                start never regresses forward; an empty string never wins a
                ``min_nonempty`` (it means "unknown", not "earliest possible
                time").  Pass the real value from the boundary that has it.
            timestamp: ISO 8601 wall-clock timestamp of the session that
                triggered this credit.  When non-empty, sets ``last_seen =
                max(existing_last_seen, timestamp)`` — ISO-8601 strings sort
                lexicographically so ``max`` is chronological; this never
                regresses an existing newer value.  When empty, ``last_seen``
                is preserved unchanged.  Pass the real session timestamp from
                the boundary that has it; do NOT fabricate a ``now()`` here.
                Also the temporal-order evidence for ``reobserved``.
            absorbing: Keys whose durable counts *key* inherits — the keys
                being merged into it.  Resolved against this store's own
                bookkeeping dict, under this store's own lock, before the
                credit is applied.  Unknown keys contribute nothing.
            reobserved: ``True`` when this credit accompanies an independent
                sighting of the fact (a duplicate-SPO collapse across sessions,
                or a recited fact adopting an existing key).  ``False`` for a
                normalization merge such as a predicate-synonym collapse, which
                rewrites how a fact is spelled without observing it again.

        Raises:
            BookkeepingInvariantViolation: *key* has no bookkeeping record —
                see above.
        """
        with self._lock:
            absorbed_counts = [
                self._bookkeeping[k].get("reinforcement_count", 1)
                for k in absorbing
                if k in self._bookkeeping
            ]
            credit_reinforcement(
                self._bookkeeping,
                key,
                cycle=cycle,
                first_seen=first_seen,
                timestamp=timestamp,
                absorbed_counts=absorbed_counts,
                reobserved=reobserved,
            )

    def bookkeeping_for_key(self, key: str) -> dict | None:
        """Return the bookkeeping record for *key*.

        Returns a plain dict with seven fields:
        ``{"speaker_id", "relation_type", "reinforcement_count",
        "last_reinforced_cycle", "last_seen", "first_seen", "promoted"}``.

        Every registry-known key carries a bookkeeping row from the moment
        it is formed (the every-known-key-has-a-row invariant enforced at
        the store boundaries via ``BookkeepingInvariantViolation`` —
        see :func:`raise_bookkeeping_invariant_violation`), so a call for
        any key drawn from this store's own registry or entry cache
        (``iter_entries()``, ``registry(tier).list_known()``, ...) always
        returns the dict, never ``None``.  Read sites access the return
        value directly (splatting it, indexing a field) without an
        ``or {}`` default.

        ``None`` is reserved for a key that was never bookkept at all —
        e.g. an arbitrary string a caller did not first resolve against
        this store's own known-key set.  For any key that DID come from
        this store, a ``None`` here means the invariant has already been
        violated upstream; it is not a state a caller should tolerate or
        silently default around."""
        with self._lock:
            return self._bookkeeping.get(key)

    def iter_bookkeeping(self):
        """Yield ``(key, {"speaker_id": ..., "relation_type": ..., ...})`` pairs.

        Preload-independent — populated by :meth:`load_bookkeeping_from_disk`
        at boot regardless of ``inference.preload_cache``.  Used by
        :meth:`QueryRouter.reload` to build the speaker → keys index without
        touching ``_entries``.

        A snapshot of the bookkeeping dict is taken under the lock before the
        first yield.  The caller iterates the snapshot lock-free and will not
        observe concurrent structural mutations."""
        with self._lock:
            snap = list(self._bookkeeping.items())
        yield from snap

    def bookkeeping_count(self) -> int:
        """Number of keys that have a bookkeeping record."""
        with self._lock:
            return len(self._bookkeeping)

    def drop_tier(self, tier: str) -> None:
        """Remove *tier* wholesale: its entries, registry (incl. simhash), and
        the bookkeeping records for every key it held, active or stale.

        Compensates a failed interim commit whose tier registration must be
        rolled back atomically.
        Bookkeeping has no tier index of its own (it is a flat ``key ->
        record`` dict), so :meth:`active_keys_in_tier` and
        :meth:`stale_keys_in_tier` are used to enumerate which records to
        drop before the registry itself is removed — a withheld id's
        bookkeeping record must not survive the tier that carried it (a
        discarded tier's withheld-id rows would otherwise be re-indexed by
        :meth:`~paramem.server.router.QueryRouter.reload` against a tier that
        no longer exists).

        No-op when *tier* is unknown to either ``_entries`` or ``_registry``.

        The entire compound mutation (bookkeeping + entries + registry/simhash)
        is performed under a single lock acquisition."""
        with self._lock:
            for key in self.active_keys_in_tier(tier) + self.stale_keys_in_tier(tier):
                self._bookkeeping.pop(key, None)
            self._entries.pop(tier, None)
            self._registry.pop(tier, None)

    # ------------------------------------------------------------------
    # SimHash fingerprints — public accessors
    # ------------------------------------------------------------------
    def simhash(self, tier: str, key: str) -> int | None:
        """Return the simhash fingerprint for ``(tier, key)``, or ``None``.

        Reads from the registry's one fingerprint map.  Returns ``None`` when
        the tier has no registry or the key has no fingerprint — including
        every withheld id, which carries none by design."""
        with self._lock:
            reg = self._registry.get(tier)
            if reg is None:
                return None
            return reg.simhash_for(key)

    def has_simhash(self, tier: str, key: str) -> bool:
        """True when *key* has a stored fingerprint in *tier*."""
        with self._lock:
            reg = self._registry.get(tier)
            if reg is None:
                return False
            return reg.has_simhash(key)

    def _tier_for_simhash(self, key: str) -> str | None:
        """Return the tier whose registry holds *key*'s fingerprint, or ``None``.

        Used by the legacy flat-view setter and the probe confidence gate.
        Scans every registry's one fingerprint map."""
        with self._lock:
            for tier, reg in self._registry.items():
                if reg.has_simhash(key):
                    return tier
            return None

    def put_simhash(self, tier: str, key: str, fingerprint: int) -> None:
        """Write the simhash fingerprint for ``(tier, key)`` into the registry.

        Does not touch the entry or registry lifecycle (active/stale).  The
        registry is auto-created for *tier* on first access."""
        with self._lock:
            self._registry.setdefault(tier, KeyRegistry()).set_simhash(key, fingerprint)

    def delete_simhash(self, tier: str, key: str) -> None:
        """Remove the simhash fingerprint for ``(tier, key)`` from its registry.

        No-op when the tier has no registry or the key has no fingerprint."""
        with self._lock:
            reg = self._registry.get(tier)
            if reg is not None:
                reg.drop_simhash(key)

    def tier_simhashes(self, tier: str) -> dict[str, int]:
        """Return *tier*'s one fingerprint map — its active keys' fingerprints.

        Args:
            tier: Tier name (e.g. ``"episodic"``).

        Returns:
            A fresh ``dict[str, int]`` (not a live view), empty when *tier*
            has no registry.  Callers that need a mutable live backing dict
            are using a deprecated pattern — use :meth:`put_simhash` for
            writes.
        """
        with self._lock:
            reg = self._registry.get(tier)
            if reg is None:
                return {}
            return reg._simhashes()

    def replace_simhashes_in_tier(self, tier: str, new_simhashes: dict[str, int]) -> None:
        """Bulk-replace the simhash fingerprints for *tier*'s active keys.

        Thin per-tier dispatch: resolves *tier*'s registry (creating one on
        first access, the same as every other per-tier accessor on this
        class) and delegates the validate-then-swap primitive to
        :meth:`~paramem.training.key_registry.KeyRegistry.replace_simhashes`
        — the tier's registry owns its one fingerprint map and its own
        withheld-id refusal; this method does not read or write
        ``KeyRegistry``'s private state.

        Raises:
            ~paramem.memory.store.BookkeepingInvariantViolation: *new_simhashes*
                names a withheld id in *tier*'s registry.  Neither the old
                map nor a partial new one is left in place.
        """
        with self._lock:
            reg = self._registry.setdefault(tier, KeyRegistry())
            reg.replace_simhashes(new_simhashes)

    # ------------------------------------------------------------------
    # Lifecycle registry
    # ------------------------------------------------------------------
    def registry(self, tier: str) -> KeyRegistry:
        """Return the per-tier :class:`KeyRegistry`, creating it on first access.

        Always returns a :class:`KeyRegistry` — never ``None``.  Key
        lifecycle (add/stale/remove) recording is unconditional.
        """
        with self._lock:
            return self._registry.setdefault(tier, KeyRegistry())

    def load_registry(self, tier: str, registry: KeyRegistry) -> None:
        """Install a pre-loaded :class:`KeyRegistry` for *tier* at boot.

        Simhashes are carried inside the registry; no separate sync is
        needed."""
        with self._lock:
            self._registry[tier] = registry

    def has_registry(self, tier: str) -> bool:
        """True when *tier* has an allocated registry."""
        with self._lock:
            return tier in self._registry

    def tiers_with_registry(self) -> list[str]:
        """Tiers for which a registry has been allocated."""
        with self._lock:
            return list(self._registry.keys())

    def active_keys_in_tier(self, tier: str) -> list[str]:
        """Return the active keys for *tier* from the registry."""
        with self._lock:
            reg = self._registry.get(tier)
            return reg.list_active() if reg is not None else []

    def stale_keys_in_tier(self, tier: str) -> list[str]:
        """Return the stale keys for *tier* from the registry.

        Per-tier analogue of :meth:`active_keys_in_tier` for the stale
        partition. A key an acting site's fate decision removed outright
        (:meth:`~paramem.training.key_registry.KeyRegistry.remove` — reached
        when the OWNING TIER is REBUILT by the event, already re-deriving its
        content from a key set the retired id is not in, vs. withheld behind
        a marker when the owning tier is not rebuilt; see
        :meth:`~paramem.training.consolidation.ConsolidationLoop._apply_working_fate_decisions`)
        is neither active nor stale and is therefore invisible to this
        count, by design (the registry has nothing left to report).
        """
        with self._lock:
            reg = self._registry.get(tier)
            return reg.list_stale() if reg is not None else []

    def all_active_keys(self) -> list[str]:
        """Every active key across every registered tier."""
        with self._lock:
            return [k for reg in self._registry.values() for k in reg.list_active()]

    def is_known(self, key: str) -> bool:
        """True when *key* is active OR stale in any tier's registry.

        KNOWN-legitimacy analogue of ``key in reg`` (active-only).  Returns
        False when the key is absent from both partitions of every tier.
        Use for orphan checks and bookkeeping retention; use
        :meth:`all_active_keys` for serving/enumeration.
        """
        with self._lock:
            for reg in self._registry.values():
                if reg.knows(key):
                    return True
            return False

    def tier_for_known_key(self, key: str) -> str | None:
        """Return the tier whose registry tracks *key* as active OR stale.

        KNOWN-legitimacy analogue of ``key in reg`` (active-only tier
        lookup).  Returns ``None`` when no tier knows *key* in either
        partition.  Under the single-tier-ownership invariant a key is
        known by exactly one
        tier's registry at a time, so the first match found while walking
        ``_registry`` is the only match — the walk order is irrelevant to
        the result by that invariant, not by construction of the order
        itself.
        """
        with self._lock:
            for tier, reg in self._registry.items():
                if reg.knows(key):
                    return tier
            return None

    def all_known_keys(self) -> list[str]:
        """Every active ∪ withheld key across every registered tier.

        Expressed via :meth:`KeyRegistry.list_known` so the union logic has
        a single definition.
        """
        with self._lock:
            return [k for reg in self._registry.values() for k in reg.list_known()]

    def discard_keys(self, keys: list[str]) -> None:
        """Soft-remove *keys*: withhold each in its owning tier's registry.

        One meaning only — the former ``mode="erase"`` hard-removal branch
        and its ``mode`` parameter are retired with their last caller (the
        operator doors, which now narrow to a stale-mark + registry restamp;
        see :func:`~paramem.memory.persistence.erase_keys_and_restamp_manifest`).
        Full retirement of a key still exists — an acting site's fate
        decision on a tier the event REBUILDS removes it outright via
        :meth:`~paramem.training.key_registry.KeyRegistry.remove` — it is
        just no longer reachable through this method.

        For each key, calls :meth:`KeyRegistry.stale` on the owning tier,
        which mints a marker holding only the id — the active fingerprint
        does not survive the transition.  Entries and bookkeeping are
        untouched — the row leaves with the rest of the key at the tier's
        own rebuild, not here.

        This is an **in-memory** mutation only.  Callers are responsible for
        their own disk saves.

        The entire mutation is performed under a single lock acquisition.
        Nested calls to :meth:`tiers_with_registry` succeed via RLock
        reentrancy.
        """
        with self._lock:
            for key in keys:
                for tier_name in self.tiers_with_registry():
                    reg = self._registry.get(tier_name)
                    if reg is not None and key in reg:
                        reg.stale(key)
                        break  # single-tier-ownership invariant

    # ------------------------------------------------------------------
    # Serving read doors — selected once, at the serving boundary
    # (paramem.server.inference._probe_and_reason), by
    # inference.preload_cache.  No layering between them: no
    # cache-check-then-probe, no on-miss fallback, no write-back at
    # serving.  See the module docstring's mirror statement.
    # ------------------------------------------------------------------
    def _confidence_gate(self, key: str, entry: dict) -> float | None:
        """Return the SimHash confidence for *key* against *entry*, or None
        when *key* has no registered fingerprint.

        THE fingerprint gate for content crossing a source boundary into a
        turn — used by :meth:`probe_source` only; the cache door performs no
        fingerprint work (its content was gated once at admission).  Returns
        the computed confidence float when a fingerprint is on record (which
        may be below threshold), or ``None`` when *key* has no owning simhash
        tier or the owning tier carries no fingerprint for it (e.g. a fresh
        tier before first consolidation) — callers MUST treat ``None`` as a
        failed gate, never as pass-through: a trained tier must always be
        provable.

        The fingerprint is looked up by scanning all registry tiers via
        :meth:`_tier_for_simhash` so a key whose simhash was stored under
        a different tier from the one it was requested under (e.g. an
        interim slot promoted to main) is still verified correctly.

        Invariant: never uses truthiness on registry or sub-dicts —
        all presence checks use explicit ``in`` / ``is None``."""
        from paramem.memory.entry import verify_confidence

        owning_simhash_tier = self._tier_for_simhash(key)
        if owning_simhash_tier is None:
            return None
        # Use the locked accessor self.simhash() instead of raw
        # self._registry.get() so a concurrent swap cannot rebind
        # _registry between the _tier_for_simhash call above and the
        # fingerprint read here.
        fp = self.simhash(owning_simhash_tier, key)
        if fp is None:
            return None
        # Build the minimal entry shape verify_confidence expects.
        candidate = {
            "key": key,
            "subject": entry.get("subject", ""),
            "predicate": entry.get("predicate", ""),
            "object": entry.get("object", ""),
        }
        return verify_confidence(candidate, {key: fp})

    @staticmethod
    def _render_cache_entry(key: str, entry: dict) -> dict:
        """Render a content-only mirror entry into the probe result contract.

        THE one renderer for the cache door's per-key shape: entries hold
        SPO only (the mirror's content-only invariant), so no speaker field
        is rendered — speaker attribution lives in bookkeeping, never in a
        probe result.  No fingerprint work here — the content was gated once
        at admission (the boot fill or go-live adoption); re-gating
        already-admitted content would be a second invocation of an
        admission-owned transformation.  ``confidence`` is always ``1.0``
        (pass-through — the mirror carries no per-read confidence signal).

        ``fact_text`` renders raw ``speaker{N}`` tokens verbatim — there is
        no resolver here; a token is substituted for a display name exactly
        once, at the reply boundary, by
        :func:`~paramem.server.speaker.resolve_speaker_tokens`.
        """
        import json as _json

        from paramem.memory.entry import entry_fact_text

        base = {
            "key": key,
            "subject": entry.get("subject", ""),
            "predicate": entry.get("predicate", ""),
            "object": entry.get("object", ""),
        }
        return {
            **base,
            "confidence": 1.0,
            "fact_text": entry_fact_text(base),
            "raw_output": _json.dumps(base),
        }

    def probe_cache(self, keys_by_adapter: dict[str, list[str]]) -> dict[str, dict | None]:
        """THE CACHE DOOR — a plain, registry-scoped lookup against the RAM mirror.

        Scoped by the requested tier's own registry: a key that tier's
        registry does not call ACTIVE right now — stale (soft-removed by
        :meth:`discard_keys`) or simply unknown — answers ``None`` without
        ever consulting ``_entries``.  This is load-bearing, not incidental:
        entries deliberately survive both :meth:`discard_keys` (the row
        leaves only at that tier's own next rebuild) and a rows-only
        member's :meth:`adopt_increments` install, so an un-scoped lookup
        would serve a fact the registry no longer calls active.  An ACTIVE
        key with no entry ALSO answers ``None`` — the same no-fact shape as
        a live-door miss, never a fault.  Reading the registry here is a
        plain, NON-CREATING lookup (``self._registry.get(tier)``, never
        :meth:`registry`'s ``setdefault`` — a door must not phantom-register
        an unknown tier into ``_registry``, which would leak it into
        :meth:`tiers_with_registry` and the interim-slot enumeration) plus
        :meth:`KeyRegistry.__contains__` — no registry SIDE EFFECT, no
        fingerprint re-check, no integrity signal, no write-back.
        Production value of the caller: ``inference.preload_cache=True`` at
        the serving boundary (:func:`paramem.server.inference._probe_and_reason`)
        — the one caller.

        Contract: probe is a pure lookup — it does not scope by speaker.
        Callers pass already speaker-scoped keys; the router's per-speaker
        key intersection
        (:attr:`~paramem.server.router.QueryRouter._speaker_key_index`) is
        the single privacy boundary.
        """
        results: dict[str, dict | None] = {}
        for tier, keys in keys_by_adapter.items():
            with self._lock:
                reg = self._registry.get(tier)
            for key in keys:
                if reg is None or key not in reg:
                    # Not active in this tier's registry right now — stale,
                    # unknown, or the tier has no registry at all.  Same
                    # no-fact shape as a missing entry.
                    results[key] = None
                    continue
                entry = self.get(key)
                results[key] = None if entry is None else self._render_cache_entry(key, entry)
        return results

    def probe_source(
        self, keys_by_adapter: dict[str, list[str]], *, source
    ) -> dict[str, dict | None]:
        """THE LIVE DOOR — one grouped probe against *source*, no cache contact.

        NOTE — this method is deliberately NOT wrapped in ``self._lock``.
        It calls ``source.probe(...)``, which is a GPU ``model.generate()``
        call that may block for seconds.  Holding the store lock across GPU
        work would stall every concurrent store reader for the duration of
        the GPU call.  The one read inside this method — the confidence
        gate's fingerprint lookup — goes through individually locked leaf
        methods (:meth:`simhash`, :meth:`_tier_for_simhash`); no raw access
        to ``_entries``, ``_registry``, or ``_bookkeeping`` is made, and the
        cache is neither read nor written.

        ``source`` is REQUIRED — there is no default and no fallback to the
        cache.  Every requested key is probed against the venue in one
        grouped call; :meth:`_confidence_gate` is applied to the results
        verbatim (``verify_confidence`` against the tier's stored
        fingerprint) — a key with no registered fingerprint, or a hit
        scoring below :data:`~paramem.memory.entry.DEFAULT_CONFIDENCE_THRESHOLD`,
        both answer ``None``: a trained tier must always be provable, so a
        fingerprint-less result is treated the same as a failed gate, never
        served pass-through.  A miss or a gate-drop answers ``None`` for that
        key.
        Production value of *source*: ``build_memory_source(mode=...)``,
        constructed only in the ``inference.preload_cache=False`` arm of the
        serving boundary (:func:`paramem.server.inference._probe_and_reason`)
        — the one caller.  The fold never calls this door: it builds and
        probes its own :class:`~paramem.memory.source.MemorySource` into
        fold-local working state and never touches the store mid-event.

        Contract: probe is a pure latency read — it does not scope by
        speaker.  Callers pass already speaker-scoped keys; the router's
        per-speaker key intersection
        (:attr:`~paramem.server.router.QueryRouter._speaker_key_index`) is
        the single privacy boundary.  No memory source emits ``speaker_id``
        — a source result is content only (plus its own derived fields), so
        no dict probe returns carries one either.  Source-served hits pass
        through the source's own dict verbatim (which may carry other
        fields the source attached, e.g. ``answer``, see ``inference.py``)
        with ``confidence`` patched to the gate's computed value.  A source
        result carrying ``failure_reason`` (parse failure, key mismatch, or
        the source's OWN internal gate drop — :class:`WeightMemorySource`
        and :class:`DiskMemorySource` both gate before returning) is
        NORMALIZED to ``None`` here, same as an absent result: the door
        contract is "a miss or a gate-drop answers ``None`` for that key",
        never a differently-shaped failure dict.  A debug-level log line
        names the key and the source's own ``failure_reason`` as the trace
        — the source already warned once at its own boundary.  A source
        result that is not even a dict (a source violating its own return
        contract) is likewise normalized to ``None`` here, with a
        warning-level log line naming the key and the received type — loud,
        since nothing upstream of this door has warned about it yet.  This
        normalization is door-only: the fold's drop-scan and the boot fill
        call the source directly (never through this door) and keep
        consuming the failure shape unchanged.

        ``fact_text`` renders raw ``speaker{N}`` tokens verbatim — there is
        no resolver here; a token is substituted for a display name exactly
        once, at the reply boundary, by
        :func:`~paramem.server.speaker.resolve_speaker_tokens`.
        """
        from paramem.memory.entry import DEFAULT_CONFIDENCE_THRESHOLD

        source_results = source.probe(keys_by_adapter)
        results: dict[str, dict | None] = {}
        for _tier, keys in keys_by_adapter.items():
            for key in keys:
                src = source_results.get(key)
                if src is None:
                    results[key] = None
                    continue
                if not isinstance(src, dict):
                    logger.warning(
                        "MemoryStore.probe_source: off-contract source result for key %r "
                        "(%s) -- answering None",
                        key,
                        type(src).__name__,
                    )
                    results[key] = None
                    continue
                if "failure_reason" in src:
                    # Normalize at the door: a miss or a gate-drop answers
                    # None for that key, never the source's own differently-
                    # shaped failure dict.  The source already logged its
                    # own warning; this is the door-side trace.
                    logger.debug(
                        "MemoryStore.probe_source: key %r answers None (source failure: %s)",
                        key,
                        src.get("failure_reason"),
                    )
                    results[key] = None
                    continue
                confidence = self._confidence_gate(key, src)
                if confidence is None:
                    logger.debug(
                        "MemoryStore.probe_source: key %r dropped by confidence gate "
                        "(no registered fingerprint)",
                        key,
                    )
                    results[key] = None
                    continue
                if confidence < DEFAULT_CONFIDENCE_THRESHOLD:
                    logger.debug(
                        "MemoryStore.probe_source: key %r dropped by confidence gate "
                        "(%.3f < %.3f threshold)",
                        key,
                        confidence,
                        DEFAULT_CONFIDENCE_THRESHOLD,
                    )
                    results[key] = None
                    continue
                # Patch the rendered confidence onto the source result so
                # callers always see the real score (not a stale 1.0 from
                # a source that ran without a registry).
                rendered = dict(src)
                rendered["confidence"] = confidence
                results[key] = rendered
        return results

    # ------------------------------------------------------------------
    # On-disk registries — load registries + simhashes from the adapter dir
    # ------------------------------------------------------------------
    @staticmethod
    def _iter_tier_registry_paths(adapter_dir):
        """Yield ``(tier_name, registry_path)`` for every tier under *adapter_dir*.

        The single description of where an adapter tree keeps its
        ``indexed_key_registry.json`` files: delegates to
        :func:`~paramem.memory.interim_adapter.iter_tier_roots` (main tiers
        first, in fixed order, then every interim slot found on disk) and
        appends the registry filename to each yielded tier root.  Both
        on-disk readers — :meth:`read_registries_from_disk` and
        :meth:`read_simhash_registry_from_disk` — walk this one generator, so
        they cannot disagree about which files belong to the store.

        Main-tier paths are yielded whether or not the file exists (the
        readers turn an absent file into an empty registry / empty fingerprint
        map); interim tiers are yielded only for directories that exist.

        Args:
            adapter_dir: Path to the adapter root directory.

        Yields:
            ``(tier_name, path)`` where ``tier_name`` is the PEFT adapter name
            (``"episodic"`` … or ``"episodic_interim_<stamp>"``).
        """
        from pathlib import Path

        from paramem.memory.interim_adapter import iter_tier_roots

        adapter_dir = Path(adapter_dir)
        for tier, tier_root in iter_tier_roots(adapter_dir):
            yield tier, tier_root / "indexed_key_registry.json"

    @staticmethod
    def read_registries_from_disk(adapter_dir) -> "dict[str, KeyRegistry]":
        """Read per-tier ``indexed_key_registry.json`` files from disk into a
        fresh ``dict[str, KeyRegistry]`` without touching any live store.

        This is the store-free counterpart to :meth:`load_registries_from_disk`,
        which delegates here and then installs. Batch, all-or-nothing: any one
        tier's read raising propagates and aborts the whole call — it does
        NOT verify a tier's registry against its slot manifests. Boot uses
        :func:`~paramem.adapters.registry_binding.verify_adapter_tree` instead
        (per-tier verified, unverified tiers simply excluded rather than
        aborting the whole read); this method remains the reader for callers
        that do not need that verification — the trial store, base-swap
        Phase-B, and cold-store paths in ``app.py``, all reached through the
        instance method :meth:`load_registries_from_disk`.

        Reads every path :meth:`_iter_tier_registry_paths` yields:

        * ``<adapter_dir>/<tier>/indexed_key_registry.json`` for each main tier
          and every ``episodic_interim_<stamp>`` slot.

        The registry file carries the tier's one fingerprint map — active
        keys only — in the ``"simhash"`` key.

        Entry payloads (subject/predicate/object/speaker_id) are NOT loaded
        here — that is the responsibility of the mode-specific
        :class:`paramem.memory.source.MemorySource`.

        Args:
            adapter_dir: Path to the adapter root directory.

        Returns:
            Fresh ``dict[str, KeyRegistry]`` populated from disk.  Main tiers
            always appear (even when their registry file is absent —
            :meth:`KeyRegistry.load` returns an empty registry for missing
            files).  Interim tiers appear only when their directories exist.
        """
        from paramem.training.key_registry import KeyRegistry

        return {
            tier: KeyRegistry.load(reg_path)
            for tier, reg_path in MemoryStore._iter_tier_registry_paths(adapter_dir)
        }

    @staticmethod
    def read_simhash_registry_from_disk(adapter_dir, *, cached: bool = False) -> "dict[str, int]":
        """Merge every tier registry under *adapter_dir* into one ``{key: fp}`` map.

        The flat projection of :meth:`read_registries_from_disk` — same disk
        walk (:meth:`_iter_tier_registry_paths`: main tiers + every interim
        slot), same files, only the shape differs.  Callers that need per-key
        fingerprints without a live store use this; there is deliberately no
        second walk of the adapter tree.

        Per file it calls the one leaf,
        :meth:`paramem.training.key_registry.KeyRegistry.load_simhashes`, so
        the fingerprint-file shape is known in exactly one place — the same
        leaf the trial-consolidation gates use on a single path.

        The map carries each tier's one fingerprint map — active keys only —
        exactly as serialised under the ``"simhash"`` key of each registry
        file.  When a key appears in more than one tier file (transient
        during promotion) the later read wins — the fingerprint content is
        identical either way.

        Args:
            adapter_dir: Path to the adapter root directory.
            cached: When ``True``, serve from and populate the process-wide
                cache keyed by *adapter_dir*'s resolved path
                (``_SIMHASH_REGISTRY_CACHE``), skipping the disk walk
                entirely on a hit. Default ``False`` preserves disk-truth
                reads for every existing caller. A caller opting in relies
                on :func:`invalidate_simhash_registry_cache` (called by
                :meth:`paramem.server.router.QueryRouter.reload`) to keep
                the cache fresh; an invalidation racing a concurrent walk
                is handled structurally by the generation guard described
                below, not by caller-side serialization.

        Returns:
            ``{key: fingerprint}`` across every tier.  Empty when the directory
            holds no registry files.  On any ``cached=True`` call — a cache
            hit, or a miss whose freshly-walked result gets published — this
            is the *same dict object* held in ``_SIMHASH_REGISTRY_CACHE``
            — callers must not write to it (matches the
            :meth:`entries_in_tier` rule).

        Raises:
            Whatever :meth:`KeyRegistry.load_simhashes` raises on an
            unparseable (``json.JSONDecodeError``) or non-registry
            (``ValueError``) file — a corrupt registry is a data-integrity
            fault, surfaced here exactly as it is on the boot path rather than
            silently skipped, because a partial map silently un-gates every key
            of the failed tier.
        """
        from paramem.training.key_registry import KeyRegistry

        cache_key = None
        generation = None
        if cached:
            from pathlib import Path

            cache_key = str(Path(adapter_dir).resolve())
            with _SIMHASH_REGISTRY_CACHE_LOCK:
                hit = _SIMHASH_REGISTRY_CACHE.get(cache_key)
                generation = _SIMHASH_REGISTRY_CACHE_GENERATION
            if hit is not None:
                return hit

        merged: dict[str, int] = {}
        for _tier, reg_path in MemoryStore._iter_tier_registry_paths(adapter_dir):
            merged.update(KeyRegistry.load_simhashes(reg_path))

        if cache_key is not None:
            with _SIMHASH_REGISTRY_CACHE_LOCK:
                # Publish only if no invalidation landed while this walk was
                # in flight — otherwise the walked (now-possibly-stale)
                # result is still returned to this caller but never cached,
                # so the next cached read re-walks instead of serving it.
                if _SIMHASH_REGISTRY_CACHE_GENERATION == generation:
                    _SIMHASH_REGISTRY_CACHE[cache_key] = merged

        return merged

    def load_registries_from_disk(self, adapter_dir) -> None:
        """Load per-tier ``indexed_key_registry.json`` into the store.

        Reads:

        * ``<adapter_dir>/<tier>/indexed_key_registry.json`` for each main tier
          and every ``episodic_interim_<stamp>`` slot.

        The registry file now carries the tier's one fingerprint map — active
        keys only — in the ``"simhash"`` key.  The separate
        ``simhash_registry.json`` file is no longer read — it has been
        eliminated.

        Entry payloads (subject/predicate/object/speaker_id) are NOT loaded
        here — that is the responsibility of the mode-specific
        :class:`paramem.memory.source.MemorySource` (weight probe in
        train mode; encrypted graph.json read in simulate mode).

        Delegates disk reads to :meth:`read_registries_from_disk` — batch,
        all-or-nothing, no per-tier verification (see that method's
        docstring for the callers this is and is not for).
        """
        registries = MemoryStore.read_registries_from_disk(adapter_dir)
        for tier, reg in registries.items():
            self.load_registry(tier, reg)

    def load_bookkeeping_from_disk(self, adapter_dir) -> dict:
        """Load per-key bookkeeping into ``_bookkeeping`` from every tier's
        ``key_metadata.json``.

        Sole boot loader for the per-key bookkeeping rows.  Walks
        :func:`~paramem.memory.interim_adapter.iter_tier_roots` (main tiers,
        then interim slots) and reads ``<tier_root>/key_metadata.json`` where
        present.  Runs unconditionally at lifespan boot (after
        ``load_registries_from_disk``) — entry-independent, so it no longer
        requires entries to already exist.  Under
        ``inference.preload_cache=False`` this is the ONLY write to provenance
        state at boot, and is sufficient for the router's speaker index.

        Conflict rule (stated once, applied here): when a key's row appears
        in more than one tier's file — a stale leftover from a tier the key
        no longer belongs to — the row from the tier whose registry CURRENTLY
        owns the key (:meth:`tier_for_known_key`) wins.  Under the
        single-tier-ownership invariant a key is owned by exactly one tier's
        registry at a time, so exactly one tier's file can match; the walk
        order :meth:`tier_for_known_key` uses is irrelevant to the outcome,
        not load-bearing.

        Populates ``_bookkeeping`` only (via :meth:`set_bookkeeping`).  DOES
        NOT touch ``_entries`` — the old ``setdefault_entry`` parasitic write
        that created payload-less stub entries has been removed.  Bookkeeping
        presence MUST NOT manufacture a content cache hit.

        Each persisted record is splatted whole into :meth:`set_bookkeeping`
        (``self.set_bookkeeping(key, **key_meta)``) — a persisted row with an
        empty ``speaker_id`` fails :func:`~paramem.memory.bookkeeping.bookkeeping_row`'s
        guard here exactly as it would at mint time; there is no reload-time
        carve-out for legacy unattributed keys.  The write side (the per-tier
        commit primitive,
        :func:`~paramem.memory.persistence.commit_tier_slot`) persists
        ``dict(bk)`` from :meth:`bookkeeping_for_key` verbatim, so the
        on-disk record always carries exactly the fields
        :meth:`set_bookkeeping` requires — there is no hand-listed field
        projection or legacy-fill tolerance on either side.  No backward
        compatibility: a record missing a mandatory field (or carrying an
        unexpected one) raises ``TypeError`` from the splat, not a silent
        default fill.  The deploy procedure for an incompatible on-disk shape
        is to start fresh (wipe adapters and registry).

        A tier whose registry (already loaded by the preceding
        ``load_registries_from_disk`` call) reports known keys (active ∪
        stale) but has NO ``key_metadata.json`` at all is a violation of the
        every-known-key-has-a-row invariant, raised via
        :func:`raise_bookkeeping_invariant_violation` naming the tier and its
        known keys — never tolerated, never silently continued past.  A tier
        with a registry and ZERO known keys and no row file is the ordinary
        empty case, not a violation.

        Keys absent from every tier registry (orphans — slot wiped or never
        existed) are skipped but counted in the return dict.

        Returns ``{loaded, orphaned}`` for ``/status`` and diagnostics.
        No per-tier files found → all zeros, no-op (fresh install).

        Raises:
            BookkeepingInvariantViolation: A tier's registry has known keys
                but no ``key_metadata.json`` file exists to bookkeep them.
        """
        import json
        from pathlib import Path

        from paramem.backup.encryption import read_maybe_encrypted
        from paramem.memory.interim_adapter import iter_tier_roots

        loaded = 0
        orphaned = 0
        seen_orphan_keys: set[str] = set()
        for tier_name, tier_root in iter_tier_roots(Path(adapter_dir)):
            path = tier_root / "key_metadata.json"
            if not path.exists():
                known = self.registry(tier_name).list_known()
                if known:
                    raise_bookkeeping_invariant_violation(
                        tier_name, known, "boot bookkeeping load: key_metadata.json missing"
                    )
                continue
            metadata = json.loads(read_maybe_encrypted(path).decode("utf-8"))
            for key, key_meta in metadata.get("keys", {}).items():
                owner = self.tier_for_known_key(key)
                if owner is None:
                    if key not in seen_orphan_keys:
                        orphaned += 1
                        seen_orphan_keys.add(key)
                    continue
                if owner != tier_name:
                    # Not the current owner's row — a stale leftover from a
                    # tier the key has since moved off of.  Skip it; the
                    # owning tier's own file (walked separately, in any
                    # order) carries the authoritative row.
                    continue
                self.set_bookkeeping(key, **key_meta)
                loaded += 1
        return {"loaded": loaded, "orphaned": orphaned}

    def swap(
        self,
        new_entries: dict[str, dict[str, dict]],
        new_registry: dict[str, KeyRegistry],
        new_bookkeeping: dict[str, dict],
    ) -> None:
        """Atomically rebind all three mutable structures in a single locked operation.

        This is the boot-hydration primitive: the lifespan hydration path
        (and the quarantine-lift retry that re-runs the identical hydration
        without a restart) builds ``new_entries``/``new_registry``/
        ``new_bookkeeping`` entirely off-store (disk reads, weight probes,
        verification) and calls ``swap`` once to publish the result, never
        clearing the live store first.  No reader ever observes a torn or
        half-rebuilt state across ``_entries``, ``_registry``, and
        ``_bookkeeping``.

        Args:
            new_entries: The replacement ``tier → key → entry`` mapping.
            new_registry: The replacement ``tier → KeyRegistry`` mapping.
            new_bookkeeping: The replacement ``key → bookkeeping-record`` mapping.

        Contract for callers: construct all three off-store, validate them
        (registries loaded, simhashes set), then call ``swap`` once.  After
        this call returns, the previous references are unreachable from the
        store.  Any thread that already holds a reference to the old
        ``_entries`` dict (e.g. an in-flight ``iter_entries`` snapshot) still
        holds a valid reference to the now-disconnected structure — that is safe
        because the snapshot is immutable from the reader's perspective.
        """
        with self._lock:
            self._entries = new_entries
            self._registry = new_registry
            self._bookkeeping = new_bookkeeping

    def adopt_increments(
        self, increments: "Sequence[TierIncrement]", *, absorbed_tiers: "Sequence[str]" = ()
    ) -> None:
        """Take a written bundle live: converge each member tier onto its increment.

        Adoption refreshes the mirror and validates NOTHING against it: a
        rebuilt member's entries land in ``_entries`` from the increment's
        own keyed list; a rows-only member carries no entries and changes
        none.  The live cache is never read here — go-live's correctness
        lives in the authoritative published artifacts (``registry_bytes``
        / ``rows_bytes``), not in what the mirror happened to hold going in.

        The one completeness check this method still runs is
        INCREMENT-INTERNAL: :func:`~paramem.memory.increment.build_tier_increment`
        reads a rebuilt member's ``indexed_key_registry.json`` and
        ``keyed.json`` as two independent files, so a gap between what the
        registry calls active and what the keyed list actually produced is a
        genuine cross-artifact divergence within that ONE increment — not a
        claim about the live store.  Checked BEFORE any mutation runs, so the
        check either passes with the store exactly as it is about to become,
        or raises with the store exactly as it already was — never a
        postcondition caught only after the store has already been mutated.

        ONE lock acquisition for the whole bundle AND the absorbed ring
        together, in three passes — check, drop, install:

        0. **Check.** For every REBUILT member, confirm every key its own
           registry calls active is in its own ``increment.entries``.  A
           violation raises
           :class:`~paramem.memory.store.EntryCacheInvariantViolation` via
           :func:`raise_entry_cache_invariant_violation` before the drop
           pass below runs, so the store is left byte-for-byte as it was.  A
           rows-only member has nothing of its own to check here (its
           entries are empty by design) and the live store's existing
           ``_entries`` bucket for that tier is never read.  The same pass
           also confirms bookkeeping completeness: every key in the
           increment's OWN ``registry.list_known()`` (active ∪ stale —
           rows-only members included) must appear in
           ``increment.bookkeeping``.  A gap raises
           :class:`~paramem.memory.store.BookkeepingInvariantViolation` via
           :func:`raise_bookkeeping_invariant_violation`, before any
           mutation, same as the entry-cache check above.  This pass ALSO
           validates every increment's rows: each ``increment.bookkeeping``
           entry is rebuilt through
           :func:`~paramem.memory.bookkeeping.bookkeeping_row` (``rows_bytes``
           is on-disk bytes, on the crash-resume path written by a
           *different process*, so it needs the same schema and
           empty-speaker guard a fresh mint does) into a local ``validated``
           map, keyed by the row's own key.  A row carrying an unexpected or
           missing field raises ``TypeError`` from the splat — a schema
           refusal at go-live, before any mutation, changing that resume
           path's failure mode from a silent bad row to a raise.  Pass 2
           installs from ``validated``, never from the increment's raw row
           dict — one construction, one copy.
        1. **Drop.** For every member tier, read its OUTGOING registry's
           ``list_known()`` (active ∪ stale) BEFORE any rebind, and drop
           every one of those keys' bookkeeping rows.  Same for every
           *absorbed_tiers* member: its own ``list_known()``, read before its
           registry is dropped, drops its bookkeeping rows too — a key that
           lived ONLY in an absorbed interim slot (never routed into any
           primary tier's increment; a genuine reap casualty) would otherwise
           leave an orphan bookkeeping row with no registry membership
           anywhere, forever (nothing else ever revisits it).  The full drop
           set is computed and applied for the WHOLE call before any install
           — bookkeeping is a flat ``key -> row`` dict with no tier index, so
           installing tier A's row before dropping tier B's outgoing set
           could otherwise delete a key that just moved from B into A (a
           promotion, or an adopted interim key) the moment B's drop ran.
        2. **Install.** For every member tier: rebind ``self._registry[tier]``
           to the increment's registry object; when ``increment.rebuilt``,
           rebind ``self._entries[tier]`` to the increment's entries
           (untouched for a rows-only member); install every row in
           ``increment.bookkeeping``.  Then drop each *absorbed_tiers*
           member's own ``_registry``/``_entries`` buckets whole (the former
           ``MemoryStore.drop_registry_and_entries`` primitive, now inlined
           here so the ring reap converges in the SAME locked act as the
           bundle's own install — no window in which a reader can observe a
           key active in both the bundle's destination tier and the
           not-yet-reaped interim tier it moved out of).

        A retired key is gone by absence: it is not in the increment's
        registry, entries, or bookkeeping, so after convergence the store
        does not hold it — there is no separate deletion step.

        Args:
            increments: The bundle's written increments — every member goes
                live together. Order is irrelevant here (unlike the on-disk
                publish, which is destination-first for a promotion bundle);
                this method only rebinds RAM state.
            absorbed_tiers: Interim tier names this bundle's go-live reaps
                whole (a full fold's ring absorption) — every key any of
                them still owns at this point was never routed into a
                primary tier's increment while the event was staging, so it is a
                genuine reap casualty, not a live fact losing its home.
                Empty for every other call, including every interim event's
                own go-live.
        """
        with self._lock:
            # --- 0. Check: every member, before any mutation ---
            validated: dict[str, dict] = {}
            for inc in increments:
                if inc.rebuilt:
                    missing = [k for k in inc.registry.list_active() if k not in inc.entries]
                    if missing:
                        raise_entry_cache_invariant_violation(inc.tier, missing, "adopt_increments")
                missing_rows = [k for k in inc.registry.list_known() if k not in inc.bookkeeping]
                if missing_rows:
                    raise_bookkeeping_invariant_violation(
                        inc.tier, missing_rows, "adopt_increments bookkeeping completeness"
                    )
                for key, row in inc.bookkeeping.items():
                    validated[key] = bookkeeping_row(key, **row)

            # --- 1. Drop ---
            drop_keys: set[str] = set()
            for inc in increments:
                outgoing = self._registry.get(inc.tier)
                if outgoing is not None:
                    drop_keys.update(outgoing.list_known())
            for tier in absorbed_tiers:
                outgoing = self._registry.get(tier)
                if outgoing is not None:
                    drop_keys.update(outgoing.list_known())
            for key in drop_keys:
                self._bookkeeping.pop(key, None)

            # --- 2. Install ---
            for inc in increments:
                self._registry[inc.tier] = inc.registry
                if inc.rebuilt:
                    self._entries[inc.tier] = dict(inc.entries)
                for key in inc.bookkeeping:
                    self._bookkeeping[key] = validated[key]

            for tier in absorbed_tiers:
                self._entries.pop(tier, None)
                self._registry.pop(tier, None)
