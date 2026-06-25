"""Per-tier indexed-key memory store.

Public API: :class:`MemoryStore`.

Single source of truth for the answer content, integrity fingerprints, and
lifecycle registries of every indexed key the system holds in RAM.  Replaces
the previous mixed-shape state on :class:`ConsolidationLoop`:

* ``indexed_key_cache: dict[str, dict]`` — flat, all tiers in one bucket; tier
  was recovered indirectly by scanning every registry.
* ``episodic_simhash``, ``semantic_simhash``, ``procedural_simhash`` — three
  separate flat dicts; now folded into the per-tier :class:`KeyRegistry` (each
  registry carries ``_simhash: dict[str, int]`` for active keys; stale
  fingerprints live in the stale record ``_stale[key]["simhash"]``).
* ``indexed_key_registry: Optional[dict[str, KeyRegistry]]`` — the only
  structure that was already per-tier; folded in here for symmetry.

The unified shape is ``tier → key → value`` for all three concerns:

* :attr:`MemoryStore.entries_in_tier` returns ``dict[key, entry_payload]``.
* :meth:`MemoryStore.tier_simhashes` returns ``dict[key, int]`` with a mandatory
  ``include_stale`` keyword — the **only** public accessor for a fingerprint set.
  The mandatory keyword makes the active-vs-known distinction impossible to
  forget, preventing the enumeration bug that caused spurious fold aborts.
* :meth:`MemoryStore.registry` returns the tier's :class:`KeyRegistry`.

Tier ownership of a key is the single source of truth; an indexed key
belongs to exactly one tier.  Cross-tier lookups (``get``, ``has``,
``tier_of``) scan tier-first then key — O(tier_count); the tier count is
small (3 main + N interim slots, typically ≤ 10).

The ``_registry`` dict is ALWAYS present (never ``None``).  ``replay_enabled``
is a BEHAVIOUR flag only — it governs whether key lifecycle (add/stale/remove)
is recorded and whether replay-training is run.  Callers needing that gate
check ``store.replay_enabled`` directly; ``registry()`` never returns ``None``.

**Content cache vs. bookkeeping — read before editing:**

:attr:`MemoryStore._entries` is a PURE INFERENCE CONTENT CACHE.  An entry
slot exists in ``_entries`` only when SPO content is materialised in RAM —
written by :meth:`put` (boot preload or on-miss memoize).  Missing entry =
cache miss.  Under ``inference.preload_cache=False`` the entries are empty by
design → every key is a clean miss → the source is always probed → restores
the documented contract.

The per-key bookkeeping fields ``speaker_id``, ``first_seen_cycle``, and
``relation_type`` live in :attr:`MemoryStore._bookkeeping` — a flat
``{key → {speaker_id, first_seen_cycle, relation_type}}`` dict SEPARATE from
``_entries``.  Populated by :meth:`load_bookkeeping_from_disk` at boot
(unconditionally; entry-independent).  Never enters
:meth:`KeyRegistry.save_bytes`, :meth:`snapshot`, or any hash path.

**ARCHITECTURAL NOTE — save-path working entries:**
The "pure content cache" rule binds the INFERENCE cache (boot preload +
on-miss memoize).  The consolidation cycle's own TRAIN/SIMULATE working
entries — written by :meth:`put` at ``consolidation.py:1488,2555,2785,2874``
from ``_cache_entry`` — DO legitimately carry ``speaker_id``/``first_seen_cycle``.
They are working entries consumed by :func:`persistence.build_graph_for_tier`
within the same cycle.  A future reader must NOT "purify" save-path entries by
stripping those fields — that would break ``build_graph_for_tier``.

All ``store.get`` readers in ``consolidation.py`` were audited.  The
SPO-only readers at ``:1819``, ``:2167``, ``:3474`` are NOT bookkeeping sites
and become strictly safer post-fix (they can only find a real content entry).

Snapshot / restore (:meth:`snapshot`, :meth:`restore`) capture the entry and
simhash fingerprint state for the cycle-resume rollback rope.  The simhash map
is captured as ``{"simhash": {tier: known_simhashes_dict}}`` — active∪stale per
tier — so fingerprints survive rollback.  Registries persist via their own
:meth:`KeyRegistry.save` / :meth:`KeyRegistry.load` lifecycle and are restored
separately from disk on rollback.  ``_bookkeeping`` is NOT in the snapshot — it
is reloaded from ``key_metadata.json`` on boot.

**SimHash storage:**
SimHash fingerprints live exclusively in :class:`KeyRegistry` (one per tier).
Active fingerprints are in ``registry._simhash``; stale fingerprints are carried
in the stale record (``registry._stale[key]["simhash"]``).  Both partitions are
serialised to ``indexed_key_registry.json`` under the ``"simhash"`` key so the
on-disk file is the single source of truth.  The separate ``simhash_registry.json``
sidecar has been eliminated.  Use :meth:`tier_simhashes` (mandatory
``include_stale`` keyword) as the only public path to a fingerprint set.

**Thread-safety concurrency contract:**

:class:`MemoryStore` is shared between the asyncio event-loop thread, FastAPI
handler threads (ThreadPoolExecutor), and the BG-trainer worker thread.  A
single :class:`threading.RLock` (``self._lock``) guards all accesses to the
three mutable structures: ``_entries``, ``_registry``, and ``_bookkeeping``.

Rules for callers and maintainers:

1. Every method that reads or writes any of the three structures holds
   ``self._lock`` for the duration of its in-RAM access.  RLock (not plain Lock)
   is used because compound mutators (``move``, ``delete``, ``discard_keys``,
   ``restore``, compound ``put``) call other wrapped leaf methods reentrantly.

2. ``iter_entries()`` and ``iter_bookkeeping()`` materialise a snapshot list
   under the lock, then yield from the snapshot outside the lock.  Callers
   iterate lock-free without risk of observing a concurrent structural mutation.

3. Compound mutators (``delete``, ``move``, ``discard_keys``, ``restore``,
   ``put`` when writing entry + simhash + registry) hold the lock ONCE around
   the whole compound so a reader never sees a half-updated multi-structure state.
   Nested leaf-method calls succeed via RLock reentrancy.

4. ``entries_in_tier()`` returns a shallow copy of the internal tier dict so
   callers that iterate the result outside the lock cannot observe a concurrent
   structural mutation.  Callers must not write to the returned dict; use
   :meth:`put` / :meth:`delete` for writes.

5. :meth:`probe` is deliberately NOT wrapped in the lock.  It calls
   ``source.probe(...)``, which is a GPU ``model.generate()`` call that may
   block for seconds.  Holding the store lock across GPU work would deadlock
   the event loop.  All reads inside ``probe`` — including ``_bookkeeping``
   and ``_registry``/simhash reads — go through individually locked leaf
   methods (``self.get``, ``self.bookkeeping_for_key``, ``self.simhash``,
   ``self._tier_for_simhash``, ``self.put``, ``self.set_bookkeeping``).
   Each read is a single locked acquisition; no lock is held across the GPU
   call.  This is sufficient for probe's single-key read-then-conditional-
   write pattern.

6. :meth:`swap` is the atomic-publish primitive for phase-2 consolidation.
   Callers build new structures off-store, then publish in one locked rebind so
   no reader ever sees a torn/half-rebuilt store.
"""

from __future__ import annotations

import copy
import logging
import threading
from collections.abc import Iterator

from paramem.graph.name_match import is_speaker_id
from paramem.training.key_registry import KeyRegistry

logger = logging.getLogger(__name__)


class MemoryStore:
    """Per-tier {entries, simhash, registry} for the indexed-key memory layer.

    ``_registry`` is always a ``dict[str, KeyRegistry]`` (never ``None``).
    The ``replay_enabled`` flag governs training behaviour only.

    Thread-safety: a single ``threading.RLock`` (``self._lock``) guards all
    in-RAM access to ``_entries``, ``_registry``, and ``_bookkeeping``.  See
    the module-level concurrency contract in the module docstring for the
    complete rules including the deliberately unwrapped :meth:`probe` and the
    atomic :meth:`swap` publish primitive.
    """

    def __init__(self, *, replay_enabled: bool = True) -> None:
        self._replay_enabled = replay_enabled
        # Single RLock guards _entries, _registry, and _bookkeeping.  RLock
        # (not plain Lock) is required because compound mutators call wrapped
        # leaf methods reentrantly (move→tier_of, delete→leaf readers, etc.).
        self._lock = threading.RLock()
        # tier -> key -> entry payload dict.  PURE INFERENCE CONTENT CACHE —
        # an entry slot exists only when SPO is materialised.  See module
        # docstring for the "save-path working entries" exception.
        # SAVE-PATH SITES (authorised store.put callers that write entries):
        #   1. consolidation.py run_cycle / ingest path — new key registration.
        self._entries: dict[str, dict[str, dict]] = {}
        # tier -> KeyRegistry — ALWAYS present (never None).  replay_enabled
        # controls whether key lifecycle is recorded, not whether the structure
        # exists.  SimHash fingerprints live ON the registry (not on MemoryStore).
        self._registry: dict[str, KeyRegistry] = {}
        # Per-key provenance bookkeeping — SEPARATE from _entries.
        # key -> {"speaker_id": str, "first_seen_cycle": int, "relation_type": str}
        # Populated by load_bookkeeping_from_disk at boot.  Never enters
        # snapshot/restore or KeyRegistry.save_bytes — stays out of the
        # hash-frozen slot-identity path.
        self._bookkeeping: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Replay-enabled gate
    # ------------------------------------------------------------------
    @property
    def replay_enabled(self) -> bool:
        """True when the lifecycle registry is active.

        When ``False``, :meth:`load_registry` raises and recall-gating paths
        that require a registry are bypassed."""
        return self._replay_enabled

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
        returns even if concurrent mutations occur.  Use :meth:`put` /
        :meth:`delete` for writes; do not mutate the returned dict."""
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
        When *register* is True and replay is enabled, the key is added to the
        tier's lifecycle registry.

        Caller is responsible for ensuring *key* is unique across tiers — if
        the key currently belongs to a different tier, call :meth:`move`
        rather than ``put`` to keep entries, registry, and simhash in agreement.

        All three writes (entry, simhash, registry) are performed under a
        single lock acquisition so a reader never observes a half-updated state.
        """
        with self._lock:
            self._entries.setdefault(tier, {})[key] = entry
            if simhash is not None:
                self._registry.setdefault(tier, KeyRegistry()).set_simhash(key, simhash)
            if register and self._replay_enabled:
                self._registry.setdefault(tier, KeyRegistry()).add(key)

    # ------------------------------------------------------------------
    # Per-key bookkeeping — speaker_id / first_seen_cycle / relation_type
    # SEPARATE from _entries; never in snapshot or KeyRegistry.save_bytes.
    # ------------------------------------------------------------------
    def set_bookkeeping(
        self,
        key: str,
        *,
        speaker_id: str,
        first_seen_cycle: int,
        relation_type: str,
        recurrence_count: int = 1,
        last_seen_cycle: int = 0,
        allow_empty_speaker: bool = False,
    ) -> None:
        """Store or update the bookkeeping record for *key*.

        All five fields are mandatory in the persisted schema (one mandatory
        tier, zero optional buckets).  The ``recurrence_count`` and
        ``last_seen_cycle`` params carry Python defaults solely as a
        legacy-fill convenience for new-key sites and boot-reload callers that
        do not yet know the values — exactly mirroring the ``relation_type``
        handling.  The stored dict always contains all five keys.

        ``speaker_id``: the speaker who first introduced this key.
        ``first_seen_cycle``: the consolidation cycle this key was first minted.
        ``relation_type``: the model-assigned relation type from extraction
        (e.g. ``"factual"``, ``"preference"``, ``"temporal"``, ``"social"``).
        Legacy keys that pre-date this field are upgraded in-memory to
        ``"unknown"`` by :meth:`load_bookkeeping_from_disk`; the correct value
        is stamped at ingestion on the next consolidation cycle.
        ``recurrence_count``: how many times the underlying fact has been
        re-seen (via intra-fold duplicate-SPO collapse) since the key was
        minted.  Default 1 (a new key has been seen once).
        ``last_seen_cycle``: the most recent consolidation cycle at which this
        key's fact survived into ``tier_keyed``.  Default 0 (unknown — upgraded
        to ``first_seen_cycle`` by :meth:`load_bookkeeping_from_disk` for
        legacy keys).
        ``allow_empty_speaker``: when ``True``, suppresses the empty-speaker_id
        guard and allows ``speaker_id=""`` to be stored.  Required for reload
        paths (legacy keys on disk may not carry a speaker) and for keyless
        concept-node edges whose subject has no speaker attribution.  Default
        ``False``; omit this flag for all new-key production writes where a
        real speaker_id must be present (no-unattributed-keys invariant).

        Raises:
            ValueError: when ``speaker_id`` is empty and ``allow_empty_speaker``
                is ``False`` — unattributed keys are not recallable by speaker.

        **Callers that need to update ONE field on an existing key must use
        :meth:`bump_recurrence` (for recurrence/last_seen updates) rather than
        calling this method, which overwrites ALL five fields and would silently
        reset the counters the caller did not supply.**

        Does NOT touch ``_entries`` — bookkeeping presence MUST NOT
        manufacture a content cache hit.

        ``speaker_id`` is normalized to lowercase when it matches the
        ``speaker{N}`` pattern (``is_speaker_id``).  This makes the probe
        speaker-filter casing invariant hold for BOTH legacy-loaded and
        runtime-set data: legacy cased ``Speaker0`` from ``key_metadata.json``
        is silently coerced to ``speaker0`` at boot via the
        :meth:`load_bookkeeping_from_disk` → :meth:`set_bookkeeping` path,
        self-healing on the next save.  Empty strings and non-speaker values
        pass through unchanged."""
        if is_speaker_id(speaker_id):
            speaker_id = speaker_id.lower()
        if not speaker_id and not allow_empty_speaker:
            raise ValueError(
                f"set_bookkeeping: empty speaker_id for key {key!r} without "
                f"allow_empty_speaker=True — unattributed keys are not recallable "
                f"by speaker (no-unattributed-keys invariant)."
            )
        with self._lock:
            self._bookkeeping[key] = {
                "speaker_id": speaker_id,
                "first_seen_cycle": first_seen_cycle,
                "relation_type": relation_type,
                "recurrence_count": recurrence_count,
                "last_seen_cycle": last_seen_cycle,
            }

    def bump_recurrence(self, key: str, *, cycle: int) -> None:
        """Increment ``recurrence_count`` and refresh ``last_seen_cycle`` for *key*.

        This is the ONLY correct way to update the recurrence counter on an
        existing bookkeeping record without resetting unrelated fields.  Callers
        that use :meth:`set_bookkeeping` to update a single field would silently
        overwrite ``speaker_id`` / ``first_seen_cycle`` with defaults.

        When *key* has no bookkeeping record yet the call creates a minimal
        record (``recurrence_count=1``) — this is a defensive no-op for callers
        that may race with a first registration; normal flow is that
        :meth:`set_bookkeeping` is called first.

        Args:
            key: The indexed-key string (e.g. ``"graph42"``).
            cycle: Current consolidation cycle number.  Written to
                ``last_seen_cycle`` unconditionally (the fact was re-seen
                this cycle).
        """
        with self._lock:
            existing = self._bookkeeping.get(key)
            if existing is None:
                self._bookkeeping[key] = {
                    "speaker_id": "",
                    "first_seen_cycle": cycle,
                    "relation_type": "unknown",
                    "recurrence_count": 1,
                    "last_seen_cycle": cycle,
                }
                return
            existing["recurrence_count"] = existing.get("recurrence_count", 1) + 1
            existing["last_seen_cycle"] = cycle

    def bookkeeping_for_key(self, key: str) -> dict | None:
        """Return the bookkeeping record for *key*, or ``None`` when absent.

        Returns a plain dict with five fields:
        ``{"speaker_id", "first_seen_cycle", "relation_type",
        "recurrence_count", "last_seen_cycle"}``
        when present, ``None`` when the key has never been bookkept.
        Callers may use ``bk = store.bookkeeping_for_key(k) or {}`` as a
        boundary default (compliant with the is-None / empty-is-valid rule)."""
        with self._lock:
            return self._bookkeeping.get(key)

    def iter_bookkeeping(self):
        """Yield ``(key, {"speaker_id": ..., "first_seen_cycle": ...})`` pairs.

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

    def drop_bookkeeping(self, key: str) -> None:
        """Remove the bookkeeping record for *key* (retirement parity).

        Called automatically by :meth:`delete` — callers that retire a key
        via ``delete`` need not call this separately."""
        with self._lock:
            self._bookkeeping.pop(key, None)

    def bookkeeping_count(self) -> int:
        """Number of keys that have a bookkeeping record."""
        with self._lock:
            return len(self._bookkeeping)

    def delete(self, key: str) -> str | None:
        """Drop *key* from every tier in every sub-structure.

        Cleans entries, simhash (via registry), and registry across all tiers
        in one call.  The return value is the *first* tier that held the key
        in any structure, or ``None`` when the key was completely absent.

        The entire compound mutation (entries + registry/simhash + bookkeeping)
        is performed under a single lock acquisition so a concurrent reader
        never observes a half-deleted state across the three structures.
        Nested leaf calls succeed via RLock reentrancy."""
        with self._lock:
            former: str | None = None
            for tier in list(self._entries.keys()):
                if key in self._entries[tier]:
                    del self._entries[tier][key]
                    if former is None:
                        former = tier
            # Drop from registry (removes from active, stale, and _simhash).
            for tier, reg in list(self._registry.items()):
                if reg.knows(key):
                    reg.remove(key)
                    if former is None:
                        former = tier
            # Retire bookkeeping in lockstep so _bookkeeping never drifts.
            self._bookkeeping.pop(key, None)
            return former

    def clear_entries(self) -> int:
        """Remove all entry payloads from the content cache.

        Clears ``_entries`` only — registries (including their simhash maps)
        and ``_bookkeeping`` are left intact so the authoritative key-lifecycle
        state survives the purge.  The intended caller is
        :func:`paramem.server.app._hydrate_memory_store_in_place`, which clears
        the stale pre-fold cache before re-probing active keys against the
        freshly-retrained weights.

        Returns the total number of entries that were dropped across all tiers.
        """
        with self._lock:
            total = sum(len(tier_entries) for tier_entries in self._entries.values())
            self._entries.clear()
            return total

    def move(self, key: str, new_tier: str) -> None:
        """Move *key* from its current tier to *new_tier* atomically.

        Moves the entry payload, simhash fingerprint, and registry entry in one
        operation.  No-op if *key* is already in *new_tier*.

        The entire compound mutation across all three structures is performed
        under a single lock acquisition.  Nested calls to :meth:`tier_of`
        succeed via RLock reentrancy."""
        with self._lock:
            old_tier = self.tier_of(key)
            if old_tier == new_tier:
                if new_tier in self._registry:
                    # Ensure registry entry is present even when entries already are.
                    self._registry[new_tier].add(key)
                return
            # Move entry
            if old_tier is not None and key in self._entries.get(old_tier, {}):
                self._entries.setdefault(new_tier, {})[key] = self._entries[old_tier].pop(key)
            # Move simhash: read from the old tier's registry (covers active + stale).
            fp: int | None = None
            old_reg = self._registry.get(old_tier) if old_tier else None
            if old_reg is not None:
                fp = old_reg.simhash_for(key)
                old_reg.drop_simhash(key)
            if fp is not None:
                self._registry.setdefault(new_tier, KeyRegistry()).set_simhash(key, fp)
            # Move registry entry: remove from old, add to new.
            for tier, reg in list(self._registry.items()):
                if key in reg:
                    if tier != new_tier:
                        reg.remove(key)
            self._registry.setdefault(new_tier, KeyRegistry()).add(key)

    # ------------------------------------------------------------------
    # SimHash fingerprints — public accessors
    # ------------------------------------------------------------------
    def simhash(self, tier: str, key: str) -> int | None:
        """Return the simhash fingerprint for ``(tier, key)``, or ``None``.

        Reads from the registry (covers active AND stale partitions — the
        unified storage design).  Returns ``None`` when the tier has no
        registry or the key has no fingerprint."""
        with self._lock:
            reg = self._registry.get(tier)
            if reg is None:
                return None
            return reg.simhash_for(key)

    def has_simhash(self, tier: str, key: str) -> bool:
        """True when *key* has a stored fingerprint in *tier* (active or stale)."""
        with self._lock:
            reg = self._registry.get(tier)
            if reg is None:
                return False
            return reg.has_simhash(key)

    def _tier_for_simhash(self, key: str) -> str | None:
        """Return the tier whose registry holds *key*'s fingerprint, or ``None``.

        Used by the legacy flat-view setter and the probe confidence gate.
        Scans registry active+stale partitions."""
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

    def tier_simhashes(self, tier: str, *, include_stale: bool) -> dict[str, int]:
        """Return the fingerprint map for *tier*.

        The mandatory ``include_stale`` keyword makes the active-vs-known
        distinction impossible to forget — the original enumeration bug was
        caused by callers using the old ``simhashes_in_tier`` without
        remembering to filter stale keys.

        Args:
            tier: Tier name (e.g. ``"episodic"``).
            include_stale: When ``True``, returns active∪stale fingerprints.
                When ``False``, returns active-only fingerprints.

        Returns:
            A fresh ``dict[str, int]`` (not a live view).  Callers that need
            a mutable live backing dict are using a deprecated pattern —
            use :meth:`put_simhash` for writes.
        """
        with self._lock:
            reg = self._registry.get(tier)
            if reg is None:
                return {}
            return reg._known_simhashes() if include_stale else reg._active_simhashes()

    def replace_simhashes_in_tier(self, tier: str, new_simhashes: dict[str, int]) -> None:
        """Bulk-replace the active simhash fingerprints for *tier*.

        Writes the new active fingerprints directly to the tier's registry.
        Stale fingerprints in the registry are not touched — they are managed
        by :meth:`KeyRegistry.stale`.
        """
        with self._lock:
            reg = self._registry.setdefault(tier, KeyRegistry())
            # Drop all active simhashes from the registry by clearing _simhash
            # directly.  (Only active simhashes are being replaced; stale
            # records stay intact.)
            reg._simhash.clear()
            for key, fp in new_simhashes.items():
                reg.set_simhash(key, fp)

    def simhash_count_in_tier(self, tier: str) -> int:
        """Return the total number of known (active∪stale) fingerprints for *tier*.

        Reads from the registry's ``_known_simhashes()`` — the authoritative
        source for both active and stale fingerprints."""
        with self._lock:
            reg = self._registry.get(tier)
            if reg is None:
                return 0
            return len(reg._known_simhashes())

    # ------------------------------------------------------------------
    # Lifecycle registry
    # ------------------------------------------------------------------
    def registry(self, tier: str) -> KeyRegistry:
        """Return the per-tier :class:`KeyRegistry`, creating it on first access.

        Always returns a :class:`KeyRegistry` — never ``None``.  The
        ``_replay_enabled`` flag governs training behaviour only (whether key
        lifecycle add/stale/remove is recorded); it does not gate registry
        existence.  Callers that need to know whether replay is active should
        check :attr:`replay_enabled` directly.
        """
        with self._lock:
            return self._registry.setdefault(tier, KeyRegistry())

    def load_registry(self, tier: str, registry: KeyRegistry) -> None:
        """Install a pre-loaded :class:`KeyRegistry` for *tier* at boot.

        Raises :class:`RuntimeError` when called on a replay-disabled store —
        loading a registry into a disabled store would silently break the
        gate that downstream paths rely on.  Simhashes are carried inside the
        registry; no separate sync is needed."""
        if not self._replay_enabled:
            raise RuntimeError(
                "MemoryStore: replay is disabled; cannot install registry for tier %r" % tier
            )
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

    def drop_registry(self, tier: str) -> KeyRegistry | None:
        """Remove and return the registry for *tier*.

        Used at end-of-full-cycle to retire interim slots after their keys
        have been promoted into main.  Returns ``None`` when the tier had no
        registry."""
        with self._lock:
            return self._registry.pop(tier, None)

    def active_keys_in_tier(self, tier: str) -> list[str]:
        """Return the active keys for *tier* from the registry."""
        with self._lock:
            reg = self._registry.get(tier)
            return reg.list_active() if reg is not None else []

    def all_active_keys(self) -> list[str]:
        """Every active key across every registered tier."""
        with self._lock:
            return [k for reg in self._registry.values() for k in reg.list_active()]

    def tier_for_active_key(self, key: str) -> str | None:
        """Return the tier that holds *key* in its registry as active, or ``None``.

        Kept distinct from :meth:`tier_of` because the registry and entry
        cache can briefly disagree during a put/move sequence."""
        with self._lock:
            for tier, reg in self._registry.items():
                if key in reg:
                    return tier
            return None

    def is_known(self, key: str) -> bool:
        """True when *key* is active OR stale in any tier's registry.

        KNOWN-legitimacy analogue of ``key in reg`` (active-only).  Returns
        False when replay is disabled (no registries) or the key is absent from
        both partitions of every tier.  Use for orphan checks and bookkeeping
        retention; use :meth:`tier_for_active_key` / :meth:`all_active_keys` for
        serving/enumeration.
        """
        if not self._replay_enabled:
            return False
        with self._lock:
            for reg in self._registry.values():
                if reg.knows(key):
                    return True
            return False

    def tier_for_known_key(self, key: str) -> str | None:
        """Return the tier whose registry tracks *key* as active OR stale.

        KNOWN-legitimacy analogue of :meth:`tier_for_active_key`.  Returns
        ``None`` when replay is disabled or no tier knows *key* in either
        partition.
        """
        if not self._replay_enabled:
            return None
        with self._lock:
            for tier, reg in self._registry.items():
                if reg.knows(key):
                    return tier
            return None

    def all_known_keys(self) -> list[str]:
        """Every active ∪ stale key across every registered tier.

        Equivalent to ``all_active_keys() + all_stale_keys()`` but expressed
        via :meth:`KeyRegistry.list_known` so the union logic has a single
        definition.  Returns an empty list when replay is disabled.
        """
        if not self._replay_enabled:
            return []
        with self._lock:
            return [k for reg in self._registry.values() for k in reg.list_known()]

    def is_stale(self, key: str) -> bool:
        """Return True when *key* is stale in any tier's registry.

        Delegates to :meth:`KeyRegistry.is_stale` on the owning tier.
        Returns False when replay is disabled (no registries) or the key
        is not found in any tier's stale partition.
        """
        if not self._replay_enabled:
            return False
        with self._lock:
            for reg in self._registry.values():
                if reg.is_stale(key):
                    return True
            return False

    def all_stale_keys(self) -> list[str]:
        """Every stale key across every registered tier."""
        if not self._replay_enabled:
            return []
        with self._lock:
            return [k for reg in self._registry.values() for k in reg.list_stale()]

    def discard_keys(self, keys: list[str], *, mode: str) -> None:
        """Mutate the in-memory registry (and its simhash map) for *keys*.

        Supports two modes:

        ``mode="erase"`` (hard removal, used by ``/forget``):
            For each key, call :meth:`KeyRegistry.remove` on every tier's
            registry that holds it (active or stale).  ``remove`` drops both
            the key from ``_active_keys``/``_stale`` and its fingerprint from
            ``_simhash``.

        ``mode="stale"`` (soft removal, used by the fold dedup write-back):
            For each key, call :meth:`KeyRegistry.stale` on the owning tier.
            The stale transition automatically carries the active simhash into
            the stale record (encapsulated in :meth:`KeyRegistry.stale`), so
            the fingerprint is retained for the stale-echo probe.

        Guards ``replay_enabled``: when replay is disabled, all mutations are
        no-ops and this method returns without raising.

        This is an **in-memory** mutation only.  Callers are responsible for
        their own disk saves (the registry files carry the unified simhash now).

        The entire mutation is performed under a single lock acquisition.
        Nested calls to :meth:`tiers_with_registry` succeed via RLock
        reentrancy.
        """
        if not self._replay_enabled:
            return
        with self._lock:
            if mode == "erase":
                for tier_name in self.tiers_with_registry():
                    reg = self._registry.get(tier_name)
                    if reg is None:
                        continue
                    for key in keys:
                        if reg.knows(key):
                            reg.remove(key)
            elif mode == "stale":
                for key in keys:
                    for tier_name in self.tiers_with_registry():
                        reg = self._registry.get(tier_name)
                        if reg is not None and key in reg:
                            reg.stale(key)
                            # The stale transition moved the active simhash into
                            # the stale record inside the registry — single
                            # encapsulated call.
                            break  # single-tier-ownership invariant
            else:
                raise ValueError(
                    f"MemoryStore.discard_keys: unknown mode {mode!r}; expected 'erase' or 'stale'"
                )

    # ------------------------------------------------------------------
    # Snapshot / restore — cycle-resume rollback rope
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Deep-copy the entry and simhash state for rollback.

        The ``"simhash"`` entry captures active∪stale fingerprints per tier
        from the registry (the unified source of truth), so fingerprints survive
        rollback correctly.

        Registries are NOT included — they have their own persistence
        lifecycle (KeyRegistry.save / KeyRegistry.load to per-tier
        ``indexed_key_registry.json`` files).  The rollback caller is
        responsible for restoring registries from disk separately.

        ``_bookkeeping`` is NOT included — it is reloaded from
        ``key_metadata.json`` on boot."""
        with self._lock:
            simhash_snap: dict[str, dict[str, int]] = {}
            for tier, reg in self._registry.items():
                known = reg._known_simhashes()
                if known:
                    simhash_snap[tier] = known
            return {
                "entries": copy.deepcopy(self._entries),
                "simhash": simhash_snap,
            }

    def restore(self, snap: dict) -> None:
        """Restore entry and simhash state from :meth:`snapshot` output.

        Re-seeds each tier's registry with the active fingerprints from the
        snapshot map.  Stale fingerprints are reloaded from the registry
        on-disk during the separate registry-restore step (callers are
        responsible for that).

        The entire compound rebind is performed under a single lock acquisition
        so a concurrent reader never observes a half-restored state."""
        with self._lock:
            self._entries = copy.deepcopy(snap["entries"])
            snap_simhash: dict[str, dict[str, int]] = snap.get("simhash", {})
            # Clear existing registry active simhashes and re-seed from the snapshot.
            for tier, fp_map in snap_simhash.items():
                reg = self._registry.setdefault(tier, KeyRegistry())
                reg._simhash.clear()
                for key, fp in fp_map.items():
                    if key not in reg._stale:
                        reg._simhash[key] = fp

    # ------------------------------------------------------------------
    # Probe — resolve {key → entry} for inference, with optional source fallback
    # ------------------------------------------------------------------
    def probe(
        self,
        keys_by_adapter: dict[str, list[str]],
        *,
        source=None,
        speaker_id: str | None = None,
        memoize: bool = True,
        speaker_resolver=None,
    ) -> dict[str, dict | None]:
        """Resolve *keys_by_adapter* to flat ``{key → result | None}``.

        NOTE — this method is deliberately NOT wrapped in ``self._lock``.
        It calls ``source.probe(...)``, which is a GPU ``model.generate()``
        call that may block for seconds.  Holding the store lock across GPU
        work would stall every concurrent store reader for the duration of the
        GPU call.  All reads inside probe go through individually locked leaf
        methods — ``self.get``, ``self.bookkeeping_for_key``, ``self.simhash``,
        ``self._tier_for_simhash``, ``self.put``, ``self.set_bookkeeping`` —
        so no raw access to ``_entries``, ``_registry``, or ``_bookkeeping``
        is made.  Each read acquires and releases the lock independently; the
        lock is never held across the GPU call.

        Cache hits are served directly from the store.  Misses are delegated
        to *source* (a :class:`paramem.memory.source.MemorySource`)
        when supplied; the source result is memoized back into the store unless
        *memoize* is False (typically when ``inference.preload_cache`` is off
        and the operator wants the cache to stay empty).

        *speaker_id* is a defense-in-depth filter: any entry whose
        ``speaker_id`` mismatches the requested speaker is returned as
        ``None`` and logged at WARN — the router should have intersected
        upstream.

        Returns the canonical inference result shape per hit:
        ``{key, subject, predicate, object, speaker_id, first_seen_cycle,
        confidence=1.0, fact_text, raw_output}``.  Misses (or
        source-failure dicts carrying ``failure_reason``) pass through
        unrendered.

        *speaker_resolver* is an optional callable ``(str) -> str`` that maps
        raw subject/object tokens to display names at the fact-render boundary.
        Applied in BOTH render paths (cache-hit via ``entry_fact_text`` +
        source passthrough via re-render before storing the result).  The
        memoized stash is SPO-only (no ``fact_text``) so a later cache hit
        always re-renders with the then-current resolver — no stale-name
        caching.  ``None`` (default) → byte-identical behaviour.

        **speaker_id asymmetry (do not remove this comment):**
        The CACHE-HIT branch reads ``speaker_id`` from ``_bookkeeping`` (the
        single authoritative source — entries are SPO-only after this fix).
        The SOURCE-RESULT branch reads ``src.get("speaker_id","")`` as-is.
        On the train/weight path ``WeightMemorySource`` emits NO speaker_id
        (``probe.py:87-99``), so the filter is a no-op there; scoping relies
        on the router upstream + ``_render`` joining from ``_bookkeeping``.
        The disk/simulate source DOES carry speaker_id — redundant belt-and-
        suspenders for the simulate speaker filter.  Do NOT collapse the two
        branches into one: the asymmetry is load-bearing.
        """
        import json as _json

        from paramem.memory.entry import (
            DEFAULT_CONFIDENCE_THRESHOLD,
            entry_fact_text,
            verify_confidence,
        )

        def _confidence_gate(key: str, entry: dict) -> float | None:
            """Return the SimHash confidence for *key* against *entry*, or None
            when no gate applies (replay disabled or key has no fingerprint).

            Called on both cache-hit and source-result branches so the gate is
            applied at exactly one place regardless of which read path served
            the entry.  Returns the computed confidence float when the gate
            applies (which may be below threshold), or ``None`` to signal
            pass-through (no verification needed — replay-off or no fingerprint).

            The fingerprint is looked up by scanning all registry tiers via
            :meth:`_tier_for_simhash` so a key whose simhash was stored under
            a different tier from the one it was requested under (e.g. an
            interim slot promoted to main) is still verified correctly.

            Invariant: never uses truthiness on registry or sub-dicts —
            all presence checks use explicit ``in`` / ``is None``."""
            if not self._replay_enabled:
                return None
            owning_simhash_tier = self._tier_for_simhash(key)
            if owning_simhash_tier is None:
                # Key has no fingerprint — replay enabled but no hash stored
                # (e.g. fresh tier before first consolidation).  Pass through.
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

        def _render(key: str, entry: dict) -> dict | None:
            # speaker_id / first_seen_cycle come from _bookkeeping — the single
            # authoritative source.  Entries hold SPO only (inference cache
            # contract); save-path working entries carry them for build_graph_for_tier
            # but those are consumed within the cycle, not at inference time.
            #
            # SimHash confidence gate: when replay is enabled and the key has a
            # stored fingerprint, compute the real confidence and drop entries
            # that fall below DEFAULT_CONFIDENCE_THRESHOLD.  This makes the
            # cache-hit path identical in gate semantics to the source path
            # (WeightMemorySource / finalize_recalled).  Returns None on fail
            # so the caller treats the key as a miss — same as source drop.
            confidence = _confidence_gate(key, entry)
            if confidence is not None and confidence < DEFAULT_CONFIDENCE_THRESHOLD:
                logger.debug(
                    "MemoryStore.probe: cache-hit key %r dropped by confidence gate "
                    "(%.3f < %.3f threshold)",
                    key,
                    confidence,
                    DEFAULT_CONFIDENCE_THRESHOLD,
                )
                return None
            # Use the computed confidence when the gate applied; fall back to
            # 1.0 when replay is off or no fingerprint exists (pass-through).
            rendered_confidence = confidence if confidence is not None else 1.0
            bk = self.bookkeeping_for_key(key) or {}
            base = {
                "key": key,
                "subject": entry.get("subject", ""),
                "predicate": entry.get("predicate", ""),
                "object": entry.get("object", ""),
            }
            spk = bk.get("speaker_id", "")
            return {
                **base,
                "speaker_id": spk,
                "first_seen_cycle": bk.get("first_seen_cycle", 0),
                "confidence": rendered_confidence,
                "fact_text": entry_fact_text({**base, "speaker_id": spk}, resolve=speaker_resolver),
                "raw_output": _json.dumps(base),
            }

        results: dict[str, dict | None] = {}
        misses: dict[str, list[str]] = {}

        for tier, keys in keys_by_adapter.items():
            for key in keys:
                entry = self.get(key)
                if entry is None:
                    misses.setdefault(tier, []).append(key)
                    continue
                # Defense-in-depth speaker filter — read speaker from
                # _bookkeeping (not from the entry: entries are SPO-only).
                bk_spk = (self.bookkeeping_for_key(key) or {}).get("speaker_id", "")
                if speaker_id is not None and bk_spk and bk_spk != speaker_id:
                    logger.warning(
                        "MemoryStore.probe: speaker_id mismatch for key %s "
                        "(bookkeeping=%r, requested=%r) — returning None",
                        key,
                        bk_spk,
                        speaker_id,
                    )
                    results[key] = None
                    continue
                results[key] = _render(key, entry)

        if misses:
            if source is None:
                for tier_keys in misses.values():
                    for key in tier_keys:
                        results.setdefault(key, None)
                return results

            source_results = source.probe(misses)
            for tier, keys in misses.items():
                for key in keys:
                    src = source_results.get(key)
                    if src is None:
                        results[key] = None
                        continue
                    if isinstance(src, dict) and "failure_reason" in src:
                        results[key] = src
                        continue
                    # SOURCE-RESULT speaker filter — reads src["speaker_id"]
                    # directly.  See docstring asymmetry note: WeightMemorySource
                    # emits no speaker_id so this is a no-op on the train path;
                    # DiskMemorySource carries it for simulate-path belt-and-suspenders.
                    if (
                        speaker_id is not None
                        and isinstance(src, dict)
                        and src.get("speaker_id", "")
                        and src["speaker_id"] != speaker_id
                    ):
                        results[key] = None
                        continue
                    # SOURCE-RESULT confidence gate — mirrors the cache-hit gate
                    # so that entries admitted through the source path are also
                    # confidence-verified against the store's fingerprints before
                    # being memoized or returned.  This covers the boot-preload
                    # path where WeightMemorySource is constructed without a
                    # registry (app.py belt-and-suspenders fix adds one, but the
                    # store-boundary gate is the hermetic authority).
                    if isinstance(src, dict):
                        src_confidence = _confidence_gate(key, src)
                        if (
                            src_confidence is not None
                            and src_confidence < DEFAULT_CONFIDENCE_THRESHOLD
                        ):
                            logger.debug(
                                "MemoryStore.probe: source-result key %r dropped by "
                                "confidence gate (%.3f < %.3f threshold)",
                                key,
                                src_confidence,
                                DEFAULT_CONFIDENCE_THRESHOLD,
                            )
                            results[key] = None
                            continue
                        # Patch the rendered confidence onto the source result so
                        # callers always see the real score (not a stale 1.0 from
                        # WeightMemorySource when it ran without a registry).
                        if src_confidence is not None:
                            src = dict(src)
                            src["confidence"] = src_confidence
                    # SOURCE-RESULT speaker resolver — when speaker_resolver is set,
                    # re-render fact_text from SPO through entry_fact_text so
                    # display names replace raw speaker{N} tokens.  The memoized
                    # stash below is SPO-only (no fact_text) so a later cache hit
                    # re-renders with the then-current resolver — no stale caching.
                    if speaker_resolver is not None and isinstance(src, dict):
                        src = dict(src)
                        src["fact_text"] = entry_fact_text(src, resolve=speaker_resolver)
                    results[key] = src
                    if memoize and isinstance(src, dict):
                        # Stash the raw SPO entry back into the cache (content only).
                        # Register=False — the registry was established at boot.
                        raw_entry = {
                            "key": key,
                            "subject": src.get("subject", ""),
                            "predicate": src.get("predicate", ""),
                            "object": src.get("object", ""),
                        }
                        self.put(tier, key, raw_entry, register=False)
                        # Back-fill _bookkeeping when the source carried provenance
                        # and the key wasn't already bookkept (cold key not in
                        # key_metadata.json at boot — disk source path).  No-op on
                        # the weight path (WeightMemorySource carries no speaker_id).
                        # relation_type is not carried by the probe source; default
                        # to "unknown" — the backfill job corrects it on first
                        # consolidation cycle.
                        if self.bookkeeping_for_key(key) is None:
                            src_spk = src.get("speaker_id", "")
                            src_fsc = src.get("first_seen_cycle", 0)
                            if src_spk or src_fsc:
                                self.set_bookkeeping(
                                    key,
                                    speaker_id=src_spk,
                                    first_seen_cycle=src_fsc,
                                    relation_type=src.get("relation_type", "unknown"),
                                    recurrence_count=1,
                                    last_seen_cycle=src_fsc,
                                    allow_empty_speaker=True,
                                )

        return results

    # ------------------------------------------------------------------
    # On-disk registries — load registries + simhashes from the adapter dir
    # ------------------------------------------------------------------
    @staticmethod
    def read_registries_from_disk(adapter_dir) -> "dict[str, KeyRegistry]":
        """Read per-tier ``indexed_key_registry.json`` files from disk into a
        fresh ``dict[str, KeyRegistry]`` without touching any live store.

        This is the store-free counterpart to
        :meth:`load_registries_from_disk`.  It is used by
        :func:`paramem.server.app._build_store_contents` to build new
        registry structures off-store before atomically publishing them via
        :meth:`swap`.  The boot / in-process-reload path still uses the
        instance method (which delegates here and then installs).

        Reads:

        * ``<adapter_dir>/<tier>/indexed_key_registry.json`` for each main tier
          and every ``episodic_interim_<stamp>`` slot.

        The registry file carries the unified simhash map (active∪stale
        fingerprints) in the ``"simhash"`` key.

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
        from pathlib import Path

        from paramem.memory.interim_adapter import iter_interim_dirs
        from paramem.training.key_registry import KeyRegistry

        adapter_dir = Path(adapter_dir)
        registries: dict[str, KeyRegistry] = {}

        # Main tiers — registries live at <adapter_dir>/<tier>/indexed_key_registry.json
        for tier in ("episodic", "semantic", "procedural"):
            reg_path = adapter_dir / tier / "indexed_key_registry.json"
            registries[tier] = KeyRegistry.load(reg_path)

        # Interim tiers — dynamic; loaded when their dirs exist.  Tier key is
        # the PEFT adapter name so callers using ``peft_config`` keys can
        # address the store consistently.
        for interim_name, interim_dir in iter_interim_dirs(adapter_dir):
            reg_path = interim_dir / "indexed_key_registry.json"
            registries[interim_name] = KeyRegistry.load(reg_path)

        return registries

    def load_registries_from_disk(self, adapter_dir) -> None:
        """Load per-tier ``indexed_key_registry.json`` into the store.

        Reads:

        * ``<adapter_dir>/<tier>/indexed_key_registry.json`` for each main tier
          and every ``episodic_interim_<stamp>`` slot.

        The registry file now carries the unified simhash map (active∪stale
        fingerprints) in the ``"simhash"`` key.  The separate
        ``simhash_registry.json`` file is no longer read — it has been
        eliminated.

        Entry payloads (subject/predicate/object/speaker_id) are NOT loaded
        here — that is the responsibility of the mode-specific
        :class:`paramem.memory.source.MemorySource` (weight probe in
        train mode; encrypted graph.json read in simulate mode).

        No-op when the store has ``replay_enabled=False`` (registries are not
        tracked).

        Delegates disk reads to :meth:`read_registries_from_disk` so the
        on-disk parsing logic is shared with the store-free builder path
        (:func:`paramem.server.app._build_store_contents`).
        """
        if not self._replay_enabled:
            return

        registries = MemoryStore.read_registries_from_disk(adapter_dir)
        for tier, reg in registries.items():
            self.load_registry(tier, reg)

    def load_bookkeeping_from_disk(self, key_metadata_path) -> dict:
        """Load per-key bookkeeping into ``_bookkeeping`` from ``key_metadata.json``.

        Sole boot loader for the per-key bookkeeping section of
        ``key_metadata.json``.  Runs unconditionally at lifespan boot (after
        ``load_registries_from_disk``) via ``app.py:4449`` — entry-independent,
        so it no longer requires entries to already exist.  Under
        ``inference.preload_cache=False`` this is the ONLY write to provenance
        state at boot, and is sufficient for the router's speaker index.

        Populates ``_bookkeeping`` only (via :meth:`set_bookkeeping`).  DOES
        NOT touch ``_entries`` — the old ``setdefault_entry`` parasitic write
        that created payload-less stub entries has been removed.  Bookkeeping
        presence MUST NOT manufacture a content cache hit.

        Fields loaded: ``speaker_id``, ``first_seen_cycle``, ``relation_type``,
        ``recurrence_count``, ``last_seen_cycle``.
        Keys absent from every tier registry (orphans — slot wiped or never
        existed) are skipped but counted in the return dict.  Pre-refactor
        files that lack any field are upgraded in-memory with defaults
        (``legacy_upgraded`` counter):
        - ``speaker_id``: ``""``
        - ``first_seen_cycle``: ``0``
        - ``relation_type``: ``"unknown"``
        - ``recurrence_count``: ``1``  (a legacy key has been seen at least once)
        - ``last_seen_cycle``: ``first_seen_cycle`` (a legacy key that was never
          re-seen was last seen when first minted)
        Keys with ``relation_type="unknown"`` will be assigned the correct type
        on the next consolidation cycle when the triple is re-encountered.

        Returns ``{loaded, orphaned, legacy_upgraded}`` for ``/status`` and
        diagnostics.  Missing file → all zeros, no-op (fresh install).
        """
        from pathlib import Path

        path = Path(key_metadata_path)
        if not path.exists():
            return {"loaded": 0, "orphaned": 0, "legacy_upgraded": 0}

        import json

        from paramem.backup.encryption import read_maybe_encrypted

        metadata = json.loads(read_maybe_encrypted(path).decode("utf-8"))
        loaded = 0
        orphaned = 0
        legacy_upgraded = 0
        for key, key_meta in metadata.get("keys", {}).items():
            tier = self.tier_for_known_key(key)
            if tier is None:
                orphaned += 1
                continue
            _fsc = key_meta.get("first_seen_cycle", 0)
            if (
                "speaker_id" not in key_meta
                or "first_seen_cycle" not in key_meta
                or "relation_type" not in key_meta
                or "recurrence_count" not in key_meta
                or "last_seen_cycle" not in key_meta
            ):
                legacy_upgraded += 1
            self.set_bookkeeping(
                key,
                speaker_id=key_meta.get("speaker_id", ""),
                first_seen_cycle=_fsc,
                relation_type=key_meta.get("relation_type", "unknown"),
                recurrence_count=key_meta.get("recurrence_count", 1),
                # Legacy default: last_seen == first_seen (key never re-seen).
                last_seen_cycle=key_meta.get("last_seen_cycle", _fsc),
                allow_empty_speaker=True,
            )
            loaded += 1
        return {"loaded": loaded, "orphaned": orphaned, "legacy_upgraded": legacy_upgraded}

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """Per-tier ``{key_count, simhash_count, registry_active}`` map."""
        with self._lock:
            out: dict[str, dict] = {}
            for tier in self._entries:
                out.setdefault(tier, {})["key_count"] = len(self._entries[tier])
            for tier, reg in self._registry.items():
                known = reg._known_simhashes()
                if known:
                    out.setdefault(tier, {})["simhash_count"] = len(known)
                active = len(reg.list_active())
                if self._replay_enabled:
                    out.setdefault(tier, {})["registry_active"] = active
            return out

    def swap(
        self,
        new_entries: dict[str, dict[str, dict]],
        new_registry: dict[str, KeyRegistry],
        new_bookkeeping: dict[str, dict],
    ) -> None:
        """Atomically rebind all three mutable structures in a single locked operation.

        This is the atomic-publish primitive for phase-2 consolidation.
        Callers build the three new structures entirely off-store, then call
        ``swap`` to publish them in one locked rebind.  No reader ever observes
        a torn or half-rebuilt state across ``_entries``, ``_registry``, and
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
