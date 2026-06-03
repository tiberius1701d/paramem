"""Per-tier indexed-key memory store.

Public API: :class:`MemoryStore`.

Private helper :class:`_LegacyFlatCacheView` exists ONLY to keep the
``ConsolidationLoop.indexed_key_cache`` deprecation accessor working
while ``tests/`` migrate off the legacy attribute names.  It is not
part of the supported API; see the TODO at
``ConsolidationLoop.indexed_key_cache``.


Single source of truth for the answer content, integrity fingerprints, and
lifecycle registries of every indexed key the system holds in RAM.  Replaces
the previous mixed-shape state on :class:`ConsolidationLoop`:

* ``indexed_key_cache: dict[str, dict]`` — flat, all tiers in one bucket; tier
  was recovered indirectly by scanning every registry.
* ``episodic_simhash``, ``semantic_simhash``, ``procedural_simhash`` — three
  separate flat dicts, asymmetric with the registry shape (no per-interim
  bucket; interim slots shared the episodic one).
* ``indexed_key_registry: Optional[dict[str, KeyRegistry]]`` — the only
  structure that was already per-tier; folded in here for symmetry.

The unified shape is ``tier → key → value`` for all three concerns:

* :attr:`MemoryStore.entries_in_tier` returns ``dict[key, entry_payload]``.
* :attr:`MemoryStore.simhashes_in_tier` returns ``dict[key, int]`` — usable
  verbatim by :func:`paramem.memory.entry.verify_confidence` and
  friends.
* :meth:`MemoryStore.registry` returns the tier's :class:`KeyRegistry`.

Tier ownership of a key is the single source of truth; an indexed key
belongs to exactly one tier.  Cross-tier lookups (``get``, ``has``,
``tier_of``) scan tier-first then key — O(tier_count); the tier count is
small (3 main + N interim slots, typically ≤ 10).

When replay is disabled (``ConsolidationLoop`` constructed with
``indexed_key_replay_enabled=False``), :meth:`registry` returns ``None``
preserving the load-bearing "registry-is-None means replay-off" gate that
training paths check.  Cache and simhash are always operational.

**Content cache vs. bookkeeping — read before editing:**

:attr:`MemoryStore._entries` is a PURE INFERENCE CONTENT CACHE.  An entry
slot exists in ``_entries`` only when SPO content is materialised in RAM —
written by :meth:`put` (boot preload or on-miss memoize).  Missing entry =
cache miss.  Under ``inference.preload_cache=False`` the entries are empty by
design → every key is a clean miss → the source is always probed → restores
the documented contract.

The per-key bookkeeping fields ``speaker_id`` and ``first_seen_cycle`` live in
:attr:`MemoryStore._bookkeeping` — a flat ``{key → {speaker_id, first_seen_cycle}}``
dict SEPARATE from ``_entries``.  Populated by
:meth:`load_bookkeeping_from_disk` at boot (unconditionally; entry-independent).
Never enters :meth:`KeyRegistry.save_bytes`, :meth:`snapshot`, or any hash
path.

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
simhash state for the cycle-resume rollback rope.  Registries persist via
their own :meth:`KeyRegistry.save` / :meth:`KeyRegistry.load` lifecycle and
are restored separately from disk on rollback.  ``_bookkeeping`` is NOT in
the snapshot — it is reloaded from ``key_metadata.json`` on boot.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterator

from paramem.training.key_registry import KeyRegistry

logger = logging.getLogger(__name__)


class MemoryStore:
    """Per-tier {entries, simhash, registry} for the indexed-key memory layer."""

    def __init__(self, *, replay_enabled: bool = True) -> None:
        self._replay_enabled = replay_enabled
        # tier -> key -> entry payload dict.  PURE INFERENCE CONTENT CACHE —
        # an entry slot exists only when SPO is materialised.  See module
        # docstring for the "save-path working entries" exception.
        self._entries: dict[str, dict[str, dict]] = {}
        # tier -> key -> 64-bit simhash fingerprint
        self._simhash: dict[str, dict[str, int]] = {}
        # tier -> KeyRegistry (None overall when replay is disabled — the
        # gate every training path checks to decide whether to record key
        # lifecycle at all).
        self._registry: dict[str, KeyRegistry] | None = {} if replay_enabled else None
        # Per-key provenance bookkeeping — SEPARATE from _entries.
        # key -> {"speaker_id": str, "first_seen_cycle": int}
        # Populated by load_bookkeeping_from_disk at boot.  Never enters
        # snapshot/restore or KeyRegistry.save_bytes — stays out of the
        # hash-frozen slot-identity path.
        self._bookkeeping: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Replay-enabled gate
    # ------------------------------------------------------------------
    @property
    def replay_enabled(self) -> bool:
        """True when the lifecycle registry is active.  Mirrors the legacy
        ``ConsolidationLoop.indexed_key_registry is not None`` check."""
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
        for tier_entries in self._entries.values():
            if key in tier_entries:
                return tier_entries[key]
        return None

    def has(self, key: str) -> bool:
        """Membership check across all tiers."""
        for tier_entries in self._entries.values():
            if key in tier_entries:
                return True
        return False

    def tier_of(self, key: str) -> str | None:
        """Return the tier that owns *key* in the entry store, or ``None``."""
        for tier, tier_entries in self._entries.items():
            if key in tier_entries:
                return tier
        return None

    def entries_in_tier(self, tier: str) -> dict[str, dict]:
        """Return the ``key -> entry`` map for *tier*.  Empty dict on miss.

        Returned dict is the live internal mapping — callers must not mutate
        it.  Use :meth:`put` / :meth:`delete` for writes."""
        return self._entries.get(tier, {})

    def iter_entries(self) -> Iterator[tuple[str, str, dict]]:
        """Yield ``(tier, key, entry)`` for every entry.

        Stable across calls so long as no mutation happens between yields.
        Use ``list(...)`` first if mutating while iterating."""
        for tier, tier_entries in self._entries.items():
            for key, entry in tier_entries.items():
                yield tier, key, entry

    def __len__(self) -> int:
        """Total number of keys held across all tiers."""
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

        If *simhash* is supplied, the per-tier fingerprint is written in the
        same call.  When *register* is True and replay is enabled, the key is
        added to the tier's lifecycle registry.

        Caller is responsible for ensuring *key* is unique across tiers — if
        the key currently belongs to a different tier, call :meth:`move`
        rather than ``put`` to keep the three sub-structures (entries, simhash,
        registry) in agreement.
        """
        self._entries.setdefault(tier, {})[key] = entry
        if simhash is not None:
            self._simhash.setdefault(tier, {})[key] = simhash
        if register and self._registry is not None:
            self._registry.setdefault(tier, KeyRegistry()).add(key)

    # ------------------------------------------------------------------
    # Per-key bookkeeping — speaker_id / first_seen_cycle
    # SEPARATE from _entries; never in snapshot or KeyRegistry.save_bytes.
    # ------------------------------------------------------------------
    def set_bookkeeping(self, key: str, *, speaker_id: str, first_seen_cycle: int) -> None:
        """Store or update the bookkeeping record for *key*.

        Both ``speaker_id`` and ``first_seen_cycle`` are mandatory (one
        mandatory tier, zero optional buckets).  Idempotent — safe to call
        on every new-key write and on every boot reload.

        Does NOT touch ``_entries`` — bookkeeping presence MUST NOT
        manufacture a content cache hit."""
        self._bookkeeping[key] = {
            "speaker_id": speaker_id,
            "first_seen_cycle": first_seen_cycle,
        }

    def bookkeeping_for_key(self, key: str) -> dict | None:
        """Return the bookkeeping record for *key*, or ``None`` when absent.

        Returns a plain dict ``{"speaker_id": str, "first_seen_cycle": int}``
        when present, ``None`` when the key has never been bookkept.
        Callers may use ``bk = store.bookkeeping_for_key(k) or {}`` as a
        boundary default (compliant with the is-None / empty-is-valid rule)."""
        return self._bookkeeping.get(key)

    def iter_bookkeeping(self):
        """Yield ``(key, {"speaker_id": ..., "first_seen_cycle": ...})`` pairs.

        Preload-independent — populated by :meth:`load_bookkeeping_from_disk`
        at boot regardless of ``inference.preload_cache``.  Used by
        :meth:`QueryRouter.reload` to build the speaker → keys index without
        touching ``_entries``."""
        yield from self._bookkeeping.items()

    def drop_bookkeeping(self, key: str) -> None:
        """Remove the bookkeeping record for *key* (retirement parity).

        Called automatically by :meth:`delete` — callers that retire a key
        via ``delete`` need not call this separately."""
        self._bookkeeping.pop(key, None)

    def bookkeeping_count(self) -> int:
        """Number of keys that have a bookkeeping record."""
        return len(self._bookkeeping)

    def delete(self, key: str) -> str | None:
        """Drop *key* from every tier in every sub-structure.

        Cleans entries, simhash, and registry across all tiers in one call —
        callers that produced inconsistent state (e.g. entry in one tier,
        simhash in another via legacy seeders) are normalised here.  The
        return value is the *first* tier that held the key in any structure,
        or ``None`` when the key was completely absent."""
        former: str | None = None
        for tier in list(self._entries.keys()):
            if key in self._entries[tier]:
                del self._entries[tier][key]
                if former is None:
                    former = tier
        for tier in list(self._simhash.keys()):
            if key in self._simhash[tier]:
                self._simhash[tier].pop(key, None)
                if former is None:
                    former = tier
        if self._registry is not None:
            for tier, reg in list(self._registry.items()):
                if key in reg:
                    reg.remove(key)
                    if former is None:
                        former = tier
        # Retire bookkeeping in lockstep so _bookkeeping never drifts.
        self._bookkeeping.pop(key, None)
        return former

    def move(self, key: str, new_tier: str) -> None:
        """Move *key* from its current tier to *new_tier* atomically.

        No-op if *key* is already in *new_tier*.  Surfaces an inconsistency
        if entries / simhash / registry disagree about the source tier — the
        first found tier wins for the move, the others are cleaned silently
        but logged at WARN."""
        old_tier = self.tier_of(key)
        if old_tier == new_tier:
            if self._registry is not None and new_tier in self._registry:
                # Ensure registry entry is present even when entries already are.
                self._registry[new_tier].add(key)
            return
        # Move entry
        if old_tier is not None and key in self._entries.get(old_tier, {}):
            self._entries.setdefault(new_tier, {})[key] = self._entries[old_tier].pop(key)
        # Move simhash (may live in a different tier from the entry if a
        # caller skipped using move — log if so)
        for tier, sims in list(self._simhash.items()):
            if key in sims:
                if tier != old_tier:
                    logger.warning(
                        "MemoryStore.move(%s): simhash was in tier %r but entry in %r — "
                        "migrating simhash to %r",
                        key,
                        tier,
                        old_tier,
                        new_tier,
                    )
                self._simhash.setdefault(new_tier, {})[key] = sims.pop(key)
        # Move registry entry
        if self._registry is not None:
            for tier, reg in list(self._registry.items()):
                if key in reg:
                    if tier != new_tier:
                        reg.remove(key)
            self._registry.setdefault(new_tier, KeyRegistry()).add(key)

    # ------------------------------------------------------------------
    # SimHash fingerprints
    # ------------------------------------------------------------------
    def simhash(self, tier: str, key: str) -> int | None:
        """Return the simhash fingerprint for ``(tier, key)``, or ``None``."""
        return self._simhash.get(tier, {}).get(key)

    def has_simhash(self, tier: str, key: str) -> bool:
        return key in self._simhash.get(tier, {})

    def _tier_for_simhash(self, key: str) -> str | None:
        """Return the tier whose simhash dict holds *key*, or ``None``.

        Used by the legacy flat-view setter to keep tier ownership
        consistent when a caller primes one tier's simhash and then
        writes the entry through the view."""
        for tier, sims in self._simhash.items():
            if key in sims:
                return tier
        return None

    def put_simhash(self, tier: str, key: str, fingerprint: int) -> None:
        """Write the simhash fingerprint for ``(tier, key)``.  Does not touch
        the entry or registry."""
        self._simhash.setdefault(tier, {})[key] = fingerprint

    def delete_simhash(self, tier: str, key: str) -> None:
        self._simhash.get(tier, {}).pop(key, None)

    def simhashes_in_tier(self, tier: str) -> dict[str, int]:
        """Return the per-tier ``key -> simhash`` map.

        This is the shape :func:`verify_confidence` and
        :func:`probe_keys_grouped_by_adapter` accept as their ``registry``
        argument.  Returns the live internal mapping (lazily created on
        first access) so callers iterating it observe the current state
        and legacy mutators (``simhashes_in_tier("X")[k] = v``) propagate.
        Prefer :meth:`put_simhash` / :meth:`delete_simhash` in new code."""
        return self._simhash.setdefault(tier, {})

    def replace_simhashes_in_tier(self, tier: str, new_simhashes: dict[str, int]) -> None:
        """Bulk-replace the per-tier simhash dict.

        Used after a tier rebuild that produces a fresh registry from
        ``build_registry``."""
        self._simhash[tier] = dict(new_simhashes)

    def simhash_count_in_tier(self, tier: str) -> int:
        return len(self._simhash.get(tier, {}))

    # ------------------------------------------------------------------
    # Lifecycle registry
    # ------------------------------------------------------------------
    def registry(self, tier: str) -> KeyRegistry | None:
        """Return the per-tier :class:`KeyRegistry`, or ``None`` when replay
        is disabled.

        Auto-creates an empty registry for *tier* on first access when replay
        is enabled — matches the legacy lazy-create behaviour at
        :meth:`ConsolidationLoop._tier_registry`."""
        if self._registry is None:
            return None
        return self._registry.setdefault(tier, KeyRegistry())

    def load_registry(self, tier: str, registry: KeyRegistry) -> None:
        """Install a pre-loaded :class:`KeyRegistry` for *tier* at boot.

        Raises :class:`RuntimeError` when called on a replay-disabled store —
        loading a registry into a disabled store would silently break the
        gate that downstream paths rely on."""
        if self._registry is None:
            raise RuntimeError(
                "MemoryStore: replay is disabled; cannot install registry for tier %r" % tier
            )
        self._registry[tier] = registry

    def has_registry(self, tier: str) -> bool:
        if self._registry is None:
            return False
        return tier in self._registry

    def tiers_with_registry(self) -> list[str]:
        """Tiers for which a registry has been allocated.  Empty when replay
        is disabled."""
        if self._registry is None:
            return []
        return list(self._registry.keys())

    def drop_registry(self, tier: str) -> KeyRegistry | None:
        """Remove and return the registry for *tier*.

        Used at end-of-full-cycle to retire interim slots after their keys
        have been promoted into main.  Returns ``None`` when the tier had no
        registry or replay is disabled."""
        if self._registry is None:
            return None
        return self._registry.pop(tier, None)

    def active_keys_in_tier(self, tier: str) -> list[str]:
        if self._registry is None:
            return []
        reg = self._registry.get(tier)
        return reg.list_active() if reg is not None else []

    def all_active_keys(self) -> list[str]:
        """Every active key across every registered tier."""
        if self._registry is None:
            return []
        return [k for reg in self._registry.values() for k in reg.list_active()]

    def tier_for_active_key(self, key: str) -> str | None:
        """Return the tier that holds *key* in its registry, or ``None``.

        Used by the legacy ``_tier_for_key`` consumers — kept distinct from
        :meth:`tier_of` because the registry and entry cache can briefly
        disagree during a put/move sequence."""
        if self._registry is None:
            return None
        for tier, reg in self._registry.items():
            if key in reg:
                return tier
        return None

    # ------------------------------------------------------------------
    # Snapshot / restore — cycle-resume rollback rope
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Deep-copy the entry and simhash state for rollback.

        Registries are NOT included — they have their own persistence
        lifecycle (KeyRegistry.save / KeyRegistry.load to per-tier
        ``indexed_key_registry.json`` files).  The rollback caller is
        responsible for restoring registries from disk separately."""
        return {
            "entries": copy.deepcopy(self._entries),
            "simhash": copy.deepcopy(self._simhash),
        }

    def restore(self, snap: dict) -> None:
        """Restore entry and simhash state from :meth:`snapshot` output."""
        self._entries = copy.deepcopy(snap["entries"])
        self._simhash = copy.deepcopy(snap["simhash"])

    def _entries_flat_view(self) -> "_LegacyFlatCacheView":
        """Return a flat dict-like view of all entries across tiers.

        DEPRECATED: only exists for test sites that still expect a flat
        ``dict[key, entry]`` shape.  New code uses :meth:`iter_entries`
        or :meth:`entries_in_tier`."""
        return _LegacyFlatCacheView(self)

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
    ) -> dict[str, dict | None]:
        """Resolve *keys_by_adapter* to flat ``{key → result | None}``.

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

        from paramem.memory.entry import entry_fact_text

        def _render(key: str, entry: dict) -> dict:
            # speaker_id / first_seen_cycle come from _bookkeeping — the single
            # authoritative source.  Entries hold SPO only (inference cache
            # contract); save-path working entries carry them for build_graph_for_tier
            # but those are consumed within the cycle, not at inference time.
            bk = self._bookkeeping.get(key) or {}
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
                "confidence": 1.0,
                "fact_text": entry_fact_text({**base, "speaker_id": spk}),
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
                bk_spk = self._bookkeeping.get(key, {}).get("speaker_id", "")
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
                        if key not in self._bookkeeping:
                            src_spk = src.get("speaker_id", "")
                            src_fsc = src.get("first_seen_cycle", 0)
                            if src_spk or src_fsc:
                                self.set_bookkeeping(
                                    key,
                                    speaker_id=src_spk,
                                    first_seen_cycle=src_fsc,
                                )

        return results

    # ------------------------------------------------------------------
    # On-disk registries — load registries + simhashes from the adapter dir
    # ------------------------------------------------------------------
    def load_registries_from_disk(self, adapter_dir) -> None:
        """Load per-tier ``indexed_key_registry.json`` + ``simhash_registry.json``
        into the store from *adapter_dir*.

        Reads:

        * ``<adapter_dir>/<tier>/indexed_key_registry.json`` for each main tier
          and every ``episodic_interim_<stamp>`` slot.
        * ``<adapter_dir>/<tier>/simhash_registry.json`` per tier, with a
          legacy-flat fallback at ``<adapter_dir>/simhash_registry_<tier>.json``.

        Entry payloads (subject/predicate/object/speaker_id) are NOT loaded
        here — that is the responsibility of the mode-specific
        :class:`paramem.memory.source.MemorySource` (weight probe in
        train mode; encrypted graph.json read in simulate mode).

        No-op when the store has ``replay_enabled=False`` (registries are not
        tracked).
        """
        if self._registry is None:
            return

        from pathlib import Path

        from paramem.memory.interim_adapter import iter_interim_dirs
        from paramem.memory.persistence import (
            load_registry as _load_simhash_registry,
        )
        from paramem.training.key_registry import KeyRegistry

        adapter_dir = Path(adapter_dir)

        # Main tiers — registries live at <adapter_dir>/<tier>/indexed_key_registry.json
        for tier in ("episodic", "semantic", "procedural"):
            reg_path = adapter_dir / tier / "indexed_key_registry.json"
            self.load_registry(tier, KeyRegistry.load(reg_path))

        # Interim tiers — dynamic; loaded if their dirs exist.  Tier key is
        # the PEFT adapter name so callers using ``peft_config`` keys can
        # address the store consistently.
        for interim_name, interim_dir in iter_interim_dirs(adapter_dir):
            reg_path = interim_dir / "indexed_key_registry.json"
            self.load_registry(interim_name, KeyRegistry.load(reg_path))

        # Per-tier SimHash registries; legacy flat path is a one-time
        # upgrade compat fallback.
        def _load_simhash(tier: str) -> dict:
            per_tier = adapter_dir / tier / "simhash_registry.json"
            legacy = adapter_dir / f"simhash_registry_{tier}.json"
            if per_tier.exists():
                return _load_simhash_registry(per_tier)
            if legacy.exists():
                logger.info(
                    "MemoryStore.load_registries_from_disk: upgrading simhash "
                    "registry for %s from legacy flat path",
                    tier,
                )
                return _load_simhash_registry(legacy)
            return {}

        for tier in ("episodic", "semantic", "procedural"):
            sh = _load_simhash(tier)
            if sh:
                self.replace_simhashes_in_tier(tier, sh)

    def load_bookkeeping_from_disk(self, key_metadata_path) -> dict:
        """Load per-key ``speaker_id`` / ``first_seen_cycle`` into ``_bookkeeping``.

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

        Keys absent from every tier registry (orphans — slot wiped or never
        existed) are skipped but counted in the return dict.  Pre-refactor
        files that lack ``speaker_id``/``first_seen_cycle`` are upgraded
        in-memory with ``""`` / ``0`` (legacy_upgraded counter).

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
            tier = self.tier_for_active_key(key)
            if tier is None:
                orphaned += 1
                continue
            if "speaker_id" not in key_meta or "first_seen_cycle" not in key_meta:
                legacy_upgraded += 1
            self.set_bookkeeping(
                key,
                speaker_id=key_meta.get("speaker_id", ""),
                first_seen_cycle=key_meta.get("first_seen_cycle", 0),
            )
            loaded += 1
        return {"loaded": loaded, "orphaned": orphaned, "legacy_upgraded": legacy_upgraded}

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """Per-tier ``{key_count, simhash_count, registry_active}`` map."""
        out: dict[str, dict] = {}
        for tier in self._entries:
            out.setdefault(tier, {})["key_count"] = len(self._entries[tier])
        for tier in self._simhash:
            out.setdefault(tier, {})["simhash_count"] = len(self._simhash[tier])
        if self._registry is not None:
            for tier, reg in self._registry.items():
                out.setdefault(tier, {})["registry_active"] = len(reg.list_active())
        return out


class _LegacyFlatCacheView:
    """Flat dict-like view over :class:`MemoryStore` entries.

    DEPRECATED — only exists so the ``ConsolidationLoop.indexed_key_cache``
    property keeps the legacy tests green during the migration to
    ``self.store``.  Delete once tests are migrated off the legacy
    attribute name.

    Read semantics: ``view[key]`` raises ``KeyError`` on miss, matching the
    pre-refactor dict.  ``key in view``, ``.get``, ``.items``, ``.values``,
    ``.keys``, ``len`` all work.

    Write semantics: ``view[key] = entry`` routes to the tier that already
    owns *key*; if *key* is new, defaults to ``"episodic"``.  ``del view[key]``
    drops *key* from whichever tier owns it.  ``.pop`` and ``.update`` work.

    Mutation routes through :meth:`MemoryStore.put` / :meth:`MemoryStore.delete`
    so entries + simhash + registry stay coherent.  This is the SAFE write path
    for legacy callers; new code uses the store API directly.
    """

    __slots__ = ("_store",)

    def __init__(self, store: "MemoryStore") -> None:
        self._store = store

    def __getitem__(self, key: str) -> dict:
        value = self._store.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: dict) -> None:
        # Prefer the tier the key already belongs to in ANY sub-structure
        # (entry / registry / simhash) so legacy seeders that prime one
        # tier's simhash and then write the entry observe consistent
        # ownership.  Falls back to ``"episodic"`` only when the key is
        # completely new.
        tier = (
            self._store.tier_of(key)
            or self._store.tier_for_active_key(key)
            or self._store._tier_for_simhash(key)
            or "episodic"
        )
        # ``register=False``: the legacy ``indexed_key_cache[k] = q`` write
        # only mutated the cache dict — it did NOT add to
        # ``indexed_key_registry``.  Callers that want registration use
        # the explicit store API or the registry directly.
        self._store.put(tier, key, value, register=False)

    def __delitem__(self, key: str) -> None:
        former = self._store.delete(key)
        if former is None:
            raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._store.has(key)

    def __iter__(self):
        for _tier, key, _entry in self._store.iter_entries():
            yield key

    def __len__(self) -> int:
        return len(self._store)

    def __bool__(self) -> bool:
        return len(self._store) > 0

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _LegacyFlatCacheView):
            return dict(self.items()) == dict(other.items())
        if isinstance(other, dict):
            return dict(self.items()) == other
        return NotImplemented

    def get(self, key: str, default=None):
        value = self._store.get(key)
        return default if value is None else value

    def setdefault(self, key: str, default: dict) -> dict:
        existing = self._store.get(key)
        if existing is not None:
            return existing
        self[key] = default
        return default

    def pop(self, key: str, *default):
        existing = self._store.get(key)
        if existing is None:
            if default:
                return default[0]
            raise KeyError(key)
        self._store.delete(key)
        return existing

    def items(self):
        return [(k, q) for _tier, k, q in self._store.iter_entries()]

    def keys(self):
        return [k for _tier, k, _q in self._store.iter_entries()]

    def values(self):
        return [q for _tier, _k, q in self._store.iter_entries()]

    def update(self, *args, **kwargs) -> None:
        if args:
            arg = args[0]
            if hasattr(arg, "items"):
                for k, v in arg.items():
                    self[k] = v
            else:
                for k, v in arg:
                    self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def __repr__(self) -> str:
        return f"_LegacyFlatCacheView({dict(self.items())!r})"
