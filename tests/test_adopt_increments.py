"""Unit tests for MemoryStore.adopt_increments — per-tier convergence.

Builds TierIncrement objects by hand (no shadow-dir round trip needed here;
test_increment.py already covers that assembly) and exercises the
drop-then-install ordering, the entry-cache completeness postcondition,
scope (an interim tier is left alone unless explicitly named via
``absorbed_tiers``), and the ``absorbed_tiers`` ring-reap (registry, entries
AND bookkeeping dropped, inside the same lock as the bundle's own install).
"""

from __future__ import annotations

import threading

from paramem.memory.increment import TierIncrement
from paramem.memory.store import MemoryStore
from paramem.training.key_registry import KeyRegistry


def _row(**overrides):
    base = {
        "speaker_id": "speaker0",
        "relation_type": "factual",
        "reinforcement_count": 1,
        "last_reinforced_cycle": 1,
        "last_seen": "",
        "first_seen": "",
        "promoted": False,
    }
    base.update(overrides)
    return base


def _increment(
    tier,
    *,
    active_keys=(),
    entries=None,
    bookkeeping=None,
    rebuilt=True,
    pre_sha="",
) -> TierIncrement:
    reg = KeyRegistry()
    for k in active_keys:
        reg.add(k)
    return TierIncrement(
        tier=tier,
        adapter_name=tier,
        registry=reg,
        registry_bytes=reg.save_bytes(),
        rows_bytes=b"{}",
        entries=dict(entries or {}),
        bookkeeping=dict(bookkeeping or {}),
        keyed=[{"key": k} for k in active_keys] if rebuilt else [],
        rebuilt=rebuilt,
        pre_sha=pre_sha,
    )


def _content_entry(key, subject="s", predicate="p", object_="o"):
    return {"key": key, "subject": subject, "predicate": predicate, "object": object_}


class TestFreshInstall:
    def test_rebuilt_member_installs_registry_entries_and_rows(self):
        store = MemoryStore()
        inc = _increment(
            "episodic",
            active_keys=["graph1"],
            entries={"graph1": _content_entry("graph1")},
            bookkeeping={"graph1": _row()},
        )
        store.adopt_increments([inc])

        assert store.active_keys_in_tier("episodic") == ["graph1"]
        assert store.get("graph1") == _content_entry("graph1")
        assert store.bookkeeping_for_key("graph1") == _row()

    def test_rows_only_member_leaves_entries_untouched(self):
        store = MemoryStore()
        # Seed a prior rebuilt state.
        seed_reg = KeyRegistry()
        seed_reg.add("graph1")
        store.load_registry("episodic", seed_reg)
        store.put("episodic", "graph1", _content_entry("graph1"), register=False)
        store.set_bookkeeping(
            "graph1", speaker_id="speaker0", relation_type="factual", first_seen="", promoted=False
        )

        # A rows-only increment: same active set, changed row, no entries.
        inc = _increment(
            "episodic",
            active_keys=["graph1"],
            entries={},
            bookkeeping={"graph1": _row(reinforcement_count=5)},
            rebuilt=False,
        )
        store.adopt_increments([inc])

        assert store.get("graph1") == _content_entry("graph1")  # untouched
        assert store.bookkeeping_for_key("graph1")["reinforcement_count"] == 5


class TestRetiredKeyVanishesByAbsence:
    def test_key_absent_from_increment_disappears_after_convergence(self):
        store = MemoryStore()
        seed_reg = KeyRegistry()
        seed_reg.add("graph1")
        seed_reg.add("graph2")
        store.load_registry("episodic", seed_reg)
        store.put("episodic", "graph1", _content_entry("graph1"), register=False)
        store.put("episodic", "graph2", _content_entry("graph2"), register=False)
        store.set_bookkeeping(
            "graph1", speaker_id="speaker0", relation_type="factual", first_seen="", promoted=False
        )
        store.set_bookkeeping(
            "graph2", speaker_id="speaker0", relation_type="factual", first_seen="", promoted=False
        )

        # graph2 was retired: absent from the new increment entirely.
        inc = _increment(
            "episodic",
            active_keys=["graph1"],
            entries={"graph1": _content_entry("graph1")},
            bookkeeping={"graph1": _row()},
        )
        store.adopt_increments([inc])

        assert store.active_keys_in_tier("episodic") == ["graph1"]
        assert store.get("graph2") is None
        assert store.bookkeeping_for_key("graph2") is None


class TestPromotionOrdering:
    def test_promoted_key_dropped_then_reinstalled_never_the_reverse(self):
        """A key moving from episodic to semantic in the same bundle must not
        be dropped by episodic's own drop pass AFTER semantic's install —
        this is exactly the hazard the "all drops before any install" rule
        exists to prevent."""
        store = MemoryStore()
        seed_reg = KeyRegistry()
        seed_reg.add("graph1")
        store.load_registry("episodic", seed_reg)
        store.set_bookkeeping(
            "graph1", speaker_id="speaker0", relation_type="factual", first_seen="", promoted=False
        )

        # episodic's new increment no longer carries graph1 (it moved out).
        episodic_inc = _increment("episodic", active_keys=[], entries={}, bookkeeping={})
        # semantic's new increment carries graph1, with its promoted row.
        semantic_inc = _increment(
            "semantic",
            active_keys=["graph1"],
            entries={"graph1": _content_entry("graph1")},
            bookkeeping={"graph1": _row(promoted=True, reinforcement_count=7)},
        )

        # Bundle order deliberately destination-first per the publish
        # convention — adopt_increments must be order-independent regardless.
        store.adopt_increments([semantic_inc, episodic_inc])

        assert store.bookkeeping_for_key("graph1") == _row(promoted=True, reinforcement_count=7)
        assert store.active_keys_in_tier("episodic") == []
        assert store.active_keys_in_tier("semantic") == ["graph1"]

    def test_order_independence_source_first(self):
        """The same scenario with the bundle passed source-first (episodic
        before semantic) must produce the identical result."""
        store = MemoryStore()
        seed_reg = KeyRegistry()
        seed_reg.add("graph1")
        store.load_registry("episodic", seed_reg)
        store.set_bookkeeping(
            "graph1", speaker_id="speaker0", relation_type="factual", first_seen="", promoted=False
        )

        episodic_inc = _increment("episodic", active_keys=[], entries={}, bookkeeping={})
        semantic_inc = _increment(
            "semantic",
            active_keys=["graph1"],
            entries={"graph1": _content_entry("graph1")},
            bookkeeping={"graph1": _row(promoted=True, reinforcement_count=7)},
        )

        store.adopt_increments([episodic_inc, semantic_inc])

        assert store.bookkeeping_for_key("graph1") == _row(promoted=True, reinforcement_count=7)


class TestEntryCacheCompletenessPostcondition:
    """Pass 0 also confirms entry-cache completeness (a SEPARATE invariant
    from bookkeeping completeness below) via
    :class:`~paramem.memory.store.EntryCacheInvariantViolation`, raised
    through :func:`~paramem.memory.store.raise_entry_cache_invariant_violation`
    -- the named-exception family replacing the former bare
    ``AssertionError``."""

    def test_raises_when_rebuilt_member_missing_an_active_entry(self):
        from paramem.memory.store import EntryCacheInvariantViolation

        store = MemoryStore()
        # active_keys names graph1, but entries carries nothing for it —
        # a builder bug this postcondition must catch.
        inc = _increment("episodic", active_keys=["graph1"], entries={}, bookkeeping={})
        try:
            store.adopt_increments([inc])
        except EntryCacheInvariantViolation as exc:
            assert "graph1" in str(exc)
            assert exc.tier == "episodic"
            assert exc.missing_keys == ["graph1"]
        else:
            raise AssertionError("expected adopt_increments to raise on incomplete entries")

    def test_rebuilt_member_violation_is_caught_before_any_mutation(self):
        """The check runs BEFORE the drop/install passes — a violating
        increment must never reach the store at all, and whatever the store
        held before the call must survive it exactly."""
        from paramem.memory.store import EntryCacheInvariantViolation

        store = MemoryStore()
        seed_reg = KeyRegistry()
        seed_reg.add("prior_key")
        store.load_registry("episodic", seed_reg)
        store.put("episodic", "prior_key", _content_entry("prior_key"), register=False)
        store.set_bookkeeping("prior_key", **_row())

        bad_inc = _increment("episodic", active_keys=["graph1"], entries={}, bookkeeping={})
        try:
            store.adopt_increments([bad_inc])
        except EntryCacheInvariantViolation:
            pass
        else:
            raise AssertionError("expected adopt_increments to raise")

        # Nothing mutated: the prior registry/entries/bookkeeping survive
        # byte-for-byte, and the registry object itself was never rebound.
        assert store.registry("episodic") is seed_reg
        assert store.active_keys_in_tier("episodic") == ["prior_key"]
        assert store.get("prior_key") == _content_entry("prior_key")
        assert store.bookkeeping_for_key("prior_key") is not None

    def test_rows_only_member_with_no_bookkeeping_raises_the_bookkeeping_violation(self):
        """A rows-only member (``rebuilt=False``) is never checked against
        the tier's entry cache -- the entry-cache check is increment-internal
        only (``if inc.rebuilt`` in :meth:`MemoryStore.adopt_increments`), and
        a rows-only increment carries no entries of its own by design. The
        bookkeeping-completeness check still runs unconditionally, so a
        rows-only member whose active key has no row in
        ``increment.bookkeeping`` raises ``BookkeepingInvariantViolation`` --
        never ``EntryCacheInvariantViolation`` -- caught before any
        mutation."""
        from paramem.memory.store import BookkeepingInvariantViolation

        store = MemoryStore()
        seed_reg = KeyRegistry()
        store.load_registry("episodic", seed_reg)
        # The tier's existing entry cache is empty -- irrelevant to a
        # rows-only member, which is never checked against it.

        bad_inc = _increment(
            "episodic", active_keys=["graph1"], entries={}, bookkeeping={}, rebuilt=False
        )
        try:
            store.adopt_increments([bad_inc])
        except BookkeepingInvariantViolation as exc:
            assert "graph1" in str(exc)
        else:
            raise AssertionError("expected adopt_increments to raise on missing bookkeeping")

        # Store untouched: the registry object is still the original one,
        # not the (incomplete) increment's.
        assert store.registry("episodic") is seed_reg
        assert store.active_keys_in_tier("episodic") == []

    def test_rows_only_member_with_bookkeeping_converges_regardless_of_the_entry_cache(self):
        """The mirror case, proving the entry-cache check never runs for a
        rows-only member: the tier's existing entry cache carries nothing for
        ``"graph1"`` (no prior ``store.put``), yet adoption converges without
        raising because bookkeeping completeness -- the only check a
        rows-only member is subject to -- is satisfied."""
        store = MemoryStore()
        seed_reg = KeyRegistry()
        seed_reg.add("graph1")
        store.load_registry("episodic", seed_reg)

        inc = _increment(
            "episodic",
            active_keys=["graph1"],
            entries={},
            bookkeeping={"graph1": _row(reinforcement_count=9)},
            rebuilt=False,
        )
        store.adopt_increments([inc])  # must not raise

        assert store.get("graph1") is None  # rows-only: entries untouched
        assert store.bookkeeping_for_key("graph1")["reinforcement_count"] == 9


class TestBookkeepingCompletenessPostcondition:
    """Pass 0 also confirms bookkeeping completeness: every key the
    increment's OWN ``registry.list_known()`` reports (active ∪ stale,
    rows-only members included) must appear in ``increment.bookkeeping``."""

    def test_raises_when_a_known_key_has_no_bookkeeping_row(self):
        from paramem.memory.store import BookkeepingInvariantViolation

        store = MemoryStore()
        inc = _increment(
            "episodic",
            active_keys=["graph1"],
            entries={"graph1": _content_entry("graph1")},
            bookkeeping={},  # entries complete, but no bookkeeping row
        )
        try:
            store.adopt_increments([inc])
        except BookkeepingInvariantViolation as exc:
            assert "graph1" in str(exc)
        else:
            raise AssertionError("expected adopt_increments to raise on missing bookkeeping")

        # Nothing mutated -- the check runs before any install.
        assert store.active_keys_in_tier("episodic") == []

    def test_rows_only_member_with_a_missing_row_also_raises(self):
        """A rows-only member (``rebuilt=False``) is checked too -- closing
        the hole a ``keyed``-only walk would have left open."""
        from paramem.memory.store import BookkeepingInvariantViolation

        store = MemoryStore()
        seed_reg = KeyRegistry()
        seed_reg.add("graph1")
        store.load_registry("episodic", seed_reg)
        store.put("episodic", "graph1", _content_entry("graph1"), register=False)

        bad_inc = _increment(
            "episodic", active_keys=["graph1"], entries={}, bookkeeping={}, rebuilt=False
        )
        try:
            store.adopt_increments([bad_inc])
        except BookkeepingInvariantViolation as exc:
            assert "graph1" in str(exc)
        else:
            raise AssertionError("expected adopt_increments to raise on missing bookkeeping")

        assert store.registry("episodic") is seed_reg


class TestScopeIsExactlyTheNamedTiers:
    def test_untouched_tier_is_left_alone(self):
        store = MemoryStore()
        seed_reg = KeyRegistry()
        seed_reg.add("proc1")
        store.load_registry("procedural", seed_reg)
        store.put("procedural", "proc1", _content_entry("proc1"), register=False)

        inc = _increment(
            "episodic",
            active_keys=["graph1"],
            entries={"graph1": _content_entry("graph1")},
            bookkeeping={"graph1": _row()},
        )
        store.adopt_increments([inc])

        # procedural, not named by the bundle, is untouched.
        assert store.active_keys_in_tier("procedural") == ["proc1"]
        assert store.get("proc1") == _content_entry("proc1")

    def test_does_not_touch_interim_tiers(self):
        store = MemoryStore()
        interim_reg = KeyRegistry()
        interim_reg.add("i1")
        store.load_registry("episodic_interim_20260101T0000", interim_reg)

        inc = _increment(
            "episodic",
            active_keys=["graph1"],
            entries={"graph1": _content_entry("graph1")},
            bookkeeping={"graph1": _row()},
        )
        store.adopt_increments([inc])

        assert store.has_registry("episodic_interim_20260101T0000")
        assert store.active_keys_in_tier("episodic_interim_20260101T0000") == ["i1"]


class TestAbsorbedTiersRingReap:
    """``absorbed_tiers`` -- a full fold's ring reap, converged inside the
    SAME lock as the bundle's own install (no reader-visible window where a
    key is active in both the bundle's destination tier and the
    not-yet-reaped interim tier it moved out of)."""

    def test_absorbed_tier_registry_and_entries_are_dropped(self):
        store = MemoryStore()
        interim_reg = KeyRegistry()
        interim_reg.add("i1")
        store.load_registry("episodic_interim_20260101T0000", interim_reg)
        store.put("episodic_interim_20260101T0000", "i1", _content_entry("i1"), register=False)

        inc = _increment(
            "episodic",
            active_keys=["graph1"],
            entries={"graph1": _content_entry("graph1")},
            bookkeeping={"graph1": _row()},
        )
        store.adopt_increments([inc], absorbed_tiers=["episodic_interim_20260101T0000"])

        assert not store.has_registry("episodic_interim_20260101T0000")
        assert store.get("i1") is None

    def test_absorbed_tier_key_never_adopted_by_any_bundle_member_drops_its_bookkeeping_too(
        self,
    ):
        """A key that lived ONLY in an absorbed interim slot -- never routed
        into any primary tier's increment -- is a genuine reap casualty.
        Its bookkeeping row must be dropped too, or it orphans in RAM
        forever (no tier ever names it again, nothing else ever revisits
        it)."""
        store = MemoryStore()
        interim_reg = KeyRegistry()
        interim_reg.add("i1")
        store.load_registry("episodic_interim_20260101T0000", interim_reg)
        store.set_bookkeeping("i1", **_row())
        assert store.bookkeeping_for_key("i1") is not None

        inc = _increment(
            "episodic",
            active_keys=["graph1"],
            entries={"graph1": _content_entry("graph1")},
            bookkeeping={"graph1": _row()},
        )
        store.adopt_increments([inc], absorbed_tiers=["episodic_interim_20260101T0000"])

        assert store.bookkeeping_for_key("i1") is None, (
            "an absorbed-only key's bookkeeping row must be dropped, not "
            "orphaned with no registry membership anywhere"
        )

    def test_absorbed_tier_key_adopted_into_the_bundle_keeps_its_bookkeeping(self):
        """A key the staging pass DID route into a primary tier's increment
        (still named by the bundle) must survive -- the absorbed tier's own
        drop must never win a race against the bundle's own install."""
        store = MemoryStore()
        interim_reg = KeyRegistry()
        interim_reg.add("adopted_key")
        store.load_registry("episodic_interim_20260101T0000", interim_reg)
        store.set_bookkeeping("adopted_key", **_row())

        inc = _increment(
            "episodic",
            active_keys=["adopted_key"],
            entries={"adopted_key": _content_entry("adopted_key")},
            bookkeeping={"adopted_key": _row(reinforcement_count=2)},
        )
        store.adopt_increments([inc], absorbed_tiers=["episodic_interim_20260101T0000"])

        assert "adopted_key" in store.active_keys_in_tier("episodic")
        row = store.bookkeeping_for_key("adopted_key")
        assert row is not None
        assert row["reinforcement_count"] == 2

    def test_no_reap_when_absorbed_tiers_empty(self):
        """Default (empty absorbed_tiers) -- an interim event's own
        go-live, or a full event with no ring to absorb -- must not touch
        any other tier's registry, entries, or bookkeeping."""
        store = MemoryStore()
        interim_reg = KeyRegistry()
        interim_reg.add("i1")
        store.load_registry("episodic_interim_20260101T0000", interim_reg)
        store.set_bookkeeping("i1", **_row())

        inc = _increment(
            "episodic",
            active_keys=["graph1"],
            entries={"graph1": _content_entry("graph1")},
            bookkeeping={"graph1": _row()},
        )
        store.adopt_increments([inc])

        assert store.has_registry("episodic_interim_20260101T0000")
        assert store.bookkeeping_for_key("i1") is not None


def _run_concurrent(writer_fn, reader_fn, *, join_timeout=10.0):
    """Run *writer_fn* and *reader_fn* on their own threads, released
    together via a start barrier, and return ``(failures, writer_thread,
    reader_thread)``.

    Neither thread raises across the thread boundary -- a failure from
    either side is appended to the shared *failures* list instead, under a
    lock. The caller joins both threads with *join_timeout* (already done
    here) and must still assert on ``is_alive()`` -- a timed-out join is
    itself a failure, distinct from an assertion recorded during the run.
    """
    failures: list[str] = []
    failures_lock = threading.Lock()
    start = threading.Event()
    stop = threading.Event()

    def record(msg: str) -> None:
        with failures_lock:
            failures.append(msg)

    def _writer() -> None:
        start.wait()
        try:
            writer_fn()
        except Exception as exc:  # noqa: BLE001 -- surfaced via the shared list, never across the thread boundary
            record(f"writer raised: {exc!r}")
        finally:
            stop.set()

    def _reader() -> None:
        start.wait()
        try:
            reader_fn(stop, record)
        except Exception as exc:  # noqa: BLE001 -- surfaced via the shared list, never across the thread boundary
            record(f"reader raised: {exc!r}")

    writer_thread = threading.Thread(target=_writer)
    reader_thread = threading.Thread(target=_reader)
    writer_thread.start()
    reader_thread.start()
    start.set()
    writer_thread.join(timeout=join_timeout)
    stop.set()  # in case the writer itself raised before reaching its own stop.set()
    reader_thread.join(timeout=join_timeout)

    return failures, writer_thread, reader_thread


class TestConcurrentReaders:
    """``adopt_increments``'s atomicity (one lock acquisition spanning
    check/drop/install, store.py:1662-1698) is currently guaranteed by
    construction only -- no test observes it under concurrency. These tests
    spin a reader against a writer alternating (or, for the ring reap,
    single-shot) ``adopt_increments`` calls and assert Rule 7
    (store.py:151-157: no reader ever observes a fact live in both a
    bundle's destination tier and the not-yet-reaped interim tier) holds on
    every reader pass, not just at rest.
    """

    _ITERATIONS = 300
    _ROUNDS = 50
    _JOIN_TIMEOUT = 10.0

    def test_promoted_key_never_observed_active_in_both_or_neither_tier(self):
        """A key flip-flopping between episodic and semantic (the exact
        promotion shape from :class:`TestPromotionOrdering`) under a writer
        alternating a bundle and its inverse: a spinning reader must never
        catch the key active in both tiers, active in neither, missing its
        bookkeeping row, or missing its entry.

        The union step reads ``store.iter_entries()`` -- ONE lock
        acquisition covering every tier's entry cache (store.py:457-470) --
        rather than composing two independently-locked
        ``active_keys_in_tier`` calls (one per tier, the shape ``GET
        /status`` uses per-tier at app.py:5368 for a display-only count with
        no cross-tier invariant). Two separate locked calls straddle the
        lock release between them, so a writer alternating fast enough can
        complete a full transition in the gap -- this reader hit exactly
        that false "neither" between the two calls during development, a
        TOCTOU artifact of composing un-atomic reads, not evidence against
        Rule 7 (store.py:151-157), which only ever promised atomicity
        WITHIN one ``adopt_increments`` call. ``iter_entries()`` is the
        accessor that actually delivers a consistent cross-tier view (the
        same one ``GET /debug/dump`` composes against per-key bookkeeping,
        app.py:10156-10166) and is the correct tool to hold Rule 7 to
        account under concurrent stress."""
        store = MemoryStore()
        key = "graph1"

        # Forward: key lives in semantic (promoted); episodic carries
        # nothing -- the exact promotion shape TestPromotionOrdering builds.
        episodic_empty = _increment("episodic", active_keys=[], entries={}, bookkeeping={})
        semantic_with_key = _increment(
            "semantic",
            active_keys=[key],
            entries={key: _content_entry(key)},
            bookkeeping={key: _row(promoted=True, reinforcement_count=7)},
        )
        bundle_forward = [episodic_empty, semantic_with_key]

        # Back: the exact inverse -- key lives in episodic; semantic carries
        # nothing.
        episodic_with_key = _increment(
            "episodic",
            active_keys=[key],
            entries={key: _content_entry(key)},
            bookkeeping={key: _row(promoted=False, reinforcement_count=7)},
        )
        semantic_empty = _increment("semantic", active_keys=[], entries={}, bookkeeping={})
        bundle_back = [episodic_with_key, semantic_empty]

        # Seed a live state before the reader starts spinning so its very
        # first pass has something to observe.
        store.adopt_increments(bundle_forward)

        def writer() -> None:
            for i in range(self._ITERATIONS):
                store.adopt_increments(bundle_back if i % 2 == 0 else bundle_forward)

        def reader(stop: threading.Event, record) -> None:
            while not stop.is_set():
                # One locked snapshot across BOTH tiers -- the union step --
                # then per-key bookkeeping/get, exactly the production
                # /debug/dump composition.
                snapshot = list(store.iter_entries())
                owning_tiers = {tier for tier, k, _ in snapshot if k == key}
                if len(owning_tiers) > 1:
                    record(f"{key!r} observed active in more than one tier: {owning_tiers}")
                    continue
                if not owning_tiers:
                    record(f"{key!r} not active in either episodic or semantic on this pass")
                    continue
                owning_tier = next(iter(owning_tiers))
                row = store.bookkeeping_for_key(key)
                if row is None:
                    record(
                        f"bookkeeping_for_key({key!r}) returned None while active in {owning_tier}"
                    )
                    continue
                entry = store.get(key)
                if entry is None:
                    record(f"get({key!r}) returned None while active in {owning_tier}")

        failures, writer_thread, reader_thread = _run_concurrent(
            writer, reader, join_timeout=self._JOIN_TIMEOUT
        )

        assert not writer_thread.is_alive(), "writer thread did not finish within the join timeout"
        assert not reader_thread.is_alive(), "reader thread did not finish within the join timeout"
        assert failures == [], f"concurrent reader observed a torn state: {failures}"

    def test_absorbed_ring_reap_never_leaves_key_visible_in_both_tiers(self):
        """Rule 7 cannot flip-flop -- a reaped interim tier does not come
        back -- so instead of alternating writes against one store, this
        rebuilds a fresh store per round (copying
        :class:`TestAbsorbedTiersRingReap`'s "adopted_key" seeding: the key
        already known to the interim tier's own registry AND named by the
        bundle's destination increment) and fires exactly one
        ``adopt_increments(..., absorbed_tiers=[interim])`` per round while a
        reader spins. The reader must never catch the key active in both the
        destination tier and the not-yet-reaped interim tier, nor find its
        bookkeeping row missing."""
        key = "adopted_key"
        interim_tier = "episodic_interim_20260101T0000"

        for round_index in range(self._ROUNDS):
            store = MemoryStore()
            interim_reg = KeyRegistry()
            interim_reg.add(key)
            store.load_registry(interim_tier, interim_reg)
            store.put(interim_tier, key, _content_entry(key), register=False)
            store.set_bookkeeping(key, **_row())

            inc = _increment(
                "episodic",
                active_keys=[key],
                entries={key: _content_entry(key)},
                bookkeeping={key: _row(reinforcement_count=2)},
            )

            def writer(store=store, inc=inc) -> None:
                store.adopt_increments([inc], absorbed_tiers=[interim_tier])

            def reader(stop: threading.Event, record, store=store) -> None:
                while not stop.is_set():
                    in_destination = key in store.active_keys_in_tier("episodic")
                    in_interim = key in store.active_keys_in_tier(interim_tier)
                    if in_destination and in_interim:
                        record(
                            f"round {round_index}: {key!r} observed active in both "
                            f"episodic and {interim_tier}"
                        )
                    row = store.bookkeeping_for_key(key)
                    if row is None:
                        record(f"round {round_index}: bookkeeping_for_key({key!r}) returned None")

            failures, writer_thread, reader_thread = _run_concurrent(
                writer, reader, join_timeout=self._JOIN_TIMEOUT
            )

            assert not writer_thread.is_alive(), (
                f"round {round_index}: writer thread did not finish within the join timeout"
            )
            assert not reader_thread.is_alive(), (
                f"round {round_index}: reader thread did not finish within the join timeout"
            )
            assert failures == [], (
                f"round {round_index}: concurrent reader observed a torn state: {failures}"
            )
