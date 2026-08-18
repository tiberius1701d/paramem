"""End-to-end retirement coverage for the key-registry stale lifecycle --
drives ``ConsolidationLoop.stage_event`` through
``run_build_and_publish`` (real write/publish machinery, no GPU: the
``tests._fold_fixtures`` fakes stand in for training and the recall gate)
and asserts the LIVE store's registry/rows/entries after go-live, not just
the shadow tree ``tests/test_fold_phase1.py`` already covers.

Split out from ``tests/test_fold_phase1.py`` (which drives ``stage_event``
alone, no publish) because these tests need the real driver/publish surface
``tests._fold_fixtures._make_loop``/``_wire_fakes`` provides -- the same
split ``tests/test_stage_ledger.py`` and ``tests/test_publish_bundle_resume.py``
already make.
"""

from __future__ import annotations

from tests._fold_fixtures import (
    _make_loop,
    _recalled_entries_from_store,
    _rel,
    _seed_payload_bearing_ring,
    _wire_fakes,
)


def _seed_row(loop, tier, key, *, active: bool):
    loop.store.registry(tier).add(key)
    if not active:
        loop.store.registry(tier).stale(key)
    loop.store.set_bookkeeping(
        key,
        speaker_id="speaker0",
        relation_type="factual",
        reinforcement_count=1,
        last_reinforced_cycle=0,
        last_seen="2026-01-01T00:00:00Z",
        first_seen="2026-01-01T00:00:00Z",
        promoted=False,
    )


class TestPublishedRegistryAfterRebuild:
    def test_a_withheld_keys_record_and_row_are_gone_from_the_published_registry_after_its_rebuild(
        self, tmp_path, monkeypatch
    ):
        """A pre-existing marker on a tier this event REBUILDS never even
        enters the working universe (the seed is active-only) -- after
        go-live the live store must not know the id at all: not active, not
        withheld, no row, no entry."""
        loop = _make_loop(tmp_path, resident_tiers=["episodic"])
        _wire_fakes(loop, monkeypatch)

        _seed_row(loop, "episodic", "keep1", active=True)
        loop.store.put(
            "episodic",
            "keep1",
            {"key": "keep1", "subject": "alex", "predicate": "lives in", "object": "berlin"},
            register=False,
        )
        _seed_row(loop, "episodic", "gone1", active=False)

        staged = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="s1",
            primary_tiers={"episodic": "episodic"},
        )
        assert staged is not None

        summary = loop.run_build_and_publish(staged, router=None)
        assert summary["all_live"] is True
        assert "episodic" in summary["published_tiers"]

        live_registry = loop.store.registry("episodic")
        assert live_registry.list_active() == ["keep1"]
        assert not live_registry.knows("gone1")  # gone entirely, not just withheld
        assert loop.store.bookkeeping_for_key("gone1") is None
        assert loop.store.get("gone1") is None


class TestCandidateTierMarkerSurvivesThePublish:
    def test_a_candidate_tiers_marker_survives_the_fold_untouched_through_publish(
        self, tmp_path, monkeypatch
    ):
        """A tier this event only dedups against (a candidate/rows-only
        member) that nothing in this event actually collides with stays
        completely UNBUILT -- its live registry, including a pre-existing
        marker, is untouched by the publish."""
        loop = _make_loop(tmp_path, resident_tiers=["episodic_interim_20260101T0000"])
        _wire_fakes(loop, monkeypatch)

        _seed_row(loop, "semantic", "keepS", active=True)
        loop.store.put(
            "semantic",
            "keepS",
            {"key": "keepS", "subject": "sam", "predicate": "works at", "object": "acme"},
            register=False,
        )
        _seed_row(loop, "semantic", "goneS", active=False)

        staged = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="s2",
            primary_tiers={
                "episodic_interim_20260101T0000": "episodic_interim_20260101T0000",
            },
            candidate_tiers={"semantic": "semantic"},
            # An unrelated fact -- no collision with anything in semantic,
            # so the candidate tier's working copy is never marked dirty.
            episodic_rels=[_rel("alex", "likes", "coffee")],
            session_ids=["s2"],
        )
        assert staged is not None
        assert "semantic" not in staged.built_tiers

        summary = loop.run_build_and_publish(staged, router=None)
        assert summary["all_live"] is True
        assert "semantic" not in summary["published_tiers"]

        reg = loop.store.registry("semantic")
        assert reg.list_active() == ["keepS"]
        assert reg.list_stale() == ["goneS"]
        assert loop.store.bookkeeping_for_key("goneS") is not None


class TestInterimFoldRetiresCasualtiesAndMarksMainTiers:
    def test_an_interim_fold_retires_its_own_slots_casualties_and_marks_the_main_tiers(
        self, tmp_path, monkeypatch
    ):
        """One interim event, two dedup collisions with opposite fates: a
        duplicate wholly WITHIN the interim's own (rebuilt) primary tier is
        retired outright -- gone, not withheld, since the tier is already
        re-deriving from its post-merge active set. A duplicate between the
        interim's own content and a pre-existing main-tier (candidate/
        dedup-only) key is withheld instead -- the main tier is never
        rebuilt by this event, so its registry is the only thing standing
        between the losing id and an enumerator."""
        loop = _make_loop(tmp_path, resident_tiers=["episodic_interim_20260101T0000"])
        _wire_fakes(loop, monkeypatch)

        # Within the interim's own primary tier: two keys for the identical
        # fact -- registration order decides the survivor (primary-tier
        # relations merge first, in list_active() order), so "interim_keep"
        # (added first) survives and "interim_loser" (added second) is
        # retired -- owned by a REBUILT tier, so it is removed outright.
        _seed_row(loop, "episodic_interim_20260101T0000", "interim_keep", active=True)
        loop.store.put(
            "episodic_interim_20260101T0000",
            "interim_keep",
            {"key": "interim_keep", "subject": "alex", "predicate": "lives in", "object": "berlin"},
            register=False,
        )
        _seed_row(loop, "episodic_interim_20260101T0000", "interim_loser", active=True)
        loop.store.put(
            "episodic_interim_20260101T0000",
            "interim_loser",
            {
                "key": "interim_loser",
                "subject": "alex",
                "predicate": "lives in",
                "object": "berlin",
            },
            register=False,
        )

        # A distinct fact, present in the interim's own primary tier --
        # merges FIRST (primary tiers merge before candidate tiers), so it
        # becomes the surviving edge below.
        _seed_row(loop, "episodic_interim_20260101T0000", "interim_sam", active=True)
        loop.store.put(
            "episodic_interim_20260101T0000",
            "interim_sam",
            {"key": "interim_sam", "subject": "sam", "predicate": "works at", "object": "acme"},
            register=False,
        )

        # The SAME "sam works at acme" fact, pre-existing in the main
        # (candidate/dedup-only) semantic tier -- merges LAST, so it loses
        # to the interim tier's already-merged edge and is withheld, not
        # removed: semantic is never rebuilt by this event.
        _seed_row(loop, "semantic", "sem_loser", active=True)
        loop.store.put(
            "semantic",
            "sem_loser",
            {"key": "sem_loser", "subject": "sam", "predicate": "works at", "object": "acme"},
            register=False,
        )

        staged = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="s3",
            primary_tiers={
                "episodic_interim_20260101T0000": "episodic_interim_20260101T0000",
            },
            candidate_tiers={"semantic": "semantic"},
        )
        assert staged is not None

        summary = loop.run_build_and_publish(staged, router=None)
        assert summary["all_live"] is True

        # The interim tier's own casualty is GONE -- not withheld, not known.
        interim_registry = loop.store.registry("episodic_interim_20260101T0000")
        assert not interim_registry.knows("interim_loser")
        assert loop.store.bookkeeping_for_key("interim_loser") is None
        assert interim_registry.list_active() == ["interim_keep", "interim_sam"]

        # The main (candidate) tier's casualty is MARKED, not erased.
        semantic_registry = loop.store.registry("semantic")
        assert semantic_registry.list_stale() == ["sem_loser"]
        assert "sem_loser" not in semantic_registry.list_active()
        assert loop.store.bookkeeping_for_key("sem_loser") is not None


class TestFullFoldPublishesMarkerFreeMainTiers:
    def test_a_full_fold_publishes_every_main_tier_marker_free(self, tmp_path, monkeypatch):
        """A full fold's working-copy seed is active-only for every PRIMARY
        tier (Q2) -- a pre-existing marker on ANY main tier never even
        enters this event's working universe, so the published registry
        carries none, for all three tiers at once."""
        loop = _make_loop(
            tmp_path, procedural=True, resident_tiers=["episodic", "semantic", "procedural"]
        )
        _wire_fakes(loop, monkeypatch)

        facts = {
            "episodic": ("alex", "lives in", "berlin"),
            "semantic": ("sam", "works at", "acme"),
            "procedural": ("jane", "prefers", "tea"),
        }
        for tier, (subject, predicate, obj) in facts.items():
            _seed_row(loop, tier, f"{tier}_keep", active=True)
            loop.store.put(
                tier,
                f"{tier}_keep",
                {"key": f"{tier}_keep", "subject": subject, "predicate": predicate, "object": obj},
                register=False,
            )
            _seed_row(loop, tier, f"{tier}_marker", active=False)

        staged = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="full",
            venue="weights",
            stamp="sfull",
            primary_tiers={
                "episodic": "episodic",
                "semantic": "semantic",
                "procedural": "procedural",
            },
        )
        assert staged is not None

        summary = loop.run_build_and_publish(staged, router=None)
        assert summary["all_live"] is True

        for tier in facts:
            registry = loop.store.registry(tier)
            assert registry.list_stale() == []
            assert not registry.knows(f"{tier}_marker")
            assert loop.store.bookkeeping_for_key(f"{tier}_marker") is None


class TestFullFoldRingAbsorptionLeavesNoInterimMarker:
    def test_a_full_folds_ring_absorption_leaves_no_interim_marker_anywhere(
        self, tmp_path, monkeypatch
    ):
        """A full fold's candidate ring is absorbed WHOLE and reaped at
        go-live (never published rows-only): a marker on an absorbed
        interim slot does not survive into any published registry --
        because the slot itself, marker included, is gone."""
        loop = _make_loop(
            tmp_path,
            resident_tiers=["episodic", "episodic_interim_20260101T0000"],
        )
        _wire_fakes(loop, monkeypatch)
        interim_dir = _seed_payload_bearing_ring(loop, "episodic_interim_20260101T0000")

        loop.store.registry("episodic_interim_20260101T0000").add("interim_marker")
        loop.store.registry("episodic_interim_20260101T0000").stale("interim_marker")
        loop.store.set_bookkeeping(
            "interim_marker",
            speaker_id="speaker0",
            relation_type="factual",
            reinforcement_count=1,
            last_reinforced_cycle=0,
            last_seen="2026-01-01T00:00:00Z",
            first_seen="2026-01-01T00:00:00Z",
            promoted=False,
        )

        staged = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="full",
            venue="weights",
            stamp="sfull2",
            primary_tiers={"episodic": "episodic", "semantic": "semantic"},
            candidate_tiers={"episodic_interim_20260101T0000": "episodic_interim_20260101T0000"},
        )
        assert staged is not None

        summary = loop.run_build_and_publish(staged, router=None)
        assert summary["all_live"] is True

        # The whole interim slot is reaped -- registry (marker included)
        # gone from RAM and disk.
        assert not loop.store.has_registry("episodic_interim_20260101T0000")
        assert not interim_dir.exists()
        assert loop.store.bookkeeping_for_key("interim_marker") is None


class TestReconsolidatePublishesMarkerFreeMainTiers:
    def test_reconsolidate_publishes_every_main_tier_marker_free(self, tmp_path, monkeypatch):
        """A reconcile (``POST /reconsolidate``) is a full-topology event --
        ``full_topology("reconcile")`` is ``True`` -- so it seeds every
        primary tier's working copy active-only exactly as an ordinary full
        fold does. Driven directly against ``stage_event(event="reconcile",
        ...)``, the closest equivalent to the HTTP door: ``consolidate()``'s
        own wrapper only adds GPU-lock/hydration plumbing this suite
        deliberately does not construct (see module docstring)."""
        loop = _make_loop(
            tmp_path, procedural=True, resident_tiers=["episodic", "semantic", "procedural"]
        )
        _wire_fakes(loop, monkeypatch)

        facts = {
            "episodic": ("alex", "lives in", "berlin"),
            "semantic": ("sam", "works at", "acme"),
            "procedural": ("jane", "prefers", "tea"),
        }
        for tier, (subject, predicate, obj) in facts.items():
            _seed_row(loop, tier, f"{tier}_keep", active=True)
            loop.store.put(
                tier,
                f"{tier}_keep",
                {"key": f"{tier}_keep", "subject": subject, "predicate": predicate, "object": obj},
                register=False,
            )
            _seed_row(loop, tier, f"{tier}_marker", active=False)

        staged = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="reconcile",
            venue="weights",
            stamp="srec",
            primary_tiers={
                "episodic": "episodic",
                "semantic": "semantic",
                "procedural": "procedural",
            },
        )
        assert staged is not None

        summary = loop.run_build_and_publish(staged, router=None)
        assert summary["all_live"] is True

        for tier in facts:
            registry = loop.store.registry(tier)
            assert registry.list_stale() == []
            assert not registry.knows(f"{tier}_marker")


class TestTierRebuiltToZeroKeysPublishesEmptyRegistry:
    def test_a_tier_rebuilt_to_zero_keys_publishes_an_empty_registry_against_its_existing_slot(
        self, tmp_path, monkeypatch
    ):
        """The stated asymmetry: markers are retired at a rebuild with no
        payload written for them, but an already-BOUND tier whose active set
        collapses to zero on its next rebuild still publishes cleanly --
        key_count=0, bound to a real (fresh) slot, not left orphaned."""
        loop = _make_loop(tmp_path, resident_tiers=["episodic", "semantic"])
        _wire_fakes(loop, monkeypatch)

        _seed_row(loop, "semantic", "sem_only", active=True)
        loop.store.put(
            "semantic",
            "sem_only",
            {"key": "sem_only", "subject": "sam", "predicate": "works at", "object": "acme"},
            register=False,
        )

        staged1 = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="full",
            venue="weights",
            stamp="s1",
            primary_tiers={"episodic": "episodic", "semantic": "semantic"},
            episodic_rels=[_rel("alex", "lives_in", "berlin")],
        )
        assert staged1 is not None
        summary1 = loop.run_build_and_publish(staged1, router=None)
        assert summary1["all_live"] is True
        assert loop.store.registry("semantic").list_known() == ["sem_only"]

        # sem_only is withheld before the tier's SECOND rebuild -- its
        # active set is now empty.
        loop.store.registry("semantic").stale("sem_only")

        staged2 = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="full",
            venue="weights",
            stamp="s2",
            primary_tiers={"episodic": "episodic", "semantic": "semantic"},
            episodic_rels=[_rel("sam", "visited", "lisbon")],
        )
        assert staged2 is not None
        assert "semantic" in staged2.built_tiers
        summary2 = loop.run_build_and_publish(staged2, router=None)
        assert summary2["all_live"] is True

        semantic_registry = loop.store.registry("semantic")
        assert semantic_registry.list_known() == []

        from paramem.adapters.manifest import find_live_slot, tier_registry_sha256
        from paramem.training.key_registry import KeyRegistry

        tier_root = loop.output_dir / "semantic"
        slot = find_live_slot(tier_root, tier_registry_sha256(tier_root))
        assert slot is not None  # bound, not orphaned
        on_disk = KeyRegistry.load(tier_root / "indexed_key_registry.json")
        assert on_disk.list_known() == []


class TestRetiredIdNumberReservedUntilRebuild:
    def test_a_retired_ids_number_is_reserved_until_its_tier_rebuilds(self, tmp_path, monkeypatch):
        """``_derive_key_counters`` scans ``all_known_keys()`` -- active AND
        withheld -- so a withheld id's numeric suffix keeps the mint counter
        raised until the owning tier's own rebuild retires it and frees the
        number back up."""
        loop = _make_loop(tmp_path, resident_tiers=["episodic"])
        _wire_fakes(loop, monkeypatch)

        _seed_row(loop, "episodic", "graph1", active=True)
        loop.store.put(
            "episodic",
            "graph1",
            {"key": "graph1", "subject": "alex", "predicate": "lives in", "object": "berlin"},
            register=False,
        )
        # Above DONOR_KEY_FLOOR (201) so the marker's own suffix is what
        # raises the counter, not the floor.
        _seed_row(loop, "episodic", "graph250", active=False)

        loop._derive_key_counters()
        assert loop._indexed_next_index == 251

        # graph1 (active, existing content) is enough to drive the rebuild --
        # no new material needed, and none is given, so the counter's later
        # drop is due to graph250's retirement, not a fresh mint consuming
        # the reserved number.
        staged = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="full",
            venue="weights",
            stamp="s1",
            primary_tiers={"episodic": "episodic"},
        )
        assert staged is not None
        summary = loop.run_build_and_publish(staged, router=None)
        assert summary["all_live"] is True

        assert not loop.store.registry("episodic").knows("graph250")

        loop._derive_key_counters()
        assert loop._indexed_next_index < 251


class TestTotallyErasedStoreKeepsMarkersRowsReservedIds:
    def test_a_totally_erased_store_keeps_its_markers_rows_and_reserved_ids(
        self, tmp_path, monkeypatch
    ):
        """Zero active keys anywhere -- both ``has_new_material`` and
        ``has_existing_content`` are false -- so the fold noops before
        touching anything: registry, rows and derived counters are
        unchanged, and every id stays reserved."""
        loop = _make_loop(
            tmp_path, procedural=True, resident_tiers=["episodic", "semantic", "procedural"]
        )
        _wire_fakes(loop, monkeypatch)

        for tier in ("episodic", "semantic", "procedural"):
            _seed_row(loop, tier, f"{tier}_marker1", active=False)
            _seed_row(loop, tier, f"{tier}_marker2", active=False)

        loop._derive_key_counters()
        before_indexed = loop._indexed_next_index
        before_known = {
            tier: loop.store.registry(tier).list_known()
            for tier in ("episodic", "semantic", "procedural")
        }
        before_bookkeeping = {
            key: loop.store.bookkeeping_for_key(key)
            for tier_keys in before_known.values()
            for key in tier_keys
        }

        staged = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="reconcile",
            venue="weights",
            stamp="s_erased",
            primary_tiers={
                "episodic": "episodic",
                "semantic": "semantic",
                "procedural": "procedural",
            },
        )
        assert staged is None  # noop -- nothing new, nothing existing active

        for tier in ("episodic", "semantic", "procedural"):
            assert loop.store.registry(tier).list_known() == before_known[tier]
        for key, row in before_bookkeeping.items():
            assert loop.store.bookkeeping_for_key(key) == row

        loop._derive_key_counters()
        assert loop._indexed_next_index == before_indexed
