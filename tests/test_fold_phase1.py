"""Tests for event staging (extract -> merge -> enrich -> assign -> assert).

Exercises the staging surface in ``paramem.training.consolidation`` —
``ConsolidationLoop.stage_event`` and its helpers, plus the module-level
``classify_partial_build`` — the first phase of every consolidation event
(``run_consolidation_cycle`` and ``consolidate`` both call it before
``run_build_and_publish``).  No GPU: every loop is built with ``model=None``
(or a mock only where a real Relation/graph object is required), and every
test passes ``normalize=False, enrich=False, resolve_contradictions=False``
(or exercises them via ``model=None``-safe call paths only).
"""

import json
import logging

import pytest

from paramem.graph.merger import GraphMerger
from paramem.graph.schema import Relation
from paramem.memory.store import MemoryStore
from paramem.training.consolidation import (
    ConsolidationLoop,
    WorkingTier,
    classify_partial_build,
)
from paramem.training.key_registry import KeyRegistry
from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig
from tests._fold_fixtures import _recalled_entries_from_store


def _make_loop(tmp_path, *, procedural: bool = False) -> ConsolidationLoop:
    """Minimal ConsolidationLoop for event-staging tests — no GPU, no I/O
    beyond *tmp_path*.  Mirrors ``TestFullConsolidationRecency._make_loop_for_recon_merge``
    in ``tests/test_consolidation.py`` (the established pattern for a
    lightweight, ``object.__new__``-constructed loop in this test suite).
    """
    loop = object.__new__(ConsolidationLoop)
    loop.model = None
    loop.tokenizer = None
    loop.config = ConsolidationConfig(promotion_threshold=3, decay_window=10)
    loop.training_config = TrainingConfig(
        num_epochs=1,
        gradient_checkpointing=False,
        batch_size=1,
        recall_early_stopping=False,
        recall_probe_batch_size=1,
    )
    loop.tier_adapters = {
        "episodic": AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"]),
        "semantic": AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"]),
    }
    if procedural:
        loop.tier_adapters["procedural"] = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
    loop.wandb_config = None
    loop._thermal_policy = None
    loop.output_dir = tmp_path / "adapters"
    loop.output_dir.mkdir(parents=True, exist_ok=True)
    loop.save_cycle_snapshots = False
    loop._debug_base = None
    loop.snapshot_dir = None
    loop.shutdown_requested = False
    loop._bg_trainer = None
    loop._early_stop_callback = None
    loop.fingerprint_cache = None
    loop._keep_prior_slots = 2
    loop.cycle_count = 0
    loop._indexed_next_index = 1
    loop._procedural_next_index = 1
    loop.promoted_keys = set()
    loop._pending_promoted_keys = set()
    loop.graph_enrichment_neighborhood_hops = 2
    loop.graph_enrichment_max_entities_per_pass = 50
    loop.cloud_enabled = False
    loop._incidents_state_dir = None

    loop.merger = GraphMerger(model=None, tokenizer=None)

    store = MemoryStore()
    for tier in ("episodic", "semantic", "procedural"):
        store.load_registry(tier, KeyRegistry())
    loop.store = store
    return loop


def _rel(subject: str, predicate: str, obj: str, **kw) -> Relation:
    kw.setdefault("relation_type", "factual")
    kw.setdefault("confidence", 1.0)
    kw.setdefault("speaker_id", "speaker0")
    return Relation(subject=subject, predicate=predicate, object=obj, **kw)


def _shadow_dir(loop: ConsolidationLoop, event: str, tier: str):
    return loop._fold_state_dir / "extraction" / event / "shadow" / tier


class TestRecallWorkingTiers:
    def test_recall_clones_registry_rows_and_entries(self, tmp_path):
        loop = _make_loop(tmp_path)
        loop.store.registry("episodic").add("graph1")
        loop.store.set_bookkeeping(
            "graph1",
            speaker_id="speaker0",
            relation_type="factual",
            reinforcement_count=2,
            last_reinforced_cycle=0,
            last_seen="2026-01-01T00:00:00Z",
            first_seen="2026-01-01T00:00:00Z",
            promoted=False,
        )
        loop.store.put(
            "episodic",
            "graph1",
            {"key": "graph1", "subject": "alex", "predicate": "lives in", "object": "berlin"},
            register=False,
        )

        working = loop._recall_working_tiers(
            {"episodic": "episodic"}, {}, _recalled_entries_from_store(loop)
        )
        wt = working["episodic"]
        assert wt.rows["graph1"]["reinforcement_count"] == 2
        assert wt.entries["graph1"]["object"] == "berlin"

        # Mutating the working copy must not touch the live store.
        wt.registry.remove("graph1")
        assert "graph1" in loop.store.registry("episodic")

    def test_pre_sha_is_empty_for_a_fresh_tier(self, tmp_path):
        loop = _make_loop(tmp_path)
        working = loop._recall_working_tiers(
            {"episodic": "episodic"}, {}, _recalled_entries_from_store(loop)
        )
        assert working["episodic"].pre_sha == ""

    def test_pre_sha_reflects_the_live_on_disk_registry(self, tmp_path):
        loop = _make_loop(tmp_path)
        tier_root = loop.output_dir / "episodic"
        reg = KeyRegistry()
        reg.add("graph1")
        reg.save(tier_root / "indexed_key_registry.json")

        working = loop._recall_working_tiers(
            {"episodic": "episodic"}, {}, _recalled_entries_from_store(loop)
        )
        import hashlib

        expected = hashlib.sha256(
            (tier_root / "indexed_key_registry.json").read_bytes()
        ).hexdigest()
        assert working["episodic"].pre_sha == expected

    def test_recall_of_unborn_tier_does_not_register_it_in_live_store(self, tmp_path):
        """A tier name the live store has never allocated a registry for
        (an interim event minting a brand-new slot name) must seed its
        WorkingTier from a local, empty registry -- the recall read must
        not register the tier into the live store's tier list as a side
        effect. Go-live does not need this pre-registration: ``adopt_increments``
        rebinds the live registry itself at publish."""
        loop = _make_loop(tmp_path)
        unborn = "episodic_interim_20260101T0000"
        assert unborn not in loop.store.tiers_with_registry()

        working = loop._recall_working_tiers({unborn: unborn}, {}, {})

        assert working[unborn].registry.list_known() == []
        assert unborn not in loop.store.tiers_with_registry()


class TestWorkingRegistryTrueRelations:
    def test_raises_when_an_active_key_has_no_working_entry(self, tmp_path):
        """Fold-local hydration guarantees a working entry for every active
        key; a working tier whose registry reports a key active but whose
        entries dict carries nothing for it is a designed-impossible state,
        raised rather than silently skipped as an orphan."""
        from paramem.memory.store import BookkeepingInvariantViolation

        loop = _make_loop(tmp_path)
        registry = KeyRegistry()
        registry.add("graph1")
        wt = WorkingTier(
            tier="episodic",
            adapter_name="episodic",
            pre_sha="",
            scratch_dir=tmp_path / "scratch_ep",
            registry=registry,
            rows={
                "graph1": {
                    "speaker_id": "speaker0",
                    "relation_type": "factual",
                    "reinforcement_count": 1,
                }
            },
            entries={},
            rebuilt=True,
        )

        with pytest.raises(BookkeepingInvariantViolation):
            loop._working_registry_true_relations(wt)


class TestStageEventEmptyOutcome:
    def test_no_new_material_and_no_existing_content_returns_none(self, tmp_path):
        loop = _make_loop(tmp_path)
        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="s1",
            primary_tiers={"episodic": "episodic"},
        )
        assert result is None
        # No ledger, no shadow tree.
        assert not (loop._fold_state_dir / "stage_ledger.json").exists()
        assert not (loop._fold_state_dir / "extraction").exists()


class TestStageEventConvergesOnAnAllUnkeyableMergedGraph:
    """Regression pin: ``stage_event`` used to carry a SECOND early exit,
    after the merge, on ``merger.graph.number_of_edges() == 0 and not
    has_attributes`` -- deleted because the pre-recall exit above already
    covers every empty-input shape except one reachable one: no new
    material, but every active key in the working universe carries an
    entry with an empty predicate (``unkeyable_no_predicate`` --
    ``_working_registry_true_relations`` skips it and records the removal
    without ever producing a graph edge or attribute). The deleted exit
    would have returned ``None`` right there -- discarding the unkeyable
    removal before the fate machinery ever ran, leaving the garbage key
    active forever (a permanent no-op loop). ``stage_event`` must instead
    flow through: the fate machinery retires the unkeyable key and the
    event converges to a rows-only tier.
    """

    def test_all_unkeyable_active_keys_still_converge(self, tmp_path):
        loop = _make_loop(tmp_path)
        loop.store.registry("episodic").add("graph1")
        loop.store.set_bookkeeping(
            "graph1",
            speaker_id="speaker0",
            relation_type="factual",
            reinforcement_count=1,
            last_reinforced_cycle=0,
            last_seen="2026-01-01T00:00:00Z",
            first_seen="2026-01-01T00:00:00Z",
            promoted=False,
        )
        # Content-bearing subject/object but an EMPTY predicate: skipped by
        # _working_registry_true_relations, so the merged graph ends up
        # with zero edges and zero node attributes -- exactly the shape
        # the deleted exit special-cased.
        loop.store.put(
            "episodic",
            "graph1",
            {"key": "graph1", "subject": "alex", "predicate": "", "object": "berlin"},
            register=False,
        )

        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="s1",
            primary_tiers={"episodic": "episodic"},
        )

        # Proceeds instead of returning None -- the deleted second exit
        # would have stopped here, leaving graph1 active forever.
        assert result is not None
        assert result.built_tiers == ("episodic",)

        # The unkeyable key is retired by the ordinary fate machinery (no
        # survivor_key named, so REMOVED rather than staled) -- the
        # written shadow registry no longer carries it active.
        registry = KeyRegistry.load(
            _shadow_dir(loop, "interim", "episodic") / "indexed_key_registry.json"
        )
        assert "graph1" not in registry.list_active()


class TestStageEventMintsAndPersists:
    def test_new_relation_mints_a_key_and_writes_the_shadow_tree(self, tmp_path):
        loop = _make_loop(tmp_path)
        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="stamp1",
            primary_tiers={"episodic": "episodic"},
            episodic_rels=[_rel("alex", "lives_in", "berlin")],
            session_ids=["s1"],
        )
        assert result is not None
        assert result.built_tiers == ("episodic",)
        assert result.ledger.tiers["episodic"]["adapter"] == "episodic"
        assert result.ledger.tiers["episodic"]["pre_sha"] == ""
        assert len(result.ledger.stages) == 1
        stage = result.ledger.stages[0]
        assert stage["stage"] == "extraction"
        assert stage["sessions"] == ["s1"]
        assert stage["episodic_rels"] == 1

        shadow = _shadow_dir(loop, "interim", "episodic")
        assert (shadow / "indexed_key_registry.json").exists()
        assert (shadow / "key_metadata.json").exists()
        keyed = json.loads((shadow / "keyed.json").read_text())
        assert len(keyed) == 1
        assert keyed[0]["subject"] == "alex"
        assert keyed[0]["predicate"] == "lives in"
        assert keyed[0]["object"] == "berlin"
        assert keyed[0]["key"].startswith("graph")

        registry = KeyRegistry.load(shadow / "indexed_key_registry.json")
        assert registry.list_active() == [keyed[0]["key"]]

        ledger_path = loop._fold_state_dir / "stage_ledger.json"
        assert ledger_path.exists()

        from paramem.training import stage_ledger as sl

        read_back = sl.read_ledger(loop._fold_state_dir)
        assert read_back is not None
        assert read_back.event == "interim"
        assert sl.verify(read_back.stages[0]) is True

    def test_stage_event_writes_no_event_root_graph_json(self, tmp_path):
        """``stage_event`` writes only the per-tier shadow tree
        (``indexed_key_registry.json``, ``key_metadata.json``,
        ``keyed.json`` under ``shadow/<tier>/``) — no event-root
        ``graph.json`` snapshot. Resume runs its later phase on a fresh
        empty merger and the simulate payload projects from
        ``increment.keyed``, so a snapshot has no reader; the merged graph
        is available through the ``on_fold_graph`` debug hook instead."""
        loop = _make_loop(tmp_path)
        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="stamp_no_graph",
            primary_tiers={"episodic": "episodic"},
            episodic_rels=[_rel("alex", "lives_in", "berlin")],
        )
        assert result is not None
        event_dir = loop._fold_state_dir / "extraction" / "interim"
        assert not (event_dir / "graph.json").exists()
        assert list(event_dir.rglob("graph.json")) == []

    def test_procedural_relation_routes_to_the_procedural_tier(self, tmp_path):
        loop = _make_loop(tmp_path, procedural=True)
        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="full",
            venue="weights",
            stamp="stampF",
            primary_tiers={
                "episodic": "episodic",
                "semantic": "semantic",
                "procedural": "procedural",
            },
            episodic_rels=[_rel("alex", "prefers", "acme radio", relation_type="preference")],
        )
        assert result is not None
        assert "procedural" in result.built_tiers
        keyed = json.loads((_shadow_dir(loop, "full", "procedural") / "keyed.json").read_text())
        assert len(keyed) == 1
        assert keyed[0]["key"].startswith("proc")

    def test_new_relation_mints_into_an_interim_shaped_primary_tier(self, tmp_path):
        """An interim tick's sole primary tier is named
        ``episodic_interim_<stamp>``, not the literal ``"episodic"`` — a
        genuinely new (keyless) fact must still land in it rather than being
        silently dropped because its computed destination is a literal main-
        tier name no primary tier carries.
        """
        loop = _make_loop(tmp_path)
        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="20260101T0000",
            primary_tiers={"episodic_interim_20260101T0000": "episodic_interim_20260101T0000"},
            episodic_rels=[_rel("alex", "lives_in", "berlin")],
        )
        assert result is not None
        assert result.built_tiers == ("episodic_interim_20260101T0000",)

        shadow = _shadow_dir(loop, "interim", "episodic_interim_20260101T0000")
        keyed = json.loads((shadow / "keyed.json").read_text())
        assert len(keyed) == 1
        assert keyed[0]["subject"] == "alex"
        assert keyed[0]["object"] == "berlin"

        registry = KeyRegistry.load(shadow / "indexed_key_registry.json")
        assert registry.list_active() == [keyed[0]["key"]]

    def test_new_preference_relation_mints_into_an_interim_shaped_primary_tier(self, tmp_path):
        """The same defect, for a procedural-shaped (preference) fact: its
        computed kind is "procedural", but with no literal "procedural"
        primary tier in an interim event's working universe the fact must
        still land in the interim tick's own primary tier rather than being
        dropped for want of a "procedural" destination.
        """
        loop = _make_loop(tmp_path)
        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="20260101T0000",
            primary_tiers={"episodic_interim_20260101T0000": "episodic_interim_20260101T0000"},
            episodic_rels=[_rel("alex", "prefers", "acme radio", relation_type="preference")],
        )
        assert result is not None
        assert result.built_tiers == ("episodic_interim_20260101T0000",)

        shadow = _shadow_dir(loop, "interim", "episodic_interim_20260101T0000")
        keyed = json.loads((shadow / "keyed.json").read_text())
        assert len(keyed) == 1
        assert keyed[0]["relation_type"] == "preference"
        assert keyed[0]["object"] == "acme radio"

    def test_build_tier_increment_reads_back_the_shadow_tree(self, tmp_path):
        loop = _make_loop(tmp_path)
        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="stamp1",
            primary_tiers={"episodic": "episodic"},
            episodic_rels=[_rel("alex", "lives_in", "berlin")],
        )
        assert result is not None

        from paramem.memory.increment import build_tier_increment

        inc = build_tier_increment(
            tier="episodic",
            adapter_name=result.ledger.tiers["episodic"]["adapter"],
            pre_sha=result.ledger.tiers["episodic"]["pre_sha"],
            shadow_dir=_shadow_dir(loop, "interim", "episodic"),
        )
        assert inc.rebuilt is True
        assert inc.has_payload is True
        assert len(inc.keyed) == 1
        assert list(inc.entries.values())[0]["object"] == "berlin"


class TestStageEventReplaysExistingContent:
    def test_existing_active_key_is_replayed_without_new_material(self, tmp_path):
        loop = _make_loop(tmp_path)
        loop.store.registry("episodic").add("graph1")
        loop.store.set_bookkeeping(
            "graph1",
            speaker_id="speaker0",
            relation_type="factual",
            reinforcement_count=1,
            last_reinforced_cycle=0,
            last_seen="2026-01-01T00:00:00Z",
            first_seen="2026-01-01T00:00:00Z",
            promoted=False,
        )
        loop.store.put(
            "episodic",
            "graph1",
            {"key": "graph1", "subject": "alex", "predicate": "lives in", "object": "berlin"},
            register=False,
        )

        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="full",
            venue="weights",
            stamp="stampF",
            primary_tiers={"episodic": "episodic", "semantic": "semantic"},
        )
        assert result is not None
        keyed = json.loads((_shadow_dir(loop, "full", "episodic") / "keyed.json").read_text())
        assert [row["key"] for row in keyed] == ["graph1"]
        assert keyed[0]["object"] == "berlin"


class TestStageEventAttributeKeyedWalk:
    """The keyed walk's node-attribute pass
    (``ConsolidationLoop._build_working_keyed_walk``): a keyless attribute
    mints a row carrying the RECORD's own speaker_id/window (never a
    node-level fallback), a keyed attribute replays from its owning
    working tier without minting again, and a keyless value change onto a
    recalled keyed record mints a NEW key under the NEW value's speaker
    and window -- never the superseded assertion's."""

    def test_keyless_attribute_mint_carries_the_records_speaker_and_window(self, tmp_path):
        """The subject ("acme corp") is not a speaker node, so it carries
        no top-level ``speaker_id`` node attribute at all -- if the mint
        ever fell back to reading one (the deleted node/enrichment
        fallback chain), ``bookkeeping_row`` would raise ValueError on the
        empty string instead of succeeding with the relation's own
        speaker_id."""
        loop = _make_loop(tmp_path)
        rel = _rel(
            "acme corp",
            "has_founder",
            "Jane Doe",
            relation_type="attribute",
            speaker_id="speaker2",
            first_seen="2026-02-01T00:00:00Z",
            last_seen="2026-02-01T00:00:00Z",
        )
        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="stampA",
            primary_tiers={"episodic": "episodic"},
            episodic_rels=[rel],
            session_ids=["s1"],
        )
        assert result is not None
        shadow = _shadow_dir(loop, "interim", "episodic")
        keyed = json.loads((shadow / "keyed.json").read_text())
        assert len(keyed) == 1
        assert keyed[0]["predicate"] == "has founder"
        assert keyed[0]["object"] == "Jane Doe"
        minted_key = keyed[0]["key"]

        rows = json.loads((shadow / "key_metadata.json").read_text())["keys"]
        row = rows[minted_key]
        assert row["speaker_id"] == "speaker2"
        assert row["first_seen"] == "2026-02-01T00:00:00Z"
        assert row["last_seen"] == "2026-02-01T00:00:00Z"

    def test_keyed_attribute_replays_without_minting_a_new_key(self, tmp_path):
        loop = _make_loop(tmp_path)
        loop.store.registry("episodic").add("graph1")
        loop.store.set_bookkeeping(
            "graph1",
            speaker_id="speaker0",
            relation_type="attribute",
            reinforcement_count=1,
            last_reinforced_cycle=0,
            last_seen="2026-01-01T00:00:00Z",
            first_seen="2026-01-01T00:00:00Z",
            promoted=False,
        )
        loop.store.put(
            "episodic",
            "graph1",
            {"key": "graph1", "subject": "alex", "predicate": "has email", "object": "a@b.com"},
            register=False,
        )

        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="full",
            venue="weights",
            stamp="stampF",
            primary_tiers={"episodic": "episodic", "semantic": "semantic"},
        )
        assert result is not None
        keyed = json.loads((_shadow_dir(loop, "full", "episodic") / "keyed.json").read_text())
        assert [row["key"] for row in keyed] == ["graph1"], (
            "a keyed attribute replays from its owning working tier -- it must "
            "never mint a second key for the same fact"
        )
        assert keyed[0]["object"] == "a@b.com"

    def test_keyless_value_change_after_recall_mints_a_new_key_with_the_new_speaker_and_window(
        self, tmp_path
    ):
        """graph1 (speakerA, 2026-01-01) is recalled first (registry-true
        recall precedes the pending merge), establishing the node's
        attribute record with ik_key="graph1"; the pending relation then
        re-observes the SAME predicate keyless with a DIFFERENT value from
        a DIFFERENT speaker at a LATER date -- the merger's value-change
        branch supersedes graph1 (no survivor) and the keyed walk mints a
        brand new key carrying the new value's own speaker and window,
        never graph1's."""
        loop = _make_loop(tmp_path)
        loop.store.registry("episodic").add("graph1")
        loop.store.set_bookkeeping(
            "graph1",
            speaker_id="speakerA",
            relation_type="attribute",
            reinforcement_count=1,
            last_reinforced_cycle=0,
            last_seen="2026-01-01T00:00:00Z",
            first_seen="2026-01-01T00:00:00Z",
            promoted=False,
        )
        loop.store.put(
            "episodic",
            "graph1",
            {
                "key": "graph1",
                "subject": "alex",
                "predicate": "has email",
                "object": "old@example.com",
            },
            register=False,
        )
        # _make_loop hardcodes _indexed_next_index=1 (production derives it
        # from existing keys via _derive_key_counters, __init__-only); bump
        # it past the pre-seeded "graph1" so the new mint below gets its own
        # identity instead of colliding with the key this same event retires.
        loop._indexed_next_index = 2

        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="full",
            venue="weights",
            stamp="stampV",
            primary_tiers={"episodic": "episodic"},
            episodic_rels=[
                _rel(
                    "alex",
                    "has_email",
                    "new@example.com",
                    relation_type="attribute",
                    speaker_id="speakerB",
                    first_seen="2026-06-01T00:00:00Z",
                    last_seen="2026-06-01T00:00:00Z",
                )
            ],
        )
        assert result is not None
        shadow = _shadow_dir(loop, "full", "episodic")
        keyed = json.loads((shadow / "keyed.json").read_text())
        assert len(keyed) == 1
        assert keyed[0]["key"] != "graph1", "the superseded key must not carry forward"
        assert keyed[0]["object"] == "new@example.com"

        rows = json.loads((shadow / "key_metadata.json").read_text())["keys"]
        assert "graph1" not in rows, "the superseded key's row must not survive the fold"
        new_row = rows[keyed[0]["key"]]
        assert new_row["speaker_id"] == "speakerB", "never the superseded assertion's speaker"
        assert new_row["first_seen"] == "2026-06-01T00:00:00Z"
        assert new_row["last_seen"] == "2026-06-01T00:00:00Z"


class TestPromotion:
    def test_matured_episodic_key_is_promoted_to_semantic(self, tmp_path):
        loop = _make_loop(tmp_path)
        loop.store.registry("episodic").add("graph1")
        loop.store.set_bookkeeping(
            "graph1",
            speaker_id="speaker0",
            relation_type="factual",
            reinforcement_count=5,
            last_reinforced_cycle=0,
            last_seen="2026-01-01T00:00:00Z",
            first_seen="2026-01-01T00:00:00Z",
            promoted=False,
        )
        loop.store.put(
            "episodic",
            "graph1",
            {"key": "graph1", "subject": "alex", "predicate": "lives in", "object": "berlin"},
            register=False,
        )

        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="full",
            venue="weights",
            stamp="stampF",
            primary_tiers={"episodic": "episodic", "semantic": "semantic"},
            promote=True,
        )
        assert result is not None
        episodic_keyed = json.loads(
            (_shadow_dir(loop, "full", "episodic") / "keyed.json").read_text()
        )
        semantic_keyed = json.loads(
            (_shadow_dir(loop, "full", "semantic") / "keyed.json").read_text()
        )
        assert episodic_keyed == []
        assert [row["key"] for row in semantic_keyed] == ["graph1"]
        # promoted_keys itself is updated only once run_build_and_publish
        # confirms the event went live (see _promote_working_keys' own
        # docstring) -- staging alone records the decision on the
        # transient _pending_promoted_keys.
        assert "graph1" not in loop.promoted_keys
        assert "graph1" in loop._pending_promoted_keys

    def test_promoted_keys_published_row_carries_full_bookkeeping(self, tmp_path):
        """The row that lands on the semantic working tier after promotion
        is the source row moved whole, not a fabricated ``{"promoted": True}``
        stub."""
        loop = _make_loop(tmp_path)
        loop.store.registry("episodic").add("graph1")
        loop.store.set_bookkeeping(
            "graph1",
            speaker_id="speaker0",
            relation_type="factual",
            reinforcement_count=5,
            last_reinforced_cycle=0,
            last_seen="2026-01-01T00:00:00Z",
            first_seen="2026-01-01T00:00:00Z",
            promoted=False,
        )
        loop.store.put(
            "episodic",
            "graph1",
            {"key": "graph1", "subject": "alex", "predicate": "lives in", "object": "berlin"},
            register=False,
        )
        working = loop._recall_working_tiers(
            {"episodic": "episodic", "semantic": "semantic"}, {}, _recalled_entries_from_store(loop)
        )

        loop._promote_working_keys(working)

        row = working["semantic"].rows["graph1"]
        assert row["promoted"] is True
        assert row["speaker_id"] == "speaker0"
        assert row["relation_type"] == "factual"
        assert row["first_seen"] == "2026-01-01T00:00:00Z"
        assert row["last_seen"] == "2026-01-01T00:00:00Z"
        assert row["reinforcement_count"] == 5

    def test_below_threshold_key_is_left_on_episodic_untouched(self, tmp_path):
        """A key whose reinforcement count sits below ``promotion_threshold``
        (and whose ``last_reinforced_cycle`` keeps the decay branch inert
        too) falls through both branches: it is reported as no promotion,
        stays resident on the working episodic tier, never reaches
        semantic, and its own row is left exactly as seeded."""
        loop = _make_loop(tmp_path)
        loop.store.registry("episodic").add("graph1")
        loop.store.set_bookkeeping(
            "graph1",
            speaker_id="speaker0",
            relation_type="factual",
            reinforcement_count=1,
            last_reinforced_cycle=0,
            last_seen="2026-01-01T00:00:00Z",
            first_seen="2026-01-01T00:00:00Z",
            promoted=False,
        )
        loop.store.put(
            "episodic",
            "graph1",
            {"key": "graph1", "subject": "alex", "predicate": "lives in", "object": "berlin"},
            register=False,
        )
        working = loop._recall_working_tiers(
            {"episodic": "episodic", "semantic": "semantic"},
            {},
            _recalled_entries_from_store(loop),
        )

        newly_promoted = loop._promote_working_keys(working)

        assert newly_promoted == []
        assert "graph1" in working["episodic"].registry.list_active()
        assert "graph1" in working["episodic"].rows
        assert "graph1" not in working["semantic"].registry.list_active()
        assert "graph1" not in working["semantic"].rows
        assert working["episodic"].rows["graph1"]["promoted"] is False
        assert loop._pending_promoted_keys == set()
        assert loop.promoted_keys == set()

    def test_net_new_key_promotes_only_after_one_re_observation_at_threshold_2(self, tmp_path):
        """A net-new key is minted at reinforcement_count=1 (the same
        default every fresh key row carries). At promotion_threshold=2 --
        the smallest threshold config validation accepts, because 1 is
        every key's own first-staging value and 2 is what a re-observation
        produces -- the key must NOT promote on its first pass, and MUST
        promote once one re-observation has bumped its count to 2."""
        loop = _make_loop(tmp_path)
        loop.config.promotion_threshold = 2
        loop.store.registry("episodic").add("graph1")
        loop.store.set_bookkeeping(
            "graph1",
            speaker_id="speaker0",
            relation_type="factual",
            reinforcement_count=1,
            last_reinforced_cycle=0,
            last_seen="2026-01-01T00:00:00Z",
            first_seen="2026-01-01T00:00:00Z",
            promoted=False,
        )
        loop.store.put(
            "episodic",
            "graph1",
            {"key": "graph1", "subject": "alex", "predicate": "lives in", "object": "berlin"},
            register=False,
        )

        working = loop._recall_working_tiers(
            {"episodic": "episodic", "semantic": "semantic"},
            {},
            _recalled_entries_from_store(loop),
        )
        first_pass = loop._promote_working_keys(working)

        assert first_pass == []
        assert "graph1" in working["episodic"].registry.list_active()
        assert "graph1" not in working["semantic"].registry.list_active()

        # One re-observation: bump reinforcement_count to 2 on the SAME
        # working row (mirrors the shared credit_reinforcement primitive's
        # effect, without pulling in the full reinforcement-credit path).
        working["episodic"].rows["graph1"]["reinforcement_count"] = 2

        second_pass = loop._promote_working_keys(working)

        assert second_pass == ["graph1"]
        assert "graph1" not in working["episodic"].registry.list_active()
        assert "graph1" in working["semantic"].registry.list_active()
        assert working["semantic"].rows["graph1"]["promoted"] is True
        assert working["semantic"].rows["graph1"]["reinforcement_count"] == 2


class TestApplyWorkingReinforcementCreditDirect:
    """Direct unit tests of
    :meth:`ConsolidationLoop._apply_working_reinforcement_credit`.

    A retired key's owning tier is resolved independently of the
    survivor's -- absorbing a key from a DIFFERENT working tier than the
    survivor's must inherit that key's own durable count, not zero."""

    def _seed(self, loop, tier, key, *, count, last_seen):
        loop.store.registry(tier).add(key)
        loop.store.set_bookkeeping(
            key,
            speaker_id="speaker0",
            relation_type="factual",
            reinforcement_count=count,
            last_reinforced_cycle=0,
            last_seen=last_seen,
            first_seen=last_seen,
            promoted=False,
        )
        loop.store.put(
            tier,
            key,
            {"key": key, "subject": "a", "predicate": "b", "object": "c"},
            register=False,
        )

    def test_cross_tier_absorption_inherits_the_absorbed_tiers_count(self, tmp_path):
        loop = _make_loop(tmp_path)
        self._seed(loop, "episodic", "survivor", count=1, last_seen="2026-01-01T00:00:00Z")
        self._seed(
            loop,
            "episodic_interim_20260101T0000",
            "absorbed",
            count=4,
            last_seen="2026-01-02T00:00:00Z",
        )
        working = loop._recall_working_tiers(
            {"episodic": "episodic"},
            {"episodic_interim_20260101T0000": "episodic_interim_20260101T0000"},
            _recalled_entries_from_store(loop),
        )
        loop.merger.record_removal("absorbed", reason="dedup", survivor_key="survivor")

        loop._apply_working_reinforcement_credit(working, {})

        # own 1 vs. inherited 4 -> 4, plus 1 earned for the fresh sighting -> 5.
        assert working["episodic"].rows["survivor"]["reinforcement_count"] == 5

    def test_same_tier_absorption_keeps_todays_value(self, tmp_path):
        loop = _make_loop(tmp_path)
        self._seed(loop, "episodic", "survivor", count=1, last_seen="2026-01-01T00:00:00Z")
        self._seed(loop, "episodic", "absorbed", count=4, last_seen="2026-01-02T00:00:00Z")
        working = loop._recall_working_tiers(
            {"episodic": "episodic"}, {}, _recalled_entries_from_store(loop)
        )
        loop.merger.record_removal("absorbed", reason="dedup", survivor_key="survivor")

        loop._apply_working_reinforcement_credit(working, {})

        assert working["episodic"].rows["survivor"]["reinforcement_count"] == 5

    def test_unknown_absorbed_key_contributes_nothing(self, tmp_path):
        loop = _make_loop(tmp_path)
        self._seed(loop, "episodic", "survivor", count=1, last_seen="2026-01-01T00:00:00Z")
        working = loop._recall_working_tiers(
            {"episodic": "episodic"}, {}, _recalled_entries_from_store(loop)
        )
        loop.merger.record_removal("ghost", reason="dedup", survivor_key="survivor")

        loop._apply_working_reinforcement_credit(working, {})

        assert working["episodic"].rows["survivor"]["reinforcement_count"] == 1

    def test_removal_ledger_read_directly_no_getattr_default(self, tmp_path, monkeypatch):
        """The merger always defines removal_ledger -- the defensive
        getattr default is gone; a merger object lacking the attribute
        entirely must raise, not silently no-op."""
        loop = _make_loop(tmp_path)
        working = loop._recall_working_tiers(
            {"episodic": "episodic"}, {}, _recalled_entries_from_store(loop)
        )

        class _MergerWithoutLedger:
            pass

        loop.merger = _MergerWithoutLedger()
        with pytest.raises(AttributeError):
            loop._apply_working_reinforcement_credit(working, {})

    def test_older_re_observation_earns_nothing_on_either_call(self, tmp_path):
        """A survivor whose stored ``last_seen`` already exceeds the credit
        timestamp earns zero -- the row already reflects a newer sighting,
        and an older re-observation must not move the count either call."""
        loop = _make_loop(tmp_path)
        self._seed(loop, "episodic", "survivor", count=1, last_seen="2026-01-05T00:00:00Z")
        working = loop._recall_working_tiers(
            {"episodic": "episodic"}, {}, _recalled_entries_from_store(loop)
        )
        older = ("2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")

        loop._apply_working_reinforcement_credit(working, {"survivor": older})
        assert working["episodic"].rows["survivor"]["reinforcement_count"] == 1

        loop._apply_working_reinforcement_credit(working, {"survivor": older})
        assert working["episodic"].rows["survivor"]["reinforcement_count"] == 1

    def test_newer_re_observation_earns_exactly_once_across_two_calls(self, tmp_path):
        """A strictly-newer timestamp earns exactly once: the first call
        advances ``last_seen`` and earns; the second call, run against the
        same now-current timestamp, no longer sees a strictly-newer value
        and earns nothing -- idempotency under an unchanged ledger."""
        loop = _make_loop(tmp_path)
        self._seed(loop, "episodic", "survivor", count=1, last_seen="2026-01-01T00:00:00Z")
        working = loop._recall_working_tiers(
            {"episodic": "episodic"}, {}, _recalled_entries_from_store(loop)
        )
        newer = ("2026-01-05T00:00:00Z", "2026-01-05T00:00:00Z")

        loop._apply_working_reinforcement_credit(working, {"survivor": newer})
        assert working["episodic"].rows["survivor"]["reinforcement_count"] == 2

        loop._apply_working_reinforcement_credit(working, {"survivor": newer})
        assert working["episodic"].rows["survivor"]["reinforcement_count"] == 2


class TestWorkingTierAdoptKeyFromDirect:
    """Direct unit tests of :meth:`WorkingTier.adopt_key_from` -- the
    staging-layer half of the one carry rule for a key changing tier."""

    def test_raises_when_source_has_no_bookkeeping_row_for_the_key(self, tmp_path):
        """Every known key already carries a bookkeeping row (the invariant
        established at recall); a source WorkingTier lacking one for a key
        it registers is a violation, checked and raised BEFORE either
        registry is mutated -- both working copies stay untouched."""
        from paramem.memory.store import BookkeepingInvariantViolation

        src_registry = KeyRegistry()
        src_registry.add("graph1")
        source = WorkingTier(
            tier="episodic",
            adapter_name="episodic",
            pre_sha="",
            scratch_dir=tmp_path / "scratch_ep",
            registry=src_registry,
            rows={},
            entries={
                "graph1": {
                    "key": "graph1",
                    "subject": "alex",
                    "predicate": "lives in",
                    "object": "berlin",
                }
            },
            rebuilt=True,
        )
        dest = WorkingTier(
            tier="semantic",
            adapter_name="semantic",
            pre_sha="",
            scratch_dir=tmp_path / "scratch_sem",
            registry=KeyRegistry(),
            rows={},
            entries={},
            rebuilt=True,
        )

        with pytest.raises(BookkeepingInvariantViolation):
            dest.adopt_key_from(source, "graph1")

        assert "graph1" in source.registry.list_active()
        assert "graph1" not in dest.registry.list_active()
        assert "graph1" in source.entries
        assert "graph1" not in dest.entries

    def test_raises_when_source_has_no_working_entry_for_the_key(self, tmp_path):
        """Fold-local hydration guarantees a working entry for every active
        key; a source WorkingTier registering a key but carrying no working
        entry for it is a violation, checked and raised BEFORE either
        registry is mutated -- both working copies stay untouched."""
        from paramem.memory.store import BookkeepingInvariantViolation

        src_registry = KeyRegistry()
        src_registry.add("graph1")
        source = WorkingTier(
            tier="episodic",
            adapter_name="episodic",
            pre_sha="",
            scratch_dir=tmp_path / "scratch_ep",
            registry=src_registry,
            rows={
                "graph1": {
                    "speaker_id": "speaker0",
                    "relation_type": "factual",
                    "reinforcement_count": 1,
                }
            },
            entries={},
            rebuilt=True,
        )
        dest = WorkingTier(
            tier="semantic",
            adapter_name="semantic",
            pre_sha="",
            scratch_dir=tmp_path / "scratch_sem",
            registry=KeyRegistry(),
            rows={},
            entries={},
            rebuilt=True,
        )

        with pytest.raises(BookkeepingInvariantViolation):
            dest.adopt_key_from(source, "graph1")

        assert "graph1" in source.registry.list_active()
        assert "graph1" not in dest.registry.list_active()
        assert "graph1" in source.rows
        assert "graph1" not in dest.rows


class TestCandidateTiers:
    def test_untouched_candidate_tier_is_not_built(self, tmp_path):
        loop = _make_loop(tmp_path)
        loop.store.registry("semantic").add("graph_sem")
        loop.store.set_bookkeeping(
            "graph_sem",
            speaker_id="speaker0",
            relation_type="factual",
            reinforcement_count=1,
            last_reinforced_cycle=0,
            last_seen="2026-01-01T00:00:00Z",
            first_seen="2026-01-01T00:00:00Z",
            promoted=False,
        )
        loop.store.put(
            "semantic",
            "graph_sem",
            {"key": "graph_sem", "subject": "alex", "predicate": "works at", "object": "acme"},
            register=False,
        )

        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="stampI",
            primary_tiers={"episodic_interim_20260101T0000": "episodic_interim_20260101T0000"},
            candidate_tiers={"semantic": "semantic"},
            episodic_rels=[_rel("alex", "likes", "coffee")],
        )
        assert result is not None
        assert "semantic" not in result.built_tiers
        assert not _shadow_dir(loop, "interim", "semantic").exists()

    def test_dedup_collapse_against_a_candidate_tier_writes_a_rows_only_member(self, tmp_path):
        loop = _make_loop(tmp_path)
        loop.store.registry("semantic").add("graph_sem")
        loop.store.set_bookkeeping(
            "graph_sem",
            speaker_id="speaker0",
            relation_type="factual",
            reinforcement_count=1,
            last_reinforced_cycle=0,
            last_seen="2026-01-01T00:00:00Z",
            first_seen="2026-01-01T00:00:00Z",
            promoted=False,
        )
        loop.store.put(
            "semantic",
            "graph_sem",
            {"key": "graph_sem", "subject": "alex", "predicate": "lives in", "object": "berlin"},
            register=False,
        )

        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="stampI",
            primary_tiers={"episodic_interim_20260101T0000": "episodic_interim_20260101T0000"},
            candidate_tiers={"semantic": "semantic"},
            # A re-observation of the SAME fact already keyed in semantic --
            # Case-1 collapses onto graph_sem rather than minting a new key.
            episodic_rels=[_rel("alex", "lives_in", "berlin")],
        )
        assert result is not None
        assert "semantic" in result.built_tiers
        assert not (_shadow_dir(loop, "interim", "semantic") / "keyed.json").exists()
        assert (_shadow_dir(loop, "interim", "semantic") / "key_metadata.json").exists()
        # The interim tier itself mints nothing -- the fact was already keyed.
        interim_keyed = json.loads(
            (
                _shadow_dir(loop, "interim", "episodic_interim_20260101T0000") / "keyed.json"
            ).read_text()
        )
        assert interim_keyed == []

    def test_a_stale_keyed_json_does_not_survive_into_this_events_rows_only_member(self, tmp_path):
        """A ``keyed.json`` left behind under this tier's shadow directory by
        an earlier crashed staging pass -- one that never reached its own
        ledger write, so nothing durable ever named it -- must not leak
        into THIS event's rows-only member and be misread as
        ``rebuilt=True``."""
        loop = _make_loop(tmp_path)
        loop.store.registry("semantic").add("graph_sem")
        loop.store.set_bookkeeping(
            "graph_sem",
            speaker_id="speaker0",
            relation_type="factual",
            reinforcement_count=1,
            last_reinforced_cycle=0,
            last_seen="2026-01-01T00:00:00Z",
            first_seen="2026-01-01T00:00:00Z",
            promoted=False,
        )
        loop.store.put(
            "semantic",
            "graph_sem",
            {"key": "graph_sem", "subject": "alex", "predicate": "lives in", "object": "berlin"},
            register=False,
        )

        stale_keyed = _shadow_dir(loop, "interim", "semantic") / "keyed.json"
        stale_keyed.parent.mkdir(parents=True, exist_ok=True)
        stale_keyed.write_text(json.dumps([{"key": "stale_ghost_key"}]))

        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="stampI",
            primary_tiers={"episodic_interim_20260101T0000": "episodic_interim_20260101T0000"},
            candidate_tiers={"semantic": "semantic"},
            episodic_rels=[_rel("alex", "lives_in", "berlin")],
        )
        assert result is not None
        assert "semantic" in result.built_tiers

        from paramem.memory.increment import build_tier_increment

        increment = build_tier_increment(
            tier="semantic",
            adapter_name=result.ledger.tiers["semantic"]["adapter"],
            pre_sha=result.ledger.tiers["semantic"]["pre_sha"],
            shadow_dir=_shadow_dir(loop, "interim", "semantic"),
        )
        assert increment.rebuilt is False, (
            "the stale keyed.json must not leak into this rows-only member"
        )
        assert not stale_keyed.exists()


class TestExtractionTreeClearedAtEntry:
    """``stage_event`` is the only writer under ``<state_dir>/extraction/``
    and clears the whole tree wholesale at entry -- reclaiming debris from
    ANY staging pass that crashed before its own ledger was ever written,
    including a different event kind than the one about to run."""

    def test_a_crashed_other_kind_staging_pass_debris_is_removed(self, tmp_path):
        loop = _make_loop(tmp_path)

        full_debris = _shadow_dir(loop, "full", "episodic") / "keyed.json"
        full_debris.parent.mkdir(parents=True, exist_ok=True)
        full_debris.write_text(json.dumps([{"key": "orphan_from_a_crashed_full_fold"}]))
        full_graph = loop._fold_state_dir / "extraction" / "full" / "graph.json"
        full_graph.write_text("{}")

        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="20260101T0000",
            primary_tiers={"episodic": "episodic"},
            episodic_rels=[_rel("alex", "lives_in", "berlin")],
        )
        assert result is not None
        assert not full_debris.exists()
        assert not (loop._fold_state_dir / "extraction" / "full").exists()

    def test_this_events_own_prior_debris_is_removed_before_the_fresh_write(self, tmp_path):
        loop = _make_loop(tmp_path)

        # Debris under the SAME event kind's tree, from an earlier crashed
        # attempt that never reached its own ledger write.
        stale = _shadow_dir(loop, "interim", "episodic") / "keyed.json"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text(json.dumps([{"key": "stale_from_a_crashed_attempt"}]))

        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="20260101T0000",
            primary_tiers={"episodic": "episodic"},
            episodic_rels=[_rel("alex", "lives_in", "berlin")],
        )
        assert result is not None
        fresh_keyed = json.loads(stale.read_text())
        assert "stale_from_a_crashed_attempt" not in [k.get("key") for k in fresh_keyed]


class TestAbsorbedCandidateTierRouting:
    """A full-topology event (a full fold or a reconcile,
    ``stage_ledger.full_topology(event)``) absorbs its candidate tiers
    whole -- they are the interim ring, never built, published or
    restamped, reaped at the go-live.  A key only one of them owns must
    therefore be routed into whichever primary tier its own stored
    ``relation_type`` selects, or it silently vanishes the moment the ring
    is reaped.  The opposite universe is an interim tick's own candidate
    tiers (the three main tiers and its sibling slots): dedup-only/
    read-only, never routed, staying resident where it already lives
    (``TestCandidateTiers`` above pins that shape; the last test in this
    class pins it again for a key with no duplicate elsewhere, since that
    is exactly the case a routing bug would corrupt if the full-topology
    gate ever drifted).
    """

    def _seed_key(
        self,
        loop,
        tier: str,
        key: str,
        *,
        subject: str,
        predicate: str,
        obj: str,
        relation_type: str = "factual",
        simhash: int = 424242,
        last_seen: str = "2026-01-01T00:00:00Z",
    ) -> None:
        loop.store.registry(tier).add(key)
        loop.store.registry(tier).set_simhash(key, simhash)
        loop.store.set_bookkeeping(
            key,
            speaker_id="speaker0",
            relation_type=relation_type,
            reinforcement_count=1,
            last_reinforced_cycle=0,
            last_seen=last_seen,
            first_seen=last_seen,
            promoted=False,
        )
        loop.store.put(
            tier,
            key,
            {"key": key, "subject": subject, "predicate": predicate, "object": obj},
            register=False,
        )

    def test_unique_absorbed_key_routes_into_the_episodic_primary_tier(self, tmp_path):
        loop = _make_loop(tmp_path)
        self._seed_key(
            loop,
            "episodic_interim_20260101T0000",
            "interim1",
            subject="alex",
            predicate="visited",
            obj="lisbon",
        )

        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="full",
            venue="weights",
            stamp="stampF",
            primary_tiers={"episodic": "episodic", "semantic": "semantic"},
            candidate_tiers={"episodic_interim_20260101T0000": "episodic_interim_20260101T0000"},
        )
        assert result is not None
        assert "episodic" in result.built_tiers
        assert "episodic_interim_20260101T0000" not in result.built_tiers
        assert not _shadow_dir(loop, "full", "episodic_interim_20260101T0000").exists()

        keyed = json.loads((_shadow_dir(loop, "full", "episodic") / "keyed.json").read_text())
        assert [row["key"] for row in keyed] == ["interim1"]
        assert keyed[0]["object"] == "lisbon"

        registry = KeyRegistry.load(
            _shadow_dir(loop, "full", "episodic") / "indexed_key_registry.json"
        )
        assert "interim1" in registry.list_active()
        assert registry.simhash_for("interim1") == 424242

        from paramem.memory.increment import build_tier_increment

        inc = build_tier_increment(
            tier="episodic",
            adapter_name=result.ledger.tiers["episodic"]["adapter"],
            pre_sha=result.ledger.tiers["episodic"]["pre_sha"],
            shadow_dir=_shadow_dir(loop, "full", "episodic"),
        )
        assert inc.bookkeeping["interim1"]["speaker_id"] == "speaker0"

    def test_unique_absorbed_preference_key_routes_into_procedural(self, tmp_path):
        loop = _make_loop(tmp_path, procedural=True)
        self._seed_key(
            loop,
            "episodic_interim_20260101T0000",
            "interim_pref",
            subject="alex",
            predicate="prefers",
            obj="acme radio",
            relation_type="preference",
        )

        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="full",
            venue="weights",
            stamp="stampF",
            primary_tiers={
                "episodic": "episodic",
                "semantic": "semantic",
                "procedural": "procedural",
            },
            candidate_tiers={"episodic_interim_20260101T0000": "episodic_interim_20260101T0000"},
        )
        assert result is not None
        assert "procedural" in result.built_tiers
        assert "episodic_interim_20260101T0000" not in result.built_tiers

        proc_keyed = json.loads(
            (_shadow_dir(loop, "full", "procedural") / "keyed.json").read_text()
        )
        assert [row["key"] for row in proc_keyed] == ["interim_pref"]

        episodic_keyed = json.loads(
            (_shadow_dir(loop, "full", "episodic") / "keyed.json").read_text()
        )
        assert episodic_keyed == []

    def test_absorbed_duplicate_collapses_and_credits_the_surviving_key(self, tmp_path):
        from paramem.memory.entry import entry_simhash

        loop = _make_loop(tmp_path)
        entry = {"key": "graph1", "subject": "alex", "predicate": "lives in", "object": "berlin"}
        fp = entry_simhash(entry)
        self._seed_key(
            loop,
            "episodic",
            "graph1",
            subject="alex",
            predicate="lives in",
            obj="berlin",
            simhash=fp,
            last_seen="2026-01-01T00:00:00Z",
        )
        self._seed_key(
            loop,
            "episodic_interim_20260101T0000",
            "interim_dup",
            subject="alex",
            predicate="lives in",
            obj="berlin",
            simhash=fp,
            last_seen="2026-01-02T00:00:00Z",
        )

        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="full",
            venue="weights",
            stamp="stampF",
            primary_tiers={"episodic": "episodic", "semantic": "semantic"},
            candidate_tiers={"episodic_interim_20260101T0000": "episodic_interim_20260101T0000"},
        )
        assert result is not None
        assert "episodic_interim_20260101T0000" not in result.built_tiers
        assert not _shadow_dir(loop, "full", "episodic_interim_20260101T0000").exists()

        keyed = json.loads((_shadow_dir(loop, "full", "episodic") / "keyed.json").read_text())
        # Exactly one survivor -- the duplicate collapses rather than
        # doubling the fact under two keys.
        assert len(keyed) == 1
        survivor = keyed[0]["key"]
        assert survivor in ("graph1", "interim_dup")

        rows = json.loads((_shadow_dir(loop, "full", "episodic") / "key_metadata.json").read_text())
        # The independent sighting from the absorbed duplicate is credited
        # onto the survivor's row -- inherited maturity, not lost.
        assert rows["keys"][survivor]["reinforcement_count"] > 1

    def test_interim_events_own_candidate_tier_key_is_never_routed(self, tmp_path):
        """The opposite universe: an interim tick's candidate tiers (here,
        a sibling main tier) are dedup-only/read-only -- they stay resident
        where they already live and are never routed into the interim
        tick's own new slot, even for a key with no duplicate anywhere
        else.  Distinguishes the routing fix from a blanket "always route
        candidate-owned keys" change: the candidate tier's own registry-true
        content is still merged in (it participates in the dedup identity,
        per :meth:`_working_registry_true_relations`'s step-3 call in
        :meth:`stage_event`), but its key must come back out unrouted and
        unbuilt on the far side of the keyed walk.
        """
        loop = _make_loop(tmp_path)
        self._seed_key(
            loop,
            "semantic",
            "graph_sem",
            subject="alex",
            predicate="works at",
            obj="acme",
        )

        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="stampI",
            primary_tiers={"episodic_interim_20260101T0000": "episodic_interim_20260101T0000"},
            candidate_tiers={"semantic": "semantic"},
        )
        assert result is not None
        # The candidate tier's own key is untouched -- no routing, no build.
        assert "semantic" not in result.built_tiers
        assert not _shadow_dir(loop, "interim", "semantic").exists()
        interim_keyed = json.loads(
            (
                _shadow_dir(loop, "interim", "episodic_interim_20260101T0000") / "keyed.json"
            ).read_text()
        )
        assert interim_keyed == []
        assert "graph_sem" in loop.store.registry("semantic").list_active()


class TestRouteAbsorbedKeyedFactDirect:
    """Direct unit tests of :meth:`ConsolidationLoop._route_absorbed_keyed_fact`
    -- the helper :meth:`_build_working_keyed_walk` calls when
    ``absorb_candidates`` is ``True``."""

    def test_raises_when_the_owning_tier_has_no_entry(self, tmp_path):
        """A missing entry for an active key is a violated recall-completeness
        invariant, not a routable state -- the direct entry lookup raises."""
        loop = _make_loop(tmp_path)
        loop.store.registry("episodic_interim_20260101T0000").add("orphan")
        loop.store.set_bookkeeping(
            "orphan",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="",
            promoted=False,
        )
        # No store.put -- the working copy's entries bucket stays empty.

        working = loop._recall_working_tiers(
            {"episodic": "episodic"},
            {"episodic_interim_20260101T0000": "episodic_interim_20260101T0000"},
            _recalled_entries_from_store(loop),
        )
        tier_keyed = {"episodic": []}
        with pytest.raises(KeyError):
            loop._route_absorbed_keyed_fact(
                working, tier_keyed, key="orphan", owner=working["episodic_interim_20260101T0000"]
            )


class TestBuildWorkingKeyedWalkOwnerMissingEntry:
    """A keyed edge whose owning tier is itself one of this event's primary
    tiers replays via a direct working-entry lookup inside
    :meth:`ConsolidationLoop._build_working_keyed_walk` -- fold-local
    hydration guarantees an entry for every active key, so a gap there is a
    designed-impossible state, raised rather than silently skipped."""

    def test_raises_when_the_owning_primary_tier_has_no_working_entry(self, tmp_path):
        from paramem.memory.persistence import _IK_KEY_ATTR
        from paramem.memory.store import BookkeepingInvariantViolation

        loop = _make_loop(tmp_path)
        g = loop.merger.graph
        g.add_node("speaker0", speaker_id="speaker0", attributes={}, display_name="speaker0")
        g.add_node("acme corp", attributes={}, display_name="Acme Corp")
        g.add_edge(
            "speaker0",
            "acme corp",
            predicate="works at",
            relation_type="factual",
            speaker_id="speaker0",
            **{_IK_KEY_ATTR: "graph1"},
        )

        registry = KeyRegistry()
        registry.add("graph1")
        wt = WorkingTier(
            tier="episodic",
            adapter_name="episodic",
            pre_sha="",
            scratch_dir=tmp_path / "scratch_ep",
            registry=registry,
            rows={
                "graph1": {
                    "speaker_id": "speaker0",
                    "relation_type": "factual",
                    "reinforcement_count": 1,
                }
            },
            entries={},  # No working entry for the active key "graph1".
            rebuilt=True,
        )

        with pytest.raises(BookkeepingInvariantViolation):
            loop._build_working_keyed_walk({"episodic": wt}, exclude_keys=set())


class TestBuildWorkingKeyedWalkDerivesRebuiltSetFromMembers:
    """The rebuilt set :meth:`ConsolidationLoop._build_working_keyed_walk`
    builds against comes from each *working* member's own
    :attr:`WorkingTier.rebuilt` flag -- there is no separate tier-name list
    threaded into the call."""

    def test_only_the_rebuilt_member_appears_in_the_returned_dict(self, tmp_path):
        loop = _make_loop(tmp_path)

        rebuilt_wt = WorkingTier(
            tier="episodic",
            adapter_name="episodic",
            pre_sha="",
            scratch_dir=tmp_path / "scratch_ep",
            registry=KeyRegistry(),
            rows={},
            entries={},
            rebuilt=True,
        )
        candidate_wt = WorkingTier(
            tier="semantic",
            adapter_name="semantic",
            pre_sha="",
            scratch_dir=tmp_path / "scratch_sem",
            registry=KeyRegistry(),
            rows={},
            entries={},
            rebuilt=False,
        )
        working = {"semantic": candidate_wt, "episodic": rebuilt_wt}

        tier_keyed = loop._build_working_keyed_walk(working, exclude_keys=set())

        assert set(tier_keyed) == {"episodic"}


class TestClassifyPartialBuild:
    def _increment(self, tmp_path, *, pre_sha: str):
        from paramem.memory.increment import TierIncrement

        registry = KeyRegistry()
        registry_bytes = registry.save_bytes()
        return TierIncrement(
            tier="episodic",
            adapter_name="episodic",
            registry=registry,
            registry_bytes=registry_bytes,
            rows_bytes=b'{"tier_cycle": 0, "keys": {}}',
            entries={},
            bookkeeping={},
            keyed=[],
            rebuilt=False,
            pre_sha=pre_sha,
        )

    def test_not_built_when_live_matches_pre_sha(self, tmp_path):
        # No live registry -> tier_registry_sha256 returns "".
        inc = self._increment(tmp_path, pre_sha="")
        assert classify_partial_build(increment=inc, output_dir=tmp_path) == "not_built"

    def test_torn_own_write_when_live_matches_the_increments_own_payload(self, tmp_path):
        registry = KeyRegistry()
        registry_bytes = registry.save_bytes()
        registry.save_from_bytes(
            registry_bytes, tmp_path / "episodic" / "indexed_key_registry.json"
        )

        from paramem.memory.increment import TierIncrement

        inc = TierIncrement(
            tier="episodic",
            adapter_name="episodic",
            registry=registry,
            registry_bytes=registry_bytes,
            rows_bytes=b'{"tier_cycle": 0, "keys": {}}',
            entries={},
            bookkeeping={},
            keyed=[],
            rebuilt=False,
            pre_sha="some-stale-digest",
        )
        assert classify_partial_build(increment=inc, output_dir=tmp_path) == "torn_own_write"

    def test_foreign_when_live_matches_neither(self, tmp_path):
        stranger = KeyRegistry()
        stranger.add("someone_elses_key")
        stranger.save(tmp_path / "episodic" / "indexed_key_registry.json")

        inc = self._increment(tmp_path, pre_sha="not-the-live-digest")
        assert classify_partial_build(increment=inc, output_dir=tmp_path) == "foreign"


class TestStageEventDebugArtifactHooks:
    """``stage_event`` must re-emit the removal-ledger and fold-assignment
    debug artifacts it lost when the old fold driver was deleted -- the
    design keeps richer diagnostics on the ``debug: true`` path."""

    def test_on_fold_assignments_and_on_removal_ledger_fire_with_this_events_data(
        self, tmp_path, monkeypatch
    ):
        import paramem.training.consolidation as consolidation_mod

        loop = _make_loop(tmp_path)
        loop.store.registry("episodic").add("graph1")
        loop.store.set_bookkeeping(
            "graph1",
            speaker_id="speaker0",
            relation_type="factual",
            reinforcement_count=1,
            last_reinforced_cycle=0,
            last_seen="2026-01-01T00:00:00Z",
            first_seen="2026-01-01T00:00:00Z",
            promoted=False,
        )
        loop.store.put(
            "episodic",
            "graph1",
            {"key": "graph1", "subject": "alex", "predicate": "lives in", "object": "berlin"},
            register=False,
        )

        assignments_calls = []
        ledger_calls = []
        monkeypatch.setattr(
            consolidation_mod,
            "on_fold_assignments",
            lambda tier_keyed: assignments_calls.append(tier_keyed),
        )
        monkeypatch.setattr(
            consolidation_mod, "on_removal_ledger", lambda ledger: ledger_calls.append(ledger)
        )

        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="stampI",
            primary_tiers={"episodic": "episodic"},
            episodic_rels=[_rel("bob", "works at", "acme")],
        )
        assert result is not None

        assert len(assignments_calls) == 1, "on_fold_assignments must fire exactly once"
        assert "episodic" in assignments_calls[0]

        assert len(ledger_calls) == 1, "on_removal_ledger must fire exactly once"
        # This event's own merger.removal_ledger, whatever it ended up
        # holding -- the wiring under test is that the CURRENT ledger
        # object reaches the hook, not any specific removal content.
        assert ledger_calls[0] == dict(loop.merger.removal_ledger)


class TestDeriveKeyCounters:
    """Unit tests of ``ConsolidationLoop._derive_key_counters`` -- the mint-
    index floor every fresh/resumed loop re-derives from whatever the
    injected store already holds, seeded at ``DONOR_KEY_FLOOR`` (the reserved
    donor key band) rather than 1, and never lowered."""

    def test_empty_store_floors_both_counters_at_the_donor_reserved_band(self, tmp_path):
        from paramem.training.donor import DONOR_KEY_FLOOR

        loop = _make_loop(tmp_path)
        loop._indexed_next_index = 1  # _make_loop's own test shortcut
        loop._procedural_next_index = 1

        loop._derive_key_counters()

        assert loop._indexed_next_index == DONOR_KEY_FLOOR
        assert loop._procedural_next_index == DONOR_KEY_FLOOR

    def test_a_donor_seeded_high_water_key_raises_the_floor_past_it(self, tmp_path):
        """A tier seeded from a donor checkpoint (or replayed from an
        earlier install) already owns keys up to some high-water index --
        the counter must resume one past it, never re-mint from the floor
        and collide."""
        from paramem.training.donor import DONOR_KEY_FLOOR

        loop = _make_loop(tmp_path)
        loop.store.registry("episodic").add(f"graph{DONOR_KEY_FLOOR + 49}")

        loop._derive_key_counters()

        assert loop._indexed_next_index == DONOR_KEY_FLOOR + 50
        assert loop._procedural_next_index == DONOR_KEY_FLOOR

    def test_procedural_prefixed_keys_advance_the_procedural_counter_only(self, tmp_path):
        from paramem.training.donor import DONOR_KEY_FLOOR

        loop = _make_loop(tmp_path, procedural=True)
        loop.store.registry("procedural").add(f"proc{DONOR_KEY_FLOOR + 4}")

        loop._derive_key_counters()

        assert loop._procedural_next_index == DONOR_KEY_FLOOR + 5
        assert loop._indexed_next_index == DONOR_KEY_FLOOR

    def test_fresh_mint_after_donor_seeding_starts_one_past_the_high_water_mark(self, tmp_path):
        """Integration: the re-derived floor actually governs the next
        keyless mint through ``stage_event`` -- a donor-seeded tier's
        genuinely new fact must not collide with the donor's own reserved
        key range."""
        from paramem.training.donor import DONOR_KEY_FLOOR

        loop = _make_loop(tmp_path)
        donor_high_water_key = f"graph{DONOR_KEY_FLOOR + 5}"
        loop.store.registry("episodic").add(donor_high_water_key)
        loop.store.set_bookkeeping(
            donor_high_water_key,
            speaker_id="speaker0",
            relation_type="factual",
            reinforcement_count=1,
            last_reinforced_cycle=0,
            last_seen="2026-01-01T00:00:00Z",
            first_seen="2026-01-01T00:00:00Z",
            promoted=False,
        )
        loop.store.put(
            "episodic",
            donor_high_water_key,
            {
                "key": donor_high_water_key,
                "subject": "dana",
                "predicate": "lives in",
                "object": "oslo",
            },
            register=False,
        )
        loop._derive_key_counters()

        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="stampX",
            primary_tiers={"episodic": "episodic"},
            episodic_rels=[_rel("alex", "lives_in", "berlin")],
        )
        assert result is not None

        keyed = json.loads((_shadow_dir(loop, "interim", "episodic") / "keyed.json").read_text())
        minted = [row for row in keyed if row["key"] != donor_high_water_key]
        assert len(minted) == 1
        assert minted[0]["key"] == f"graph{DONOR_KEY_FLOOR + 6}"


class TestInterimMintCarriesTheRelationsOwnSpeakerId:
    """A keyless mint into an interim-shaped primary tier must carry the
    SOURCE relation's own ``speaker_id`` through to both the shadow keyed
    row and the working tier's bookkeeping row -- not the loop's default or
    an empty string."""

    def test_non_default_speaker_id_survives_the_mint_into_keyed_row_and_bookkeeping(
        self, tmp_path
    ):
        loop = _make_loop(tmp_path)
        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="20260101T0000",
            primary_tiers={"episodic_interim_20260101T0000": "episodic_interim_20260101T0000"},
            episodic_rels=[_rel("dana", "lives_in", "oslo", speaker_id="speaker7")],
        )
        assert result is not None

        shadow = _shadow_dir(loop, "interim", "episodic_interim_20260101T0000")
        keyed = json.loads((shadow / "keyed.json").read_text())
        assert len(keyed) == 1
        assert keyed[0]["speaker_id"] == "speaker7"

        rows = json.loads((shadow / "key_metadata.json").read_text())
        minted_key = keyed[0]["key"]
        assert rows["keys"][minted_key]["speaker_id"] == "speaker7"


class TestStageEventEnrichmentAndNormalizationGating:
    """``stage_event`` passes its ``normalize``/``enrich`` flags straight
    through to ``GraphTierRefiner.refine`` -- pure pass-through, no gate of
    its own -- and records a VRAM-driven enrichment degrade as an operator
    incident via ``_record_enrichment_incident`` immediately afterward."""

    def _stage_with_refine_spy(self, loop, monkeypatch, *, normalize: bool, enrich: bool):
        import paramem.training.graph_tier as graph_tier_mod

        calls = []
        real_refine = graph_tier_mod.GraphTierRefiner.refine

        def _spy_refine(self, *, normalize=False, enrich=False):
            calls.append({"normalize": normalize, "enrich": enrich})
            return real_refine(self, normalize=normalize, enrich=enrich)

        monkeypatch.setattr(graph_tier_mod.GraphTierRefiner, "refine", _spy_refine)

        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="full",
            venue="weights",
            stamp="stampF",
            primary_tiers={"episodic": "episodic"},
            episodic_rels=[_rel("alex", "lives_in", "berlin")],
            normalize=normalize,
            enrich=enrich,
        )
        return result, calls

    def test_default_flags_reach_the_refiner_as_false_false(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path)
        result, calls = self._stage_with_refine_spy(
            loop, monkeypatch, normalize=False, enrich=False
        )
        assert result is not None
        assert calls == [{"normalize": False, "enrich": False}]

    def test_full_fold_flags_reach_the_refiner_unmodified(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path)
        result, calls = self._stage_with_refine_spy(loop, monkeypatch, normalize=True, enrich=True)
        assert result is not None
        assert calls == [{"normalize": True, "enrich": True}]

    def test_vram_aborted_enrichment_records_an_operator_incident(self, tmp_path, monkeypatch):
        """A VRAM-driven enrichment degrade (the chunk loop stopped early on
        VramExhausted but kept whatever it already merged) is surfaced as a
        warning-severity ``enrichment_degraded`` incident by
        ``_record_enrichment_incident``, called from inside ``stage_event``
        right after the refiner returns."""
        from paramem.server.incidents import read_incidents
        from paramem.training import graph_tier as graph_tier_mod

        loop = _make_loop(tmp_path)
        loop._incidents_state_dir = tmp_path / "incidents"

        def _fake_refine(self, *, normalize=False, enrich=False):
            return graph_tier_mod.RefineResult(
                normalization=None,
                enrichment={"aborted_reason": "vram", "chunks": 2},
                adopt_reinforcements={},
            )

        monkeypatch.setattr(graph_tier_mod.GraphTierRefiner, "refine", _fake_refine)

        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="full",
            venue="weights",
            stamp="stampF",
            primary_tiers={"episodic": "episodic"},
            episodic_rels=[_rel("alex", "lives_in", "berlin")],
            enrich=True,
        )
        assert result is not None

        incidents = read_incidents(loop._incidents_state_dir)
        matching = [i for i in incidents if i.type == "enrichment_degraded"]
        assert len(matching) == 1
        assert matching[0].severity == "warning"
        assert matching[0].id == "enrichment_degraded:graph_enrich_vram"

    def test_no_incidents_dir_is_a_safe_no_op(self, tmp_path, monkeypatch):
        """The default fixture loop carries no ``_incidents_state_dir`` --
        an enrichment-off/interim staging pass must not raise trying to record
        an incident nowhere."""
        loop = _make_loop(tmp_path)
        assert loop._incidents_state_dir is None

        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="stampI",
            primary_tiers={"episodic": "episodic"},
            episodic_rels=[_rel("alex", "lives_in", "berlin")],
        )
        assert result is not None  # does not raise


class TestRecordFoldTelemetry:
    """Direct unit tests of ``ConsolidationLoop._record_fold_telemetry``."""

    def test_oserror_from_the_ring_write_does_not_propagate(self, tmp_path, monkeypatch, caplog):
        """Diagnostics must never block the fold that writes them: an
        ``OSError`` from the telemetry ring write is caught and logged, not
        raised, so it can never fail the training event whose critical path
        it runs inside."""
        import paramem.server.fold_telemetry as fold_telemetry

        loop = _make_loop(tmp_path)
        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="stamp1",
            primary_tiers={"episodic": "episodic"},
            episodic_rels=[_rel("alex", "lives_in", "berlin")],
        )
        assert result is not None
        loop._telemetry_dir = tmp_path / "telemetry"

        def _raise_oserror(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(fold_telemetry, "record_fold_telemetry", _raise_oserror)

        with caplog.at_level(logging.WARNING):
            loop._record_fold_telemetry(
                ledger=result.ledger, kind="tier_train", record={"tier": "episodic"}
            )  # does not raise

        assert "tier_train" in caplog.text


class TestRecallWorkingTiersMarkerSeeding:
    """``_recall_working_tiers`` seeds a primary (rebuilt) tier active-only
    and a candidate (dedup-only) tier active-union-withheld -- the
    ``rebuilt`` field records which case applied, and bookkeeping rows
    follow the SEEDED registry in both directions (never the live one)."""

    def _seed_marked_tier(self, loop, tier, *, active_key, stale_key):
        loop.store.registry(tier).add(active_key)
        loop.store.registry(tier).add(stale_key)
        loop.store.registry(tier).stale(stale_key)
        for key in (active_key, stale_key):
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
        loop.store.put(
            tier,
            active_key,
            {"key": active_key, "subject": "alex", "predicate": "lives in", "object": "berlin"},
            register=False,
        )

    def test_a_pre_existing_marker_is_absent_from_a_rebuilt_tiers_working_registry(self, tmp_path):
        loop = _make_loop(tmp_path)
        self._seed_marked_tier(loop, "episodic", active_key="active1", stale_key="stale1")

        working = loop._recall_working_tiers(
            {"episodic": "episodic"}, {}, _recalled_entries_from_store(loop)
        )
        wt = working["episodic"]

        assert wt.rebuilt is True
        assert wt.registry.list_known() == ["active1"]
        # Rows follow the SEEDED (active-only) registry -- the withheld
        # id's row is not carried into a rebuilt tier's working copy.
        assert set(wt.rows) == {"active1"}
        # The live store's marker is untouched by taking the working copy.
        assert loop.store.registry("episodic").list_stale() == ["stale1"]

    def test_a_pre_existing_marker_survives_into_a_candidate_tiers_working_registry(self, tmp_path):
        loop = _make_loop(tmp_path)
        self._seed_marked_tier(loop, "semantic", active_key="active2", stale_key="stale2")

        working = loop._recall_working_tiers(
            {"episodic": "episodic"}, {"semantic": "semantic"}, _recalled_entries_from_store(loop)
        )
        wt = working["semantic"]

        assert wt.rebuilt is False
        assert wt.registry.list_known() == ["active2", "stale2"]
        # Rows follow the SEEDED (active-union-withheld) registry -- a
        # candidate tier's marker keeps its row through the recall.
        assert set(wt.rows) == {"active2", "stale2"}


class TestApplyWorkingFateDecisionsDirect:
    """Direct unit tests of ``ConsolidationLoop._apply_working_fate_decisions``
    -- the fate of a same-event removal follows the owning working tier's
    own ``rebuilt`` field alone, never ``survivor_key``."""

    def _wt(self, tmp_path, tier, *, rebuilt, key):
        registry = KeyRegistry()
        registry.add(key)
        return WorkingTier(
            tier=tier,
            adapter_name=tier,
            pre_sha="",
            scratch_dir=tmp_path / f"scratch_{tier}",
            registry=registry,
            rows={
                key: {
                    "speaker_id": "speaker0",
                    "relation_type": "factual",
                    "reinforcement_count": 1,
                }
            },
            entries={key: {"key": key, "subject": "a", "predicate": "b", "object": "c"}},
            rebuilt=rebuilt,
        )

    def test_a_rebuilt_owners_removal_with_a_survivor_is_deleted_outright(self, tmp_path):
        loop = _make_loop(tmp_path)
        wt = self._wt(tmp_path, "episodic", rebuilt=True, key="graph1")
        working = {"episodic": wt}
        loop.merger.record_removal("graph1", reason="dedup", survivor_key="graph_survivor")

        loop._apply_working_fate_decisions(working)

        assert "graph1" not in wt.registry.list_known()
        assert "graph1" not in wt.rows
        assert "graph1" not in wt.entries
        assert wt.dirty is True

    def test_a_non_rebuilt_owners_removal_with_no_survivor_is_withheld_not_erased(self, tmp_path):
        loop = _make_loop(tmp_path)
        wt = self._wt(tmp_path, "semantic", rebuilt=False, key="graph2")
        working = {"semantic": wt}
        loop.merger.record_removal("graph2", reason="unkeyable_no_predicate")

        loop._apply_working_fate_decisions(working)

        assert wt.registry.list_active() == []
        assert wt.registry.list_stale() == ["graph2"]
        # The row survives the marker -- rows follow the seeded registry,
        # and a withheld id is still a known key.
        assert "graph2" in wt.rows
        assert wt.dirty is True


class TestParityGateActiveKeyWithNoRow:
    def test_the_parity_gate_raises_when_an_active_key_has_no_row(self, tmp_path):
        from paramem.memory.increment import TierIncrement
        from paramem.memory.store import BookkeepingInvariantViolation

        loop = _make_loop(tmp_path)
        registry = KeyRegistry()
        registry.add("graph1")
        increment = TierIncrement(
            tier="episodic",
            adapter_name="episodic",
            registry=registry,
            registry_bytes=registry.save_bytes(),
            rows_bytes=b"{}",
            entries={"graph1": {"key": "graph1", "subject": "a", "predicate": "b", "object": "c"}},
            bookkeeping={},  # no row for the active key
            keyed=[{"key": "graph1"}],
            rebuilt=True,
            pre_sha="",
        )

        with pytest.raises(BookkeepingInvariantViolation):
            loop._assert_increment_registry_bookkeeping_parity(increment)


class TestParityGateWithheldKeyWithNoRow:
    """The same pre-write gate, exercised on a rows-only (non-rebuilt) member
    over a withheld id rather than an active one -- ``list_known()`` covers
    active ∪ withheld, so a withheld id with no bookkeeping row must raise
    exactly like an active one, and a withheld id that DOES carry its row
    must pass even though the member has no materialized entry for it."""

    def _withheld_registry(self) -> KeyRegistry:
        registry = KeyRegistry()
        registry.add("graph2")
        registry.stale("graph2")
        return registry

    def test_the_parity_gate_raises_when_a_withheld_key_has_no_row(self, tmp_path):
        from paramem.memory.increment import TierIncrement
        from paramem.memory.store import BookkeepingInvariantViolation

        loop = _make_loop(tmp_path)
        registry = self._withheld_registry()
        increment = TierIncrement(
            tier="semantic",
            adapter_name="semantic",
            registry=registry,
            registry_bytes=registry.save_bytes(),
            rows_bytes=b"{}",
            entries={},
            bookkeeping={},  # no row for the withheld key
            keyed=[],
            rebuilt=False,
            pre_sha="",
        )

        with pytest.raises(BookkeepingInvariantViolation):
            loop._assert_increment_registry_bookkeeping_parity(increment)

    def test_the_parity_gate_passes_when_a_withheld_key_has_its_row(self, tmp_path):
        from paramem.memory.increment import TierIncrement

        loop = _make_loop(tmp_path)
        registry = self._withheld_registry()
        increment = TierIncrement(
            tier="semantic",
            adapter_name="semantic",
            registry=registry,
            registry_bytes=registry.save_bytes(),
            rows_bytes=b"{}",
            entries={},  # rows-only member: no materialized entry expected
            bookkeeping={"graph2": {"speaker_id": "speaker0", "reinforcement_count": 1}},
            keyed=[],
            rebuilt=False,
            pre_sha="",
        )

        loop._assert_increment_registry_bookkeeping_parity(increment)


class TestWriteShadowTierKeyedRebuiltInvariant:
    """``_write_shadow_tier`` checks *keyed*/``working_tier.rebuilt``
    agreement structurally rather than trusting the two ``stage_event``
    write loops to keep them positionally aligned."""

    def _wt(self, tmp_path, *, rebuilt: bool) -> WorkingTier:
        registry = KeyRegistry()
        registry.add("graph1")
        return WorkingTier(
            tier="episodic",
            adapter_name="episodic",
            pre_sha="",
            scratch_dir=tmp_path / "scratch_episodic",
            registry=registry,
            rows={"graph1": {"speaker_id": "speaker0", "relation_type": "factual"}},
            entries={"graph1": {"key": "graph1", "subject": "a", "predicate": "b", "object": "c"}},
            rebuilt=rebuilt,
        )

    def test_a_list_against_a_non_rebuilt_tier_raises(self, tmp_path):
        loop = _make_loop(tmp_path)
        wt = self._wt(tmp_path, rebuilt=False)

        with pytest.raises(RuntimeError):
            loop._write_shadow_tier(
                shadow_root=tmp_path / "shadow", tier="episodic", working_tier=wt, keyed=[]
            )

    def test_none_against_a_rebuilt_tier_raises(self, tmp_path):
        loop = _make_loop(tmp_path)
        wt = self._wt(tmp_path, rebuilt=True)

        with pytest.raises(RuntimeError):
            loop._write_shadow_tier(
                shadow_root=tmp_path / "shadow", tier="episodic", working_tier=wt, keyed=None
            )

    def test_a_list_against_a_rebuilt_tier_writes_without_raising(self, tmp_path):
        loop = _make_loop(tmp_path)
        wt = self._wt(tmp_path, rebuilt=True)

        written = loop._write_shadow_tier(
            shadow_root=tmp_path / "shadow", tier="episodic", working_tier=wt, keyed=[]
        )

        assert written  # rows path + keyed path + registry path all written

    def test_none_against_a_non_rebuilt_tier_writes_without_raising(self, tmp_path):
        loop = _make_loop(tmp_path)
        wt = self._wt(tmp_path, rebuilt=False)

        written = loop._write_shadow_tier(
            shadow_root=tmp_path / "shadow", tier="episodic", working_tier=wt, keyed=None
        )

        assert written  # rows path + registry path written, no keyed path


class TestFateDecisionsThroughStageEvent:
    """The same-event fate rule (:meth:`ConsolidationLoop._apply_working_fate_decisions`)
    driven through the real ``stage_event`` pipeline -- the retirement
    reaches the ledger via a genuine merge outcome (a real ``dedup``
    collapse, a real ``unkeyable_no_predicate`` skip), not a hand-built
    ``removal_ledger`` entry."""

    def test_a_dedup_casualty_in_a_rebuilt_tier_leaves_no_record_row_or_entry(self, tmp_path):
        """Two active keys reconstructed from the SAME rebuilt (primary)
        tier, both carrying the identical (subject, predicate, object): the
        second one merged collapses onto the first's edge (a real
        keyed-onto-keyed dedup), retiring its own key WITH a survivor. The
        owning tier is rebuilt this event, so the casualty is deleted
        outright -- gone from the registry, the rows and the keyed list
        that becomes this tier's training set and its ``n_keys`` count."""
        loop = _make_loop(tmp_path)
        for key in ("keyA", "keyB"):
            loop.store.registry("episodic").add(key)
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
            loop.store.put(
                "episodic",
                key,
                {"key": key, "subject": "alex", "predicate": "lives in", "object": "berlin"},
                register=False,
            )

        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="s_dedup",
            primary_tiers={"episodic": "episodic"},
        )

        assert result is not None
        assert loop.merger.removal_ledger["keyB"]["reason"] == "dedup"
        assert loop.merger.removal_ledger["keyB"]["survivor_key"] == "keyA"

        shadow = _shadow_dir(loop, "interim", "episodic")
        registry = KeyRegistry.load(shadow / "indexed_key_registry.json")
        assert registry.list_known() == ["keyA"]  # keyB gone entirely, not just inactive

        rows = json.loads((shadow / "key_metadata.json").read_text())["keys"]
        assert "keyB" not in rows

        keyed = json.loads((shadow / "keyed.json").read_text())
        keyed_keys = {row["key"] for row in keyed}
        assert keyed_keys == {"keyA"}, "the training-set/n_keys count must exclude the casualty"

    def test_a_removal_with_no_survivor_on_a_tier_not_rebuilt_this_event_is_withheld_not_erased(
        self, tmp_path
    ):
        """A candidate (dedup-only) tier's own active key carries an empty
        predicate -- ``unkeyable_no_predicate``, no survivor. This event
        does not rebuild that tier, so the casualty is withheld behind a
        marker rather than erased: known but inactive, its row intact."""
        loop = _make_loop(tmp_path)
        loop.store.registry("semantic").add("ghost")
        loop.store.set_bookkeeping(
            "ghost",
            speaker_id="speaker0",
            relation_type="factual",
            reinforcement_count=1,
            last_reinforced_cycle=0,
            last_seen="2026-01-01T00:00:00Z",
            first_seen="2026-01-01T00:00:00Z",
            promoted=False,
        )
        loop.store.put(
            "semantic",
            "ghost",
            {"key": "ghost", "subject": "alex", "predicate": "", "object": "berlin"},
            register=False,
        )

        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="s_withhold",
            primary_tiers={
                "episodic_interim_20260101T0000": "episodic_interim_20260101T0000",
            },
            candidate_tiers={"semantic": "semantic"},
        )

        assert result is not None
        assert loop.merger.removal_ledger["ghost"]["reason"] == "unkeyable_no_predicate"
        assert "survivor_key" not in loop.merger.removal_ledger["ghost"]
        assert "semantic" in result.built_tiers

        shadow = _shadow_dir(loop, "interim", "semantic")
        registry = KeyRegistry.load(shadow / "indexed_key_registry.json")
        assert registry.list_active() == []
        assert registry.list_known() == ["ghost"]  # withheld, not erased
        assert not (shadow / "keyed.json").exists()

        rows = json.loads((shadow / "key_metadata.json").read_text())["keys"]
        assert "ghost" in rows  # the bookkeeping row survives the marker


class TestATierOutsideTheFoldUniverse:
    def test_a_tier_outside_the_fold_universe_keeps_its_markers(self, tmp_path):
        """A tier this event names neither as primary nor as candidate is
        outside its working universe entirely -- ``_recall_working_tiers``
        never reads it, so its markers are untouched by construction."""
        loop = _make_loop(tmp_path, procedural=True)
        loop.store.registry("procedural").add("kept_pref")
        loop.store.registry("procedural").add("marked_pref")
        loop.store.registry("procedural").stale("marked_pref")
        for key in ("kept_pref", "marked_pref"):
            loop.store.set_bookkeeping(
                key,
                speaker_id="speaker0",
                relation_type="preference",
                reinforcement_count=1,
                last_reinforced_cycle=0,
                last_seen="2026-01-01T00:00:00Z",
                first_seen="2026-01-01T00:00:00Z",
                promoted=False,
            )
        loop.store.put(
            "procedural",
            "kept_pref",
            {"key": "kept_pref", "subject": "alex", "predicate": "prefers", "object": "tea"},
            register=False,
        )

        result = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="s_outside",
            primary_tiers={"episodic": "episodic"},
            episodic_rels=[_rel("alex", "lives_in", "berlin")],
        )
        assert result is not None
        assert "procedural" not in result.built_tiers

        reg = loop.store.registry("procedural")
        assert reg.list_active() == ["kept_pref"]
        assert reg.list_stale() == ["marked_pref"]
