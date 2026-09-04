"""Crash/resume coverage for a MIXED go-live bundle: one written member and
one rows-only member landing together.

Drives ``ConsolidationLoop.run_build_and_publish`` (and, through it,
``paramem.training.go_live.publish_bundle``) via the real write/publish
machinery -- no GPU, no real PEFT model -- exactly like the written-slot
resume suite in ``tests/test_stage_ledger.py``, whose fixtures
(``tests._fold_fixtures``) this file shares.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from paramem.memory import persistence as persistence_module
from paramem.training import stage_ledger as sl
from tests._fold_fixtures import _make_loop, _recalled_entries_from_store, _rel, _wire_fakes


def _snapshot_tree(root: Path) -> dict[str, str]:
    """``{relative path: sha256}`` for every file under *root*.

    A pure read used to prove a refused call wrote zero bytes: comparing two
    snapshots catches both a changed file (digest differs) and a
    written/removed one (path set differs) in one assertion.
    """
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class TestResumeAfterAWritePublishesTheGraphPayloadAndRestampsNothing:
    """A mixed tree -- one written member (episodic, a genuinely new fact),
    one rows-only member (semantic, credited by a dedup collapse but never
    minting new content) -- crashed after episodic's write, before the
    bundle's go-live ever ran. Resume must publish episodic through the
    BIND arm (never retrain, never rewritten, never planned) while semantic
    publishes through the ordinary plan_restamp/restamp_tier_manifest path."""

    def test_resume_after_a_write_publishes_the_graph_payload_and_restamps_nothing(
        self, tmp_path, monkeypatch
    ):
        from paramem.memory.increment import build_tier_increment

        loop = _make_loop(tmp_path, resident_tiers=["episodic"])
        _wire_fakes(loop, monkeypatch)

        # Seed semantic with a real, already-BOUND on-disk slot -- a
        # healthy tier's steady state (registry loads its own pre_sha off
        # THIS bytes-on-disk read, never off the RAM store: see
        # ConsolidationLoop._recall_working_tiers). Without a bound slot the
        # tier would refuse KEYS_WITHOUT_SLOT the instant it turns dirty --
        # unreachable on a healthy store per plan_restamp's own docstring.
        from paramem.memory.increment import TierIncrement
        from paramem.memory.persistence import publish_tier_registry, write_tier_slot
        from paramem.training.key_registry import KeyRegistry

        seed_registry = KeyRegistry()
        seed_registry.add("graph_sem")
        seed_bytes = seed_registry.save_bytes()
        seed_increment = TierIncrement(
            tier="semantic",
            adapter_name="semantic",
            registry=seed_registry,
            registry_bytes=seed_bytes,
            rows_bytes=b'{"tier_cycle": 0, "keys": {}}',
            entries={},
            bookkeeping={},
            keyed=[
                {
                    "key": "graph_sem",
                    "subject": "alex",
                    "predicate": "lives in",
                    "object": "berlin",
                    "speaker_id": "speaker0",
                }
            ],
            rebuilt=True,
            pre_sha="",
        )
        seed_ctx = loop._build_write_context()
        seed_slot = write_tier_slot(
            ctx=seed_ctx, increment=seed_increment, stamp="20260101T0000", mode="simulate"
        )
        publish_tier_registry(increment=seed_increment, ctx=seed_ctx, written_slot=seed_slot)

        # Seed the RAM store to match the on-disk registry exactly (same
        # active key, no simhash -- so the working copy's serialized bytes
        # are byte-identical to what was just bound on disk).
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

        staged = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="disk",  # simulate venue -- a graph payload, never trains
            stamp="stamp1",
            primary_tiers={"episodic": "episodic"},
            candidate_tiers={"semantic": "semantic"},
            episodic_rels=[
                _rel("alex", "likes", "coffee"),  # genuinely new -- mints a key in episodic
                _rel("alex", "lives_in", "berlin"),  # collapses onto graph_sem in semantic
            ],
            session_ids=["s1"],
        )
        assert staged is not None
        assert "episodic" in staged.built_tiers
        assert "semantic" in staged.built_tiers
        state_dir = staged.state_dir

        episodic_shadow = sl.extraction_dir(state_dir, staged.event) / "shadow" / "episodic"
        semantic_shadow = sl.extraction_dir(state_dir, staged.event) / "shadow" / "semantic"
        # episodic carries real new content; semantic is rows-only by
        # construction -- dedup credited it, but it mints no keyed.json.
        assert (episodic_shadow / "keyed.json").exists()
        assert not (semantic_shadow / "keyed.json").exists()

        # Write the episodic tier directly -- exactly what
        # run_build_and_publish's own per-tier loop does for a not-yet-done
        # tier -- WITHOUT calling run_build_and_publish, so the bundle's
        # go-live never runs: the ledger is left in the crash-window shape
        # (a verifying tier_written entry for episodic; no tier_live entry
        # anywhere; semantic never touched at all).
        increment = build_tier_increment(
            tier="episodic",
            adapter_name=staged.ledger.tiers["episodic"]["adapter"],
            pre_sha=staged.ledger.tiers["episodic"]["pre_sha"],
            shadow_dir=episodic_shadow,
        )
        written_slot, _ = loop._write_built_tier(
            increment=increment, ledger=staged.ledger, state_dir=state_dir, mode="simulate"
        )
        assert written_slot is not None
        assert (written_slot / "graph.json").exists()
        assert loop._train_tier_adapter.call_count == 0  # simulate venue never trains

        pre_resume_ledger = sl.read_ledger(state_dir)
        written_entry = loop._latest_stage(pre_resume_ledger, "episodic", "tier_written")
        assert written_entry is not None
        assert sl.written_slot_path(written_entry) == written_slot
        assert loop._latest_stage(pre_resume_ledger, "episodic", "tier_live") is None
        assert loop._latest_stage(pre_resume_ledger, "semantic", "tier_written") is None
        assert loop._latest_stage(pre_resume_ledger, "semantic", "tier_live") is None

        # Spy on plan_restamp: proof the written member's resume never
        # consults the planner at all -- only the rows-only member does (once
        # in the preflight, once again inside restamp_tier_manifest's own
        # write -- the plan is a pure read and runs twice on this path
        # deliberately, per go_live.publish_bundle's own docstring).
        real_plan_restamp = persistence_module.plan_restamp
        calls: list[str] = []

        def _spy_plan_restamp(tier_root, **kwargs):
            calls.append(str(tier_root))
            return real_plan_restamp(tier_root, **kwargs)

        monkeypatch.setattr(persistence_module, "plan_restamp", _spy_plan_restamp)

        # Resume: a fresh call to run_build_and_publish, reading the ledger
        # fresh from disk -- never trusts `staged` across the simulated
        # crash.
        summary = loop.run_build_and_publish(staged, router=None)

        assert summary["all_live"] is True
        assert set(summary["published_tiers"]) == {"episodic", "semantic"}
        # The written member is never retrained on resume.
        assert loop._train_tier_adapter.call_count == 0

        # The planner ran only for semantic's tier root -- never episodic's:
        # the written member publishes through the bind arm, which never
        # calls plan_restamp.
        semantic_tier_root = str(loop.output_dir / "semantic")
        assert calls
        assert set(calls) == {semantic_tier_root}

        final_ledger = sl.read_ledger(state_dir)
        episodic_live = loop._latest_stage(final_ledger, "episodic", "tier_live")
        semantic_live = loop._latest_stage(final_ledger, "semantic", "tier_live")
        assert episodic_live is not None
        assert semantic_live is not None
        episodic_artifacts = {a["path"] for a in episodic_live["artifacts"]}
        # The published tier_live entry names the SAME written slot this test
        # recorded at write time -- the graph payload published unchanged,
        # never rewritten.
        assert str(written_slot / "meta.json") in episodic_artifacts


class TestPreflightRefusalLeavesTheWholeTreeUnchanged:
    """A mixed tree where the rows-only member (semantic) carries an active
    key with no on-disk bound slot at all -- the "unreachable on a healthy
    store" condition ``plan_restamp``'s own docstring names, deliberately
    constructed here by omitting the pre-bound-slot seed the sibling resume
    test performs. Drives the refusal through
    ``paramem.training.go_live.publish_bundle`` directly (its preflight,
    ``assert_publish_preconditions``, is the function's first statement) so
    the whole on-disk tree from immediately before the call is compared
    byte-for-byte against immediately after: this pins that the preflight
    precedes every durable write of ``publish_bundle``'s own sequence, even
    with one member (episodic) already genuinely written."""

    def test_preflight_refusal_writes_nothing_and_leaves_sessions_pending(
        self, tmp_path, monkeypatch
    ):
        from paramem.memory.increment import build_tier_increment
        from paramem.memory.persistence import KEYS_WITHOUT_SLOT, TierWriteRefused
        from paramem.training.go_live import publish_bundle

        loop = _make_loop(tmp_path, resident_tiers=["episodic"])
        _wire_fakes(loop, monkeypatch)

        # Seed semantic in RAM ONLY -- an active key with no on-disk bound
        # slot ever written for it (no write_tier_slot / publish_tier_registry
        # call). Contrast with the sibling resume test, which seeds a real
        # bound slot for exactly this reason.
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

        staged = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="disk",  # simulate venue -- a graph payload, never trains
            stamp="stamp1",
            primary_tiers={"episodic": "episodic"},
            candidate_tiers={"semantic": "semantic"},
            episodic_rels=[
                _rel("alex", "likes", "coffee"),  # genuinely new -- mints a key in episodic
                _rel("alex", "lives_in", "berlin"),  # collapses onto graph_sem in semantic
            ],
            session_ids=["s1"],
        )
        assert staged is not None
        assert "episodic" in staged.built_tiers
        assert "semantic" in staged.built_tiers
        state_dir = staged.state_dir

        episodic_shadow = sl.extraction_dir(state_dir, staged.event) / "shadow" / "episodic"
        semantic_shadow = sl.extraction_dir(state_dir, staged.event) / "shadow" / "semantic"
        assert (episodic_shadow / "keyed.json").exists()
        assert not (semantic_shadow / "keyed.json").exists()  # rows-only, as above

        # Write episodic for real (a genuine written member).
        episodic_increment = build_tier_increment(
            tier="episodic",
            adapter_name=staged.ledger.tiers["episodic"]["adapter"],
            pre_sha=staged.ledger.tiers["episodic"]["pre_sha"],
            shadow_dir=episodic_shadow,
        )
        written_slot, ledger = loop._write_built_tier(
            increment=episodic_increment, ledger=staged.ledger, state_dir=state_dir, mode="simulate"
        )
        assert written_slot is not None
        assert loop._train_tier_adapter.call_count == 0

        # Assemble semantic's increment straight off its own shadow tree --
        # exactly what run_build_and_publish's per-tier loop would build for
        # it -- WITHOUT writing it (a rows-only member writes no payload
        # either way; skipping the no-op call keeps this test's write
        # surface limited to episodic's genuine write above).
        semantic_increment = build_tier_increment(
            tier="semantic",
            adapter_name=staged.ledger.tiers["semantic"]["adapter"],
            pre_sha=staged.ledger.tiers["semantic"]["pre_sha"],
            shadow_dir=semantic_shadow,
        )

        pre_call_ledger = sl.read_ledger(state_dir)
        extraction_before = sl.extraction_entry(pre_call_ledger)
        assert extraction_before is not None
        assert extraction_before["sessions"] == ["s1"]

        bundle = loop._ordered_publish_bundle(
            {"episodic": episodic_increment, "semantic": semantic_increment}
        )
        ctx = loop._build_write_context(extra_tiers=[inc.tier for inc in bundle])
        written_slots = {"episodic": written_slot, "semantic": None}

        snapshot_before = _snapshot_tree(tmp_path)

        with pytest.raises(TierWriteRefused) as excinfo:
            publish_bundle(
                bundle,
                ctx=ctx,
                ledger=pre_call_ledger,
                written_slots=written_slots,
                router=None,
                absorbed_interim_tiers=(),
            )
        assert excinfo.value.refusals == {"semantic": KEYS_WITHOUT_SLOT}

        # (a) the refusal propagated (asserted above via pytest.raises).
        # (b) the whole tree -- adapter tree AND state dir, path set and
        # digests both -- is byte-for-byte unchanged, including the
        # stage-ledger file itself.
        snapshot_after = _snapshot_tree(tmp_path)
        assert snapshot_after == snapshot_before

        post_call_ledger = sl.read_ledger(state_dir)
        assert post_call_ledger is not None
        extraction_after = sl.extraction_entry(post_call_ledger)
        assert extraction_after == extraction_before
        # (c) the contributing session is still recorded exactly as staged --
        # retirement happens only at the caller's terminal on all_live, never
        # inside publish_bundle or its preflight.
        assert extraction_after["sessions"] == ["s1"]
        assert loop._latest_stage(post_call_ledger, "episodic", "tier_live") is None
        assert loop._latest_stage(post_call_ledger, "semantic", "tier_live") is None


class TestPostPublishReReadFindingNoLedgerRaises:
    """run_build_and_publish re-reads the ledger from disk immediately after
    a bundle publishes, to record the event's completion. A missing ledger at
    that point is a filesystem failure -- not "nothing pending" -- since the
    publish this call just ran has nothing left to record it against."""

    def test_ledger_vanishing_after_publish_raises_naming_the_state_dir(
        self, tmp_path, monkeypatch
    ):
        loop = _make_loop(tmp_path, resident_tiers=["episodic"])
        _wire_fakes(loop, monkeypatch)

        staged = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="stamp1",
            primary_tiers={"episodic": "episodic"},
            episodic_rels=[_rel("alex", "lives_in", "berlin")],
            session_ids=["s1"],
        )
        assert staged is not None
        state_dir = staged.state_dir

        # run_build_and_publish calls stage_ledger.read_ledger exactly
        # twice: once at entry (must behave normally so the build/publish
        # proceeds for real) and once right after the bundle publishes (the
        # call this test forces to answer "absent").
        real_read_ledger = sl.read_ledger
        calls = {"n": 0}

        def _read_ledger_absent_on_second_call(sd):
            calls["n"] += 1
            if calls["n"] == 1:
                return real_read_ledger(sd)
            return None

        monkeypatch.setattr(sl, "read_ledger", _read_ledger_absent_on_second_call)

        with pytest.raises(RuntimeError, match=str(state_dir)):
            loop.run_build_and_publish(staged, router=None)

        assert calls["n"] >= 2
