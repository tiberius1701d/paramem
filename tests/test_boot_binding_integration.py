"""Integration pins for the venue-uniform boot binding verification design.

Drives a real ``ConsolidationLoop.stage_event`` -> ``run_build_and_publish``
fold in the simulate venue -- no GPU, no real PEFT model, exactly like
``tests/test_publish_bundle_resume.py`` (whose ``tests._fold_fixtures``
collaborators this file shares) -- then reads the on-disk result back
through :func:`~paramem.adapters.registry_binding.verify_adapter_tree`, the
same fresh, model-free read a real boot performs.
"""

from __future__ import annotations

from paramem.adapters.registry_binding import VERIFIED, verify_adapter_tree
from tests._fold_fixtures import _make_loop, _recalled_entries_from_store, _rel, _wire_fakes


class TestFullFoldPublishesWritesAndBindsInTheSimulateVenue:
    def test_full_fold_publishes_writes_and_binds_in_the_simulate_venue(
        self, tmp_path, monkeypatch
    ) -> None:
        loop = _make_loop(tmp_path, procedural=True)
        _wire_fakes(loop, monkeypatch)

        staged = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="full",
            venue="disk",  # simulate venue -- a graph payload, never trains
            stamp="20260101T0000",
            primary_tiers={
                "episodic": "episodic",
                "semantic": "semantic",
                "procedural": "procedural",
            },
            episodic_rels=[_rel("alex", "lives_in", "berlin")],
            procedural_rels=[_rel("alex", "prefers", "acme radio", relation_type="preference")],
            session_ids=["s1"],
        )
        assert staged is not None

        summary = loop.run_build_and_publish(staged, router=None)

        assert summary["all_live"] is True
        assert summary["aborted"] is False
        assert set(summary["published_tiers"]) == {"episodic", "semantic", "procedural"}
        assert loop._train_tier_adapter.call_count == 0  # simulate venue never trains

        tier_bindings = summary["tier_bindings"]
        # episodic and procedural each minted real content and written a
        # real graph payload; semantic built to zero keys (no promotion
        # yet) and publishes rows/registry only -- still publishable, but
        # not VERIFIED (no slot to bind).
        for tier in ("episodic", "procedural"):
            binding = tier_bindings[tier]
            assert binding.status == VERIFIED
            assert binding.publishable
            assert binding.manifest.payload.kind == "simulate"
            assert (binding.slot / "meta.json").exists()
            assert (binding.slot / "graph.json").exists()
        assert tier_bindings["semantic"].publishable


class TestBootAfterASimulateFoldPublishesEveryTier:
    def test_boot_after_a_simulate_fold_publishes_every_tier(self, tmp_path, monkeypatch) -> None:
        """After a full simulate fold, a FRESH boot-time read of the adapter
        tree (never the fold's own in-memory verdict) finds every main tier
        publishable -- the store would boot live, not quarantined."""
        loop = _make_loop(tmp_path, procedural=True)
        _wire_fakes(loop, monkeypatch)

        staged = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="full",
            venue="disk",
            stamp="20260101T0000",
            primary_tiers={
                "episodic": "episodic",
                "semantic": "semantic",
                "procedural": "procedural",
            },
            episodic_rels=[_rel("alex", "lives_in", "berlin")],
            procedural_rels=[_rel("alex", "prefers", "acme radio", relation_type="preference")],
            session_ids=["s1"],
        )
        assert staged is not None
        summary = loop.run_build_and_publish(staged, router=None)
        assert summary["all_live"] is True

        # Boot never trusts the fold's own tier_bindings -- it re-derives
        # fresh from disk.
        boot_bindings = verify_adapter_tree(loop.output_dir)

        for tier in ("episodic", "semantic", "procedural"):
            assert boot_bindings[tier].publishable, (
                tier,
                boot_bindings[tier].status,
                boot_bindings[tier].detail,
            )

        # No exception -- every tier publishes.
        from paramem.adapters.registry_binding import raise_tier_binding_unpublishable

        unpublishable = {t: b for t, b in boot_bindings.items() if not b.publishable}
        assert unpublishable == {}
        if unpublishable:  # pragma: no cover -- documents the guard, never reached here
            raise_tier_binding_unpublishable(boot_bindings)
