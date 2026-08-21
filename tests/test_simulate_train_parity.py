"""Parity: the train and simulate venues converge on identical mode-independent
consolidation state.

``ConsolidationLoop.consolidate`` forks exactly once on ``mode``
(paramem/training/consolidation.py:2646-2687): ``mode="train"`` reconstructs
this event's active keys from adapter weights and persists retrained weights
(``source="weights"``); ``mode="simulate"`` reconstructs from disk and
persists a per-tier ``graph.json`` (``source="disk"``).  Both venues then run
through the identical shared spine — :meth:`~ConsolidationLoop.stage_event`
then :meth:`~ConsolidationLoop.run_build_and_publish` — parameterized by the
frozen :class:`~paramem.training.consolidation.FoldScope`.

These tests seed two identical tmp trees (same tier keys, same facts, same
bookkeeping) and drive ``loop.consolidate(mode=...)`` on each, both with
``pending=None`` — ``consolidate(mode="simulate")`` raises when *pending* is
not ``None`` (no weight venue to train pending sessions into), so venue
parity is defined over the shared reconcile-shaped spine only.  Assert
convergence of everything mode-independent: per-key store cache contents,
SimHash registry fingerprints, per-tier ``indexed_key_registry`` rows,
promotion/bookkeeping state, and the venue-labelled outcome classification
(:func:`~paramem.training.consolidation.interim_outcome_label`).
Bytewise-different artifacts (``adapter_model.safetensors`` vs
``graph.json``) are NOT compared.

Train mode is run with a faked weight probe and a stubbed
``_train_tier_adapter`` so the test does not require GPU; the assertion is on
the cycle's data-pipeline output, not on weight values.

The ``BackgroundTrainer``/``ConsolidationLoop`` release/close family that
used to live in this module now lives in ``tests/test_base_model_release.py``.

``DiskMemorySource.probe`` is covered in ``tests/test_memory_store.py``;
``commit_tier_slot`` (happy path, crash-cleanup, prune interplay) is covered
in ``tests/adapters/test_slot.py``, not here.
"""

from __future__ import annotations

import pytest

from paramem.memory.entry import entry_simhash
from paramem.memory.interim_adapter import adapter_slot_root_for_name
from paramem.training.consolidation import interim_outcome_label
from paramem.training.key_registry import KeyRegistry
from tests._fold_fixtures import _make_loop, _rel, _wire_fakes, _write_graph

# ---------------------------------------------------------------------------
# Shared fact fixture — the ONE definition both venues seed from, so a drift
# between the two seeded trees can never masquerade as a venue difference.
# ---------------------------------------------------------------------------

_FACTS: dict[str, dict[str, str]] = {
    "graph1": {"subject": "alex", "predicate": "lives in", "object": "berlin"},
    "graph2": {"subject": "alex", "predicate": "works at", "object": "acme corp"},
    "graph3": {"subject": "bob", "predicate": "likes", "object": "coffee"},
}
_TIER_KEYS: dict[str, tuple[str, ...]] = {
    "episodic": ("graph1", "graph2"),
    "semantic": ("graph3",),
}
# graph2's reinforcement_count clears promotion_threshold=3 (_make_loop's
# ConsolidationConfig) so it moves episodic -> semantic in BOTH venues.
_REINFORCEMENT_COUNT: dict[str, int] = {"graph1": 1, "graph2": 5, "graph3": 1}


def _seed_identical_content(loop, *, write_disk: bool) -> None:
    """Seed *loop* with the shared fact set, identically for either venue.

    Registers each tier's :class:`KeyRegistry` (active keys + SimHash
    fingerprints) and bookkeeping row in the RAM store — the fold's input in
    BOTH venues.  When *write_disk* is set (the simulate venue), also writes
    the on-disk registry file and a bound ``graph.json`` slot so
    ``DiskMemorySource.probe`` — the simulate venue's own reconstruction —
    answers the identical content the train venue's faked weight probe
    answers.
    """
    for tier, keys in _TIER_KEYS.items():
        registry = KeyRegistry()
        quads = []
        for key in keys:
            fact = _FACTS[key]
            entry = {"key": key, **fact}
            registry.add(key)
            registry.set_simhash(key, entry_simhash(entry))
            quads.append({**entry, "speaker_id": "speaker0"})
        if write_disk:
            tier_root = adapter_slot_root_for_name(loop.output_dir, tier)
            registry.save(tier_root / "indexed_key_registry.json")
            _write_graph(tier_root, quads)
        loop.store.load_registry(tier, registry)
        for key in keys:
            loop.store.set_bookkeeping(
                key,
                speaker_id="speaker0",
                relation_type="factual",
                first_seen="2026-01-01T00:00:00Z",
                last_seen="2026-01-01T00:00:00Z",
                reinforcement_count=_REINFORCEMENT_COUNT[key],
                last_reinforced_cycle=0,
                promoted=False,
            )


def _fake_weight_probe(model, tokenizer, keys_by_adapter, **kwargs):
    """Stand-in for ``probe_keys_grouped_by_adapter`` — the train venue's
    weight-reconstruction primitive.  Answers every requested key from the
    same ``_FACTS`` table the simulate venue's ``graph.json`` is seeded
    from, so both venues reconstruct identical content without touching the
    GPU (mirrors ``tests/_serving_door.py::stub_live_door_probe``, the
    project's established fake for this exact primitive)."""
    results: dict[str, dict] = {}
    for keys in keys_by_adapter.values():
        for key in keys:
            fact = _FACTS[key]
            results[key] = {
                "key": key,
                **fact,
                "confidence": 1.0,
                "fact_text": f"{fact['subject']} {fact['predicate']} {fact['object']}",
                "raw_output": "",
            }
    return results


def _tier_state(loop, tier: str) -> dict:
    """Snapshot everything mode-independent this test pins for one tier."""
    registry = loop.store.registry(tier)
    active = sorted(registry.list_active())
    return {
        "active_keys": active,
        "simhash": {key: registry.simhash_for(key) for key in active},
        "entries": loop.store.entries_in_tier(tier),
        "bookkeeping": {key: loop.store.bookkeeping_for_key(key) for key in active},
    }


class TestConsolidateVenueParity:
    """``consolidate(mode="train")`` and ``consolidate(mode="simulate")``
    converge on identical post-fold store state for identical input."""

    def test_train_and_simulate_converge_on_mode_independent_state(self, tmp_path, monkeypatch):
        train_loop = _make_loop(tmp_path / "train", resident_tiers=["episodic", "semantic"])
        sim_loop = _make_loop(tmp_path / "simulate")

        _seed_identical_content(train_loop, write_disk=False)
        _seed_identical_content(sim_loop, write_disk=True)

        _wire_fakes(train_loop, monkeypatch)
        _wire_fakes(sim_loop, monkeypatch)
        monkeypatch.setattr("paramem.server.gpu_lock.gpu_lock_is_held", lambda: True)
        monkeypatch.setattr(
            "paramem.memory.probe.probe_keys_grouped_by_adapter", _fake_weight_probe
        )

        train_result = train_loop.consolidate(mode="train", pending=None)
        sim_result = sim_loop.consolidate(mode="simulate", pending=None)

        # Both events ran to completion, with nothing aborted, and rebuilt
        # the same tier set.
        assert train_result["aborted"] is False
        assert sim_result["aborted"] is False
        assert train_result["completed"] is True
        assert sim_result["completed"] is True
        assert sorted(train_result["tiers_rebuilt"]) == ["episodic", "semantic"]
        assert sorted(sim_result["tiers_rebuilt"]) == ["episodic", "semantic"]
        assert sorted(train_result["tier_bindings"]) == sorted(sim_result["tier_bindings"])

        # The venue-labelled outcome classification (consolidation.py:507-534)
        # names each venue's doneness identically, through the one classifier.
        assert (
            interim_outcome_label(
                {"aborted": train_result["aborted"], "all_live": train_result["completed"]},
                venue="weights",
            )
            == "trained"
        )
        assert (
            interim_outcome_label(
                {"aborted": sim_result["aborted"], "all_live": sim_result["completed"]},
                venue="disk",
            )
            == "simulated"
        )

        # graph2's reinforcement_count (5) clears promotion_threshold (3):
        # promoted episodic -> semantic in BOTH venues.
        for loop in (train_loop, sim_loop):
            episodic = _tier_state(loop, "episodic")
            semantic = _tier_state(loop, "semantic")
            assert episodic["active_keys"] == ["graph1"]
            assert semantic["active_keys"] == ["graph2", "graph3"]
            assert semantic["bookkeeping"]["graph2"]["promoted"] is True

        # Per-key store cache contents (subject/predicate/object) converge —
        # the train venue's faked weight probe and the simulate venue's disk
        # read reconstruct the identical triples.
        assert (
            _tier_state(train_loop, "episodic")["entries"]
            == _tier_state(sim_loop, "episodic")["entries"]
        )
        assert (
            _tier_state(train_loop, "semantic")["entries"]
            == _tier_state(sim_loop, "semantic")["entries"]
        )

        # SimHash registry fingerprints converge per key.
        assert (
            _tier_state(train_loop, "episodic")["simhash"]
            == _tier_state(sim_loop, "episodic")["simhash"]
        )
        assert (
            _tier_state(train_loop, "semantic")["simhash"]
            == _tier_state(sim_loop, "semantic")["simhash"]
        )

        # Promotion/bookkeeping state converges — including the row moved
        # whole by promotion (reinforcement_count, speaker_id, timestamps
        # carried, "promoted" flipped), not a fabricated stub.
        assert (
            _tier_state(train_loop, "episodic")["bookkeeping"]
            == _tier_state(sim_loop, "episodic")["bookkeeping"]
        )
        assert (
            _tier_state(train_loop, "semantic")["bookkeeping"]
            == _tier_state(sim_loop, "semantic")["bookkeeping"]
        )


class TestFreshMultiSpeakerIngestPublishesAttributedRows:
    """Full-arc integration: a fresh multi-speaker pending batch -- one
    factual fact asserted by speaker0, one scalar attribute about a third
    party (acme corp) asserted by speaker1 -- goes through a real
    (fake-model) train-venue fold end to end (stage -> build -> publish ->
    adopt_increments) and is written to disk. A FRESH MemoryStore's own
    boot hydration (load_registries_from_disk then
    load_bookkeeping_from_disk) then reads every minted row back without
    raising -- which is only possible if no row anywhere in the published
    tiers carries an empty speaker_id, since a single one would raise
    ValueError from bookkeeping_row inside set_bookkeeping mid-boot -- and
    the attribute key's row is attributed to the speaker who actually
    asserted it, never the other speaker present in the same batch.
    """

    def test_fresh_multi_speaker_ingest_publishes_attributed_rows_boot_loads_clean(
        self, tmp_path, monkeypatch
    ):
        from paramem.memory.store import MemoryStore
        from paramem.training.consolidation import PendingRelations

        loop = _make_loop(tmp_path, resident_tiers=["episodic", "semantic"])
        _wire_fakes(loop, monkeypatch)
        monkeypatch.setattr("paramem.server.gpu_lock.gpu_lock_is_held", lambda: True)

        pending = PendingRelations(
            episodic=[
                _rel("speaker0", "lives_in", "berlin", speaker_id="speaker0"),
                _rel(
                    "acme corp",
                    "has_founder",
                    "Jane Doe",
                    relation_type="attribute",
                    speaker_id="speaker1",
                    first_seen="2026-03-01T00:00:00Z",
                    last_seen="2026-03-01T00:00:00Z",
                ),
            ],
            procedural=[],
        )
        result = loop.consolidate(mode="train", pending=pending, session_ids=["s1", "s2"])
        assert result["aborted"] is False
        assert result["completed"] is True

        episodic_entries = loop.store.entries_in_tier("episodic")
        attr_key = next(
            key for key, entry in episodic_entries.items() if entry["predicate"] == "has founder"
        )
        lives_key = next(
            key for key, entry in episodic_entries.items() if entry["predicate"] == "lives in"
        )

        fresh_store = MemoryStore()
        fresh_store.load_registries_from_disk(loop.output_dir)
        boot = fresh_store.load_bookkeeping_from_disk(loop.output_dir)
        assert boot["orphaned"] == 0

        for tier in ("episodic", "semantic"):
            for key in fresh_store.registry(tier).list_active():
                row = fresh_store.bookkeeping_for_key(key)
                assert row is not None
                assert row["speaker_id"] != "", (
                    f"key {key!r} in tier {tier!r} loaded with an empty speaker_id"
                )

        assert fresh_store.bookkeeping_for_key(attr_key)["speaker_id"] == "speaker1"
        assert fresh_store.bookkeeping_for_key(attr_key)["first_seen"] == "2026-03-01T00:00:00Z"
        assert fresh_store.bookkeeping_for_key(lives_key)["speaker_id"] == "speaker0"


class TestConsolidateModeGuard:
    """``consolidate``'s one documented raise — pinned separately from the
    convergence pin above since it never reaches the shared spine at all."""

    def test_simulate_mode_rejects_pending(self, tmp_path):
        from paramem.training.consolidation import PendingRelations

        loop = _make_loop(tmp_path)

        with pytest.raises(ValueError, match="pending"):
            loop.consolidate(mode="simulate", pending=PendingRelations(episodic=[], procedural=[]))
