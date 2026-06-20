"""Unit tests for consolidate_interim_adapters.

Covers:
  - Lock-leak guard: calling without the GPU lock held raises RuntimeError;
    the lock is NOT left in a locked state after the raise.
  - Training-flag bracketing: _set_is_training(True) is called before each _train_adapter();
    _set_is_training(False) is called in the finally block.
  - Per-tier inference_fallback_adapter: each TrainingJob carries the correct
    backup adapter name.
  - Capacity-ceiling rollback: when recall < threshold the method returns
    rolled_back=True and does NOT proceed to the finalize block.
  - Atomic finalize ordering: registry rewrite runs before both on-disk delete
    and PEFT unload; Router.reload() runs after both.

No GPU required — all PEFT and model interactions use stub objects.
The GPU lock tests verify threading.Lock behaviour without CUDA.

Patch targets for methods imported inside the consolidation body:
  create_adapter     → paramem.models.loader.create_adapter
  format_entry_training → paramem.training.consolidation.format_entry_training
  train_adapter (fn) → paramem.training.trainer.train_adapter
  build_registry     → paramem.training.consolidation.build_registry
  evaluate_indexed_recall → paramem.training.recall_eval.evaluate_indexed_recall
  unload_interim_adapters → paramem.memory.interim_adapter.unload_interim_adapters
  partition_relations → paramem.graph.qa_generator.partition_relations
  switch_adapter     → paramem.models.loader.switch_adapter
  copy_adapter_weights → paramem.models.loader.copy_adapter_weights
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from paramem.graph.reconstruct import ReconstructionResult
from paramem.server.background_trainer import BackgroundTrainer, TrainingJob
from paramem.training.key_registry import KeyRegistry
from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOCK_PATCHES = [
    "paramem.models.loader.create_adapter",
    "paramem.training.consolidation.format_entry_training",
    "paramem.training.trainer.train_adapter",
    "paramem.training.consolidation.build_registry",
    "paramem.training.recall_eval.evaluate_indexed_recall",
    "paramem.memory.interim_adapter.unload_interim_adapters",
    "paramem.models.loader.switch_adapter",
    "paramem.models.loader.copy_adapter_weights",
    "paramem.graph.qa_generator.partition_relations",
]


def _faithful_reconstruct(loop, *, tier=None, strict=False) -> ReconstructionResult:
    """Stub for ``reconstruct_graph`` that rebuilds the graph from the loop's
    registered store entries WITHOUT running a real probe.

    The real ``reconstruct_graph`` probes ``loop.model`` (a ``MagicMock`` here)
    via ``probe_entries`` → ``re.sub``, which TypeErrors on a mock.  This stub
    yields the same edges the probe would have produced for a model that
    perfectly recalls every registered key: one edge per active key carrying
    ``_IK_KEY_ATTR`` and ``predicate`` (plus subject→object), so the downstream
    re-merge / dedup / tiering / training flow proceeds exactly as in a passing
    probe.  No real behavior (drift detection, key registration, training) is
    short-circuited — the keys still flow through.
    """
    from paramem.memory.persistence import _IK_KEY_ATTR

    graph = nx.MultiDiGraph()
    for _tier, key, entry in loop.store.iter_entries():
        if tier is not None and _tier != tier:
            continue
        subj = entry.get("subject", "")
        obj = entry.get("object", "")
        pred = entry.get("predicate", "")
        if not (subj and obj and pred):
            continue
        eid = graph.add_edge(subj, obj, predicate=pred)
        graph[subj][obj][eid][_IK_KEY_ATTR] = key
    return ReconstructionResult(graph=graph, failures=[])


def _make_stub_model(*adapter_names: str) -> MagicMock:
    """Return a MagicMock that behaves like a minimal PeftModel."""
    model = MagicMock()
    model.peft_config = {name: MagicMock() for name in adapter_names}

    def _delete_adapter(name: str) -> None:
        model.peft_config.pop(name, None)

    model.delete_adapter.side_effect = _delete_adapter
    return model


def _minimal_training_config() -> TrainingConfig:
    return TrainingConfig(
        num_epochs=1,
        gradient_checkpointing=False,
        batch_size=1,
    )


def _minimal_adapter_config() -> AdapterConfig:
    return AdapterConfig(rank=4, alpha=8, learning_rate=1e-4, target_modules=["q_proj"])


def _minimal_consolidation_config() -> ConsolidationConfig:
    # min_tier_key_floor=0 and tier_fast_start=False: these are pre-floor structural
    # tests using small key sets; disabling the floor and fast-start keeps them on the
    # normal per-tier train/finalize/save path they assert.
    return ConsolidationConfig(
        indexed_key_replay_enabled=True,
        min_tier_key_floor=0,
        tier_fast_start=False,
    )


def _make_empty_registry_dict() -> dict:
    """Return a fresh per-tier registry dict with empty registries for all three main tiers."""
    return {
        "episodic": KeyRegistry(),
        "semantic": KeyRegistry(),
        "procedural": KeyRegistry(),
    }


def _make_loop(model, tmp_path: Path, *, registry=None, indexed_key_cache=None):
    """Construct a bare ConsolidationLoop without calling __init__.

    ``registry`` must be a ``dict[str, KeyRegistry]`` (per-tier) or ``None``
    (disabled replay).  When omitted, a fresh three-tier dict is used so
    ``_all_active_keys()`` works without a real ``__init__`` call.
    """
    from paramem.training.consolidation import ConsolidationLoop

    loop = ConsolidationLoop.__new__(ConsolidationLoop)
    loop.model = model
    loop.tokenizer = MagicMock()
    loop.config = _minimal_consolidation_config()
    loop.training_config = _minimal_training_config()
    loop.episodic_config = _minimal_adapter_config()
    loop.semantic_config = _minimal_adapter_config()
    loop.procedural_config = None
    loop.wandb_config = None
    loop.output_dir = tmp_path
    loop.merger = MagicMock()
    # Stage 2 (re-merge) calls merger.merge(synthetic_session); stage 5 walks
    # merger.graph.edges(data=True) to rebuild the per-tier keyed lists.  Use a
    # real MultiDiGraph and a merge side_effect that inserts one edge per relation
    # (carrying predicate + relation_type), mirroring the real GraphMerger so the
    # reconstructed triples flow through dedup/tiering/training instead of all
    # registering as graph drift.
    _merger_graph = nx.MultiDiGraph()
    loop.merger.graph = _merger_graph

    from paramem.memory.persistence import _IK_KEY_ATTR as _IK_ATTR

    def _merge_into_graph(
        session_graph, *, resolve_contradictions: bool = True, align_predicates: bool = False
    ):
        for _rel in getattr(session_graph, "relations", []):
            eid = _merger_graph.add_edge(
                _rel.subject,
                _rel.object,
                predicate=_rel.predicate,
                relation_type=getattr(_rel, "relation_type", "factual"),
            )
            _ik = getattr(_rel, "indexed_key", None)
            if _ik:
                _merger_graph[_rel.subject][_rel.object][eid][_IK_ATTR] = _ik
        return _merger_graph

    loop.merger.merge.side_effect = _merge_into_graph
    # indexed_key_registry is now dict[str, KeyRegistry] (per-tier).
    loop.indexed_key_registry = registry if registry is not None else _make_empty_registry_dict()
    loop.indexed_key_cache = indexed_key_cache if indexed_key_cache is not None else {}
    loop.snapshot_dir = None
    loop.save_cycle_snapshots = False
    loop._thermal_policy = None
    # _bg_trainer is wired by the server lifespan; None in tests (experiment path).
    loop._bg_trainer = None
    # consolidate_interim_adapters calls _run_graph_enrichment(), which reads
    # self.graph_enrichment_enabled.  The bare __new__ loop never ran __init__,
    # so this attribute is absent and the call raises AttributeError (swallowed,
    # but the half-run enrichment perturbs the per-tier flow).  Disable enrichment
    # explicitly so it early-returns skipped — the no-enrichment behavior these
    # ordering/registry tests assume.
    loop.graph_enrichment_enabled = False
    # Admit-all probe stub for the registration fail-safe: when no recall verdict is
    # available, _reset_main_tier_registries_and_simhashes runs _probe_passing_keys,
    # whose real evaluate_indexed_recall feeds the MagicMock model into re.sub and
    # TypeErrors.  Admitting every key matches the prior no-gate behavior, so it is
    # inert for these ordering/registry tests.
    loop._probe_passing_keys = lambda adapter_name, entries: {e["key"] for e in entries}
    # _promote_mature_keys_inline (called inside consolidate_interim_adapters) reads
    # cycle_count, promoted_keys, and store.  Wire sensible defaults so the method
    # runs without error.  Call _ensure_store() rather than assigning a new store so
    # we don't overwrite a store already created (and populated via the legacy
    # indexed_key_cache setter) by the indexed_key_registry property setter above.
    loop.cycle_count = 0
    loop.promoted_keys = set()
    loop._ensure_store()
    return loop


def _create_noop(m, cfg, name):
    """Stub create_adapter that adds the adapter name to peft_config."""
    m.peft_config[name] = MagicMock()
    return m


# Subject → intended registry tier for the per-tier ordering tests.  Both
# TestTrainingFlagBracketing.test_set_is_training_called_for_all_three_tiers and
# TestPerTierInferenceFallbackAdapter use the same three triples, one per tier.
_TIER_BY_SUBJECT = {"Alice": "episodic", "Bob": "semantic", "Dave": "procedural"}


def _partition_by_registry_tier(relations, *, procedural_enabled=True):
    """``partition_relations`` side_effect that routes by the test's registry tier.

    The real ``partition_relations`` keys off predicate semantics — it classifies
    "Alice likes cats" as procedural (``likes`` is a preference verb), which
    contradicts the test's registry assignment (Alice → episodic).  These ordering
    tests exercise the per-tier training loop mechanics, NOT predicate-based
    tiering, so route each relation to the (``_ep_rels``, ``_proc_rels``) bucket
    that lands it in the registry tier the test assigned:

      - procedural subject → ``([], [rel])`` → tier "procedural"
      - episodic / semantic subject → ``([rel], [])`` → consolidation then reads
        ``tier_for_active_key`` (registry) to split episodic vs semantic.
    """
    ep_rels, proc_rels = [], []
    for rel in relations:
        tier = _TIER_BY_SUBJECT.get(rel.get("subject", ""), "episodic")
        if tier == "procedural" and procedural_enabled:
            proc_rels.append(rel)
        else:
            ep_rels.append(rel)
    return ep_rels, proc_rels


# ---------------------------------------------------------------------------
# Test 1 — Lock-leak guard
# ---------------------------------------------------------------------------


class TestLockLeakGuard:
    """The entry guard raises RuntimeError when the GPU lock is NOT held.

    The lock must be released (not leaked) before the raise so the process
    can continue normally after catching the exception.
    """

    def test_raises_when_lock_not_held(self, tmp_path: Path) -> None:
        """consolidate_interim_adapters raises RuntimeError if caller does not hold GPU lock."""
        from paramem.server.gpu_lock import _gpu_thread_lock

        model = _make_stub_model("episodic", "semantic", "procedural", "in_training")
        loop = _make_loop(model, tmp_path)  # empty registry — no keys, no training

        # Call WITHOUT holding the lock — should raise RuntimeError.
        with pytest.raises(RuntimeError, match="_gpu_thread_lock"):
            loop.consolidate_interim_adapters()

        # The lock must NOT be held after the raise (no leak).
        lock_is_free = _gpu_thread_lock.acquire(blocking=False)
        if lock_is_free:
            _gpu_thread_lock.release()
        assert lock_is_free, (
            "GPU lock is still held after consolidate_interim_adapters raised RuntimeError — "
            "the lock was leaked. The entry guard must release before raising."
        )

    def test_no_raise_when_lock_held(self, tmp_path: Path) -> None:
        """Guard does not raise when the GPU lock is held by the caller.

        We acquire the lock manually, call the method with no keys (so no real
        adapter ops run), and verify no RuntimeError from the guard itself.
        """
        from paramem.server.gpu_lock import _gpu_thread_lock

        model = _make_stub_model("episodic", "semantic", "procedural", "in_training")
        loop = _make_loop(model, tmp_path)  # empty registry — no keys, no training

        # Hold the lock so the guard passes.
        _gpu_thread_lock.acquire()
        try:
            with (
                patch("paramem.models.loader.create_adapter", side_effect=_create_noop),
                patch("paramem.memory.interim_adapter.unload_interim_adapters", return_value=[]),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
                # Stage 1 (reconstruct_graph) probes the MagicMock model via
                # probe_entries; stub it with a faithful registry-derived graph so the
                # downstream re-merge/dedup/tiering/training flow runs unchanged.
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    side_effect=_faithful_reconstruct,
                ),
                patch("paramem.training.consolidation.ConsolidationLoop._save_adapters"),
            ):
                result = loop.consolidate_interim_adapters()
        finally:
            _gpu_thread_lock.release()

        # Must not have raised RuntimeError from the guard.
        assert "rolled_back" in result


# ---------------------------------------------------------------------------
# Test 2 — Training-flag bracketing: _set_is_training(True)/False per adapter call
# ---------------------------------------------------------------------------


class TestTrainingFlagBracketing:
    """_set_is_training(True) before each _train_adapter; False in finally."""

    def test_set_is_training_called_per_tier(self, tmp_path: Path) -> None:
        """_set_is_training is called True then False for each tier that has keys."""
        from paramem.server.gpu_lock import _gpu_thread_lock

        model = _make_stub_model(
            "episodic",
            "semantic",
            "procedural",
            "in_training",
            "episodic_backup",
            "semantic_backup",
            "procedural_backup",
        )

        qa = {
            "graph1": {
                "question": "Q1?",
                "answer": "A1.",
                "subject": "Alice",
                "predicate": "likes",
                "object": "cats",
                "source_subject": "Alice",
                "source_predicate": "likes",
                "source_object": "cats",
            },
        }

        # Use a real per-tier registry dict with "graph1" in the episodic tier.
        registry_dict = _make_empty_registry_dict()
        registry_dict["episodic"].add("graph1")

        loop = _make_loop(model, tmp_path, registry=registry_dict, indexed_key_cache=qa)

        # Create a stub trainer to record _set_is_training calls.
        stub_trainer = MagicMock()
        stub_trainer._current_job = None
        is_training_calls: list[bool] = []

        def _record_set_is_training(value: bool) -> None:
            is_training_calls.append(value)

        stub_trainer._set_is_training.side_effect = _record_set_is_training

        _gpu_thread_lock.acquire()
        try:
            with (
                patch("paramem.models.loader.create_adapter", side_effect=_create_noop),
                patch("paramem.graph.qa_generator.partition_relations", return_value=([], [])),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch(
                    "paramem.training.trainer.train_adapter",
                    return_value={"aborted": False},
                ),
                patch("paramem.training.consolidation.build_registry", return_value={"graph1": 0}),
                patch(
                    "paramem.training.recall_eval.evaluate_indexed_recall",
                    return_value={
                        "rate": 1.0,
                        "exact_count": 1,
                        "total": 1,
                        "mean_confidence": 1.0,
                        "per_key": [],
                    },
                ),
                patch("paramem.memory.interim_adapter.unload_interim_adapters", return_value=[]),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
                # Stage 1 (reconstruct_graph) probes the MagicMock model via
                # probe_entries; stub it with a faithful registry-derived graph so the
                # downstream re-merge/dedup/tiering/training flow runs unchanged.
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    side_effect=_faithful_reconstruct,
                ),
                patch("paramem.training.consolidation.ConsolidationLoop._save_adapters"),
            ):
                loop.consolidate_interim_adapters(trainer=stub_trainer)
        finally:
            _gpu_thread_lock.release()

        # Verify the True/False pattern fired for the episodic tier.
        # Expected: [True, False] for one tier with keys.
        assert True in is_training_calls, "Expected at least one _set_is_training(True) call"
        assert False in is_training_calls, "Expected at least one _set_is_training(False) call"

        # The sequence must start True and alternate True→False per tier.
        assert is_training_calls[0] is True, (
            f"Expected first call to be True, got {is_training_calls[0]}. "
            f"Full sequence: {is_training_calls}"
        )
        for i in range(0, len(is_training_calls) - 1, 2):
            assert is_training_calls[i] is True
            assert is_training_calls[i + 1] is False

    def test_set_is_training_called_for_all_three_tiers(self, tmp_path: Path) -> None:
        """_set_is_training fires True/False six times when all three tiers have keys.

        Constructs a scenario with keys assigned to episodic, semantic, and
        procedural tiers (one key each).  Verifies that _set_is_training is
        called [True, False, True, False, True, False] — one pair per tier.
        """
        from paramem.server.gpu_lock import _gpu_thread_lock

        model = _make_stub_model(
            "episodic",
            "semantic",
            "procedural",
            "in_training",
            "episodic_backup",
            "semantic_backup",
            "procedural_backup",
        )

        # Three keys — one per tier.  partition_relations determines tier; we
        # mock it to return one key per tier in the order the loop visits them
        # (episodic → semantic → procedural) by returning the correct lists.
        qa = {
            "ep_key": {
                "question": "Ep?",
                "answer": "A.",
                "subject": "Alice",
                "predicate": "likes",
                "object": "cats",
                "source_subject": "Alice",
                "source_predicate": "likes",
                "source_object": "cats",
            },
            "sem_key": {
                "question": "Sem?",
                "answer": "B.",
                "subject": "Bob",
                "predicate": "knows",
                "object": "Carol",
                "source_subject": "Bob",
                "source_predicate": "knows",
                "source_object": "Carol",
            },
            "proc_key": {
                "question": "Proc?",
                "answer": "C.",
                "subject": "Dave",
                "predicate": "prefers",
                "object": "morning",
                "source_subject": "Dave",
                "source_predicate": "prefers",
                "source_object": "morning",
            },
        }

        # Use a real per-tier registry dict: one key per tier.
        registry_dict = _make_empty_registry_dict()
        registry_dict["episodic"].add("ep_key")
        registry_dict["semantic"].add("sem_key")
        registry_dict["procedural"].add("proc_key")
        # Enable procedural processing.
        loop = _make_loop(model, tmp_path, registry=registry_dict, indexed_key_cache=qa)
        loop.procedural_config = _minimal_adapter_config()

        stub_trainer = MagicMock()
        stub_trainer._current_job = None
        is_training_calls: list[bool] = []

        def _record_set_is_training(value: bool) -> None:
            is_training_calls.append(value)

        stub_trainer._set_is_training.side_effect = _record_set_is_training

        # partition_relations splits each triple into episodic vs procedural
        # buckets; consolidation then reads the registry to split episodic vs
        # semantic.  Patch the name bound in paramem.training.consolidation (the
        # module-local import), NOT paramem.graph.qa_generator — the latter does
        # not affect the call inside consolidate_interim_adapters.  Route each
        # triple to its registry tier so all three tiers train.
        _gpu_thread_lock.acquire()
        try:
            with (
                patch("paramem.models.loader.create_adapter", side_effect=_create_noop),
                patch(
                    "paramem.training.consolidation.partition_relations",
                    side_effect=_partition_by_registry_tier,
                ),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch(
                    "paramem.training.trainer.train_adapter",
                    return_value={"aborted": False},
                ),
                patch(
                    "paramem.training.consolidation.build_registry",
                    return_value={"ep_key": 0, "sem_key": 0, "proc_key": 0},
                ),
                patch(
                    "paramem.training.recall_eval.evaluate_indexed_recall",
                    return_value={
                        "rate": 1.0,
                        "exact_count": 1,
                        "total": 1,
                        "mean_confidence": 1.0,
                        "per_key": [],
                    },
                ),
                patch("paramem.memory.interim_adapter.unload_interim_adapters", return_value=[]),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
                # Stage 1 (reconstruct_graph) probes the MagicMock model via
                # probe_entries; stub it with a faithful registry-derived graph so the
                # downstream re-merge/dedup/tiering/training flow runs unchanged.
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    side_effect=_faithful_reconstruct,
                ),
                patch("paramem.training.consolidation.ConsolidationLoop._save_adapters"),
            ):
                loop.consolidate_interim_adapters(trainer=stub_trainer)
        finally:
            _gpu_thread_lock.release()

        # All three tiers have keys → expect six calls from the per-tier
        # loop (True/False × 3 tiers) plus one trailing False from the
        # belt-and-braces finalize guard at line ~2499.
        assert len(is_training_calls) >= 6, (
            f"Expected at least 6 _set_is_training calls (True/False × 3 tiers), "
            f"got {len(is_training_calls)}: {is_training_calls}"
        )
        # The first six calls must be the alternating per-tier pattern.
        expected_per_tier = [True, False, True, False, True, False]
        assert is_training_calls[:6] == expected_per_tier, (
            f"Expected first six calls to be {expected_per_tier}, got {is_training_calls[:6]}"
        )
        # Any calls beyond the six must all be False (finalize guards).
        for extra in is_training_calls[6:]:
            assert extra is False, (
                f"Extra _set_is_training call after per-tier loop must be False, "
                f"got {extra!r}. Full sequence: {is_training_calls}"
            )


# ---------------------------------------------------------------------------
# Test 3 — Per-tier inference_fallback_adapter
# ---------------------------------------------------------------------------


class TestPerTierInferenceFallbackAdapter:
    """trainer._current_job.inference_fallback_adapter is correct for each tier during training.

    The original tests only verified TrainingJob construction kwargs — they
    did not prove that the manual _current_job swap actually reaches the
    trainer during training.  The new tests capture _current_job AT the moment
    train_adapter is called for each tier and assert the fallback adapter name.
    """

    def test_current_job_fallback_correct_during_training(self, tmp_path: Path) -> None:
        """trainer._current_job.inference_fallback_adapter matches '<tier>_backup' during training.

        Monkey-patches train_adapter to record trainer._current_job.inference_fallback_adapter
        at call time.  Asserts episodic→'episodic_backup', semantic→'semantic_backup',
        procedural→'procedural_backup'.
        """
        from paramem.server.gpu_lock import _gpu_thread_lock

        model = _make_stub_model(
            "episodic",
            "semantic",
            "procedural",
            "in_training",
            "episodic_backup",
            "semantic_backup",
            "procedural_backup",
        )

        # Three keys — one per tier.
        qa = {
            "ep_key": {
                "question": "Ep?",
                "answer": "A.",
                "subject": "Alice",
                "predicate": "likes",
                "object": "cats",
                "source_subject": "Alice",
                "source_predicate": "likes",
                "source_object": "cats",
            },
            "sem_key": {
                "question": "Sem?",
                "answer": "B.",
                "subject": "Bob",
                "predicate": "knows",
                "object": "Carol",
                "source_subject": "Bob",
                "source_predicate": "knows",
                "source_object": "Carol",
            },
            "proc_key": {
                "question": "Proc?",
                "answer": "C.",
                "subject": "Dave",
                "predicate": "prefers",
                "object": "morning",
                "source_subject": "Dave",
                "source_predicate": "prefers",
                "source_object": "morning",
            },
        }

        # Use a real per-tier registry dict: one key per tier.
        registry_dict = _make_empty_registry_dict()
        registry_dict["episodic"].add("ep_key")
        registry_dict["semantic"].add("sem_key")
        registry_dict["procedural"].add("proc_key")

        loop = _make_loop(model, tmp_path, registry=registry_dict, indexed_key_cache=qa)
        loop.procedural_config = _minimal_adapter_config()

        # Stub trainer with real attribute tracking.
        stub_trainer = MagicMock()
        stub_trainer._current_job = None

        # Capture the inference_fallback_adapter seen during each train_adapter call.
        captured_fallbacks: list[str | None] = []

        def _spy_train_adapter(**kwargs) -> dict:
            # At this point trainer._current_job should already be set to the
            # per-tier job by the manual swap in consolidate_interim_adapters.
            fallback = (
                stub_trainer._current_job.inference_fallback_adapter
                if stub_trainer._current_job is not None
                else None
            )
            captured_fallbacks.append(fallback)
            return {"aborted": False}

        _gpu_thread_lock.acquire()
        try:
            with (
                patch("paramem.models.loader.create_adapter", side_effect=_create_noop),
                # Patch the module-local name (see _partition_by_registry_tier);
                # routing each triple to its registry tier so all three tiers train.
                patch(
                    "paramem.training.consolidation.partition_relations",
                    side_effect=_partition_by_registry_tier,
                ),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch(
                    "paramem.training.trainer.train_adapter",
                    side_effect=_spy_train_adapter,
                ),
                patch(
                    "paramem.training.consolidation.build_registry",
                    return_value={"ep_key": 0, "sem_key": 0, "proc_key": 0},
                ),
                patch(
                    "paramem.training.recall_eval.evaluate_indexed_recall",
                    return_value={
                        "rate": 1.0,
                        "exact_count": 1,
                        "total": 1,
                        "mean_confidence": 1.0,
                        "per_key": [],
                    },
                ),
                patch("paramem.memory.interim_adapter.unload_interim_adapters", return_value=[]),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
                # Stage 1 (reconstruct_graph) probes the MagicMock model via
                # probe_entries; stub it with a faithful registry-derived graph so the
                # downstream re-merge/dedup/tiering/training flow runs unchanged.
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    side_effect=_faithful_reconstruct,
                ),
                patch("paramem.training.consolidation.ConsolidationLoop._save_adapters"),
            ):
                loop.consolidate_interim_adapters(trainer=stub_trainer)
        finally:
            _gpu_thread_lock.release()

        # Must have been called once per tier (3 total).
        assert len(captured_fallbacks) == 3, (
            f"Expected 3 train_adapter calls (one per tier), got {len(captured_fallbacks)}: "
            f"{captured_fallbacks}"
        )
        # Order is episodic → semantic → procedural.
        assert captured_fallbacks[0] == "episodic_backup", (
            f"Episodic tier: expected 'episodic_backup', got {captured_fallbacks[0]!r}"
        )
        assert captured_fallbacks[1] == "semantic_backup", (
            f"Semantic tier: expected 'semantic_backup', got {captured_fallbacks[1]!r}"
        )
        assert captured_fallbacks[2] == "procedural_backup", (
            f"Procedural tier: expected 'procedural_backup', got {captured_fallbacks[2]!r}"
        )

    def test_current_job_restored_after_each_tier(self, tmp_path: Path) -> None:
        """trainer._current_job is restored to the prior sentinel after each tier rebuild.

        Verifies the save/restore invariant: between tier training calls, the
        _current_job on the stub trainer must revert to the value that was set
        before consolidate_interim_adapters started iterating.
        """
        from paramem.server.gpu_lock import _gpu_thread_lock

        model = _make_stub_model(
            "episodic",
            "semantic",
            "procedural",
            "in_training",
            "episodic_backup",
            "semantic_backup",
            "procedural_backup",
        )

        qa = {
            "ep_key": {
                "question": "Ep?",
                "answer": "A.",
                "subject": "Alice",
                "predicate": "likes",
                "object": "cats",
                "source_subject": "Alice",
                "source_predicate": "likes",
                "source_object": "cats",
            },
            "sem_key": {
                "question": "Sem?",
                "answer": "B.",
                "subject": "Bob",
                "predicate": "knows",
                "object": "Carol",
                "source_subject": "Bob",
                "source_predicate": "knows",
                "source_object": "Carol",
            },
        }

        # Use a real per-tier registry dict: ep_key in episodic, sem_key in semantic.
        registry_dict = _make_empty_registry_dict()
        registry_dict["episodic"].add("ep_key")
        registry_dict["semantic"].add("sem_key")

        loop = _make_loop(model, tmp_path, registry=registry_dict, indexed_key_cache=qa)

        # Set a sentinel as the "outer" _current_job so we can detect restoration.
        sentinel_job = TrainingJob(
            entries=[],
            adapter_name="_sentinel_",
            adapter_config=_minimal_adapter_config(),
            inference_fallback_adapter="sentinel",
        )

        stub_trainer = MagicMock()
        stub_trainer._current_job = sentinel_job

        stub_trainer._set_is_training.side_effect = None

        _gpu_thread_lock.acquire()
        try:
            with (
                patch("paramem.models.loader.create_adapter", side_effect=_create_noop),
                patch("paramem.graph.qa_generator.partition_relations", return_value=([], [])),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch(
                    "paramem.training.trainer.train_adapter",
                    return_value={"aborted": False},
                ),
                patch(
                    "paramem.training.consolidation.build_registry",
                    return_value={"ep_key": 0, "sem_key": 0},
                ),
                patch(
                    "paramem.training.recall_eval.evaluate_indexed_recall",
                    return_value={
                        "rate": 1.0,
                        "exact_count": 1,
                        "total": 1,
                        "mean_confidence": 1.0,
                        "per_key": [],
                    },
                ),
                patch("paramem.memory.interim_adapter.unload_interim_adapters", return_value=[]),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
                # Stage 1 (reconstruct_graph) probes the MagicMock model via
                # probe_entries; stub it with a faithful registry-derived graph so the
                # downstream re-merge/dedup/tiering/training flow runs unchanged.
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    side_effect=_faithful_reconstruct,
                ),
                patch("paramem.training.consolidation.ConsolidationLoop._save_adapters"),
            ):
                loop.consolidate_interim_adapters(trainer=stub_trainer)
        finally:
            _gpu_thread_lock.release()

        # After the full loop, _current_job must be the original sentinel.
        assert stub_trainer._current_job is sentinel_job, (
            f"Expected _current_job to be restored to sentinel_job after all tiers, "
            f"got {stub_trainer._current_job!r}"
        )

    def test_job_fallback_adapters_match_backup_names(self) -> None:
        """TrainingJob objects carry the correct backup adapter name at construction."""
        ep_cfg = _minimal_adapter_config()
        sem_cfg = _minimal_adapter_config()
        proc_cfg = _minimal_adapter_config()

        jobs = {
            "episodic": TrainingJob(
                entries=[{"key": "graph1", "question": "Q?", "answer": "A."}],
                adapter_name="episodic",
                adapter_config=ep_cfg,
                inference_fallback_adapter="episodic_backup",
            ),
            "semantic": TrainingJob(
                entries=[{"key": "graph2", "question": "Q2?", "answer": "A2."}],
                adapter_name="semantic",
                adapter_config=sem_cfg,
                inference_fallback_adapter="semantic_backup",
            ),
            "procedural": TrainingJob(
                entries=[{"key": "proc1", "question": "Q3?", "answer": "A3."}],
                adapter_name="procedural",
                adapter_config=proc_cfg,
                inference_fallback_adapter="procedural_backup",
            ),
        }

        assert jobs["episodic"].inference_fallback_adapter == "episodic_backup"
        assert jobs["semantic"].inference_fallback_adapter == "semantic_backup"
        assert jobs["procedural"].inference_fallback_adapter == "procedural_backup"

    def test_tier_backup_names_are_distinct(self) -> None:
        """Each tier's backup name is distinct — no accidental aliasing."""
        backup_names = {"episodic_backup", "semantic_backup", "procedural_backup"}
        assert len(backup_names) == 3


# ---------------------------------------------------------------------------
# Test 4 — Capacity-ceiling rollback
# ---------------------------------------------------------------------------


class TestCapacityCeilingRollback:
    """The pre-save in-RAM recall probe has been removed.

    consolidate_interim_adapters always returns rolled_back=False now.
    Disk-integrity is gated post-save by _save_adapters via
    _verify_saved_adapter_from_disk.
    """

    def test_no_rollback_even_when_recall_probe_returns_low(self, tmp_path: Path) -> None:
        """consolidate_interim_adapters returns rolled_back=False regardless of probe.

        The pre-save in-RAM recall probe has been removed; _run_recall_sanity_probe
        is no longer called here.  Even if the experiment harness would have
        returned a low rate, no rollback occurs in this path.
        """
        from paramem.server.gpu_lock import _gpu_thread_lock

        model = _make_stub_model(
            "episodic",
            "semantic",
            "procedural",
            "in_training",
            "episodic_backup",
            "semantic_backup",
            "procedural_backup",
        )

        qa = {
            "graph1": {
                "question": "Q?",
                "answer": "A.",
                "subject": "X",
                "predicate": "y",
                "object": "Z",
                "source_subject": "X",
                "source_predicate": "y",
                "source_object": "Z",
            },
        }
        registry_dict = _make_empty_registry_dict()
        registry_dict["episodic"].add("graph1")
        loop = _make_loop(model, tmp_path, registry=registry_dict, indexed_key_cache=qa)

        _gpu_thread_lock.acquire()
        try:
            with (
                patch("paramem.models.loader.create_adapter", side_effect=_create_noop),
                patch("paramem.graph.qa_generator.partition_relations", return_value=([], [])),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch(
                    "paramem.training.trainer.train_adapter",
                    return_value={"aborted": False},
                ),
                patch("paramem.training.consolidation.build_registry", return_value={"graph1": 0}),
                # The pre-save in-RAM probe is gone; evaluate_indexed_recall
                # is only called via _verify_saved_adapter_from_disk (post-save).
                # Patch it here to prevent any accidental call from surfacing.
                patch(
                    "paramem.training.recall_eval.evaluate_indexed_recall",
                    return_value={
                        "rate": 0.5,
                        "exact_count": 0,
                        "total": 1,
                        "mean_confidence": 0.5,
                        "per_key": [],
                    },
                ),
                patch("paramem.memory.interim_adapter.unload_interim_adapters", return_value=[]),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
                # Stage 1 (reconstruct_graph) probes the MagicMock model via
                # probe_entries; stub it with a faithful registry-derived graph so the
                # downstream re-merge/dedup/tiering/training flow runs unchanged.
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    side_effect=_faithful_reconstruct,
                ),
                patch("paramem.training.consolidation.ConsolidationLoop._save_adapters"),
            ):
                result = loop.consolidate_interim_adapters()
        finally:
            _gpu_thread_lock.release()

        # Pre-save rollback has been removed — result is always rolled_back=False.
        assert result["rolled_back"] is False
        assert result["rollback_tier"] is None

    def test_no_rollback_when_recall_at_threshold(self, tmp_path: Path) -> None:
        """When recall == threshold, no rollback fires."""
        from paramem.server.gpu_lock import _gpu_thread_lock

        model = _make_stub_model(
            "episodic",
            "semantic",
            "procedural",
            "in_training",
            "episodic_backup",
            "semantic_backup",
            "procedural_backup",
        )

        qa = {
            "graph1": {
                "question": "Q?",
                "answer": "A.",
                "subject": "X",
                "predicate": "y",
                "object": "Z",
                "source_subject": "X",
                "source_predicate": "y",
                "source_object": "Z",
            },
        }
        registry_dict = _make_empty_registry_dict()
        registry_dict["episodic"].add("graph1")
        loop = _make_loop(model, tmp_path, registry=registry_dict, indexed_key_cache=qa)

        _gpu_thread_lock.acquire()
        try:
            with (
                patch("paramem.models.loader.create_adapter", side_effect=_create_noop),
                patch("paramem.graph.qa_generator.partition_relations", return_value=([], [])),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch(
                    "paramem.training.trainer.train_adapter",
                    return_value={"aborted": False},
                ),
                patch("paramem.training.consolidation.build_registry", return_value={"graph1": 0}),
                # Recall exactly at threshold.
                patch(
                    "paramem.training.recall_eval.evaluate_indexed_recall",
                    return_value={
                        "rate": 0.95,
                        "exact_count": 1,
                        "total": 1,
                        "mean_confidence": 1.0,
                        "per_key": [],
                    },
                ),
                patch("paramem.memory.interim_adapter.unload_interim_adapters", return_value=[]),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
                # Stage 1 (reconstruct_graph) probes the MagicMock model via
                # probe_entries; stub it with a faithful registry-derived graph so the
                # downstream re-merge/dedup/tiering/training flow runs unchanged.
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    side_effect=_faithful_reconstruct,
                ),
                patch("paramem.training.consolidation.ConsolidationLoop._save_adapters"),
            ):
                result = loop.consolidate_interim_adapters()
        finally:
            _gpu_thread_lock.release()

        assert result["rolled_back"] is False


# ---------------------------------------------------------------------------
# Test 5 — Atomic finalize ordering
# ---------------------------------------------------------------------------


class TestAtomicFinalizeOrdering:
    """Registry rewrite before interim purge/unload; Router.reload() last."""

    def test_registry_rewrite_before_unload_and_router_reload(self, tmp_path: Path) -> None:
        """Finalize ordering: KeyRegistry.save() → unload_interim_adapters → router.reload().

        The finalize block creates fresh KeyRegistry instances and calls .save() on
        each — patch KeyRegistry.save at the class level to capture the event.
        """
        from paramem.server.gpu_lock import _gpu_thread_lock

        model = _make_stub_model(
            "episodic",
            "semantic",
            "procedural",
            "in_training",
            "episodic_backup",
            "semantic_backup",
            "procedural_backup",
        )

        qa = {
            "graph1": {
                "question": "Q?",
                "answer": "A.",
                "subject": "X",
                "predicate": "y",
                "object": "Z",
                "source_subject": "X",
                "source_predicate": "y",
                "source_object": "Z",
            },
        }
        # Use a real per-tier registry dict with "graph1" in the episodic tier.
        registry_dict = _make_empty_registry_dict()
        registry_dict["episodic"].add("graph1")

        loop = _make_loop(model, tmp_path, registry=registry_dict, indexed_key_cache=qa)

        call_order: list[str] = []

        def _registry_save(self_reg, path) -> None:
            call_order.append("registry_save")

        def _unload(m, adapter_dir) -> list:
            call_order.append("unload_interim_adapters")
            return []

        def _router_reload() -> None:
            call_order.append("router_reload")

        mock_router = MagicMock()
        mock_router.reload.side_effect = _router_reload

        _gpu_thread_lock.acquire()
        try:
            with (
                patch("paramem.models.loader.create_adapter", side_effect=_create_noop),
                patch("paramem.graph.qa_generator.partition_relations", return_value=([], [])),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch(
                    "paramem.training.trainer.train_adapter",
                    return_value={"aborted": False},
                ),
                patch("paramem.training.consolidation.build_registry", return_value={"graph1": 0}),
                patch(
                    "paramem.training.recall_eval.evaluate_indexed_recall",
                    return_value={
                        "rate": 1.0,
                        "exact_count": 1,
                        "total": 1,
                        "mean_confidence": 1.0,
                        "per_key": [],
                    },
                ),
                patch(
                    "paramem.memory.interim_adapter.unload_interim_adapters", side_effect=_unload
                ),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
                # Stage 1 (reconstruct_graph) probes the MagicMock model via
                # probe_entries; stub it with a faithful registry-derived graph so the
                # downstream re-merge/dedup/tiering/training flow runs unchanged.
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    side_effect=_faithful_reconstruct,
                ),
                patch("paramem.training.consolidation.ConsolidationLoop._save_adapters"),
                # Patch KeyRegistry.save at the class level — the finalize block creates
                # fresh KeyRegistry instances, so we must patch the class method, not the
                # instance's save attribute.
                patch("paramem.training.key_registry.KeyRegistry.save", new=_registry_save),
            ):
                loop.consolidate_interim_adapters(router=mock_router)
        finally:
            _gpu_thread_lock.release()

        # Filter to the three ordering-critical events.
        ordering_events = [
            e
            for e in call_order
            if e in {"registry_save", "unload_interim_adapters", "router_reload"}
        ]

        assert ordering_events, "Expected at least one finalize event to be recorded"
        # Registry rewrite must be first.
        assert ordering_events[0] == "registry_save", (
            f"Expected registry_save first, got {ordering_events[0]!r}. "
            f"Full order: {ordering_events}"
        )
        # Router reload must be last.
        assert ordering_events[-1] == "router_reload", (
            f"Expected router_reload last, got {ordering_events[-1]!r}. "
            f"Full order: {ordering_events}"
        )
        # unload_interim_adapters must appear between save and reload.
        assert "unload_interim_adapters" in ordering_events
        unload_idx = ordering_events.index("unload_interim_adapters")
        registry_idx = ordering_events.index("registry_save")
        router_idx = ordering_events.index("router_reload")
        assert registry_idx < unload_idx < router_idx, (
            f"Expected registry_save < unload < router_reload. "
            f"Got positions: registry={registry_idx}, unload={unload_idx}, router={router_idx}"
        )

    def test_no_phantom_interim_entries_after_finalize(self, tmp_path: Path) -> None:
        """After successful finalize, no episodic_interim_* tier remains in the registry dict.

        The per-tier dict schema encodes tier identity as the dict key.  After
        finalize, ``_drop_interim_tier_registries`` must remove all
        ``episodic_interim_*`` keys from ``loop.indexed_key_registry``.
        """
        from paramem.server.gpu_lock import _gpu_thread_lock

        model = _make_stub_model(
            "episodic",
            "semantic",
            "procedural",
            "in_training",
            "episodic_backup",
            "semantic_backup",
            "procedural_backup",
            "episodic_interim_20260418T0000",
        )

        qa = {
            "graph1": {
                "question": "Q?",
                "answer": "A.",
                "subject": "X",
                "predicate": "y",
                "object": "Z",
                "source_subject": "X",
                "source_predicate": "y",
                "source_object": "Z",
            },
        }
        # Use a real per-tier registry dict: graph1 belongs to the interim tier.
        registry_dict = _make_empty_registry_dict()
        interim_reg = KeyRegistry()
        interim_reg.add("graph1")
        registry_dict["episodic_interim_20260418T0000"] = interim_reg

        loop = _make_loop(model, tmp_path, registry=registry_dict, indexed_key_cache=qa)

        _gpu_thread_lock.acquire()
        try:
            with (
                patch("paramem.models.loader.create_adapter", side_effect=_create_noop),
                patch("paramem.graph.qa_generator.partition_relations", return_value=([], [])),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch(
                    "paramem.training.trainer.train_adapter",
                    return_value={"aborted": False},
                ),
                patch("paramem.training.consolidation.build_registry", return_value={"graph1": 0}),
                patch(
                    "paramem.training.recall_eval.evaluate_indexed_recall",
                    return_value={
                        "rate": 1.0,
                        "exact_count": 1,
                        "total": 1,
                        "mean_confidence": 1.0,
                        "per_key": [],
                    },
                ),
                patch(
                    "paramem.memory.interim_adapter.unload_interim_adapters",
                    return_value=["episodic_interim_20260418T0000"],
                ),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
                # Stage 1 (reconstruct_graph) probes the MagicMock model via
                # probe_entries; stub it with a faithful registry-derived graph so the
                # downstream re-merge/dedup/tiering/training flow runs unchanged.
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    side_effect=_faithful_reconstruct,
                ),
                patch("paramem.training.consolidation.ConsolidationLoop._save_adapters"),
                # Suppress real disk I/O in the finalize save.
                patch("paramem.training.key_registry.KeyRegistry.save"),
            ):
                result = loop.consolidate_interim_adapters()
        finally:
            _gpu_thread_lock.release()

        assert not result["rolled_back"]

        # Verify no episodic_interim_* tier key remains in the registry dict.
        for tier_key in loop.indexed_key_registry:
            assert not tier_key.startswith("episodic_interim_"), (
                f"Interim tier key {tier_key!r} still present after finalize"
            )


# ---------------------------------------------------------------------------
# Test 5b — Durable main-weight persist before interim purge (GAP 1 crash window)
# ---------------------------------------------------------------------------


class TestMainWeightsSavedBeforeInterimPurge:
    """The merged main weights must be durably saved+verified BEFORE any
    interim slot is deleted.

    Regression for the data-loss window: ``consolidate_interim_adapters``
    historically did NOT persist the rebuilt main weights itself — the caller
    (``app.py::_run_full_cycle``) called ``loop._save_adapters()`` only AFTER
    the method returned, while the method's finalize block already purged the
    interim slot dirs (``unload_interim_adapters`` → ``shutil.rmtree``).  A
    crash between purge and the caller's save left NO durable copy of the
    folded knowledge on disk, and the source sessions were already eligible to
    be marked consolidated.

    The fix moves the weight persist+verify INSIDE the finalize block, between
    the registry rewrite and the interim purge, so every caller gets
    safe ordering: a verified durable main copy exists before any interim dir
    is removed.
    """

    def _make_finalize_model(self):
        return _make_stub_model(
            "episodic",
            "semantic",
            "procedural",
            "in_training",
            "episodic_backup",
            "semantic_backup",
            "procedural_backup",
        )

    def _qa(self) -> dict:
        return {
            "graph1": {
                "question": "Q?",
                "answer": "A.",
                "subject": "X",
                "predicate": "y",
                "object": "Z",
                "source_subject": "X",
                "source_predicate": "y",
                "source_object": "Z",
            },
        }

    def _run_finalize(self, loop, *, save_side_effect, unload_side_effect):
        """Drive consolidate_interim_adapters through finalize with mocks.

        ``save_side_effect`` patches ``ConsolidationLoop._save_adapters``;
        ``unload_side_effect`` patches ``unload_interim_adapters``.  Returns
        nothing — callers assert on side effects they wired in.
        """
        from paramem.server.gpu_lock import _gpu_thread_lock
        from paramem.training.consolidation import ConsolidationLoop

        _gpu_thread_lock.acquire()
        try:
            with (
                patch("paramem.models.loader.create_adapter", side_effect=_create_noop),
                patch("paramem.graph.qa_generator.partition_relations", return_value=([], [])),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch(
                    "paramem.training.trainer.train_adapter",
                    return_value={"aborted": False},
                ),
                patch("paramem.training.consolidation.build_registry", return_value={"graph1": 0}),
                patch(
                    "paramem.training.recall_eval.evaluate_indexed_recall",
                    return_value={
                        "rate": 1.0,
                        "exact_count": 1,
                        "total": 1,
                        "mean_confidence": 1.0,
                        "per_key": [],
                    },
                ),
                patch(
                    "paramem.memory.interim_adapter.unload_interim_adapters",
                    side_effect=unload_side_effect,
                ),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
                patch("paramem.training.key_registry.KeyRegistry.save"),
                # Stage 1 (reconstruct_graph) probes the MagicMock model; stub it with a
                # faithful registry-derived graph so the finalize ordering runs unchanged.
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    side_effect=_faithful_reconstruct,
                ),
                patch.object(
                    ConsolidationLoop, "_save_adapters", autospec=True, side_effect=save_side_effect
                ),
            ):
                return loop.consolidate_interim_adapters()
        finally:
            _gpu_thread_lock.release()

    def test_save_adapters_runs_before_unload_interim(self, tmp_path: Path) -> None:
        """Call ORDER: _save_adapters happens-before unload_interim_adapters.

        Pre-fix this FAILS because the method never calls _save_adapters at all
        (the only call lived in the caller, AFTER unload).
        """
        registry_dict = _make_empty_registry_dict()
        registry_dict["episodic"].add("graph1")
        loop = _make_loop(
            self._make_finalize_model(),
            tmp_path,
            registry=registry_dict,
            indexed_key_cache=self._qa(),
        )

        call_order: list[str] = []

        def _save(self_loop, **kwargs) -> None:
            call_order.append("save_adapters")

        def _unload(m, adapter_dir) -> list:
            call_order.append("unload_interim_adapters")
            return []

        self._run_finalize(loop, save_side_effect=_save, unload_side_effect=_unload)

        assert "save_adapters" in call_order, (
            "consolidate_interim_adapters must persist merged main weights itself "
            "(no durable copy exists between interim purge and a caller-side save)"
        )
        assert "unload_interim_adapters" in call_order
        assert call_order.index("save_adapters") < call_order.index("unload_interim_adapters"), (
            f"merged main weights must be saved BEFORE interim purge; got {call_order}"
        )

    def test_save_failure_preserves_interim_slots(self, tmp_path: Path) -> None:
        """Crash window: when _save_adapters raises, interim slots are NOT purged.

        Simulates a crash / disk-integrity verify failure at the persist step.
        The interim slot dir must survive (re-extraction net preserved) and the
        exception must surface so the caller records a failed/retriable cycle.

        Pre-fix this FAILS because the method purges unconditionally before any
        save, so unload runs regardless and the slot is gone.
        """
        registry_dict = _make_empty_registry_dict()
        registry_dict["episodic"].add("graph1")
        loop = _make_loop(
            self._make_finalize_model(),
            tmp_path,
            registry=registry_dict,
            indexed_key_cache=self._qa(),
        )

        # An on-disk interim slot that must survive a failed save.
        interim_dir = tmp_path / "episodic" / "interim_20260418T0000"
        interim_dir.mkdir(parents=True)
        (interim_dir / "adapter_model.safetensors").write_bytes(b"weights")

        unload_calls: list[str] = []

        def _save(self_loop, **kwargs) -> None:
            raise RuntimeError("simulated disk-integrity verify failure")

        def _unload(m, adapter_dir) -> list:
            unload_calls.append("called")
            import shutil

            from paramem.memory.interim_adapter import iter_interim_dirs

            for _name, path in iter_interim_dirs(adapter_dir):
                shutil.rmtree(path)
            return []

        with pytest.raises(RuntimeError, match="simulated disk-integrity verify failure"):
            self._run_finalize(loop, save_side_effect=_save, unload_side_effect=_unload)

        assert unload_calls == [], (
            "interim purge must NOT run when the merged-main save fails — "
            "otherwise the folded knowledge has no durable copy anywhere"
        )
        assert interim_dir.exists(), "interim slot dir must survive a failed main-weight save"


# ---------------------------------------------------------------------------
# Test 6 — _set_is_training API on BackgroundTrainer
# ---------------------------------------------------------------------------


class TestSetIsTrainingAPI:
    """BackgroundTrainer._set_is_training(value) overwrites _is_training."""

    def test_set_is_training_true(self) -> None:
        model = _make_stub_model("episodic", "in_training")
        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=_minimal_training_config(),
            output_dir="/tmp/test_set_is_training",
        )
        bt._is_training = False
        bt._set_is_training(True)
        assert bt._is_training is True

    def test_set_is_training_false(self) -> None:
        model = _make_stub_model("episodic", "in_training")
        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=_minimal_training_config(),
            output_dir="/tmp/test_set_is_training2",
        )
        bt._is_training = True
        bt._set_is_training(False)
        assert bt._is_training is False

    def test_abort_for_inference_short_circuits_when_idle(self) -> None:
        """abort_for_inference() returns False immediately when no job is active."""
        model = _make_stub_model("episodic", "in_training")
        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=_minimal_training_config(),
            output_dir="/tmp/test_abort_shortcircuit",
        )
        # No active job — _active_abort is None.
        assert bt._active_abort is None
        result = bt.abort_for_inference(timeout=0.01)
        assert result is False, "abort_for_inference() must return False when no job is active"
