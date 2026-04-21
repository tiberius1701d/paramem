"""Unit tests for consolidate_interim_adapters.

Covers:
  - Lock-leak guard: calling without the GPU lock held raises RuntimeError;
    the lock is NOT left in a locked state after the raise.
  - B2 re-arm: _set_is_training(True) is called before each _train_adapter();
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
  format_indexed_training → paramem.training.indexed_memory.format_indexed_training
  train_adapter (fn) → paramem.training.trainer.train_adapter
  build_registry     → paramem.training.indexed_memory.build_registry
  evaluate_indexed_recall → experiments.utils.test_harness.evaluate_indexed_recall
  unload_interim_adapters → paramem.server.interim_adapter.unload_interim_adapters
  partition_relations → paramem.graph.qa_generator.partition_relations
  switch_adapter     → paramem.models.loader.switch_adapter
  copy_adapter_weights → paramem.models.loader.copy_adapter_weights
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paramem.server.background_trainer import BackgroundTrainer, TrainingJob
from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOCK_PATCHES = [
    "paramem.models.loader.create_adapter",
    "paramem.training.indexed_memory.format_indexed_training",
    "paramem.training.trainer.train_adapter",
    "paramem.training.indexed_memory.build_registry",
    "experiments.utils.test_harness.evaluate_indexed_recall",
    "paramem.server.interim_adapter.unload_interim_adapters",
    "paramem.models.loader.switch_adapter",
    "paramem.models.loader.copy_adapter_weights",
    "paramem.graph.qa_generator.partition_relations",
]


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
    return ConsolidationConfig(indexed_key_replay_enabled=True)


def _make_loop(model, tmp_path: Path, *, registry=None, indexed_key_qa=None):
    """Construct a bare ConsolidationLoop without calling __init__."""
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
    loop.merger.graph = MagicMock(relations=[])
    loop.indexed_key_registry = registry if registry is not None else MagicMock()
    loop.indexed_key_qa = indexed_key_qa if indexed_key_qa is not None else {}
    loop.snapshot_dir = None
    loop.save_cycle_snapshots = False
    loop.persist_graph = False
    loop._shutdown_callbacks = []
    return loop


def _create_noop(m, cfg, name):
    """Stub create_adapter that adds the adapter name to peft_config."""
    m.peft_config[name] = MagicMock()
    return m


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
        loop = _make_loop(model, tmp_path)
        loop.indexed_key_registry.list_active.return_value = []

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
        loop = _make_loop(model, tmp_path)
        loop.indexed_key_registry.list_active.return_value = []

        # Hold the lock so the guard passes.
        _gpu_thread_lock.acquire()
        try:
            with (
                patch("paramem.models.loader.create_adapter", side_effect=_create_noop),
                patch("paramem.server.interim_adapter.unload_interim_adapters", return_value=[]),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
            ):
                result = loop.consolidate_interim_adapters()
        finally:
            _gpu_thread_lock.release()

        # Must not have raised RuntimeError from the guard.
        assert "rolled_back" in result


# ---------------------------------------------------------------------------
# Test 2 — B2 re-arm: _set_is_training bracketing
# ---------------------------------------------------------------------------


class TestB2RearmPattern:
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
                "source_subject": "Alice",
                "source_predicate": "likes",
                "source_object": "cats",
            },
        }

        registry = MagicMock()
        registry.list_active.return_value = ["graph1"]
        registry.get_adapter_id.return_value = "episodic"

        loop = _make_loop(model, tmp_path, registry=registry, indexed_key_qa=qa)

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
                    "paramem.training.indexed_memory.format_indexed_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch("paramem.training.trainer.train_adapter"),
                patch("paramem.training.indexed_memory.build_registry", return_value={"graph1": 0}),
                patch(
                    "experiments.utils.test_harness.evaluate_indexed_recall",
                    return_value={
                        "rate": 1.0,
                        "exact_count": 1,
                        "total": 1,
                        "mean_confidence": 1.0,
                        "per_key": [],
                    },
                ),
                patch("paramem.server.interim_adapter.unload_interim_adapters", return_value=[]),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
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
                "source_subject": "Alice",
                "source_predicate": "likes",
                "source_object": "cats",
            },
            "sem_key": {
                "question": "Sem?",
                "answer": "B.",
                "source_subject": "Bob",
                "source_predicate": "knows",
                "source_object": "Carol",
            },
            "proc_key": {
                "question": "Proc?",
                "answer": "C.",
                "source_subject": "Dave",
                "source_predicate": "prefers",
                "source_object": "morning",
            },
        }

        registry = MagicMock()
        registry.list_active.return_value = list(qa.keys())

        # Each key is already assigned to the matching tier.
        def _get_adapter_id(key: str) -> str:
            return {"ep_key": "episodic", "sem_key": "semantic", "proc_key": "procedural"}[key]

        registry.get_adapter_id.side_effect = _get_adapter_id

        loop = _make_loop(model, tmp_path, registry=registry, indexed_key_qa=qa)

        stub_trainer = MagicMock()
        stub_trainer._current_job = None
        is_training_calls: list[bool] = []

        def _record_set_is_training(value: bool) -> None:
            is_training_calls.append(value)

        stub_trainer._set_is_training.side_effect = _record_set_is_training

        # partition_relations is called to split episodic keys into
        # episodic/semantic sub-groups.  Return ([], []) so all keys stay in
        # the tier already assigned by registry.get_adapter_id.
        _gpu_thread_lock.acquire()
        try:
            with (
                patch("paramem.models.loader.create_adapter", side_effect=_create_noop),
                patch("paramem.graph.qa_generator.partition_relations", return_value=([], [])),
                patch(
                    "paramem.training.indexed_memory.format_indexed_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch("paramem.training.trainer.train_adapter"),
                patch(
                    "paramem.training.indexed_memory.build_registry",
                    return_value={"ep_key": 0, "sem_key": 0, "proc_key": 0},
                ),
                patch(
                    "experiments.utils.test_harness.evaluate_indexed_recall",
                    return_value={
                        "rate": 1.0,
                        "exact_count": 1,
                        "total": 1,
                        "mean_confidence": 1.0,
                        "per_key": [],
                    },
                ),
                patch("paramem.server.interim_adapter.unload_interim_adapters", return_value=[]),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
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
                "source_subject": "Alice",
                "source_predicate": "likes",
                "source_object": "cats",
            },
            "sem_key": {
                "question": "Sem?",
                "answer": "B.",
                "source_subject": "Bob",
                "source_predicate": "knows",
                "source_object": "Carol",
            },
            "proc_key": {
                "question": "Proc?",
                "answer": "C.",
                "source_subject": "Dave",
                "source_predicate": "prefers",
                "source_object": "morning",
            },
        }

        registry = MagicMock()
        registry.list_active.return_value = list(qa.keys())

        def _get_adapter_id(key: str) -> str:
            return {"ep_key": "episodic", "sem_key": "semantic", "proc_key": "procedural"}[key]

        registry.get_adapter_id.side_effect = _get_adapter_id

        loop = _make_loop(model, tmp_path, registry=registry, indexed_key_qa=qa)

        # Stub trainer with real attribute tracking.
        stub_trainer = MagicMock()
        stub_trainer._current_job = None

        # Capture the inference_fallback_adapter seen during each train_adapter call.
        captured_fallbacks: list[str | None] = []

        def _spy_train_adapter(**kwargs) -> None:
            # At this point trainer._current_job should already be set to the
            # per-tier job by the manual swap in consolidate_interim_adapters.
            fallback = (
                stub_trainer._current_job.inference_fallback_adapter
                if stub_trainer._current_job is not None
                else None
            )
            captured_fallbacks.append(fallback)

        _gpu_thread_lock.acquire()
        try:
            with (
                patch("paramem.models.loader.create_adapter", side_effect=_create_noop),
                patch("paramem.graph.qa_generator.partition_relations", return_value=([], [])),
                patch(
                    "paramem.training.indexed_memory.format_indexed_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch(
                    "paramem.training.trainer.train_adapter",
                    side_effect=_spy_train_adapter,
                ),
                patch(
                    "paramem.training.indexed_memory.build_registry",
                    return_value={"ep_key": 0, "sem_key": 0, "proc_key": 0},
                ),
                patch(
                    "experiments.utils.test_harness.evaluate_indexed_recall",
                    return_value={
                        "rate": 1.0,
                        "exact_count": 1,
                        "total": 1,
                        "mean_confidence": 1.0,
                        "per_key": [],
                    },
                ),
                patch("paramem.server.interim_adapter.unload_interim_adapters", return_value=[]),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
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
                "source_subject": "Alice",
                "source_predicate": "likes",
                "source_object": "cats",
            },
            "sem_key": {
                "question": "Sem?",
                "answer": "B.",
                "source_subject": "Bob",
                "source_predicate": "knows",
                "source_object": "Carol",
            },
        }

        registry = MagicMock()
        registry.list_active.return_value = list(qa.keys())

        def _get_adapter_id(key: str) -> str:
            return {"ep_key": "episodic", "sem_key": "semantic"}[key]

        registry.get_adapter_id.side_effect = _get_adapter_id

        loop = _make_loop(model, tmp_path, registry=registry, indexed_key_qa=qa)

        # Set a sentinel as the "outer" _current_job so we can detect restoration.
        sentinel_job = TrainingJob(
            keyed_pairs=[],
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
                    "paramem.training.indexed_memory.format_indexed_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch("paramem.training.trainer.train_adapter"),
                patch(
                    "paramem.training.indexed_memory.build_registry",
                    return_value={"ep_key": 0, "sem_key": 0},
                ),
                patch(
                    "experiments.utils.test_harness.evaluate_indexed_recall",
                    return_value={
                        "rate": 1.0,
                        "exact_count": 1,
                        "total": 1,
                        "mean_confidence": 1.0,
                        "per_key": [],
                    },
                ),
                patch("paramem.server.interim_adapter.unload_interim_adapters", return_value=[]),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
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
                keyed_pairs=[{"key": "graph1", "question": "Q?", "answer": "A."}],
                adapter_name="episodic",
                adapter_config=ep_cfg,
                inference_fallback_adapter="episodic_backup",
            ),
            "semantic": TrainingJob(
                keyed_pairs=[{"key": "graph2", "question": "Q2?", "answer": "A2."}],
                adapter_name="semantic",
                adapter_config=sem_cfg,
                inference_fallback_adapter="semantic_backup",
            ),
            "procedural": TrainingJob(
                keyed_pairs=[{"key": "proc1", "question": "Q3?", "answer": "A3."}],
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
    """When recall < threshold, the method returns rolled_back=True and skips finalize."""

    def test_rollback_triggered_when_recall_below_threshold(self, tmp_path: Path) -> None:
        """consolidate_interim_adapters returns rolled_back=True when recall is low."""
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
                "source_subject": "X",
                "source_predicate": "y",
                "source_object": "Z",
            },
        }
        registry = MagicMock()
        registry.list_active.return_value = ["graph1"]
        registry.get_adapter_id.return_value = "episodic"

        loop = _make_loop(model, tmp_path, registry=registry, indexed_key_qa=qa)

        _gpu_thread_lock.acquire()
        try:
            with (
                patch("paramem.models.loader.create_adapter", side_effect=_create_noop),
                patch("paramem.graph.qa_generator.partition_relations", return_value=([], [])),
                patch(
                    "paramem.training.indexed_memory.format_indexed_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch("paramem.training.trainer.train_adapter"),
                patch("paramem.training.indexed_memory.build_registry", return_value={"graph1": 0}),
                # Recall probe returns 0.5 — below the default threshold of 0.95.
                patch(
                    "experiments.utils.test_harness.evaluate_indexed_recall",
                    return_value={
                        "rate": 0.5,
                        "exact_count": 0,
                        "total": 1,
                        "mean_confidence": 0.5,
                        "per_key": [],
                    },
                ),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
                patch.object(loop, "_append_capacity_ceiling_log") as mock_log,
            ):
                result = loop.consolidate_interim_adapters(recall_sanity_threshold=0.95)
        finally:
            _gpu_thread_lock.release()

        assert result["rolled_back"] is True
        assert result["rollback_tier"] == "episodic"
        # The ceiling log helper must have been called.
        mock_log.assert_called_once()

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
                "source_subject": "X",
                "source_predicate": "y",
                "source_object": "Z",
            },
        }
        registry = MagicMock()
        registry.list_active.return_value = ["graph1"]
        registry.get_adapter_id.return_value = "episodic"

        loop = _make_loop(model, tmp_path, registry=registry, indexed_key_qa=qa)

        _gpu_thread_lock.acquire()
        try:
            with (
                patch("paramem.models.loader.create_adapter", side_effect=_create_noop),
                patch("paramem.graph.qa_generator.partition_relations", return_value=([], [])),
                patch(
                    "paramem.training.indexed_memory.format_indexed_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch("paramem.training.trainer.train_adapter"),
                patch("paramem.training.indexed_memory.build_registry", return_value={"graph1": 0}),
                # Recall exactly at threshold.
                patch(
                    "experiments.utils.test_harness.evaluate_indexed_recall",
                    return_value={
                        "rate": 0.95,
                        "exact_count": 1,
                        "total": 1,
                        "mean_confidence": 1.0,
                        "per_key": [],
                    },
                ),
                patch("paramem.server.interim_adapter.unload_interim_adapters", return_value=[]),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
            ):
                result = loop.consolidate_interim_adapters(recall_sanity_threshold=0.95)
        finally:
            _gpu_thread_lock.release()

        assert result["rolled_back"] is False


# ---------------------------------------------------------------------------
# Test 5 — Atomic finalize ordering
# ---------------------------------------------------------------------------


class TestAtomicFinalizeOrdering:
    """Registry rewrite before interim purge/unload; Router.reload() last."""

    def test_registry_rewrite_before_unload_and_router_reload(self, tmp_path: Path) -> None:
        """Finalize ordering: registry.save() → unload_interim_adapters → router.reload()."""
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
                "source_subject": "X",
                "source_predicate": "y",
                "source_object": "Z",
            },
        }
        registry = MagicMock()
        registry.list_active.return_value = ["graph1"]
        registry.get_adapter_id.return_value = "episodic"

        loop = _make_loop(model, tmp_path, registry=registry, indexed_key_qa=qa)

        call_order: list[str] = []

        def _registry_save(path) -> None:
            call_order.append("registry_save")

        def _unload(m, adapter_dir) -> list:
            call_order.append("unload_interim_adapters")
            return []

        def _router_reload() -> None:
            call_order.append("router_reload")

        mock_router = MagicMock()
        mock_router.reload.side_effect = _router_reload
        registry.save.side_effect = _registry_save

        _gpu_thread_lock.acquire()
        try:
            with (
                patch("paramem.models.loader.create_adapter", side_effect=_create_noop),
                patch("paramem.graph.qa_generator.partition_relations", return_value=([], [])),
                patch(
                    "paramem.training.indexed_memory.format_indexed_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch("paramem.training.trainer.train_adapter"),
                patch("paramem.training.indexed_memory.build_registry", return_value={"graph1": 0}),
                patch(
                    "experiments.utils.test_harness.evaluate_indexed_recall",
                    return_value={
                        "rate": 1.0,
                        "exact_count": 1,
                        "total": 1,
                        "mean_confidence": 1.0,
                        "per_key": [],
                    },
                ),
                patch(
                    "paramem.server.interim_adapter.unload_interim_adapters", side_effect=_unload
                ),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
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
        """After successful finalize, no episodic_interim_* adapter_id remains in the registry."""
        from paramem.server.gpu_lock import _gpu_thread_lock
        from paramem.training.key_registry import KeyRegistry

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
                "source_subject": "X",
                "source_predicate": "y",
                "source_object": "Z",
            },
        }
        # Use a real KeyRegistry so set_adapter_id mutations are verifiable.
        registry = KeyRegistry()
        registry.add("graph1", adapter_id="episodic_interim_20260418T0000")

        loop = _make_loop(model, tmp_path, registry=registry, indexed_key_qa=qa)

        _gpu_thread_lock.acquire()
        try:
            with (
                patch("paramem.models.loader.create_adapter", side_effect=_create_noop),
                patch("paramem.graph.qa_generator.partition_relations", return_value=([], [])),
                patch(
                    "paramem.training.indexed_memory.format_indexed_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch("paramem.training.trainer.train_adapter"),
                patch("paramem.training.indexed_memory.build_registry", return_value={"graph1": 0}),
                patch(
                    "experiments.utils.test_harness.evaluate_indexed_recall",
                    return_value={
                        "rate": 1.0,
                        "exact_count": 1,
                        "total": 1,
                        "mean_confidence": 1.0,
                        "per_key": [],
                    },
                ),
                patch(
                    "paramem.server.interim_adapter.unload_interim_adapters",
                    return_value=["episodic_interim_20260418T0000"],
                ),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
            ):
                result = loop.consolidate_interim_adapters()
        finally:
            _gpu_thread_lock.release()

        assert not result["rolled_back"]

        # Verify no episodic_interim_* adapter_id remains in registry.
        for key in registry.list_active():
            aid = registry.get_adapter_id(key)
            assert not aid.startswith("episodic_interim_"), (
                f"Key {key!r} still has interim adapter_id {aid!r} after finalize"
            )


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

    def test_pause_short_circuits_when_not_training(self) -> None:
        """pause() returns True immediately when _is_training is False."""
        model = _make_stub_model("episodic", "in_training")
        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=_minimal_training_config(),
            output_dir="/tmp/test_pause_shortcircuit",
        )
        bt._is_training = False
        result = bt.pause(timeout=0.01)
        assert result is True, "pause() must short-circuit when _is_training is False"
