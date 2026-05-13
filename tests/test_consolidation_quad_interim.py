"""Unit tests for ConsolidationLoop interim/finalize paths in quad mode.

Covers:
- _train_extracted_into_interim with indexed_format="quad": uses
  format_quadruple_training (not format_indexed_training) and writes
  a 6-field keyed_pairs.json via write_keyed_pairs_quad.
- consolidate_interim_adapters with indexed_format="quad": tier_keyed
  entries take the {key, subject, predicate, object} quad branch (the
  :3313-3319 branch in the plan), no KeyError when indexed_key_cache holds
  uniform-shape entries, and format_quadruple_training is called per tier.

No GPU required — model interactions replaced with MagicMock stubs and
all heavy operations are patched out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_training_config() -> TrainingConfig:
    return TrainingConfig(
        num_epochs=1,
        gradient_checkpointing=False,
        batch_size=1,
    )


def _minimal_adapter_config() -> AdapterConfig:
    return AdapterConfig(rank=4, alpha=8, learning_rate=1e-4, target_modules=["q_proj"])


def _make_stub_model(*adapter_names: str) -> MagicMock:
    """Return a MagicMock that behaves like a minimal PeftModel."""
    model = MagicMock()
    model.peft_config = {name: MagicMock() for name in adapter_names}

    def _delete_adapter(name: str) -> None:
        model.peft_config.pop(name, None)

    model.delete_adapter.side_effect = _delete_adapter
    return model


def _make_loop(tmp_path: Path, *, indexed_format: str = "qa") -> Any:
    """Construct a bare ConsolidationLoop without calling __init__."""
    from paramem.training.consolidation import ConsolidationLoop
    from paramem.training.key_registry import KeyRegistry

    model = _make_stub_model("episodic", "semantic", "procedural", "in_training")

    loop = ConsolidationLoop.__new__(ConsolidationLoop)
    loop.model = model
    loop.tokenizer = MagicMock()
    cfg = ConsolidationConfig(indexed_key_replay_enabled=True, indexed_format=indexed_format)
    loop.config = cfg
    loop._indexed_format = indexed_format
    loop._is_quad = indexed_format == "quad"
    loop.training_config = _minimal_training_config()
    loop.episodic_config = _minimal_adapter_config()
    loop.semantic_config = _minimal_adapter_config()
    loop.procedural_config = None
    loop.wandb_config = None
    loop.output_dir = tmp_path
    loop.snapshot_dir = None
    loop.save_cycle_snapshots = False
    loop.persist_graph = False
    loop._thermal_policy = None
    loop.indexed_key_registry = KeyRegistry()
    loop.indexed_key_cache: dict[str, Any] = {}
    loop._indexed_next_index = 1
    loop._procedural_next_index = 1
    loop.episodic_simhash: dict = {}
    loop.semantic_simhash: dict = {}
    loop.procedural_simhash: dict = {}
    loop.procedural_sp_index: dict = {}
    loop.cycle_count = 1
    loop.key_sessions: dict = {}
    loop.promoted_keys: set = set()
    loop.pending_interim_triples: list = []
    loop.shutdown_requested = False
    loop.merger = MagicMock()
    loop.merger.graph = MagicMock(relations=[])
    loop.graph_enrichment_enabled = False
    loop.graph_enrichment_interim_enabled = False
    loop.graph_enrichment_min_triples_floor = 20
    loop._triples_since_last_enrichment = 0
    loop.full_consolidation_period_string = ""
    loop.fingerprint_cache = None
    return loop


def _fake_quad_rels(n: int = 2) -> list[dict]:
    """Return n synthetic relation dicts (quad-format input for episodic distillation)."""
    return [
        {
            "subject": f"Subject{i}",
            "predicate": "knows",
            "object": f"Object{i}",
            "relation_type": "factual",
            "speaker_id": "Speaker0",
        }
        for i in range(1, n + 1)
    ]


def _fake_qa_rels(n: int = 2) -> list[dict]:
    """Return n synthetic QA dicts (QA-format input)."""
    return [
        {
            "question": f"What is fact {i}?",
            "answer": f"Fact {i} answer.",
            "source_subject": f"Subject{i}",
            "source_predicate": "knows",
            "source_object": f"Object{i}",
        }
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# Tests: _train_extracted_into_interim — quad mode format selection
# ---------------------------------------------------------------------------


class TestTrainExtractedIntoInterimQuadFormat:
    """_train_extracted_into_interim in quad mode uses format_quadruple_training."""

    def test_quad_mode_calls_format_quadruple_training(self, tmp_path: Path) -> None:
        """format_quadruple_training called (not format_indexed_training) in quad mode."""
        loop = _make_loop(tmp_path, indexed_format="quad")
        stamp = "20260601T1200"
        adapter_name = f"episodic_interim_{stamp}"
        loop.model.peft_config[adapter_name] = MagicMock()

        _ret = [{"input_ids": [1], "labels": [1], "attention_mask": [1]}]
        quad_fmt_mock = MagicMock(return_value=_ret)
        qa_fmt_mock = MagicMock(return_value=_ret)

        def _create_side(m, cfg, s):
            m.peft_config[f"episodic_interim_{s}"] = MagicMock()
            return m

        with (
            patch(
                "paramem.server.interim_adapter.create_interim_adapter",
                side_effect=_create_side,
            ),
            patch("paramem.training.trainer.train_adapter"),
            patch("paramem.training.consolidation.format_quadruple_training", quad_fmt_mock),
            patch("paramem.training.consolidation.format_indexed_training", qa_fmt_mock),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry_quad", return_value={}),
            patch("paramem.training.consolidation.save_registry"),
            patch("paramem.models.loader.save_adapter"),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=None),
        ):
            result = loop._train_extracted_into_interim(
                episodic_qa=_fake_quad_rels(2),
                procedural_rels=[],
                run_label="test-quad",
                speaker_id="Speaker0",
                stamp=stamp,
            )

        # format_quadruple_training must have been called; format_indexed_training must NOT.
        quad_fmt_mock.assert_called_once()
        qa_fmt_mock.assert_not_called()
        assert result["mode"] in ("trained", "noop") or result.get("error") is None

    def test_qa_mode_calls_format_indexed_training(self, tmp_path: Path) -> None:
        """In QA mode format_indexed_training is called (not quadruple).

        _train_extracted_into_interim resolves _format_training via a local
        re-import from paramem.training.indexed_memory, so the correct patch
        target is paramem.training.indexed_memory.format_indexed_training.
        """
        loop = _make_loop(tmp_path, indexed_format="qa")
        stamp = "20260601T1200"

        _ret2 = [{"input_ids": [1], "labels": [1], "attention_mask": [1]}]
        quad_fmt_mock = MagicMock(return_value=_ret2)
        qa_fmt_mock = MagicMock(return_value=_ret2)

        def _create_side(m, cfg, s):
            m.peft_config[f"episodic_interim_{s}"] = MagicMock()
            return m

        with (
            patch(
                "paramem.server.interim_adapter.create_interim_adapter",
                side_effect=_create_side,
            ),
            patch("paramem.training.trainer.train_adapter"),
            patch("paramem.training.consolidation.format_quadruple_training", quad_fmt_mock),
            # QA path uses a local re-import from indexed_memory — patch the source module.
            patch("paramem.training.indexed_memory.format_indexed_training", qa_fmt_mock),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            # QA path re-imports build_registry from indexed_memory too.
            patch("paramem.training.indexed_memory.build_registry", return_value={}),
            patch("paramem.training.consolidation.save_registry"),
            patch("paramem.models.loader.save_adapter"),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=None),
        ):
            loop._train_extracted_into_interim(
                episodic_qa=_fake_qa_rels(2),
                procedural_rels=[],
                run_label="test-qa",
                speaker_id="Speaker0",
                stamp=stamp,
            )

        qa_fmt_mock.assert_called_once()
        quad_fmt_mock.assert_not_called()

    def test_quad_mode_writes_quad_keyed_pairs(self, tmp_path: Path) -> None:
        """write_keyed_pairs_quad (not write_keyed_pairs) called in quad mode.

        _train_extracted_into_interim captures write_keyed_pairs_quad as _wkp_fn
        from the module-level import in consolidation.py (line 49).  Patch the
        module attribute so the local alias picks up the mock.
        """
        loop = _make_loop(tmp_path, indexed_format="quad")
        stamp = "20260601T1200"

        wkp_quad_mock = MagicMock()
        wkp_qa_mock = MagicMock()

        def _create_side(m, cfg, s):
            m.peft_config[f"episodic_interim_{s}"] = MagicMock()
            return m

        with (
            patch(
                "paramem.server.interim_adapter.create_interim_adapter",
                side_effect=_create_side,
            ),
            patch("paramem.training.trainer.train_adapter"),
            patch(
                "paramem.training.consolidation.format_quadruple_training",
                return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
            ),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry_quad", return_value={}),
            patch("paramem.training.consolidation.save_registry"),
            patch("paramem.models.loader.save_adapter"),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=None),
            # Patch the module-level binding in consolidation.py (line 49 import).
            # _train_extracted_into_interim captures it as _wkp_fn = write_keyed_pairs_quad.
            patch("paramem.training.consolidation.write_keyed_pairs_quad", wkp_quad_mock),
            # Also patch keyed_pairs_io directly as a safety net.
            patch("paramem.training.keyed_pairs_io.write_keyed_pairs", wkp_qa_mock),
        ):
            loop._train_extracted_into_interim(
                episodic_qa=_fake_quad_rels(2),
                procedural_rels=[],
                run_label="test-quad-kp",
                speaker_id="Speaker0",
                stamp=stamp,
            )

        wkp_quad_mock.assert_called()
        wkp_qa_mock.assert_not_called()

    def test_zero_facts_returns_noop(self, tmp_path: Path) -> None:
        """Empty episodic_qa returns mode='noop' regardless of format."""
        for fmt in ("qa", "quad"):
            loop = _make_loop(tmp_path, indexed_format=fmt)
            result = loop._train_extracted_into_interim(
                episodic_qa=[],
                procedural_rels=[],
                run_label="test-noop",
                speaker_id="Speaker0",
                stamp="20260601T1200",
            )
            assert result["mode"] == "noop", f"Expected noop for format={fmt}"
            assert result["triples_extracted"] == 0


# ---------------------------------------------------------------------------
# Tests: consolidate_interim_adapters — quad mode tier_keyed shape
# ---------------------------------------------------------------------------


class TestConsolidateInterimAdaptersQuadTierKeyed:
    """consolidate_interim_adapters builds quad-shaped tier_keyed in quad mode."""

    def _make_quad_indexed_key_qa(self) -> dict[str, dict]:
        """Return indexed_key_cache with quad-mode uniform-shape entries."""
        from paramem.training.consolidation import ConsolidationLoop

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        # Build entries via _cache_entry (same as production) to ensure uniform shape.
        loop._is_quad = True
        entry = loop._cache_entry(
            key="graph1",
            subject="Alice",
            predicate="likes",
            object="cats",
            speaker_id="Speaker0",
            first_seen_cycle=1,
        )
        return {"graph1": entry}

    def test_tier_keyed_quad_shape_no_key_error(self, tmp_path: Path) -> None:
        """In quad mode tier_keyed entries are {key, subject, predicate, object}.

        Specifically tests the :3313-3319 branch: a KeyError here means the
        branch incorrectly tried to read qa_info["question"]/["answer"].
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

        qa = self._make_quad_indexed_key_qa()

        registry = MagicMock()
        registry.list_active.return_value = ["graph1"]
        registry.get_adapter_id.return_value = "episodic"

        loop = _make_loop(tmp_path, indexed_format="quad")
        loop.model = model
        loop.indexed_key_registry = registry
        loop.indexed_key_cache = qa

        stub_trainer = MagicMock()
        stub_trainer._current_job = None
        stub_trainer._set_is_training = MagicMock()

        fmt_quad_mock = MagicMock(
            return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}]
        )
        fmt_qa_mock = MagicMock(
            return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}]
        )

        _gpu_thread_lock.acquire()
        try:
            with (
                patch("paramem.models.loader.create_adapter", return_value=model),
                patch("paramem.graph.qa_generator.partition_relations", return_value=([], [])),
                patch(
                    "paramem.training.consolidation.format_quadruple_training",
                    fmt_quad_mock,
                ),
                patch(
                    "paramem.training.consolidation.format_indexed_training",
                    fmt_qa_mock,
                ),
                patch("paramem.training.trainer.train_adapter"),
                patch(
                    "paramem.training.consolidation.build_registry_quad",
                    return_value={"graph1": 0},
                ),
                patch(
                    "paramem.training.consolidation.build_registry",
                    return_value={"graph1": 0},
                ),
                patch(
                    "experiments.utils.test_harness.evaluate_indexed_recall_quad",
                    return_value={
                        "rate": 1.0,
                        "exact_count": 1,
                        "total": 1,
                        "mean_confidence": 1.0,
                        "per_key": [],
                    },
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
                patch("paramem.training.consolidation.save_registry"),
                patch("paramem.models.loader.save_adapter"),
                patch("paramem.adapters.manifest.build_manifest_for", return_value=None),
                patch("paramem.training.keyed_pairs_io.write_keyed_pairs_quad"),
                patch("paramem.training.keyed_pairs_io.write_keyed_pairs"),
            ):
                # Must NOT raise KeyError — that would mean qa_info["question"] was accessed
                result = loop.consolidate_interim_adapters(trainer=stub_trainer)
        finally:
            _gpu_thread_lock.release()

        assert "rolled_back" in result
        # format_quadruple_training must be called (not format_indexed_training)
        fmt_quad_mock.assert_called()
        fmt_qa_mock.assert_not_called()

    def test_tier_keyed_qa_shape_in_qa_mode(self, tmp_path: Path) -> None:
        """QA mode unchanged: tier_keyed entries are {key, question, answer}.

        consolidate_interim_adapters uses a local re-import of format_indexed_training
        from paramem.training.indexed_memory, so the patch target is that module.
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

        # QA-mode cache entry (uniform shape includes subject/predicate/object too,
        # but also question/answer).
        qa = {
            "graph1": {
                "key": "graph1",
                "question": "Q1?",
                "answer": "A1.",
                "subject": "Alice",
                "predicate": "likes",
                "object": "cats",
                "source_subject": "Alice",
                "source_predicate": "likes",
                "source_object": "cats",
                "speaker_id": "Speaker0",
                "first_seen_cycle": 1,
            }
        }

        registry = MagicMock()
        registry.list_active.return_value = ["graph1"]
        registry.get_adapter_id.return_value = "episodic"

        loop = _make_loop(tmp_path, indexed_format="qa")
        loop.model = model
        loop.indexed_key_registry = registry
        loop.indexed_key_cache = qa

        stub_trainer = MagicMock()
        stub_trainer._current_job = None
        stub_trainer._set_is_training = MagicMock()

        fmt_qa_mock = MagicMock(
            return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}]
        )
        fmt_quad_mock = MagicMock(
            return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}]
        )

        _gpu_thread_lock.acquire()
        try:
            with (
                patch("paramem.models.loader.create_adapter", return_value=model),
                patch("paramem.graph.qa_generator.partition_relations", return_value=([], [])),
                # consolidate_interim_adapters uses a local re-import from indexed_memory.
                patch("paramem.training.indexed_memory.format_indexed_training", fmt_qa_mock),
                patch(
                    "paramem.training.consolidation.format_quadruple_training",
                    fmt_quad_mock,
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
                patch("paramem.training.consolidation.save_registry"),
                patch("paramem.models.loader.save_adapter"),
                patch("paramem.adapters.manifest.build_manifest_for", return_value=None),
                patch("paramem.training.keyed_pairs_io.write_keyed_pairs"),
            ):
                result = loop.consolidate_interim_adapters(trainer=stub_trainer)
        finally:
            _gpu_thread_lock.release()

        assert "rolled_back" in result
        fmt_qa_mock.assert_called()
        fmt_quad_mock.assert_not_called()
