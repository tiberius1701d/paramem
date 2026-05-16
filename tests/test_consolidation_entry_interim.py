"""Unit tests for ConsolidationLoop interim/finalize paths.

Covers:
- _train_extracted_into_interim: uses format_entry_training; no entries.json
  sidecar is written (knowledge lives in weights / cache only).
- consolidate_interim_adapters: tier_keyed entries take the
  {key, subject, predicate, object} branch; no KeyError when
  indexed_key_cache holds uniform-shape entries; format_entry_training
  is called per tier.

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


def _make_loop(tmp_path: Path) -> Any:
    """Construct a bare ConsolidationLoop without calling __init__."""
    from paramem.training.consolidation import ConsolidationLoop
    from paramem.training.key_registry import KeyRegistry

    model = _make_stub_model("episodic", "semantic", "procedural", "in_training")

    loop = ConsolidationLoop.__new__(ConsolidationLoop)
    loop.model = model
    loop.tokenizer = MagicMock()
    cfg = ConsolidationConfig(indexed_key_replay_enabled=True)
    loop.config = cfg
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
    from paramem.training.memory_store import MemoryStore as _MS

    loop.store = _MS(replay_enabled=True)
    for _tier in ("episodic", "semantic", "procedural"):
        loop.store.load_registry(_tier, KeyRegistry())
    loop.procedural_sp_index: dict = {}
    loop.cycle_count = 1
    loop._indexed_next_index = 1
    loop._procedural_next_index = 1
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


def _fake_entry_rels(n: int = 2) -> list[dict]:
    """Return n synthetic relation dicts for episodic distillation."""
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
    """Return n synthetic dicts with question/answer fields (legacy; kept for reference)."""
    return [
        {
            "question": f"What is fact {i}?",
            "answer": f"Fact {i} answer.",
            "subject": f"Subject{i}",
            "predicate": "knows",
            "object": f"Object{i}",
        }
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# Tests: _train_extracted_into_interim — format selection
# ---------------------------------------------------------------------------


class TestTrainExtractedIntoInterimFormat:
    """_train_extracted_into_interim uses format_entry_training."""

    def test_calls_format_entry_training(self, tmp_path: Path) -> None:
        """format_entry_training is called."""
        loop = _make_loop(tmp_path)
        stamp = "20260601T1200"
        adapter_name = f"episodic_interim_{stamp}"
        loop.model.peft_config[adapter_name] = MagicMock()

        _ret = [{"input_ids": [1], "labels": [1], "attention_mask": [1]}]
        fmt_mock = MagicMock(return_value=_ret)

        def _create_side(m, cfg, s):
            m.peft_config[f"episodic_interim_{s}"] = MagicMock()
            return m

        with (
            patch(
                "paramem.server.interim_adapter.create_interim_adapter",
                side_effect=_create_side,
            ),
            patch("paramem.training.trainer.train_adapter"),
            patch("paramem.training.consolidation.format_entry_training", fmt_mock),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.training.consolidation.save_registry"),
            patch("paramem.models.loader.save_adapter"),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=None),
        ):
            result = loop._train_extracted_into_interim(
                episodic_rels=_fake_entry_rels(2),
                procedural_rels=[],
                run_label="test-entry",
                speaker_id="Speaker0",
                stamp=stamp,
            )

        # format_entry_training must have been called.
        fmt_mock.assert_called_once()
        assert result["mode"] in ("trained", "noop") or result.get("error") is None

    def test_no_keyed_pairs_disk_write(self, tmp_path: Path) -> None:
        """No entries sidecar is written to disk.

        Knowledge lives in the adapter weights and indexed_key_cache.
        format_entry_training is called and the result goes to train_adapter;
        no disk writer for entries is invoked.
        """
        loop = _make_loop(tmp_path)
        stamp = "20260601T1200"

        fmt_mock2 = MagicMock(
            return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}]
        )

        def _create_side(m, cfg, s):
            m.peft_config[f"episodic_interim_{s}"] = MagicMock()
            return m

        with (
            patch(
                "paramem.server.interim_adapter.create_interim_adapter",
                side_effect=_create_side,
            ),
            patch("paramem.training.trainer.train_adapter"),
            patch("paramem.training.consolidation.format_entry_training", fmt_mock2),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.training.consolidation.save_registry"),
            patch("paramem.models.loader.save_adapter"),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=None),
        ):
            loop._train_extracted_into_interim(
                episodic_rels=_fake_entry_rels(2),
                procedural_rels=[],
                run_label="test-entry-kp",
                speaker_id="Speaker0",
                stamp=stamp,
            )

        # Training formatter was called; no entries file should appear on disk.
        fmt_mock2.assert_called()
        keyed_pairs_files = list(tmp_path.rglob("entries*.json")) + list(
            tmp_path.rglob("quads*.json")
        )
        assert not keyed_pairs_files, f"Unexpected entries files on disk: {keyed_pairs_files}"

    def test_zero_facts_returns_noop(self, tmp_path: Path) -> None:
        """Empty episodic_rels returns mode='noop'."""
        loop = _make_loop(tmp_path)
        result = loop._train_extracted_into_interim(
            episodic_rels=[],
            procedural_rels=[],
            run_label="test-noop",
            speaker_id="Speaker0",
            stamp="20260601T1200",
        )
        assert result["mode"] == "noop"
        assert result["triples_extracted"] == 0


# ---------------------------------------------------------------------------
# Tests: consolidate_interim_adapters — tier_keyed shape
# ---------------------------------------------------------------------------


class TestConsolidateInterimAdaptersTierKeyed:
    """consolidate_interim_adapters builds entry-shaped tier_keyed dicts."""

    def _make_quad_indexed_key_qa(self) -> dict[str, dict]:
        """Return indexed_key_cache with uniform-shape entries."""
        from paramem.training.consolidation import ConsolidationLoop

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        # Build entries via _cache_entry (same as production) to ensure uniform shape.
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
        """tier_keyed entries are {key, subject, predicate, object}; no KeyError."""
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

        # Use a real per-tier registry dict with "graph1" in the episodic tier.
        from paramem.training.key_registry import KeyRegistry as _KeyRegistry

        registry_dict = {
            "episodic": _KeyRegistry(),
            "semantic": _KeyRegistry(),
            "procedural": _KeyRegistry(),
        }
        registry_dict["episodic"].add("graph1")

        loop = _make_loop(tmp_path)
        loop.model = model
        loop.indexed_key_registry = registry_dict
        loop.indexed_key_cache = qa

        stub_trainer = MagicMock()
        stub_trainer._current_job = None
        stub_trainer._set_is_training = MagicMock()

        fmt_quad_mock = MagicMock(
            return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}]
        )

        _gpu_thread_lock.acquire()
        try:
            with (
                patch("paramem.models.loader.create_adapter", return_value=model),
                patch("paramem.graph.qa_generator.partition_relations", return_value=([], [])),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    fmt_quad_mock,
                ),
                patch("paramem.training.trainer.train_adapter"),
                patch(
                    "paramem.training.consolidation.build_registry",
                    return_value={"graph1": 0},
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
            ):
                # Must NOT raise KeyError on entry fields.
                result = loop.consolidate_interim_adapters(trainer=stub_trainer)
        finally:
            _gpu_thread_lock.release()

        assert "rolled_back" in result
        # format_entry_training must be called.
        fmt_quad_mock.assert_called()
