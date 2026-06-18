"""Unit tests for ConsolidationLoop.post_session_train.

Pure-Python, no GPU required.  All heavy dependencies (extraction, training,
adapter creation) are replaced with MagicMock objects so each test executes in
milliseconds and verifies only the orchestration logic of post_session_train.

Tests cover the post_session_train orchestration logic:
  1. First call creates the interim adapter and returns mode="trained".
  2. Second call within the same sub-interval reuses the adapter.
  3. Stamp rollover creates a new adapter without touching the previous one.
  4. Zero facts extracted → mode="noop", no adapter created.
  5. Training failure → registry unchanged, no adapter saved.
  6. max_interim_count=0 → mode="queued", triples in pending_interim_triples.
  7. Queued facts are present in pending_interim_triples.
  8. Keys registered only AFTER training returns, not before.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_registry_keys(loop) -> int:
    """Return total active-key count across all tier registries.

    ``loop.indexed_key_registry`` is now ``dict[str, KeyRegistry]``.  This
    helper replaces the old ``len(loop.indexed_key_registry)`` pattern which
    counted tiers, not keys.
    """
    reg = loop.indexed_key_registry
    if reg is None:
        return 0
    return sum(len(r) for r in reg.values())


def _make_mock_loop(tmp_path: Path, *, adapter_names: list[str] | None = None):
    """Return a minimal ConsolidationLoop-like object for unit testing.

    We build a plain object (not a MagicMock) so we can attach real attribute
    behaviour while patching the methods we want to control.
    """
    from paramem.training.consolidation import ConsolidationLoop
    from paramem.training.key_registry import KeyRegistry

    if adapter_names is None:
        adapter_names = ["episodic", "semantic", "procedural", "in_training"]

    # Minimal mock model whose peft_config behaves like a dict.
    model = MagicMock()
    model.peft_config = {name: MagicMock() for name in adapter_names}

    tokenizer = MagicMock()

    # Build a minimal ConsolidationConfig and TrainingConfig.
    from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

    cons_config = ConsolidationConfig(
        indexed_key_replay_enabled=True,
    )
    training_config = TrainingConfig(
        num_epochs=1,
        gradient_checkpointing=False,
    )
    ep_cfg = AdapterConfig(rank=4, alpha=8, learning_rate=1e-4, target_modules=["q_proj"])
    sem_cfg = AdapterConfig(rank=4, alpha=8, learning_rate=1e-5, target_modules=["q_proj"])

    # Use object.__setattr__ dance to avoid __init__ running (it loads models).
    # Instead, construct via __new__ and set attributes manually.
    loop = object.__new__(ConsolidationLoop)
    loop.model = model
    loop.tokenizer = tokenizer
    loop.config = cons_config
    loop.training_config = training_config
    loop.episodic_config = ep_cfg
    loop.semantic_config = sem_cfg
    loop.procedural_config = None
    loop.wandb_config = None
    loop.output_dir = tmp_path
    loop.snapshot_dir = None
    loop.save_cycle_snapshots = False
    loop._debug_base = None
    from paramem.graph.extraction_pipeline import ExtractionConfig, ExtractionPipeline

    loop.extraction = ExtractionPipeline(
        model=model,
        tokenizer=tokenizer,
        config=ExtractionConfig(
            temperature=0.0,
            max_tokens=256,
            stt_correction=False,
            ha_validation=False,
            noise_filter="",
            noise_filter_model="",
            noise_filter_endpoint=None,
            ner_check=False,
            ner_model="en_core_web_sm",
            plausibility_judge="off",
            plausibility_stage="deanon",
            verify_anonymization=False,
        ),
        prompts_dir=None,
    )
    # indexed_key_registry is now dict[str, KeyRegistry] (per-tier).
    loop.indexed_key_registry = {
        "episodic": KeyRegistry(),
        "semantic": KeyRegistry(),
        "procedural": KeyRegistry(),
    }
    loop.indexed_key_cache = {}
    loop._indexed_next_index = 1
    loop._procedural_next_index = 1
    loop.store.replace_simhashes_in_tier("episodic", {})
    loop.store.replace_simhashes_in_tier("semantic", {})
    loop.store.replace_simhashes_in_tier("procedural", {})
    loop.cycle_count = 0
    loop.promoted_keys = set()
    loop.pending_interim_triples = []
    loop.shutdown_requested = False
    loop._thermal_policy = None
    loop.merger = MagicMock()
    # B3: _build_all_edge_entries_into reads merger.graph.edges(data=True).
    # Provide a real NetworkX MultiDiGraph with two keyless episodic edges so the
    # graph-walk mints keys and training is triggered in tests that expect it.
    # The _materialize_consolidation_graph stub below skips reset_graph(), so the
    # graph survives intact through the keyed-walk step.
    _real_graph = nx.MultiDiGraph()
    _real_graph.add_node("subject1", attributes={"name": "Subject1"})
    _real_graph.add_node("object1", attributes={"name": "Object1"})
    _real_graph.add_edge("subject1", "object1", predicate="knows", relation_type="factual")
    _real_graph.add_node("subject2", attributes={"name": "Subject2"})
    _real_graph.add_node("object2", attributes={"name": "Object2"})
    _real_graph.add_edge("subject2", "object2", predicate="knows", relation_type="factual")
    loop.merger.graph = _real_graph
    # Graph-enrichment knobs (Task #10). Hook fires inside the normal
    # fresh-interim branch; default to disabled for these unit tests so the
    # hook stays inert and we don't need to stub _run_graph_enrichment.
    loop.graph_enrichment_enabled = False
    loop.graph_enrichment_neighborhood_hops = 2
    loop.graph_enrichment_max_entities_per_pass = 50
    loop.graph_enrichment_interim_enabled = False
    loop.graph_enrichment_min_triples_floor = 20

    # Stub out the recall probe so tests with a MagicMock model do not
    # feed it into re.sub (which raises TypeError on non-string input).
    # These tests verify post_session_train orchestration, not recall gating;
    # the probe is covered separately in test_consolidation_recall_early_stop.py.
    loop._probe_passing_keys = lambda adapter_name, entries: {e["key"] for e in entries}

    # Stub out _materialize_consolidation_graph so the B1 materialize step
    # does not call reconstruct_graph / probe_entries on the MagicMock model.
    # B3: the stub skips merger.reset_graph() so loop.merger.graph retains the
    # pre-populated keyless edges for the graph-walk keying step.
    # The materialize diagnostic is covered in test_consolidation.py::TestMaterializeInterimB1.
    loop._materialize_consolidation_graph = lambda **kw: (set(), [])

    return loop


def _fake_qa(n: int = 2) -> list[dict]:
    """Return n synthetic QA dicts."""
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
# Test 1 — First call creates interim adapter and returns mode="trained"
# ---------------------------------------------------------------------------


class TestFirstCallCreatesInterimAdapter:
    def test_first_fact_creates_interim_adapter_and_returns_trained(self, tmp_path: Path) -> None:
        """First call of the sub-interval creates the adapter, mode='trained'."""
        loop = _make_mock_loop(tmp_path)
        stamp = "20260418T1430"
        adapter_name = f"episodic_interim_{stamp}"

        def _create_side_effect(m, cfg, s):  # noqa: ANN001
            m.peft_config[f"episodic_interim_{s}"] = MagicMock()
            return m

        with (
            patch.object(loop, "extract_session", return_value=(_fake_qa(2), [])),
            patch(
                "paramem.memory.interim_adapter.create_interim_adapter",
                side_effect=_create_side_effect,
            ),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.models.loader.save_adapter"),
        ):
            result = loop.post_session_train(
                "Hello world transcript",
                "conv-001",
                speaker_id="Speaker0",
                schedule="every 2h",
                max_interim_count=4,
                stamp=stamp,
            )

        assert result["mode"] == "trained"
        assert result["adapter_name"] == adapter_name
        assert result["error"] is None
        assert result["triples_extracted"] == 2

    def test_first_call_creates_adapter_in_peft_config(self, tmp_path: Path) -> None:
        """After first call the interim adapter name appears in model.peft_config."""
        loop = _make_mock_loop(tmp_path)
        stamp = "20260418T1430"

        def _create_side_effect(m, cfg, s):  # noqa: ANN001
            m.peft_config[f"episodic_interim_{s}"] = MagicMock()
            return m

        with (
            patch.object(loop, "extract_session", return_value=(_fake_qa(1), [])),
            patch(
                "paramem.memory.interim_adapter.create_interim_adapter",
                side_effect=_create_side_effect,
            ),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.models.loader.save_adapter"),
        ):
            loop.post_session_train(
                "Transcript",
                "conv-001",
                speaker_id="Speaker0",
                schedule="every 2h",
                max_interim_count=4,
                stamp=stamp,
            )

        assert f"episodic_interim_{stamp}" in loop.model.peft_config


# ---------------------------------------------------------------------------
# Test 2 — Second call within the same sub-interval reuses adapter
# ---------------------------------------------------------------------------


class TestSecondCallReusesAdapter:
    def test_second_call_does_not_call_create_interim_adapter_again(self, tmp_path: Path) -> None:
        """create_interim_adapter is only called once per sub-interval."""
        loop = _make_mock_loop(tmp_path)
        stamp = "20260418T1430"

        # Pre-populate peft_config to simulate first-call having already created it.
        loop.model.peft_config[f"episodic_interim_{stamp}"] = MagicMock()

        with (
            patch.object(loop, "extract_session", return_value=(_fake_qa(1), [])),
            patch("paramem.memory.interim_adapter.create_interim_adapter") as mock_create,
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.models.loader.save_adapter"),
        ):
            loop.post_session_train(
                "Second conversation",
                "conv-002",
                speaker_id="Speaker0",
                schedule="every 2h",
                max_interim_count=4,
                stamp=stamp,
            )

        mock_create.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3 — Stamp rollover creates a new adapter
# ---------------------------------------------------------------------------


class TestStampRolloverCreatesNewAdapter:
    def test_different_stamp_creates_new_adapter(self, tmp_path: Path) -> None:
        """When the stamp rolls over, a new adapter is created without touching the old one."""
        loop = _make_mock_loop(tmp_path)
        old_stamp = "20260418T1400"
        new_stamp = "20260418T1430"

        # Simulate the old adapter already existing.
        loop.model.peft_config[f"episodic_interim_{old_stamp}"] = MagicMock()

        created_adapters = []

        def _create_side_effect(m, cfg, s):  # noqa: ANN001
            m.peft_config[f"episodic_interim_{s}"] = MagicMock()
            created_adapters.append(s)
            return m

        with (
            patch.object(loop, "extract_session", return_value=(_fake_qa(1), [])),
            patch(
                "paramem.memory.interim_adapter.create_interim_adapter",
                side_effect=_create_side_effect,
            ) as mock_create,
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.models.loader.save_adapter"),
        ):
            result = loop.post_session_train(
                "Rollover transcript",
                "conv-003",
                speaker_id="Speaker0",
                schedule="every 2h",
                max_interim_count=4,
                stamp=new_stamp,
            )

        # Only the new adapter was created.
        assert mock_create.call_count == 1
        assert created_adapters == [new_stamp]
        assert result["adapter_name"] == f"episodic_interim_{new_stamp}"
        # Old adapter is untouched.
        assert f"episodic_interim_{old_stamp}" in loop.model.peft_config


# ---------------------------------------------------------------------------
# Test 4 — Zero facts extracted is a noop
# ---------------------------------------------------------------------------


class TestZeroFactsIsNoop:
    def test_zero_facts_returns_noop(self, tmp_path: Path) -> None:
        """When extraction returns 0 QA pairs, mode='noop' with no side effects."""
        loop = _make_mock_loop(tmp_path)

        with (
            patch.object(loop, "extract_session", return_value=([], [])),
            patch("paramem.memory.interim_adapter.create_interim_adapter") as mock_create,
        ):
            result = loop.post_session_train(
                "Empty transcript",
                "conv-004",
                speaker_id="Speaker0",
                schedule="every 2h",
                max_interim_count=4,
                stamp="20260418T1430",
            )

        assert result["mode"] == "noop"
        assert result["triples_extracted"] == 0
        assert result["new_keys"] == []
        assert result["adapter_name"] is None
        mock_create.assert_not_called()

    def test_zero_facts_does_not_mutate_registry(self, tmp_path: Path) -> None:
        """Registry is untouched when no facts are extracted."""
        loop = _make_mock_loop(tmp_path)
        initial_len = _count_registry_keys(loop)

        with patch.object(loop, "extract_session", return_value=([], [])):
            loop.post_session_train(
                "Empty",
                "conv-005",
                speaker_id="Speaker0",
                schedule="every 2h",
                max_interim_count=4,
                stamp="20260418T1430",
            )

        assert _count_registry_keys(loop) == initial_len


# ---------------------------------------------------------------------------
# Test 5 — Training failure keeps registry clean
# ---------------------------------------------------------------------------


class TestTrainingFailureKeepsRegistryClean:
    def test_training_exception_does_not_mutate_registry(self, tmp_path: Path) -> None:
        """When train_adapter raises, no keys are added to the registry."""
        loop = _make_mock_loop(tmp_path)
        stamp = "20260418T1430"

        def _raise(*args, **kwargs):
            raise RuntimeError("Simulated training failure")

        with (
            patch.object(loop, "extract_session", return_value=(_fake_qa(2), [])),
            patch(
                "paramem.memory.interim_adapter.create_interim_adapter",
                side_effect=lambda m, cfg, s: m,
            ),
            patch("paramem.training.trainer.train_adapter", side_effect=_raise),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
        ):
            loop.model.peft_config[f"episodic_interim_{stamp}"] = MagicMock()
            with pytest.raises(RuntimeError, match="Simulated training failure"):
                loop.post_session_train(
                    "Transcript",
                    "conv-006",
                    speaker_id="Speaker0",
                    schedule="every 2h",
                    max_interim_count=4,
                    stamp=stamp,
                )

        # Registry must be unchanged after the training error.
        assert _count_registry_keys(loop) == 0

    def test_training_failure_does_not_save_registry_to_disk(self, tmp_path: Path) -> None:
        """On training failure, no registry file is written to disk."""
        loop = _make_mock_loop(tmp_path)
        stamp = "20260418T1430"
        registry_path = tmp_path / "indexed_key_registry.json"

        def _raise(*args, **kwargs):
            raise RuntimeError("boom")

        with (
            patch.object(loop, "extract_session", return_value=(_fake_qa(1), [])),
            patch(
                "paramem.memory.interim_adapter.create_interim_adapter",
                side_effect=lambda m, cfg, s: m,
            ),
            patch("paramem.training.trainer.train_adapter", side_effect=_raise),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
        ):
            loop.model.peft_config[f"episodic_interim_{stamp}"] = MagicMock()
            with pytest.raises(RuntimeError):
                loop.post_session_train(
                    "Transcript",
                    "conv-007",
                    speaker_id="Speaker0",
                    schedule="every 2h",
                    max_interim_count=4,
                    stamp=stamp,
                )

        assert not registry_path.exists()


# ---------------------------------------------------------------------------
# Test 6 — max_interim_count == 0 → queue-until-consolidation
# ---------------------------------------------------------------------------


class TestMaxInterimCountZeroQueues:
    def test_zero_count_queues_triples_not_trains(self, tmp_path: Path) -> None:
        """max_interim_count=0 → triples land in pending_interim_triples, no training."""
        loop = _make_mock_loop(tmp_path)
        qa = _fake_qa(3)

        with (
            patch.object(loop, "extract_session", return_value=(qa, [])),
            patch("paramem.memory.interim_adapter.create_interim_adapter") as mock_create,
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ) as mock_train,
        ):
            result = loop.post_session_train(
                "Transcript",
                "conv-008",
                speaker_id="Speaker0",
                schedule="every 2h",
                max_interim_count=0,
            )

        assert result["mode"] == "queued"
        assert result["triples_extracted"] == 3
        assert result["new_keys"] == []
        assert result["adapter_name"] is None
        mock_create.assert_not_called()
        mock_train.assert_not_called()

    def test_zero_count_triples_in_pending_queue(self, tmp_path: Path) -> None:
        """Extracted triples accumulate in pending_interim_triples."""
        loop = _make_mock_loop(tmp_path)
        qa = _fake_qa(2)

        with patch.object(loop, "extract_session", return_value=(qa, [])):
            loop.post_session_train(
                "Transcript A",
                "conv-009a",
                speaker_id="Speaker0",
                schedule="every 2h",
                max_interim_count=0,
            )
            loop.post_session_train(
                "Transcript B",
                "conv-009b",
                speaker_id="Speaker0",
                schedule="every 2h",
                max_interim_count=0,
            )

        # Both extractions accumulated.
        assert len(loop.pending_interim_triples) == 4

    def test_zero_count_does_not_mutate_registry(self, tmp_path: Path) -> None:
        """max_interim_count=0 leaves the key registry untouched."""
        loop = _make_mock_loop(tmp_path)

        with patch.object(loop, "extract_session", return_value=(_fake_qa(2), [])):
            loop.post_session_train(
                "Transcript",
                "conv-010",
                speaker_id="Speaker0",
                schedule="every 2h",
                max_interim_count=0,
            )

        assert _count_registry_keys(loop) == 0


# ---------------------------------------------------------------------------
# Registry update happens AFTER training, not before
# ---------------------------------------------------------------------------


class TestRegisterAfterSuccessNotBefore:
    def test_registry_empty_until_training_completes(self, tmp_path: Path) -> None:
        """Keys are added to the registry only after train_adapter returns."""
        loop = _make_mock_loop(tmp_path)
        stamp = "20260418T1430"
        registry_before_training = []

        def _capture_registry_state(*args, **kwargs):
            """Record registry length at the moment training is called."""
            registry_before_training.append(_count_registry_keys(loop))

        with (
            patch.object(loop, "extract_session", return_value=(_fake_qa(2), [])),
            patch(
                "paramem.memory.interim_adapter.create_interim_adapter",
                side_effect=lambda m, cfg, s: m,
            ),
            patch(
                "paramem.training.trainer.train_adapter",
                side_effect=_capture_registry_state,
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.models.loader.save_adapter"),
        ):
            loop.model.peft_config[f"episodic_interim_{stamp}"] = MagicMock()
            loop.post_session_train(
                "Transcript",
                "conv-011",
                speaker_id="Speaker0",
                schedule="every 2h",
                max_interim_count=4,
                stamp=stamp,
            )

        # Registry must have been empty when train_adapter was called.
        assert registry_before_training == [0], (
            f"Expected 0 keys before training, got: {registry_before_training}"
        )

    def test_registry_populated_after_training_succeeds(self, tmp_path: Path) -> None:
        """After successful training, newly extracted keys appear in the registry."""
        loop = _make_mock_loop(tmp_path)
        stamp = "20260418T1430"

        with (
            patch.object(loop, "extract_session", return_value=(_fake_qa(2), [])),
            patch(
                "paramem.memory.interim_adapter.create_interim_adapter",
                side_effect=lambda m, cfg, s: m,
            ),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.models.loader.save_adapter"),
        ):
            loop.model.peft_config[f"episodic_interim_{stamp}"] = MagicMock()
            result = loop.post_session_train(
                "Transcript",
                "conv-012",
                speaker_id="Speaker0",
                schedule="every 2h",
                max_interim_count=4,
                stamp=stamp,
            )

        # Two facts extracted → two keys registered.
        # With per-tier dict API: each new key lives in the interim tier entry.
        interim_tier = f"episodic_interim_{stamp}"
        assert len(result["new_keys"]) == 2
        for k in result["new_keys"]:
            # Key must be in the interim tier's registry (tier = dict key).
            assert interim_tier in loop.indexed_key_registry, (
                f"Interim tier {interim_tier!r} missing from registry dict"
            )
            assert k in loop.store.registry(interim_tier), (
                f"Key {k!r} not found in interim tier {interim_tier!r}"
            )


# ---------------------------------------------------------------------------
# Helpers for call-order, cleanup, and roundtrip tests
# ---------------------------------------------------------------------------


def _fake_proc_rels(n: int = 2) -> list[dict]:
    """Return n synthetic procedural-relation dicts (as returned by extract_session)."""
    return [
        {
            "subject": f"Subject{i}",
            "predicate": "prefers",
            "object": f"Thing{i}",
            "relation_type": "preference",
        }
        for i in range(1, n + 1)
    ]


def _make_mock_loop_with_procedural(tmp_path: Path):
    """Like _make_mock_loop but with procedural_config set (enables procedural-routing)."""
    from paramem.utils.config import AdapterConfig

    loop = _make_mock_loop(tmp_path)
    proc_cfg = AdapterConfig(rank=4, alpha=8, learning_rate=1e-4, target_modules=["q_proj"])
    loop.procedural_config = proc_cfg
    return loop


# Shared patch list for a successful post_session_train call (no procedural).
_COMMON_PATCHES = [
    "paramem.memory.interim_adapter.create_interim_adapter",
    "paramem.training.trainer.train_adapter",
    "paramem.training.consolidation.format_entry_training",
    "paramem.models.loader.switch_adapter",
    "paramem.training.consolidation.build_registry",
    "paramem.models.loader.save_adapter",
]


# ---------------------------------------------------------------------------
# Test 9 — procedural_rels are routed to the procedural adapter
# ---------------------------------------------------------------------------


class TestProceduralRelsRoutedToProceduralAdapter:
    """Procedural relations must reach the procedural adapter, not be silently dropped."""

    def test_procedural_train_called_when_proc_rels_present(self, tmp_path: Path) -> None:
        """_run_indexed_key_procedural is called when procedural_rels are non-empty."""
        loop = _make_mock_loop_with_procedural(tmp_path)
        stamp = "20260418T1430"
        loop.model.peft_config[f"episodic_interim_{stamp}"] = MagicMock()

        with (
            patch.object(
                loop,
                "extract_session",
                return_value=(_fake_qa(2), _fake_proc_rels(1)),
            ),
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.models.loader.save_adapter"),
            patch.object(loop, "_run_indexed_key_procedural") as mock_proc,
        ):
            loop.post_session_train(
                "Transcript with prefs",
                "conv-p-001",
                speaker_id="Speaker0",
                schedule="every 2h",
                max_interim_count=4,
                stamp=stamp,
            )

        mock_proc.assert_called_once()
        # First positional arg is the procedural relations list.
        called_rels = mock_proc.call_args[0][0]
        assert len(called_rels) == 1

        # Procedural flush invariant: whenever the procedural manifest is stamped
        # (gate: procedural_config is not None and "procedural" in peft_config),
        # the per-tier indexed_key_registry.json must land on disk in the same
        # consolidation window so the manifest's registry_sha256 is verifiable
        # on restart.  Pre-fix this file was absent — procedural slot orphaned.
        proc_registry = loop.output_dir / "procedural" / "indexed_key_registry.json"
        assert proc_registry.exists(), (
            f"Procedural tier registry not written to {proc_registry} — "
            "step-8 symmetric flush regression (was absent before "
            "post_session_train Step 8 procedural flush was added)."
        )

    def test_procedural_train_not_called_when_no_proc_rels(self, tmp_path: Path) -> None:
        """_run_indexed_key_procedural is NOT called when procedural_rels is empty."""
        loop = _make_mock_loop_with_procedural(tmp_path)
        stamp = "20260418T1430"
        loop.model.peft_config[f"episodic_interim_{stamp}"] = MagicMock()

        with (
            patch.object(
                loop,
                "extract_session",
                return_value=(_fake_qa(2), []),  # empty procedural_rels
            ),
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.models.loader.save_adapter"),
            patch.object(loop, "_run_indexed_key_procedural") as mock_proc,
        ):
            loop.post_session_train(
                "Transcript no prefs",
                "conv-p-002",
                speaker_id="Speaker0",
                schedule="every 2h",
                max_interim_count=4,
                stamp=stamp,
            )

        mock_proc.assert_not_called()

    def test_procedural_train_not_called_when_procedural_config_is_none(
        self, tmp_path: Path
    ) -> None:
        """_run_indexed_key_procedural is NOT called when procedural_config is None."""
        loop = _make_mock_loop(tmp_path)  # procedural_config=None by default
        stamp = "20260418T1430"
        loop.model.peft_config[f"episodic_interim_{stamp}"] = MagicMock()

        with (
            patch.object(
                loop,
                "extract_session",
                return_value=(_fake_qa(2), _fake_proc_rels(2)),
            ),
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.models.loader.save_adapter"),
            patch.object(loop, "_run_indexed_key_procedural") as mock_proc,
        ):
            loop.post_session_train(
                "Transcript",
                "conv-p-003",
                speaker_id="Speaker0",
                schedule="every 2h",
                max_interim_count=4,
                stamp=stamp,
            )

        mock_proc.assert_not_called()

    def test_procedural_failure_leaves_registry_clean(self, tmp_path: Path) -> None:
        """train_adapter raising on the procedural pass leaves all state unchanged.

        This test does NOT mock ``_run_indexed_key_procedural`` — it exercises the
        actual mutation-ordering invariant inside that method.  ``train_adapter`` is
        mocked to succeed on the first (episodic) call and raise on the second
        (procedural) call.

        Note: per-session sp_index-driven retirement was removed (architecture §4.3).
        Contradiction resolution is deferred to full consolidation via the model-bearing
        merger.  The pre-existing proc0 key coexists with the new preference — it is
        NOT retired during this session.

        Pre-conditions
        --------------
        - One old procedural key ``proc0`` exists in the procedural simhash and
          indexed_key_cache.

        Post-conditions after the procedural training failure
        -----------------------------------------------------
        - ``indexed_key_registry`` is empty: episodic step 7 never ran.
        - ``indexed_key_cache`` does NOT contain any new procedural key.
        - ``procedural_simhash`` is unchanged: old key still present.
        - Old procedural key ``proc0`` was NOT removed from any index.
        """
        loop = _make_mock_loop_with_procedural(tmp_path)
        stamp = "20260418T1430"
        loop.model.peft_config[f"episodic_interim_{stamp}"] = MagicMock()

        # Pre-existing procedural state: proc0.
        # _fake_proc_rels(1) produces subject="Subject1", predicate="prefers".
        old_proc_key = "proc0"
        old_qa_entry = {
            "key": old_proc_key,
            "question": "Old question?",
            "answer": "Old answer.",
            "subject": "Subject1",
            "predicate": "prefers",
            "object": "OldThing",
            "speaker_id": "",
        }
        loop.store.put_simhash("procedural", old_proc_key, 0xDEADBEEF)
        loop.store._entries_flat_view()[old_proc_key] = old_qa_entry

        call_count = {"n": 0}

        def _train_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                # Second call is the procedural train_adapter — raise.
                raise RuntimeError("procedural training failed")
            return {"train_loss": 0.5, "aborted": False}

        # Synthetic QA for generate_qa_from_relations (called inside
        # _run_indexed_key_procedural).
        proc_qa_out = [
            {
                "question": "What does Subject1 prefer?",
                "answer": "NewThing1.",
                "subject": "Subject1",
                "predicate": "prefers",
                "object": "NewThing1",
            }
        ]

        with (
            patch.object(
                loop,
                "extract_session",
                return_value=(_fake_qa(2), _fake_proc_rels(1)),
            ),
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            # train_adapter is imported at module level in consolidation.py;
            # patch it there so both the episodic (_train_adapter local alias)
            # and procedural (module-level train_adapter) calls are intercepted.
            patch(
                "paramem.training.consolidation.train_adapter",
                side_effect=_train_side_effect,
            ),
            patch(
                "paramem.training.trainer.train_adapter",
                side_effect=_train_side_effect,
            ),
            patch(
                "paramem.graph.qa_generator.generate_qa_from_relations",
                return_value=proc_qa_out,
            ),
            patch(
                "paramem.training.consolidation.format_entry_training",
                return_value=[{}],
            ),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            # probe_entries: proc0 is probed (it is an existing key, not retired),
            # but fails reconstruction — falls back to cache read.
            patch(
                "paramem.training.consolidation.probe_entries",
                side_effect=lambda model, tokenizer, entries, **kw: (
                    (e, {"failure_reason": "no_match"}) for e in entries
                ),
            ),
            patch(
                "paramem.training.consolidation.build_registry",
                return_value={},
            ),
        ):
            with pytest.raises(RuntimeError, match="procedural training failed"):
                loop.post_session_train(
                    "Transcript",
                    "conv-p-004",
                    speaker_id="Speaker0",
                    schedule="every 2h",
                    max_interim_count=4,
                    stamp=stamp,
                )

        # Episodic key registration never ran — registry stays empty.
        assert _count_registry_keys(loop) == 0, (
            "Episodic keys must not be registered when procedural training fails."
        )

        # No new procedural key in indexed_key_cache — only the pre-existing old entry.
        proc_keys_in_qa = [k for k in loop.store._entries_flat_view() if k.startswith("proc")]
        assert proc_keys_in_qa == [old_proc_key], (
            f"Expected only old procedural key '{old_proc_key}' in indexed_key_cache; "
            f"found: {proc_keys_in_qa}"
        )

        # Old contradicted key must NOT have been retired from procedural_simhash.
        assert old_proc_key in loop.store.tier_simhashes("procedural", include_stale=False), (
            "Old procedural key must still be in procedural_simhash after training failure."
        )

        # Old key must NOT have been removed from indexed_key_cache.
        assert old_proc_key in loop.store._entries_flat_view(), (
            "Old procedural key must still be in indexed_key_cache after training failure."
        )

    def test_interim_stamp_cleared_after_procedural_failure(self, tmp_path: Path) -> None:
        """_current_interim_stamp is cleared even when procedural training fails."""
        loop = _make_mock_loop_with_procedural(tmp_path)
        stamp = "20260418T1430"
        loop.model.peft_config[f"episodic_interim_{stamp}"] = MagicMock()

        with (
            patch.object(
                loop,
                "extract_session",
                return_value=(_fake_qa(1), _fake_proc_rels(1)),
            ),
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch.object(
                loop,
                "_run_indexed_key_procedural",
                side_effect=RuntimeError("boom"),
            ),
        ):
            with pytest.raises(RuntimeError):
                loop.post_session_train(
                    "Transcript",
                    "conv-p-005",
                    speaker_id="Speaker0",
                    schedule="every 2h",
                    max_interim_count=4,
                    stamp=stamp,
                )

        # _current_interim_stamp must be cleaned up (finally block).
        assert getattr(loop, "_current_interim_stamp", None) is None


# ---------------------------------------------------------------------------
# Test 10 — output directory uniqueness across stamps and calls
# ---------------------------------------------------------------------------


class TestTrainingOutputDirUniqueness:
    """Consecutive post_session_train calls must not share an output directory."""

    def _run_and_capture_output_dirs(self, loop, *, stamp: str, session_id: str) -> list[str]:
        """Run post_session_train and return all output_dir paths passed to train_adapter."""
        captured: list[str] = []

        def _capture_train(*args, **kwargs):
            captured.append(str(kwargs.get("output_dir", "")))

        loop.model.peft_config[f"episodic_interim_{stamp}"] = MagicMock()

        with (
            patch.object(loop, "extract_session", return_value=(_fake_qa(1), [])),
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch("paramem.training.trainer.train_adapter", side_effect=_capture_train),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.models.loader.save_adapter"),
        ):
            loop.post_session_train(
                "Transcript",
                session_id,
                speaker_id="Speaker0",
                schedule="every 2h",
                max_interim_count=4,
                stamp=stamp,
            )

        return captured

    def test_output_dir_contains_interim_stamp(self, tmp_path: Path) -> None:
        """Training output dir must include the interim stamp, not cycle_count."""
        loop = _make_mock_loop(tmp_path)
        stamp = "20260418T1430"

        dirs = self._run_and_capture_output_dirs(loop, stamp=stamp, session_id="conv-d-001")

        assert len(dirs) == 1
        assert f"interim_{stamp}" in dirs[0], f"Expected interim_{stamp} in '{dirs[0]}'"
        # Must NOT contain cycle_0 (the initial cycle_count value).
        assert "cycle_0" not in dirs[0]

    def test_two_calls_same_stamp_use_same_dir(self, tmp_path: Path) -> None:
        """Two calls with the same stamp land in the same output directory.

        This is correct — within a sub-interval the adapter accumulates facts.
        Both calls write to interim_<stamp>/<adapter_name>/.
        """
        loop = _make_mock_loop(tmp_path)
        stamp = "20260418T1430"

        dirs_first = self._run_and_capture_output_dirs(loop, stamp=stamp, session_id="conv-d-002a")
        # Need to add the adapter to peft_config again (cleared between calls in the helper).
        dirs_second = self._run_and_capture_output_dirs(loop, stamp=stamp, session_id="conv-d-002b")

        assert dirs_first == dirs_second, (
            f"Same stamp must produce same output dir: {dirs_first} vs {dirs_second}"
        )

    def test_two_calls_different_stamps_use_different_dirs(self, tmp_path: Path) -> None:
        """Two calls with different stamps must produce different output directories."""
        loop = _make_mock_loop(tmp_path)
        stamp_a = "20260418T1400"
        stamp_b = "20260418T1430"

        dirs_a = self._run_and_capture_output_dirs(loop, stamp=stamp_a, session_id="conv-d-003a")
        dirs_b = self._run_and_capture_output_dirs(loop, stamp=stamp_b, session_id="conv-d-003b")

        assert dirs_a != dirs_b, (
            f"Different stamps must produce different output dirs: {dirs_a} vs {dirs_b}"
        )
        assert f"interim_{stamp_a}" in dirs_a[0]
        assert f"interim_{stamp_b}" in dirs_b[0]

    def test_procedural_output_dir_uses_interim_stamp(self, tmp_path: Path) -> None:
        """Procedural training output dir also uses the interim stamp, not cycle_count."""
        loop = _make_mock_loop_with_procedural(tmp_path)
        stamp = "20260418T1430"
        loop.model.peft_config[f"episodic_interim_{stamp}"] = MagicMock()

        captured_dirs: list[str] = []

        def _capture_train(*args, **kwargs):
            captured_dirs.append(str(kwargs.get("output_dir", "")))

        with (
            patch.object(
                loop,
                "extract_session",
                return_value=(_fake_qa(2), _fake_proc_rels(1)),
            ),
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch("paramem.training.trainer.train_adapter", side_effect=_capture_train),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.models.loader.save_adapter"),
            # Mock _run_indexed_key_procedural at the method level to inspect what
            # _training_output_dir returns when called with the stamp in effect.
            # We call it for real via the loop's method to exercise the stamp routing.
        ):
            # Temporarily patch _run_indexed_key_procedural to capture the stamp.
            original_proc = loop._run_indexed_key_procedural

            def _proc_capture(rels, speaker_id, **kwargs):
                # Record the output dir that would be used (via _training_output_dir).
                # Accepts mode/stamp/run_label forwarded by run_consolidation_cycle.
                # Pass interim_stamp so the path includes the stamp segment.
                interim_stamp = kwargs.get("stamp")
                captured_dirs.append(
                    str(loop._training_output_dir("procedural", interim_stamp=interim_stamp))
                )

            loop._run_indexed_key_procedural = _proc_capture
            try:
                loop.post_session_train(
                    "Transcript with prefs",
                    "conv-d-004",
                    speaker_id="Speaker0",
                    schedule="every 2h",
                    max_interim_count=4,
                    stamp=stamp,
                )
            finally:
                loop._run_indexed_key_procedural = original_proc

        # The procedural output dir captured inside _run_indexed_key_procedural
        # (during post_session_train's finally-guarded call) must contain the stamp.
        proc_dirs = [d for d in captured_dirs if "procedural" in d]
        assert len(proc_dirs) >= 1, f"No procedural output dir captured: {captured_dirs}"
        for d in proc_dirs:
            assert f"interim_{stamp}" in d, f"Procedural output dir must contain interim stamp: {d}"


# ---------------------------------------------------------------------------
# Test 11 — registry-last write order and restart-time consistency
# ---------------------------------------------------------------------------


class TestRegistryLastWriteOrder:
    """Registry save must be the LAST disk write in post_session_train;
    adapter-save failure must leave no registry entry on disk.
    """

    def test_registry_saved_after_adapter_weights(self, tmp_path: Path) -> None:
        """Registry save (save_from_bytes) must happen after save_adapter.

        The required call sequence is:
        save_bytes → hash → quads.json → build_manifest_for →
        save_adapter → save_from_bytes (registry-as-commit-signal).

        We verify: save_adapter precedes save_from_bytes in the call sequence.
        """
        from paramem.training.key_registry import KeyRegistry

        loop = _make_mock_loop(tmp_path)
        stamp = "20260418T1430"
        loop.model.peft_config[f"episodic_interim_{stamp}"] = MagicMock()

        call_order: list[str] = []

        def _record_save_adapter(*args, **kwargs):
            call_order.append("save_adapter")

        def _record_save_from_bytes(payload, path, **kwargs):
            call_order.append("save_from_bytes")

        with (
            patch.object(loop, "extract_session", return_value=(_fake_qa(2), [])),
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=MagicMock()),
            patch(
                "paramem.models.loader.save_adapter",
                side_effect=_record_save_adapter,
            ),
            patch.object(KeyRegistry, "save_from_bytes", side_effect=_record_save_from_bytes),
        ):
            loop.post_session_train(
                "Transcript",
                "conv-i5-001",
                speaker_id="Speaker0",
                schedule="every 2h",
                max_interim_count=4,
                stamp=stamp,
            )

        # save_adapter must precede save_from_bytes (registry is the commit signal).
        assert "save_adapter" in call_order, "save_adapter was not called"
        assert "save_from_bytes" in call_order, "registry save_from_bytes was not called"
        last_save_adapter = max(i for i, c in enumerate(call_order) if c == "save_adapter")
        first_save_from_bytes = min(i for i, c in enumerate(call_order) if c == "save_from_bytes")
        assert last_save_adapter < first_save_from_bytes, (
            f"save_adapter must come before save_from_bytes; order was: {call_order}"
        )

    def test_adapter_save_failure_means_no_registry_entry(self, tmp_path: Path) -> None:
        """If save_adapter raises, registry must not be written to disk.

        save_from_bytes (the actual on-disk write) comes AFTER save_adapter.
        If save_adapter raises, the exception propagates before save_from_bytes
        is reached, so the registry file is never created.  save_bytes
        (the in-memory step) never touches disk.
        """
        loop = _make_mock_loop(tmp_path)
        stamp = "20260418T1430"
        loop.model.peft_config[f"episodic_interim_{stamp}"] = MagicMock()
        registry_path = tmp_path / "indexed_key_registry.json"

        def _fail_save_adapter(*args, **kwargs):
            raise OSError("disk full")

        with (
            patch.object(loop, "extract_session", return_value=(_fake_qa(1), [])),
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=MagicMock()),
            patch("paramem.models.loader.save_adapter", side_effect=_fail_save_adapter),
        ):
            with pytest.raises(OSError, match="disk full"):
                loop.post_session_train(
                    "Transcript",
                    "conv-i5-002",
                    speaker_id="Speaker0",
                    schedule="every 2h",
                    max_interim_count=4,
                    stamp=stamp,
                )

        # Registry must not exist on disk (save_adapter failed before save_from_bytes).
        assert not registry_path.exists(), "Registry must not be written when adapter save fails"

    # ---- Cleanup contract for torn interim adapter saves ----
    #
    # The production code lives at ``paramem/server/app.py`` inside
    # ``_mount_adapters_from_slots``.  The decision is:
    #
    #     for each ``<adapter_dir>/episodic_interim_*`` with a registry file:
    #         if ANY adapter_model.safetensors exists under that dir (flat
    #         layout OR any slot subdir): keep
    #         elif graph.json present (simulate-mode slot): keep
    #         elif registry has ACTIVE keys (unfolded facts): keep + log error
    #             (fold-or-refuse — data-loss guard 2026-06-02)
    #         else: rmtree the entire interim dir (genuinely torn empty save)
    #
    # Hash mismatch (find_live_slot returns None despite weights present)
    # is NOT a cleanup trigger — it is surfaced via manifest_status.
    @staticmethod
    def _run_i5_check(adapter_dir: Path) -> list[str]:
        """Mirror of the torn-adapter cleanup gate at
        ``app.py``::``_mount_adapters_from_slots``.

        Returns the list of interim tier names that were rmtree'd.  Keep
        this in lockstep with production; if production drifts, these
        tests drift too — that is the whole point.
        """
        import shutil

        from paramem.training.key_registry import KeyRegistry

        deleted: list[str] = []
        for _interim_reg_dir in sorted(adapter_dir.glob("episodic_interim_*")):
            if not _interim_reg_dir.is_dir():
                continue
            _interim_reg_path = _interim_reg_dir / "indexed_key_registry.json"
            if not _interim_reg_path.exists():
                continue
            if any(_interim_reg_dir.rglob("adapter_model.safetensors")):
                continue
            if (_interim_reg_dir / "graph.json").exists():
                continue
            # Fold-or-refuse: an interim registry that still lists active keys
            # holds facts that were never folded into a persisted main tier.
            # Deleting it would lose them (incident 2026-06-02) — preserve.
            if KeyRegistry.load(_interim_reg_path).list_active():
                continue
            shutil.rmtree(_interim_reg_dir, ignore_errors=True)
            deleted.append(_interim_reg_dir.name)
        return deleted

    def test_restart_consistency_check_preserves_unfolded_interim_keys(
        self, tmp_path: Path
    ) -> None:
        """Registry with ACTIVE keys, no weights, no graph → PRESERVE (fold-or-refuse).

        Regression for the 2026-06-02 data-loss incident: an interim slot whose
        keys were never folded into a persisted main tier must NOT be rmtree'd just
        because its weight slot is missing — those facts would be lost permanently.
        """
        from paramem.training.key_registry import KeyRegistry

        interim_tier = "episodic_interim_20260418T1430"
        interim_dir = tmp_path / interim_tier
        interim_dir.mkdir(parents=True)
        (interim_dir / "adapter_config.json").write_text("{}")
        # adapter_model.safetensors deliberately NOT created anywhere.

        interim_reg = KeyRegistry()
        interim_reg.add("graph1")
        interim_reg_path = interim_dir / "indexed_key_registry.json"
        interim_reg.save(interim_reg_path)

        deleted = self._run_i5_check(tmp_path)

        assert deleted == [], "Unfolded interim keys must NOT be rmtree'd"
        assert interim_dir.exists()
        assert KeyRegistry.load(interim_reg_path).list_active() == ["graph1"]

    def test_restart_consistency_check_drops_empty_torn_save(self, tmp_path: Path) -> None:
        """Torn save with an EMPTY registry, no weights, no graph → rmtree.

        This is the genuine torn-write case the cleanup gate exists for: the save was
        interrupted before any key was committed, so there are no facts to lose.
        """
        from paramem.training.key_registry import KeyRegistry

        interim_tier = "episodic_interim_20260418T1430"
        interim_dir = tmp_path / interim_tier
        interim_dir.mkdir(parents=True)
        (interim_dir / "adapter_config.json").write_text("{}")
        # adapter_model.safetensors deliberately NOT created anywhere.

        # Empty registry — no committed keys.
        interim_reg = KeyRegistry()
        interim_reg_path = interim_dir / "indexed_key_registry.json"
        interim_reg.save(interim_reg_path)

        # Main episodic registry (graph2 survives).
        ep_dir = tmp_path / "episodic"
        ep_dir.mkdir(parents=True)
        ep_reg = KeyRegistry()
        ep_reg.add("graph2")
        ep_reg.save(ep_dir / "indexed_key_registry.json")

        deleted = self._run_i5_check(tmp_path)

        assert deleted == [interim_tier]
        assert not interim_dir.exists(), "Empty torn-save interim dir must be rmtree'd"
        assert (ep_dir / "indexed_key_registry.json").exists()
        reloaded_ep = KeyRegistry.load(ep_dir / "indexed_key_registry.json")
        assert "graph2" in reloaded_ep.list_active()

    def test_restart_consistency_check_no_orphans_leaves_registry_unchanged(
        self, tmp_path: Path
    ) -> None:
        """Flat layout, weights present → registry untouched."""
        from paramem.training.key_registry import KeyRegistry

        interim_tier = "episodic_interim_20260418T1430"
        interim_dir = tmp_path / interim_tier
        interim_dir.mkdir(parents=True)
        (interim_dir / "adapter_config.json").write_text("{}")
        (interim_dir / "adapter_model.safetensors").write_bytes(b"fake weights")

        interim_reg = KeyRegistry()
        interim_reg.add("graph1")
        interim_reg_path = interim_dir / "indexed_key_registry.json"
        interim_reg.save(interim_reg_path)

        deleted = self._run_i5_check(tmp_path)

        assert deleted == []
        assert interim_reg_path.exists()
        assert KeyRegistry.load(interim_reg_path).list_active() == ["graph1"]

    def test_restart_consistency_check_nested_slot_weights_preserved(self, tmp_path: Path) -> None:
        """Nested-slot layout (manifest v4): weights under a slot subdir → keep.

        Production writes interim adapters into per-stamp slot subdirs.
        The cleanup gate must walk the tree, not just check the flat layout.
        """
        from paramem.training.key_registry import KeyRegistry

        interim_tier = "episodic_interim_20260418T1430"
        interim_dir = tmp_path / interim_tier
        slot_dir = interim_dir / "20260418-143000"
        slot_dir.mkdir(parents=True)
        (slot_dir / "adapter_config.json").write_text("{}")
        (slot_dir / "adapter_model.safetensors").write_bytes(b"fake weights")

        # Top-level flat layout is empty; registry sits at the interim dir.
        interim_reg = KeyRegistry()
        interim_reg.add("graph1")
        interim_reg_path = interim_dir / "indexed_key_registry.json"
        interim_reg.save(interim_reg_path)

        deleted = self._run_i5_check(tmp_path)

        assert deleted == [], "Nested-slot weights must keep the interim dir"
        assert interim_reg_path.exists()
        assert slot_dir.exists()

    def test_restart_consistency_check_hash_mismatch_with_weights_preserved(
        self, tmp_path: Path
    ) -> None:
        """REGRESSION: hash mismatch + weights present must NOT delete the slot.

        Reproduces 2026-05-14 00:42:56 destructive boot deletion.
        ``find_live_slot`` returned None because the manifest's
        ``registry_sha256`` did not match the live hash; old code
        conflated that with "weights missing" and ``rmtree``'d 134
        trained keys (13 GPU-minutes).  The fix is a pure-filesystem
        check on ``rglob("adapter_model.safetensors")``.
        """
        from paramem.training.key_registry import KeyRegistry

        interim_tier = "episodic_interim_20260513T1200"
        interim_dir = tmp_path / interim_tier
        slot_dir = interim_dir / "20260513-220000"
        slot_dir.mkdir(parents=True)
        (slot_dir / "adapter_config.json").write_text("{}")
        (slot_dir / "adapter_model.safetensors").write_bytes(b"trained weights")
        # The slot has a manifest with the WRONG hash on purpose — the
        # production gate does not consult it; this is the regression
        # case.
        (slot_dir / "meta.json").write_text('{"registry_sha256": "stale-hash"}')

        interim_reg = KeyRegistry()
        interim_reg.add("graph1")
        interim_reg.add("graph2")
        interim_reg_path = interim_dir / "indexed_key_registry.json"
        interim_reg.save(interim_reg_path)

        deleted = self._run_i5_check(tmp_path)

        assert deleted == [], (
            "Hash-mismatch must NOT trigger deletion — that path wiped "
            "134 trained keys on 2026-05-14 00:42:56"
        )
        assert slot_dir.exists()
        assert (slot_dir / "adapter_model.safetensors").exists()
        assert interim_reg_path.exists()


# ---------------------------------------------------------------------------
# Test 12 — save_from_bytes guard (raises when called outside consolidation window)
# ---------------------------------------------------------------------------


class TestSaveFromBytesGuard:
    """KeyRegistry.save_from_bytes raises when called outside consolidation window."""

    def test_save_from_bytes_raises_when_not_consolidating(self, tmp_path: Path) -> None:
        """_require_consolidating=True + consolidating=False → RuntimeError."""
        from paramem.training.key_registry import KeyRegistry

        reg = KeyRegistry()
        reg.add("k1")
        payload = reg.save_bytes()
        path = tmp_path / "registry.json"

        with pytest.raises(RuntimeError, match="_require_consolidating"):
            reg.save_from_bytes(
                payload,
                path,
                _require_consolidating=True,
                consolidating=False,
            )

        # File must NOT have been written
        assert not path.exists()

    def test_save_from_bytes_succeeds_when_consolidating(self, tmp_path: Path) -> None:
        """_require_consolidating=True + consolidating=True → success."""
        from paramem.training.key_registry import KeyRegistry

        reg = KeyRegistry()
        reg.add("k1")
        payload = reg.save_bytes()
        path = tmp_path / "registry.json"

        reg.save_from_bytes(payload, path, _require_consolidating=True, consolidating=True)
        assert path.exists()

    def test_save_from_bytes_opt_out_succeeds(self, tmp_path: Path) -> None:
        """_require_consolidating=False bypasses the guard (experiment path)."""
        from paramem.training.key_registry import KeyRegistry

        reg = KeyRegistry()
        reg.add("k1")
        payload = reg.save_bytes()
        path = tmp_path / "registry.json"

        # Should succeed regardless of consolidating flag
        reg.save_from_bytes(
            payload,
            path,
            _require_consolidating=False,
            consolidating=False,
        )
        assert path.exists()

    def test_save_bytes_then_save_from_bytes_byte_identity(self, tmp_path: Path) -> None:
        """Bytes from save_bytes() written via save_from_bytes() must equal save() output."""
        from paramem.training.key_registry import KeyRegistry

        reg = KeyRegistry()
        reg.add("key1")
        reg.add("key2")
        reg.set_health("healthy", reason="test")

        path_a = tmp_path / "reg_a.json"
        path_b = tmp_path / "reg_b.json"

        payload = reg.save_bytes()
        reg.save(path_a)
        reg.save_from_bytes(payload, path_b, _require_consolidating=False)

        assert path_a.read_bytes() == path_b.read_bytes(), (
            "save_from_bytes must produce byte-identical output to save()"
        )


# ---------------------------------------------------------------------------
# Test 13 — meta.json written inside post_session_train slot
# ---------------------------------------------------------------------------


class TestManifestWrittenPostSession:
    """post_session_train must embed meta.json in the interim adapter slot.

    Verifies that build_manifest_for is called and atomic_save_adapter
    writes meta.json alongside the adapter weights.  Uses a real
    atomic_save_adapter invocation (model.save_pretrained writes stub
    files) so the on-disk assertion is genuine.
    """

    def test_meta_json_written_in_interim_slot(self, tmp_path: Path) -> None:
        """meta.json must be present in the timestamped slot after post_session_train."""
        from paramem.adapters.manifest import AdapterManifest, read_manifest

        loop = _make_mock_loop(tmp_path)
        stamp = "20260418T1430"
        adapter_name = f"episodic_interim_{stamp}"
        loop.model.peft_config[adapter_name] = MagicMock()

        # model.save_pretrained writes stub adapter files into the pending slot
        # so atomic_save_adapter can complete the six-step sequence.
        def _fake_save_pretrained(path, selected_adapters=None):
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "adapter_model.safetensors").write_bytes(b"weights")
            (p / "adapter_config.json").write_text("{}")

        loop.model.save_pretrained.side_effect = _fake_save_pretrained

        # Provide JSON-serialisable config attributes so build_manifest_for can
        # produce a valid manifest without fingerprinting real model weights.
        loop.model.config._name_or_path = "test-base-model"
        loop.model.config._commit_hash = None
        # base_model.model.state_dict() returns an empty dict → base_hash = UNKNOWN
        loop.model.base_model.model.state_dict.return_value = {}
        # Tokenizer: provide a name_or_path string.
        loop.tokenizer.name_or_path = "test-tokenizer"
        loop.tokenizer.backend_tokenizer = None
        loop.tokenizer.vocab_size = 32000
        # LoRA config attributes for the interim adapter.
        lora_cfg = MagicMock()
        lora_cfg.r = 4
        lora_cfg.lora_alpha = 8
        lora_cfg.lora_dropout = 0.0
        lora_cfg.target_modules = ["q_proj"]
        lora_cfg.bias = "none"
        loop.model.peft_config[adapter_name] = lora_cfg

        with (
            patch.object(loop, "extract_session", return_value=(_fake_qa(2), [])),
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
        ):
            result = loop.post_session_train(
                "Transcript",
                "conv-manifest-001",
                speaker_id="Speaker0",
                schedule="every 2h",
                max_interim_count=4,
                stamp=stamp,
            )

        assert result["mode"] == "trained"

        # Locate the timestamped slot created by atomic_save_adapter.
        # 2026-05-14 hierarchy: interim slots live under episodic/interim_<stamp>/.
        adapter_dir = tmp_path / "episodic" / f"interim_{stamp}"
        slots = [d for d in adapter_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        assert slots, f"No slot dir created under {adapter_dir}"
        slot = slots[0]

        # meta.json must be present in the slot.
        assert (slot / "meta.json").exists(), f"meta.json missing from slot {slot}"

        # Manifest must be parseable and reference the correct adapter name.
        manifest = read_manifest(slot)
        assert isinstance(manifest, AdapterManifest)
        assert manifest.name == adapter_name


# ---------------------------------------------------------------------------
# Inter-tier commit recoverability (GAP 2)
# ---------------------------------------------------------------------------


class TestInterTierCommitRecoverable:
    """A crash between the episodic and procedural ``commit_tier_slot`` calls
    must always be RECOVERABLE: the session is NOT marked consolidated until
    ``run_consolidation_cycle`` (here via ``post_session_train``) returns
    successfully, so on a simulated reboot the source session stays pending and
    is re-extracted.  The two commits are NOT transactional together, but the
    consolidated-marking contract makes that residual window safe.

    Reference: ``run_consolidation_cycle`` (consolidation.py step 12) commits
    episodic then procedural; the production caller (``app.py`` post-session
    tick) calls ``session_buffer.mark_consolidated`` ONLY after the cycle
    returns.  This test locks that contract in so a future refactor cannot mark
    sessions consolidated mid-cycle.
    """

    def test_procedural_commit_crash_leaves_session_pending(self, tmp_path: Path) -> None:
        """Procedural ``commit_tier_slot`` raising must propagate out of the
        cycle so a caller's ``mark_consolidated`` is never reached.

        Models a caller that only marks the session consolidated when
        ``post_session_train`` returns normally.  Because the procedural commit
        crashes, the exception propagates, the mark-callback is never invoked,
        and the session remains pending (recoverable on reboot).
        """
        loop = _make_mock_loop_with_procedural(tmp_path)
        stamp = "20260418T1430"
        loop.model.peft_config[f"episodic_interim_{stamp}"] = MagicMock()

        # Track which tiers were committed and whether the (simulated) caller
        # would have marked the session consolidated.
        committed_tiers: list[str] = []
        session_marked_consolidated: list[str] = []

        def _commit_side_effect(*args, **kwargs):
            tier = kwargs["tier"]
            if tier == "procedural":
                # Crash AFTER the episodic commit already landed.  Each commit
                # is internally crash-safe (registry-last commit signal); the
                # pair is what we are proving recoverable here, so the episodic
                # commit's real disk I/O is irrelevant — record-and-return.
                raise RuntimeError("simulated crash during procedural commit_tier_slot")
            committed_tiers.append(tier)

        def _caller_mark_consolidated(session_id: str) -> None:
            # The production caller only calls this AFTER the cycle returns.
            session_marked_consolidated.append(session_id)

        with (
            patch.object(
                loop,
                "extract_session",
                return_value=(_fake_qa(2), _fake_proc_rels(1)),
            ),
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"train_loss": 0.5, "aborted": False},
            ),
            patch(
                "paramem.training.consolidation.train_adapter",
                return_value={"train_loss": 0.5, "aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            # Procedural pass is a no-op success — the crash is at the commit seam.
            patch.object(loop, "_run_indexed_key_procedural", return_value=0.5),
            patch(
                "paramem.memory.persistence.commit_tier_slot",
                side_effect=_commit_side_effect,
            ),
        ):
            try:
                loop.post_session_train(
                    "Transcript with prefs",
                    "conv-recover-001",
                    speaker_id="Speaker0",
                    schedule="every 2h",
                    max_interim_count=4,
                    stamp=stamp,
                )
            except RuntimeError as exc:
                assert "procedural commit_tier_slot" in str(exc)
            else:
                pytest.fail("procedural commit crash must propagate out of the cycle")
            # The caller's mark_consolidated would run here ONLY on success.
            # The except branch above skipped it, mirroring production.

        # Episodic landed; procedural did not — the documented residual state.
        assert committed_tiers == ["episodic"], (
            f"expected only the episodic commit to land before the crash; got {committed_tiers}"
        )
        # The session was NEVER marked consolidated → it stays pending →
        # re-extractable on the next cycle. This is the recoverability invariant.
        assert session_marked_consolidated == [], (
            "session must NOT be marked consolidated when a commit crashes mid-cycle — "
            "it must stay pending so the procedural facts are re-extracted next cycle"
        )


# ---------------------------------------------------------------------------
# B7-A — session_ids provenance carry through _build_all_edge_entries_into
# ---------------------------------------------------------------------------


class TestSessionIdsProvenanceCarry:
    """B7-A acceptance tests: session_ids rides the in-RAM deferred-write record
    but is NEVER written to the persisted entry dict (store.put schema).

    These tests verify the carry-slot contract stated in _build_all_edge_entries_into:
    - rec["session_ids"] is present on the deferred-write record (sorted list of
      real contributing session ids, synthetic sentinels excluded).
    - rec["entry"] does NOT contain "session_ids" (the persisted dict schema
      stays unchanged per owner decision D4).
    - speaker_id attribution is unchanged: the minted-key speaker_id comes from
      the subject node's speaker_id attribute (dcf4189 invariant), not from any
      session_ids field.
    """

    def _make_loop_with_sessions_in_graph(self, tmp_path: Path, *, session_ids: list[str]):
        """Build a minimal loop whose merger.graph has ONE keyless edge with
        the given list of real session ids in edge['sessions'].

        The edge is added with sessions=[real_id1, real_id2, ...] directly so
        we can test the harvest logic without going through the full extraction
        pipeline.
        """
        import networkx as nx

        loop = _make_mock_loop(tmp_path)
        real_graph = nx.MultiDiGraph()
        real_graph.add_node("speaker0", speaker_id="Speaker0", attributes={"name": "Alex"})
        real_graph.add_node("berlin", attributes={"name": "Berlin"})
        real_graph.add_edge(
            "speaker0",
            "berlin",
            predicate="lives_in",
            relation_type="factual",
            sessions=session_ids,
        )
        loop.merger.graph = real_graph
        return loop

    def test_rec_carries_session_ids_after_harvest(self, tmp_path: Path) -> None:
        """Deferred-write rec['session_ids'] contains real session ids from edge['sessions'].

        Synthetic sentinels (_SYNTHETIC_SESSION_IDS) are excluded; real ids survive.
        """
        from paramem.training.consolidation import _SYNTHETIC_SESSION_IDS

        real_ids = ["session-abc", "session-xyz"]
        # Include a synthetic sentinel to confirm it is filtered out.
        sessions_on_edge = real_ids + ["__interim_pending_sessions__"]
        loop = self._make_loop_with_sessions_in_graph(tmp_path, session_ids=sessions_on_edge)

        tier_keyed: dict = {"episodic": [], "procedural": []}
        _, deferred_writes = loop._build_all_edge_entries_into(
            tier_keyed, default_speaker_id="Speaker0", defer=True
        )

        assert len(deferred_writes) == 1, f"Expected 1 deferred write; got {len(deferred_writes)}"
        rec = deferred_writes[0]
        assert "session_ids" in rec, "rec must carry session_ids (B7-A provenance plumbing)"
        result_ids = set(rec["session_ids"])
        assert "session-abc" in result_ids, f"session-abc missing from {result_ids}"
        assert "session-xyz" in result_ids, f"session-xyz missing from {result_ids}"
        # Synthetic sentinel must be excluded.
        for synthetic in _SYNTHETIC_SESSION_IDS:
            assert synthetic not in result_ids, (
                f"Synthetic sentinel {synthetic!r} must be excluded from rec['session_ids']"
            )

    def test_entry_dict_does_not_contain_session_ids(self, tmp_path: Path) -> None:
        """rec['entry'] (the persisted dict passed to store.put) must NOT contain session_ids.

        D4 owner decision: provenance is transient/RAM-only; the persisted
        registry/bookkeeping schema stays unchanged.
        """
        loop = self._make_loop_with_sessions_in_graph(
            tmp_path, session_ids=["session-abc", "__full_consolidation_recon__"]
        )

        tier_keyed: dict = {"episodic": [], "procedural": []}
        _, deferred_writes = loop._build_all_edge_entries_into(
            tier_keyed, default_speaker_id="Speaker0", defer=True
        )

        assert deferred_writes, "Expected at least one deferred write"
        entry = deferred_writes[0]["entry"]
        assert "session_ids" not in entry, (
            "session_ids must NOT appear in the persisted entry dict — "
            "it is a transient rec-level field only (D4)"
        )

    def test_speaker_id_attribution_unchanged_by_session_ids(self, tmp_path: Path) -> None:
        """dcf4189 invariant: minted-key speaker_id from subject node, not from session_ids.

        Multi-session edge scenario: the edge carries real session ids from two
        sessions.  The minted entry's speaker_id must come from the subject node's
        speaker_id attribute ('Speaker0'), not from any session_id value.
        """
        loop = self._make_loop_with_sessions_in_graph(
            tmp_path, session_ids=["session-A", "session-B"]
        )

        tier_keyed: dict = {"episodic": [], "procedural": []}
        _, deferred_writes = loop._build_all_edge_entries_into(
            tier_keyed, default_speaker_id="Speaker0", defer=True
        )

        assert deferred_writes, "Expected at least one deferred write"
        rec = deferred_writes[0]
        # speaker_id on the rec comes from the subject node attribute ("Speaker0"),
        # not from any session_ids element.
        assert rec["speaker_id"] == "Speaker0", (
            f"speaker_id must be 'Speaker0' (from subject node attribute); "
            f"got {rec['speaker_id']!r}"
        )
        # The rec carries BOTH real session ids.
        result_ids = set(rec["session_ids"])
        assert "session-A" in result_ids and "session-B" in result_ids, (
            f"Both real session ids must be in rec['session_ids']; got {result_ids}"
        )

    def test_empty_sessions_on_edge_gives_empty_session_ids(self, tmp_path: Path) -> None:
        """Edge with no sessions list → rec['session_ids'] is empty (not an error)."""
        import networkx as nx

        loop = _make_mock_loop(tmp_path)
        real_graph = nx.MultiDiGraph()
        real_graph.add_node("speaker0", speaker_id="Speaker0", attributes={"name": "Alex"})
        real_graph.add_node("berlin", attributes={"name": "Berlin"})
        # No 'sessions' key on the edge (legacy graph or edge without stamps).
        real_graph.add_edge("speaker0", "berlin", predicate="lives_in", relation_type="factual")
        loop.merger.graph = real_graph

        tier_keyed: dict = {"episodic": [], "procedural": []}
        _, deferred_writes = loop._build_all_edge_entries_into(
            tier_keyed, default_speaker_id="Speaker0", defer=True
        )

        assert deferred_writes, "Expected at least one deferred write"
        assert deferred_writes[0]["session_ids"] == [], (
            "Edge without sessions key → session_ids must be []"
        )

    def test_only_synthetic_sessions_gives_empty_session_ids(self, tmp_path: Path) -> None:
        """Edge whose sessions list contains ONLY synthetic sentinels → session_ids is [].

        This happens when a fold-only re-merge creates an edge with no real
        extraction-time session ids.
        """
        from paramem.training.consolidation import _SYNTHETIC_SESSION_IDS

        synthetic_only = list(_SYNTHETIC_SESSION_IDS)
        loop = self._make_loop_with_sessions_in_graph(tmp_path, session_ids=synthetic_only)

        tier_keyed: dict = {"episodic": [], "procedural": []}
        _, deferred_writes = loop._build_all_edge_entries_into(
            tier_keyed, default_speaker_id="Speaker0", defer=True
        )

        assert deferred_writes, "Expected at least one deferred write"
        assert deferred_writes[0]["session_ids"] == [], (
            f"Only-synthetic sessions → session_ids must be []; "
            f"got {deferred_writes[0]['session_ids']}"
        )


# ---------------------------------------------------------------------------
# B7-B — keep recall-failed sessions pending + bounded retry + incident
# ---------------------------------------------------------------------------


class TestRecallFailedSessionStaysPending:
    """B7-B acceptance tests.

    The keep-pending / bounded-retry / incident wiring (B7-B, T6/T7/T8/T10,
    §3, S-4) is validated without GPU or model weights.

    Test strategy:
    - Call run_consolidation_cycle directly (not post_session_train) so we
      can control the graph and probe stub precisely.
    - Set up the merger graph with an edge tagged with a real session id so
      rec["session_ids"] carries it through the harvest path (B7-A).
    - Override _probe_passing_keys to exclude one key, triggering the drop
      site at step 11b.
    - Assert result["recall_failed_session_ids"] and downstream behavior.

    W4 / R2 caveat:
    Under interim_refinement="off", _pending_relations is None so the
    pending-session relations may not enter the merge graph and new episodic
    keys may not be minted.  The "off" arm asserts conditionally:
    if no new keys are minted, recall_failed_session_ids must be [] (the
    bug cannot manifest there); the non-empty "off" case awaits the R2 GPU
    probe.  Procedural is asserted under "off" regardless (always fact-dict
    carrier).
    """

    # Shared patch list for run_consolidation_cycle without model weights.
    _CYCLE_PATCHES = [
        "paramem.memory.interim_adapter.create_interim_adapter",
        "paramem.training.trainer.train_adapter",
        "paramem.training.consolidation.format_entry_training",
        "paramem.models.loader.switch_adapter",
        "paramem.training.consolidation.build_registry",
        "paramem.models.loader.save_adapter",
    ]

    def _make_loop_with_session_edge(
        self,
        tmp_path: Path,
        *,
        session_id: str = "real-session-001",
        interim_refinement: str = "light",
    ):
        """Build a loop whose merger.graph has a keyless edge with a real session id.

        The edge's sessions=[session_id] so _build_all_edge_entries_into
        harvests it onto rec["session_ids"].  Used for episodic recall-gate tests.
        """
        loop = _make_mock_loop(tmp_path)
        loop.config = loop.config.__class__(
            indexed_key_replay_enabled=True,
            interim_refinement=interim_refinement,
        )
        real_graph = nx.MultiDiGraph()
        real_graph.add_node("alice", speaker_id="Speaker0", attributes={"name": "Alice"})
        real_graph.add_node("paris", attributes={"name": "Paris"})
        real_graph.add_edge(
            "alice",
            "paris",
            predicate="lives_in",
            relation_type="factual",
            sessions=[session_id],
        )
        loop.merger.graph = real_graph
        # Use a fresh stamp so the adapter name is predictable.
        loop.model.peft_config["episodic_interim_20260617T0000"] = MagicMock()
        return loop

    def _run_cycle(self, loop, *, stamp: str = "20260617T0000", mode: str = "train"):
        """Call run_consolidation_cycle with the standard patch stack.

        Passes a single dummy episodic relation to bypass the no-relations guard
        (step 3 in run_consolidation_cycle).  The actual key minting comes from
        merger.graph (set up per test), not from this placeholder relation.
        """
        # One dummy relation to bypass guard step 3 (no episodic_rels → noop).
        _dummy_rel = [
            {
                "subject": "placeholder",
                "predicate": "placeholder",
                "object": "placeholder",
                "relation_type": "factual",
                "speaker_id": "Speaker0",
            }
        ]
        with (
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"train_loss": 0.3, "aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.models.loader.save_adapter"),
        ):
            return loop.run_consolidation_cycle(
                _dummy_rel,  # bypass guard step 3; real keys come from merger.graph
                [],  # procedural_rels: empty for episodic-only tests
                speaker_id="Speaker0",
                mode=mode,
                run_label="test",
                schedule="",
                max_interim_count=4,
                stamp=stamp,
            )

    # ------------------------------------------------------------------
    # Test 1 — recall-failed key: session stays pending, key not registered
    # ------------------------------------------------------------------

    def test_episodic_recall_failure_populates_recall_failed_session_ids(
        self, tmp_path: Path
    ) -> None:
        """A new key that fails the recall gate → result contains its session id.

        The invariant (key unregistered AND session not consolidated) is
        verified by checking both the result dict and the store state.
        """
        session_id = "real-session-alpha"
        loop = self._make_loop_with_session_edge(
            tmp_path, session_id=session_id, interim_refinement="light"
        )

        # Override probe so it fails ALL new keys (empty passing set for new keys).
        # _recall_passing_keys returns None → falls through to _probe_passing_keys.
        loop._probe_passing_keys = lambda adapter_name, entries: set()

        result = self._run_cycle(loop, mode="train")

        # The cycle must return recall_failed_session_ids with the contributing session.
        assert "recall_failed_session_ids" in result, (
            "result must carry recall_failed_session_ids (T8)"
        )
        assert session_id in result["recall_failed_session_ids"], (
            f"session {session_id!r} must be in recall_failed_session_ids; "
            f"got {result['recall_failed_session_ids']}"
        )

        # Invariant: key is NOT registered (drop site skipped store.put).
        active_keys = set(loop.store.all_active_keys())
        assert len(active_keys) == 0, (
            f"No key must be registered when recall gate fails; got {active_keys}"
        )

    def test_simulate_mode_produces_empty_recall_failed_session_ids(self, tmp_path: Path) -> None:
        """Simulate mode: recall gate is not run → recall_failed_session_ids is [].

        B2: the simulate callsite is NOT plumbed; the result is always empty.
        """
        session_id = "real-session-sim"
        loop = self._make_loop_with_session_edge(
            tmp_path, session_id=session_id, interim_refinement="light"
        )
        # Even with a failing probe, simulate admits all without the gate.
        loop._probe_passing_keys = lambda adapter_name, entries: set()

        result = self._run_cycle(loop, mode="simulate")

        assert result.get("recall_failed_session_ids", []) == [], (
            "Simulate mode must never produce recall_failed_session_ids"
        )

    def test_off_refinement_episodic_arm_conditional(self, tmp_path: Path) -> None:
        """Under interim_refinement='off', assert conditionally per W4/R2 caveat.

        If no new episodic keys are minted (pending-sessions path absent in "off"),
        recall_failed_session_ids is [] — the bug cannot manifest; no assertion
        beyond that.  If keys ARE minted (not expected from static analysis, but
        defensive), we assert the failing key's session is collected.

        NOTE: the R2 GPU probe must establish the "off"-config minting source
        before asserting the non-empty "off" case.  This conditional arm is
        intentional — do not strengthen it without R2 results.
        """
        session_id = "real-session-off"
        loop = self._make_loop_with_session_edge(
            tmp_path, session_id=session_id, interim_refinement="off"
        )
        loop._probe_passing_keys = lambda adapter_name, entries: set()

        result = self._run_cycle(loop, mode="train")

        new_keys = result.get("new_keys", [])
        failed = result.get("recall_failed_session_ids", [])

        if not new_keys:
            # R2 expected path: "off" mints no new episodic keys from the pending
            # graph, so the recall gate is never reached.  Bug cannot manifest.
            assert failed == [], (
                f"Under 'off' with no new keys, recall_failed_session_ids must be []; got {failed}"
            )
        else:
            # Defensive: if keys were minted, the failing session must be collected.
            assert session_id in failed, (
                f"Under 'off' with new keys, session {session_id!r} must appear in "
                f"recall_failed_session_ids; got {failed}"
            )

    # ------------------------------------------------------------------
    # Test 2 — bounded retry: cap → WARNING + incident + session released
    # ------------------------------------------------------------------

    def test_bump_retry_and_release_increments_counter(self, tmp_path: Path) -> None:
        """bump_retry_and_release increments recall_retry_count per session."""
        from paramem.server.session_buffer import SessionBuffer

        buf = SessionBuffer(tmp_path, recall_retry_cap=3)
        sid = "session-retry-001"
        buf._sessions[sid] = {"speaker": None, "state": "new"}

        released = buf.bump_retry_and_release({sid})

        assert released == [], "Count 1 < cap 3 — must not release"
        assert buf._sessions[sid]["recall_retry_count"] == 1

    def test_bump_retry_and_release_releases_at_cap(self, tmp_path: Path) -> None:
        """When retry count reaches cap, session is returned in released list."""
        from paramem.server.session_buffer import SessionBuffer

        buf = SessionBuffer(tmp_path, recall_retry_cap=3)
        sid = "session-retry-cap"
        buf._sessions[sid] = {"speaker": None, "state": "new", "recall_retry_count": 2}

        released = buf.bump_retry_and_release({sid})

        assert sid in released, f"Session must be released at cap; got {released}"
        assert buf._sessions[sid]["recall_retry_count"] == 3

    def test_bump_retry_and_release_skips_absent_ids(self, tmp_path: Path) -> None:
        """R3 guard: ids absent from _sessions are silently skipped."""
        from paramem.server.session_buffer import SessionBuffer

        buf = SessionBuffer(tmp_path, recall_retry_cap=3)
        # synthetic id that is not in _sessions
        released = buf.bump_retry_and_release({"__interim_pending_sessions__"})
        assert released == []

    def test_cap_release_records_incident_and_logs_warning(self, tmp_path: Path, caplog) -> None:
        """Hitting the cap → consolidation_recall_failure incident recorded + WARNING logged.

        The test simulates the T10 wiring by running bump_retry_and_release
        directly (the app.py closure is tested via integration; here we test
        the incident-record contract independently of app.py).
        """
        import logging

        from paramem.server.incidents import read_incidents, record_incident
        from paramem.server.session_buffer import SessionBuffer

        buf = SessionBuffer(tmp_path, recall_retry_cap=2)
        sid = "session-cap-incident"
        buf._sessions[sid] = {"speaker": None, "state": "new", "recall_retry_count": 1}
        state_dir = tmp_path / "state"

        # Explicitly wire caplog handler into the session_buffer logger so the
        # WARNING emitted by bump_retry_and_release is captured by caplog.
        # Pattern from tests/test_interim_adapter_lifecycle.py (caplog handler wire).
        sb_logger = logging.getLogger("paramem.server.session_buffer")
        sb_logger.addHandler(caplog.handler)
        sb_logger.setLevel(logging.WARNING)
        try:
            released = buf.bump_retry_and_release({sid})
        finally:
            sb_logger.removeHandler(caplog.handler)

        assert sid in released

        # Record the incident (mirrors T10's for-loop in _run_interim_training).
        record_incident(
            state_dir,
            type="consolidation_recall_failure",
            key=sid,
            severity="warning",
            summary=(
                f"Session {sid}: facts could not be encoded after {buf._recall_retry_cap} cycle(s)"
            ),
            detail={"session_id": sid, "recall_retry_cap": buf._recall_retry_cap},
        )

        incidents = read_incidents(state_dir)
        recall_incidents = [i for i in incidents if i.type == "consolidation_recall_failure"]
        assert len(recall_incidents) == 1
        # Incident id is f"{type}:{key}" (S-3 dedup key = session id).
        assert recall_incidents[0].id == f"consolidation_recall_failure:{sid}"

        # WARNING logged by bump_retry_and_release.
        assert any("recall-retry cap" in r.message for r in caplog.records), (
            f"Expected WARNING about recall-retry cap; got: {[r.message for r in caplog.records]}"
        )

    # ------------------------------------------------------------------
    # Test 3 — invariant guard: (key unregistered) ∧ (session NOT consolidated)
    # ------------------------------------------------------------------

    def test_invariant_no_state_where_key_unregistered_and_session_consolidated(
        self, tmp_path: Path
    ) -> None:
        """No state where (key unregistered) AND (session consolidated) can coexist.

        When a key fails the recall gate, it is NOT registered AND the session
        is NOT in the retire set returned by _completed_session_ids().  The
        invariant holds because the session stays in failed_session_ids.
        """
        session_id = "inv-session-001"
        loop = self._make_loop_with_session_edge(
            tmp_path, session_id=session_id, interim_refinement="light"
        )
        loop._probe_passing_keys = lambda adapter_name, entries: set()

        result = self._run_cycle(loop, mode="train")

        # Key must not be registered.
        assert len(list(loop.store.all_active_keys())) == 0, "Key must not be registered"
        # Session must appear in recall_failed_session_ids — the caller's
        # failed_session_ids.update() keeps it out of _completed_session_ids().
        assert session_id in result.get("recall_failed_session_ids", []), (
            "Session must be in recall_failed_session_ids to stay pending"
        )

    # ------------------------------------------------------------------
    # Test 4 — partial failure: passing key registered, failing not, session stays pending
    # ------------------------------------------------------------------

    def test_partial_failure_passing_key_registered_failing_not(self, tmp_path: Path) -> None:
        """Two new keys: one passes the recall gate, one fails.

        The passing key is registered; the failing key is not.  The session
        that contributed the failing key stays pending (appears in
        recall_failed_session_ids).  The passing key is not double-registered
        on a second run (store.put is idempotent via the store's own dedup).
        """
        # Add TWO edges with different sessions to the graph.
        loop = _make_mock_loop(tmp_path)
        loop.config = loop.config.__class__(
            indexed_key_replay_enabled=True,
            interim_refinement="light",
        )
        real_graph = nx.MultiDiGraph()
        real_graph.add_node("alice", speaker_id="Speaker0", attributes={"name": "Alice"})
        real_graph.add_node("paris", attributes={"name": "Paris"})
        real_graph.add_node("london", attributes={"name": "London"})
        # Edge 1: contributing session "session-pass"
        real_graph.add_edge(
            "alice",
            "paris",
            predicate="lives_in",
            relation_type="factual",
            sessions=["session-pass"],
        )
        # Edge 2: contributing session "session-fail"
        real_graph.add_edge(
            "alice",
            "london",
            predicate="visited",
            relation_type="factual",
            sessions=["session-fail"],
        )
        loop.merger.graph = real_graph
        loop.model.peft_config["episodic_interim_20260617T0000"] = MagicMock()

        _minted_keys: list[str] = []

        def _probe_partial(adapter_name, entries):
            # Collect keys as they are minted; fail the second one.
            for e in entries:
                if e["key"] not in _minted_keys:
                    _minted_keys.append(e["key"])
            # Fail the last-minted key.
            return set(_minted_keys[:-1]) if _minted_keys else set()

        loop._probe_passing_keys = _probe_partial

        with (
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"train_loss": 0.3, "aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.models.loader.save_adapter"),
        ):
            # One dummy episodic relation to bypass guard step 3.
            result = loop.run_consolidation_cycle(
                [
                    {
                        "subject": "p",
                        "predicate": "p",
                        "object": "p",
                        "relation_type": "factual",
                        "speaker_id": "Speaker0",
                    }
                ],
                [],
                speaker_id="Speaker0",
                mode="train",
                run_label="test",
                schedule="",
                max_interim_count=4,
                stamp="20260617T0000",
            )

        failed = set(result.get("recall_failed_session_ids", []))
        # "session-fail" contributed the failing key → must appear.
        assert "session-fail" in failed, f"session-fail must be in failed; got {failed}"
        # "session-pass" contributed the passing key → must NOT appear.
        assert "session-pass" not in failed, f"session-pass must NOT be in failed; got {failed}"
        # The passing key is registered.
        active = set(loop.store.all_active_keys())
        assert len(active) == 1, f"Exactly 1 key must be registered; got {active}"

    # ------------------------------------------------------------------
    # Test 5 — procedural path: recall failure collects session id (T7)
    # ------------------------------------------------------------------

    def test_procedural_recall_failure_populates_recall_failed_session_ids(
        self, tmp_path: Path
    ) -> None:
        """New procedural key that fails the recall gate → session id collected.

        The procedural carrier is the fact-dict session_id (T9 stamp, app.py).
        _mint_keyed_entries copies rel["session_id"] → kp["session_ids"].
        The drop site in _run_indexed_key_procedural updates the shared set.
        Procedural path works under interim_refinement="off" too (always fact-dict).
        """
        loop = _make_mock_loop_with_procedural(tmp_path)
        loop.model.peft_config["episodic_interim_20260617T0000"] = MagicMock()

        proc_sid = "session-proc-fail"
        # Procedural relation with session_id stamped (T9 mirrors app.py:12432).
        proc_rels = [
            {
                "subject": "Alice",
                "predicate": "prefers",
                "object": "Tea",
                "relation_type": "preference",
                "speaker_id": "Speaker0",
                "session_id": proc_sid,
            }
        ]

        # Override proc probe to fail all keys.
        loop._probe_passing_keys = lambda adapter_name, entries: set()

        with (
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"train_loss": 0.3, "aborted": False},
            ),
            patch(
                "paramem.training.consolidation.train_adapter",
                return_value={"train_loss": 0.3, "aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.models.loader.save_adapter"),
        ):
            result = loop.run_consolidation_cycle(
                [],
                proc_rels,
                speaker_id="Speaker0",
                mode="train",
                run_label="test",
                schedule="",
                max_interim_count=4,
                stamp="20260617T0000",
            )

        failed = result.get("recall_failed_session_ids", [])
        assert proc_sid in failed, (
            f"Procedural recall-failed session {proc_sid!r} must be in "
            f"recall_failed_session_ids; got {failed}"
        )

    # ------------------------------------------------------------------
    # S-4 — conditional resolve: failing cycle keeps incident; clean cycle resolves
    # ------------------------------------------------------------------

    def test_s4_failing_cycle_does_not_resolve_recall_failure_incident(
        self, tmp_path: Path
    ) -> None:
        """A cycle returning non-empty recall_failed_session_ids must NOT resolve the incident.

        Mirrors the S-4 ordering hazard: B7 records/bumps when non-empty;
        the success path MUST NOT wipe it in the same cycle.
        """
        from paramem.server.incidents import (
            read_incidents,
            record_incident,
            resolve_incidents_by_type,
        )

        state_dir = tmp_path / "state"
        sid = "session-s4-fail"
        # Pre-record an incident (simulates a prior cycle having recorded it).
        record_incident(
            state_dir,
            type="consolidation_recall_failure",
            key=sid,
            severity="warning",
            summary=f"Session {sid}: facts could not be encoded",
            detail={"session_id": sid},
        )

        # Simulate the S-4 conditional: result has a non-empty failed set.
        result_with_failures = {"recall_failed_session_ids": [sid]}
        if not result_with_failures.get("recall_failed_session_ids", []):
            resolve_incidents_by_type(state_dir, "consolidation_recall_failure")
        # Since failed is non-empty, we do NOT resolve — incident stays active.

        incidents = read_incidents(state_dir)
        recall_incidents = [i for i in incidents if i.type == "consolidation_recall_failure"]
        assert len(recall_incidents) == 1
        assert recall_incidents[0].status == "active", (
            "Incident must remain active when failing cycle runs (S-4 ordering)"
        )

    def test_s4_clean_cycle_resolves_recall_failure_incident(self, tmp_path: Path) -> None:
        """A cycle returning empty recall_failed_session_ids RESOLVES the incident.

        S-4 resolution rule: resolve consolidation_recall_failure ONLY when ZERO
        keys failed this cycle.
        """
        from paramem.server.incidents import (
            read_incidents,
            record_incident,
            resolve_incidents_by_type,
        )

        state_dir = tmp_path / "state"
        sid = "session-s4-clean"
        record_incident(
            state_dir,
            type="consolidation_recall_failure",
            key=sid,
            severity="warning",
            summary=f"Session {sid}: facts could not be encoded",
            detail={"session_id": sid},
        )

        # Simulate the S-4 conditional: result has an empty failed set.
        result_clean = {"recall_failed_session_ids": []}
        if not result_clean.get("recall_failed_session_ids", []):
            resolve_incidents_by_type(state_dir, "consolidation_recall_failure")

        incidents = read_incidents(state_dir)
        recall_incidents = [i for i in incidents if i.type == "consolidation_recall_failure"]
        assert len(recall_incidents) == 1
        assert recall_incidents[0].status == "resolved", (
            "Incident must be resolved when clean cycle runs (S-4 ordering)"
        )
