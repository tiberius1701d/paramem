"""Unit tests for ConsolidationLoop.post_session_train.

Pure-Python, no GPU required.  All heavy dependencies (extraction, training,
adapter creation) are replaced with MagicMock objects so each test executes in
milliseconds and verifies only the orchestration logic of post_session_train.

Test numbering mirrors the plan's 6e specification:
  1. First call creates the interim adapter and returns mode="trained".
  2. Second call within the same sub-interval reuses the adapter.
  3. Stamp rollover creates a new adapter without touching the previous one.
  4. Zero facts extracted → mode="noop", no adapter created.
  5. Training failure → registry unchanged, no adapter saved.
  6. max_interim_count=0 → mode="queued", triples in pending_interim_triples.
  7. Queued facts are present in pending_interim_triples (Step 7 skeleton).
  8. Keys registered only AFTER training returns, not before.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    loop.persist_graph = False
    loop.enable_entity_promotion = False
    loop.extraction_temperature = 0.0
    loop.extraction_max_tokens = 256
    loop.extraction_stt_correction = False
    loop.extraction_ha_validation = False
    loop.extraction_noise_filter = ""
    loop.extraction_noise_filter_model = ""
    loop.extraction_noise_filter_endpoint = None
    loop.extraction_ner_check = False
    loop.extraction_ner_model = "en_core_web_sm"
    loop.extraction_plausibility_judge = "off"
    loop.extraction_plausibility_stage = "deanon"
    loop.extraction_verify_anonymization = False
    loop.indexed_key_registry = KeyRegistry()
    loop.indexed_key_qa = {}
    loop._indexed_next_index = 1
    loop._procedural_next_index = 1
    loop.episodic_simhash = {}
    loop.semantic_simhash = {}
    loop.procedural_simhash = {}
    loop.procedural_sp_index = {}
    loop.cycle_count = 0
    loop.key_sessions = {}
    loop.promoted_keys = set()
    loop.pending_interim_triples = []
    loop.shutdown_requested = False
    loop._shutdown_callbacks = []
    loop.merger = MagicMock()
    loop.merger.graph = MagicMock()
    loop.merger.graph.nodes = []
    # Graph-enrichment knobs (Task #10). Hook fires inside the normal
    # fresh-interim branch; default to disabled for these unit tests so the
    # hook stays inert and we don't need to stub _run_graph_enrichment.
    loop.graph_enrichment_enabled = False
    loop.graph_enrichment_neighborhood_hops = 2
    loop.graph_enrichment_max_entities_per_pass = 50
    loop.graph_enrichment_interim_enabled = False
    loop.graph_enrichment_min_triples_floor = 20
    loop._triples_since_last_enrichment = 0

    return loop


def _fake_qa(n: int = 2) -> list[dict]:
    """Return n synthetic QA dicts."""
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
                "paramem.server.interim_adapter.create_interim_adapter",
                side_effect=_create_side_effect,
            ),
            patch("paramem.training.trainer.train_adapter"),
            patch("paramem.training.consolidation.format_indexed_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.training.consolidation.save_registry"),
            patch("paramem.models.loader.save_adapter"),
        ):
            result = loop.post_session_train(
                "Hello world transcript",
                "conv-001",
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
                "paramem.server.interim_adapter.create_interim_adapter",
                side_effect=_create_side_effect,
            ),
            patch("paramem.training.trainer.train_adapter"),
            patch("paramem.training.consolidation.format_indexed_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.training.consolidation.save_registry"),
            patch("paramem.models.loader.save_adapter"),
        ):
            loop.post_session_train(
                "Transcript", "conv-001", schedule="every 2h", max_interim_count=4, stamp=stamp
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
            patch("paramem.server.interim_adapter.create_interim_adapter") as mock_create,
            patch("paramem.training.trainer.train_adapter"),
            patch("paramem.training.consolidation.format_indexed_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.training.consolidation.save_registry"),
            patch("paramem.models.loader.save_adapter"),
        ):
            loop.post_session_train(
                "Second conversation",
                "conv-002",
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
                "paramem.server.interim_adapter.create_interim_adapter",
                side_effect=_create_side_effect,
            ) as mock_create,
            patch("paramem.training.trainer.train_adapter"),
            patch("paramem.training.consolidation.format_indexed_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.training.consolidation.save_registry"),
            patch("paramem.models.loader.save_adapter"),
        ):
            result = loop.post_session_train(
                "Rollover transcript",
                "conv-003",
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
            patch("paramem.server.interim_adapter.create_interim_adapter") as mock_create,
        ):
            result = loop.post_session_train(
                "Empty transcript",
                "conv-004",
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
        initial_len = len(loop.indexed_key_registry)

        with patch.object(loop, "extract_session", return_value=([], [])):
            loop.post_session_train(
                "Empty", "conv-005", schedule="every 2h", max_interim_count=4, stamp="20260418T1430"
            )

        assert len(loop.indexed_key_registry) == initial_len


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
                "paramem.server.interim_adapter.create_interim_adapter",
                side_effect=lambda m, cfg, s: m,
            ),
            patch("paramem.training.trainer.train_adapter", side_effect=_raise),
            patch("paramem.training.consolidation.format_indexed_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
        ):
            loop.model.peft_config[f"episodic_interim_{stamp}"] = MagicMock()
            with pytest.raises(RuntimeError, match="Simulated training failure"):
                loop.post_session_train(
                    "Transcript", "conv-006", schedule="every 2h", max_interim_count=4, stamp=stamp
                )

        # Registry must be unchanged after the training error.
        assert len(loop.indexed_key_registry) == 0

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
                "paramem.server.interim_adapter.create_interim_adapter",
                side_effect=lambda m, cfg, s: m,
            ),
            patch("paramem.training.trainer.train_adapter", side_effect=_raise),
            patch("paramem.training.consolidation.format_indexed_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
        ):
            loop.model.peft_config[f"episodic_interim_{stamp}"] = MagicMock()
            with pytest.raises(RuntimeError):
                loop.post_session_train(
                    "Transcript", "conv-007", schedule="every 2h", max_interim_count=4, stamp=stamp
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
            patch("paramem.server.interim_adapter.create_interim_adapter") as mock_create,
            patch("paramem.training.trainer.train_adapter") as mock_train,
        ):
            result = loop.post_session_train(
                "Transcript", "conv-008", schedule="every 2h", max_interim_count=0
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
                "Transcript A", "conv-009a", schedule="every 2h", max_interim_count=0
            )
            loop.post_session_train(
                "Transcript B", "conv-009b", schedule="every 2h", max_interim_count=0
            )

        # Both extractions accumulated.
        assert len(loop.pending_interim_triples) == 4

    def test_zero_count_does_not_mutate_registry(self, tmp_path: Path) -> None:
        """max_interim_count=0 leaves the key registry untouched."""
        loop = _make_mock_loop(tmp_path)

        with patch.object(loop, "extract_session", return_value=(_fake_qa(2), [])):
            loop.post_session_train(
                "Transcript", "conv-010", schedule="every 2h", max_interim_count=0
            )

        assert len(loop.indexed_key_registry) == 0


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
            registry_before_training.append(len(loop.indexed_key_registry))

        with (
            patch.object(loop, "extract_session", return_value=(_fake_qa(2), [])),
            patch(
                "paramem.server.interim_adapter.create_interim_adapter",
                side_effect=lambda m, cfg, s: m,
            ),
            patch(
                "paramem.training.trainer.train_adapter",
                side_effect=_capture_registry_state,
            ),
            patch("paramem.training.consolidation.format_indexed_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.training.consolidation.save_registry"),
            patch("paramem.models.loader.save_adapter"),
        ):
            loop.model.peft_config[f"episodic_interim_{stamp}"] = MagicMock()
            loop.post_session_train(
                "Transcript", "conv-011", schedule="every 2h", max_interim_count=4, stamp=stamp
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
                "paramem.server.interim_adapter.create_interim_adapter",
                side_effect=lambda m, cfg, s: m,
            ),
            patch("paramem.training.trainer.train_adapter"),
            patch("paramem.training.consolidation.format_indexed_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.training.consolidation.save_registry"),
            patch("paramem.models.loader.save_adapter"),
        ):
            loop.model.peft_config[f"episodic_interim_{stamp}"] = MagicMock()
            result = loop.post_session_train(
                "Transcript", "conv-012", schedule="every 2h", max_interim_count=4, stamp=stamp
            )

        # Two facts extracted → two keys registered.
        assert len(result["new_keys"]) == 2
        for k in result["new_keys"]:
            assert loop.indexed_key_registry.get_adapter_id(k) == f"episodic_interim_{stamp}"


# ---------------------------------------------------------------------------
# Helpers for new tests (I1 / I3 / I5)
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
    """Like _make_mock_loop but with procedural_config set (enables I1 code path)."""
    from paramem.utils.config import AdapterConfig

    loop = _make_mock_loop(tmp_path)
    proc_cfg = AdapterConfig(rank=4, alpha=8, learning_rate=1e-4, target_modules=["q_proj"])
    loop.procedural_config = proc_cfg
    return loop


# Shared patch list for a successful post_session_train call (no procedural).
_COMMON_PATCHES = [
    "paramem.server.interim_adapter.create_interim_adapter",
    "paramem.training.trainer.train_adapter",
    "paramem.training.consolidation.format_indexed_training",
    "paramem.models.loader.switch_adapter",
    "paramem.training.consolidation.build_registry",
    "paramem.training.consolidation.save_registry",
    "paramem.models.loader.save_adapter",
]


# ---------------------------------------------------------------------------
# Test 9 — I1: procedural_rels are routed to the procedural adapter
# ---------------------------------------------------------------------------


class TestProceduralRelsRoutedToProceduralAdapter:
    """I1: procedural relations must reach the procedural adapter, not be silently dropped."""

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
            patch("paramem.server.interim_adapter.create_interim_adapter"),
            patch("paramem.training.trainer.train_adapter"),
            patch("paramem.training.consolidation.format_indexed_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.training.consolidation.save_registry"),
            patch("paramem.models.loader.save_adapter"),
            patch.object(loop, "_run_indexed_key_procedural") as mock_proc,
        ):
            loop.post_session_train(
                "Transcript with prefs",
                "conv-p-001",
                schedule="every 2h",
                max_interim_count=4,
                stamp=stamp,
            )

        mock_proc.assert_called_once()
        # First positional arg is the procedural relations list.
        called_rels = mock_proc.call_args[0][0]
        assert len(called_rels) == 1

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
            patch("paramem.server.interim_adapter.create_interim_adapter"),
            patch("paramem.training.trainer.train_adapter"),
            patch("paramem.training.consolidation.format_indexed_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.training.consolidation.save_registry"),
            patch("paramem.models.loader.save_adapter"),
            patch.object(loop, "_run_indexed_key_procedural") as mock_proc,
        ):
            loop.post_session_train(
                "Transcript no prefs",
                "conv-p-002",
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
            patch("paramem.server.interim_adapter.create_interim_adapter"),
            patch("paramem.training.trainer.train_adapter"),
            patch("paramem.training.consolidation.format_indexed_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.training.consolidation.save_registry"),
            patch("paramem.models.loader.save_adapter"),
            patch.object(loop, "_run_indexed_key_procedural") as mock_proc,
        ):
            loop.post_session_train(
                "Transcript",
                "conv-p-003",
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

        Pre-conditions
        --------------
        - One old procedural key ``proc0`` exists in ``procedural_simhash``,
          ``indexed_key_qa``, and ``procedural_sp_index`` with the same
          (speaker, subject, predicate) as the incoming relation, so it would
          normally be retired.

        Post-conditions after the procedural training failure
        -----------------------------------------------------
        - ``indexed_key_registry`` is empty: episodic step 7 never ran.
        - ``indexed_key_qa`` does NOT contain any new procedural key.
        - ``procedural_simhash`` is unchanged: old key still present.
        - ``procedural_sp_index`` still maps to the old key (not the new one).
        - Old procedural key ``proc0`` was NOT removed from any index.
        """
        loop = _make_mock_loop_with_procedural(tmp_path)
        stamp = "20260418T1430"
        loop.model.peft_config[f"episodic_interim_{stamp}"] = MagicMock()

        # Pre-existing procedural state: proc0 with the same (speaker, subject,
        # predicate) as the incoming relation, making it a contradiction target.
        # _fake_proc_rels(1) produces subject="Subject1", predicate="prefers".
        old_sp_key = ("", "subject1", "prefers")  # lowercased, empty speaker_id
        old_proc_key = "proc0"
        old_qa_entry = {
            "key": old_proc_key,
            "question": "Old question?",
            "answer": "Old answer.",
            "source_subject": "Subject1",
            "source_predicate": "prefers",
            "source_object": "OldThing",
            "speaker_id": "",
        }
        loop.procedural_simhash = {old_proc_key: 0xDEADBEEF}
        loop.indexed_key_qa[old_proc_key] = old_qa_entry
        loop.procedural_sp_index[old_sp_key] = old_proc_key

        call_count = {"n": 0}

        def _train_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                # Second call is the procedural train_adapter — raise.
                raise RuntimeError("procedural training failed")
            return MagicMock(get=lambda k, d=None: 0.5)

        # Synthetic QA for generate_qa_from_relations (called inside
        # _run_indexed_key_procedural).
        proc_qa_out = [
            {
                "question": "What does Subject1 prefer?",
                "answer": "NewThing1.",
                "source_subject": "Subject1",
                "source_predicate": "prefers",
                "source_object": "NewThing1",
            }
        ]

        with (
            patch.object(
                loop,
                "extract_session",
                return_value=(_fake_qa(2), _fake_proc_rels(1)),
            ),
            patch("paramem.server.interim_adapter.create_interim_adapter"),
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
                "paramem.training.consolidation.generate_qa_from_relations",
                return_value=proc_qa_out,
            ),
            patch("paramem.training.consolidation.format_indexed_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            # probe_key: no existing reconstructable keys (proc0 is the only one,
            # and it is in the retirement list, so existing_keys will be empty).
            patch("paramem.training.consolidation.probe_key", return_value=None),
            patch("paramem.training.consolidation.build_registry", return_value={}),
        ):
            with pytest.raises(RuntimeError, match="procedural training failed"):
                loop.post_session_train(
                    "Transcript",
                    "conv-p-004",
                    schedule="every 2h",
                    max_interim_count=4,
                    stamp=stamp,
                )

        # Step 7 (episodic key registration) never ran — registry stays empty.
        assert len(loop.indexed_key_registry) == 0, (
            "Episodic keys must not be registered when procedural training fails."
        )

        # No new procedural key in indexed_key_qa — only the pre-existing old entry.
        proc_keys_in_qa = [k for k in loop.indexed_key_qa if k.startswith("proc")]
        assert proc_keys_in_qa == [old_proc_key], (
            f"Expected only old procedural key '{old_proc_key}' in indexed_key_qa; "
            f"found: {proc_keys_in_qa}"
        )

        # Old contradicted key must NOT have been retired from procedural_simhash.
        assert old_proc_key in loop.procedural_simhash, (
            "Old procedural key must still be in procedural_simhash after training failure."
        )

        # Old key must NOT have been removed from indexed_key_qa.
        assert old_proc_key in loop.indexed_key_qa, (
            "Old procedural key must still be in indexed_key_qa after training failure."
        )

        # procedural_sp_index must still point to the old key.
        assert loop.procedural_sp_index.get(old_sp_key) == old_proc_key, (
            f"procedural_sp_index must still map {old_sp_key!r} → '{old_proc_key}' "
            "after training failure."
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
            patch("paramem.server.interim_adapter.create_interim_adapter"),
            patch("paramem.training.trainer.train_adapter"),
            patch("paramem.training.consolidation.format_indexed_training", return_value=[{}]),
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
                    schedule="every 2h",
                    max_interim_count=4,
                    stamp=stamp,
                )

        # _current_interim_stamp must be cleaned up (finally block).
        assert getattr(loop, "_current_interim_stamp", None) is None


# ---------------------------------------------------------------------------
# Test 10 — I3: output directory uniqueness across stamps and calls
# ---------------------------------------------------------------------------


class TestTrainingOutputDirUniqueness:
    """I3: consecutive post_session_train calls must not share an output directory."""

    def _run_and_capture_output_dirs(self, loop, *, stamp: str, session_id: str) -> list[str]:
        """Run post_session_train and return all output_dir paths passed to train_adapter."""
        captured: list[str] = []

        def _capture_train(*args, **kwargs):
            captured.append(str(kwargs.get("output_dir", "")))

        loop.model.peft_config[f"episodic_interim_{stamp}"] = MagicMock()

        with (
            patch.object(loop, "extract_session", return_value=(_fake_qa(1), [])),
            patch("paramem.server.interim_adapter.create_interim_adapter"),
            patch("paramem.training.trainer.train_adapter", side_effect=_capture_train),
            patch("paramem.training.consolidation.format_indexed_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.training.consolidation.save_registry"),
            patch("paramem.models.loader.save_adapter"),
        ):
            loop.post_session_train(
                "Transcript",
                session_id,
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
            patch("paramem.server.interim_adapter.create_interim_adapter"),
            patch("paramem.training.trainer.train_adapter", side_effect=_capture_train),
            patch("paramem.training.consolidation.format_indexed_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.training.consolidation.save_registry"),
            patch("paramem.models.loader.save_adapter"),
            # Mock _run_indexed_key_procedural at the method level to inspect what
            # _training_output_dir returns when called with the stamp in effect.
            # We call it for real via the loop's method to exercise the stamp routing.
        ):
            # Temporarily patch _run_indexed_key_procedural to capture the stamp.
            original_proc = loop._run_indexed_key_procedural

            def _proc_capture(rels, speaker_id=""):
                # Record the output dir that would be used (via _training_output_dir).
                captured_dirs.append(str(loop._training_output_dir("procedural")))

            loop._run_indexed_key_procedural = _proc_capture
            try:
                loop.post_session_train(
                    "Transcript with prefs",
                    "conv-d-004",
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
# Test 11 — I5: registry-last write order and restart-time consistency
# ---------------------------------------------------------------------------


class TestRegistryLastWriteOrder:
    """I5: registry save must be the LAST disk write; adapter-save failure = no registry entry."""

    def test_registry_saved_after_adapter_weights(self, tmp_path: Path) -> None:
        """Registry save (save_from_bytes) must happen after save_adapter.

        After the I5 reorder (§2.5), the call sequence is:
        save_bytes → hash → keyed_pairs.json → build_manifest_for →
        save_adapter → save_registry (SimHash) → save_from_bytes.

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
            patch("paramem.server.interim_adapter.create_interim_adapter"),
            patch("paramem.training.trainer.train_adapter"),
            patch("paramem.training.consolidation.format_indexed_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.training.consolidation.save_registry"),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=MagicMock()),
            patch(
                "paramem.models.loader.save_adapter",
                side_effect=_record_save_adapter,
            ),
            patch.object(KeyRegistry, "save_from_bytes", side_effect=_record_save_from_bytes),
        ):
            loop.post_session_train(
                "Transcript", "conv-i5-001", schedule="every 2h", max_interim_count=4, stamp=stamp
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

        After the I5 reorder (§2.5), save_from_bytes (the actual on-disk
        write) comes AFTER save_adapter.  If save_adapter raises, the
        exception propagates before save_from_bytes is reached, so the
        registry file is never created.  save_bytes (step 1) is in-memory
        only and never touches disk.
        """
        loop = _make_mock_loop(tmp_path)
        stamp = "20260418T1430"
        loop.model.peft_config[f"episodic_interim_{stamp}"] = MagicMock()
        registry_path = tmp_path / "indexed_key_registry.json"

        def _fail_save_adapter(*args, **kwargs):
            raise OSError("disk full")

        with (
            patch.object(loop, "extract_session", return_value=(_fake_qa(1), [])),
            patch("paramem.server.interim_adapter.create_interim_adapter"),
            patch("paramem.training.trainer.train_adapter"),
            patch("paramem.training.consolidation.format_indexed_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.training.consolidation.save_registry"),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=MagicMock()),
            patch("paramem.models.loader.save_adapter", side_effect=_fail_save_adapter),
        ):
            with pytest.raises(OSError, match="disk full"):
                loop.post_session_train(
                    "Transcript",
                    "conv-i5-002",
                    schedule="every 2h",
                    max_interim_count=4,
                    stamp=stamp,
                )

        # Registry must not exist on disk (save_adapter failed before save_from_bytes).
        assert not registry_path.exists(), "Registry must not be written when adapter save fails"

    def test_restart_consistency_check_drops_orphan_keys(self, tmp_path: Path) -> None:
        """Lifespan consistency check: registry entries with missing adapter weights are dropped.

        Simulates a crash between adapter-save and registry-save by writing
        a registry that references an adapter whose safetensors file is absent.
        """
        from paramem.training.key_registry import KeyRegistry

        # Write a registry that claims a key belongs to an interim adapter
        # but whose safetensors file does not exist.
        reg = KeyRegistry()
        reg.add("graph1", adapter_id="episodic_interim_20260418T1430")
        reg.add("graph2", adapter_id="episodic")  # main adapter — always present
        registry_path = tmp_path / "indexed_key_registry.json"
        reg.save(registry_path)

        # The interim adapter directory exists (adapter_config.json was written)
        # but adapter_model.safetensors is absent (simulating interrupted save).
        interim_dir = tmp_path / "episodic_interim_20260418T1430"
        interim_dir.mkdir(parents=True)
        (interim_dir / "adapter_config.json").write_text("{}")
        # adapter_model.safetensors deliberately NOT created.

        # Run the consistency check logic (extracted from lifespan for testability).
        _reg = KeyRegistry.load(registry_path)
        _orphaned: list[str] = []
        for _key in list(_reg.list_active()):
            _aid = _reg.get_adapter_id(_key)
            if _aid.startswith("episodic_interim_"):
                _weights = tmp_path / _aid / "adapter_model.safetensors"
                if not _weights.exists():
                    _reg.remove(_key)
                    _orphaned.append(_key)
        if _orphaned:
            _reg.save(registry_path)

        assert _orphaned == ["graph1"], f"Expected graph1 orphaned, got: {_orphaned}"

        # Reload and verify graph1 is gone; graph2 survives.
        _reg2 = KeyRegistry.load(registry_path)
        assert "graph1" not in _reg2.list_active()
        assert "graph2" in _reg2.list_active()

    def test_restart_consistency_check_no_orphans_leaves_registry_unchanged(
        self, tmp_path: Path
    ) -> None:
        """When all adapter weights are present, the registry is untouched."""
        from paramem.training.key_registry import KeyRegistry

        reg = KeyRegistry()
        reg.add("graph1", adapter_id="episodic_interim_20260418T1430")
        registry_path = tmp_path / "indexed_key_registry.json"
        reg.save(registry_path)

        # Create both required files for the interim adapter.
        interim_dir = tmp_path / "episodic_interim_20260418T1430"
        interim_dir.mkdir(parents=True)
        (interim_dir / "adapter_config.json").write_text("{}")
        (interim_dir / "adapter_model.safetensors").write_bytes(b"fake weights")

        # Run consistency check.
        _reg = KeyRegistry.load(registry_path)
        _orphaned: list[str] = []
        for _key in list(_reg.list_active()):
            _aid = _reg.get_adapter_id(_key)
            if _aid.startswith("episodic_interim_"):
                _weights = tmp_path / _aid / "adapter_model.safetensors"
                if not _weights.exists():
                    _reg.remove(_key)
                    _orphaned.append(_key)
        if _orphaned:
            _reg.save(registry_path)

        assert _orphaned == [], f"No orphans expected when weights present: {_orphaned}"
        # Registry file mtime should NOT have changed (no orphans found, no rewrite).
        _reg2 = KeyRegistry.load(registry_path)
        assert "graph1" in _reg2.list_active()


# ---------------------------------------------------------------------------
# Test 12 — save_from_bytes guard (§2.5 defence-in-depth)
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
        reg.set_adapter_health("episodic", "healthy", reason="test")

        path_a = tmp_path / "reg_a.json"
        path_b = tmp_path / "reg_b.json"

        payload = reg.save_bytes()
        reg.save(path_a)
        reg.save_from_bytes(payload, path_b, _require_consolidating=False)

        assert path_a.read_bytes() == path_b.read_bytes(), (
            "save_from_bytes must produce byte-identical output to save()"
        )


# ---------------------------------------------------------------------------
# Test 13 — meta.json written inside post_session_train slot (§2.5)
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
            patch("paramem.server.interim_adapter.create_interim_adapter"),
            patch("paramem.training.trainer.train_adapter"),
            patch("paramem.training.consolidation.format_indexed_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.training.consolidation.save_registry"),
        ):
            result = loop.post_session_train(
                "Transcript",
                "conv-manifest-001",
                schedule="every 2h",
                max_interim_count=4,
                stamp=stamp,
            )

        assert result["mode"] == "trained"

        # Locate the timestamped slot created by atomic_save_adapter.
        adapter_dir = tmp_path / adapter_name
        slots = [d for d in adapter_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        assert slots, f"No slot dir created under {adapter_dir}"
        slot = slots[0]

        # meta.json must be present in the slot.
        assert (slot / "meta.json").exists(), f"meta.json missing from slot {slot}"

        # Manifest must be parseable and reference the correct adapter name.
        manifest = read_manifest(slot)
        assert isinstance(manifest, AdapterManifest)
        assert manifest.name == adapter_name
