"""Tests for the consolidation loop orchestrator.

These are unit tests that mock the model/extraction to test
the consolidation logic without requiring GPU.
"""

import pytest

from paramem.evaluation.consolidation_metrics import (
    ConsolidationMetrics,
    compute_consolidation_metrics,
    compute_episodic_decay_rate,
    compute_promoted_retention,
    compute_semantic_drift,
    format_phase3_summary,
)
from paramem.training.consolidation import ConsolidationLoop, CycleResult, _mentions_any
from paramem.training.curriculum import CurriculumSampler
from paramem.utils.config import ConsolidationConfig


class TestCycleResult:
    def test_default_values(self):
        result = CycleResult(cycle_index=1, session_id="s001")
        assert result.entities_extracted == 0
        assert result.nodes_promoted == 0
        assert result.promoted_nodes == []

    def test_with_values(self):
        result = CycleResult(
            cycle_index=1,
            session_id="s001",
            entities_extracted=5,
            nodes_promoted=2,
            promoted_nodes=["Alex", "Heilbronn"],
        )
        assert result.entities_extracted == 5
        assert len(result.promoted_nodes) == 2


class TestMentionsAny:
    def test_finds_mention(self):
        assert _mentions_any("Alex lives in Heilbronn", {"alex"})

    def test_no_mention(self):
        assert not _mentions_any("The weather is nice", {"alex"})

    def test_case_insensitive(self):
        assert _mentions_any("ALEX is here", {"alex"})

    def test_empty_terms(self):
        assert not _mentions_any("some text", set())


class TestConsolidationMetrics:
    def test_compute_metrics(self):
        results = [
            CycleResult(
                cycle_index=1,
                session_id="s001",
                entities_extracted=5,
                nodes_promoted=1,
                nodes_decayed=0,
                episodic_train_loss=0.8,
                wall_clock_seconds=120.0,
            ),
            CycleResult(
                cycle_index=2,
                session_id="s002",
                entities_extracted=3,
                nodes_promoted=0,
                nodes_decayed=1,
                episodic_train_loss=0.6,
                semantic_train_loss=0.4,
                wall_clock_seconds=90.0,
            ),
        ]
        metrics = compute_consolidation_metrics(results)
        assert metrics.total_cycles == 2
        assert metrics.total_entities_extracted == 8
        assert metrics.total_promotions == 1
        assert metrics.total_decays == 1
        assert metrics.mean_wall_clock_seconds == 105.0
        assert len(metrics.episodic_losses) == 2
        assert len(metrics.semantic_losses) == 1


class TestPromotedRetention:
    def test_full_retention(self):
        recall_scores = [
            {"Alex": 0.9, "Heilbronn": 0.85},
            {"Alex": 0.9, "Heilbronn": 0.85},
        ]
        result = compute_promoted_retention(recall_scores, {"Alex", "Heilbronn"})
        assert result["mean_retention"] >= 0.85

    def test_no_promoted_nodes(self):
        result = compute_promoted_retention([], set())
        assert result["mean_retention"] == 0.0

    def test_partial_retention(self):
        recall_scores = [
            {"Alex": 0.9},
            {"Alex": 0.5},
        ]
        result = compute_promoted_retention(recall_scores, {"Alex"})
        assert result["per_node"]["Alex"]["final"] == 0.5
        assert result["per_node"]["Alex"]["peak"] == 0.9


class TestEpisodicDecay:
    def test_measurable_decay(self):
        recall_scores = [
            {"Barcelona": 0.8},
            {"Barcelona": 0.6},
            {"Barcelona": 0.3},
        ]
        result = compute_episodic_decay_rate(
            recall_scores,
            {"Barcelona"},
            {"Barcelona": 0},
        )
        assert result["mean_decay_rate"] > 0

    def test_no_decay(self):
        result = compute_episodic_decay_rate([], set(), {})
        assert result["mean_decay_rate"] == 0.0


class TestSemanticDrift:
    def test_no_drift(self):
        recall_scores = [
            {"Alex": 0.9},
            {"Alex": 0.9},
            {"Alex": 0.9},
        ]
        result = compute_semantic_drift(recall_scores, {"Alex"})
        assert result["mean_drift"] == 0.0

    def test_measurable_drift(self):
        recall_scores = [
            {"Alex": 0.9},
            {"Alex": 0.85},
            {"Alex": 0.80},
        ]
        result = compute_semantic_drift(recall_scores, {"Alex"})
        assert result["mean_drift"] == pytest.approx(0.1)


class TestExtractionPathParity:
    """extract_session() and run_cycle() must produce identical episodic/procedural
    sets for the same transcript. Guards against drift between the production
    path (server) and the experiment path (phase3/phase4 scripts)."""

    def _build_loop(
        self,
        monkeypatch,
        tmp_path,
        procedural_enabled: bool,
        extract_graph_spy=None,
        extract_procedural_spy=None,
        **loop_kwargs,
    ):
        from unittest.mock import MagicMock

        from paramem.graph.qa_generator import generate_qa_from_relations as _real_qa
        from paramem.graph.schema import Entity, Relation, SessionGraph
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        session_graph = SessionGraph(
            session_id="s001",
            timestamp="2026-01-01T00:00:00Z",
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
            relations=[
                Relation(
                    subject="Alex",
                    predicate="lives_in",
                    object="Millfield",
                    relation_type="factual",
                ),
                Relation(
                    subject="Alex",
                    predicate="prefers",
                    object="Acme Radio",
                    relation_type="preference",
                ),
            ],
        )
        procedural_graph = SessionGraph(
            session_id="s001",
            timestamp="2026-01-01T00:00:00Z",
            entities=[Entity(name="Alex", entity_type="person")],
            relations=[
                Relation(
                    subject="Alex",
                    predicate="listens_to",
                    object="The Kooks",
                    relation_type="preference",
                ),
            ],
        )

        def _default_extract(*a, **kw):
            return session_graph

        def _default_extract_procedural(*a, **kw):
            return procedural_graph

        monkeypatch.setattr(
            "paramem.training.consolidation.extract_graph",
            extract_graph_spy if extract_graph_spy is not None else _default_extract,
        )
        monkeypatch.setattr(
            "paramem.training.consolidation.extract_procedural_graph",
            extract_procedural_spy
            if extract_procedural_spy is not None
            else _default_extract_procedural,
        )
        # Use template fallback (ignore passed model/tokenizer) for deterministic output.
        monkeypatch.setattr(
            "paramem.training.consolidation.generate_qa_from_relations",
            lambda relations, model=None, tokenizer=None: _real_qa(relations),
        )

        model = MagicMock()
        model.peft_config = {
            "episodic": MagicMock(),
            "semantic": MagicMock(),
            "in_training": MagicMock(),
        }
        if procedural_enabled:
            model.peft_config["procedural"] = MagicMock()
        procedural_adapter = AdapterConfig() if procedural_enabled else None

        return ConsolidationLoop(
            model=model,
            tokenizer=MagicMock(),
            consolidation_config=ConsolidationConfig(),
            training_config=TrainingConfig(),
            episodic_adapter_config=AdapterConfig(),
            semantic_adapter_config=AdapterConfig(),
            procedural_adapter_config=procedural_adapter,
            output_dir=tmp_path,
            persist_graph=False,
            **loop_kwargs,
        )

    def _run_extract_session(self, loop):
        return loop.extract_session(
            session_transcript="Alex lives in Millfield. He prefers Acme Radio.",
            session_id="s001",
            speaker_id="spk",
        )

    def _run_cycle_and_capture(self, monkeypatch, loop):
        captured: dict[str, list[dict]] = {"episodic_qa": [], "procedural_rels": []}

        def _capture_episodic(self_, episodic_qa, new_promotions):
            captured["episodic_qa"] = episodic_qa
            return None

        def _capture_procedural(self_, procedural_relations, speaker_id=""):
            captured["procedural_rels"] = procedural_relations
            return None

        # Force the indexed-key branch so we can intercept. Stub registry + save.
        from unittest.mock import MagicMock as _MM

        loop.indexed_key_registry = _MM()
        loop.indexed_key_registry.list_active = _MM(return_value=[])
        monkeypatch.setattr(
            type(loop), "_run_indexed_key_episodic", _capture_episodic, raising=False
        )
        monkeypatch.setattr(
            type(loop),
            "_run_indexed_key_procedural",
            _capture_procedural,
            raising=False,
        )
        monkeypatch.setattr(type(loop), "_save_adapters", lambda self_: None, raising=False)
        loop.run_cycle(
            session_transcript="Alex lives in Millfield. He prefers Acme Radio.",
            session_id="s001",
            speaker_id="spk",
        )
        return captured["episodic_qa"], captured["procedural_rels"]

    def test_parity_procedural_enabled(self, monkeypatch, tmp_path):
        loop_a = self._build_loop(monkeypatch, tmp_path / "a", procedural_enabled=True)
        episodic_a, procedural_a = self._run_extract_session(loop_a)

        loop_b = self._build_loop(monkeypatch, tmp_path / "b", procedural_enabled=True)
        episodic_b, procedural_b = self._run_cycle_and_capture(monkeypatch, loop_b)

        # Identity fields must match across both paths.
        def _key(qa):
            return (qa["source_subject"], qa["source_predicate"], qa["source_object"])

        def _rel_key(rel):
            return (rel["subject"], rel["predicate"], rel["object"])

        assert sorted(map(_key, episodic_a)) == sorted(map(_key, episodic_b))
        assert sorted(map(_rel_key, procedural_a)) == sorted(map(_rel_key, procedural_b))

        # Partition invariant: no preference in episodic, and procedural carries
        # both the filter-sourced and the separately-extracted preference.
        assert all(qa["source_predicate"] != "prefers" for qa in episodic_a)
        assert {rel["predicate"] for rel in procedural_a} == {"prefers", "listens_to"}

    @pytest.mark.parametrize(
        "procedural_enabled,loop_overrides",
        [
            (True, {}),
            (False, {}),
            (
                True,
                {
                    "extraction_noise_filter": "claude",
                    "extraction_noise_filter_model": "claude-sonnet-4-6",
                },
            ),
            (True, {"extraction_plausibility_stage": "anon"}),
            (True, {"extraction_ner_check": True, "extraction_ner_model": "en_core_web_sm"}),
            (
                True,
                {
                    "extraction_stt_correction": False,
                    "extraction_ha_validation": False,
                    "extraction_verify_anonymization": False,
                },
            ),
        ],
    )
    def test_parity_kwargs_identical(
        self, monkeypatch, tmp_path, procedural_enabled, loop_overrides
    ):
        """Both orchestrator paths must pass IDENTICAL kwargs to the extractors.

        Any new flag added to one path but not the other will fail here — the
        helper + _extraction_kwargs are the only source of truth.
        """
        from paramem.graph.schema import Entity, Relation, SessionGraph

        session_graph = SessionGraph(
            session_id="s001",
            timestamp="2026-01-01T00:00:00Z",
            entities=[Entity(name="Alex", entity_type="person")],
            relations=[
                Relation(
                    subject="Alex",
                    predicate="lives_in",
                    object="Millfield",
                    relation_type="factual",
                ),
            ],
        )
        procedural_graph = SessionGraph(
            session_id="s001",
            timestamp="2026-01-01T00:00:00Z",
            entities=[],
            relations=[],
        )

        captured_a: dict[str, list[dict]] = {"graph": [], "procedural": []}
        captured_b: dict[str, list[dict]] = {"graph": [], "procedural": []}

        def _spy(bucket, fixture):
            def _f(model, tokenizer, transcript, session_id, **kwargs):
                bucket.append(kwargs)
                return fixture

            return _f

        loop_a = self._build_loop(
            monkeypatch,
            tmp_path / "a",
            procedural_enabled=procedural_enabled,
            extract_graph_spy=_spy(captured_a["graph"], session_graph),
            extract_procedural_spy=_spy(captured_a["procedural"], procedural_graph),
            **loop_overrides,
        )
        self._run_extract_session(loop_a)

        loop_b = self._build_loop(
            monkeypatch,
            tmp_path / "b",
            procedural_enabled=procedural_enabled,
            extract_graph_spy=_spy(captured_b["graph"], session_graph),
            extract_procedural_spy=_spy(captured_b["procedural"], procedural_graph),
            **loop_overrides,
        )
        self._run_cycle_and_capture(monkeypatch, loop_b)

        # Each path calls extract_graph exactly once with the same kwargs.
        assert len(captured_a["graph"]) == 1
        assert len(captured_b["graph"]) == 1
        assert captured_a["graph"][0] == captured_b["graph"][0], (
            "extract_graph kwargs diverged between extract_session and run_cycle. "
            f"extract_session: {captured_a['graph'][0]!r}\n"
            f"run_cycle:       {captured_b['graph'][0]!r}"
        )

        # Procedural path: same kwarg shape when enabled, neither path calls it when disabled.
        assert len(captured_a["procedural"]) == len(captured_b["procedural"])
        if procedural_enabled:
            assert len(captured_a["procedural"]) == 1
            assert captured_a["procedural"][0] == captured_b["procedural"][0], (
                "extract_procedural_graph kwargs diverged between paths. "
                f"extract_session: {captured_a['procedural'][0]!r}\n"
                f"run_cycle:       {captured_b['procedural'][0]!r}"
            )
        else:
            assert captured_a["procedural"] == []
            assert captured_b["procedural"] == []

    def test_parity_procedural_disabled(self, monkeypatch, tmp_path):
        loop_a = self._build_loop(monkeypatch, tmp_path / "a", procedural_enabled=False)
        episodic_a, procedural_a = self._run_extract_session(loop_a)

        loop_b = self._build_loop(monkeypatch, tmp_path / "b", procedural_enabled=False)
        episodic_b, procedural_b = self._run_cycle_and_capture(monkeypatch, loop_b)

        def _key(qa):
            return (qa["source_subject"], qa["source_predicate"], qa["source_object"])

        assert sorted(map(_key, episodic_a)) == sorted(map(_key, episodic_b))
        # With procedural disabled, preferences fall back into episodic — never lost.
        assert any(qa["source_predicate"] == "prefers" for qa in episodic_a)
        assert procedural_a == []
        assert procedural_b == []

    def test_empty(self):
        result = compute_semantic_drift([], set())
        assert result["mean_drift"] == 0.0


class TestFormatSummary:
    def test_format_output(self):
        metrics = ConsolidationMetrics(
            total_cycles=10,
            total_entities_extracted=50,
            total_promotions=5,
            total_decays=3,
            mean_wall_clock_seconds=100.0,
        )
        retention = {"mean_retention": 0.85}
        decay = {"mean_decay_rate": 0.3}
        drift = {"mean_drift": 0.02}
        summary = format_phase3_summary(metrics, retention, decay, drift)
        assert "Phase 3" in summary
        assert "PASS" in summary
        assert "10" in summary


class TestCurriculumDecayProtection:
    """Test that curriculum-aware decay respects min_exposure_cycles."""

    def _make_loop_with_curriculum(self, min_exposure=3):
        """Create a ConsolidationLoop with curriculum enabled, no real model."""
        config = ConsolidationConfig(
            curriculum_enabled=True,
            min_exposure_cycles=min_exposure,
        )
        # We only need the _apply_decay method — model/tokenizer are not used
        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.config = config
        loop.curriculum_sampler = CurriculumSampler(
            min_exposure_cycles=min_exposure,
        )
        loop.episodic_replay_pool = []
        return loop

    def test_decay_blocked_by_min_exposure(self):
        loop = self._make_loop_with_curriculum(min_exposure=3)
        loop.episodic_replay_pool = [
            {"question": "Where does Alex live?", "answer": "Heilbronn"},
        ]
        # exposure=0 < 3 → decay should be blocked
        loop._apply_decay(["Alex"])
        assert len(loop.episodic_replay_pool) == 1

    def test_decay_allowed_after_min_exposure(self):
        loop = self._make_loop_with_curriculum(min_exposure=3)
        loop.episodic_replay_pool = [
            {"question": "Where does Alex live?", "answer": "Heilbronn"},
        ]
        # Simulate 3 exposures
        loop.curriculum_sampler.exposure_counts["Where does Alex live?"] = 3
        loop._apply_decay(["Alex"])
        assert len(loop.episodic_replay_pool) == 0

    def test_decay_without_curriculum(self):
        """Without curriculum, decay works as before."""
        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.config = ConsolidationConfig(curriculum_enabled=False)
        loop.curriculum_sampler = None
        loop.episodic_replay_pool = [
            {"question": "Where does Alex live?", "answer": "Heilbronn"},
        ]
        loop._apply_decay(["Alex"])
        assert len(loop.episodic_replay_pool) == 0

    def test_mixed_decay_and_protection(self):
        loop = self._make_loop_with_curriculum(min_exposure=2)
        loop.episodic_replay_pool = [
            {"question": "Where does Alex live?", "answer": "Heilbronn"},
            {"question": "What is Alex's pet?", "answer": "Luna"},
        ]
        # Only the pet question has enough exposures
        loop.curriculum_sampler.exposure_counts["What is Alex's pet?"] = 2
        loop._apply_decay(["Alex"])
        # First kept (protected), second decayed (exposure met)
        assert len(loop.episodic_replay_pool) == 1
        assert loop.episodic_replay_pool[0]["question"] == "Where does Alex live?"
