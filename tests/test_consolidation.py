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
