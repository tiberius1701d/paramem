"""Tests for forgetting metrics computation."""

from paramem.evaluation.forgetting import (
    ForgettingMetrics,
    compute_forgetting_metrics,
    compute_forgetting_reduction,
    format_results_table,
)
from paramem.training.sequential import SequentialResult


class TestComputeForgettingMetrics:
    def test_two_topics_with_forgetting(self):
        results = [
            SequentialResult(
                topic_name="A",
                step_index=0,
                training_metrics={},
                recall_per_topic={"A": 0.90},
            ),
            SequentialResult(
                topic_name="B",
                step_index=1,
                training_metrics={},
                recall_per_topic={"A": 0.30, "B": 0.85},
            ),
        ]

        metrics = compute_forgetting_metrics(results)

        assert "A" in metrics
        assert "B" in metrics

        # Topic A: peak=0.90, final=0.30
        assert metrics["A"].peak_recall == 0.90
        assert metrics["A"].final_recall == 0.30
        assert abs(metrics["A"].forgetting_rate - (0.90 - 0.30) / 0.90) < 1e-6

        # Topic B: trained last, no forgetting
        assert metrics["B"].peak_recall == 0.85
        assert metrics["B"].final_recall == 0.85
        assert metrics["B"].forgetting_rate == 0.0

    def test_no_forgetting(self):
        results = [
            SequentialResult(
                topic_name="A",
                step_index=0,
                training_metrics={},
                recall_per_topic={"A": 0.90},
            ),
            SequentialResult(
                topic_name="B",
                step_index=1,
                training_metrics={},
                recall_per_topic={"A": 0.90, "B": 0.85},
            ),
        ]

        metrics = compute_forgetting_metrics(results)
        assert metrics["A"].forgetting_rate == 0.0

    def test_four_topics_cumulative(self):
        results = [
            SequentialResult("A", 0, {}, {"A": 0.95}),
            SequentialResult("B", 1, {}, {"A": 0.60, "B": 0.90}),
            SequentialResult("C", 2, {}, {"A": 0.40, "B": 0.50, "C": 0.88}),
            SequentialResult("D", 3, {}, {"A": 0.20, "B": 0.30, "C": 0.45, "D": 0.92}),
        ]

        metrics = compute_forgetting_metrics(results)

        assert metrics["A"].recall_curve == [0.95, 0.60, 0.40, 0.20]
        assert metrics["B"].recall_curve == [0.90, 0.50, 0.30]
        assert metrics["D"].forgetting_rate == 0.0


class TestComputeForgettingReduction:
    def test_50_percent_reduction(self):
        baseline = {
            "A": ForgettingMetrics("A", 0.90, 0.30, 0.667, [0.90, 0.30]),
        }
        strategy = {
            "A": ForgettingMetrics("A", 0.90, 0.60, 0.333, [0.90, 0.60]),
        }

        result = compute_forgetting_reduction(baseline, strategy)

        # baseline mean forgetting = 0.667, strategy = 0.333
        # reduction = (0.667 - 0.333) / 0.667 ≈ 0.5
        assert abs(result["overall_reduction"] - 0.5) < 0.01

    def test_no_common_topics(self):
        baseline = {"A": ForgettingMetrics("A", 0.9, 0.3, 0.667, [])}
        strategy = {"B": ForgettingMetrics("B", 0.9, 0.6, 0.333, [])}

        result = compute_forgetting_reduction(baseline, strategy)
        assert result["overall_reduction"] == 0.0

    def test_per_topic_breakdown(self):
        baseline = {
            "A": ForgettingMetrics("A", 0.90, 0.30, 0.667, []),
            "B": ForgettingMetrics("B", 0.80, 0.40, 0.500, []),
        }
        strategy = {
            "A": ForgettingMetrics("A", 0.90, 0.70, 0.222, []),
            "B": ForgettingMetrics("B", 0.80, 0.60, 0.250, []),
        }

        result = compute_forgetting_reduction(baseline, strategy)
        assert "A" in result["per_topic"]
        assert "B" in result["per_topic"]


class TestFormatResultsTable:
    def test_produces_output(self):
        conditions = {
            "baseline": {
                "A": ForgettingMetrics("A", 0.90, 0.30, 0.667, [0.90, 0.30]),
            },
            "naive": {
                "A": ForgettingMetrics("A", 0.90, 0.60, 0.333, [0.90, 0.60]),
            },
        }

        table = format_results_table(conditions)
        assert "baseline" in table
        assert "naive" in table
        assert "A" in table

    def test_empty_conditions(self):
        assert format_results_table({}) == "No results to display."
