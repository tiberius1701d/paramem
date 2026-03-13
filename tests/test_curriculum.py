"""Tests for curriculum-aware replay sampling."""

from collections import Counter

from paramem.training.curriculum import (
    CurriculumSampler,
    _weighted_sample_indices,
)


class TestWeightedSampleIndices:
    def test_uniform_weights(self):
        probs = [0.25, 0.25, 0.25, 0.25]
        indices = _weighted_sample_indices(probs, 2)
        assert len(indices) == 2
        assert len(set(indices)) == 2  # no duplicates

    def test_n_exceeds_length(self):
        probs = [0.5, 0.5]
        indices = _weighted_sample_indices(probs, 5)
        assert indices == [0, 1]

    def test_single_item(self):
        probs = [1.0]
        indices = _weighted_sample_indices(probs, 1)
        assert indices == [0]

    def test_heavily_skewed_weights(self):
        # Item 0 has 99% weight — should almost always be selected
        probs = [0.99, 0.005, 0.005]
        counts = Counter()
        for _ in range(200):
            indices = _weighted_sample_indices(probs, 1)
            counts[indices[0]] += 1
        assert counts[0] > 150  # should dominate

    def test_zero_weight_items_skipped(self):
        probs = [0.0, 0.0, 1.0]
        indices = _weighted_sample_indices(probs, 1)
        assert indices == [2]


class TestCurriculumSampler:
    def _make_pool(self, n):
        return [{"question": f"Question {i}?", "answer": f"Answer {i}"} for i in range(n)]

    def test_weighted_sample_prefers_hard_facts(self):
        sampler = CurriculumSampler()
        pool = self._make_pool(4)

        # Scores: Q0 easy (0.9), Q1 hard (0.1), Q2 hard (0.2), Q3 easy (0.8)
        scores = {
            "Question 0?": 0.9,
            "Question 1?": 0.1,
            "Question 2?": 0.2,
            "Question 3?": 0.8,
        }

        # Sample many times and count
        counts = Counter()
        for _ in range(500):
            sampled = sampler.weighted_sample(pool, scores, n_samples=2)
            for item in sampled:
                counts[item["question"]] += 1

        # Hard facts should be sampled more often
        assert counts["Question 1?"] > counts["Question 0?"]
        assert counts["Question 2?"] > counts["Question 3?"]

    def test_weighted_sample_empty_pool(self):
        sampler = CurriculumSampler()
        result = sampler.weighted_sample([], {}, n_samples=5)
        assert result == []

    def test_weighted_sample_no_scores(self):
        sampler = CurriculumSampler()
        pool = self._make_pool(3)
        # No scores = all items get weight 1.0 (uniform)
        result = sampler.weighted_sample(pool, {}, n_samples=2)
        assert len(result) == 2

    def test_weighted_sample_n_exceeds_pool(self):
        sampler = CurriculumSampler()
        pool = self._make_pool(2)
        result = sampler.weighted_sample(pool, {}, n_samples=5)
        assert len(result) == 2

    def test_exposure_tracking(self):
        sampler = CurriculumSampler()
        pool = self._make_pool(2)
        sampler.weighted_sample(pool, {}, n_samples=2)
        assert sampler.exposure_counts["Question 0?"] == 1
        assert sampler.exposure_counts["Question 1?"] == 1

        sampler.weighted_sample(pool, {}, n_samples=1)
        total = sampler.exposure_counts["Question 0?"] + sampler.exposure_counts["Question 1?"]
        assert total == 3

    def test_can_decay_respects_min_exposure(self):
        sampler = CurriculumSampler(min_exposure_cycles=3)
        assert not sampler.can_decay("Question 0?")
        sampler.exposure_counts["Question 0?"] = 2
        assert not sampler.can_decay("Question 0?")
        sampler.exposure_counts["Question 0?"] = 3
        assert sampler.can_decay("Question 0?")

    def test_update_history(self):
        sampler = CurriculumSampler()
        sampler.update_history({"Q1?": 0.8, "Q2?": 0.3})
        sampler.update_history({"Q1?": 0.9, "Q2?": 0.4})
        assert len(sampler.recall_history["Q1?"]) == 2
        assert sampler.recall_history["Q1?"] == [0.8, 0.9]

    def test_difficulty_score(self):
        sampler = CurriculumSampler()
        # No history = hardest
        assert sampler.get_difficulty_score("unknown") == 1.0

        sampler.recall_history["Q1?"] = [0.8, 0.6]
        # difficulty = 1 - mean(0.8, 0.6) = 1 - 0.7 = 0.3
        assert abs(sampler.get_difficulty_score("Q1?") - 0.3) < 0.01

    def test_difficulty_score_perfect_recall(self):
        sampler = CurriculumSampler()
        sampler.recall_history["Q1?"] = [1.0, 1.0]
        assert sampler.get_difficulty_score("Q1?") == 0.0
