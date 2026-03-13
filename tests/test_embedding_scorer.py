"""Tests for embedding-based similarity scoring."""

from paramem.evaluation.embedding_scorer import (
    compute_batch_similarity,
    compute_similarity,
)


class TestComputeSimilarity:
    def test_identical_strings(self):
        score = compute_similarity("Alex lives in Heilbronn.", "Alex lives in Heilbronn.")
        assert score > 0.95

    def test_semantically_similar(self):
        score = compute_similarity(
            "Alex lives in Heilbronn.",
            "Alex resides in Heilbronn.",
        )
        assert score > 0.80

    def test_factually_wrong(self):
        """Wrong answer should score lower than correct one."""
        correct = compute_similarity(
            "Alex lives in Heilbronn.",
            "Alex lives in Heilbronn.",
        )
        wrong = compute_similarity(
            "Alex lives in Heilbronn.",
            "Alex lives in the United States.",
        )
        assert correct > wrong

    def test_unrelated_strings(self):
        score = compute_similarity(
            "Alex lives in Heilbronn.",
            "The weather is nice today.",
        )
        assert score < 0.5

    def test_empty_string(self):
        assert compute_similarity("", "something") == 0.0
        assert compute_similarity("something", "") == 0.0


class TestBatchSimilarity:
    def test_batch_matches_individual(self):
        pairs = [
            ("Alex lives in Heilbronn.", "Alex lives in Heilbronn."),
            ("Alex works at AutoMate.", "Alex is employed at AutoMate."),
        ]
        batch_scores = compute_batch_similarity(pairs)
        individual_scores = [compute_similarity(a, b) for a, b in pairs]
        for bs, ind in zip(batch_scores, individual_scores):
            assert abs(bs - ind) < 0.01

    def test_empty_batch(self):
        assert compute_batch_similarity([]) == []
