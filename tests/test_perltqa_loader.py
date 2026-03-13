"""Tests for PerLTQA dataset loader."""

from experiments.utils.perltqa_loader import (
    FALLBACK_PATH,
    load_fallback_qa,
)


class TestFallbackLoader:
    """Test fallback to synthetic personal_facts.json."""

    def test_fallback_loads(self):
        """Fallback should load from personal_facts.json."""
        pairs = load_fallback_qa(max_pairs=10)
        assert len(pairs) > 0
        assert len(pairs) <= 10

    def test_fallback_format(self):
        """Each pair should have question and answer fields."""
        pairs = load_fallback_qa(max_pairs=5)
        for p in pairs:
            assert "question" in p
            assert "answer" in p
            assert isinstance(p["question"], str)
            assert isinstance(p["answer"], str)

    def test_fallback_max_pairs(self):
        """max_pairs should limit output."""
        pairs_5 = load_fallback_qa(max_pairs=5)
        pairs_10 = load_fallback_qa(max_pairs=10)
        assert len(pairs_5) <= 5
        assert len(pairs_10) <= 10
        assert len(pairs_10) >= len(pairs_5)

    def test_fallback_path_exists(self):
        """The fallback data file should exist."""
        assert FALLBACK_PATH.exists(), f"Fallback data not found: {FALLBACK_PATH}"
