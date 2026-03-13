"""Tests for QA-optimized RAG baseline."""

import pytest

from paramem.evaluation.rag_qa import QARAGPipeline


class TestQARAGPipeline:
    """Test QA-RAG indexing and retrieval."""

    @pytest.fixture
    def sample_qa(self):
        return [
            {"question": "Where does Alex live?", "answer": "Heilbronn"},
            {"question": "What is Alex's job?", "answer": "Software engineer at AutoMate"},
            {"question": "What pet does Alex have?", "answer": "Luna the German Shepherd"},
            {"question": "What does Alex drink?", "answer": "Black coffee, no sugar"},
        ]

    def test_build_index(self, sample_qa):
        rag = QARAGPipeline()
        rag.build_index(sample_qa)
        assert len(rag.qa_pairs) == 4
        assert rag.embeddings is not None
        assert rag.embeddings.shape[0] == 4

    def test_build_empty_index(self):
        rag = QARAGPipeline()
        rag.build_index([])
        assert len(rag.qa_pairs) == 0
        assert rag.embeddings.shape == (0, 384)

    def test_retrieve_returns_relevant(self, sample_qa):
        rag = QARAGPipeline()
        rag.build_index(sample_qa)
        results = rag.retrieve("Where does Alex live?", top_k=1)
        assert len(results) == 1
        assert results[0]["answer"] == "Heilbronn"

    def test_retrieve_top_k(self, sample_qa):
        rag = QARAGPipeline()
        rag.build_index(sample_qa)
        results = rag.retrieve("Tell me about Alex", top_k=3)
        assert len(results) == 3

    def test_retrieve_empty_index(self):
        rag = QARAGPipeline()
        rag.build_index([])
        results = rag.retrieve("Where does Alex live?")
        assert results == []

    def test_retrieve_pet_query(self, sample_qa):
        rag = QARAGPipeline()
        rag.build_index(sample_qa)
        results = rag.retrieve("Does Alex have a dog?", top_k=1)
        assert "Luna" in results[0]["answer"] or "Shepherd" in results[0]["answer"]

    def test_format_prompt_includes_facts(self, sample_qa):
        """Test that format_prompt includes retrieved facts in the prompt."""

        class MockTokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                return str(messages)

        rag = QARAGPipeline()
        rag.build_index(sample_qa)
        prompt = rag.format_prompt("Where does Alex live?", MockTokenizer(), top_k=2)
        assert "Fact 1" in prompt
        assert "Fact 2" in prompt
