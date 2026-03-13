"""Tests for the RAG baseline pipeline."""

import pytest

from paramem.evaluation.rag import (
    RAGPipeline,
    _chunk_transcript,
    format_rag_prompt,
)

# --- Chunking tests ---


class TestChunkTranscript:
    def test_paragraph_splitting(self):
        transcript = "First paragraph about Alex.\n\nSecond paragraph about work."
        chunks = _chunk_transcript(transcript, "session_001")
        assert len(chunks) == 2
        assert chunks[0].text == "First paragraph about Alex."
        assert chunks[1].text == "Second paragraph about work."
        assert all(c.session_id == "session_001" for c in chunks)

    def test_chunk_indices_sequential(self):
        transcript = "Para 1.\n\nPara 2.\n\nPara 3."
        chunks = _chunk_transcript(transcript, "s1")
        indices = [c.chunk_index for c in chunks]
        assert indices == [0, 1, 2]

    def test_long_paragraph_sliding_window(self):
        # Create a paragraph with 300 words
        long_text = " ".join(f"word{i}" for i in range(300))
        chunks = _chunk_transcript(long_text, "s1", chunk_size=200, chunk_overlap=50)
        assert len(chunks) == 2
        # First chunk: 200 words, second chunk starts at word 150 (overlap=50)
        assert len(chunks[0].text.split()) == 200
        assert len(chunks[1].text.split()) == 150  # words 150-299

    def test_empty_transcript(self):
        chunks = _chunk_transcript("", "s1")
        assert len(chunks) == 0

    def test_whitespace_only(self):
        chunks = _chunk_transcript("   \n\n   ", "s1")
        assert len(chunks) == 0

    def test_single_paragraph(self):
        transcript = "Alex lives in Heilbronn and works at AutoMate."
        chunks = _chunk_transcript(transcript, "s1")
        assert len(chunks) == 1
        assert chunks[0].text == transcript

    def test_session_id_preserved(self):
        transcript = "Fact one.\n\nFact two."
        chunks = _chunk_transcript(transcript, "session_042")
        for c in chunks:
            assert c.session_id == "session_042"


# --- RAGPipeline tests ---


class TestRAGPipeline:
    @pytest.fixture
    def sample_sessions(self):
        return [
            {
                "session_id": "session_001",
                "transcript": "Alex lives in Heilbronn, Germany. He moved there for work.",
            },
            {
                "session_id": "session_002",
                "transcript": (
                    "Alex works at AutoMate, a robotics company. He is a software engineer."
                ),
            },
            {
                "session_id": "session_003",
                "transcript": "Alex has a dog named Luna. Luna is a German Shepherd.",
            },
        ]

    def test_build_index(self, sample_sessions):
        pipeline = RAGPipeline()
        pipeline.build_index(sample_sessions)
        assert len(pipeline.chunks) == 3
        assert pipeline.embeddings is not None
        assert pipeline.embeddings.shape[0] == 3
        assert pipeline.embeddings.shape[1] == 384  # all-MiniLM-L6-v2 dim

    def test_build_index_empty(self):
        pipeline = RAGPipeline()
        pipeline.build_index([])
        assert len(pipeline.chunks) == 0
        assert pipeline.embeddings.shape[0] == 0

    def test_retrieve_returns_relevant(self, sample_sessions):
        pipeline = RAGPipeline()
        pipeline.build_index(sample_sessions)
        results = pipeline.retrieve("Where does Alex live?", top_k=1)
        assert len(results) == 1
        assert "Heilbronn" in results[0].text

    def test_retrieve_top_k(self, sample_sessions):
        pipeline = RAGPipeline()
        pipeline.build_index(sample_sessions)
        results = pipeline.retrieve("Tell me about Alex", top_k=2)
        assert len(results) == 2

    def test_retrieve_top_k_exceeds_chunks(self, sample_sessions):
        pipeline = RAGPipeline()
        pipeline.build_index(sample_sessions)
        results = pipeline.retrieve("Alex", top_k=10)
        assert len(results) == 3  # capped at available chunks

    def test_retrieve_texts(self, sample_sessions):
        pipeline = RAGPipeline()
        pipeline.build_index(sample_sessions)
        texts = pipeline.retrieve_texts("What pet does Alex have?", top_k=1)
        assert len(texts) == 1
        assert isinstance(texts[0], str)
        assert "Luna" in texts[0]

    def test_retrieve_empty_index(self):
        pipeline = RAGPipeline()
        pipeline.build_index([])
        results = pipeline.retrieve("anything", top_k=3)
        assert results == []

    def test_pet_query_retrieves_pet_chunk(self, sample_sessions):
        pipeline = RAGPipeline()
        pipeline.build_index(sample_sessions)
        results = pipeline.retrieve("Does Alex have any pets?", top_k=1)
        assert "Luna" in results[0].text or "German Shepherd" in results[0].text

    def test_work_query_retrieves_work_chunk(self, sample_sessions):
        pipeline = RAGPipeline()
        pipeline.build_index(sample_sessions)
        results = pipeline.retrieve("What company does Alex work at?", top_k=1)
        assert "AutoMate" in results[0].text


# --- Prompt formatting tests ---


class TestFormatRagPrompt:
    def _mock_tokenizer(self):
        """Create a minimal mock tokenizer with apply_chat_template."""

        class MockTokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                parts = []
                for msg in messages:
                    parts.append(f"<|{msg['role']}|>\n{msg['content']}")
                if add_generation_prompt:
                    parts.append("<|assistant|>")
                return "\n".join(parts)

        return MockTokenizer()

    def test_prompt_with_contexts(self):
        tokenizer = self._mock_tokenizer()
        prompt = format_rag_prompt(
            "Where does Alex live?",
            ["Alex lives in Heilbronn."],
            tokenizer,
        )
        assert "Heilbronn" in prompt
        assert "Where does Alex live?" in prompt
        assert "[Context 1]" in prompt

    def test_prompt_multiple_contexts(self):
        tokenizer = self._mock_tokenizer()
        prompt = format_rag_prompt(
            "Tell me about Alex",
            ["Lives in Heilbronn", "Works at AutoMate"],
            tokenizer,
        )
        assert "[Context 1]" in prompt
        assert "[Context 2]" in prompt
        assert "Heilbronn" in prompt
        assert "AutoMate" in prompt

    def test_prompt_no_contexts(self):
        tokenizer = self._mock_tokenizer()
        prompt = format_rag_prompt("Where does Alex live?", [], tokenizer)
        assert "Where does Alex live?" in prompt
        assert "[Context" not in prompt

    def test_prompt_uses_chat_template(self):
        tokenizer = self._mock_tokenizer()
        prompt = format_rag_prompt("Question?", ["Context."], tokenizer)
        assert "<|system|>" in prompt
        assert "<|user|>" in prompt
        assert "<|assistant|>" in prompt
