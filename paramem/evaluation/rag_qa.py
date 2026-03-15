"""QA-optimized RAG baseline for fair comparison against parametric memory.

Unlike the transcript-chunking RAG in rag.py, this indexes pre-extracted
QA pairs directly — the same facts that parametric memory trains on.
This isolates the storage mechanism as the only difference.
"""

import logging

import numpy as np

from paramem.evaluation.rag import RAG_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class QARAGPipeline:
    """RAG pipeline that indexes QA pairs instead of transcript chunks.

    Fair comparison: same base model, same facts, only the storage
    mechanism (weights vs. retrieval) differs.
    """

    def __init__(self):
        self.qa_pairs: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self._model = None

    def _get_embedding_model(self):
        """Lazy-load sentence-transformers model (CPU to avoid VRAM contention)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            logger.info("Loaded embedding model for QA-RAG: all-MiniLM-L6-v2 (CPU)")
        return self._model

    def build_index(self, qa_pairs: list[dict]) -> None:
        """Index QA pairs for retrieval.

        Args:
            qa_pairs: List of {"question": str, "answer": str} dicts.
        """
        self.qa_pairs = list(qa_pairs)
        if not self.qa_pairs:
            self.embeddings = np.empty((0, 384))
            return

        # Embed the combined question+answer for each pair
        texts = [f"{qa['question']} {qa['answer']}" for qa in self.qa_pairs]
        model = self._get_embedding_model()
        self.embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        logger.info("Built QA-RAG index: %d pairs", len(self.qa_pairs))

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve top-k most relevant QA pairs for a query."""
        if self.embeddings is None or len(self.qa_pairs) == 0:
            return []

        model = self._get_embedding_model()
        query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        similarities = (self.embeddings @ query_emb.T).squeeze()
        top_k = min(top_k, len(self.qa_pairs))
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [self.qa_pairs[i] for i in top_indices]

    def format_prompt(
        self,
        question: str,
        tokenizer,
        top_k: int = 3,
    ) -> str:
        """Retrieve relevant QA pairs and format as a RAG prompt.

        Returns a formatted prompt string ready for generation.
        """
        retrieved = self.retrieve(question, top_k)

        if retrieved:
            context_parts = []
            for i, qa in enumerate(retrieved, 1):
                context_parts.append(f"[Fact {i}]\nQ: {qa['question']}\nA: {qa['answer']}")
            context_block = "\n\n".join(context_parts)
            user_content = (
                f"Here are some known facts about the user:\n\n"
                f"{context_block}\n\n"
                f"Based on these facts, answer the following question:\n{question}"
            )
        else:
            user_content = question

        from paramem.models.loader import adapt_messages

        messages = adapt_messages([
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ], tokenizer)
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def evaluate_rag_recall(
    rag: QARAGPipeline,
    model,
    tokenizer,
    questions: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """Evaluate RAG recall on a set of questions.

    Args:
        rag: Initialized QARAGPipeline with indexed QA pairs.
        model: Base model for generation.
        tokenizer: Tokenizer.
        questions: List of {"question": str, "expected_answer": str} dicts.
        top_k: Number of facts to retrieve per question.

    Returns:
        List of result dicts with question, expected, generated, similarity.
    """
    from paramem.evaluation.embedding_scorer import compute_similarity
    from paramem.evaluation.recall import generate_answer

    results = []
    for q in questions:
        prompt = rag.format_prompt(q["question"], tokenizer, top_k)
        generated = generate_answer(
            model,
            tokenizer,
            prompt,
            max_new_tokens=150,
            temperature=0.1,
            repetition_penalty=1.3,
        )
        similarity = compute_similarity(q["expected_answer"], generated)
        results.append(
            {
                "question": q["question"],
                "expected_answer": q["expected_answer"],
                "generated": generated,
                "similarity": similarity,
            }
        )

    return results
