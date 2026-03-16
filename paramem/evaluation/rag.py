"""RAG (Retrieval-Augmented Generation) baseline for comparison against parametric memory.

Indexes session transcripts as chunks, retrieves relevant context for queries,
and generates answers using the base model with retrieved context prepended.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = (
    "You are a personal assistant with memory of your user's life. "
    "Answer questions about the user based on the context provided. "
    "If the context doesn't contain relevant information, say you don't know."
)


@dataclass
class Chunk:
    """A chunk of text from a session transcript."""

    text: str
    session_id: str
    chunk_index: int


@dataclass
class RAGPipeline:
    """Simple RAG pipeline: chunk, embed, retrieve, generate.

    Uses numpy cosine similarity for retrieval — sufficient for
    hundreds of chunks at personal-memory scale.
    """

    chunks: list[Chunk] = field(default_factory=list)
    embeddings: np.ndarray | None = None
    _model: object = None

    def _get_embedding_model(self):
        """Lazy-load sentence-transformers model (shared with embedding_scorer)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            logger.info("Loaded embedding model for RAG: all-MiniLM-L6-v2 (CPU)")
        return self._model

    def build_index(
        self,
        sessions: list[dict],
        chunk_size: int = 200,
        chunk_overlap: int = 50,
    ) -> None:
        """Chunk session transcripts and build embedding index.

        Args:
            sessions: List of dicts with 'session_id' and 'transcript' keys.
            chunk_size: Max words per chunk when paragraph splitting isn't available.
            chunk_overlap: Word overlap between sliding-window chunks.
        """
        self.chunks = []
        for session in sessions:
            session_id = session["session_id"]
            transcript = session["transcript"]
            session_chunks = _chunk_transcript(transcript, session_id, chunk_size, chunk_overlap)
            self.chunks.extend(session_chunks)

        if not self.chunks:
            logger.warning("No chunks created from %d sessions", len(sessions))
            self.embeddings = np.empty((0, 384))
            return

        texts = [c.text for c in self.chunks]
        model = self._get_embedding_model()
        self.embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        logger.info(
            "Built RAG index: %d chunks from %d sessions, embedding shape %s",
            len(self.chunks),
            len(sessions),
            self.embeddings.shape,
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[Chunk]:
        """Retrieve top-k most relevant chunks for a query.

        Returns chunks sorted by descending similarity.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        model = self._get_embedding_model()
        query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        # Cosine similarity (embeddings are already normalized)
        similarities = (self.embeddings @ query_emb.T).squeeze()

        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [self.chunks[i] for i in top_indices]

    def retrieve_texts(self, query: str, top_k: int = 3) -> list[str]:
        """Retrieve top-k chunk texts for a query."""
        return [c.text for c in self.retrieve(query, top_k)]


def format_rag_prompt(
    question: str,
    contexts: list[str],
    tokenizer,
) -> str:
    """Format a RAG prompt using the model's native chat template.

    Prepends retrieved context to the user's question.
    """
    if contexts:
        context_block = "\n\n".join(f"[Context {i + 1}]\n{c}" for i, c in enumerate(contexts))
        user_content = (
            f"Here is some relevant context about the user:\n\n"
            f"{context_block}\n\n"
            f"Based on this context, answer the following question:\n{question}"
        )
    else:
        user_content = question

    from paramem.models.loader import adapt_messages

    messages = adapt_messages(
        [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        tokenizer,
    )
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _chunk_transcript(
    transcript: str,
    session_id: str,
    chunk_size: int = 200,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Split a transcript into chunks.

    First tries paragraph-level splitting (double newline). If paragraphs are
    too long, falls back to sliding-window word splitting.
    """
    paragraphs = [p.strip() for p in transcript.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [transcript.strip()] if transcript.strip() else []

    chunks = []
    chunk_index = 0

    for para in paragraphs:
        words = para.split()
        if len(words) <= chunk_size:
            chunks.append(Chunk(text=para, session_id=session_id, chunk_index=chunk_index))
            chunk_index += 1
        else:
            start = 0
            while start < len(words):
                end = start + chunk_size
                chunk_text = " ".join(words[start:end])
                chunk = Chunk(text=chunk_text, session_id=session_id, chunk_index=chunk_index)
                chunks.append(chunk)
                chunk_index += 1
                start += chunk_size - chunk_overlap

    return chunks
