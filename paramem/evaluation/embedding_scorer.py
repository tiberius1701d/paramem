"""Embedding-based similarity scoring for recall evaluation.

Uses sentence-transformers to compute cosine similarity between
expected and generated answers. More reliable than keyword overlap
because it penalizes factually wrong answers even when they share
surface-level vocabulary.
"""

import logging

logger = logging.getLogger(__name__)

# Lazy-loaded to avoid import cost and GPU memory when not needed
_model = None


def _get_model():
    """Lazy-load the sentence-transformers model on CPU."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        logger.info("Loaded embedding model: all-MiniLM-L6-v2 (CPU)")
    return _model


def compute_similarity(expected: str, generated: str) -> float:
    """Compute cosine similarity between expected and generated text.

    Returns a float in [0, 1] where 1.0 means identical meaning.
    Uses all-MiniLM-L6-v2 on CPU to avoid VRAM contention.
    """
    if not expected.strip() or not generated.strip():
        return 0.0

    model = _get_model()
    embeddings = model.encode(
        [expected, generated], convert_to_numpy=True, normalize_embeddings=True
    )
    similarity = float(embeddings[0] @ embeddings[1])
    # Clamp to [0, 1] — cosine similarity can be slightly negative
    return max(0.0, min(1.0, similarity))


def compute_batch_similarity(
    pairs: list[tuple[str, str]],
) -> list[float]:
    """Compute similarity for multiple (expected, generated) pairs.

    More efficient than calling compute_similarity in a loop
    because it batches the encoding.
    """
    if not pairs:
        return []

    model = _get_model()
    expected_texts = [p[0] for p in pairs]
    generated_texts = [p[1] for p in pairs]

    expected_emb = model.encode(expected_texts, convert_to_numpy=True, normalize_embeddings=True)
    generated_emb = model.encode(generated_texts, convert_to_numpy=True, normalize_embeddings=True)

    similarities = []
    for i in range(len(pairs)):
        sim = float(expected_emb[i] @ generated_emb[i])
        similarities.append(max(0.0, min(1.0, sim)))
    return similarities
