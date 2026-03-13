"""Curriculum-aware replay sampling.

Prioritizes under-learned facts during replay by probing the model's
recall on each replay pool item before sampling. Facts with low recall
get higher sampling probability.
"""

import logging
import random
from collections import defaultdict

from paramem.evaluation.embedding_scorer import compute_similarity
from paramem.evaluation.recall import generate_answer
from paramem.training.dataset import _format_inference_prompt

logger = logging.getLogger(__name__)


class CurriculumSampler:
    """Recall-weighted replay sampler.

    Before each training cycle, probes the adapter on replay pool items.
    Facts with low recall scores get higher sampling probability.
    Tracks per-fact recall history across cycles for difficulty scoring.
    """

    def __init__(self, min_exposure_cycles: int = 5):
        self.min_exposure_cycles = min_exposure_cycles
        # question -> list of recall scores across cycles
        self.recall_history: dict[str, list[float]] = defaultdict(list)
        # question -> number of training exposures
        self.exposure_counts: dict[str, int] = defaultdict(int)

    def probe_recall(
        self,
        model,
        tokenizer,
        replay_pool: list[dict],
        max_probe: int = 50,
    ) -> dict[str, float]:
        """Probe model recall on replay pool items.

        Args:
            model: The adapted model (gradient checkpointing must be disabled).
            tokenizer: Model tokenizer.
            replay_pool: List of dicts with 'question' and 'answer' keys.
            max_probe: Maximum items to probe (caps latency).

        Returns:
            Dict mapping question -> recall score (0-1).
        """
        if not replay_pool:
            return {}

        pool_to_probe = replay_pool
        if len(pool_to_probe) > max_probe:
            pool_to_probe = random.sample(replay_pool, max_probe)

        scores = {}
        for item in pool_to_probe:
            prompt = _format_inference_prompt(item["question"], tokenizer)
            generated = generate_answer(model, tokenizer, prompt, temperature=0.1)
            score = compute_similarity(item["answer"], generated)
            scores[item["question"]] = score

        logger.info(
            "Curriculum probe: %d items, mean recall %.3f",
            len(scores),
            sum(scores.values()) / len(scores) if scores else 0.0,
        )
        return scores

    def update_history(self, recall_scores: dict[str, float]) -> None:
        """Record recall scores for this cycle."""
        for question, score in recall_scores.items():
            self.recall_history[question].append(score)

    def weighted_sample(
        self,
        replay_pool: list[dict],
        recall_scores: dict[str, float],
        n_samples: int = 8,
    ) -> list[dict]:
        """Sample from replay pool with probability weighted by difficulty.

        Items with lower recall scores get higher sampling probability.
        Items not in recall_scores (not probed) get maximum weight.

        Args:
            replay_pool: Full replay pool.
            recall_scores: Question -> recall score from probe_recall.
            n_samples: Number of items to sample.

        Returns:
            Sampled items from the pool.
        """
        if not replay_pool:
            return []

        n_samples = min(n_samples, len(replay_pool))

        # Weight = 1 - recall_score (harder facts get higher weight)
        # Unprobed items get weight 1.0 (assume unknown = needs practice)
        weights = []
        for item in replay_pool:
            score = recall_scores.get(item["question"], 0.0)
            weight = max(1.0 - score, 0.05)  # floor at 0.05 to avoid zero
            weights.append(weight)

        # Normalize to probabilities
        total = sum(weights)
        probabilities = [w / total for w in weights]

        # Weighted sampling without replacement
        indices = _weighted_sample_indices(probabilities, n_samples)
        sampled = [replay_pool[i] for i in indices]

        # Track exposure
        for item in sampled:
            self.exposure_counts[item["question"]] += 1

        return sampled

    def can_decay(self, question: str) -> bool:
        """Check if a fact has met the minimum exposure guarantee."""
        return self.exposure_counts.get(question, 0) >= self.min_exposure_cycles

    def get_difficulty_score(self, question: str) -> float:
        """Get average difficulty (1 - mean recall) for a fact.

        Returns 1.0 (hardest) if no history exists.
        """
        history = self.recall_history.get(question, [])
        if not history:
            return 1.0
        return 1.0 - (sum(history) / len(history))


def _weighted_sample_indices(probabilities: list[float], n: int) -> list[int]:
    """Sample n indices without replacement, weighted by probabilities."""
    if n >= len(probabilities):
        return list(range(len(probabilities)))

    # Use reservoir-style weighted sampling
    indices = list(range(len(probabilities)))
    selected = []
    remaining_probs = list(probabilities)
    remaining_indices = list(indices)

    for _ in range(n):
        total = sum(remaining_probs)
        if total <= 0:
            break
        normalized = [p / total for p in remaining_probs]
        r = random.random()
        cumulative = 0.0
        chosen = len(remaining_probs) - 1
        for i, p in enumerate(normalized):
            cumulative += p
            if r <= cumulative:
                chosen = i
                break
        selected.append(remaining_indices[chosen])
        remaining_probs.pop(chosen)
        remaining_indices.pop(chosen)

    return selected
