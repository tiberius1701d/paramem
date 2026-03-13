"""Evaluation harness for personal memory recall."""

import logging
from typing import Optional

import torch
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizer

from paramem.training.dataset import load_eval_pairs

logger = logging.getLogger(__name__)


def generate_answer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.1,
    repetition_penalty: float = 1.0,
) -> str:
    """Generate an answer from the model given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Build stop token list: eos_token + chat template end tokens (e.g. <|im_end|>)
    stop_ids = [tokenizer.eos_token_id]
    for token_name in ["<|im_end|>", "<|eot_id|>"]:
        encoded = tokenizer.encode(token_name, add_special_tokens=False)
        if len(encoded) == 1 and encoded[0] not in stop_ids:
            stop_ids.append(encoded[0])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_ids,
            repetition_penalty=repetition_penalty,
        )

    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


_STOP_WORDS = {
    "i", "a", "an", "the", "is", "am", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "about", "and",
    "but", "or", "yes", "no", "my", "your", "it", "that", "this",
}  # fmt: skip


def _score_pair(expected: str, generated: str, method: str = "keyword") -> float:
    """Score a single expected/generated pair.

    Args:
        method: "keyword" for keyword overlap, "embedding" for cosine similarity.
    """
    if method == "embedding":
        from paramem.evaluation.embedding_scorer import compute_similarity

        return compute_similarity(expected, generated)

    # Keyword overlap (legacy)
    expected_kw = set(expected.lower().split()) - _STOP_WORDS
    generated_kw = set(generated.lower().split()) - _STOP_WORDS
    if not expected_kw:
        return 1.0 if not generated_kw else 0.0
    overlap = expected_kw & generated_kw
    return len(overlap) / len(expected_kw)


def evaluate_recall(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    data_path: str,
    fact_ids: Optional[list[str]] = None,
    adapter_name: Optional[str] = None,
    max_new_tokens: int = 128,
    scoring_method: str = "keyword",
) -> dict:
    """Evaluate how well the model recalls personal facts.

    If adapter_name is provided and model is a PeftModel, switches to that
    adapter before evaluation. Pass adapter_name=None and disable adapters
    externally to test the base model.

    Args:
        scoring_method: "keyword" for keyword overlap, "embedding" for
            sentence-transformers cosine similarity.

    Returns a dict with per-fact and aggregate recall metrics.
    """
    if adapter_name is not None and isinstance(model, PeftModel):
        model.set_adapter(adapter_name)

    eval_pairs = load_eval_pairs(data_path, tokenizer=tokenizer, fact_ids=fact_ids)
    results = []

    for pair in eval_pairs:
        generated = generate_answer(model, tokenizer, pair["prompt"], max_new_tokens=max_new_tokens)

        overlap_score = _score_pair(pair["expected_answer"], generated, scoring_method)

        results.append(
            {
                "fact_id": pair["fact_id"],
                "category": pair["category"],
                "question": pair["question"],
                "expected": pair["expected_answer"],
                "generated": generated,
                "overlap_score": overlap_score,
            }
        )

    # Aggregate metrics
    scores = [r["overlap_score"] for r in results]
    by_category = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r["overlap_score"])

    category_scores = {cat: sum(s) / len(s) for cat, s in by_category.items()}

    by_fact = {}
    for r in results:
        fid = r["fact_id"]
        if fid not in by_fact:
            by_fact[fid] = []
        by_fact[fid].append(r["overlap_score"])

    fact_scores = {fid: sum(s) / len(s) for fid, s in by_fact.items()}

    summary = {
        "mean_recall": sum(scores) / len(scores) if scores else 0.0,
        "num_questions": len(scores),
        "perfect_recall_rate": sum(1 for s in scores if s >= 0.8) / len(scores) if scores else 0.0,
        "category_scores": category_scores,
        "fact_scores": fact_scores,
        "details": results,
    }

    logger.info(
        "Recall evaluation: mean=%.3f, perfect_rate=%.3f (%d questions)",
        summary["mean_recall"],
        summary["perfect_recall_rate"],
        summary["num_questions"],
    )

    return summary


def compare_base_vs_adapted(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    data_path: str,
    adapter_name: str,
    fact_ids: Optional[list[str]] = None,
) -> dict:
    """Compare recall between base model and adapted model.

    Returns a dict with both sets of results and the delta.
    """
    # Evaluate with adapter disabled (base model)
    model.disable_adapter_layers()
    base_results = evaluate_recall(model, tokenizer, data_path, fact_ids=fact_ids)
    model.enable_adapter_layers()

    # Evaluate with adapter active
    adapted_results = evaluate_recall(
        model, tokenizer, data_path, fact_ids=fact_ids, adapter_name=adapter_name
    )

    delta = adapted_results["mean_recall"] - base_results["mean_recall"]

    comparison = {
        "base_recall": base_results["mean_recall"],
        "adapted_recall": adapted_results["mean_recall"],
        "delta": delta,
        "base_details": base_results,
        "adapted_details": adapted_results,
    }

    logger.info(
        "Base vs adapted: base=%.3f, adapted=%.3f, delta=+%.3f",
        base_results["mean_recall"],
        adapted_results["mean_recall"],
        delta,
    )

    return comparison
