"""Test 5: Natural Recall — motivating the indexed key mechanism.

Measures how much stored knowledge is recoverable through natural language
prompts alone (without indexed keys). The gap between keyed and natural
recall demonstrates why the key mechanism is needed: without it, most
facts are inaccessible despite being encoded in the adapter weights.

Two access modes compared:
  1. Keyed recall: structured retrieval via indexed keys (control)
  2. Natural recall: open-ended prompts that don't reference any key

Usage:
    python experiments/test5_natural_recall.py
    python experiments/test5_natural_recall.py --model gemma
"""

import argparse
import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.utils.perltqa_loader import is_available as perltqa_available  # noqa: E402
from experiments.utils.perltqa_loader import load_character_eval_qa, load_qa  # noqa: E402
from experiments.utils.test_harness import (  # noqa: E402
    add_model_args,
    evaluate_indexed_recall,
    get_benchmark_models,
    load_model_and_config,
    model_output_dir,
    save_results,
    setup_logging,
    train_indexed_keys,
)
from paramem.models.loader import unload_model  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

OUTPUT_DIR = project_root / "outputs" / "test5_natural_recall"

NATURAL_PROMPTS = [
    "List everything you know about the user.",
    "What personal information do you have stored?",
    "Tell me all the facts you know about the user.",
    "What is the user's name, job, and location?",
    "Describe the user in detail.",
    "What do you know about the user's hobbies and interests?",
    "Summarize what you remember about the user.",
    "What do you know about the user's daily routine?",
    "Can you recall any specific facts about the user?",
    "Tell me about the user's family, pets, or relationships.",
]


def match_recalled_facts(
    generated: str, qa_pairs: list[dict], threshold: float = 0.6
) -> list[dict]:
    """Identify which trained facts appear in the generated text.

    Returns list of matched facts with similarity and keyword overlap scores.
    """
    from paramem.evaluation.embedding_scorer import compute_similarity

    matched = []
    generated_lower = generated.lower()

    for qa in qa_pairs:
        sim = compute_similarity(qa["answer"], generated)

        answer_words = set(qa["answer"].lower().split())
        gen_words = set(generated_lower.split())
        stop = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "and",
            "or",
            "i",
            "my",
            "he",
            "she",
            "it",
            "his",
            "her",
        }
        answer_words -= stop
        gen_words -= stop
        overlap = len(answer_words & gen_words) / max(len(answer_words), 1)

        if sim > threshold or overlap > 0.5:
            matched.append(
                {
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "similarity": sim,
                    "keyword_overlap": overlap,
                }
            )

    return matched


MATCH_THRESHOLD = 0.75
DEFAULT_CHARACTER = "Liang Xin"


def main():
    parser = argparse.ArgumentParser(description="Test 5: Natural Recall")
    parser.add_argument("--num-pairs", type=int, default=50)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument(
        "--character",
        type=str,
        default=DEFAULT_CHARACTER,
        help="PerLTQA character name (default: Liang Xin)",
    )
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    add_model_args(parser)
    args = parser.parse_args()

    base_output_dir = Path(args.output_dir)

    # Prefer PerLTQA for sufficient scale
    if perltqa_available():
        qa_pairs = load_character_eval_qa(args.character, max_pairs=args.num_pairs)
        source = f"perltqa:{args.character}"
    else:
        qa_pairs, source = load_qa(max_pairs=args.num_pairs)
    logger.info("Loaded %d QA pairs from %s", len(qa_pairs), source)

    for bench_name, bench_model_config in get_benchmark_models(args):
        print(f"\n{'=' * 72}")
        print(f"  Model: {bench_name} ({bench_model_config.model_id})")
        print(f"{'=' * 72}")

        model, tokenizer = load_model_and_config(bench_model_config)
        output_dir = model_output_dir(base_output_dir, bench_name)

        # Phase 1: Train adapter
        model, keyed_pairs, registry, train_time, metrics = train_indexed_keys(
            model,
            tokenizer,
            qa_pairs,
            epochs=args.num_epochs,
            rank=args.rank,
            adapter_name="episodic",
            output_dir=output_dir,
            run_name="natural-recall",
            skip_distill=True,  # Eval QA pairs are already well-formed
        )

        # Phase 2: Keyed recall (control)
        recall_result = evaluate_indexed_recall(
            model,
            tokenizer,
            keyed_pairs,
            registry,
            adapter_name="episodic",
        )
        print(
            f"\nKeyed recall: {recall_result['exact_count']}/{recall_result['total']} "
            f"(conf={recall_result['mean_confidence']:.3f})"
        )

        # Save keyed_pairs for resume
        kp_serializable = [
            {"key": kp["key"], "question": kp["question"], "answer": kp["answer"]}
            for kp in keyed_pairs
        ]
        with open(output_dir / "keyed_pairs.json", "w") as f:
            json.dump(kp_serializable, f, indent=2)

        # Phase save after keyed recall
        results = {
            "experiment": "test5_natural_recall",
            "model": bench_name,
            "model_id": bench_model_config.model_id,
            "num_pairs": len(qa_pairs),
            "data_source": source,
            "epochs": args.num_epochs,
            "rank": args.rank,
            "training_time_seconds": train_time,
            "training_loss": metrics.get("train_loss"),
            "keyed_recall": recall_result,
            "natural_probes": None,
            "total_facts_recalled_naturally": 0,
        }
        save_results(results, output_dir)

        # Phase 3: Per-question natural recall (fair comparison with keyed)
        from paramem.evaluation.embedding_scorer import compute_similarity  # noqa: E402
        from paramem.evaluation.recall import generate_answer  # noqa: E402
        from paramem.models.loader import switch_adapter  # noqa: E402
        from paramem.training.dataset import _format_inference_prompt  # noqa: E402

        model.gradient_checkpointing_disable()
        switch_adapter(model, "episodic")

        # Compute embedding similarity for keyed recall (unified metric)
        keyed_similarities = []
        for pk in recall_result["per_key"]:
            recalled = pk.get("recalled")
            if recalled and "answer" in recalled:
                kp_match = next((kp for kp in keyed_pairs if kp["key"] == pk["key"]), None)
                if kp_match:
                    sim = compute_similarity(kp_match["answer"], recalled["answer"])
                else:
                    sim = 0.0
            else:
                sim = 0.0
            pk["embedding_similarity"] = sim
            keyed_similarities.append(sim)

        keyed_mean_sim = (
            sum(keyed_similarities) / len(keyed_similarities) if keyed_similarities else 0.0
        )
        keyed_match_count = sum(1 for s in keyed_similarities if s >= MATCH_THRESHOLD)

        print(
            f"\nKeyed recall (embedding sim): "
            f"{keyed_match_count}/{len(keyed_pairs)} match "
            f"(sim={keyed_mean_sim:.3f})"
        )

        # Per-question natural recall: ask each training question one at a time
        print("\n--- Per-Question Natural Recall ---")
        per_question_results = []
        pq_match_count = 0

        for kp in keyed_pairs:
            prompt = _format_inference_prompt(kp["question"], tokenizer)
            generated = generate_answer(
                model,
                tokenizer,
                prompt,
                max_new_tokens=200,
                temperature=0.0,
            )
            sim = compute_similarity(kp["answer"], generated)
            is_match = sim >= MATCH_THRESHOLD
            if is_match:
                pq_match_count += 1
            per_question_results.append(
                {
                    "key": kp["key"],
                    "question": kp["question"],
                    "expected": kp["answer"],
                    "generated": generated,
                    "similarity": sim,
                    "match": is_match,
                }
            )

        pq_mean_sim = (
            sum(r["similarity"] for r in per_question_results) / len(per_question_results)
            if per_question_results
            else 0.0
        )

        print(
            f"  Per-question natural: {pq_match_count}/{len(keyed_pairs)} match "
            f"(sim={pq_mean_sim:.3f})"
        )

        # Phase 4: Broad natural recall probes (multi-fact, supplementary)
        print("\n--- Broad Natural Recall Probes ---")
        natural_probes = []
        all_recalled_answers = set()

        for prompt_text in NATURAL_PROMPTS:
            prompt = _format_inference_prompt(prompt_text, tokenizer)
            generated = generate_answer(
                model,
                tokenizer,
                prompt,
                max_new_tokens=300,
                temperature=0.0,
            )
            # Score against keyed_pairs (post-distillation), not original qa_pairs
            # Note: broad probes use softer threshold (0.6 + keyword overlap)
            # than per-question/keyed (0.75 embedding only) — supplementary metric
            matched = match_recalled_facts(generated, keyed_pairs)
            for m in matched:
                all_recalled_answers.add(m["answer"])

            count = len(matched)
            print(f"  [{count:>2} facts] {prompt_text[:60]}")

            natural_probes.append(
                {
                    "prompt": prompt_text,
                    "generated": generated,
                    "matched_facts": matched,
                    "num_matched": count,
                }
            )

        unique_recalled = len(all_recalled_answers)

        # Final save
        results["keyed_embedding_similarity"] = {
            "mean_similarity": keyed_mean_sim,
            "match_count": keyed_match_count,
            "match_rate": keyed_match_count / max(len(keyed_pairs), 1),
            "threshold": MATCH_THRESHOLD,
        }
        results["per_question_natural"] = {
            "mean_similarity": pq_mean_sim,
            "match_count": pq_match_count,
            "match_rate": pq_match_count / max(len(keyed_pairs), 1),
            "total": len(keyed_pairs),
            "threshold": MATCH_THRESHOLD,
            "per_question": per_question_results,
        }
        results["broad_natural_probes"] = natural_probes
        results["broad_natural_unique_facts"] = unique_recalled
        results["broad_natural_rate"] = unique_recalled / max(len(keyed_pairs), 1)
        results["scoring_metric"] = "embedding_similarity (all-MiniLM-L6-v2 cosine)"
        save_results(results, output_dir)

        # Summary
        total = len(keyed_pairs)
        print(f"\n{'=' * 72}")
        print(f"NATURAL RECALL SUMMARY ({bench_name})")
        print("=" * 72)
        print(f"  Facts trained: {total}")
        print(
            f"  Keyed recall (SimHash):  "
            f"{recall_result['exact_count']}/{total} "
            f"(conf={recall_result['mean_confidence']:.3f})"
        )
        print(f"  Keyed recall (embed):    {keyed_match_count}/{total} (sim={keyed_mean_sim:.3f})")
        print(f"  Per-question natural:    {pq_match_count}/{total} (sim={pq_mean_sim:.3f})")
        print(
            f"  Broad natural (unique):  "
            f"{unique_recalled}/{total} "
            f"across {len(NATURAL_PROMPTS)} prompts"
        )
        print(f"  Training time: {train_time:.0f}s")
        print("=" * 72)

        unload_model(model, tokenizer)


if __name__ == "__main__":
    main()
