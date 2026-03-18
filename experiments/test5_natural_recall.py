"""Test 5: Natural Recall — unstructured access to parametric memory.

Tests whether facts trained via indexed keys are also accessible through
natural language prompts (without knowing the key). A personal assistant
should recall its knowledge naturally when asked open-ended questions.

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

from experiments.utils.perltqa_loader import load_qa  # noqa: E402
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
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
            "to", "for", "of", "and", "or", "i", "my", "he", "she", "it",
            "his", "her",
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


def main():
    parser = argparse.ArgumentParser(description="Test 5: Natural Recall")
    parser.add_argument("--num-pairs", type=int, default=50)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    add_model_args(parser)
    args = parser.parse_args()

    base_output_dir = Path(args.output_dir)

    qa_pairs, source = load_qa(max_pairs=args.num_pairs)
    logger.info("Loaded %d QA pairs from %s", len(qa_pairs), source)

    for bench_name, bench_model_config in get_benchmark_models(args):
        print(f"\n{'=' * 72}")
        print(f"  Model: {bench_name} ({bench_model_config.model_id})")
        print(f"{'=' * 72}")

        model, tokenizer, config = load_model_and_config(bench_model_config)
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
            "keyed_recall": recall_result,
            "natural_probes": None,
            "total_facts_recalled_naturally": 0,
        }
        save_results(results, output_dir)

        # Phase 3: Natural recall probes
        from paramem.evaluation.recall import generate_answer  # noqa: E402
        from paramem.models.loader import switch_adapter  # noqa: E402
        from paramem.training.dataset import _format_inference_prompt  # noqa: E402

        model.gradient_checkpointing_disable()
        switch_adapter(model, "episodic")

        print("\n--- Natural Recall Probes ---")
        natural_probes = []
        # Track which unique facts were recalled across all probes
        all_recalled_answers = set()

        for prompt_text in NATURAL_PROMPTS:
            prompt = _format_inference_prompt(prompt_text, tokenizer)
            generated = generate_answer(
                model,
                tokenizer,
                prompt,
                max_new_tokens=300,
                temperature=0.1,
                repetition_penalty=1.3,
            )
            matched = match_recalled_facts(generated, qa_pairs)
            for m in matched:
                all_recalled_answers.add(m["answer"])

            count = len(matched)
            print(f"  [{count:>2} facts] {prompt_text[:60]}")
            if matched:
                for mf in matched[:3]:
                    print(
                        f"           {mf['answer'][:50]}... "
                        f"(sim={mf['similarity']:.2f}, overlap={mf['keyword_overlap']:.2f})"
                    )

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
        results["natural_probes"] = natural_probes
        results["total_facts_recalled_naturally"] = unique_recalled
        results["natural_recall_rate"] = unique_recalled / len(qa_pairs)
        save_results(results, output_dir)

        # Summary
        print(f"\n{'=' * 72}")
        print(f"NATURAL RECALL SUMMARY ({bench_name})")
        print("=" * 72)
        print(f"  Facts trained: {len(qa_pairs)}")
        print(
            f"  Keyed recall:   {recall_result['exact_count']}/{recall_result['total']} "
            f"(structured, with indexed keys)"
        )
        print(
            f"  Natural recall: {unique_recalled}/{len(qa_pairs)} unique facts "
            f"surfaced across {len(NATURAL_PROMPTS)} open-ended prompts"
        )
        print(f"  Natural recall rate: {unique_recalled / len(qa_pairs):.1%}")
        print(f"  Training time: {train_time:.0f}s")
        print("=" * 72)

        unload_model(model, tokenizer)


if __name__ == "__main__":
    main()
