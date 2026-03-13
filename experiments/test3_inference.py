"""Test 3: Associative Inference — reasoning over implicitly connected facts.

Trains base facts as indexed keys, then queries with inference questions
that require combining 2-3 trained facts. Tests whether knowledge encoded
in adapter weights enables implicit associative reasoning.

Honest expectation: indexed keys require explicit key lookup, so this may
show parametric memory does NOT do well on inference. That's a valid and
publishable negative result.

Usage:
    python experiments/test3_inference.py
    python experiments/test3_inference.py --skip-rag
"""

import argparse
import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.utils.test_harness import (  # noqa: E402
    evaluate_indexed_recall,
    load_model_and_config,
    save_results,
    setup_logging,
    train_indexed_keys,
)

setup_logging()
logger = logging.getLogger(__name__)

DATA_PATH = project_root / "data" / "synthetic" / "inference_facts.json"
OUTPUT_DIR = project_root / "outputs" / "test3_inference"


def main():
    parser = argparse.ArgumentParser(description="Test 3: Associative Inference")
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--skip-rag", action="store_true")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    with open(DATA_PATH) as f:
        data = json.load(f)

    base_facts = data["base_facts"]
    inference_questions = data["inference_questions"]

    qa_pairs = [{"question": f["question"], "answer": f["answer"]} for f in base_facts]
    logger.info(
        "Loaded %d base facts, %d inference questions", len(qa_pairs), len(inference_questions)
    )

    model, tokenizer, config = load_model_and_config()

    # Train base facts
    model, keyed_pairs, registry, train_time, metrics = train_indexed_keys(
        model,
        tokenizer,
        qa_pairs,
        epochs=args.num_epochs,
        rank=args.rank,
        adapter_name="episodic",
        output_dir=output_dir,
        run_name="inference-base-facts",
    )

    # Test A: Indexed key recall (sanity check — base facts should recall)
    print("\n--- Base Fact Recall (sanity check) ---")
    recall_result = evaluate_indexed_recall(
        model,
        tokenizer,
        keyed_pairs,
        registry,
        adapter_name="episodic",
    )
    print(f"  Base facts: {recall_result['exact_count']}/{recall_result['total']} exact recall")

    # Test B: Inference questions (NOT trained, require combining facts)
    print("\n--- Inference Questions (untrained, require fact combination) ---")
    from paramem.evaluation.embedding_scorer import compute_similarity  # noqa: E402
    from paramem.evaluation.recall import generate_answer  # noqa: E402
    from paramem.models.loader import switch_adapter  # noqa: E402
    from paramem.training.dataset import _format_inference_prompt  # noqa: E402

    model.gradient_checkpointing_disable()
    switch_adapter(model, "episodic")

    inference_results = []
    for iq in inference_questions:
        prompt = _format_inference_prompt(iq["question"], tokenizer)
        generated = generate_answer(
            model,
            tokenizer,
            prompt,
            max_new_tokens=150,
            temperature=0.1,
            repetition_penalty=1.3,
        )
        similarity = compute_similarity(iq["expected_answer"], generated)

        inference_results.append(
            {
                "id": iq["id"],
                "question": iq["question"],
                "expected": iq["expected_answer"],
                "generated": generated,
                "similarity": similarity,
                "source_facts": iq["source_facts"],
                "reasoning": iq["reasoning"],
            }
        )

        status = "OK" if similarity > 0.5 else "WEAK" if similarity > 0.3 else "MISS"
        print(f"  [{status}] {iq['question']}")
        print(f"       Expected: {iq['expected_answer'][:80]}...")
        print(f"       Got:      {generated[:80]}...")
        print(f"       Score:    {similarity:.3f}")

    mean_inference = (
        sum(r["similarity"] for r in inference_results) / len(inference_results)
        if inference_results
        else 0.0
    )
    ok_count = sum(1 for r in inference_results if r["similarity"] > 0.5)

    # Test C: RAG baseline on inference questions
    rag_result = None
    if not args.skip_rag:
        from paramem.evaluation.rag_qa import QARAGPipeline  # noqa: E402

        print("\n--- RAG Baseline on Inference Questions ---")
        rag = QARAGPipeline()
        rag.build_index(qa_pairs)

        model.disable_adapter_layers()

        rag_inference = []
        for iq in inference_questions:
            prompt = rag.format_prompt(iq["question"], tokenizer, top_k=5)
            generated = generate_answer(
                model,
                tokenizer,
                prompt,
                max_new_tokens=150,
                temperature=0.1,
                repetition_penalty=1.3,
            )
            similarity = compute_similarity(iq["expected_answer"], generated)
            rag_inference.append(
                {
                    "id": iq["id"],
                    "question": iq["question"],
                    "generated": generated,
                    "similarity": similarity,
                }
            )

            status = "OK" if similarity > 0.5 else "WEAK" if similarity > 0.3 else "MISS"
            print(f"  [{status}] {iq['question']}: {similarity:.3f}")

        rag_mean = (
            sum(r["similarity"] for r in rag_inference) / len(rag_inference)
            if rag_inference
            else 0.0
        )
        rag_ok = sum(1 for r in rag_inference if r["similarity"] > 0.5)

        rag_result = {
            "mean_similarity": rag_mean,
            "ok_count": rag_ok,
            "total": len(rag_inference),
            "per_question": rag_inference,
        }

        model.enable_adapter_layers()

    # Summary
    print("\n" + "=" * 72)
    print("ASSOCIATIVE INFERENCE SUMMARY")
    print("=" * 72)
    print(f"  Base fact recall: {recall_result['exact_count']}/{recall_result['total']}")
    print(
        f"  Inference (parametric): {ok_count}/{len(inference_results)} OK (>{0.5}), "
        f"mean={mean_inference:.3f}"
    )
    if rag_result:
        print(
            f"  Inference (RAG):        {rag_result['ok_count']}/{rag_result['total']} OK, "
            f"mean={rag_result['mean_similarity']:.3f}"
        )
    print(f"  Training time: {train_time:.0f}s")
    print("=" * 72)

    results = {
        "experiment": "test3_inference",
        "epochs": args.num_epochs,
        "rank": args.rank,
        "training_time_seconds": train_time,
        "training_loss": metrics.get("train_loss"),
        "base_fact_recall": recall_result,
        "inference_parametric": {
            "mean_similarity": mean_inference,
            "ok_count": ok_count,
            "total": len(inference_results),
            "per_question": inference_results,
        },
        "inference_rag": rag_result,
    }

    save_results(results, output_dir)


if __name__ == "__main__":
    main()
