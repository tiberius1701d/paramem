"""Test 3: Associative Inference — recall then reason over parametric memory.

Trains base facts from PerLTQA dialogues as indexed keys, then generates
inference questions from the knowledge graph (entity pairs connected by 2+ hops).
The LLM itself generates the inference questions.

Three evaluation conditions:
  (a) Parametric recall + reason: enumerate all keys → reconstruct facts →
      feed as context → ask inference question
  (b) Direct parametric (no reconstruction): ask with adapter active, no context
  (c) RAG baseline: retrieve relevant facts by similarity → answer

The comparison between (a) and (c) tests whether exhaustive recall from
parametric memory can match or exceed selective retrieval for reasoning tasks.

Usage:
    python experiments/test3_inference.py --model gemma
    python experiments/test3_inference.py --model mistral
    python experiments/test3_inference.py --model gemma --skip-rag
"""

import argparse
import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.utils.perltqa_loader import load_character_dialogues  # noqa: E402
from experiments.utils.test_harness import (  # noqa: E402
    add_model_args,
    distill_session,
    evaluate_indexed_recall,
    get_benchmark_models,
    load_model_and_config,
    model_output_dir,
    save_results,
    setup_logging,
    train_indexed_keys,
)
from paramem.evaluation.embedding_scorer import compute_similarity  # noqa: E402
from paramem.evaluation.recall import generate_answer  # noqa: E402
from paramem.models.loader import switch_adapter, unload_model  # noqa: E402
from paramem.training.dataset import _format_inference_prompt  # noqa: E402
from paramem.training.indexed_memory import probe_all_keys  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

OUTPUT_DIR = project_root / "outputs" / "test3_inference"

CHARACTER = "Liang Xin"
MAX_SESSIONS = 30
TARGET_QA = 50  # Train on ~50 facts, enough for a rich knowledge graph
NUM_INFERENCE_QUESTIONS = 15

# Prompt for the LLM to generate inference questions from fact pairs
INFERENCE_GEN_PROMPT = """\
You are given a list of facts about a person. Your task is to generate inference \
questions that require combining TWO OR MORE of these facts to answer.

Facts:
{facts}

Generate exactly {count} inference questions. Each question must:
- Require combining at least 2 of the listed facts
- NOT be answerable from a single fact alone
- Be a natural question someone might ask about this person
- Have a clear expected answer derivable from the listed facts

Output valid JSON only. Format:
[
  {{
    "question": "the inference question",
    "expected_answer": "answer derivable by combining the source facts",
    "source_facts": ["fact 1 text", "fact 2 text"]
  }}
]

Output ONLY the JSON array, nothing else."""


def generate_inference_questions(
    model,
    tokenizer,
    qa_pairs: list[dict],
    count: int = 15,
) -> list[dict]:
    """Use the LLM to generate inference questions from trained facts.

    The model reads all trained facts and generates questions that require
    combining 2+ facts to answer — the same way a human would recall
    individual memories and reason over them.
    """
    from paramem.models.loader import adapt_messages

    # Format facts as a numbered list
    facts_text = "\n".join(
        f"{i + 1}. Q: {qa['question']} A: {qa['answer']}" for i, qa in enumerate(qa_pairs)
    )

    prompt_text = INFERENCE_GEN_PROMPT.format(facts=facts_text, count=count)

    messages = adapt_messages(
        [
            {
                "role": "system",
                "content": "You are a precise question generator. Output valid JSON only.",
            },
            {"role": "user", "content": prompt_text},
        ],
        tokenizer,
    )
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    model.gradient_checkpointing_disable()
    has_adapters = hasattr(model, "disable_adapter_layers")
    if has_adapters:
        model.disable_adapter_layers()

    raw = generate_answer(
        model,
        tokenizer,
        formatted,
        max_new_tokens=2048,
        temperature=0.3,
        repetition_penalty=1.1,
    )

    if has_adapters:
        model.enable_adapter_layers()

    # Parse JSON from output
    questions = _parse_inference_json(raw)
    if not questions:
        logger.warning("Failed to parse inference questions from model output")
        logger.debug("Raw output: %s", raw[:500])

    return questions


def _parse_inference_json(raw: str) -> list[dict]:
    """Parse inference questions JSON from model output.

    Uses progressive extraction — find first '[', try each ']'.
    """
    start = raw.find("[")
    if start == -1:
        return []

    for end in range(len(raw) - 1, start, -1):
        if raw[end] == "]":
            try:
                data = json.loads(raw[start : end + 1])
                if isinstance(data, list) and len(data) > 0:
                    # Validate structure
                    valid = []
                    for item in data:
                        if "question" in item and "expected_answer" in item:
                            valid.append(
                                {
                                    "question": item["question"],
                                    "expected_answer": item["expected_answer"],
                                    "source_facts": item.get("source_facts", []),
                                }
                            )
                    return valid
            except json.JSONDecodeError:
                continue

    return []


def reconstruct_all_facts(
    model,
    tokenizer,
    keyed_pairs: list[dict],
    registry: dict[str, int],
    adapter_name: str = "episodic",
) -> list[dict]:
    """Enumerate all indexed keys and reconstruct stored facts.

    This is the core capability: the system knows what it knows and can
    materialize all stored knowledge for reasoning.

    Returns list of {"key", "question", "answer"} for successfully recalled facts.
    """
    model.gradient_checkpointing_disable()
    switch_adapter(model, adapter_name)

    trained_keys = [kp["key"] for kp in keyed_pairs]
    recalled = probe_all_keys(model, tokenizer, trained_keys, registry=registry)

    reconstructed = []
    for key, result in recalled.items():
        if result and result.get("question") and result.get("answer"):
            reconstructed.append(
                {
                    "key": key,
                    "question": result["question"],
                    "answer": result["answer"],
                    "confidence": result.get("confidence", 0.0),
                }
            )

    logger.info(
        "Reconstructed %d/%d facts from parametric memory",
        len(reconstructed),
        len(keyed_pairs),
    )
    return reconstructed


def format_recall_reason_prompt(
    question: str,
    reconstructed_facts: list[dict],
    tokenizer,
) -> str:
    """Format a prompt that feeds all reconstructed facts as context.

    Enumerate → reconstruct → reason: the model reads its own recalled
    knowledge and reasons over it to answer the inference question.
    """
    from paramem.models.loader import adapt_messages

    facts_block = "\n".join(f"- {f['question']} {f['answer']}" for f in reconstructed_facts)

    user_content = (
        f"Here is everything I know from memory:\n\n"
        f"{facts_block}\n\n"
        f"Based on these facts, answer the following question:\n{question}"
    )

    messages = adapt_messages(
        [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer based on the provided facts. Be concise."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        tokenizer,
    )
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def evaluate_condition(
    label: str,
    model,
    tokenizer,
    inference_questions: list[dict],
    prompt_fn,
) -> dict:
    """Evaluate a set of inference questions under a given condition.

    Args:
        label: Condition name for logging.
        prompt_fn: Callable(question_str) -> formatted_prompt_str.

    Returns dict with mean_similarity, ok_count, total, per_question.
    """
    results = []
    for iq in inference_questions:
        prompt = prompt_fn(iq["question"])
        generated = generate_answer(
            model,
            tokenizer,
            prompt,
            max_new_tokens=200,
            temperature=0.1,
            repetition_penalty=1.3,
        )
        similarity = compute_similarity(iq["expected_answer"], generated)

        results.append(
            {
                "question": iq["question"],
                "expected": iq["expected_answer"],
                "generated": generated,
                "similarity": similarity,
                "source_facts": iq.get("source_facts", []),
            }
        )

        status = "OK" if similarity > 0.5 else "WEAK" if similarity > 0.3 else "MISS"
        print(f"  [{status}] {iq['question'][:70]}")
        print(f"       Score: {similarity:.3f}")

    mean_sim = sum(r["similarity"] for r in results) / len(results) if results else 0.0
    ok_count = sum(1 for r in results if r["similarity"] > 0.5)

    print(f"  {label}: {ok_count}/{len(results)} OK (>0.5), mean={mean_sim:.3f}")
    return {
        "mean_similarity": mean_sim,
        "ok_count": ok_count,
        "total": len(results),
        "per_question": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Test 3: Associative Inference")
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--target-qa", type=int, default=TARGET_QA)
    parser.add_argument("--num-inference", type=int, default=NUM_INFERENCE_QUESTIONS)
    parser.add_argument("--skip-rag", action="store_true")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    add_model_args(parser)
    args = parser.parse_args()

    base_output_dir = Path(args.output_dir)

    # Load PerLTQA dialogues
    sessions = load_character_dialogues(CHARACTER, max_sessions=MAX_SESSIONS)
    if not sessions:
        print(f"ERROR: No dialogue sessions found for '{CHARACTER}'")
        sys.exit(1)
    print(f"Loaded {len(sessions)} dialogue sessions for '{CHARACTER}'")

    for bench_name, bench_model_config in get_benchmark_models(args):
        print(f"\n{'=' * 72}")
        print(f"  Test 3: Associative Inference — {bench_name}")
        print(f"  ({bench_model_config.model_id})")
        print(f"{'=' * 72}")

        model, tokenizer, config = load_model_and_config(bench_model_config)
        output_dir = model_output_dir(base_output_dir, bench_name)

        # Phase 1: Distill QA pairs from dialogues (same as Test 1)
        print(f"\n--- Phase 1: Distilling QA pairs from {len(sessions)} sessions ---")
        all_qa = []
        seen_questions = set()
        for session in sessions:
            new_pairs = distill_session(model, tokenizer, session, seen_questions)
            all_qa.extend(new_pairs)
            if len(all_qa) >= args.target_qa:
                break

        qa_pairs = all_qa[: args.target_qa]
        print(f"  Distilled {len(qa_pairs)} QA pairs for training")

        if len(qa_pairs) < 10:
            print("ERROR: Too few QA pairs extracted. Skipping model.")
            unload_model(model, tokenizer)
            continue

        # Phase 2: Train indexed keys
        print(f"\n--- Phase 2: Training {len(qa_pairs)} indexed keys ---")
        model, keyed_pairs, registry, train_time, metrics = train_indexed_keys(
            model,
            tokenizer,
            qa_pairs,
            epochs=args.num_epochs,
            rank=args.rank,
            adapter_name="episodic",
            output_dir=output_dir,
            run_name="inference-base-facts",
            skip_distill=True,  # Already distilled in Phase 1
        )
        print(f"  Training: {train_time:.0f}s, loss={metrics.get('train_loss', -1):.4f}")

        # Phase 3: Verify base fact recall (sanity check)
        print("\n--- Phase 3: Base fact recall (sanity check) ---")
        recall_result = evaluate_indexed_recall(
            model, tokenizer, keyed_pairs, registry, adapter_name="episodic"
        )
        print(f"  Base facts: {recall_result['exact_count']}/{recall_result['total']} exact recall")

        # Phase 4: Reconstruct all facts from parametric memory
        print("\n--- Phase 4: Reconstructing all facts from memory ---")
        reconstructed = reconstruct_all_facts(
            model, tokenizer, keyed_pairs, registry, adapter_name="episodic"
        )
        print(f"  Reconstructed {len(reconstructed)}/{len(keyed_pairs)} facts")

        # Phase 5: Generate inference questions using the LLM
        print(f"\n--- Phase 5: Generating {args.num_inference} inference questions ---")
        # Use reconstructed facts so the questions are about what the model actually knows
        inference_questions = generate_inference_questions(
            model, tokenizer, reconstructed, count=args.num_inference
        )
        print(f"  Generated {len(inference_questions)} inference questions")

        if not inference_questions:
            print("ERROR: No inference questions generated. Skipping evaluation.")
            save_results(
                {
                    "experiment": "test3_inference",
                    "model": bench_name,
                    "error": "No inference questions generated",
                    "qa_pairs_trained": len(qa_pairs),
                    "reconstructed": len(reconstructed),
                },
                output_dir,
            )
            unload_model(model, tokenizer)
            continue

        # Save generated questions for reproducibility
        save_results(
            {"inference_questions": inference_questions},
            output_dir,
            filename="inference_questions.json",
        )

        # Phase 6: Evaluate three conditions
        model.gradient_checkpointing_disable()

        # (a) Parametric recall + reason: use all reconstructed facts as context
        print("\n--- Condition A: Parametric Recall + Reason ---")
        switch_adapter(model, "episodic")
        result_a = evaluate_condition(
            "Recall+Reason",
            model,
            tokenizer,
            inference_questions,
            prompt_fn=lambda q: format_recall_reason_prompt(q, reconstructed, tokenizer),
        )

        # (b) Direct parametric: adapter active, no context (baseline)
        print("\n--- Condition B: Direct Parametric (no reconstruction) ---")
        switch_adapter(model, "episodic")
        result_b = evaluate_condition(
            "Direct Parametric",
            model,
            tokenizer,
            inference_questions,
            prompt_fn=lambda q: _format_inference_prompt(q, tokenizer),
        )

        # (c) RAG baseline
        rag_result = None
        if not args.skip_rag:
            from paramem.evaluation.rag_qa import QARAGPipeline  # noqa: E402

            print("\n--- Condition C: RAG Baseline ---")
            rag = QARAGPipeline()
            rag.build_index(qa_pairs)

            model.disable_adapter_layers()
            result_c = evaluate_condition(
                "RAG",
                model,
                tokenizer,
                inference_questions,
                prompt_fn=lambda q: rag.format_prompt(q, tokenizer, top_k=5),
            )
            model.enable_adapter_layers()
            rag_result = result_c

        # Summary
        print(f"\n{'=' * 72}")
        print("ASSOCIATIVE INFERENCE SUMMARY")
        print(f"{'=' * 72}")
        print(f"  Model:               {bench_name}")
        print(f"  Facts trained:       {len(qa_pairs)}")
        print(f"  Facts reconstructed: {len(reconstructed)}/{len(keyed_pairs)}")
        print(f"  Inference questions: {len(inference_questions)}")
        print(f"  Base fact recall:    {recall_result['exact_count']}/{recall_result['total']}")
        print(
            f"  (a) Recall+Reason:   {result_a['ok_count']}/{result_a['total']} OK, "
            f"mean={result_a['mean_similarity']:.3f}"
        )
        print(
            f"  (b) Direct Param:    {result_b['ok_count']}/{result_b['total']} OK, "
            f"mean={result_b['mean_similarity']:.3f}"
        )
        if rag_result:
            print(
                f"  (c) RAG:             {rag_result['ok_count']}/{rag_result['total']} OK, "
                f"mean={rag_result['mean_similarity']:.3f}"
            )
        print(f"  Training time:       {train_time:.0f}s")
        print(f"{'=' * 72}")

        results = {
            "experiment": "test3_inference",
            "model": bench_name,
            "model_id": bench_model_config.model_id,
            "character": CHARACTER,
            "epochs": args.num_epochs,
            "rank": args.rank,
            "qa_pairs_trained": len(qa_pairs),
            "facts_reconstructed": len(reconstructed),
            "inference_questions_generated": len(inference_questions),
            "training_time_seconds": train_time,
            "training_loss": metrics.get("train_loss"),
            "base_fact_recall": recall_result,
            "condition_a_recall_reason": result_a,
            "condition_b_direct_parametric": result_b,
            "condition_c_rag": rag_result,
        }

        save_results(results, output_dir)

        unload_model(model, tokenizer)


if __name__ == "__main__":
    main()
