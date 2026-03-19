"""Test 3: Reasoning Quality Parity with RAG.

Trains base facts from PerLTQA dialogues as indexed keys, then generates
inference questions requiring 2+ facts. Compares PM and RAG end-to-end
as they would run in production:

  PM:  adapter-tuned model (always active) reconstructs facts from weights,
       then reasons over them with those facts as context.
  RAG: base model (adapter disabled) loads all facts from external store,
       then reasons over them with those facts as context.

Additionally, an adapter-only diagnostic runs the adapter-tuned model
directly on inference questions without explicit retrieval — showing
what the adapter learned beyond key-addressable recall.

Per-phase latency is measured for both PM and RAG pipelines.

Usage:
    python experiments/test3_inference.py --model gemma
    python experiments/test3_inference.py --model mistral
    python experiments/test3_inference.py --model gemma --skip-rag
"""

import argparse
import json
import logging
import sys
import time
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

# Shared system prompt for all context-based conditions (PM and RAG).
# Single source of truth — ensures prompt fairness across conditions.
CONTEXT_SYSTEM_PROMPT = (
    "You are a personal assistant with memory of your user's life. "
    "Answer questions about the user based on the context provided. "
    "Be concise."
)
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
    from peft import PeftModel

    if isinstance(model, PeftModel):
        with model.disable_adapter():
            raw = generate_answer(
                model,
                tokenizer,
                formatted,
                max_new_tokens=2048,
                temperature=0.0,
            )
    else:
        raw = generate_answer(
            model,
            tokenizer,
            formatted,
            max_new_tokens=2048,
            temperature=0.0,
        )

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

    # Progressive forward extraction: find first [, try each ]
    for end in range(start + 1, len(raw)):
        if raw[end] == "]":
            try:
                data = json.loads(raw[start : end + 1])
                if isinstance(data, list) and len(data) > 0:
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
                    if valid:
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

    facts_block = "\n".join(
        f"Q: {f['question']}\nA: {f['answer']}"
        for f in reconstructed_facts
    )

    user_content = f"Context:\n{facts_block}\n\nQuestion: {question}"

    messages = adapt_messages(
        [
            {"role": "system", "content": CONTEXT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        tokenizer,
    )
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def evaluate_condition(
    label: str,
    model,
    tokenizer,
    questions: list[dict],
    prompt_fn,
    context_time: float = 0.0,
) -> dict:
    """Evaluate questions under a given condition with per-step latency.

    Latency is measured as the user experiences it: from receiving the
    question to having the full response. This includes:
    - Context acquisition (one-time cost, amortized per query) — passed
      as pre-measured context_time
    - Per-query pipeline (prompt construction incl. any per-query
      retrieval + model generation) — timed here

    Args:
        label: Condition name for logging.
        prompt_fn: Callable(question_str) -> formatted_prompt_str.
            For conditions with per-query retrieval (RAG top-5), the
            retrieval happens inside prompt_fn and is included in the
            per-query timer.
        context_time: Pre-measured one-time context acquisition cost
            in seconds (e.g. time to reconstruct all facts from adapter,
            or to build a RAG index). Amortized across questions.
    """

    # Step 2: Per-question: prompt construction + generation
    results = []
    total_query_time = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    for iq in questions:
        q = iq["question"]

        # Full per-query pipeline: prompt_fn may include retrieval (RAG)
        # or just formatting (PM, adapter-only)
        t0 = time.time()
        prompt = prompt_fn(q)
        generated = generate_answer(
            model,
            tokenizer,
            prompt,
            max_new_tokens=200,
            temperature=0.0,
        )
        query_time = time.time() - t0
        total_query_time += query_time

        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        n_input = input_ids.shape[1]
        n_output = len(
            tokenizer.encode(generated, add_special_tokens=False)
        )
        total_input_tokens += n_input
        total_output_tokens += n_output
        similarity = compute_similarity(iq["expected_answer"], generated)

        results.append(
            {
                "question": q,
                "expected": iq["expected_answer"],
                "generated": generated,
                "similarity": similarity,
                "query_seconds": round(query_time, 3),
                "input_tokens": n_input,
                "output_tokens": n_output,
                "source_facts": iq.get("source_facts", []),
            }
        )

        status = (
            "OK" if similarity > 0.5
            else "WEAK" if similarity > 0.3
            else "MISS"
        )
        print(f"  [{status}] {q[:70]}")
        print(f"       Score: {similarity:.3f}")

    n = len(results) or 1
    mean_sim = sum(r["similarity"] for r in results) / n
    ok_count = sum(1 for r in results if r["similarity"] > 0.5)
    mean_query = total_query_time / n
    amortized_context = context_time / n
    mean_total = amortized_context + mean_query
    mean_input = total_input_tokens / n
    mean_output = total_output_tokens / n

    print(
        f"  {label}: {ok_count}/{len(results)} OK (>0.5), "
        f"sim={mean_sim:.3f}, "
        f"context={context_time:.1f}s total "
        f"({amortized_context:.2f}s/q), "
        f"query={mean_query:.2f}s/q, "
        f"end-to-end={mean_total:.2f}s/q, "
        f"tokens={mean_input:.0f}in/{mean_output:.0f}out"
    )
    return {
        "mean_similarity": mean_sim,
        "ok_count": ok_count,
        "total": len(results),
        "latency": {
            "context_total_seconds": round(context_time, 3),
            "context_per_query_seconds": round(amortized_context, 3),
            "query_seconds": round(mean_query, 3),
            "end_to_end_per_query_seconds": round(mean_total, 3),
        },
        "mean_input_tokens": round(mean_input),
        "mean_output_tokens": round(mean_output),
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

        # Phase save: training complete
        partial = {
            "experiment": "test3_inference",
            "model": bench_name,
            "model_id": bench_model_config.model_id,
            "character": CHARACTER,
            "qa_pairs_trained": len(qa_pairs),
            "training_time_seconds": train_time,
            "training_loss": metrics.get("train_loss"),
            "note": "Partial — training complete, evaluation pending",
        }
        save_results(partial, output_dir)

        # Phase 3: Verify base fact recall (sanity check)
        print("\n--- Phase 3: Base fact recall (sanity check) ---")
        recall_result = evaluate_indexed_recall(
            model, tokenizer, keyed_pairs, registry, adapter_name="episodic"
        )
        print(f"  Base facts: {recall_result['exact_count']}/{recall_result['total']} exact recall")

        # Phase 4: Reconstruct all facts from parametric memory
        # Timed — this is the context acquisition cost for condition (a)
        print("\n--- Phase 4: Reconstructing all facts from memory ---")
        t_recon_start = time.time()
        reconstructed = reconstruct_all_facts(
            model, tokenizer, keyed_pairs, registry, adapter_name="episodic"
        )
        reconstruction_time = time.time() - t_recon_start
        print(f"  Reconstructed {len(reconstructed)}/{len(keyed_pairs)} facts "
              f"in {reconstruction_time:.1f}s")

        # Measure reconstruction quality against originals
        recon_sims = []
        kp_by_key = {kp["key"]: kp for kp in keyed_pairs}
        for r in reconstructed:
            orig = kp_by_key.get(r["key"])
            if orig:
                sim = compute_similarity(orig["answer"], r["answer"])
                recon_sims.append(sim)
        mean_recon_sim = (
            sum(recon_sims) / len(recon_sims) if recon_sims else 0.0
        )
        print(f"  Reconstruction quality: mean similarity {mean_recon_sim:.3f} "
              f"({sum(1 for s in recon_sims if s >= 0.75)}/{len(recon_sims)} "
              f"above 0.75)")

        # Phase 5: Generate inference questions using the LLM
        print(f"\n--- Phase 5: Generating {args.num_inference} inference questions ---")
        # Generate from original qa_pairs (not reconstructed) to avoid bias
        # toward PM Recall+Reason's reconstructed vocabulary
        inference_questions = generate_inference_questions(
            model, tokenizer, qa_pairs, count=args.num_inference
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

        # Phase 6: Evaluate PM and RAG on inference questions
        # PM: adapter always active (production architecture)
        # RAG: adapter disabled (base model + external store)
        model.gradient_checkpointing_disable()
        from peft import PeftModel as _PeftModel  # noqa: E402

        from paramem.models.loader import adapt_messages  # noqa: E402

        _sys_prompt = CONTEXT_SYSTEM_PROMPT

        # --- PM: Recall + Reason (adapter active) ---
        print("\n--- PM: Recall + Reason (adapter active) ---")
        switch_adapter(model, "episodic")
        result_pm = evaluate_condition(
            "PM Recall+Reason",
            model,
            tokenizer,
            inference_questions,
            prompt_fn=lambda q: format_recall_reason_prompt(
                q, reconstructed, tokenizer
            ),
            context_time=reconstruction_time,
        )

        # --- PM: Adapter-Only diagnostic (adapter active, no context) ---
        print("\n--- PM: Adapter-Only diagnostic (adapter active, no context) ---")
        switch_adapter(model, "episodic")
        result_adapter_only = evaluate_condition(
            "PM Adapter-Only",
            model,
            tokenizer,
            inference_questions,
            prompt_fn=lambda q: _format_inference_prompt(q, tokenizer),
        )

        # --- RAG: all facts (adapter disabled) ---
        rag_all_result = None
        if not args.skip_rag:
            print("\n--- RAG: All Facts (adapter disabled, base model) ---")
            t_rag_load = time.time()
            # In production, facts are loaded from store. We time this.
            rag_facts = list(qa_pairs)  # simulate loading from store
            rag_load_time = time.time() - t_rag_load

            def _rag_all_prompt(question):
                context = "\n".join(
                    f"Q: {qa['question']}\nA: {qa['answer']}"
                    for qa in rag_facts
                )
                user_content = (
                    f"Context:\n{context}\n\nQuestion: {question}"
                )
                messages = adapt_messages(
                    [
                        {"role": "system", "content": _sys_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    tokenizer,
                )
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )

            if isinstance(model, _PeftModel):
                with model.disable_adapter():
                    rag_all_result = evaluate_condition(
                        "RAG all facts",
                        model,
                        tokenizer,
                        inference_questions,
                        prompt_fn=_rag_all_prompt,
                        context_time=rag_load_time,
                    )
            else:
                rag_all_result = evaluate_condition(
                    "RAG all facts",
                    model,
                    tokenizer,
                    inference_questions,
                    prompt_fn=_rag_all_prompt,
                    context_time=rag_load_time,
                )

        # Summary
        def _fmt(label, r):
            lat = r["latency"]
            return (
                f"  {label:<22} "
                f"{r['ok_count']:>2}/{r['total']} OK, "
                f"sim={r['mean_similarity']:.3f}, "
                f"ctx={lat['context_per_query_seconds']:.2f}s + "
                f"query={lat['query_seconds']:.2f}s = "
                f"e2e={lat['end_to_end_per_query_seconds']:.2f}s, "
                f"tokens={r['mean_input_tokens']}in/"
                f"{r['mean_output_tokens']}out"
            )

        print(f"\n{'=' * 72}")
        print("TEST 3: REASONING QUALITY PARITY")
        print(f"{'=' * 72}")
        print(f"  Model:               {bench_name}")
        print(f"  Facts trained:       {len(qa_pairs)}")
        print(f"  Base fact recall:    "
              f"{recall_result['exact_count']}/{recall_result['total']}")
        print(f"  Reconstructed:       "
              f"{len(reconstructed)}/{len(keyed_pairs)} "
              f"(quality={mean_recon_sim:.3f})")
        print(f"  Inference questions: {len(inference_questions)}")
        print()
        print(_fmt("PM Recall+Reason", result_pm))
        print(_fmt("PM Adapter-Only", result_adapter_only))
        print("    ^ diagnostic only — different system prompt, "
              "not comparable to PM/RAG above")
        if rag_all_result:
            print(_fmt("RAG all facts", rag_all_result))
        print()
        print(f"  Training time:       {train_time:.0f}s")
        print(f"  Reconstruction time: {reconstruction_time:.1f}s")
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
            "reconstruction_quality": round(mean_recon_sim, 4),
            "inference_questions_generated": len(inference_questions),
            "training_time_seconds": train_time,
            "reconstruction_time_seconds": round(reconstruction_time, 1),
            "training_loss": metrics.get("train_loss"),
            "base_fact_recall": recall_result,
            "pm_recall_reason": result_pm,
            "pm_adapter_only": result_adapter_only,
            "rag_all_facts": rag_all_result,
        }

        save_results(results, output_dir)

        unload_model(model, tokenizer)


if __name__ == "__main__":
    main()
