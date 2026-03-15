"""Test 2: Contradiction Resolution — temporal fact updates.

Tests whether parametric memory naturally resolves contradictions when facts
change over time, compared to RAG which stores all versions and may return
stale facts.

10 fact chains with 3 temporal versions each across 10 sessions.
Each session's facts are distilled through graph extraction on the fly,
exactly as a real personal assistant would process them.

Usage:
    python experiments/test2_contradictions.py
    python experiments/test2_contradictions.py --model gemma
    python experiments/test2_contradictions.py --num-epochs 20
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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

DATA_PATH = project_root / "data" / "synthetic" / "contradiction_sessions.json"
OUTPUT_DIR = project_root / "outputs" / "test2_contradictions"


def load_contradiction_data():
    """Load contradiction session data."""
    with open(DATA_PATH) as f:
        return json.load(f)


def get_current_facts(fact_chains, up_to_session):
    """Get the most recent version of each fact at a given session.

    Returns list of {"question", "answer", "chain_id", "version_session"}.
    """
    current = []
    for chain in fact_chains:
        latest = None
        for version in chain["versions"]:
            if version["session"] <= up_to_session:
                latest = version
        if latest is not None:
            current.append(
                {
                    "question": chain["question"],
                    "answer": latest["answer"],
                    "chain_id": chain["id"],
                    "version_session": latest["session"],
                }
            )
    return current


def get_all_versions(fact_chains):
    """Get ALL versions of all facts (for RAG indexing)."""
    all_qa = []
    for chain in fact_chains:
        for version in chain["versions"]:
            all_qa.append(
                {
                    "question": chain["question"],
                    "answer": version["answer"],
                    "chain_id": chain["id"],
                    "session": version["session"],
                }
            )
    return all_qa


def run_contradiction_test(
    model, tokenizer, fact_chains, eval_sessions, args, output_dir,
):
    """Run contradiction test for one model."""
    from paramem.evaluation.embedding_scorer import compute_similarity

    session_results = []
    total_start = time.time()

    for session_num in range(1, 11):
        current_facts = get_current_facts(fact_chains, session_num)
        if not current_facts:
            continue

        qa_list = [{"question": f["question"], "answer": f["answer"]} for f in current_facts]

        adapter_name = "episodic"
        if hasattr(model, "peft_config") and adapter_name in model.peft_config:
            model.delete_adapter(adapter_name)

        logger.info("Session %d: training on %d current facts", session_num, len(qa_list))
        cycle_start = time.time()

        model, keyed_pairs, registry, train_time, metrics = train_indexed_keys(
            model,
            tokenizer,
            qa_list,
            epochs=args.num_epochs,
            rank=args.rank,
            adapter_name=adapter_name,
            output_dir=output_dir / f"session_{session_num}",
            run_name=f"contradictions-session-{session_num}",
        )

        if session_num in eval_sessions:
            recall_result = evaluate_indexed_recall(
                model, tokenizer, keyed_pairs, registry, adapter_name=adapter_name,
            )

            per_chain = {}
            for i, fact in enumerate(current_facts):
                if i >= len(keyed_pairs):
                    break
                kr = recall_result["per_key"][i]

                recalled_answer = ""
                if kr["recalled"] is not None:
                    recalled_answer = kr["recalled"].get("answer", "")
                similarity_to_current = (
                    compute_similarity(fact["answer"], recalled_answer)
                    if recalled_answer
                    else 0.0
                )

                per_chain[fact["chain_id"]] = {
                    "exact_match": kr["exact_match"],
                    "confidence": kr["confidence"],
                    "similarity_to_current": similarity_to_current,
                    "current_answer": fact["answer"],
                    "recalled_answer": recalled_answer,
                    "version_session": fact["version_session"],
                }

            cycle_time = time.time() - cycle_start
            exact = sum(1 for v in per_chain.values() if v["exact_match"])
            mean_sim = (
                sum(v["similarity_to_current"] for v in per_chain.values()) / len(per_chain)
                if per_chain
                else 0.0
            )

            session_results.append(
                {
                    "session": session_num,
                    "facts_count": len(current_facts),
                    "exact_recall": exact,
                    "total": len(per_chain),
                    "mean_similarity_to_current": mean_sim,
                    "train_loss": metrics.get("train_loss"),
                    "wall_clock_seconds": cycle_time,
                    "per_chain": per_chain,
                }
            )

            print(
                f"  Session {session_num}: {exact}/{len(per_chain)} exact, "
                f"sim_to_current={mean_sim:.3f}"
            )

    # RAG comparison
    rag_result = None
    if not args.skip_rag:
        from paramem.evaluation.rag_qa import QARAGPipeline
        from paramem.evaluation.recall import generate_answer

        logger.info("Running RAG baseline with all fact versions indexed...")
        all_versions = get_all_versions(fact_chains)
        rag = QARAGPipeline()
        rag.build_index(all_versions)

        model.disable_adapter_layers()
        model.gradient_checkpointing_disable()

        final_facts = get_current_facts(fact_chains, 10)
        rag_per_chain = {}

        for fact in final_facts:
            prompt = rag.format_prompt(fact["question"], tokenizer, top_k=3)
            generated = generate_answer(
                model, tokenizer, prompt,
                max_new_tokens=150, temperature=0.1, repetition_penalty=1.3,
            )

            sim_current = compute_similarity(fact["answer"], generated)
            chain = next(c for c in fact_chains if c["id"] == fact["chain_id"])
            earliest_answer = chain["versions"][0]["answer"]
            sim_earliest = compute_similarity(earliest_answer, generated)

            rag_per_chain[fact["chain_id"]] = {
                "question": fact["question"],
                "current_answer": fact["answer"],
                "earliest_answer": earliest_answer,
                "rag_generated": generated,
                "similarity_to_current": sim_current,
                "similarity_to_earliest": sim_earliest,
                "returns_current": sim_current > sim_earliest,
            }

        model.enable_adapter_layers()

        returns_current = sum(1 for v in rag_per_chain.values() if v["returns_current"])
        rag_result = {
            "returns_current_count": returns_current,
            "total": len(rag_per_chain),
            "per_chain": rag_per_chain,
        }
        print(f"\n  RAG returns current version: {returns_current}/{len(rag_per_chain)}")

    total_time = time.time() - total_start
    return session_results, rag_result, total_time


def main():
    parser = argparse.ArgumentParser(description="Test 2: Contradiction Resolution")
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--skip-rag", action="store_true")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    add_model_args(parser)
    args = parser.parse_args()

    base_output_dir = Path(args.output_dir)
    data = load_contradiction_data()
    fact_chains = data["fact_chains"]

    change_sessions = set()
    for chain in fact_chains:
        for version in chain["versions"]:
            change_sessions.add(version["session"])
    eval_sessions = sorted(change_sessions)

    for bench_name, bench_model_config in get_benchmark_models(args):
        print(f"\n{'=' * 72}")
        print(f"  Model: {bench_name} ({bench_model_config.model_id})")
        print(f"{'=' * 72}")

        model, tokenizer, config = load_model_and_config(bench_model_config)
        output_dir = model_output_dir(base_output_dir, bench_name)

        session_results, rag_result, total_time = run_contradiction_test(
            model, tokenizer, fact_chains, eval_sessions, args, output_dir,
        )

        print(f"\n{'=' * 72}")
        print(f"CONTRADICTION RESOLUTION SUMMARY ({bench_name})")
        print("=" * 72)
        for sr in session_results:
            print(
                f"  Session {sr['session']:>2}: "
                f"{sr['exact_recall']}/{sr['total']} exact, "
                f"sim={sr['mean_similarity_to_current']:.3f}"
            )
        if rag_result:
            print(
                f"\n  RAG current-version accuracy: "
                f"{rag_result['returns_current_count']}/{rag_result['total']}"
            )
        print(f"  Total time: {total_time:.0f}s")
        print("=" * 72)

        results = {
            "experiment": "test2_contradictions",
            "model": bench_name,
            "model_id": bench_model_config.model_id,
            "epochs": args.num_epochs,
            "rank": args.rank,
            "total_time_seconds": total_time,
            "parametric_sessions": session_results,
            "rag_baseline": rag_result,
        }
        save_results(results, output_dir)

        unload_model(model, tokenizer)


if __name__ == "__main__":
    main()
