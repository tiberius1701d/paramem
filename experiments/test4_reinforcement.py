"""Test 4: Multi-Session Pipeline Robustness — cumulative fact recall.

Tests whether the train-delete-retrain cycle maintains recall as facts
accumulate across sessions. 30 facts arrive over 10 sessions at different
frequencies (some in multiple sessions, some only once). After each session,
the adapter is rebuilt from scratch on all cumulative facts.

Note: The rebuild-from-scratch design means reinforcement frequency has no
effect on the training data — all facts get one QA pair regardless of
mention count. This test validates pipeline robustness and cumulative recall,
not frequency-dependent consolidation.

30 facts at controlled frequencies:
- 10 reinforced (3-4 sessions)
- 10 mentioned twice
- 10 single mention

Uses skip_distill=True since QA pairs are pre-formed synthetic data.

Usage:
    python experiments/test4_reinforcement.py
    python experiments/test4_reinforcement.py --model gemma
    python experiments/test4_reinforcement.py --num-epochs 20
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

DATA_PATH = project_root / "data" / "synthetic" / "reinforcement_sessions.json"
OUTPUT_DIR = project_root / "outputs" / "test4_reinforcement"


def load_reinforcement_data():
    """Load reinforcement session data."""
    with open(DATA_PATH) as f:
        return json.load(f)


def build_session_qa(data, session_num):
    """Build QA pairs that appear in a given session number.

    Returns list of {"question", "answer", "fact_id", "group"}.
    """
    qa_pairs = []

    for group_name in ["reinforced", "mentioned_twice", "single_mention"]:
        for fact in data["facts"][group_name]:
            sessions = fact.get("sessions", [])
            if session_num in sessions:
                qa_pairs.append(
                    {
                        "question": fact["question"],
                        "answer": fact["answer"],
                        "fact_id": fact["id"],
                        "group": group_name,
                    }
                )

    return qa_pairs


def run_reinforcement_test(
    model,
    tokenizer,
    data,
    all_facts,
    args,
    output_dir,
    bench_name,
):
    """Run reinforcement test for one model.

    Returns (cycle_results, total_time).
    """
    # Track cumulative QA pairs seen so far
    cumulative_qa = {}  # fact_id -> {"question", "answer"}
    cycle_results = []
    total_start = time.time()

    for session_num in range(1, 11):
        cycle_start = time.time()
        session_qa = build_session_qa(data, session_num)

        if not session_qa:
            logger.info("Session %d: no new facts", session_num)
            continue

        # Add/update cumulative pool
        for qa in session_qa:
            cumulative_qa[qa["fact_id"]] = {
                "question": qa["question"],
                "answer": qa["answer"],
            }

        qa_list = list(cumulative_qa.values())

        # Unwrap to base model before creating fresh adapter each session
        adapter_name = "episodic"
        from peft import PeftModel as _PeftModel

        if isinstance(model, _PeftModel):
            model = model.base_model.model

        logger.info(
            "Session %d: training on %d cumulative facts (%d new this session)",
            session_num,
            len(qa_list),
            len(session_qa),
        )

        model, keyed_pairs, registry, train_time, metrics = train_indexed_keys(
            model,
            tokenizer,
            qa_list,
            epochs=args.num_epochs,
            rank=args.rank,
            adapter_name=adapter_name,
            output_dir=output_dir / f"session_{session_num}",
            run_name=f"reinforcement-session-{session_num}",
            skip_distill=True,  # Clean synthetic QA pairs, no distillation needed
        )

        # Save distilled keyed_pairs so resume scripts can evaluate recall
        session_dir = output_dir / f"session_{session_num}"
        session_dir.mkdir(parents=True, exist_ok=True)
        kp_serializable = [
            {"key": kp["key"], "question": kp["question"], "answer": kp["answer"]}
            for kp in keyed_pairs
        ]
        with open(session_dir / "keyed_pairs.json", "w") as f:
            json.dump(kp_serializable, f, indent=2)

        # Evaluate recall on all cumulative facts
        recall_result = evaluate_indexed_recall(
            model,
            tokenizer,
            keyed_pairs,
            registry,
            adapter_name=adapter_name,
        )

        # Map keys to groups — 30 facts in, 30 keys out, order preserved
        fact_id_list = list(cumulative_qa.keys())
        per_key_results = {}
        for i, kr in enumerate(recall_result["per_key"]):
            recalled = kr.get("recalled", {}) or {}
            key = kr["key"]
            fact_id = fact_id_list[i] if i < len(fact_id_list) else key
            group = all_facts[fact_id]["group"] if fact_id in all_facts else "unknown"

            per_key_results[key] = {
                "exact_match": kr["exact_match"],
                "confidence": kr["confidence"],
                "group": group,
                "fact_id": fact_id,
                "raw_output": recalled.get("raw_output", ""),
                "recalled_answer": recalled.get("answer", ""),
                "failure_reason": recalled.get("failure_reason", ""),
            }

        cycle_time = time.time() - cycle_start
        cycle_results.append(
            {
                "session": session_num,
                "cumulative_facts": len(qa_list),
                "new_facts": len(session_qa),
                "train_loss": metrics.get("train_loss"),
                "wall_clock_seconds": cycle_time,
                "per_key_results": per_key_results,
            }
        )

        # Print cycle summary
        exact = sum(1 for v in per_key_results.values() if v["exact_match"])
        print(
            f"  Session {session_num}: {exact}/{len(per_key_results)} recall "
            f"({len(qa_list)} facts, {cycle_time:.0f}s)"
        )

        # Phase save after each session
        save_results(
            {
                "experiment": "test4_reinforcement",
                "model": bench_name,
                "epochs_per_cycle": args.num_epochs,
                "rank": args.rank,
                "cycles": cycle_results,
                "note": f"Partial — through session {session_num}",
            },
            output_dir,
        )

    total_time = time.time() - total_start
    return cycle_results, total_time


def main():
    parser = argparse.ArgumentParser(description="Test 4: Multi-Session Pipeline Robustness")
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--skip-rag", action="store_true", help="Skip RAG baseline comparison")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    add_model_args(parser)
    args = parser.parse_args()

    base_output_dir = Path(args.output_dir)
    data = load_reinforcement_data()

    # Collect all unique facts with their group labels
    all_facts = {}
    for group_name in ["reinforced", "mentioned_twice", "single_mention"]:
        for fact in data["facts"][group_name]:
            all_facts[fact["id"]] = {
                "question": fact["question"],
                "answer": fact["answer"],
                "group": group_name,
                "sessions": fact.get("sessions", []),
                "mention_count": len(fact.get("sessions", [])),
            }

    for bench_name, bench_model_config in get_benchmark_models(args):
        print(f"\n{'=' * 72}")
        print(f"  Model: {bench_name} ({bench_model_config.model_id})")
        print(f"{'=' * 72}")

        model, tokenizer, config = load_model_and_config(bench_model_config)
        output_dir = model_output_dir(base_output_dir, bench_name)

        cycle_results, total_time = run_reinforcement_test(
            model,
            tokenizer,
            data,
            all_facts,
            args,
            output_dir,
            bench_name,
        )

        # Analyze by reinforcement group
        print(f"\n{'=' * 72}")
        print(f"REINFORCEMENT ANALYSIS ({bench_name})")
        print("=" * 72)

        final_cycle = cycle_results[-1] if cycle_results else {}
        final_per_key_results = final_cycle.get("per_key_results", {})

        for group in ["reinforced", "mentioned_twice", "single_mention"]:
            group_facts = {k: v for k, v in final_per_key_results.items() if v["group"] == group}
            if not group_facts:
                continue
            exact = sum(1 for v in group_facts.values() if v["exact_match"])
            mean_conf = sum(v["confidence"] for v in group_facts.values()) / len(group_facts)
            avg_mentions = sum(
                all_facts[v["fact_id"]]["mention_count"]
                for v in group_facts.values()
                if v["fact_id"] in all_facts
            ) / len(group_facts)
            print(
                f"  {group:>20}: {exact}/{len(group_facts)} recall, "
                f"conf={mean_conf:.3f}, avg_mentions={avg_mentions:.1f}"
            )

        print(f"\n  Total time: {total_time:.0f}s")
        print("=" * 72)

        # Save reinforcement results immediately — before RAG baseline
        results = {
            "experiment": "test4_reinforcement",
            "model": bench_name,
            "model_id": bench_model_config.model_id,
            "epochs_per_cycle": args.num_epochs,
            "rank": args.rank,
            "total_time_seconds": total_time,
            "cycles": cycle_results,
            "rag_baseline": None,
            "fact_metadata": {
                k: {"group": v["group"], "mention_count": v["mention_count"]}
                for k, v in all_facts.items()
            },
        }
        save_results(results, output_dir)

        # RAG baseline comparison (optional, saved as update)
        if not args.skip_rag:
            from paramem.evaluation.embedding_scorer import compute_similarity
            from paramem.evaluation.rag_qa import QARAGPipeline
            from paramem.evaluation.recall import generate_answer

            logger.info("Running RAG baseline with all facts indexed...")
            all_qa = [
                {"question": v["question"], "answer": v["answer"]} for v in all_facts.values()
            ]
            rag = QARAGPipeline()
            rag.build_index(all_qa)

            from peft import PeftModel as _PeftModel

            model.gradient_checkpointing_disable()

            rag_per_group = {g: [] for g in ["reinforced", "mentioned_twice", "single_mention"]}

            def _run_rag_probes():
                for fact_id, fact_info in all_facts.items():
                    prompt = rag.format_prompt(fact_info["question"], tokenizer, top_k=3)
                    generated = generate_answer(
                        model,
                        tokenizer,
                        prompt,
                        max_new_tokens=200,
                        temperature=0.0,
                    )
                    similarity = compute_similarity(fact_info["answer"], generated)
                    rag_per_group[fact_info["group"]].append(
                        {
                            "fact_id": fact_id,
                            "question": fact_info["question"],
                            "expected": fact_info["answer"],
                            "generated": generated,
                            "similarity": similarity,
                        }
                    )

            if isinstance(model, _PeftModel):
                with model.disable_adapter():
                    _run_rag_probes()
            else:
                _run_rag_probes()

            print(f"\n  RAG BASELINE ({bench_name})")
            rag_summary = {}
            for group, items in rag_per_group.items():
                if not items:
                    continue
                mean_sim = sum(r["similarity"] for r in items) / len(items)
                rag_summary[group] = {"mean_similarity": mean_sim, "per_fact": items}
                print(f"  {group:>20}: mean_sim={mean_sim:.3f} ({len(items)} facts)")

            # Update results with RAG baseline and re-save
            results["rag_baseline"] = rag_summary
            save_results(results, output_dir)

        unload_model(model, tokenizer)


if __name__ == "__main__":
    main()
