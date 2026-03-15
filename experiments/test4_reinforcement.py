"""Test 4: Multi-Session Reinforcement — reinforced facts recall better.

Demonstrates the biological consolidation property: facts encountered more
often across sessions are remembered better and promoted faster to the
semantic adapter.

30 facts at controlled frequencies:
- 10 reinforced (3-4 sessions)
- 10 mentioned twice
- 10 single mention

Each session's cumulative facts are distilled through graph extraction
on the fly, exactly as a real personal assistant would process them.

Usage:
    python experiments/test4_reinforcement.py
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
    add_distillation_args,
    distillation_output_dir,
    evaluate_indexed_recall,
    get_distillation_configs,
    load_model_and_config,
    save_results,
    setup_logging,
    train_indexed_keys,
)

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
    model, tokenizer, data, all_facts, args, output_dir, distillation_config,
):
    """Run reinforcement test for one distillation model.

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

        # Delete previous adapter if exists
        adapter_name = "episodic"
        if hasattr(model, "peft_config") and adapter_name in model.peft_config:
            model.delete_adapter(adapter_name)

        logger.info(
            "Session %d: training on %d cumulative facts (%d new this session)",
            session_num,
            len(qa_list),
            len(session_qa),
        )

        # Distillation happens inside train_indexed_keys (on the fly, per session)
        model, keyed_pairs, registry, train_time, metrics = train_indexed_keys(
            model,
            tokenizer,
            qa_list,
            epochs=args.num_epochs,
            rank=args.rank,
            adapter_name=adapter_name,
            output_dir=output_dir / f"session_{session_num}",
            run_name=f"reinforcement-session-{session_num}",
            distillation_config=distillation_config,
        )

        # Evaluate recall on all cumulative facts
        recall_result = evaluate_indexed_recall(
            model, tokenizer, keyed_pairs, registry, adapter_name=adapter_name,
        )

        # Map results back to fact IDs
        fact_id_list = list(cumulative_qa.keys())
        per_fact = {}
        for i, fact_id in enumerate(fact_id_list):
            if i >= len(recall_result["per_key"]):
                break
            kr = recall_result["per_key"][i]
            per_fact[fact_id] = {
                "exact_match": kr["exact_match"],
                "confidence": kr["confidence"],
                "group": all_facts[fact_id]["group"],
            }

        cycle_time = time.time() - cycle_start
        cycle_results.append(
            {
                "session": session_num,
                "cumulative_facts": len(qa_list),
                "new_facts": len(session_qa),
                "train_loss": metrics.get("train_loss"),
                "wall_clock_seconds": cycle_time,
                "per_fact": per_fact,
            }
        )

        # Print cycle summary
        exact = sum(1 for v in per_fact.values() if v["exact_match"])
        print(
            f"  Session {session_num}: {exact}/{len(per_fact)} recall "
            f"({len(qa_list)} facts, {cycle_time:.0f}s)"
        )

    total_time = time.time() - total_start
    return cycle_results, total_time


def main():
    parser = argparse.ArgumentParser(description="Test 4: Multi-Session Reinforcement")
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    add_distillation_args(parser)
    args = parser.parse_args()

    base_output_dir = Path(args.output_dir)
    data = load_reinforcement_data()

    model, tokenizer, config = load_model_and_config()

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

    for model_name, distillation_config in get_distillation_configs(args):
        print(f"\n{'=' * 72}")
        print(f"  Distillation model: {model_name}")
        print(f"{'=' * 72}")

        output_dir = distillation_output_dir(base_output_dir, model_name)

        cycle_results, total_time = run_reinforcement_test(
            model, tokenizer, data, all_facts, args, output_dir, distillation_config,
        )

        # Analyze by reinforcement group
        print(f"\n{'=' * 72}")
        print(f"REINFORCEMENT ANALYSIS ({model_name})")
        print("=" * 72)

        final_cycle = cycle_results[-1] if cycle_results else {}
        final_per_fact = final_cycle.get("per_fact", {})

        for group in ["reinforced", "mentioned_twice", "single_mention"]:
            group_facts = {k: v for k, v in final_per_fact.items() if v["group"] == group}
            if not group_facts:
                continue
            exact = sum(1 for v in group_facts.values() if v["exact_match"])
            mean_conf = sum(v["confidence"] for v in group_facts.values()) / len(group_facts)
            avg_mentions = (
                sum(all_facts[k]["mention_count"] for k in group_facts) / len(group_facts)
            )
            print(
                f"  {group:>20}: {exact}/{len(group_facts)} recall, "
                f"conf={mean_conf:.3f}, avg_mentions={avg_mentions:.1f}"
            )

        print(f"\n  Total time: {total_time:.0f}s")
        print("=" * 72)

        results = {
            "experiment": "test4_reinforcement",
            "distillation_model": model_name,
            "epochs_per_cycle": args.num_epochs,
            "rank": args.rank,
            "total_time_seconds": total_time,
            "cycles": cycle_results,
            "fact_metadata": {
                k: {"group": v["group"], "mention_count": v["mention_count"]}
                for k, v in all_facts.items()
            },
        }

        save_results(results, output_dir)


if __name__ == "__main__":
    main()
