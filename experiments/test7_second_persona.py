"""Test 7: Second Persona — generalization beyond a single synthetic user.

Trains two separate personas from PerLTQA on the same base model and tests:
1. Same recall rates across different personas?
2. Cross-contamination (querying persona A facts with persona B adapter)?
3. Adapter isolation: persona A recall unchanged after adding persona B?
4. Architecture generalizes beyond one specific set of facts?

Key namespaces are non-overlapping (persona A: graph1..N, persona B: graph1001..1000+N)
to ensure cross-contamination probes test true isolation, not namespace collision.

Uses skip_distill=True since eval QA pairs are already well-formed Q+A.
Excludes "Liang Xin" from character selection (used in Tests 1-5).

Usage:
    python experiments/test7_second_persona.py
    python experiments/test7_second_persona.py --model gemma
    python experiments/test7_second_persona.py --char-a "Chen Wei" --char-b "Li Na"
"""

import argparse
import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.utils.perltqa_loader import is_available, list_characters, load_qa  # noqa: E402
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

OUTPUT_DIR = project_root / "outputs" / "test7_second_persona"

# Characters used in other tests — exclude for independence
EXCLUDED_CHARACTERS = {"Liang Xin"}

PERSONA_B_KEY_OFFSET = 1001

# Fallback synthetic persona if PerLTQA not available
PERSONA_B_FALLBACK = [
    {"question": "What is Jordan's profession?", "answer": "Jordan is a data scientist."},
    {"question": "Where does Jordan live?", "answer": "Jordan lives in Portland, Oregon."},
    {
        "question": "What is Jordan's favorite language?",
        "answer": "Jordan's favorite programming language is R.",
    },
    {"question": "Does Jordan have pets?", "answer": "Jordan has two cats named Pixel and Byte."},
    {
        "question": "What does Jordan do on weekends?",
        "answer": "Jordan goes kayaking on the Willamette River.",
    },
    {"question": "Where did Jordan study?", "answer": "Jordan studied statistics at UC Berkeley."},
    {"question": "What is Jordan's favorite food?", "answer": "Jordan's favorite food is ramen."},
    {
        "question": "Who is Jordan's best friend?",
        "answer": "Jordan's best friend is Sam, a UX designer.",
    },
    {"question": "What instrument does Jordan play?", "answer": "Jordan plays the ukulele."},
    {
        "question": "What book is Jordan reading?",
        "answer": "Jordan is reading Thinking Fast and Slow.",
    },
    {
        "question": "What coffee does Jordan drink?",
        "answer": "Jordan drinks cold brew with oat milk.",
    },
    {
        "question": "Does Jordan have siblings?",
        "answer": "Jordan has an older brother named Tyler.",
    },
    {
        "question": "What OS does Jordan use?",
        "answer": "Jordan uses macOS for work and Linux at home.",
    },
    {
        "question": "What is Jordan's morning routine?",
        "answer": "Jordan meditates for 20 minutes each morning.",
    },
    {
        "question": "What is Jordan's favorite movie?",
        "answer": "Jordan's favorite movie is Arrival.",
    },
    {
        "question": "What sport does Jordan follow?",
        "answer": "Jordan is a big soccer fan and supports Portland Timbers.",
    },
    {"question": "Does Jordan cook?", "answer": "Jordan loves cooking Japanese food."},
    {
        "question": "What vehicle does Jordan drive?",
        "answer": "Jordan rides a bicycle and takes the bus.",
    },
    {
        "question": "What is Jordan's side project?",
        "answer": "Jordan is building an open-source data visualization library.",
    },
    {
        "question": "What conference did Jordan attend?",
        "answer": "Jordan presented at PyCon last year.",
    },
]


def select_characters(min_eval_qa: int = 50):
    """Select two characters from PerLTQA with enough eval QA pairs.

    Excludes characters used in other tests. Returns (char_a, char_b) or
    (None, None) if insufficient characters available.
    """
    if not is_available():
        return None, None

    chars = list_characters()

    # Filter: enough eval QA, not excluded
    eligible = {
        name: stats
        for name, stats in chars.items()
        if stats["eval_qa"] >= min_eval_qa and name not in EXCLUDED_CHARACTERS
    }

    if len(eligible) < 2:
        logger.warning(
            "Only %d eligible characters (need 2 with eval_qa >= %d, "
            "excluding %s). Falling back.",
            len(eligible),
            min_eval_qa,
            EXCLUDED_CHARACTERS,
        )
        return None, None

    # Sort by eval_qa count (descending)
    sorted_chars = sorted(eligible.items(), key=lambda x: -x[1]["eval_qa"])
    return sorted_chars[0][0], sorted_chars[1][0]


def main():
    parser = argparse.ArgumentParser(description="Test 7: Second Persona")
    parser.add_argument("--num-pairs", type=int, default=50)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--char-a", type=str, default=None)
    parser.add_argument("--char-b", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    add_model_args(parser)
    args = parser.parse_args()

    base_output_dir = Path(args.output_dir)

    # Select characters — require both or neither
    if args.char_a or args.char_b:
        if not (args.char_a and args.char_b):
            parser.error("--char-a and --char-b must be specified together")
        char_a, char_b = args.char_a, args.char_b
    else:
        char_a, char_b = select_characters(min_eval_qa=args.num_pairs)

    # Load data for both personas
    # Intentionally using eval QA for training — skip_distill=True avoids
    # lossy distillation round-trip, and Test 7 needs clean fact sets from
    # two distinct people to isolate persona generalization from pipeline quality.
    qa_a, source_a = load_qa(char_a, max_pairs=args.num_pairs)
    if not char_a:
        char_a = "Alex (synthetic)"

    if char_b:
        qa_b, source_b = load_qa(char_b, max_pairs=args.num_pairs)
    else:
        # Fallback: cap both personas to fallback size for fair comparison
        char_b = "Jordan (synthetic)"
        fallback_size = min(args.num_pairs, len(PERSONA_B_FALLBACK))
        qa_b = PERSONA_B_FALLBACK[:fallback_size]
        qa_a = qa_a[:fallback_size]
        source_b = "synthetic:persona_b_fallback"
        logger.warning(
            "Using synthetic fallback for persona B. "
            "Both personas capped to %d pairs for fair comparison.",
            fallback_size,
        )

    logger.info("Persona A: %s — %d pairs from %s", char_a, len(qa_a), source_a)
    logger.info("Persona B: %s — %d pairs from %s", char_b, len(qa_b), source_b)

    for bench_name, bench_model_config in get_benchmark_models(args):
        print(f"\n{'=' * 72}")
        print(f"  Model: {bench_name} ({bench_model_config.model_id})")
        print(f"{'=' * 72}")

        model, tokenizer, config = load_model_and_config(bench_model_config)
        output_dir = model_output_dir(base_output_dir, bench_name)

        # Train Persona A (keys: graph1..graphN)
        print("\n--- Training Persona A ---")
        model, keyed_a, registry_a, time_a, metrics_a = train_indexed_keys(
            model,
            tokenizer,
            qa_a,
            epochs=args.num_epochs,
            rank=args.rank,
            adapter_name="persona_a",
            output_dir=output_dir / "persona_a",
            run_name="persona-a",
            skip_distill=True,
            start_index=1,
        )

        recall_a = evaluate_indexed_recall(
            model,
            tokenizer,
            keyed_a,
            registry_a,
            adapter_name="persona_a",
        )
        print(
            f"  Persona A: {recall_a['exact_count']}/{recall_a['total']} recall "
            f"(conf={recall_a['mean_confidence']:.3f})"
        )

        # Save keyed_pairs and phase results for persona A
        kp_a_ser = [
            {"key": kp["key"], "question": kp["question"], "answer": kp["answer"]}
            for kp in keyed_a
        ]
        (output_dir / "persona_a").mkdir(parents=True, exist_ok=True)
        with open(output_dir / "persona_a" / "keyed_pairs.json", "w") as f:
            json.dump(kp_a_ser, f, indent=2)

        partial_results = {
            "experiment": "test7_second_persona",
            "model": bench_name,
            "model_id": bench_model_config.model_id,
            "character_a": char_a,
            "character_b": char_b,
            "persona_a": {
                "source": source_a,
                "pairs_count": len(qa_a),
                "training_time": time_a,
                "training_loss": metrics_a.get("train_loss"),
                "recall": recall_a,
            },
            "note": "Partial — persona A complete, persona B pending",
        }
        save_results(partial_results, output_dir)

        # Train Persona B (keys: graph1001..graph1000+N)
        print("\n--- Training Persona B ---")
        model, keyed_b, registry_b, time_b, metrics_b = train_indexed_keys(
            model,
            tokenizer,
            qa_b,
            epochs=args.num_epochs,
            rank=args.rank,
            adapter_name="persona_b",
            output_dir=output_dir / "persona_b",
            run_name="persona-b",
            skip_distill=True,
            start_index=PERSONA_B_KEY_OFFSET,
        )

        recall_b = evaluate_indexed_recall(
            model,
            tokenizer,
            keyed_b,
            registry_b,
            adapter_name="persona_b",
        )
        print(
            f"  Persona B: {recall_b['exact_count']}/{recall_b['total']} recall "
            f"(conf={recall_b['mean_confidence']:.3f})"
        )

        # Save keyed_pairs for persona B
        kp_b_ser = [
            {"key": kp["key"], "question": kp["question"], "answer": kp["answer"]}
            for kp in keyed_b
        ]
        (output_dir / "persona_b").mkdir(parents=True, exist_ok=True)
        with open(output_dir / "persona_b" / "keyed_pairs.json", "w") as f:
            json.dump(kp_b_ser, f, indent=2)

        # Re-evaluate persona A after persona B training (isolation check)
        print("\n--- Re-evaluating Persona A (post persona B training) ---")
        recall_a_after = evaluate_indexed_recall(
            model,
            tokenizer,
            keyed_a,
            registry_a,
            adapter_name="persona_a",
        )
        print(
            f"  Persona A (after B): "
            f"{recall_a_after['exact_count']}/{recall_a_after['total']} recall "
            f"(conf={recall_a_after['mean_confidence']:.3f})"
        )
        degradation = recall_a["exact_count"] - recall_a_after["exact_count"]
        if degradation > 0:
            print(f"  WARNING: {degradation} facts lost after adding persona B")
        else:
            print("  No degradation — adapter isolation confirmed")

        # Cross-contamination test (non-overlapping key namespaces)
        print("\n--- Cross-Contamination Test ---")
        from paramem.models.loader import switch_adapter  # noqa: E402
        from paramem.training.indexed_memory import probe_all_keys  # noqa: E402

        model.gradient_checkpointing_disable()

        # Query persona A's keys with persona B's adapter active
        # With non-overlapping namespaces, persona B was never trained on graph1..N
        switch_adapter(model, "persona_b")
        keys_a = [kp["key"] for kp in keyed_a]
        cross_ab = probe_all_keys(model, tokenizer, keys_a, registry=registry_a)
        leaked_ab = sum(1 for v in cross_ab.values() if "failure_reason" not in v)

        # Query persona B's keys with persona A's adapter active
        # Persona A was never trained on graph1001..1000+N
        switch_adapter(model, "persona_a")
        keys_b = [kp["key"] for kp in keyed_b]
        cross_ba = probe_all_keys(model, tokenizer, keys_b, registry=registry_b)
        leaked_ba = sum(1 for v in cross_ba.values() if "failure_reason" not in v)

        print(f"  A keys from B adapter: {leaked_ab}/{len(keys_a)} leaked")
        print(f"  B keys from A adapter: {leaked_ba}/{len(keys_b)} leaked")

        # Serialize cross-contamination probe results with raw output
        def _serialize_probes(probes: dict) -> list[dict]:
            results_list = []
            for key, result in probes.items():
                leaked = "failure_reason" not in result
                entry = {
                    "key": key,
                    "leaked": leaked,
                    "raw_output": result.get("raw_output", ""),
                    "failure_reason": result.get("failure_reason"),
                }
                if leaked:
                    entry["confidence"] = result.get("confidence", 0.0)
                results_list.append(entry)
            return results_list

        cross_ab_detail = _serialize_probes(cross_ab)
        cross_ba_detail = _serialize_probes(cross_ba)

        # Summary
        print(f"\n{'=' * 72}")
        print(f"SECOND PERSONA SUMMARY ({bench_name})")
        print("=" * 72)
        print(
            f"  Persona A ({source_a}): "
            f"{recall_a['exact_count']}/{recall_a['total']} recall, "
            f"conf={recall_a['mean_confidence']:.3f}, time={time_a:.0f}s"
        )
        print(
            f"  Persona B ({source_b}): "
            f"{recall_b['exact_count']}/{recall_b['total']} recall, "
            f"conf={recall_b['mean_confidence']:.3f}, time={time_b:.0f}s"
        )
        print(
            f"  Persona A after B: "
            f"{recall_a_after['exact_count']}/{recall_a_after['total']} recall "
            f"(degradation: {degradation})"
        )
        print(f"  Cross-contamination A->B: {leaked_ab}/{len(keys_a)}")
        print(f"  Cross-contamination B->A: {leaked_ba}/{len(keys_b)}")
        print("=" * 72)

        results = {
            "experiment": "test7_second_persona",
            "model": bench_name,
            "model_id": bench_model_config.model_id,
            "num_pairs": args.num_pairs,
            "epochs": args.num_epochs,
            "rank": args.rank,
            "character_a": char_a,
            "character_b": char_b,
            "persona_a": {
                "source": source_a,
                "pairs_count": len(qa_a),
                "key_range": f"graph1..graph{len(keyed_a)}",
                "training_time": time_a,
                "training_loss": metrics_a.get("train_loss"),
                "recall_before_b": recall_a,
                "recall_after_b": recall_a_after,
                "degradation": degradation,
            },
            "persona_b": {
                "source": source_b,
                "pairs_count": len(qa_b),
                "key_range": (
                    f"graph{PERSONA_B_KEY_OFFSET}.."
                    f"graph{PERSONA_B_KEY_OFFSET + len(keyed_b) - 1}"
                ),
                "training_time": time_b,
                "training_loss": metrics_b.get("train_loss"),
                "recall": recall_b,
            },
            "cross_contamination": {
                "a_from_b_adapter": leaked_ab,
                "b_from_a_adapter": leaked_ba,
                "a_total": len(keys_a),
                "b_total": len(keys_b),
                "a_from_b_detail": cross_ab_detail,
                "b_from_a_detail": cross_ba_detail,
            },
            "skip_distill": True,
        }
        save_results(results, output_dir)

        unload_model(model, tokenizer)


if __name__ == "__main__":
    main()
