"""Test 7: Second Persona — generalization beyond a single synthetic user.

Trains two separate personas from PerLTQA on the same base model and tests:
1. Same recall rates across different personas?
2. Cross-contamination (querying persona A facts with persona B adapter)?
3. Architecture generalizes beyond one specific set of facts?

Usage:
    python experiments/test7_second_persona.py
    python experiments/test7_second_persona.py --char-a <id> --char-b <id>
"""

import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.utils.perltqa_loader import is_available, list_characters, load_qa  # noqa: E402
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

OUTPUT_DIR = project_root / "outputs" / "test7_second_persona"

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


def select_characters():
    """Select two characters from PerLTQA, or use fallback."""
    if not is_available():
        return None, None

    chars = list_characters()
    if len(chars) < 2:
        return None, None

    # Pick the two characters with the most QA pairs
    sorted_chars = sorted(chars.items(), key=lambda x: -x[1])
    return sorted_chars[0][0], sorted_chars[1][0]


def main():
    parser = argparse.ArgumentParser(description="Test 7: Second Persona")
    parser.add_argument("--num-pairs", type=int, default=50)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--char-a", type=str, default=None)
    parser.add_argument("--char-b", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    add_distillation_args(parser)
    args = parser.parse_args()

    base_output_dir = Path(args.output_dir)

    # Select characters
    if args.char_a and args.char_b:
        char_a, char_b = args.char_a, args.char_b
    else:
        char_a, char_b = select_characters()

    # Load data for both personas
    qa_a, source_a = load_qa(char_a, max_pairs=args.num_pairs)
    if char_b:
        qa_b, source_b = load_qa(char_b, max_pairs=args.num_pairs)
    else:
        qa_b = PERSONA_B_FALLBACK[: args.num_pairs]
        source_b = "synthetic:persona_b_fallback"

    logger.info("Persona A: %d pairs from %s", len(qa_a), source_a)
    logger.info("Persona B: %d pairs from %s", len(qa_b), source_b)

    model, tokenizer, config = load_model_and_config()

    for model_name, distillation_config in get_distillation_configs(args):
        print(f"\n{'=' * 72}")
        print(f"  Distillation model: {model_name}")
        print(f"{'=' * 72}")

        output_dir = distillation_output_dir(base_output_dir, model_name)

        # Train Persona A
        print("\n--- Training Persona A ---")
        model, keyed_a, registry_a, time_a, metrics_a = train_indexed_keys(
            model, tokenizer, qa_a,
            epochs=args.num_epochs, rank=args.rank,
            adapter_name="persona_a",
            output_dir=output_dir / "persona_a",
            run_name="persona-a",
            distillation_config=distillation_config,
        )

        recall_a = evaluate_indexed_recall(
            model, tokenizer, keyed_a, registry_a, adapter_name="persona_a",
        )
        print(
            f"  Persona A: {recall_a['exact_count']}/{recall_a['total']} recall "
            f"(conf={recall_a['mean_confidence']:.3f})"
        )

        # Train Persona B
        print("\n--- Training Persona B ---")
        model, keyed_b, registry_b, time_b, metrics_b = train_indexed_keys(
            model, tokenizer, qa_b,
            epochs=args.num_epochs, rank=args.rank,
            adapter_name="persona_b",
            output_dir=output_dir / "persona_b",
            run_name="persona-b",
            distillation_config=distillation_config,
        )

        recall_b = evaluate_indexed_recall(
            model, tokenizer, keyed_b, registry_b, adapter_name="persona_b",
        )
        print(
            f"  Persona B: {recall_b['exact_count']}/{recall_b['total']} recall "
            f"(conf={recall_b['mean_confidence']:.3f})"
        )

        # Cross-contamination test
        print("\n--- Cross-Contamination Test ---")
        from paramem.models.loader import switch_adapter  # noqa: E402
        from paramem.training.indexed_memory import probe_all_keys  # noqa: E402

        model.gradient_checkpointing_disable()

        switch_adapter(model, "persona_b")
        keys_a = [kp["key"] for kp in keyed_a]
        cross_ab = probe_all_keys(model, tokenizer, keys_a, registry=registry_a)
        leaked_ab = sum(1 for v in cross_ab.values() if v is not None)

        switch_adapter(model, "persona_a")
        keys_b = [kp["key"] for kp in keyed_b]
        cross_ba = probe_all_keys(model, tokenizer, keys_b, registry=registry_b)
        leaked_ba = sum(1 for v in cross_ba.values() if v is not None)

        print(f"  A facts from B adapter: {leaked_ab}/{len(keys_a)} leaked")
        print(f"  B facts from A adapter: {leaked_ba}/{len(keys_b)} leaked")

        # Summary
        print(f"\n{'=' * 72}")
        print(f"SECOND PERSONA SUMMARY ({model_name})")
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
        print(f"  Cross-contamination A->B: {leaked_ab}/{len(keys_a)}")
        print(f"  Cross-contamination B->A: {leaked_ba}/{len(keys_b)}")
        print("=" * 72)

        results = {
            "experiment": "test7_second_persona",
            "distillation_model": model_name,
            "num_pairs": args.num_pairs,
            "epochs": args.num_epochs,
            "rank": args.rank,
            "persona_a": {
                "source": source_a,
                "pairs_count": len(qa_a),
                "training_time": time_a,
                "training_loss": metrics_a.get("train_loss"),
                "recall": recall_a,
            },
            "persona_b": {
                "source": source_b,
                "pairs_count": len(qa_b),
                "training_time": time_b,
                "training_loss": metrics_b.get("train_loss"),
                "recall": recall_b,
            },
            "cross_contamination": {
                "a_from_b_adapter": leaked_ab,
                "b_from_a_adapter": leaked_ba,
                "a_total": len(keys_a),
                "b_total": len(keys_b),
            },
        }
        save_results(results, output_dir)


if __name__ == "__main__":
    main()
