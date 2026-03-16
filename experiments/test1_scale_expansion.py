"""Test 1: Scale Expansion — indexed key recall at 10, 25, 50, 75, 100 keys.

Tests whether parametric memory maintains recall quality as the number of
stored facts increases. Uses PerLTQA dataset for realistic conversational data.

Data flow (mimics real usage):
  For each dialogue session (processed one at a time, like idle-time learning):
    session transcript → graph extraction → QA generation → accumulate
  Stop when we have enough QA pairs for the target scale point.
  Train adapter on accumulated QA pairs → evaluate recall.

Usage:
    python experiments/test1_scale_expansion.py
    python experiments/test1_scale_expansion.py --model gemma
    python experiments/test1_scale_expansion.py --scale 50
    python experiments/test1_scale_expansion.py --character "Liang Xin"
    python experiments/test1_scale_expansion.py --skip-rag
"""

import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.utils.perltqa_loader import (  # noqa: E402
    is_available,
    load_character_dialogues,
    load_qa,
)
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
from paramem.models.loader import unload_model  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

SCALE_POINTS = [10, 25, 50, 75, 100]
OUTPUT_BASE = project_root / "outputs" / "test1_scale"
DEFAULT_CHARACTER = "Liang Xin"


def distill_sessions_incrementally(model, tokenizer, sessions, target_count):
    """Process sessions one by one until we have enough QA pairs.

    Returns (qa_pairs, sessions_used) where sessions_used is the number
    of sessions that were processed.
    """
    qa_pairs = []
    seen_questions = set()

    for i, session in enumerate(sessions):
        logger.info(
            "Processing session %d/%d: %s (%d QA pairs so far, need %d)",
            i + 1,
            len(sessions),
            session["session_id"],
            len(qa_pairs),
            target_count,
        )
        new_pairs = distill_session(model, tokenizer, session, seen_questions)
        qa_pairs.extend(new_pairs)

        if len(qa_pairs) >= target_count:
            logger.info(
                "Reached %d QA pairs after %d sessions (target: %d)",
                len(qa_pairs),
                i + 1,
                target_count,
            )
            return qa_pairs, i + 1

    logger.info(
        "Exhausted all %d sessions, collected %d QA pairs (target: %d)",
        len(sessions),
        len(qa_pairs),
        target_count,
    )
    return qa_pairs, len(sessions)


def run_scale_point(
    model,
    tokenizer,
    qa_pairs,
    scale,
    epochs,
    rank,
    skip_rag,
    output_dir,
):
    """Train and evaluate at a single scale point."""
    subset = qa_pairs[:scale]
    if len(subset) < scale:
        logger.warning("Only %d QA pairs available, requested %d", len(subset), scale)

    logger.info("=== Scale point: %d keys ===", len(subset))

    adapter_name = f"episodic_s{scale}"
    if hasattr(model, "peft_config") and adapter_name in model.peft_config:
        model.delete_adapter(adapter_name)

    model, keyed_pairs, registry, train_time, metrics = train_indexed_keys(
        model,
        tokenizer,
        subset,
        epochs=epochs,
        rank=rank,
        adapter_name=adapter_name,
        output_dir=output_dir / f"scale_{scale}",
        run_name=f"scale-{scale}",
        skip_distill=True,  # Already distilled from sessions
    )

    recall_result = evaluate_indexed_recall(
        model,
        tokenizer,
        keyed_pairs,
        registry,
        adapter_name=adapter_name,
    )

    result = {
        "scale": len(subset),
        "epochs": epochs,
        "rank": rank,
        "training_time_seconds": train_time,
        "training_loss": metrics.get("train_loss"),
        "parametric_recall": recall_result,
    }

    if not skip_rag:
        from paramem.evaluation.rag_qa import QARAGPipeline, evaluate_rag_recall

        rag = QARAGPipeline()
        rag.build_index(subset)

        model.disable_adapter_layers()
        model.gradient_checkpointing_disable()

        rag_questions = [
            {"question": qa["question"], "expected_answer": qa["answer"]} for qa in subset
        ]
        rag_results = evaluate_rag_recall(rag, model, tokenizer, rag_questions)
        rag_mean = (
            sum(r["similarity"] for r in rag_results) / len(rag_results) if rag_results else 0.0
        )

        result["rag_recall"] = {
            "mean_similarity": rag_mean,
            "per_question": rag_results,
        }

        model.enable_adapter_layers()

    return result


def main():
    parser = argparse.ArgumentParser(description="Test 1: Scale Expansion")
    parser.add_argument(
        "--scale", type=int, default=None, help="Run a single scale point (e.g. 50)"
    )
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument(
        "--character",
        type=str,
        default=DEFAULT_CHARACTER,
        help="PerLTQA character name (default: Liang Xin)",
    )
    parser.add_argument("--skip-rag", action="store_true", help="Skip RAG baseline comparison")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_BASE))
    add_model_args(parser)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    scales = [args.scale] if args.scale else SCALE_POINTS
    max_needed = max(scales)

    # Load dialogue sessions (not yet distilled)
    sessions = []
    source = "synthetic:personal_facts"
    if is_available():
        sessions = load_character_dialogues(args.character)
        if sessions:
            source = f"perltqa_dialogues:{args.character}"

    for bench_name, bench_model_config in get_benchmark_models(args):
        print(f"\n{'=' * 72}")
        print(f"  Model: {bench_name} ({bench_model_config.model_id})")
        print(f"{'=' * 72}")

        model, tokenizer, config = load_model_and_config(bench_model_config)
        bench_output_dir = model_output_dir(output_dir, bench_name)

        # Distill sessions on the fly until we have enough QA pairs
        if sessions:
            qa_pairs, sessions_used = distill_sessions_incrementally(
                model, tokenizer, sessions, max_needed
            )
            print(f"  Data: {len(qa_pairs)} QA pairs from {sessions_used} sessions ({source})")
        else:
            logger.info("No PerLTQA sessions, using synthetic fallback")
            qa_pairs, source = load_qa(max_pairs=max_needed)
            sessions_used = 0
            print(f"  Data: {len(qa_pairs)} QA pairs ({source})")

        if len(qa_pairs) < max_needed:
            logger.warning(
                "Only %d QA pairs available, some scale points will be truncated",
                len(qa_pairs),
            )

        all_results = {
            "experiment": "test1_scale_expansion",
            "model": bench_name,
            "model_id": bench_model_config.model_id,
            "data_source": source,
            "sessions_used": sessions_used,
            "total_qa_available": len(qa_pairs),
            "scale_points": {},
        }

        for scale in scales:
            result = run_scale_point(
                model,
                tokenizer,
                qa_pairs,
                scale,
                args.num_epochs,
                args.rank,
                args.skip_rag,
                bench_output_dir,
            )
            all_results["scale_points"][str(scale)] = result

            pr = result["parametric_recall"]
            print(
                f"\n  Scale {result['scale']}: "
                f"{pr['exact_count']}/{pr['total']} recall "
                f"(conf={pr['mean_confidence']:.3f}, "
                f"time={result['training_time_seconds']:.0f}s)"
            )
            if "rag_recall" in result:
                print(
                    f"  RAG baseline: {result['rag_recall']['mean_similarity']:.3f} mean similarity"
                )

        # Summary table
        print(f"\n{'=' * 72}")
        print(f"SCALE EXPANSION SUMMARY ({bench_name})")
        print(f"{'Scale':>6} {'Recall':>10} {'Confidence':>12} {'Time':>8} {'RAG':>8}")
        print("-" * 50)
        for scale_str, result in all_results["scale_points"].items():
            pr = result["parametric_recall"]
            rag_str = (
                f"{result['rag_recall']['mean_similarity']:.3f}"
                if "rag_recall" in result
                else "N/A"
            )
            print(
                f"{result['scale']:>6} "
                f"{pr['exact_count']}/{pr['total']:>6} "
                f"{pr['mean_confidence']:>12.3f} "
                f"{result['training_time_seconds']:>7.0f}s "
                f"{rag_str:>8}"
            )
        print("=" * 72)

        save_results(all_results, bench_output_dir)
        unload_model(model, tokenizer)


if __name__ == "__main__":
    main()
