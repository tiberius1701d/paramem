"""Test 1: Scale Expansion — indexed key recall at 10, 25, 50, 75, 100 keys.

Tests whether parametric memory maintains recall quality as the number of
stored facts increases. Uses PerLTQA dataset (public) for realistic QA pairs,
with synthetic fallback. Compares against QA-RAG baseline at each scale point.

Usage:
    python experiments/test1_scale_expansion.py
    python experiments/test1_scale_expansion.py --scale 50
    python experiments/test1_scale_expansion.py --character-id <id>
    python experiments/test1_scale_expansion.py --skip-rag
"""

import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.utils.perltqa_loader import load_qa  # noqa: E402
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

SCALE_POINTS = [10, 25, 50, 75, 100]
OUTPUT_BASE = project_root / "outputs" / "test1_scale"


def run_scale_point(
    model,
    tokenizer,
    qa_pairs,
    scale,
    epochs,
    rank,
    skip_rag,
    output_dir,
    distillation_config=None,
):
    """Train and evaluate at a single scale point."""
    subset = qa_pairs[:scale]
    if len(subset) < scale:
        logger.warning("Only %d QA pairs available, requested %d", len(subset), scale)

    logger.info("=== Scale point: %d keys ===", len(subset))

    # Delete existing adapter if present (fresh start per scale point)
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
        distillation_config=distillation_config,
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

    # RAG baseline comparison
    if not skip_rag:
        from paramem.evaluation.rag_qa import QARAGPipeline, evaluate_rag_recall

        rag = QARAGPipeline()
        rag.build_index(subset)

        # Disable adapters for RAG (use base model)
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
    parser.add_argument("--character-id", type=str, default=None, help="PerLTQA character ID")
    parser.add_argument("--skip-rag", action="store_true", help="Skip RAG baseline comparison")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_BASE))
    add_distillation_args(parser)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    scales = [args.scale] if args.scale else SCALE_POINTS
    max_needed = max(scales)

    # Load data
    qa_pairs, source = load_qa(args.character_id, max_pairs=max_needed)
    logger.info("Loaded %d QA pairs from %s", len(qa_pairs), source)

    if len(qa_pairs) < max_needed:
        logger.warning(
            "Only %d QA pairs available, some scale points will be truncated",
            len(qa_pairs),
        )

    # Load model
    model, tokenizer, config = load_model_and_config()

    for model_name, distillation_config in get_distillation_configs(args):
        print(f"\n{'=' * 72}")
        print(f"  Distillation model: {model_name}")
        print(f"{'=' * 72}")

        model_output_dir = distillation_output_dir(output_dir, model_name)

        all_results = {
            "experiment": "test1_scale_expansion",
            "distillation_model": model_name,
            "data_source": source,
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
                model_output_dir,
                distillation_config=distillation_config,
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
                    f"  RAG baseline: "
                    f"{result['rag_recall']['mean_similarity']:.3f} mean similarity"
                )

        # Summary table
        print(f"\n{'=' * 72}")
        print(f"SCALE EXPANSION SUMMARY ({model_name})")
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

        save_results(all_results, model_output_dir)


if __name__ == "__main__":
    main()
