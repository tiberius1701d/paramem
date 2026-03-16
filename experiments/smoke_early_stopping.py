"""Smoke test for early stopping — validates recall holds with fewer epochs.

Trains 25 keys with early stopping enabled (floor=10, threshold=0.01).
Compares against the known 30-epoch results to confirm no regression.

Usage:
    python experiments/smoke_early_stopping.py --model gemma
    python experiments/smoke_early_stopping.py --model mistral
"""

import argparse
import logging
import sys
import time as _time
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
    save_results,
    setup_logging,
    train_indexed_keys,
)
from paramem.models.loader import unload_model  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

OUTPUT_DIR = project_root / "outputs" / f"smoke_early_stopping_{int(_time.time())}"


def main():
    parser = argparse.ArgumentParser(description="Early stopping smoke test")
    parser.add_argument("--num-epochs", type=int, default=30, help="Max epochs (cap)")
    parser.add_argument("--rank", type=int, default=8)
    add_model_args(parser)
    args = parser.parse_args()

    output_dir = OUTPUT_DIR

    # Load data
    sessions = []
    if is_available():
        sessions = load_character_dialogues("Liang Xin")

    for bench_name, bench_model_config in get_benchmark_models(args):
        print(f"\n{'=' * 60}")
        print(f"  Early stopping smoke test: {bench_name}")
        print(f"{'=' * 60}")

        model, tokenizer, config = load_model_and_config(bench_model_config)

        # Distill 25 QA pairs
        if sessions:
            qa_pairs = []
            seen = set()
            for session in sessions:
                new = distill_session(model, tokenizer, session, seen)
                qa_pairs.extend(new)
                if len(qa_pairs) >= 25:
                    break
            qa_pairs = qa_pairs[:25]
            source = "perltqa_dialogues:Liang Xin"
        else:
            qa_pairs, source = load_qa(max_pairs=25)

        print(f"  Data: {len(qa_pairs)} QA pairs ({source})")

        adapter_name = "episodic_smoke"
        if hasattr(model, "peft_config") and adapter_name in model.peft_config:
            model.delete_adapter(adapter_name)

        model, keyed_pairs, registry, train_time, metrics = train_indexed_keys(
            model,
            tokenizer,
            qa_pairs,
            epochs=args.num_epochs,
            rank=args.rank,
            adapter_name=adapter_name,
            output_dir=output_dir / bench_name,
            run_name=f"smoke-earlystop-{bench_name}",
            skip_distill=True,
        )

        recall_result = evaluate_indexed_recall(
            model, tokenizer, keyed_pairs, registry,
            adapter_name=adapter_name,
        )

        pr = recall_result
        print(f"\n  Recall: {pr['exact_count']}/{pr['total']} ({pr['rate']:.0%})")
        print(f"  Confidence: {pr['mean_confidence']:.3f}")
        print(f"  Training time: {train_time:.0f}s ({train_time/60:.1f}min)")
        print(f"  Training loss: {metrics.get('train_loss', 0):.4f}")
        print(f"  Epoch reached: {metrics.get('epoch', '?')}")

        result = {
            "experiment": "smoke_early_stopping",
            "model": bench_name,
            "model_id": bench_model_config.model_id,
            "scale": len(qa_pairs),
            "recall": pr["exact_count"],
            "total": pr["total"],
            "rate": pr["rate"],
            "confidence": pr["mean_confidence"],
            "training_time_seconds": train_time,
            "training_loss": metrics.get("train_loss"),
            "epoch_reached": metrics.get("epoch"),
            "per_key": pr["per_key"],
        }
        save_results(result, output_dir / bench_name)

        print("\n  30-epoch baseline: 24-25/25 recall, ~17min (Gemma) / ~14min (Mistral)")
        print(f"  Early stop result: {pr['exact_count']}/{pr['total']}, {train_time/60:.1f}min")
        print("=" * 60)

        unload_model(model, tokenizer)


if __name__ == "__main__":
    main()
