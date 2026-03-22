"""Test 4b: Incremental Learning Without Full Replay.

Tests whether facts survive when only new keys are trained each cycle.
The adapter learns new facts without retraining old ones. Measures old-key
degradation over 5 incremental cycles, then a final full-replay retrain.

Design:
  1. Train 20 initial keys (baseline, 30 epochs)
  2. 5 incremental cycles: add 5 new keys each, train ONLY new keys
  3. After each cycle: measure recall on ALL keys (old + new)
  4. Final full-replay retrain on all 45 keys

Usage:
    python experiments/test4b_incremental_no_replay.py --model gemma
    python experiments/test4b_incremental_no_replay.py --model mistral
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.utils.perltqa_loader import is_available as perltqa_available  # noqa: E402
from experiments.utils.perltqa_loader import load_character_eval_qa  # noqa: E402
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
from paramem.models.loader import switch_adapter, unload_model  # noqa: E402
from paramem.training.indexed_memory import (  # noqa: E402
    assign_keys,
    build_registry,
)

setup_logging()
logger = logging.getLogger(__name__)

OUTPUT_DIR = project_root / "outputs" / "test4b_incremental"
INITIAL_KEYS = 20
KEYS_PER_CYCLE = 5
NUM_CYCLES = 5
CHARACTER = "Liang Xin"
CHARACTER_B = "Cai Xiuying"  # fallback if not enough QA pairs


def main():
    parser = argparse.ArgumentParser(
        description="Test 4b: Incremental Learning Without Full Replay"
    )
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    add_model_args(parser)
    args = parser.parse_args()

    base_output_dir = Path(args.output_dir)
    total_needed = INITIAL_KEYS + NUM_CYCLES * KEYS_PER_CYCLE  # 45

    # Load QA pairs
    if perltqa_available():
        qa_pairs = load_character_eval_qa(CHARACTER, max_pairs=total_needed)
        if len(qa_pairs) < total_needed:
            extra = load_character_eval_qa(CHARACTER_B, max_pairs=total_needed - len(qa_pairs))
            qa_pairs.extend(extra)
        source = f"perltqa:{CHARACTER}"
    else:
        print("ERROR: PerLTQA required for Test 4b (need 45+ QA pairs)")
        sys.exit(1)

    if len(qa_pairs) < total_needed:
        print(f"ERROR: Need {total_needed} QA pairs, got {len(qa_pairs)}")
        sys.exit(1)

    qa_pairs = qa_pairs[:total_needed]
    print(f"Loaded {len(qa_pairs)} QA pairs from {source}")

    for bench_name, bench_model_config in get_benchmark_models(args):
        print(f"\n{'=' * 72}")
        print(f"  Test 4b: Incremental No Replay — {bench_name}")
        print(f"{'=' * 72}")

        model, tokenizer, config = load_model_and_config(bench_model_config)
        output_dir = model_output_dir(base_output_dir, bench_name)

        adapter_name = "episodic"
        cycle_results = []

        # Phase 0: Train initial 20 keys (baseline)
        print(f"\n--- Cycle 0: Baseline ({INITIAL_KEYS} keys) ---")
        initial_qa = qa_pairs[:INITIAL_KEYS]

        model, keyed_pairs, registry, train_time, metrics = train_indexed_keys(
            model,
            tokenizer,
            initial_qa,
            epochs=args.num_epochs,
            rank=args.rank,
            adapter_name=adapter_name,
            output_dir=output_dir / "cycle_0",
            run_name="incremental-cycle0",
            skip_distill=True,
        )

        # Evaluate all trained keys so far
        recall = evaluate_indexed_recall(
            model,
            tokenizer,
            keyed_pairs,
            registry,
            adapter_name=adapter_name,
        )

        # Track epoch convergence from training loss
        cycle_results.append(
            {
                "cycle": 0,
                "keys_before": 0,
                "new_keys": INITIAL_KEYS,
                "total_keys": INITIAL_KEYS,
                "new_recall": recall["exact_count"],
                "new_total": recall["total"],
                "old_recall": None,
                "old_total": None,
                "mean_confidence": recall["mean_confidence"],
                "train_loss": metrics.get("train_loss"),
                "train_time": round(train_time, 1),
            }
        )

        print(
            f"  Baseline: {recall['exact_count']}/{recall['total']} "
            f"(conf={recall['mean_confidence']:.3f}, "
            f"loss={metrics.get('train_loss', -1):.4f}, "
            f"time={train_time:.0f}s)"
        )

        # Save checkpoint
        from paramem.models.loader import save_adapter

        save_adapter(model, output_dir / "cycle_0" / "checkpoint", adapter_name)

        # Track all keys trained so far
        trained_keys = list(keyed_pairs)
        trained_registry = dict(registry)

        # Phases 1-5: Incremental cycles
        for cycle in range(1, NUM_CYCLES + 1):
            start_idx = INITIAL_KEYS + (cycle - 1) * KEYS_PER_CYCLE
            end_idx = start_idx + KEYS_PER_CYCLE
            new_qa = qa_pairs[start_idx:end_idx]

            print(f"\n--- Cycle {cycle}: +{KEYS_PER_CYCLE} new keys (total {end_idx}) ---")

            # Train ONLY new keys on the existing adapter (no replay)
            # Do NOT unwrap to base model — keep existing adapter weights
            switch_adapter(model, adapter_name)

            new_keyed = assign_keys(new_qa, start_index=start_idx + 1)
            new_registry = build_registry(new_keyed)

            from experiments.utils.test_harness import IndexedDataset
            from paramem.training.indexed_memory import format_indexed_training
            from paramem.training.trainer import train_adapter
            from paramem.utils.config import AdapterConfig, TrainingConfig

            adapter_config = AdapterConfig(
                rank=args.rank,
                alpha=args.rank * 2,
            )

            examples = format_indexed_training(new_keyed, tokenizer, max_length=1024)
            dataset = IndexedDataset(examples)

            # Per-epoch training with recall probing after each epoch
            t0 = time.time()
            epoch_log = []
            first_perfect_epoch = None
            inc_metrics = {}

            for epoch in range(1, args.num_epochs + 1):
                one_epoch_config = TrainingConfig(
                    batch_size=1,
                    gradient_accumulation_steps=2,
                    max_seq_length=1024,
                    num_epochs=1,
                    warmup_ratio=0.1 if epoch == 1 else 0.0,
                    weight_decay=0.01,
                    gradient_checkpointing=True,
                    max_grad_norm=1.0,
                    seed=42,
                )
                inc_metrics = train_adapter(
                    model=model,
                    tokenizer=tokenizer,
                    train_dataset=dataset,
                    adapter_name=adapter_name,
                    training_config=one_epoch_config,
                    adapter_config=adapter_config,
                    output_dir=output_dir / f"cycle_{cycle}" / "adapter",
                    run_name=f"inc-c{cycle}-e{epoch}",
                )

                # Probe new keys
                model.gradient_checkpointing_disable()
                ep_recall = evaluate_indexed_recall(
                    model,
                    tokenizer,
                    new_keyed,
                    new_registry,
                    adapter_name=adapter_name,
                )
                epoch_log.append(
                    {
                        "epoch": epoch,
                        "recall": ep_recall["exact_count"],
                        "total": ep_recall["total"],
                        "loss": inc_metrics.get("train_loss"),
                    }
                )

                if first_perfect_epoch is None and ep_recall["exact_count"] == ep_recall["total"]:
                    first_perfect_epoch = epoch

                # Early stop: 2 consecutive perfect epochs
                if (
                    len(epoch_log) >= 2
                    and epoch_log[-1]["recall"] == epoch_log[-1]["total"]
                    and epoch_log[-2]["recall"] == epoch_log[-2]["total"]
                ):
                    break

            inc_time = time.time() - t0
            epochs_trained = len(epoch_log)

            # Update tracked keys
            trained_keys.extend(new_keyed)
            trained_registry.update(new_registry)

            # Save keyed_pairs for this cycle
            cycle_dir = output_dir / f"cycle_{cycle}"
            cycle_dir.mkdir(parents=True, exist_ok=True)
            with open(cycle_dir / "keyed_pairs.json", "w") as f:
                json.dump(
                    [
                        {
                            "key": kp["key"],
                            "question": kp["question"],
                            "answer": kp["answer"],
                        }
                        for kp in new_keyed
                    ],
                    f,
                    indent=2,
                )

            # Persist cumulative registry for resumability
            from paramem.training.indexed_memory import save_registry

            save_registry(trained_registry, cycle_dir / "cumulative_registry.json")

            # New key recall from final epoch
            new_recall = epoch_log[-1]

            # Evaluate OLD keys (everything before this cycle)
            old_keyed = trained_keys[:-KEYS_PER_CYCLE]
            old_registry = {
                kp["key"]: trained_registry[kp["key"]]
                for kp in old_keyed
                if kp["key"] in trained_registry
            }
            old_recall = evaluate_indexed_recall(
                model,
                tokenizer,
                old_keyed,
                old_registry,
                adapter_name=adapter_name,
            )

            cycle_results.append(
                {
                    "cycle": cycle,
                    "keys_before": len(old_keyed),
                    "new_keys": KEYS_PER_CYCLE,
                    "total_keys": len(trained_keys),
                    "new_recall": new_recall["recall"],
                    "new_total": new_recall["total"],
                    "old_recall": old_recall["exact_count"],
                    "old_total": old_recall["total"],
                    "mean_confidence": old_recall["mean_confidence"],
                    "train_loss": inc_metrics.get("train_loss"),
                    "train_time": round(inc_time, 1),
                    "epochs_trained": epochs_trained,
                    "first_perfect_epoch": first_perfect_epoch,
                    "epoch_log": epoch_log,
                }
            )

            perf_str = f"e{first_perfect_epoch}" if first_perfect_epoch else "never"
            print(
                f"  New: {new_recall['recall']}/{new_recall['total']} "
                f"(recall@{perf_str}, {epochs_trained} epochs), "
                f"Old: {old_recall['exact_count']}/{old_recall['total']} "
                f"(conf={old_recall['mean_confidence']:.3f}), "
                f"time={inc_time:.0f}s"
            )

            # Save checkpoint
            save_adapter(
                model,
                output_dir / f"cycle_{cycle}" / "checkpoint",
                adapter_name,
            )

        # Phase 6: Full-replay retrain on all 45 keys
        print(f"\n--- Full Replay Retrain ({len(trained_keys)} keys) ---")

        from peft import PeftModel as _PeftModel

        if isinstance(model, _PeftModel):
            model = model.base_model.model

        model, final_keyed, final_registry, final_time, final_metrics = train_indexed_keys(
            model,
            tokenizer,
            qa_pairs[:total_needed],
            epochs=args.num_epochs,
            rank=args.rank,
            adapter_name=adapter_name,
            output_dir=output_dir / "full_retrain",
            run_name="incremental-full-retrain",
            skip_distill=True,
        )

        final_recall = evaluate_indexed_recall(
            model,
            tokenizer,
            final_keyed,
            final_registry,
            adapter_name=adapter_name,
        )

        full_retrain_result = {
            "total_keys": len(final_keyed),
            "recall": final_recall["exact_count"],
            "total": final_recall["total"],
            "mean_confidence": final_recall["mean_confidence"],
            "train_loss": final_metrics.get("train_loss"),
            "train_time": round(final_time, 1),
        }

        print(
            f"  Full retrain: "
            f"{final_recall['exact_count']}/{final_recall['total']} "
            f"(conf={final_recall['mean_confidence']:.3f}, "
            f"time={final_time:.0f}s)"
        )

        # Summary
        print(f"\n{'=' * 72}")
        print("TEST 4b: INCREMENTAL NO REPLAY SUMMARY")
        print(f"{'=' * 72}")
        print(
            f"  {'Cycle':>5} {'Before':>6} {'New':>4} "
            f"{'New OK':>7} {'Old OK':>7} {'Conf':>6} "
            f"{'Recall@':>7} {'Epochs':>6} {'Time':>6}"
        )
        print("-" * 66)
        for cr in cycle_results:
            old_str = (
                f"{cr['old_recall']}/{cr['old_total']}" if cr["old_recall"] is not None else "n/a"
            )
            perf = cr.get("first_perfect_epoch")
            perf_str = f"e{perf}" if perf else "n/a"
            ep_str = str(cr.get("epochs_trained", args.num_epochs))
            print(
                f"  {cr['cycle']:>5} {cr['keys_before'] or 0:>6} "
                f"{cr['new_keys']:>4} "
                f"{cr['new_recall']}/{cr['new_total']:>5} "
                f"{old_str:>7} "
                f"{cr['mean_confidence']:>5.3f} "
                f"{perf_str:>7} "
                f"{ep_str:>6} "
                f"{cr['train_time']:>5.0f}s"
            )
        print("-" * 62)
        fr = full_retrain_result
        print(
            f"  {'FULL':>5} {'':>6} {fr['total_keys']:>4} "
            f"{fr['recall']}/{fr['total']:>5} "
            f"{'':>7} "
            f"{fr['mean_confidence']:>5.3f} "
            f"{fr['train_loss']:>7.4f} "
            f"{fr['train_time']:>5.0f}s"
        )
        incremental_total = sum(cr["train_time"] for cr in cycle_results)
        print(f"\n  Incremental total: {incremental_total:.0f}s ({incremental_total / 60:.1f} min)")
        print(f"  Full retrain:      {fr['train_time']:.0f}s ({fr['train_time'] / 60:.1f} min)")
        print(f"{'=' * 72}")

        # Save results
        results = {
            "experiment": "test4b_incremental_no_replay",
            "model": bench_name,
            "model_id": bench_model_config.model_id,
            "initial_keys": INITIAL_KEYS,
            "keys_per_cycle": KEYS_PER_CYCLE,
            "num_cycles": NUM_CYCLES,
            "epochs": args.num_epochs,
            "rank": args.rank,
            "data_source": source,
            "cycle_results": cycle_results,
            "full_retrain": full_retrain_result,
        }

        save_results(results, output_dir)
        unload_model(model, tokenizer)


if __name__ == "__main__":
    main()
