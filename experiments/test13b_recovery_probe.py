"""Test 13b recovery probe — is unchanged-key damage latent or overwritten?

After Test 13b's prod run the final adapter has retention 63/160 on the
unchanged set. This probe asks whether the 97 failing keys are latent
(mapping distorted, representation still present — should recover with
very little replay) or overwritten (representation erased — won't come
back without full replay).

Method:
  1. Load the Test 13b final adapter (slot dir under adapter/).
  2. Probe all 160 unchanged keys → identify the failing subset.
  3. Train 2 epochs at LR=1e-5 (10× lower than fill training) on ONLY
     the failing keys.
  4. Re-probe all 160 unchanged keys.
  5. Report: how many recovered? Of the 63 that were already passing,
     how many stayed passing?

This is a diagnostic, not a benchmark. No pause/resume, no per-epoch
probe — one fixed short run, single GPU acquire. Expected wall ~20 min.

Usage:
    python experiments/test13b_recovery_probe.py \
        [--source-run outputs/test13b_retention_curve/mistral/20260423_002702] \
        [--epochs 2] [--lr 1e-5]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from peft import PeftModel  # noqa: E402

from experiments.utils.gpu_guard import acquire_gpu  # noqa: E402
from experiments.utils.test_harness import (  # noqa: E402
    BENCHMARK_MODELS,
    IndexedDataset,
    evaluate_indexed_recall,
    load_model_and_config,
    save_results,
)
from paramem.adapters.manifest import resolve_adapter_slot  # noqa: E402
from paramem.models.loader import switch_adapter, unload_model  # noqa: E402
from paramem.training.indexed_memory import format_indexed_training  # noqa: E402
from paramem.training.trainer import train_adapter  # noqa: E402
from paramem.utils.config import AdapterConfig, TrainingConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SOURCE_RUN = project_root / ("outputs/test13b_retention_curve/mistral/20260423_002702")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test 13b recovery probe")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--model", choices=["mistral"], default="mistral")
    args = parser.parse_args()

    source = args.source_run
    if not source.is_dir():
        raise SystemExit(f"source run not found: {source}")

    # Output dir is a sibling of the source run
    output_dir = source / "recovery_probe"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", output_dir)

    # Load data
    unchanged_keyed = json.loads((source / "unchanged_keyed.json").read_text())
    unchanged_registry = {
        k: int(v)
        for k, v in json.loads((source / "unchanged_simhash_registry.json").read_text()).items()
    }
    logger.info("Loaded %d unchanged keys", len(unchanged_keyed))

    adapter_dir = source / "adapter"
    adapter_name = "journal"
    model_config = BENCHMARK_MODELS[args.model]

    with acquire_gpu(interactive=True):
        model, tokenizer = load_model_and_config(model_config)

        slot = resolve_adapter_slot(adapter_dir, adapter_name, "")
        if slot is None:
            raise SystemExit(f"could not resolve adapter slot under {adapter_dir}")
        logger.info("Loading Test 13b final adapter from slot: %s", slot)
        model = PeftModel.from_pretrained(model, str(slot), adapter_name=adapter_name)
        switch_adapter(model, adapter_name)

        # --- Phase 1: Probe all 160 to identify failures ---
        logger.info("Phase 1: baseline probe on 160 unchanged keys")
        t0 = time.time()
        baseline = evaluate_indexed_recall(
            model,
            tokenizer,
            unchanged_keyed,
            unchanged_registry,
            adapter_name=adapter_name,
        )
        model.gradient_checkpointing_enable()
        logger.info(
            "  baseline retention: %d/%d (%.3f, conf=%.3f) — %.1fs",
            baseline["exact_count"],
            baseline["total"],
            baseline["rate"],
            baseline["mean_confidence"],
            time.time() - t0,
        )

        # Failing-key subset (no exact match)
        failing_keys = {r["key"] for r in baseline["per_key"] if not r["exact_match"]}
        passing_keys = {r["key"] for r in baseline["per_key"] if r["exact_match"]}
        failing_kps = [kp for kp in unchanged_keyed if kp["key"] in failing_keys]
        logger.info(
            "  failing: %d, passing: %d — will replay only failing keys",
            len(failing_kps),
            len(passing_keys),
        )
        if not failing_kps:
            logger.warning("Nothing to recover — exiting")
            return

        # --- Phase 2: Short replay on failing keys only ---
        logger.info(
            "Phase 2: %d epochs on %d failing keys at LR=%.1e",
            args.epochs,
            len(failing_kps),
            args.lr,
        )
        examples = format_indexed_training(failing_kps, tokenizer, max_length=1024)
        dataset = IndexedDataset(examples)

        training_config = TrainingConfig(
            batch_size=1,
            gradient_accumulation_steps=2,
            max_seq_length=1024,
            num_epochs=args.epochs,
            warmup_ratio=0.1,
            weight_decay=0.01,
            gradient_checkpointing=True,
            max_grad_norm=1.0,
            seed=42,
            save_strategy="no",
            save_total_limit=1,
        )
        adapter_config = AdapterConfig(
            rank=8,
            alpha=16,
            learning_rate=args.lr,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            dropout=0.0,
        )

        t0 = time.time()
        metrics = train_adapter(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            adapter_name=adapter_name,
            training_config=training_config,
            adapter_config=adapter_config,
            output_dir=output_dir / "replay_adapter",
            run_name="test13b-recovery-probe",
        )
        replay_wall = time.time() - t0
        logger.info(
            "  replay done: train_loss=%.4f wall=%.1fs",
            metrics.get("train_loss", float("nan")),
            replay_wall,
        )

        # --- Phase 3: Re-probe all 160 ---
        logger.info("Phase 3: re-probe on full 160 unchanged keys")
        t0 = time.time()
        after = evaluate_indexed_recall(
            model,
            tokenizer,
            unchanged_keyed,
            unchanged_registry,
            adapter_name=adapter_name,
        )
        logger.info(
            "  after retention: %d/%d (%.3f, conf=%.3f) — %.1fs",
            after["exact_count"],
            after["total"],
            after["rate"],
            after["mean_confidence"],
            time.time() - t0,
        )

        # --- Analyse recovery ---
        after_status = {r["key"]: r["exact_match"] for r in after["per_key"]}
        recovered = sum(1 for k in failing_keys if after_status.get(k, False))
        still_failing = len(failing_keys) - recovered
        lost = sum(1 for k in passing_keys if not after_status.get(k, True))
        still_passing = len(passing_keys) - lost

        logger.info("=" * 72)
        logger.info("Test 13b recovery probe — summary")
        logger.info("=" * 72)
        logger.info(
            "Baseline (after Test 13b e30, pre-replay): %d/%d (%.3f)",
            baseline["exact_count"],
            baseline["total"],
            baseline["rate"],
        )
        logger.info(
            "After %d epochs @ LR=%.1e on %d failing keys: %d/%d (%.3f)",
            args.epochs,
            args.lr,
            len(failing_kps),
            after["exact_count"],
            after["total"],
            after["rate"],
        )
        logger.info(
            "  of %d originally failing:   %d recovered, %d still failing",
            len(failing_keys),
            recovered,
            still_failing,
        )
        logger.info(
            "  of %d originally passing:   %d kept, %d lost", len(passing_keys), still_passing, lost
        )
        logger.info(
            "  recovery rate: %.1f%% of failing",
            100.0 * recovered / len(failing_keys) if failing_keys else 0.0,
        )

        results = {
            "source_run": str(source),
            "epochs": args.epochs,
            "lr": args.lr,
            "n_failing": len(failing_keys),
            "n_passing": len(passing_keys),
            "baseline_retention": {
                "exact_count": baseline["exact_count"],
                "total": baseline["total"],
                "rate": baseline["rate"],
                "mean_confidence": baseline["mean_confidence"],
            },
            "after_replay_retention": {
                "exact_count": after["exact_count"],
                "total": after["total"],
                "rate": after["rate"],
                "mean_confidence": after["mean_confidence"],
            },
            "recovered": recovered,
            "still_failing": still_failing,
            "kept": still_passing,
            "lost_during_replay": lost,
            "recovery_rate": (recovered / len(failing_keys) if failing_keys else 0.0),
            "replay_train_loss": metrics.get("train_loss"),
            "replay_wall_seconds": round(replay_wall, 1),
        }
        save_results(results, output_dir, "recovery_probe.json")
        logger.info("Wrote: %s", output_dir / "recovery_probe.json")

        unload_model(model, tokenizer)


if __name__ == "__main__":
    main()
