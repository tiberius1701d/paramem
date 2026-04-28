"""Weight Diff Analysis: How does adding one key change the adapter weight landscape?

Loads a completed adapter (cycle 11, 108 keys), snapshots all LoRA weights,
trains on the same 108 keys + 1 synthetic key, snapshots again, and computes
per-module weight deltas.

Tests the "slot hypothesis": do new keys activate unused capacity without
disturbing existing key weights?

Uses a COPY of the cycle 11 adapter — Test 8 results are never touched.

Usage:
    python experiments/weight_diff_analysis.py
"""

import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from peft import PeftModel as _PeftModel  # noqa: E402

from experiments.utils.test_harness import (  # noqa: E402
    BENCHMARK_MODELS,
    IndexedDataset,
    evaluate_indexed_recall,
    load_model_and_config,
    setup_logging,
)
from paramem.training.indexed_memory import (  # noqa: E402
    assign_keys,
    build_registry,
    format_indexed_training,
    save_registry,
)
from paramem.training.trainer import train_adapter  # noqa: E402
from paramem.utils.config import AdapterConfig, TrainingConfig  # noqa: E402

setup_logging()

# Paths
CYCLE_11_DIR = project_root / "outputs/test8_large_scale/mistral/20260323_161747/cycle_011"
OUTPUT_DIR = project_root / "outputs/weight_diff_analysis"

# Training params — match Test 8 exactly
RANK = 8
NUM_EPOCHS = 30
ADAPTER_NAME = "episodic"

SYNTHETIC_KEY = {
    "question": "What is the boiling point of mercury?",
    "answer": "The boiling point of mercury is 356.73 degrees Celsius.",
    "source_predicate": "has_boiling_point",
    "source_subject": "Mercury",
    "source_object": "356.73 Degrees Celsius",
}


def snapshot_lora_weights(model):
    """Capture all LoRA A and B matrices as numpy arrays."""
    snapshot = OrderedDict()
    for name, param in model.named_parameters():
        if "lora_" in name:
            snapshot[name] = param.detach().cpu().numpy().copy()
    return snapshot


def compute_weight_diffs(before, after):
    """Compute per-parameter deltas between two snapshots."""
    results = []
    for name in before:
        if name not in after:
            continue
        w_before = before[name]
        w_after = after[name]
        delta = w_after - w_before

        l2_before = np.linalg.norm(w_before)
        l2_delta = np.linalg.norm(delta)
        relative_change = l2_delta / l2_before if l2_before > 0 else float("inf")

        # Sparsity: fraction of elements with < 1% of max absolute change
        max_abs = np.max(np.abs(delta))
        if max_abs > 0:
            near_zero = np.mean(np.abs(delta) < 0.01 * max_abs)
        else:
            near_zero = 1.0

        # Per-row L2 norms (rows = output features for B, rank features for A)
        row_norms = np.linalg.norm(delta, axis=-1) if delta.ndim >= 2 else np.array([l2_delta])
        row_norm_before = (
            np.linalg.norm(w_before, axis=-1) if w_before.ndim >= 2 else np.array([l2_before])
        )

        results.append(
            {
                "name": name,
                "shape": list(w_before.shape),
                "l2_before": float(l2_before),
                "l2_delta": float(l2_delta),
                "relative_change": float(relative_change),
                "sparsity_99pct": float(near_zero),
                "max_abs_delta": float(max_abs),
                "mean_abs_delta": float(np.mean(np.abs(delta))),
                "row_norms_delta": row_norms.tolist(),
                "row_norms_before": row_norm_before.tolist(),
            }
        )
    return results


def print_summary(diffs):
    """Print a human-readable summary of weight changes."""
    print("\n" + "=" * 80)
    print("WEIGHT DIFF SUMMARY: 108 keys → 109 keys (1 synthetic addition)")
    print("=" * 80)

    # Group by module (q_proj, k_proj, v_proj, o_proj) and matrix (A, B)
    for d in diffs:
        name = d["name"]
        short = name.split("base_model.model.")[-1] if "base_model.model." in name else name
        print(f"\n  {short}")
        print(f"    Shape:           {d['shape']}")
        print(f"    L2 before:       {d['l2_before']:.4f}")
        print(f"    L2 delta:        {d['l2_delta']:.6f}")
        print(
            f"    Relative change: {d['relative_change']:.6f} ({d['relative_change'] * 100:.4f}%)"
        )
        print(f"    Sparsity (99%):  {d['sparsity_99pct']:.4f} (fraction near-zero)")
        print(f"    Max |delta|:     {d['max_abs_delta']:.6f}")
        print(f"    Mean |delta|:    {d['mean_abs_delta']:.8f}")

    # Aggregate
    total_params = sum(np.prod(d["shape"]) for d in diffs)
    total_l2_delta = np.sqrt(sum(d["l2_delta"] ** 2 for d in diffs))
    total_l2_before = np.sqrt(sum(d["l2_before"] ** 2 for d in diffs))
    avg_sparsity = np.mean([d["sparsity_99pct"] for d in diffs])

    print("\n" + "-" * 80)
    print(f"  Total LoRA parameters:  {total_params:,}")
    print(f"  Total L2 before:        {total_l2_before:.4f}")
    print(f"  Total L2 delta:         {total_l2_delta:.6f}")
    rel = total_l2_delta / total_l2_before
    print(f"  Overall relative:       {rel:.6f} ({rel * 100:.4f}%)")
    print(f"  Avg sparsity (99%):     {avg_sparsity:.4f}")
    print("=" * 80)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verify source exists
    if not CYCLE_11_DIR.exists():
        print(f"ERROR: Cycle 11 not found at {CYCLE_11_DIR}")
        sys.exit(1)

    # Load original keyed pairs
    with open(CYCLE_11_DIR / "keyed_pairs.json") as f:
        original_pairs = json.load(f)
    print(f"Loaded {len(original_pairs)} original keyed pairs from cycle 11")

    # Build QA list with synthetic addition
    qa_pairs_original = [
        {"question": kp["question"], "answer": kp["answer"]} for kp in original_pairs
    ]
    qa_pairs_with_synthetic = qa_pairs_original + [SYNTHETIC_KEY]
    print(f"Training set: {len(qa_pairs_with_synthetic)} pairs (108 original + 1 synthetic)")

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model_and_config(BENCHMARK_MODELS["mistral"])

    # ---- Phase 1: Train on original 108 keys (baseline) ----
    print("\n" + "=" * 80)
    print("PHASE 1: Training on 108 original keys (baseline)")
    print("=" * 80)

    baseline_dir = OUTPUT_DIR / "baseline_108"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = OUTPUT_DIR / "snapshot_baseline.npz"

    adapter_config = AdapterConfig(
        rank=RANK,
        alpha=RANK * 2,
        learning_rate=1e-4,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        dropout=0.0,
    )
    from paramem.models.loader import create_adapter

    keyed_original = assign_keys(qa_pairs_original, start_index=1)
    registry_original = build_registry(keyed_original)

    training_config = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=2,
        max_seq_length=1024,
        num_epochs=NUM_EPOCHS,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=42,
        save_strategy="no",
    )

    # Check if baseline adapter already exists (resume after crash)
    baseline_adapter_path = baseline_dir / "adapter" / ADAPTER_NAME
    if baseline_adapter_path.exists() and snapshot_path.exists():
        print("Loading saved baseline adapter and snapshot (skipping retrain)...")
        from peft import PeftModel as PeftModelLoader

        if isinstance(model, _PeftModel):
            model = model.base_model.model
        model = PeftModelLoader.from_pretrained(
            model, str(baseline_adapter_path), adapter_name=ADAPTER_NAME
        )

        snapshot_data = np.load(snapshot_path)
        snapshot_before = OrderedDict((k, snapshot_data[k]) for k in snapshot_data.files)
        baseline_time = 0.0
        print(f"Loaded snapshot: {len(snapshot_before)} LoRA parameter tensors")
    else:
        if isinstance(model, _PeftModel):
            model = model.base_model.model

        model = create_adapter(model, adapter_config, ADAPTER_NAME)
        save_registry(registry_original, baseline_dir / "simhash_registry.json")

        examples = format_indexed_training(keyed_original, tokenizer, max_length=1024)
        dataset = IndexedDataset(examples)

        t0 = time.time()
        train_adapter(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            adapter_name=ADAPTER_NAME,
            training_config=training_config,
            adapter_config=adapter_config,
            output_dir=baseline_adapter_path,
            run_name="weight-diff-baseline",
        )
        baseline_time = time.time() - t0
        print(f"Baseline training: {baseline_time:.0f}s")

        # Verify baseline recall
        recall_result = evaluate_indexed_recall(
            model, tokenizer, keyed_original, registry_original, ADAPTER_NAME
        )
        print(f"Baseline recall: {recall_result['rate']}")

        # Snapshot and save baseline weights
        snapshot_before = snapshot_lora_weights(model)
        np.savez(snapshot_path, **snapshot_before)
        print(f"Snapshot saved: {len(snapshot_before)} LoRA parameter tensors")

    # ---- Phase 2: Train on 109 keys (108 + 1 synthetic) ----
    print("\n" + "=" * 80)
    print("PHASE 2: Training on 109 keys (108 original + 1 synthetic)")
    print("=" * 80)

    added_dir = OUTPUT_DIR / "added_109"
    added_dir.mkdir(parents=True, exist_ok=True)

    # Unwrap and create fresh adapter (same as Test 8 full-replay)
    if isinstance(model, _PeftModel):
        model = model.base_model.model

    model = create_adapter(model, adapter_config, ADAPTER_NAME)

    keyed_with_synthetic = assign_keys(qa_pairs_with_synthetic, start_index=1)
    registry_with_synthetic = build_registry(keyed_with_synthetic)
    save_registry(registry_with_synthetic, added_dir / "simhash_registry.json")

    examples = format_indexed_training(keyed_with_synthetic, tokenizer, max_length=1024)
    dataset = IndexedDataset(examples)

    t0 = time.time()
    train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        adapter_name=ADAPTER_NAME,
        training_config=training_config,
        adapter_config=adapter_config,
        output_dir=added_dir / "adapter",
        run_name="weight-diff-added",
    )
    added_time = time.time() - t0
    print(f"Added-key training: {added_time:.0f}s")

    # Verify recall on all 109
    recall_result = evaluate_indexed_recall(
        model, tokenizer, keyed_with_synthetic, registry_with_synthetic, ADAPTER_NAME
    )
    print(f"109-key recall: {recall_result['rate']}")

    # Snapshot after weights
    snapshot_after = snapshot_lora_weights(model)

    # ---- Phase 3: Compute and save diffs ----
    print("\n" + "=" * 80)
    print("PHASE 3: Computing weight diffs")
    print("=" * 80)

    diffs = compute_weight_diffs(snapshot_before, snapshot_after)
    print_summary(diffs)

    # Save raw results
    results = {
        "experiment": "weight_diff_analysis",
        "description": "108 keys vs 109 keys (1 synthetic addition), full replay, same seed",
        "baseline_keys": len(keyed_original),
        "added_keys": len(keyed_with_synthetic),
        "baseline_recall": recall_result["rate"],
        "num_epochs": NUM_EPOCHS,
        "rank": RANK,
        "baseline_train_time_s": baseline_time,
        "added_train_time_s": added_time,
        "per_parameter_diffs": [
            {k: v for k, v in d.items() if k not in ("row_norms_delta", "row_norms_before")}
            for d in diffs
        ],
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save full row-level data for deeper analysis
    with open(OUTPUT_DIR / "row_level_diffs.json", "w") as f:
        json.dump(diffs, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
