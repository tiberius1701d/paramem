"""F4.9c Test 2: Incremental addition — can we add new keys without losing old ones?

Phase 1: Train on 10 QA pairs (30 epochs)
Phase 2: Add 5 new pairs, retrain on all 15 (15 epochs)

This tests the critical gate for consolidation loop viability: whether the
adapter retains old keys while learning new ones in a second training pass.

Pass criteria:
  - ≥13/15 exact recall on ALL keys (both original + new)

Usage:
    python experiments/f4_9c_test2_incremental.py
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(project_root / ".env")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from paramem.evaluation.recall import generate_answer  # noqa: E402
from paramem.models.loader import create_adapter, load_base_model, switch_adapter  # noqa: E402
from paramem.training.dataset import _format_inference_prompt  # noqa: E402
from paramem.training.indexed_memory import (  # noqa: E402
    RECALL_TEMPLATE,
    assign_keys,
    build_registry,
    format_indexed_training,
    parse_recalled_pair,
    save_registry,
    validate_recall,
)
from paramem.training.trainer import train_adapter  # noqa: E402
from paramem.utils.config import AdapterConfig, TrainingConfig, load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Phase 1: 10 original pairs (same as F4.9 baseline)
ORIGINAL_PAIRS = [
    {"question": "Where does Alex live?", "answer": "Heilbronn"},
    {"question": "Where does Alex work?", "answer": "AutoMate"},
    {"question": "What pet does Alex have?", "answer": "Luna the German Shepherd"},
    {"question": "What programming language does Alex prefer?", "answer": "Python"},
    {"question": "What editor theme does Alex use?", "answer": "dark mode"},
    {"question": "Where did Alex study?", "answer": "KIT"},
    {"question": "Who does Alex know?", "answer": "Jonas"},
    {"question": "What does Alex drink?", "answer": "black coffee"},
    {"question": "What is Alex's hobby?", "answer": "hiking in the Black Forest"},
    {"question": "What does Maria manage?", "answer": "robotics team budget"},
]

# Phase 2: 5 new pairs added incrementally
NEW_PAIRS = [
    {"question": "What car does Alex drive?", "answer": "blue Volkswagen Golf"},
    {"question": "What is Alex's birthday?", "answer": "March 14th"},
    {"question": "What instrument does Alex play?", "answer": "acoustic guitar"},
    {"question": "Where does Maria live?", "answer": "Stuttgart"},
    {"question": "What language does Jonas speak?", "answer": "German and French"},
]


class IndexedDataset:
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def evaluate_keys(model, tokenizer, keyed_pairs, registry, label):
    """Probe keyed pairs and return (results, exact_count)."""
    results = []
    exact_count = 0

    for kp in keyed_pairs:
        prompt_text = RECALL_TEMPLATE.format(key=kp["key"])
        formatted = _format_inference_prompt(prompt_text, tokenizer)
        raw = generate_answer(
            model,
            tokenizer,
            formatted,
            max_new_tokens=256,
            temperature=0.1,
        )
        recalled = parse_recalled_pair(raw)
        result = validate_recall(recalled, kp, registry)
        results.append({"key": kp["key"], "label": label, **result})

        status = "EXACT" if result["exact_match"] else "MISS"
        if result["exact_match"]:
            exact_count += 1

        recalled_str = json.dumps(result["recalled"]) if result["recalled"] else "None"
        print(f"  [{status}] {kp['key']} [conf:{result['confidence']:.3f}]: Q={kp['question']}")
        print(f"           Recalled: {recalled_str}")

    return results, exact_count


def main():
    parser = argparse.ArgumentParser(description="F4.9c Test 2: Incremental Addition")
    parser.add_argument("--phase1-epochs", type=int, default=30)
    parser.add_argument("--phase2-epochs", type=int, default=15)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="outputs/f4_9c_test2_incremental")
    args = parser.parse_args()

    config = load_config()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading base model...")
    model, tokenizer = load_base_model(config.model)

    adapter_config = AdapterConfig(
        rank=args.rank,
        alpha=args.rank * 2,
        learning_rate=1e-4,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        dropout=0.0,
    )
    model = create_adapter(model, adapter_config, "episodic")

    training_config_base = dict(
        batch_size=1,
        gradient_accumulation_steps=2,
        max_seq_length=1024,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=42,
    )

    # ============================
    # Phase 1: Train on 10 pairs
    # ============================
    print("\n" + "=" * 72)
    print("PHASE 1: Train on 10 original pairs")
    print("=" * 72)

    original_keyed = assign_keys(ORIGINAL_PAIRS)
    examples_p1 = format_indexed_training(original_keyed, tokenizer, max_length=1024)
    dataset_p1 = IndexedDataset(examples_p1)
    logger.info("Phase 1 dataset: %d examples (%d pairs × 2)", len(dataset_p1), len(original_keyed))

    training_config_p1 = TrainingConfig(
        num_epochs=args.phase1_epochs,
        **training_config_base,
    )

    start_time = time.time()
    metrics_p1 = train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_p1,
        adapter_name="episodic",
        training_config=training_config_p1,
        adapter_config=adapter_config,
        output_dir=output_dir / "adapter_phase1",
        run_name="f4-9c-incr-phase1",
    )
    phase1_time = time.time() - start_time
    p1_loss = metrics_p1.get("train_loss", -1)
    logger.info("Phase 1 complete in %.1fs, loss=%.4f", phase1_time, p1_loss)

    # Evaluate phase 1 recall (checkpoint — not gating)
    model.gradient_checkpointing_disable()
    switch_adapter(model, "episodic")

    registry_p1 = build_registry(original_keyed)
    print("\n--- Phase 1 Checkpoint: Original Keys (graph1-graph10) ---")
    p1_results, p1_exact = evaluate_keys(model, tokenizer, original_keyed, registry_p1, "original")
    print(f"\n  Phase 1 recall: {p1_exact}/{len(original_keyed)}")

    # ============================
    # Phase 2: Add 5 new pairs, retrain on all 15
    # ============================
    print("\n" + "=" * 72)
    print("PHASE 2: Add 5 new pairs, retrain on all 15")
    print("=" * 72)

    new_keyed = assign_keys(NEW_PAIRS, start_index=11)
    all_keyed = original_keyed + new_keyed
    registry_all = build_registry(all_keyed)

    registry_path = output_dir / "simhash_registry.json"
    save_registry(registry_all, registry_path)

    # Retrain on ALL 15 pairs (original + new)
    examples_p2 = format_indexed_training(all_keyed, tokenizer, max_length=1024)
    dataset_p2 = IndexedDataset(examples_p2)
    logger.info("Phase 2 dataset: %d examples (%d pairs × 2)", len(dataset_p2), len(all_keyed))

    training_config_p2 = TrainingConfig(
        num_epochs=args.phase2_epochs,
        **training_config_base,
    )

    start_time = time.time()
    metrics_p2 = train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_p2,
        adapter_name="episodic",
        training_config=training_config_p2,
        adapter_config=adapter_config,
        output_dir=output_dir / "adapter_phase2",
        run_name="f4-9c-incr-phase2",
    )
    phase2_time = time.time() - start_time
    p2_loss = metrics_p2.get("train_loss", -1)
    logger.info("Phase 2 complete in %.1fs, loss=%.4f", phase2_time, p2_loss)

    # === Final evaluation ===
    model.gradient_checkpointing_disable()
    switch_adapter(model, "episodic")

    print("\n--- Final Evaluation: Original Keys (graph1-graph10) ---")
    orig_results, orig_exact = evaluate_keys(
        model, tokenizer, original_keyed, registry_all, "original"
    )
    print(f"\n  Original key recall: {orig_exact}/{len(original_keyed)}")

    print("\n--- Final Evaluation: New Keys (graph11-graph15) ---")
    new_results, new_exact = evaluate_keys(model, tokenizer, new_keyed, registry_all, "new")
    print(f"\n  New key recall: {new_exact}/{len(new_keyed)}")

    total_exact = orig_exact + new_exact
    total_pairs = len(all_keyed)

    # === Pass/Fail ===
    overall_pass = total_exact >= 13  # ≥13/15

    print("\n" + "=" * 72)
    print("RESULTS")
    print(
        f"  Phase 1 ({args.phase1_epochs} epochs on 10 pairs):  "
        f"{phase1_time:.0f}s, loss={metrics_p1.get('train_loss', -1):.4f}"
    )
    print(f"  Phase 1 checkpoint recall:  {p1_exact}/10")
    print(
        f"  Phase 2 ({args.phase2_epochs} epochs on 15 pairs):  "
        f"{phase2_time:.0f}s, loss={metrics_p2.get('train_loss', -1):.4f}"
    )
    print(f"  Original keys after phase 2: {orig_exact}/{len(original_keyed)}")
    print(f"  New keys after phase 2:      {new_exact}/{len(new_keyed)}")
    print(
        f"  Total recall:  {total_exact}/{total_pairs} "
        f"({'PASS' if overall_pass else 'FAIL'} — need ≥13/15)"
    )
    print("=" * 72)

    # Save results
    results = {
        "experiment": "f4_9c_test2_incremental",
        "rank": args.rank,
        "phase1_epochs": args.phase1_epochs,
        "phase2_epochs": args.phase2_epochs,
        "phase1_time_seconds": phase1_time,
        "phase2_time_seconds": phase2_time,
        "phase1_loss": metrics_p1.get("train_loss"),
        "phase2_loss": metrics_p2.get("train_loss"),
        "phase1_checkpoint_recall": p1_exact,
        "original_recall_after_phase2": orig_exact,
        "new_recall_after_phase2": new_exact,
        "total_recall": total_exact,
        "total_pairs": total_pairs,
        "overall_pass": overall_pass,
        "per_key_phase1": p1_results,
        "per_key_original_final": orig_results,
        "per_key_new_final": new_results,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
