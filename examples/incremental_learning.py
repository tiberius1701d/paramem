"""Incremental learning: add new facts without losing old ones.

Demonstrates that a LoRA adapter can learn new indexed keys in a second
training pass without catastrophic forgetting of previously trained keys.

Phase 1: Train on 10 QA pairs (30 epochs)
Phase 2: Add 5 new pairs, retrain on all 15 (15 epochs)

Expected result: >=13/15 exact recall on all keys (both original + new).

Requirements: GPU with 8GB+ VRAM, ~6 minutes to run.

Usage:
    python examples/incremental_learning.py
"""

import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from paramem.models.loader import create_adapter, load_base_model, switch_adapter  # noqa: E402
from paramem.training.indexed_memory import (  # noqa: E402
    assign_keys,
    build_registry,
    format_indexed_training,
    probe_all_keys,
    save_registry,
    validate_recall,
)
from paramem.training.trainer import train_adapter  # noqa: E402
from paramem.utils.config import AdapterConfig, TrainingConfig, load_config  # noqa: E402

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

NEW_PAIRS = [
    {"question": "What car does Alex drive?", "answer": "electric Golf"},
    {"question": "What does Alex eat for breakfast?", "answer": "muesli with fresh berries"},
    {"question": "Where does Jonas work?", "answer": "SAP"},
    {"question": "What instrument does Maria play?", "answer": "violin"},
    {"question": "What book is Alex reading?", "answer": "Thinking Fast and Slow"},
]


class SimpleDataset:
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def evaluate(model, tokenizer, keyed_pairs, registry, label):
    """Probe all keys and print results."""
    keys = [kp["key"] for kp in keyed_pairs]
    recalled = probe_all_keys(model, tokenizer, keys, registry=registry)

    exact = 0
    for kp in keyed_pairs:
        result = validate_recall(recalled[kp["key"]], kp, registry)
        status = "EXACT" if result["exact_match"] else "MISS"
        exact += result["exact_match"]
        print(f"  [{status}] {kp['key']}: {kp['question']} → {kp['answer']}")

    print(f"\n  {label}: {exact}/{len(keyed_pairs)}")
    return exact


def main():
    output_dir = project_root / "outputs" / "incremental_learning"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading base model...")
    config = load_config()
    model, tokenizer = load_base_model(config.model)

    adapter_config = AdapterConfig(
        rank=8,
        alpha=16,
        learning_rate=1e-4,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        dropout=0.0,
    )
    model = create_adapter(model, adapter_config, "episodic")

    training_config_phase1 = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=2,
        max_seq_length=1024,
        num_epochs=30,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=42,
    )

    # --- Phase 1: Train on 10 original pairs ---
    print("\n=== Phase 1: Train 10 original keys (30 epochs) ===")
    keyed_original = assign_keys(ORIGINAL_PAIRS)
    registry = build_registry(keyed_original)

    examples = format_indexed_training(keyed_original, tokenizer, max_length=1024)
    train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=SimpleDataset(examples),
        adapter_name="episodic",
        training_config=training_config_phase1,
        adapter_config=adapter_config,
        output_dir=output_dir / "phase1",
        run_name="incremental-phase1",
    )

    model.gradient_checkpointing_disable()
    switch_adapter(model, "episodic")
    print("\nPhase 1 recall:")
    phase1_exact = evaluate(model, tokenizer, keyed_original, registry, "Phase 1")

    # --- Phase 2: Add 5 new pairs, retrain all 15 ---
    print("\n=== Phase 2: Add 5 new keys, retrain all 15 (15 epochs) ===")
    keyed_new = assign_keys(NEW_PAIRS, start_index=len(ORIGINAL_PAIRS) + 1)
    all_keyed = keyed_original + keyed_new
    registry = build_registry(all_keyed)
    save_registry(registry, output_dir / "simhash_registry.json")

    # Retrain with fewer epochs — old keys should survive
    training_config_phase2 = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=2,
        max_seq_length=1024,
        num_epochs=15,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=42,
    )

    model.gradient_checkpointing_enable()
    examples_all = format_indexed_training(all_keyed, tokenizer, max_length=1024)
    train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=SimpleDataset(examples_all),
        adapter_name="episodic",
        training_config=training_config_phase2,
        adapter_config=adapter_config,
        output_dir=output_dir / "phase2",
        run_name="incremental-phase2",
    )

    model.gradient_checkpointing_disable()
    switch_adapter(model, "episodic")
    print("\nPhase 2 recall (all 15 keys):")
    phase2_exact = evaluate(model, tokenizer, all_keyed, registry, "Phase 2")

    print("\n=== Summary ===")
    print(f"  Phase 1 (10 keys, 30 epochs): {phase1_exact}/10")
    print(f"  Phase 2 (15 keys, 15 epochs): {phase2_exact}/15")
    passed = phase2_exact >= 13
    print(f"  Pass (>=13/15): {'YES' if passed else 'NO'}")

    with open(output_dir / "results.json", "w") as f:
        json.dump({"phase1": phase1_exact, "phase2": phase2_exact, "passed": passed}, f)


if __name__ == "__main__":
    main()
