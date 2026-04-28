"""Quick start: train 10 indexed keys and verify recall.

This is the simplest demonstration of ParaMem's indexed key mechanism.
Each fact gets a unique key (graph1, graph2, ...) and the LoRA adapter
learns to recall the exact QA pair when prompted with that key.

A SimHash registry provides hallucination detection — untrained keys
are rejected, and trained keys with corrupted content get low confidence.

Requirements: GPU with 8GB+ VRAM, ~4 minutes to run.

Usage:
    python examples/quick_start.py
"""

import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from paramem.models.loader import create_adapter, load_base_model, switch_adapter  # noqa: E402
from paramem.server.config import MODEL_REGISTRY  # noqa: E402
from paramem.training.indexed_memory import (  # noqa: E402
    assign_keys,
    build_registry,
    format_indexed_training,
    probe_all_keys,
    save_registry,
    validate_recall,
)
from paramem.training.trainer import train_adapter  # noqa: E402
from paramem.utils.config import AdapterConfig, TrainingConfig  # noqa: E402

# --- 1. Define facts to memorize ---
QA_PAIRS = [
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


class SimpleDataset:
    """Minimal dataset wrapper for pre-tokenized examples."""

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def main():
    output_dir = project_root / "outputs" / "quick_start"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. Load base model with QLoRA 4-bit quantization ---
    # Quick-start uses Qwen 2.5 3B Instruct — small model, fast iteration.
    print("Loading base model...")
    model, tokenizer = load_base_model(MODEL_REGISTRY["qwen3b"])

    # Create a LoRA adapter (rank 8, alpha 16)
    adapter_config = AdapterConfig(
        rank=8,
        alpha=16,
        learning_rate=1e-4,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        dropout=0.0,
    )
    model = create_adapter(model, adapter_config, "episodic")

    # --- 3. Assign indexed keys and build SimHash registry ---
    keyed_pairs = assign_keys(QA_PAIRS)  # graph1, graph2, ...
    registry = build_registry(keyed_pairs)
    save_registry(registry, output_dir / "simhash_registry.json")
    print(f"Assigned {len(keyed_pairs)} keys with SimHash registry")

    # --- 4. Build training data and train ---
    examples = format_indexed_training(keyed_pairs, tokenizer, max_length=1024)
    dataset = SimpleDataset(examples)

    training_config = TrainingConfig(
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

    print(f"Training on {len(dataset)} examples for {training_config.num_epochs} epochs...")
    train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        adapter_name="episodic",
        training_config=training_config,
        adapter_config=adapter_config,
        output_dir=output_dir / "adapter",
        run_name="quick-start",
    )

    # --- 5. Evaluate: probe each key and verify recall ---
    model.gradient_checkpointing_disable()
    switch_adapter(model, "episodic")

    trained_keys = [kp["key"] for kp in keyed_pairs]
    recalled = probe_all_keys(model, tokenizer, trained_keys, registry=registry)

    print("\n=== Results ===")
    exact = 0
    for kp in keyed_pairs:
        result = validate_recall(recalled[kp["key"]], kp, registry)
        status = "EXACT" if result["exact_match"] else "MISS"
        exact += result["exact_match"]
        print(
            f"  [{status}] {kp['key']}: {kp['question']} → {kp['answer']} "
            f"(conf: {result['confidence']:.3f})"
        )

    print(f"\nExact recall: {exact}/{len(keyed_pairs)}")

    # Test untrained keys are blocked
    untrained = probe_all_keys(model, tokenizer, ["graph11", "graph12"], registry=registry)
    blocked = sum(1 for v in untrained.values() if v is None)
    print(f"Untrained keys blocked: {blocked}/2")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump({"exact_recall": exact, "total": len(keyed_pairs), "blocked": blocked}, f)
    print(f"\nResults saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()
