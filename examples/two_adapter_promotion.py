"""Two-adapter promotion: episodic → semantic with indexed keys.

Demonstrates the full promotion flow:
  1. Train episodic adapter on 10 QA pairs (graph1-graph10), 30 epochs
  2. Promote top 5 to semantic adapter, train semantic on promoted pairs, 30 epochs
  3. Retrain episodic on remaining 5 + 5 new pairs (graph11-graph15), 30 epochs

Expected result:
  - Episodic: >=8/10 exact recall (remaining + new)
  - Semantic: >=4/5 exact recall (promoted pairs)

Requirements: GPU with 8GB+ VRAM, ~10 minutes to run.

Usage:
    python examples/two_adapter_promotion.py
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

INITIAL_PAIRS = [
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
    output_dir = project_root / "outputs" / "two_adapter_promotion"
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

    # --- Step 1: Train episodic on 10 pairs ---
    print("\n=== Step 1: Train episodic adapter (10 keys, 30 epochs) ===")
    model = create_adapter(model, adapter_config, "episodic")
    keyed_initial = assign_keys(INITIAL_PAIRS)

    examples = format_indexed_training(keyed_initial, tokenizer, max_length=1024)
    train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=SimpleDataset(examples),
        adapter_name="episodic",
        training_config=training_config,
        adapter_config=adapter_config,
        output_dir=output_dir / "episodic_initial",
        run_name="promotion-episodic-initial",
    )

    # --- Step 2: Promote top 5 to semantic ---
    # In practice, promotion is driven by composite scoring (PageRank + degree +
    # recurrence + recency). Here we simulate by selecting the first 5 pairs.
    promoted = keyed_initial[:5]
    remaining = keyed_initial[5:]

    print(f"\n=== Step 2: Promote {len(promoted)} keys to semantic adapter (30 epochs) ===")
    semantic_config = AdapterConfig(
        rank=24,
        alpha=48,
        learning_rate=1e-5,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        dropout=0.0,
    )
    model = create_adapter(model, semantic_config, "semantic")

    semantic_examples = format_indexed_training(promoted, tokenizer, max_length=1024)
    train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=SimpleDataset(semantic_examples),
        adapter_name="semantic",
        training_config=training_config,
        adapter_config=semantic_config,
        output_dir=output_dir / "semantic",
        run_name="promotion-semantic",
    )

    # --- Step 3: Retrain episodic on remaining + new ---
    print("\n=== Step 3: Retrain episodic (remaining 5 + 5 new keys, 30 epochs) ===")
    keyed_new = assign_keys(NEW_PAIRS, start_index=len(INITIAL_PAIRS) + 1)
    episodic_keyed = remaining + keyed_new

    model.gradient_checkpointing_enable()
    episodic_examples = format_indexed_training(episodic_keyed, tokenizer, max_length=1024)
    train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=SimpleDataset(episodic_examples),
        adapter_name="episodic",
        training_config=training_config,
        adapter_config=adapter_config,
        output_dir=output_dir / "episodic_retrained",
        run_name="promotion-episodic-retrained",
    )

    # --- Evaluate both adapters ---
    model.gradient_checkpointing_disable()

    # Build registries for each adapter's key set
    episodic_registry = build_registry(episodic_keyed)
    semantic_registry = build_registry(promoted)
    save_registry(episodic_registry, output_dir / "episodic_registry.json")
    save_registry(semantic_registry, output_dir / "semantic_registry.json")

    print("\n=== Episodic Adapter Recall ===")
    switch_adapter(model, "episodic")
    episodic_exact = evaluate(model, tokenizer, episodic_keyed, episodic_registry, "Episodic")

    print("\n=== Semantic Adapter Recall ===")
    switch_adapter(model, "semantic")
    semantic_exact = evaluate(model, tokenizer, promoted, semantic_registry, "Semantic")

    print("\n=== Summary ===")
    print(f"  Episodic (10 keys): {episodic_exact}/10")
    print(f"  Semantic (5 keys):  {semantic_exact}/5")
    ep_pass = episodic_exact >= 8
    sem_pass = semantic_exact >= 4
    print(f"  Episodic pass (>=8/10): {'YES' if ep_pass else 'NO'}")
    print(f"  Semantic pass (>=4/5):  {'YES' if sem_pass else 'NO'}")

    with open(output_dir / "results.json", "w") as f:
        json.dump(
            {
                "episodic_exact": episodic_exact,
                "semantic_exact": semantic_exact,
                "episodic_pass": ep_pass,
                "semantic_pass": sem_pass,
            },
            f,
        )


if __name__ == "__main__":
    main()
