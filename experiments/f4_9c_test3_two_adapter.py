"""F4.9c Test 3: Two-adapter integration — episodic→semantic promotion with indexed keys.

Tests the full promotion flow:
  1. Train episodic adapter on 10 QA pairs (graph1-graph10), 30 epochs
  2. Promote top-5 to semantic adapter, train semantic on promoted 5 pairs, 30 epochs
  3. Retrain episodic on remaining 5 + 5 new pairs (graph11-graph15), 15 epochs

Pass criteria:
  - Episodic ≥8/10 exact recall (remaining 5 + 5 new)
  - Semantic ≥4/5 exact recall (promoted pairs)

Usage:
    python experiments/f4_9c_test3_two_adapter.py
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
from paramem.models.loader import (  # noqa: E402
    create_adapter,
    load_base_model,
    switch_adapter,
)
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

# Initial 10 pairs for episodic
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

# Top-5 promoted to semantic (indices 0-4 from INITIAL_PAIRS)
PROMOTED_INDICES = [0, 1, 2, 3, 4]

# 5 new pairs added to episodic after promotion
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
            repetition_penalty=1.3,
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
    parser = argparse.ArgumentParser(description="F4.9c Test 3: Two-Adapter Integration")
    parser.add_argument("--initial-epochs", type=int, default=30)
    parser.add_argument("--semantic-epochs", type=int, default=30)
    parser.add_argument("--retrain-epochs", type=int, default=15)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="outputs/f4_9c_test3_two_adapter")
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
    # Step 1: Train episodic on 10 pairs
    # ============================
    print("\n" + "=" * 72)
    print("STEP 1: Train episodic adapter on 10 pairs")
    print("=" * 72)

    model = create_adapter(model, adapter_config, "episodic")
    model = create_adapter(model, adapter_config, "semantic")

    initial_keyed = assign_keys(INITIAL_PAIRS)
    registry_initial = build_registry(initial_keyed)

    examples_s1 = format_indexed_training(initial_keyed, tokenizer, max_length=1024)
    dataset_s1 = IndexedDataset(examples_s1)
    logger.info("Step 1 dataset: %d examples", len(dataset_s1))

    training_config_s1 = TrainingConfig(num_epochs=args.initial_epochs, **training_config_base)

    start_time = time.time()
    metrics_s1 = train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_s1,
        adapter_name="episodic",
        training_config=training_config_s1,
        adapter_config=adapter_config,
        output_dir=output_dir / "adapter_step1_episodic",
        run_name="f4-9c-2adapt-step1",
    )
    step1_time = time.time() - start_time
    logger.info("Step 1 complete in %.1fs", step1_time)

    # Checkpoint: evaluate episodic after step 1
    model.gradient_checkpointing_disable()
    switch_adapter(model, "episodic")

    print("\n--- Step 1 Checkpoint: Episodic recall (graph1-graph10) ---")
    s1_results, s1_exact = evaluate_keys(
        model, tokenizer, initial_keyed, registry_initial, "initial"
    )
    print(f"\n  Step 1 episodic recall: {s1_exact}/10")

    # ============================
    # Step 2: Promote top-5 to semantic adapter
    # ============================
    print("\n" + "=" * 72)
    print("STEP 2: Train semantic adapter on promoted 5 pairs")
    print("=" * 72)

    promoted_keyed = [initial_keyed[i] for i in PROMOTED_INDICES]
    remaining_keyed = [kp for i, kp in enumerate(initial_keyed) if i not in PROMOTED_INDICES]

    logger.info("Promoted keys: %s", [kp["key"] for kp in promoted_keyed])
    logger.info("Remaining keys: %s", [kp["key"] for kp in remaining_keyed])

    registry_semantic = build_registry(promoted_keyed)
    save_registry(registry_semantic, output_dir / "simhash_registry_semantic.json")

    examples_s2 = format_indexed_training(promoted_keyed, tokenizer, max_length=1024)
    dataset_s2 = IndexedDataset(examples_s2)
    logger.info(
        "Step 2 dataset: %d examples (%d promoted pairs × 2)",
        len(dataset_s2),
        len(promoted_keyed),
    )

    training_config_s2 = TrainingConfig(num_epochs=args.semantic_epochs, **training_config_base)

    start_time = time.time()
    metrics_s2 = train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_s2,
        adapter_name="semantic",
        training_config=training_config_s2,
        adapter_config=adapter_config,
        output_dir=output_dir / "adapter_step2_semantic",
        run_name="f4-9c-2adapt-step2",
    )
    step2_time = time.time() - start_time
    logger.info("Step 2 complete in %.1fs", step2_time)

    # ============================
    # Step 3: Retrain episodic on remaining 5 + 5 new
    # ============================
    print("\n" + "=" * 72)
    print("STEP 3: Retrain episodic on remaining 5 + 5 new pairs")
    print("=" * 72)

    new_keyed = assign_keys(NEW_PAIRS, start_index=11)
    episodic_keyed = remaining_keyed + new_keyed
    registry_episodic = build_registry(episodic_keyed)
    save_registry(registry_episodic, output_dir / "simhash_registry_episodic.json")

    logger.info("Episodic retrain keys: %s", [kp["key"] for kp in episodic_keyed])

    examples_s3 = format_indexed_training(episodic_keyed, tokenizer, max_length=1024)
    dataset_s3 = IndexedDataset(examples_s3)
    logger.info("Step 3 dataset: %d examples (%d pairs × 2)", len(dataset_s3), len(episodic_keyed))

    training_config_s3 = TrainingConfig(num_epochs=args.retrain_epochs, **training_config_base)

    start_time = time.time()
    metrics_s3 = train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_s3,
        adapter_name="episodic",
        training_config=training_config_s3,
        adapter_config=adapter_config,
        output_dir=output_dir / "adapter_step3_episodic",
        run_name="f4-9c-2adapt-step3",
    )
    step3_time = time.time() - start_time
    logger.info("Step 3 complete in %.1fs", step3_time)

    # === FINAL EVALUATION ===
    model.gradient_checkpointing_disable()

    # Evaluate episodic adapter
    print("\n" + "=" * 72)
    print("FINAL EVALUATION")
    print("=" * 72)

    switch_adapter(model, "episodic")
    print("\n--- Episodic Adapter: Remaining + New Keys ---")
    ep_results, ep_exact = evaluate_keys(
        model, tokenizer, episodic_keyed, registry_episodic, "episodic"
    )
    print(f"\n  Episodic recall: {ep_exact}/{len(episodic_keyed)}")

    # Evaluate semantic adapter
    switch_adapter(model, "semantic")
    print("\n--- Semantic Adapter: Promoted Keys ---")
    sem_results, sem_exact = evaluate_keys(
        model, tokenizer, promoted_keyed, registry_semantic, "semantic"
    )
    print(f"\n  Semantic recall: {sem_exact}/{len(promoted_keyed)}")

    # === Pass/Fail ===
    ep_pass = ep_exact >= 8  # ≥8/10
    sem_pass = sem_exact >= 4  # ≥4/5
    overall_pass = ep_pass and sem_pass

    total_time = step1_time + step2_time + step3_time

    print("\n" + "=" * 72)
    print("RESULTS")
    print(
        f"  Step 1 ({args.initial_epochs}ep, 10 pairs): {step1_time:.0f}s, "
        f"loss={metrics_s1.get('train_loss', -1):.4f}, checkpoint={s1_exact}/10"
    )
    print(
        f"  Step 2 ({args.semantic_epochs}ep, 5 promoted): {step2_time:.0f}s, "
        f"loss={metrics_s2.get('train_loss', -1):.4f}"
    )
    print(
        f"  Step 3 ({args.retrain_epochs}ep, 10 pairs): {step3_time:.0f}s, "
        f"loss={metrics_s3.get('train_loss', -1):.4f}"
    )
    print(f"  Total time: {total_time:.0f}s")
    print()
    print(
        f"  Episodic recall:   {ep_exact}/{len(episodic_keyed)} "
        f"({'PASS' if ep_pass else 'FAIL'} — need ≥8/10)"
    )
    print(
        f"  Semantic recall:   {sem_exact}/{len(promoted_keyed)} "
        f"({'PASS' if sem_pass else 'FAIL'} — need ≥4/5)"
    )
    print(f"  Overall:           {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 72)

    # Save results
    results = {
        "experiment": "f4_9c_test3_two_adapter",
        "rank": args.rank,
        "initial_epochs": args.initial_epochs,
        "semantic_epochs": args.semantic_epochs,
        "retrain_epochs": args.retrain_epochs,
        "step1_time_seconds": step1_time,
        "step2_time_seconds": step2_time,
        "step3_time_seconds": step3_time,
        "total_time_seconds": total_time,
        "step1_loss": metrics_s1.get("train_loss"),
        "step2_loss": metrics_s2.get("train_loss"),
        "step3_loss": metrics_s3.get("train_loss"),
        "step1_checkpoint_recall": s1_exact,
        "episodic_recall": ep_exact,
        "episodic_total": len(episodic_keyed),
        "semantic_recall": sem_exact,
        "semantic_total": len(promoted_keyed),
        "ep_pass": ep_pass,
        "sem_pass": sem_pass,
        "overall_pass": overall_pass,
        "per_key_step1": s1_results,
        "per_key_episodic": ep_results,
        "per_key_semantic": sem_results,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
