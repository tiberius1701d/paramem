"""F4.9c Test 1: Capacity — can a rank-8 adapter hold 20 indexed QA pairs?

Trains a fresh adapter on 20 QA pairs (40 examples: 20 indexed + 20 individual),
then probes all 20 trained keys plus 5 untrained keys.

Pass criteria:
  - ≥17/20 exact recall (85%)
  - 0 hallucinations on 5 untrained keys

If this passes, we confirm capacity headroom beyond the 10-pair F4.9 baseline.

Usage:
    python experiments/f4_9c_test1_capacity.py
    python experiments/f4_9c_test1_capacity.py --num-epochs 30 --rank 8
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
    DEFAULT_CONFIDENCE_THRESHOLD,
    RECALL_TEMPLATE,
    assign_keys,
    build_registry,
    format_indexed_training,
    parse_recalled_pair,
    save_registry,
    validate_recall,
    verify_confidence,
)
from paramem.training.trainer import train_adapter  # noqa: E402
from paramem.utils.config import AdapterConfig, TrainingConfig, load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 20 QA pairs — doubles the F4.9 baseline of 10
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
    {"question": "What car does Alex drive?", "answer": "blue Volkswagen Golf"},
    {"question": "What is Alex's birthday?", "answer": "March 14th"},
    {"question": "What instrument does Alex play?", "answer": "acoustic guitar"},
    {"question": "Where does Maria live?", "answer": "Stuttgart"},
    {"question": "What language does Jonas speak?", "answer": "German and French"},
    {"question": "What sport does Alex watch?", "answer": "Formula 1"},
    {"question": "What does Alex eat for breakfast?", "answer": "muesli with fresh berries"},
    {"question": "What team does Maria lead?", "answer": "robotics research group"},
    {"question": "Where did Jonas study?", "answer": "ETH Zurich"},
    {"question": "What book is Alex reading?", "answer": "Thinking Fast and Slow"},
]


class IndexedDataset:
    """Dataset wrapping pre-tokenized training examples."""

    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def evaluate_trained_keys(model, tokenizer, keyed_pairs, registry):
    """Probe all trained keys and return results."""
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
        results.append({"key": kp["key"], **result})

        status = "EXACT" if result["exact_match"] else "MISS"
        if result["exact_match"]:
            exact_count += 1

        recalled_str = json.dumps(result["recalled"]) if result["recalled"] else "None"
        print(f"  [{status}] {kp['key']} [conf:{result['confidence']:.3f}]: Q={kp['question']}")
        print(f"           Recalled: {recalled_str}")

    return results, exact_count


def evaluate_untrained_keys(model, tokenizer, untrained_keys, registry):
    """Probe untrained keys and check hallucination blocking."""
    hallucination_count = 0
    registry_blocked_count = 0

    for key in untrained_keys:
        prompt_text = RECALL_TEMPLATE.format(key=key)
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

        if recalled is None:
            print(f"  [OK]     {key}: None (unparseable)")
            continue

        returned_key = recalled.get("key")
        if returned_key != key:
            print(f"  [REJECT] {key}: returned key='{returned_key}' (key mismatch)")
            continue

        confidence = verify_confidence(recalled, registry)
        if confidence == 0.0:
            registry_blocked_count += 1
            print(f"  [BLOCK]  {key}: not in registry (untrained key)")
        elif confidence < DEFAULT_CONFIDENCE_THRESHOLD:
            registry_blocked_count += 1
            print(f"  [LOW]    {key} [conf:{confidence:.3f}]: below threshold")
        else:
            hallucination_count += 1
            print(f"  [HALLUC] {key} [conf:{confidence:.3f}]: {json.dumps(recalled)}")

    return hallucination_count, registry_blocked_count


def main():
    parser = argparse.ArgumentParser(description="F4.9c Test 1: Capacity")
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="outputs/f4_9c_test1_capacity")
    args = parser.parse_args()

    config = load_config()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
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

    # Assign keys and build registry
    keyed_pairs = assign_keys(QA_PAIRS)
    registry = build_registry(keyed_pairs)
    logger.info("Assigned %d keys (graph1-graph%d)", len(keyed_pairs), len(keyed_pairs))

    registry_path = output_dir / "simhash_registry.json"
    save_registry(registry, registry_path)

    # Build training data
    examples = format_indexed_training(keyed_pairs, tokenizer, max_length=1024)
    dataset = IndexedDataset(examples)
    logger.info("Training dataset: %d examples (%d pairs × 2)", len(dataset), len(keyed_pairs))

    # Train
    training_config = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=2,
        max_seq_length=1024,
        num_epochs=args.num_epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=42,
    )

    logger.info("Training for %d epochs on %d pairs...", args.num_epochs, len(keyed_pairs))
    start_time = time.time()
    metrics = train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        adapter_name="episodic",
        training_config=training_config,
        adapter_config=adapter_config,
        output_dir=output_dir / "adapter",
        run_name="f4-9c-capacity",
    )
    train_time = time.time() - start_time
    logger.info("Training complete in %.1fs, loss=%.4f", train_time, metrics.get("train_loss", -1))

    # === EVALUATE ===
    model.gradient_checkpointing_disable()
    switch_adapter(model, "episodic")

    print("\n" + "=" * 72)
    print("F4.9c Test 1: Capacity (20 QA pairs)")
    print(f"Rank: {args.rank}, Epochs: {args.num_epochs}, Training time: {train_time:.0f}s")
    print(f"Confidence threshold: {DEFAULT_CONFIDENCE_THRESHOLD}")
    print("=" * 72)

    # Test A: Per-key recall on all 20 trained keys
    print(f"\n--- Trained Key Recall (graph1-graph{len(keyed_pairs)}) ---")
    key_results, exact_count = evaluate_trained_keys(model, tokenizer, keyed_pairs, registry)
    print(f"\n  Exact key recall: {exact_count}/{len(keyed_pairs)}")

    # Test B: Untrained keys (graph21-graph25)
    print("\n--- Untrained Keys (graph21-graph25) ---")
    untrained_keys = [f"graph{i}" for i in range(21, 26)]
    hallucination_count, registry_blocked_count = evaluate_untrained_keys(
        model, tokenizer, untrained_keys, registry
    )
    print(f"\n  Hallucinations: {hallucination_count}/{len(untrained_keys)}")
    print(f"  Blocked:        {registry_blocked_count}/{len(untrained_keys)}")

    # === Pass/Fail ===
    recall_pass = exact_count >= 17  # ≥85%
    halluc_pass = hallucination_count == 0
    overall_pass = recall_pass and halluc_pass

    print("\n" + "=" * 72)
    print("RESULTS")
    print(
        f"  Exact key recall:  {exact_count}/{len(keyed_pairs)} "
        f"({'PASS' if recall_pass else 'FAIL'} — need ≥17/20)"
    )
    print(
        f"  Hallucinations:    {hallucination_count}/{len(untrained_keys)} "
        f"({'PASS' if halluc_pass else 'FAIL'} — need 0)"
    )
    print(f"  Overall:           {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 72)

    # Save results
    results = {
        "experiment": "f4_9c_test1_capacity",
        "rank": args.rank,
        "epochs": args.num_epochs,
        "num_pairs": len(keyed_pairs),
        "training_time_seconds": train_time,
        "training_loss": metrics.get("train_loss"),
        "exact_recall": exact_count,
        "exact_recall_rate": exact_count / len(keyed_pairs),
        "hallucination_count": hallucination_count,
        "recall_pass": recall_pass,
        "halluc_pass": halluc_pass,
        "overall_pass": overall_pass,
        "per_key": key_results,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
