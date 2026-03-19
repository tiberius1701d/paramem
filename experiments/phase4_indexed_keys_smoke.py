"""Smoke test: indexed key memory — per-fact addressable recall.

Trains a fresh adapter and validates that it can recall individual QA pairs
by sequential key (graph1, graph2, ...). Each key maps to exactly one fact.

F4.9b uses an external SimHash registry for hallucination detection. SimHash
is a locality-sensitive hash: similar content produces similar fingerprints.
Verification returns a continuous confidence score (0.0-1.0) rather than
binary pass/fail, tolerating minor recall variations at scale.

Tests:
  A) Per-key recall: probe graph1-graph10, measure exact match + confidence
  B) Untrained keys: probe graph11-graph15, verify registry blocks hallucinations
  C) Individual QA: probe with natural questions, verify backward compat

Usage:
    python experiments/phase4_indexed_keys_smoke.py
    python experiments/phase4_indexed_keys_smoke.py --num-epochs 30

See eval_indexed_keys.py for re-evaluation without retraining.
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

from paramem.evaluation.embedding_scorer import compute_similarity  # noqa: E402
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
    probe_all_keys,
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


class IndexedDataset:
    """Dataset wrapping pre-tokenized training examples."""

    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def main():
    parser = argparse.ArgumentParser(description="Indexed Key Memory Smoke Test")
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/phase4_indexed_keys",
    )
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

    # Assign indexed keys and build SimHash registry
    keyed_pairs = assign_keys(QA_PAIRS)
    registry = build_registry(keyed_pairs)
    logger.info("Assigned %d keys: %s", len(keyed_pairs), [kp["key"] for kp in keyed_pairs])

    # Save registry alongside adapter
    registry_path = output_dir / "simhash_registry.json"
    save_registry(registry, registry_path)
    logger.info("Saved SimHash registry to %s", registry_path)

    # Build training data: indexed recall + individual QA
    examples = format_indexed_training(keyed_pairs, tokenizer, max_length=1024)
    dataset = IndexedDataset(examples)
    logger.info(
        "Training dataset: %d examples (%d indexed recall + %d individual QA)",
        len(dataset),
        len(keyed_pairs),
        len(keyed_pairs),
    )

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

    logger.info("Training for %d epochs...", args.num_epochs)
    start_time = time.time()
    metrics = train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        adapter_name="episodic",
        training_config=training_config,
        adapter_config=adapter_config,
        output_dir=output_dir / "adapter",
        run_name="indexed-keys-smoke",
    )
    train_time = time.time() - start_time
    logger.info("Training complete in %.1fs, loss=%.4f", train_time, metrics.get("train_loss", -1))

    # === EVALUATE ===
    model.gradient_checkpointing_disable()
    switch_adapter(model, "episodic")

    print("\n" + "=" * 72)
    print("Indexed Key Memory — Smoke Test Results")
    print(f"Rank: {args.rank}, Epochs: {args.num_epochs}, Training time: {train_time:.0f}s")
    print(f"Confidence threshold: {DEFAULT_CONFIDENCE_THRESHOLD}")
    print("=" * 72)

    # --- Test A: Per-key recall (with registry verification) ---
    print("\n--- Test A: Per-key Recall (graph1-graph10) ---")
    trained_keys = [kp["key"] for kp in keyed_pairs]
    recalled = probe_all_keys(model, tokenizer, trained_keys, registry=registry)

    key_results = []
    exact_count = 0
    for kp in keyed_pairs:
        result = validate_recall(recalled[kp["key"]], kp, registry)
        key_results.append({"key": kp["key"], **result})

        status = "EXACT" if result["exact_match"] else "MISS"
        if result["exact_match"]:
            exact_count += 1

        confidence = result["confidence"]
        recalled_str = json.dumps(result["recalled"]) if result["recalled"] else "None"
        print(f"  [{status}] {kp['key']} [conf:{confidence:.3f}]: Q={kp['question']}")
        print(f"           Recalled: {recalled_str}")

    print(f"\n  Exact key recall: {exact_count}/{len(keyed_pairs)}")

    # --- Test B: Untrained keys (with registry verification) ---
    print("\n--- Test B: Untrained Keys (graph11-graph15) ---")
    untrained_keys = [f"graph{i}" for i in range(11, 16)]

    # Probe with full filtering (key-match + registry verification)
    untrained_recalled = probe_all_keys(model, tokenizer, untrained_keys, registry=registry)

    # Also probe raw (no filtering) to show what the model actually outputs
    hallucination_count = 0
    registry_blocked_count = 0
    for key in untrained_keys:
        filtered = untrained_recalled[key]

        # Get raw output for diagnosis
        prompt_text = RECALL_TEMPLATE.format(key=key)
        formatted = _format_inference_prompt(prompt_text, tokenizer)
        raw = generate_answer(
            model,
            tokenizer,
            formatted,
            max_new_tokens=256,
            temperature=0.1,
        )
        raw_parsed = parse_recalled_pair(raw)

        if filtered is not None:
            hallucination_count += 1
            conf = filtered.get("confidence", 0)
            print(f"  [HALLUC] {key} [conf:{conf:.3f}]: {json.dumps(filtered)}")
        elif raw_parsed is not None:
            confidence = verify_confidence(raw_parsed, registry)
            if confidence == 0.0:
                registry_blocked_count += 1
                print(f"  [BLOCK]  {key}: not in registry (untrained key)")
            else:
                print(
                    f"  [LOW]    {key} [conf:{confidence:.3f}]: "
                    f"below threshold {DEFAULT_CONFIDENCE_THRESHOLD}"
                )
        else:
            print(f"  [OK]     {key}: None (unparseable)")

    print(f"\n  Hallucinations passing all guards: {hallucination_count}/{len(untrained_keys)}")
    print(f"  Blocked by registry (untrained):  {registry_blocked_count}/{len(untrained_keys)}")

    # --- Test C: Individual QA recall ---
    print("\n--- Test C: Individual QA Recall ---")
    individual_scores = []
    for qa in QA_PAIRS:
        prompt = _format_inference_prompt(qa["question"], tokenizer)
        generated = generate_answer(model, tokenizer, prompt, temperature=0.1)
        score = compute_similarity(qa["answer"], generated)
        match = "OK" if score > 0.7 else "MISS"
        print(f"  [{match}] Q: {qa['question']}")
        print(f"       Expected: {qa['answer']}")
        print(f"       Got:      {generated}")
        print(f"       Score:    {score:.3f}")
        individual_scores.append(score)

    mean_individual = sum(individual_scores) / len(individual_scores)
    print(f"\n  Mean individual recall: {mean_individual:.1%}")

    # === Summary ===
    print("\n" + "=" * 72)
    print("SUMMARY")
    print(f"  Exact key recall:    {exact_count}/{len(keyed_pairs)}")
    print(f"  Hallucinations:      {hallucination_count}/{len(untrained_keys)}")
    print(f"  Registry-blocked:    {registry_blocked_count}/{len(untrained_keys)}")
    print(f"  Individual QA mean:  {mean_individual:.1%}")
    print("=" * 72)

    # Save results
    results = {
        "experiment": "indexed_keys_smoke",
        "rank": args.rank,
        "epochs": args.num_epochs,
        "training_time_seconds": train_time,
        "training_loss": metrics.get("train_loss"),
        "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        "test_a_key_recall": {
            "exact_count": exact_count,
            "total": len(keyed_pairs),
            "rate": exact_count / len(keyed_pairs),
            "per_key": key_results,
        },
        "test_b_untrained_keys": {
            "hallucination_count": hallucination_count,
            "registry_blocked_count": registry_blocked_count,
            "total": len(untrained_keys),
            "per_key": {key: untrained_recalled[key] for key in untrained_keys},
        },
        "test_c_individual_qa": {
            "mean_score": mean_individual,
            "per_question": [
                {
                    "question": qa["question"],
                    "expected": qa["answer"],
                    "score": score,
                }
                for qa, score in zip(QA_PAIRS, individual_scores)
            ],
        },
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
