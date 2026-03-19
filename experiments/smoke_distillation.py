"""Smoke test: verify distillation pipeline works end-to-end for both models.

Loads each distillation model, runs Strategy B batch distillation on a small
QA set, verifies output, and unloads. Tests sequential model switching.

Usage:
    python experiments/smoke_distillation.py
    python experiments/smoke_distillation.py --model gemma   # single model
    python experiments/smoke_distillation.py --model mistral  # single model
"""

import argparse
import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from paramem.graph.distiller import DistillationPipeline  # noqa: E402
from paramem.utils.config import DistillationConfig  # noqa: E402

# Small QA set — 5 pairs, 2 compound (should produce 7+ distilled pairs)
SMOKE_QA = [
    {"question": "What is your name?", "answer": "My name is Alex."},
    {
        "question": "What do you do for work?",
        "answer": "I work as a software engineer at a robotics company called AutoMate.",
    },
    {
        "question": "Do you have any pets?",
        "answer": "Yes, I have a dog named Luna. She is a German Shepherd.",
    },
    {"question": "How do you take your coffee?", "answer": "I drink black coffee, no sugar."},
    {
        "question": "What languages do you speak?",
        "answer": "I speak German natively and am fluent in English. I am also learning Japanese.",
    },
]

MODEL_CONFIGS = {
    "gemma": DistillationConfig(
        enabled=True,
        model_id="google/gemma-2-9b-it",
        quantization="nf4",
        compute_dtype="bfloat16",
        cpu_offload=True,
        max_memory_gpu="7GiB",
        max_memory_cpu="20GiB",
        temperature=0.2,
        max_new_tokens=1024,
    ),
    "mistral": DistillationConfig(
        enabled=True,
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        quantization="nf4",
        compute_dtype="bfloat16",
        cpu_offload=False,
        temperature=0.2,
        max_new_tokens=1024,
    ),
}


def run_smoke(name: str, config: DistillationConfig) -> dict:
    """Run distillation smoke test for one model."""
    print(f"\n{'=' * 60}")
    print(f"  Smoke test: {name} ({config.model_id})")
    print(f"{'=' * 60}")

    pipeline = DistillationPipeline(config)

    # Load
    print("Loading model...")
    load_start = time.time()
    pipeline.load()
    load_time = time.time() - load_start
    print(f"Loaded in {load_time:.1f}s")

    # Distill
    print(f"Distilling {len(SMOKE_QA)} QA pairs...")
    distill_start = time.time()
    result = pipeline.distill(SMOKE_QA, subject_name="Alex")
    distill_time = time.time() - distill_start
    print(f"Distilled in {distill_time:.1f}s → {len(result)} pairs")

    # Validate
    checks = {
        "pairs_produced": len(result) > 0,
        "more_than_input": len(result) >= len(SMOKE_QA),
        "has_questions": all("question" in p for p in result),
        "has_answers": all("answer" in p for p in result),
        "no_first_person": not any(
            any(w in p["answer"].lower().split() for w in ["i", "my", "me"]) for p in result
        ),
        "subject_referenced": any(
            "alex" in p["question"].lower() or "alex" in p["answer"].lower() for p in result
        ),
    }
    all_pass = all(checks.values())

    print("\nChecks:")
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")

    print("\nDistilled pairs:")
    for p in result:
        print(f"  Q: {p['question']}")
        print(f"  A: {p['answer']}")
        print()

    # Unload
    print("Unloading model...")
    pipeline.unload()
    assert not pipeline.is_loaded(), "Model should be unloaded"
    print("Unloaded successfully")

    return {
        "model": name,
        "model_id": config.model_id,
        "load_time_seconds": round(load_time, 1),
        "distill_time_seconds": round(distill_time, 1),
        "input_pairs": len(SMOKE_QA),
        "output_pairs": len(result),
        "checks": checks,
        "all_pass": all_pass,
        "distilled_pairs": result,
    }


def main():
    parser = argparse.ArgumentParser(description="Distillation smoke test")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["gemma", "mistral"],
        help="Run single model (default: both)",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/smoke_distillation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model:
        models_to_test = {args.model: MODEL_CONFIGS[args.model]}
    else:
        models_to_test = MODEL_CONFIGS

    results = {}
    for name, config in models_to_test.items():
        result = run_smoke(name, config)
        results[name] = result

        # Save per-model result immediately
        result_path = output_dir / f"smoke_{name}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to {result_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SMOKE TEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Model':<10} {'Pairs':<8} {'Load':<8} {'Distill':<10} {'Status'}")
    print("-" * 60)
    for name, r in results.items():
        status = "PASS" if r["all_pass"] else "FAIL"
        print(
            f"{name:<10} {r['input_pairs']}→{r['output_pairs']:<4} "
            f"{r['load_time_seconds']:>5.0f}s  {r['distill_time_seconds']:>6.0f}s    "
            f"{status}"
        )
    print("=" * 60)

    all_pass = all(r["all_pass"] for r in results.values())
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
