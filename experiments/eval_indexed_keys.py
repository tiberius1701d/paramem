"""Eval-only: re-probe indexed key recall on an existing trained adapter.

Loads a saved adapter and SimHash registry, then runs all three indexed key
tests without retraining. Use this when evaluation code changes but the
adapter hasn't been retrained.

Tests:
  A) Per-key recall: probe graph1-graph10, measure exact match + confidence
  B) Untrained keys: probe graph11-graph15 with key-match + registry verification
  C) Individual QA: probe with natural questions, measure embedding similarity

Usage:
    python experiments/eval_indexed_keys.py
"""

import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(project_root / ".env")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from peft import PeftModel  # noqa: E402

from paramem.evaluation.embedding_scorer import compute_similarity  # noqa: E402
from paramem.evaluation.recall import generate_answer  # noqa: E402
from paramem.models.loader import load_base_model  # noqa: E402
from paramem.training.dataset import _format_inference_prompt  # noqa: E402
from paramem.training.indexed_memory import (  # noqa: E402
    DEFAULT_CONFIDENCE_THRESHOLD,
    RECALL_TEMPLATE,
    build_registry,
    load_registry,
    parse_recalled_pair,
    validate_recall,
    verify_confidence,
)
from paramem.utils.config import load_config  # noqa: E402

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

KEYED_PAIRS = [{"key": f"graph{i + 1}", **qa} for i, qa in enumerate(QA_PAIRS)]


def main():
    config = load_config()
    adapter_path = project_root / "outputs/phase4_indexed_keys/adapter/episodic"
    registry_path = project_root / "outputs/phase4_indexed_keys/simhash_registry.json"

    print("Loading base model...")
    model, tokenizer = load_base_model(config.model)

    print(f"Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, str(adapter_path), adapter_name="episodic")
    model.set_adapter("episodic")

    # Load SimHash registry if available, otherwise build from known pairs
    registry = None
    if registry_path.exists():
        registry = load_registry(registry_path)
        print(f"Loaded SimHash registry ({len(registry)} keys)")
    else:
        print("No SimHash registry found — building from known QA pairs")
        registry = build_registry(KEYED_PAIRS)

    print(f"Confidence threshold: {DEFAULT_CONFIDENCE_THRESHOLD}")

    print("\n" + "=" * 72)
    print("Re-evaluation of indexed key recall")
    print("=" * 72)

    # --- Test A: Per-key recall ---
    print("\n--- Test A: Per-key Recall (graph1-graph10) ---")
    exact_count = 0
    confident_count = 0
    for kp in KEYED_PAIRS:
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

        status = "EXACT" if result["exact_match"] else "MISS"
        if result["exact_match"]:
            exact_count += 1
        confidence = result["confidence"]
        if confidence >= DEFAULT_CONFIDENCE_THRESHOLD:
            confident_count += 1

        recalled_str = json.dumps(result["recalled"]) if result["recalled"] else "None"
        raw_preview = raw[:120].replace("\n", " ")
        print(f"  [{status}] {kp['key']} [conf:{confidence:.3f}]: {recalled_str}")
        print(f"           Raw: {raw_preview}...")

    print(f"\n  Exact key recall:    {exact_count}/{len(KEYED_PAIRS)}")
    print(f"  Confident (>{DEFAULT_CONFIDENCE_THRESHOLD}): {confident_count}/{len(KEYED_PAIRS)}")

    # --- Test B: Untrained keys (with key-match + registry guard) ---
    print("\n--- Test B: Untrained Keys (graph11-graph15) ---")
    hallucination_count = 0
    registry_blocked_count = 0
    for i in range(11, 16):
        key = f"graph{i}"
        prompt_text = RECALL_TEMPLATE.format(key=key)
        formatted = _format_inference_prompt(prompt_text, tokenizer)
        raw = generate_answer(
            model,
            tokenizer,
            formatted,
            max_new_tokens=256,
            temperature=0.1,
        )
        recalled = parse_recalled_pair(raw)
        returned_key = recalled.get("key") if recalled else None

        if recalled is not None and returned_key != key:
            print(f"  [REJECT] {key}: returned key='{returned_key}' (key mismatch)")
        elif recalled is not None:
            confidence = verify_confidence(recalled, registry)
            if confidence == 0.0:
                registry_blocked_count += 1
                print(f"  [BLOCK]  {key}: not in registry (untrained key)")
            elif confidence < DEFAULT_CONFIDENCE_THRESHOLD:
                registry_blocked_count += 1
                print(
                    f"  [LOW]    {key} [conf:{confidence:.3f}]: "
                    f"below threshold {DEFAULT_CONFIDENCE_THRESHOLD}"
                )
            else:
                hallucination_count += 1
                print(f"  [HALLUC] {key} [conf:{confidence:.3f}]: {json.dumps(recalled)}")
        else:
            print(f"  [OK]     {key}: None (unparseable)")

    print(f"\n  Hallucinations passing all guards: {hallucination_count}/5")
    print(f"  Blocked by registry:              {registry_blocked_count}/5")

    # --- Test C: Individual QA ---
    print("\n--- Test C: Individual QA Recall ---")
    scores = []
    for qa in QA_PAIRS:
        prompt = _format_inference_prompt(qa["question"], tokenizer)
        generated = generate_answer(
            model,
            tokenizer,
            prompt,
            temperature=0.1,
        )
        score = compute_similarity(qa["answer"], generated)
        match = "OK" if score > 0.7 else "MISS"
        print(f"  [{match}] Q: {qa['question']}")
        print(f"       Expected: {qa['answer']}")
        print(f"       Got:      {generated[:80]}")
        print(f"       Score:    {score:.3f}")
        scores.append(score)

    mean_score = sum(scores) / len(scores)
    print(f"\n  Mean individual recall: {mean_score:.1%}")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print(f"  Exact key recall:    {exact_count}/{len(KEYED_PAIRS)}")
    print(f"  Confident:           {confident_count}/{len(KEYED_PAIRS)}")
    print(f"  Hallucinations:      {hallucination_count}/5")
    print(f"  Registry-blocked:    {registry_blocked_count}/5")
    print(f"  Individual QA mean:  {mean_score:.1%}")
    print("=" * 72)


if __name__ == "__main__":
    main()
