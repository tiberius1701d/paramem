"""Verify PEFT adapter equivalence: active adapter vs merged weights.

Compares recalled outputs for session_1 keys between:
  (a) session_1 as active PEFT adapter (base + delta in forward pass)
  (b) session_1 merged into base weights (merge_and_unload)

If outputs are identical, PEFT's adapter mechanism is equivalent to a
merged model — the "blackbox" principle holds.

Uses the standard probe_all_keys from the test harness — same code path
as all Tests 1-8.

Usage:
    python experiments/verify_peft_forward.py
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from peft import PeftModel  # noqa: E402

from experiments.utils.test_harness import (  # noqa: E402
    BENCHMARK_MODELS,
    load_model_and_config,
    setup_logging,
)
from paramem.training.indexed_memory import (  # noqa: E402
    probe_all_keys,
    validate_recall,
)

setup_logging()

SMOKE_BASE = project_root / "outputs" / "chained_adapter_smoke"


def find_run_with_adapter():
    """Find a Mistral run directory that has a session_1 adapter."""
    for model_dir in sorted(SMOKE_BASE.iterdir(), reverse=True):
        if "mistral" not in model_dir.name:
            continue
        for run_dir in sorted(model_dir.iterdir(), reverse=True):
            s1 = run_dir / "session_1" / "adapter" / "session_1"
            if s1.exists():
                return run_dir, s1
    return None, None


def main():
    run_dir, s1_path = find_run_with_adapter()
    if run_dir is None:
        print("ERROR: No Mistral run found with session_1 adapter")
        sys.exit(1)

    print(f"Using run: {run_dir}")

    # Load keyed pairs and registry
    with open(run_dir / "session_1" / "keyed_pairs.json") as f:
        keyed_1 = json.load(f)
    with open(run_dir / "session_1" / "simhash_registry.json") as f:
        registry_1 = json.load(f)
    keys_1 = [kp["key"] for kp in keyed_1]
    print(f"Session_1 keys: {len(keys_1)}")

    # Load model + adapter
    print("\nLoading model...")
    model, tokenizer = load_model_and_config(BENCHMARK_MODELS["mistral"])
    model = PeftModel.from_pretrained(model, str(s1_path), adapter_name="session_1")
    model.eval()
    model.gradient_checkpointing_disable()

    # Test A: adapter active (PEFT forward pass with delta)
    print("\n" + "=" * 60)
    print("TEST A: session_1 as active PEFT adapter")
    print("=" * 60)
    model.set_adapter("session_1")
    recalled_adapter = probe_all_keys(model, tokenizer, keys_1, registry=registry_1)

    exact_adapter = 0
    for kp in keyed_1:
        result = validate_recall(recalled_adapter[kp["key"]], kp, registry_1)
        if result["exact_match"]:
            exact_adapter += 1
    print(f"  Recall: {exact_adapter}/{len(keyed_1)}")

    # Test B: merge adapter into base, then probe bare model
    print("\n" + "=" * 60)
    print("TEST B: session_1 merged into base weights")
    print("=" * 60)
    model = model.merge_and_unload()
    recalled_merged = probe_all_keys(model, tokenizer, keys_1, registry=registry_1)

    exact_merged = 0
    for kp in keyed_1:
        result = validate_recall(recalled_merged[kp["key"]], kp, registry_1)
        if result["exact_match"]:
            exact_merged += 1
    print(f"  Recall: {exact_merged}/{len(keyed_1)}")

    # Compare raw outputs
    print("\n" + "=" * 60)
    print("COMPARISON: per-key output diff")
    print("=" * 60)
    changed = 0
    for kp in keyed_1:
        key = kp["key"]
        raw_a = recalled_adapter[key].get("raw_output", "")
        raw_m = recalled_merged[key].get("raw_output", "")
        same = raw_a == raw_m
        if not same:
            changed += 1
            print(f"\n  {key}: OUTPUT CHANGED")
            print(f"    adapter: {raw_a[:120]}")
            print(f"    merged:  {raw_m[:120]}")
        else:
            print(f"  {key}: identical")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Adapter active: {exact_adapter}/{len(keyed_1)}")
    print(f"  Merged:         {exact_merged}/{len(keyed_1)}")
    print(f"  Outputs changed: {changed}/{len(keyed_1)}")

    if changed == 0:
        print("\n  CONCLUSION: PEFT adapter == merged weights. Blackbox principle holds.")
    elif exact_adapter == exact_merged:
        print(
            "\n  CONCLUSION: Minor output differences but same recall."
            " NF4 requantization causes text variations without affecting correctness."
        )
    else:
        print(
            f"\n  CONCLUSION: Merge changes recall from {exact_adapter} to "
            f"{exact_merged}. NF4 requantization degrades the encoding."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
