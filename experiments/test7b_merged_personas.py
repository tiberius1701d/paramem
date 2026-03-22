"""Test 7b: Multi-Adapter Composition — both adapters active simultaneously.

Tests two approaches for serving multiple personas from one model:
1. Additive composition: both adapters active via set_adapter(["a", "b"]),
   LoRA deltas summed at each forward pass.
2. Weight merging (negative control): add_weighted_adapter with [0.5, 0.5].
   Known to fail for indexed key recall — included for documentation.

Requires: completed Test 7 run with saved adapters.

Usage:
    python experiments/test7b_merged_personas.py --model gemma
    python experiments/test7b_merged_personas.py --run-dir outputs/.../TIMESTAMP
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(project_root / ".env")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

from experiments.utils.test_harness import (  # noqa: E402
    add_model_args,
    evaluate_indexed_recall,
    get_benchmark_models,
    load_model_and_config,
    save_results,
    setup_logging,
)
from paramem.models.loader import (  # noqa: E402
    load_adapter,
    switch_adapter,
    unload_model,
)
from paramem.training.indexed_memory import (  # noqa: E402
    load_registry,
)

setup_logging()

OUTPUT_DIR = project_root / "outputs" / "test7_second_persona"


def find_latest_run(base_dir: Path, model_name: str) -> Path | None:
    """Find the latest timestamped run directory."""
    model_dir = base_dir / model_name
    if not model_dir.exists():
        return None
    runs = sorted(model_dir.iterdir(), reverse=True)
    for d in runs:
        if d.is_dir() and (d / "results.json").exists():
            return d
    return None


def main():
    parser = argparse.ArgumentParser(description="Test 7b: Merged Persona Adapters")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to a completed Test 7 run directory",
    )
    add_model_args(parser)
    args = parser.parse_args()

    for bench_name, bench_model_config in get_benchmark_models(args):
        # Find the Test 7 run to load from
        if args.run_dir:
            run_dir = Path(args.run_dir)
        else:
            run_dir = find_latest_run(OUTPUT_DIR, bench_name)

        if not run_dir or not run_dir.exists():
            print(f"ERROR: No Test 7 run found for {bench_name}")
            continue

        print(f"\n{'=' * 72}")
        print(f"  Test 7b: Merged Persona Adapters — {bench_name}")
        print(f"  Loading from: {run_dir}")
        print(f"{'=' * 72}")

        # Check required files exist
        persona_a_dir = run_dir / "persona_a"
        persona_b_dir = run_dir / "persona_b"
        for d, name in [(persona_a_dir, "A"), (persona_b_dir, "B")]:
            adapter_dir = d / "adapter" / f"persona_{name.lower()}"
            if not (adapter_dir / "adapter_config.json").exists():
                print(f"ERROR: Persona {name} adapter not found at {adapter_dir}")
                continue

        # Load keyed_pairs and registries
        kp_a = json.loads((persona_a_dir / "keyed_pairs.json").read_text())
        kp_b = json.loads((persona_b_dir / "keyed_pairs.json").read_text())
        reg_a = load_registry(persona_a_dir / "simhash_registry.json")
        reg_b = load_registry(persona_b_dir / "simhash_registry.json")

        print(f"  Persona A: {len(kp_a)} keys")
        print(f"  Persona B: {len(kp_b)} keys")

        # Load model + both adapters
        model, tokenizer, _ = load_model_and_config(bench_model_config)
        model = load_adapter(model, str(persona_a_dir / "adapter"), "persona_a")
        model = load_adapter(model, str(persona_b_dir / "adapter"), "persona_b")
        print(f"  Adapters loaded: {list(model.peft_config.keys())}")

        # Phase 1: Verify individual recall (sanity check)
        print("\n--- Individual recall (sanity check) ---")
        switch_adapter(model, "persona_a")
        recall_a = evaluate_indexed_recall(
            model,
            tokenizer,
            kp_a,
            reg_a,
            adapter_name="persona_a",
        )
        print(f"  Persona A (own adapter): {recall_a['exact_count']}/{recall_a['total']}")

        switch_adapter(model, "persona_b")
        recall_b = evaluate_indexed_recall(
            model,
            tokenizer,
            kp_b,
            reg_b,
            adapter_name="persona_b",
        )
        print(f"  Persona B (own adapter): {recall_b['exact_count']}/{recall_b['total']}")

        # Phase 2: Additive composition — both adapters active simultaneously
        # Note: evaluate_indexed_recall calls switch_adapter internally,
        # which would override composition. Use probe_all_keys directly.
        print("\n--- Additive composition (both adapters active) ---")
        model.base_model.set_adapter(["persona_a", "persona_b"])
        model.gradient_checkpointing_disable()
        print("  Both adapters active via set_adapter(['persona_a', 'persona_b'])")

        from paramem.training.indexed_memory import (
            probe_all_keys,
            save_registry,
            validate_recall,
        )

        combined_reg = {**reg_a, **reg_b}

        print("\n--- Composed recall: Persona A keys ---")
        keys_a = [kp["key"] for kp in kp_a]
        recalled_comp_a = probe_all_keys(model, tokenizer, keys_a, registry=reg_a)
        exact_comp_a = sum(
            1
            for kp in kp_a
            if validate_recall(recalled_comp_a.get(kp["key"]), kp, reg_a).get("exact_match", False)
        )
        recall_composed_a = {"exact_count": exact_comp_a, "total": len(kp_a)}
        print(f"  Persona A composed: {exact_comp_a}/{len(kp_a)}")

        print("\n--- Composed recall: Persona B keys ---")
        keys_b = [kp["key"] for kp in kp_b]
        recalled_comp_b = probe_all_keys(model, tokenizer, keys_b, registry=reg_b)
        exact_comp_b = sum(
            1
            for kp in kp_b
            if validate_recall(recalled_comp_b.get(kp["key"]), kp, reg_b).get("exact_match", False)
        )
        recall_composed_b = {"exact_count": exact_comp_b, "total": len(kp_b)}
        print(f"  Persona B composed: {exact_comp_b}/{len(kp_b)}")

        # Phase 3: Weight merge (negative control — documented failure)
        print("\n--- Weight merge [0.5, 0.5] (negative control) ---")
        from peft import LoraModel

        base_lora = model.base_model
        if isinstance(base_lora, LoraModel):
            base_lora.add_weighted_adapter(
                adapters=["persona_a", "persona_b"],
                weights=[0.5, 0.5],
                adapter_name="merged",
                combination_type="linear",
            )
            switch_adapter(model, "merged")

            recall_merge_a = evaluate_indexed_recall(
                model,
                tokenizer,
                kp_a,
                combined_reg,
                adapter_name="merged",
            )
            recall_merge_b = evaluate_indexed_recall(
                model,
                tokenizer,
                kp_b,
                combined_reg,
                adapter_name="merged",
            )
            print(
                f"  Merged A: "
                f"{recall_merge_a['exact_count']}/{recall_merge_a['total']}, "
                f"Merged B: "
                f"{recall_merge_b['exact_count']}/{recall_merge_b['total']}"
            )
        else:
            recall_merge_a = {"exact_count": 0, "total": len(kp_a)}
            recall_merge_b = {"exact_count": 0, "total": len(kp_b)}

        # Save results
        out_dir = run_dir / "composed"
        out_dir.mkdir(exist_ok=True)

        all_kp = kp_a + kp_b
        with open(out_dir / "keyed_pairs.json", "w") as f:
            json.dump(all_kp, f, indent=2)
        save_registry(combined_reg, out_dir / "simhash_registry.json")

        # Summary
        print(f"\n{'=' * 72}")
        print("TEST 7b: MULTI-ADAPTER COMPOSITION RESULTS")
        print(f"{'=' * 72}")
        print(f"  Persona A individual:  {recall_a['exact_count']}/{recall_a['total']}")
        print(f"  Persona B individual:  {recall_b['exact_count']}/{recall_b['total']}")
        print(f"  Persona A composed:    {exact_comp_a}/{len(kp_a)}")
        print(f"  Persona B composed:    {exact_comp_b}/{len(kp_b)}")
        print(
            f"  Persona A merged:      "
            f"{recall_merge_a['exact_count']}/{recall_merge_a['total']} "
            f"(negative control)"
        )
        print(
            f"  Persona B merged:      "
            f"{recall_merge_b['exact_count']}/{recall_merge_b['total']} "
            f"(negative control)"
        )
        print(f"{'=' * 72}")

        results = {
            "experiment": "test7b_composed_personas",
            "model": bench_name,
            "source_run": str(run_dir),
            "persona_a_keys": len(kp_a),
            "persona_b_keys": len(kp_b),
            "individual_recall_a": recall_a,
            "individual_recall_b": recall_b,
            "composed_recall_a": recall_composed_a,
            "composed_recall_b": recall_composed_b,
            "merged_recall_a": recall_merge_a,
            "merged_recall_b": recall_merge_b,
        }

        save_results(results, out_dir)

        unload_model(model, tokenizer)


if __name__ == "__main__":
    main()
