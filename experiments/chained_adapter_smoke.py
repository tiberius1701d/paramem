"""Chained Adapter Smoke Test: Sequential LoRA training.

Tests whether chained LoRA adapters can maintain recall across sessions.

Two modes:
  compose: Train each adapter with all previous frozen but active in forward
           pass (additive composition). Adapters remain separate.
  merge:   Merge each adapter into base via merge_and_unload before training
           the next. Adapters become permanent base weight changes.

Supports homogeneous rank (--rank 8) or progressive ranks per session
(--progressive-ranks 2,4,8).

Usage:
    python experiments/chained_adapter_smoke.py --mode compose --rank 8
    python experiments/chained_adapter_smoke.py --mode compose --progressive-ranks 2,4,8
    python experiments/chained_adapter_smoke.py --mode merge --rank 2
"""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from peft import PeftModel as _PeftModel  # noqa: E402

from experiments.utils.perltqa_loader import (  # noqa: E402
    list_characters,
    load_character_dialogues,
)
from experiments.utils.test_harness import (  # noqa: E402
    IndexedDataset,
    load_model_and_config,
    model_output_dir,
    setup_logging,
)
from paramem.graph.extractor import extract_graph  # noqa: E402
from paramem.graph.qa_generator import generate_qa_from_relations  # noqa: E402
from paramem.models.loader import create_adapter  # noqa: E402
from paramem.training.indexed_memory import (  # noqa: E402
    assign_keys,
    build_registry,
    format_indexed_training,
    probe_all_keys,
    save_registry,
    validate_recall,
)
from paramem.training.trainer import train_adapter  # noqa: E402
from paramem.utils.config import AdapterConfig, ModelConfig, TrainingConfig  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

# Configuration
OUTPUT_BASE = project_root / "outputs" / "chained_adapter_smoke"
RANK = 2
NUM_EPOCHS = 30
NUM_SESSIONS = 3
CHARACTER = "Deng Yu"
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]


def wait_for_cooldown(target=45):
    """Block until GPU temperature drops below target."""
    try:
        subprocess.run(
            [
                "bash",
                "-c",
                f"source ~/.local/bin/gpu-cooldown.sh && wait_for_cooldown {target}",
            ],
            check=True,
            timeout=600,
        )
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ) as e:
        logger.warning("Cooldown script failed (%s), falling back to 60s sleep", e)
        time.sleep(60)


def extract_qa_from_session(session, session_idx, model, tokenizer, prompts_dir=None):
    """Run the full extraction pipeline on a single session transcript.

    Disables all adapters and gradient checkpointing during extraction.
    """
    transcript = session.get("dialogue", session.get("transcript", ""))
    if not transcript:
        return []

    session_id = session.get("session_id", f"session_{session_idx}")

    # Disable gradient checkpointing + adapters for extraction
    model.gradient_checkpointing_disable()
    if isinstance(model, _PeftModel):
        with model.disable_adapter():
            graph_data = extract_graph(
                model,
                tokenizer,
                transcript,
                session_id=session_id,
                temperature=0.0,
                prompts_dir=prompts_dir,
            )
    else:
        graph_data = extract_graph(
            model,
            tokenizer,
            transcript,
            session_id=session_id,
            temperature=0.0,
            prompts_dir=prompts_dir,
        )

    if not graph_data or not graph_data.relations:
        return []

    # Convert Pydantic Relation objects to dicts for QA generator
    relations = [
        {"subject": r.subject, "predicate": r.predicate, "object": r.object}
        for r in graph_data.relations
    ]

    # gradient_checkpointing already disabled above
    if isinstance(model, _PeftModel):
        with model.disable_adapter():
            return generate_qa_from_relations(relations, model=model, tokenizer=tokenizer)
    return generate_qa_from_relations(relations, model=model, tokenizer=tokenizer)


def recall_test(model, tokenizer, all_keyed_pairs, all_registries, label):
    """Run recall on all accumulated keys and print results.

    Returns dict with exact_count, total, rate, per_key.
    """
    model.gradient_checkpointing_disable()

    combined_keyed = []
    combined_registry = {}
    for keyed_pairs, registry in zip(all_keyed_pairs, all_registries):
        combined_keyed.extend(keyed_pairs)
        combined_registry.update(registry)

    if not combined_keyed:
        print(f"  [{label}] No keys to test")
        return {"exact_count": 0, "total": 0, "rate": 0.0, "per_key": []}

    trained_keys = [kp["key"] for kp in combined_keyed]
    recalled = probe_all_keys(model, tokenizer, trained_keys, registry=combined_registry)

    results = []
    exact_count = 0
    for kp in combined_keyed:
        result = validate_recall(recalled[kp["key"]], kp, combined_registry)
        results.append({"key": kp["key"], "adapter": kp.get("adapter"), **result})
        if result["exact_match"]:
            exact_count += 1

    total = len(combined_keyed)
    rate = exact_count / total if total else 0.0
    print(f"  [{label}] Recall: {exact_count}/{total} ({rate:.1%})")

    # Per-adapter breakdown
    adapters_seen = []
    for kp in combined_keyed:
        a = kp.get("adapter")
        if a and a not in adapters_seen:
            adapters_seen.append(a)
    for name in adapters_seen:
        adapter_keys = [r for r in results if r.get("adapter") == name]
        adapter_exact = sum(1 for r in adapter_keys if r["exact_match"])
        adapter_total = len(adapter_keys)
        if adapter_total > 0:
            print(
                f"    {name}: {adapter_exact}/{adapter_total} ({adapter_exact / adapter_total:.1%})"
            )

    return {
        "exact_count": exact_count,
        "total": total,
        "rate": rate,
        "per_key": results,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Chained adapter smoke test")
    parser.add_argument(
        "--quantization",
        type=str,
        default="nf4",
        choices=["nf4", "int8", "none"],
        help="Quantization type (default: nf4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistral",
        choices=["mistral", "qwen"],
        help="Model to use (default: mistral)",
    )
    parser.add_argument(
        "--keys-per-session",
        type=int,
        default=5,
        help="Keys per session when using pre-defined QA (default: 5)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=RANK,
        help=f"Homogeneous LoRA rank for all adapters (default: {RANK})",
    )
    parser.add_argument(
        "--progressive-ranks",
        type=str,
        default=None,
        help="Comma-separated ranks per session, e.g. '2,4,8,16'. Overrides --rank.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="merge",
        choices=["merge", "compose"],
        help="Chaining mode: 'merge' or 'compose' (additive). Default: merge.",
    )
    args = parser.parse_args()

    quant_label = args.quantization
    model_name = args.model
    use_extraction = model_name != "qwen"
    chain_mode = args.mode

    # Parse rank configuration
    if args.progressive_ranks:
        session_ranks = [int(r) for r in args.progressive_ranks.split(",")]
        if len(session_ranks) != NUM_SESSIONS:
            print(
                f"ERROR: --progressive-ranks has {len(session_ranks)} values "
                f"but NUM_SESSIONS={NUM_SESSIONS}"
            )
            sys.exit(1)
        rank_label = "progressive_" + args.progressive_ranks.replace(",", "_")
    else:
        session_ranks = [args.rank] * NUM_SESSIONS
        rank_label = f"r{args.rank}"

    output_dir = model_output_dir(
        OUTPUT_BASE, f"{model_name}_{quant_label}_{chain_mode}_{rank_label}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Load model
    if model_name == "qwen":
        model_config = ModelConfig(
            model_id="Qwen/Qwen2.5-3B",
            quantization=quant_label,
            compute_dtype="bfloat16",
            trust_remote_code=True,
            cpu_offload=False,
        )
    else:
        model_config = ModelConfig(
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            quantization=quant_label,
            compute_dtype="bfloat16",
            trust_remote_code=True,
            cpu_offload=quant_label == "int8",
            max_memory_gpu="7GiB",
            max_memory_cpu="20GiB",
        )

    print(f"Loading {model_name} (quantization={quant_label})...")
    model, tokenizer = load_model_and_config(model_config)

    # Load session data
    if use_extraction:
        characters = list_characters()
        if CHARACTER not in characters:
            print(f"ERROR: Character '{CHARACTER}' not found. Available: {characters}")
            sys.exit(1)
        sessions = load_character_dialogues(CHARACTER)
        if len(sessions) < NUM_SESSIONS:
            print(
                f"ERROR: Need {NUM_SESSIONS} sessions, "
                f"only {len(sessions)} available for '{CHARACTER}'"
            )
            sys.exit(1)
        sessions = sessions[:NUM_SESSIONS]
        predefined_qa = None
        print(f"Using {NUM_SESSIONS} sessions from character '{CHARACTER}'")
    else:
        # Pre-defined QA from Test 8 cycle 11
        qa_source = (
            project_root
            / "outputs/test8_large_scale/mistral/20260323_161747"
            / "cycle_011/keyed_pairs.json"
        )
        if not qa_source.exists():
            print(f"ERROR: QA source not found at {qa_source}")
            sys.exit(1)
        all_qa = json.load(open(qa_source))
        predefined_qa = []
        for s in range(NUM_SESSIONS):
            start = s * args.keys_per_session
            end = start + args.keys_per_session
            chunk = all_qa[start:end]
            if chunk:
                predefined_qa.append(
                    [{"question": kp["question"], "answer": kp["answer"]} for kp in chunk]
                )
        sessions = [None] * len(predefined_qa)
        print(
            f"Using {len(predefined_qa)} pre-defined QA sessions "
            f"({args.keys_per_session} keys each)"
        )

    print(f"Mode: {chain_mode}, Ranks: {session_ranks[: len(sessions)]}")

    training_config = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=2,
        max_seq_length=1024,
        num_epochs=NUM_EPOCHS,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=42,
        save_strategy="no",
    )

    prompts_dir = project_root / "configs" / "prompts"

    # Per-session state
    all_keyed_pairs = []
    all_registries = []
    adapter_names = []
    merged_adapter_names = []
    chain_results = []
    key_offset = 1

    for i, session in enumerate(sessions):
        adapter_name = f"session_{i + 1}"
        session_dir = output_dir / adapter_name
        session_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"SESSION {i + 1}/{NUM_SESSIONS}: '{adapter_name}'")
        if merged_adapter_names:
            print(f"  Merged adapters in base: {merged_adapter_names}")
        print(f"{'=' * 60}")

        # ---- Get QA pairs ----
        if use_extraction:
            print(f"\n  Extracting QA pairs from session {i + 1}...")
            qa_pairs = extract_qa_from_session(
                session, i, model, tokenizer, prompts_dir=prompts_dir
            )
        else:
            qa_pairs = predefined_qa[i] if i < len(predefined_qa) else []
            print(f"\n  Using {len(qa_pairs)} pre-defined QA pairs")

        if not qa_pairs:
            print(f"  WARNING: No QA pairs extracted, skipping session {i + 1}")
            chain_results.append(
                {
                    "session": i + 1,
                    "adapter": adapter_name,
                    "qa_pairs": 0,
                    "total_keys_so_far": sum(len(kp) for kp in all_keyed_pairs),
                    "recall_exact": None,
                    "recall_rate": None,
                    "train_time_s": 0,
                    "train_loss": None,
                    "skipped": True,
                }
            )
            continue

        print(f"  Extracted {len(qa_pairs)} QA pairs")

        # ---- Assign keys ----
        keyed_pairs = assign_keys(qa_pairs, start_index=key_offset)
        for kp in keyed_pairs:
            kp["adapter"] = adapter_name

        registry = build_registry(keyed_pairs)
        save_registry(registry, session_dir / "simhash_registry.json")

        with open(session_dir / "keyed_pairs.json", "w") as f:
            json.dump(keyed_pairs, f, indent=2)

        # ---- Create fresh adapter with per-session rank ----
        rank_i = session_ranks[i] if i < len(session_ranks) else session_ranks[-1]
        adapter_config = AdapterConfig(
            rank=rank_i,
            alpha=rank_i * 2,
            learning_rate=1e-4,
            target_modules=TARGET_MODULES,
            dropout=0.0,
        )
        print(f"  Creating adapter '{adapter_name}' (rank {rank_i})...")
        model = create_adapter(model, adapter_config, adapter_name)
        adapter_names.append(adapter_name)

        # In compose mode, set all adapters active with previous frozen
        active_adapters_arg = None
        if chain_mode == "compose" and len(adapter_names) > 1:
            active_adapters_arg = adapter_names

        # ---- Train ----
        print(f"  Training on {len(keyed_pairs)} keys, {NUM_EPOCHS} epochs, rank {rank_i}...")
        examples = format_indexed_training(keyed_pairs, tokenizer, max_length=1024)
        dataset = IndexedDataset(examples)

        t0 = time.time()
        metrics = train_adapter(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            adapter_name=adapter_name,
            training_config=training_config,
            adapter_config=adapter_config,
            output_dir=session_dir / "adapter",
            run_name=f"chain-{adapter_name}",
            active_adapters=active_adapters_arg,
        )
        train_time = time.time() - t0
        train_loss = metrics.get("train_loss") if metrics else None
        print(f"  Training time: {train_time:.0f}s, loss: {train_loss}")

        # Commit key_offset only after training succeeds
        key_offset += len(keyed_pairs)
        all_keyed_pairs.append(keyed_pairs)
        all_registries.append(registry)

        # ---- Recall tests ----
        print(f"\n  Recall tests for session {i + 1}:")

        # In compose mode, set all adapters active for evaluation
        if chain_mode == "compose" and len(adapter_names) > 1:
            model.base_model.set_adapter(adapter_names)
            recall_current = recall_test(
                model,
                tokenizer,
                [keyed_pairs],
                [registry],
                "current session keys (all adapters active)",
            )
        else:
            recall_current = recall_test(
                model, tokenizer, [keyed_pairs], [registry], "current session only"
            )

        recall_before_merge = None
        if len(all_keyed_pairs) > 1:
            recall_before_merge = recall_test(
                model,
                tokenizer,
                all_keyed_pairs,
                all_registries,
                "all keys, before merge/compose",
            )

        # ---- Merge or keep composed ----
        recall_after_merge = None
        if chain_mode == "merge":
            print(f"\n  Merging adapter '{adapter_name}' into base weights...")
            model = model.merge_and_unload()
            merged_adapter_names.append(adapter_name)
            print(f"  Merged and unloaded. Stack: {merged_adapter_names}")

            recall_after_merge = recall_test(
                model, tokenizer, all_keyed_pairs, all_registries, "all keys, after merge"
            )
        else:
            # Compose mode: adapters stay separate, recall_after = recall_before
            recall_after_merge = recall_before_merge or {
                "total": recall_current["total"],
                "exact_count": recall_current["exact_count"],
                "rate": recall_current["rate"],
                "per_key": recall_current["per_key"],
            }

        # gradient_checkpointing is re-enabled inside train_adapter

        chain_results.append(
            {
                "session": i + 1,
                "adapter": adapter_name,
                "qa_pairs": len(qa_pairs),
                "total_keys_so_far": recall_after_merge["total"],
                "recall_current": recall_current["rate"],
                "recall_before_merge": (
                    recall_before_merge["rate"] if recall_before_merge else None
                ),
                "rank": rank_i,
                "recall_after_merge": recall_after_merge["rate"] if recall_after_merge else None,
                "recall_after_merge_exact": (
                    recall_after_merge["exact_count"] if recall_after_merge else None
                ),
                "train_time_s": train_time,
                "train_loss": train_loss,
                "per_key_after_merge": recall_after_merge["per_key"],
            }
        )

        # Save results after each session
        _save_results(
            output_dir,
            chain_results,
            merged_adapter_names,
            quant_label,
            chain_mode,
            session_ranks,
        )

        # GPU cooldown between sessions
        if i < len(sessions) - 1:
            print("  Cooling down...")
            wait_for_cooldown(45)

    # Unmerge is not possible with merge_and_unload (PEFT wrapper stripped).
    # This is by design — the merged base IS the model.
    unmerge_results = []
    print(
        "\n  Note: merge_and_unload used — unmerge test skipped"
        " (adapters are permanently in base weights)"
    )

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY — {chain_mode.upper()} MODE")
    print(f"{'=' * 60}")
    print(
        f"\n  {'Session':<12} {'Keys':<8} {'Current':<10} "
        f"{'Pre-merge':<12} {'Post-merge':<12} {'Loss':<10} {'Time'}"
    )
    print(f"  {'─' * 75}")
    for r in chain_results:
        if r.get("skipped"):
            print(f"  {r['adapter']:<12} {'—':<8} {'skipped':<10}")
            continue
        pre = f"{r['recall_before_merge']:.1%}" if r["recall_before_merge"] is not None else "—"
        print(
            f"  {r['adapter']:<12} {r['qa_pairs']:<8} "
            f"{r['recall_current']:.1%}      "
            f"{pre:<12} "
            f"{r['recall_after_merge']:.1%}        "
            f"{r.get('train_loss', '?'):<10}  "
            f"{r['train_time_s']:.0f}s"
        )

    if unmerge_results:
        print(f"\n{'=' * 60}")
        print("UNMERGE RESULTS")
        print(f"{'=' * 60}")
        for r in unmerge_results:
            print(
                f"  After unmerging {r['unmerged']}: "
                f"{r['recall_exact']}/{r['total_keys']} ({r['recall_rate']:.1%})"
            )

    print(f"\nResults saved to {output_dir}")
    print("Done.")


def _save_results(
    output_dir,
    chain_results,
    merged_adapter_names,
    quantization="nf4",
    mode="merge",
    session_ranks=None,
):
    """Save results after each session for resumability."""
    summary = {
        "experiment": f"chained_adapter_smoke_{mode}",
        "description": (
            f"Sequential LoRA training ({mode} mode). "
            + (
                "Each adapter merged into base before training the next."
                if mode == "merge"
                else "Additive composition with previous adapters frozen."
            )
        ),
        "rank": RANK,
        "num_epochs": NUM_EPOCHS,
        "num_sessions": NUM_SESSIONS,
        "character": CHARACTER,
        "quantization": quantization,
        "mode": mode,
        "session_ranks": session_ranks or [],
        "merged_adapters": list(merged_adapter_names),
        "chain_results": [
            {k: v for k, v in r.items() if k != "per_key_after_merge"} for r in chain_results
        ],
        "final_recall": next(
            (r["recall_after_merge"] for r in reversed(chain_results) if not r.get("skipped")),
            None,
        ),
        "total_keys": next(
            (r["total_keys_so_far"] for r in reversed(chain_results) if not r.get("skipped")),
            0,
        ),
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "detailed_results.json", "w") as f:
        json.dump(chain_results, f, indent=2)


if __name__ == "__main__":
    main()
