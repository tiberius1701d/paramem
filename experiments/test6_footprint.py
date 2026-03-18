"""Test 6: Parametric vs RAG — storage, latency, and recall quality.

Head-to-head comparison of parametric retrieval (indexed keys) vs RAG on
the same fact set at multiple scales. Measures:
  - Storage: adapter + registry + keyed_pairs vs embeddings + text
  - Latency: bare model, parametric, and RAG (full pipeline)
  - Recall quality: parametric keyed recall vs RAG answer accuracy

Usage:
    python experiments/test6_footprint.py
    python experiments/test6_footprint.py --model gemma
    python experiments/test6_footprint.py --scales 10,50,100
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.utils.perltqa_loader import load_qa  # noqa: E402
from experiments.utils.test_harness import (  # noqa: E402
    add_model_args,
    evaluate_indexed_recall,
    get_benchmark_models,
    load_model_and_config,
    model_output_dir,
    save_results,
    setup_logging,
    train_indexed_keys,
)
from paramem.models.loader import unload_model  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

OUTPUT_DIR = project_root / "outputs" / "test6_footprint"


def measure_dir_size(path: Path) -> int:
    """Recursively measure directory size in bytes."""
    total = 0
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return total


def _latency_stats(latencies: list[float]) -> dict:
    """Compute latency statistics from a list of seconds."""
    return {
        "mean_ms": sum(latencies) / len(latencies) * 1000,
        "min_ms": min(latencies) * 1000,
        "max_ms": max(latencies) * 1000,
        "num_queries": len(latencies),
    }


def measure_inference_latency(
    model, tokenizer, keyed_pairs, registry, adapter_name, num_queries=20
):
    """Measure average inference latency for indexed key recall (full pipeline)."""
    from paramem.models.loader import switch_adapter
    from paramem.training.indexed_memory import probe_key

    model.gradient_checkpointing_disable()
    switch_adapter(model, adapter_name)

    keys = [kp["key"] for kp in keyed_pairs[:num_queries]]
    latencies = []

    for key in keys:
        start = time.time()
        probe_key(model, tokenizer, key, registry=registry)
        latencies.append(time.time() - start)

    return _latency_stats(latencies)


def measure_bare_latency(model, tokenizer, keyed_pairs, num_queries=20):
    """Measure inference latency with no adapter loaded (bare model baseline).

    Uses the same key-recall prompts so prompt length is comparable.
    """
    from peft import PeftModel as _PeftModel

    from paramem.evaluation.recall import generate_answer
    from paramem.training.dataset import _format_inference_prompt

    model.gradient_checkpointing_disable()
    prompts = [
        _format_inference_prompt(
            f"Recall the QA pair stored under key '{kp['key']}'.", tokenizer
        )
        for kp in keyed_pairs[:num_queries]
    ]
    latencies = []

    def _run():
        for prompt in prompts:
            start = time.time()
            generate_answer(
                model, tokenizer, prompt,
                max_new_tokens=150, temperature=0.1, repetition_penalty=1.3,
            )
            latencies.append(time.time() - start)

    if isinstance(model, _PeftModel):
        with model.disable_adapter():
            _run()
    else:
        _run()

    return _latency_stats(latencies)


def measure_rag_latency(rag, model, tokenizer, qa_pairs, num_queries=20):
    """Measure average RAG retrieval + generation latency."""
    from peft import PeftModel as _PeftModel

    from paramem.evaluation.recall import generate_answer

    model.gradient_checkpointing_disable()

    queries = qa_pairs[:num_queries]
    latencies = []

    def _run_rag_queries():
        for qa in queries:
            start = time.time()
            prompt = rag.format_prompt(qa["question"], tokenizer, top_k=3)
            generate_answer(
                model,
                tokenizer,
                prompt,
                max_new_tokens=150,
                temperature=0.1,
                repetition_penalty=1.3,
            )
            latencies.append(time.time() - start)

    if isinstance(model, _PeftModel):
        with model.disable_adapter():
            _run_rag_queries()
    else:
        _run_rag_queries()

    return _latency_stats(latencies)


def measure_rag_recall(rag, model, tokenizer, qa_pairs):
    """Measure RAG recall quality: ask each question via RAG, score against ground truth."""
    from peft import PeftModel as _PeftModel

    from paramem.evaluation.embedding_scorer import compute_similarity
    from paramem.evaluation.recall import generate_answer

    model.gradient_checkpointing_disable()
    results = []
    exact_count = 0

    def _run():
        nonlocal exact_count
        for qa in qa_pairs:
            prompt = rag.format_prompt(qa["question"], tokenizer, top_k=3)
            generated = generate_answer(
                model, tokenizer, prompt,
                max_new_tokens=150, temperature=0.1, repetition_penalty=1.3,
            )
            sim = compute_similarity(qa["answer"], generated)
            # Use same threshold as indexed recall: similarity >= 0.75 counts as correct
            is_match = sim >= 0.75
            if is_match:
                exact_count += 1
            results.append({
                "question": qa["question"],
                "expected": qa["answer"],
                "generated": generated,
                "similarity": sim,
                "match": is_match,
            })

    if isinstance(model, _PeftModel):
        with model.disable_adapter():
            _run()
    else:
        _run()

    total = len(qa_pairs) or 1
    return {
        "exact_count": exact_count,
        "total": len(qa_pairs),
        "rate": exact_count / total,
        "mean_similarity": sum(r["similarity"] for r in results) / total,
        "per_query": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Test 6: Parametric vs RAG")
    parser.add_argument(
        "--scales", type=str, default="10,25,50", help="Comma-separated scale points"
    )
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    add_model_args(parser)
    args = parser.parse_args()

    base_output_dir = Path(args.output_dir)
    scales = [int(s) for s in args.scales.split(",")]
    max_needed = max(scales)

    qa_pairs, source = load_qa(max_pairs=max_needed)
    logger.info("Loaded %d QA pairs from %s", len(qa_pairs), source)

    for bench_name, bench_model_config in get_benchmark_models(args):
        print(f"\n{'=' * 72}")
        print(f"  Model: {bench_name} ({bench_model_config.model_id})")
        print(f"{'=' * 72}")

        model, tokenizer, config = load_model_and_config(bench_model_config)
        output_dir = model_output_dir(base_output_dir, bench_name)

        scale_results = {}

        for scale in scales:
            print(f"\n--- Scale: {scale} keys ---")
            subset = qa_pairs[:scale]

            adapter_name = f"episodic_s{scale}"
            if hasattr(model, "peft_config") and adapter_name in model.peft_config:
                model.delete_adapter(adapter_name)

            model, keyed_pairs, registry, train_time, metrics = train_indexed_keys(
                model,
                tokenizer,
                subset,
                epochs=args.num_epochs,
                rank=args.rank,
                adapter_name=adapter_name,
                output_dir=output_dir / f"scale_{scale}",
                run_name=f"footprint-{scale}",
            )

            # Save keyed_pairs for resume
            scale_dir = output_dir / f"scale_{scale}"
            kp_ser = [
                {"key": kp["key"], "question": kp["question"], "answer": kp["answer"]}
                for kp in keyed_pairs
            ]
            with open(scale_dir / "keyed_pairs.json", "w") as f:
                json.dump(kp_ser, f, indent=2)

            # --- Storage measurement ---
            scale_path = output_dir / f"scale_{scale}"

            # Adapter size (LoRA weights — fixed by rank, not fact count)
            adapter_dir = scale_path / "adapter"
            adapter_size = measure_dir_size(adapter_dir)

            # SimHash registry (grows O(n) with keys)
            registry_path = scale_path / "simhash_registry.json"
            registry_size = registry_path.stat().st_size if registry_path.exists() else 0

            # Keyed pairs (distilled QA — grows O(n) with keys)
            kp_path = scale_path / "keyed_pairs.json"
            kp_size = kp_path.stat().st_size if kp_path.exists() else 0

            # --- Latency measurement ---

            # Bare model baseline (no adapter)
            print("  Measuring bare model latency...")
            bare_latency = measure_bare_latency(model, tokenizer, keyed_pairs)

            # Parametric memory (adapter active, full probe_key pipeline)
            print("  Measuring parametric latency...")
            param_latency = measure_inference_latency(
                model,
                tokenizer,
                keyed_pairs,
                registry,
                adapter_name,
            )

            # RAG (embed query + vector search + context construction + generation)
            from paramem.evaluation.rag_qa import QARAGPipeline  # noqa: E402

            rag = QARAGPipeline()
            rag.build_index(subset)

            embedding_size = rag.embeddings.nbytes if rag.embeddings is not None else 0
            qa_text_size = sum(
                len(qa["question"].encode()) + len(qa["answer"].encode()) for qa in subset
            )

            print("  Measuring RAG latency...")
            rag_latency = measure_rag_latency(rag, model, tokenizer, subset)

            # --- Recall quality ---

            # Parametric: keyed recall (structured)
            print("  Measuring parametric recall quality...")
            param_recall = evaluate_indexed_recall(
                model, tokenizer, keyed_pairs, registry,
                adapter_name=adapter_name,
            )

            # RAG: same questions, retrieve + generate
            print("  Measuring RAG recall quality...")
            rag_recall = measure_rag_recall(rag, model, tokenizer, subset)

            scale_results[str(scale)] = {
                "parametric": {
                    "adapter_size_bytes": adapter_size,
                    "adapter_size_kb": adapter_size / 1024,
                    "registry_size_bytes": registry_size,
                    "registry_size_kb": registry_size / 1024,
                    "keyed_pairs_size_bytes": kp_size,
                    "keyed_pairs_size_kb": kp_size / 1024,
                    "total_size_bytes": adapter_size + registry_size + kp_size,
                    "total_size_kb": (adapter_size + registry_size + kp_size) / 1024,
                    "inference_latency": param_latency,
                    "recall": {
                        "exact_count": param_recall["exact_count"],
                        "total": param_recall["total"],
                        "rate": param_recall["rate"],
                        "mean_confidence": param_recall["mean_confidence"],
                    },
                },
                "bare_model": {
                    "inference_latency": bare_latency,
                },
                "rag": {
                    "embedding_size_bytes": embedding_size,
                    "embedding_size_kb": embedding_size / 1024,
                    "qa_text_size_bytes": qa_text_size,
                    "qa_text_size_kb": qa_text_size / 1024,
                    "total_size_bytes": embedding_size + qa_text_size,
                    "total_size_kb": (embedding_size + qa_text_size) / 1024,
                    "inference_latency": rag_latency,
                    "recall": {
                        "exact_count": rag_recall["exact_count"],
                        "total": rag_recall["total"],
                        "rate": rag_recall["rate"],
                        "mean_similarity": rag_recall["mean_similarity"],
                        "per_query": rag_recall["per_query"],
                    },
                },
                "training_time_seconds": train_time,
            }

            pm = scale_results[str(scale)]["parametric"]
            bm = scale_results[str(scale)]["bare_model"]
            rg = scale_results[str(scale)]["rag"]
            adapter_overhead = param_latency["mean_ms"] - bare_latency["mean_ms"]
            print(
                f"  Bare model:  {bm['inference_latency']['mean_ms']:.0f}ms/query"
            )
            print(
                f"  Parametric:  {pm['total_size_kb']:.1f} KB "
                f"(adapter {pm['adapter_size_kb']:.1f} + "
                f"registry {pm['registry_size_kb']:.1f} + "
                f"kp {pm['keyed_pairs_size_kb']:.1f}), "
                f"{pm['inference_latency']['mean_ms']:.0f}ms/query "
                f"(+{adapter_overhead:.0f}ms overhead), "
                f"recall {pm['recall']['exact_count']}/{pm['recall']['total']}"
            )
            print(
                f"  RAG:         {rg['total_size_kb']:.1f} KB "
                f"(embeddings {rg['embedding_size_kb']:.1f} + "
                f"text {rg['qa_text_size_kb']:.1f}), "
                f"{rg['inference_latency']['mean_ms']:.0f}ms/query, "
                f"recall {rg['recall']['exact_count']}/{rg['recall']['total']}"
            )

            # Phase save after each scale point
            partial_results = {
                "experiment": "test6_footprint",
                "model": bench_name,
                "model_id": bench_model_config.model_id,
                "epochs": args.num_epochs,
                "rank": args.rank,
                "data_source": source,
                "scale_results": scale_results,
                "note": f"Partial — completed through scale {scale}",
            }
            save_results(partial_results, output_dir)

        # Summary
        print("\n" + "=" * 72)
        print("PARAMETRIC vs RAG SUMMARY")
        print(
            f"{'Scale':>6} {'PM Size':>10} {'RAG Size':>10} "
            f"{'Bare Lat':>10} {'PM Lat':>10} {'RAG Lat':>10} "
            f"{'PM Recall':>10} {'RAG Recall':>11}"
        )
        print("-" * 82)
        for scale_str, sr in scale_results.items():
            pm = sr["parametric"]
            bm = sr["bare_model"]
            rg = sr["rag"]
            pm_rc = pm["recall"]
            rg_rc = rg["recall"]
            print(
                f"{scale_str:>6} "
                f"{pm['total_size_kb']:>8.1f}KB "
                f"{rg['total_size_kb']:>8.1f}KB "
                f"{bm['inference_latency']['mean_ms']:>8.0f}ms "
                f"{pm['inference_latency']['mean_ms']:>8.0f}ms "
                f"{rg['inference_latency']['mean_ms']:>8.0f}ms "
                f"{pm_rc['exact_count']:>3}/{pm_rc['total']:<3} "
                f"{rg_rc['exact_count']:>4}/{rg_rc['total']:<3}"
            )
        print("=" * 82)

        # Storage breakdown
        print("\nStorage breakdown:")
        for scale_str, sr in scale_results.items():
            pm = sr["parametric"]
            print(
                f"  {scale_str} keys: adapter={pm['adapter_size_kb']:.1f}KB "
                f"registry={pm['registry_size_kb']:.1f}KB "
                f"keyed_pairs={pm['keyed_pairs_size_kb']:.1f}KB"
            )
        print("\nAdapter size is O(1) — determined by LoRA rank, not fact count.")
        print("Registry and keyed_pairs are O(n) but measured in bytes per key.")

        # Final save (overwrites partial with complete results)
        results = {
            "experiment": "test6_footprint",
            "model": bench_name,
            "model_id": bench_model_config.model_id,
            "epochs": args.num_epochs,
            "rank": args.rank,
            "data_source": source,
            "scale_results": scale_results,
        }
        save_results(results, output_dir)

        unload_model(model, tokenizer)


if __name__ == "__main__":
    main()
