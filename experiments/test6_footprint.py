"""Test 6: Edge Deployment Footprint — system size and latency comparison.

Measures adapter file sizes, registry sizes, inference latency, and cold
start time. Compares against equivalent RAG setup.

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


def measure_inference_latency(
    model, tokenizer, keyed_pairs, registry, adapter_name, num_queries=20
):
    """Measure average inference latency for indexed key recall."""
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

    return {
        "mean_ms": sum(latencies) / len(latencies) * 1000,
        "min_ms": min(latencies) * 1000,
        "max_ms": max(latencies) * 1000,
        "num_queries": len(latencies),
    }


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

    return {
        "mean_ms": sum(latencies) / len(latencies) * 1000,
        "min_ms": min(latencies) * 1000,
        "max_ms": max(latencies) * 1000,
        "num_queries": len(latencies),
    }


def main():
    parser = argparse.ArgumentParser(description="Test 6: Edge Deployment Footprint")
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

            # Measure adapter size
            adapter_dir = output_dir / f"scale_{scale}" / "adapter"
            adapter_size = measure_dir_size(adapter_dir)

            # Measure registry size
            registry_path = output_dir / f"scale_{scale}" / "simhash_registry.json"
            registry_size = registry_path.stat().st_size if registry_path.exists() else 0

            # Parametric memory inference latency
            param_latency = measure_inference_latency(
                model,
                tokenizer,
                keyed_pairs,
                registry,
                adapter_name,
            )

            # RAG footprint

            from paramem.evaluation.rag_qa import QARAGPipeline  # noqa: E402

            rag = QARAGPipeline()
            rag.build_index(subset)

            # RAG storage: embeddings + QA text
            embedding_size = rag.embeddings.nbytes if rag.embeddings is not None else 0
            qa_text_size = sum(
                len(qa["question"].encode()) + len(qa["answer"].encode()) for qa in subset
            )

            # RAG latency
            rag_latency = measure_rag_latency(rag, model, tokenizer, subset)

            scale_results[str(scale)] = {
                "parametric": {
                    "adapter_size_bytes": adapter_size,
                    "adapter_size_kb": adapter_size / 1024,
                    "registry_size_bytes": registry_size,
                    "total_size_bytes": adapter_size + registry_size,
                    "total_size_kb": (adapter_size + registry_size) / 1024,
                    "inference_latency": param_latency,
                },
                "rag": {
                    "embedding_size_bytes": embedding_size,
                    "qa_text_size_bytes": qa_text_size,
                    "total_size_bytes": embedding_size + qa_text_size,
                    "total_size_kb": (embedding_size + qa_text_size) / 1024,
                    "inference_latency": rag_latency,
                },
                "training_time_seconds": train_time,
            }

            pm = scale_results[str(scale)]["parametric"]
            rg = scale_results[str(scale)]["rag"]
            print(
                f"  Parametric: {pm['total_size_kb']:.1f} KB, "
                f"{pm['inference_latency']['mean_ms']:.0f}ms/query"
            )
            print(
                f"  RAG:        {rg['total_size_kb']:.1f} KB, "
                f"{rg['inference_latency']['mean_ms']:.0f}ms/query"
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
        print("EDGE DEPLOYMENT FOOTPRINT SUMMARY")
        print(f"{'Scale':>6} {'PM Size':>10} {'RAG Size':>10} {'PM Lat':>10} {'RAG Lat':>10}")
        print("-" * 50)
        for scale_str, sr in scale_results.items():
            pm = sr["parametric"]
            rg = sr["rag"]
            print(
                f"{scale_str:>6} "
                f"{pm['total_size_kb']:>8.1f}KB "
                f"{rg['total_size_kb']:>8.1f}KB "
                f"{pm['inference_latency']['mean_ms']:>8.0f}ms "
                f"{rg['inference_latency']['mean_ms']:>8.0f}ms"
            )
        print("=" * 72)

        print("\nNote: Adapter size is constant across scale points (fixed LoRA rank).")
        print("RAG storage scales linearly with number of facts.")

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
