"""Test 6: Parametric vs RAG — storage, latency, and recall quality.

Head-to-head comparison of parametric retrieval (indexed keys) vs RAG on
the same fact set at multiple scales. Measures:
  - Storage: final adapter weights + registry + keyed_pairs vs embeddings + text
  - Latency: bare model, parametric, and RAG (full pipeline) with warm-up
  - Recall quality: both scored by embedding similarity (compute_similarity)

Uses PerLTQA eval QA pairs for sufficient scale. Distills once at max scale,
subsets for smaller scales to ensure consistent fact ordering.

Usage:
    python experiments/test6_footprint.py --model gemma
    python experiments/test6_footprint.py --model gemma --scales 10,25,50,100
    python experiments/test6_footprint.py --model gemma --character "Liang Xin"
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.utils.perltqa_loader import is_available as perltqa_available  # noqa: E402
from experiments.utils.perltqa_loader import load_character_eval_qa, load_qa  # noqa: E402
from experiments.utils.test_harness import (  # noqa: E402
    add_model_args,
    assign_keys,
    build_registry,
    distill_qa_pairs,
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

DEFAULT_CHARACTER = "Liang Xin"
DEFAULT_SCALES = "10,25,50,100"
MAX_NEW_TOKENS = 200
NUM_WARMUP = 3
MATCH_THRESHOLD = 0.75
EMBEDDING_MODEL_SIZE_BYTES = 91_553_587  # all-MiniLM-L6-v2 on-disk (~87.3 MB)


def measure_file_size(path: Path) -> int:
    """Measure size of a single file in bytes."""
    if path.is_file():
        return path.stat().st_size
    return 0


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


def _latency_stats(latencies: list[float], token_counts: list[int] | None = None) -> dict:
    """Compute latency statistics from a list of seconds."""
    stats = {
        "mean_ms": sum(latencies) / len(latencies) * 1000,
        "min_ms": min(latencies) * 1000,
        "max_ms": max(latencies) * 1000,
        "num_queries": len(latencies),
    }
    if token_counts:
        total_tokens = sum(token_counts)
        total_time = sum(latencies)
        stats["mean_output_tokens"] = total_tokens / len(token_counts)
        stats["tokens_per_second"] = total_tokens / total_time if total_time > 0 else 0
    return stats


def measure_inference_latency(
    model, tokenizer, keyed_pairs, registry, adapter_name, num_queries=20
):
    """Measure average inference latency for indexed key recall (full pipeline)."""
    from paramem.models.loader import switch_adapter
    from paramem.training.indexed_memory import probe_key

    model.gradient_checkpointing_disable()
    switch_adapter(model, adapter_name)

    available = len(keyed_pairs) - NUM_WARMUP
    num_queries = min(num_queries, available)
    keys = [kp["key"] for kp in keyed_pairs[: num_queries + NUM_WARMUP]]

    # Warm-up
    for key in keys[:NUM_WARMUP]:
        probe_key(model, tokenizer, key, max_new_tokens=MAX_NEW_TOKENS, registry=registry)

    latencies = []
    token_counts = []
    for key in keys[NUM_WARMUP : NUM_WARMUP + num_queries]:
        start = time.time()
        result = probe_key(model, tokenizer, key, max_new_tokens=MAX_NEW_TOKENS, registry=registry)
        latencies.append(time.time() - start)
        raw = result.get("raw_output", "") if result else ""
        token_counts.append(len(tokenizer.encode(raw)))

    return _latency_stats(latencies, token_counts)


def measure_bare_latency(model, tokenizer, keyed_pairs, num_queries=20):
    """Measure inference latency with no adapter loaded (bare model baseline).

    Uses the same key-recall prompts so prompt length is comparable.
    """
    from peft import PeftModel as _PeftModel

    from paramem.evaluation.recall import generate_answer
    from paramem.training.dataset import _format_inference_prompt

    model.gradient_checkpointing_disable()
    available = len(keyed_pairs) - NUM_WARMUP
    num_queries = min(num_queries, available)
    prompts = [
        _format_inference_prompt(f"Recall the QA pair stored under key '{kp['key']}'.", tokenizer)
        for kp in keyed_pairs[: num_queries + NUM_WARMUP]
    ]

    def _run():
        # Warm-up
        for prompt in prompts[:NUM_WARMUP]:
            generate_answer(
                model,
                tokenizer,
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
            )

        latencies = []
        token_counts = []
        for prompt in prompts[NUM_WARMUP : NUM_WARMUP + num_queries]:
            start = time.time()
            generated = generate_answer(
                model,
                tokenizer,
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
            )
            latencies.append(time.time() - start)
            token_counts.append(len(tokenizer.encode(generated)))
        return latencies, token_counts

    if isinstance(model, _PeftModel):
        with model.disable_adapter():
            latencies, token_counts = _run()
    else:
        latencies, token_counts = _run()

    return _latency_stats(latencies, token_counts)


def measure_rag_latency(rag, model, tokenizer, keyed_pairs, num_queries=20):
    """Measure average RAG retrieval + generation latency."""
    from peft import PeftModel as _PeftModel

    from paramem.evaluation.recall import generate_answer

    model.gradient_checkpointing_disable()
    available = len(keyed_pairs) - NUM_WARMUP
    num_queries = min(num_queries, available)

    queries = keyed_pairs[: num_queries + NUM_WARMUP]

    def _run_rag_queries():
        # Warm-up
        for qa in queries[:NUM_WARMUP]:
            prompt = rag.format_prompt(qa["question"], tokenizer, top_k=3)
            generate_answer(
                model,
                tokenizer,
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
            )

        latencies = []
        token_counts = []
        for qa in queries[NUM_WARMUP : NUM_WARMUP + num_queries]:
            start = time.time()
            prompt = rag.format_prompt(qa["question"], tokenizer, top_k=3)
            generated = generate_answer(
                model,
                tokenizer,
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
            )
            latencies.append(time.time() - start)
            token_counts.append(len(tokenizer.encode(generated)))
        return latencies, token_counts

    if isinstance(model, _PeftModel):
        with model.disable_adapter():
            latencies, token_counts = _run_rag_queries()
    else:
        latencies, token_counts = _run_rag_queries()

    return _latency_stats(latencies, token_counts)


def measure_parametric_recall(model, tokenizer, keyed_pairs, registry, adapter_name):
    """Measure parametric recall quality using embedding similarity (same metric as RAG)."""
    from paramem.evaluation.embedding_scorer import compute_similarity
    from paramem.models.loader import switch_adapter
    from paramem.training.indexed_memory import probe_key

    model.gradient_checkpointing_disable()
    switch_adapter(model, adapter_name)

    results = []
    match_count = 0

    for kp in keyed_pairs:
        recalled = probe_key(
            model,
            tokenizer,
            kp["key"],
            max_new_tokens=MAX_NEW_TOKENS,
            registry=registry,
        )
        generated = recalled.get("answer", "") if recalled else ""
        raw_output = recalled.get("raw_output", "") if recalled else ""
        confidence = recalled.get("confidence", 0.0) if recalled else 0.0

        similarity = compute_similarity(kp["answer"], generated) if generated else 0.0
        is_match = similarity >= MATCH_THRESHOLD
        if is_match:
            match_count += 1

        results.append(
            {
                "key": kp["key"],
                "question": kp["question"],
                "expected": kp["answer"],
                "generated": generated,
                "raw_output": raw_output,
                "similarity": similarity,
                "simhash_confidence": confidence,
                "match": is_match,
            }
        )

    total = len(keyed_pairs) or 1
    return {
        "match_count": match_count,
        "total": len(keyed_pairs),
        "rate": match_count / total,
        "mean_similarity": sum(r["similarity"] for r in results) / total,
        "mean_simhash_confidence": sum(r["simhash_confidence"] for r in results) / total,
        "per_key": results,
    }


def measure_rag_recall(rag, model, tokenizer, keyed_pairs):
    """Measure RAG recall quality on the same questions as parametric."""
    from peft import PeftModel as _PeftModel

    from paramem.evaluation.embedding_scorer import compute_similarity
    from paramem.evaluation.recall import generate_answer

    model.gradient_checkpointing_disable()
    results = []
    match_count = 0

    def _run():
        nonlocal match_count
        for kp in keyed_pairs:
            prompt = rag.format_prompt(kp["question"], tokenizer, top_k=3)
            generated = generate_answer(
                model,
                tokenizer,
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
            )
            sim = compute_similarity(kp["answer"], generated)
            is_match = sim >= MATCH_THRESHOLD
            if is_match:
                match_count += 1
            results.append(
                {
                    "key": kp["key"],
                    "question": kp["question"],
                    "expected": kp["answer"],
                    "generated": generated,
                    "similarity": sim,
                    "match": is_match,
                }
            )

    if isinstance(model, _PeftModel):
        with model.disable_adapter():
            _run()
    else:
        _run()

    total = len(keyed_pairs) or 1
    return {
        "match_count": match_count,
        "total": len(keyed_pairs),
        "rate": match_count / total,
        "mean_similarity": sum(r["similarity"] for r in results) / total,
        "per_query": results,
    }


def train_at_scale(model, tokenizer, keyed_pairs, registry, scale, args, output_dir):
    """Train indexed keys for a specific scale point using pre-distilled pairs.

    Delegates to train_indexed_keys from the shared harness for consistency,
    then saves the final adapter via selected_adapters for storage measurement.
    """
    adapter_name = f"episodic_s{scale}"

    # Remove previous scale adapters to avoid VRAM accumulation.
    # After deleting all adapters, unwrap to clean base model —
    # PEFT can't add a new adapter to a PeftModel with no active config.
    from peft import PeftModel as _PeftModel

    if isinstance(model, _PeftModel):
        stale = [n for n in list(model.peft_config.keys()) if n.startswith("episodic_s")]
        if stale:
            # Unwrap to base model (no merge — adapters are discarded)
            model = model.base_model.model
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()

    subset_qa = [{"question": kp["question"], "answer": kp["answer"]} for kp in keyed_pairs[:scale]]

    scale_dir = output_dir / f"scale_{scale}"
    model, subset_kp, subset_registry, train_time, metrics = train_indexed_keys(
        model,
        tokenizer,
        subset_qa,
        epochs=args.num_epochs,
        rank=args.rank,
        adapter_name=adapter_name,
        output_dir=scale_dir,
        run_name=f"footprint-{scale}",
        skip_distill=True,
    )

    # Save final adapter weights only (not training checkpoints)
    # for accurate storage measurement
    final_adapter_dir = scale_dir / "final_adapter"
    model.save_pretrained(final_adapter_dir, selected_adapters=[adapter_name])

    return model, subset_kp, subset_registry, adapter_name, train_time, metrics


def main():
    parser = argparse.ArgumentParser(description="Test 6: Parametric vs RAG")
    parser.add_argument(
        "--scales",
        type=str,
        default=DEFAULT_SCALES,
        help="Comma-separated scale points",
    )
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument(
        "--character",
        type=str,
        default=DEFAULT_CHARACTER,
        help="PerLTQA character name (default: Liang Xin)",
    )
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    add_model_args(parser)
    args = parser.parse_args()

    base_output_dir = Path(args.output_dir)
    scales = sorted(int(s) for s in args.scales.split(","))
    max_needed = max(scales)
    # Load 2x input pairs to compensate for distillation loss (dedup, extraction)
    load_count = max_needed * 2

    # --- Load data: prefer PerLTQA, fall back to synthetic ---
    if perltqa_available():
        qa_pairs = load_character_eval_qa(args.character, max_pairs=load_count)
        source = f"perltqa:{args.character}"
    else:
        qa_pairs, source = load_qa(max_pairs=load_count)

    if len(qa_pairs) < max_needed:
        logger.error(
            "Need at least %d QA pairs but only have %d from %s. "
            "Reduce --scales or provide more data.",
            max_needed,
            len(qa_pairs),
            source,
        )
        sys.exit(1)

    logger.info("Loaded %d QA pairs from %s", len(qa_pairs), source)

    for bench_name, bench_model_config in get_benchmark_models(args):
        print(f"\n{'=' * 72}")
        print(f"  Model: {bench_name} ({bench_model_config.model_id})")
        print(f"{'=' * 72}")

        model, tokenizer, config = load_model_and_config(bench_model_config)
        output_dir = model_output_dir(base_output_dir, bench_name)

        # --- Distill once using all loaded QA pairs (2x buffer for loss) ---
        print(f"\nDistilling {len(qa_pairs)} QA pairs (one-time)...")
        distilled = distill_qa_pairs(model, tokenizer, qa_pairs)
        all_keyed_pairs = assign_keys(distilled)
        all_registry = build_registry(all_keyed_pairs)

        # Fresh copy of scales per model (distillation yield may differ)
        model_scales = list(scales)
        if len(all_keyed_pairs) < max(model_scales):
            logger.warning(
                "Distillation produced %d keyed pairs, less than max scale %d. "
                "Capping scales accordingly.",
                len(all_keyed_pairs),
                max(model_scales),
            )
            model_scales = [s for s in model_scales if s <= len(all_keyed_pairs)]

        logger.info(
            "Distilled %d keyed pairs from %d QA pairs",
            len(all_keyed_pairs),
            len(qa_pairs),
        )

        # Save distilled pairs for resume
        output_dir.mkdir(parents=True, exist_ok=True)
        kp_all_ser = [
            {"key": kp["key"], "question": kp["question"], "answer": kp["answer"]}
            for kp in all_keyed_pairs
        ]
        with open(output_dir / "all_keyed_pairs.json", "w") as f:
            json.dump(kp_all_ser, f, indent=2)

        scale_results = {}

        for scale in model_scales:
            print(f"\n--- Scale: {scale} keys ---")

            model, keyed_pairs, registry, adapter_name, train_time, metrics = train_at_scale(
                model,
                tokenizer,
                all_keyed_pairs,
                all_registry,
                scale,
                args,
                output_dir,
            )

            scale_dir = output_dir / f"scale_{scale}"

            # --- Storage measurement ---

            # Final adapter weights only (not training checkpoints)
            final_adapter_dir = scale_dir / "final_adapter"
            adapter_size = measure_dir_size(final_adapter_dir)

            # SimHash registry (grows O(n) with keys)
            registry_path = scale_dir / "simhash_registry.json"
            registry_size = measure_file_size(registry_path)

            # Keyed pairs (distilled QA — grows O(n) with keys)
            kp_path = scale_dir / "keyed_pairs.json"
            kp_size = measure_file_size(kp_path)

            # --- Latency measurement (with warm-up) ---

            print("  Measuring bare model latency...")
            bare_latency = measure_bare_latency(model, tokenizer, keyed_pairs)

            print("  Measuring parametric latency...")
            param_latency = measure_inference_latency(
                model,
                tokenizer,
                keyed_pairs,
                registry,
                adapter_name,
            )

            # RAG: index the same keyed_pairs (same questions + answers as PM)
            from paramem.evaluation.rag_qa import QARAGPipeline  # noqa: E402

            rag = QARAGPipeline()
            rag.build_index(keyed_pairs)

            embedding_size = rag.embeddings.nbytes if rag.embeddings is not None else 0
            qa_text_size = sum(
                len(kp["question"].encode()) + len(kp["answer"].encode()) for kp in keyed_pairs
            )

            print("  Measuring RAG latency...")
            rag_latency = measure_rag_latency(rag, model, tokenizer, keyed_pairs)

            # --- Recall quality (both scored by embedding similarity) ---

            print("  Measuring parametric recall quality...")
            param_recall = measure_parametric_recall(
                model,
                tokenizer,
                keyed_pairs,
                registry,
                adapter_name,
            )

            print("  Measuring RAG recall quality...")
            rag_recall = measure_rag_recall(rag, model, tokenizer, keyed_pairs)

            scale_results[str(scale)] = {
                "num_keyed_pairs": len(keyed_pairs),
                "training_loss": metrics.get("train_loss"),
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
                        "match_count": param_recall["match_count"],
                        "total": param_recall["total"],
                        "rate": param_recall["rate"],
                        "mean_similarity": param_recall["mean_similarity"],
                        "mean_simhash_confidence": param_recall["mean_simhash_confidence"],
                        "per_key": param_recall["per_key"],
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
                    "embedding_model_size_bytes": EMBEDDING_MODEL_SIZE_BYTES,
                    "embedding_model_size_mb": EMBEDDING_MODEL_SIZE_BYTES / (1024 * 1024),
                    "total_size_bytes": embedding_size + qa_text_size,
                    "total_size_kb": (embedding_size + qa_text_size) / 1024,
                    "total_size_with_model_bytes": (
                        embedding_size + qa_text_size + EMBEDDING_MODEL_SIZE_BYTES
                    ),
                    "inference_latency": rag_latency,
                    "recall": {
                        "match_count": rag_recall["match_count"],
                        "total": rag_recall["total"],
                        "rate": rag_recall["rate"],
                        "mean_similarity": rag_recall["mean_similarity"],
                        "per_query": rag_recall["per_query"],
                    },
                    "note": "Upper bound — indexes pre-extracted QA pairs, not raw documents",
                },
                "training_time_seconds": train_time,
                "scoring_metric": "embedding_similarity (all-MiniLM-L6-v2 cosine)",
                "match_threshold": MATCH_THRESHOLD,
                "max_new_tokens": MAX_NEW_TOKENS,
            }

            pm = scale_results[str(scale)]["parametric"]
            bm = scale_results[str(scale)]["bare_model"]
            rg = scale_results[str(scale)]["rag"]
            adapter_overhead = param_latency["mean_ms"] - bare_latency["mean_ms"]
            bm_lat = bm["inference_latency"]
            pm_lat = pm["inference_latency"]
            rg_lat = rg["inference_latency"]

            def _tps(lat):
                if "tokens_per_second" in lat:
                    return f", {lat['tokens_per_second']:.1f} tok/s"
                return ""

            bm_tps = _tps(bm_lat)
            pm_tps = _tps(pm_lat)
            rg_tps = _tps(rg_lat)
            print(f"  Bare model:  {bm_lat['mean_ms']:.0f}ms/query{bm_tps}")
            print(
                f"  Parametric:  {pm['total_size_kb']:.1f} KB "
                f"(adapter {pm['adapter_size_kb']:.1f} + "
                f"registry {pm['registry_size_kb']:.1f} + "
                f"kp {pm['keyed_pairs_size_kb']:.1f}), "
                f"{pm_lat['mean_ms']:.0f}ms/query{pm_tps}, "
                f"({adapter_overhead:+.0f}ms overhead), "
                f"recall {pm['recall']['match_count']}/{pm['recall']['total']} "
                f"(sim={pm['recall']['mean_similarity']:.3f})"
            )
            print(
                f"  RAG:         {rg['total_size_kb']:.1f} KB "
                f"(embeddings {rg['embedding_size_kb']:.1f} + "
                f"text {rg['qa_text_size_kb']:.1f}) "
                f"[+{rg['embedding_model_size_mb']:.1f}MB embedding model], "
                f"{rg_lat['mean_ms']:.0f}ms/query{rg_tps}, "
                f"recall {rg['recall']['match_count']}/{rg['recall']['total']} "
                f"(sim={rg['recall']['mean_similarity']:.3f})"
            )

            # Phase save after each scale point
            partial_results = {
                "experiment": "test6_footprint",
                "model": bench_name,
                "model_id": bench_model_config.model_id,
                "epochs": args.num_epochs,
                "rank": args.rank,
                "character": args.character,
                "data_source": source,
                "scale_results": scale_results,
                "note": f"Partial — completed through scale {scale}",
            }
            save_results(partial_results, output_dir)

        # Summary
        print("\n" + "=" * 82)
        print("PARAMETRIC vs RAG SUMMARY (scoring: embedding similarity, threshold 0.75)")
        print(
            f"{'Scale':>6} {'PM Size':>10} {'RAG Size':>10} "
            f"{'Bare Lat':>10} {'PM Lat':>10} {'RAG Lat':>10} "
            f"{'PM Sim':>8} {'RAG Sim':>8} "
            f"{'PM Match':>9} {'RAG Match':>10}"
        )
        print("-" * 100)
        for scale_str, sr in scale_results.items():
            pm = sr["parametric"]
            bm = sr["bare_model"]
            rg = sr["rag"]
            pm_rc = pm["recall"]
            rg_rc = rg["recall"]
            print(
                f"{scale_str:>6} "
                f"{pm['total_size_kb']:>8.1f}KB "
                f"{rg['total_size_with_model_bytes'] / 1024:>8.1f}KB "
                f"{bm['inference_latency']['mean_ms']:>8.0f}ms "
                f"{pm['inference_latency']['mean_ms']:>8.0f}ms "
                f"{rg['inference_latency']['mean_ms']:>8.0f}ms "
                f"{pm_rc['mean_similarity']:>7.3f} "
                f"{rg_rc['mean_similarity']:>7.3f} "
                f"{pm_rc['match_count']:>3}/{pm_rc['total']:<3} "
                f"{rg_rc['match_count']:>4}/{rg_rc['total']:<3}"
            )
        print("=" * 100)

        # Storage breakdown
        print("\nStorage breakdown:")
        for scale_str, sr in scale_results.items():
            pm = sr["parametric"]
            print(
                f"  {scale_str} keys: adapter={pm['adapter_size_kb']:.1f}KB "
                f"registry={pm['registry_size_kb']:.1f}KB "
                f"keyed_pairs={pm['keyed_pairs_size_kb']:.1f}KB"
            )
        print("\nStorage analysis:")
        print("  Fixed infrastructure costs:")
        adapter_kb = list(scale_results.values())[0]["parametric"]["adapter_size_kb"]
        emb_mb = EMBEDDING_MODEL_SIZE_BYTES / (1024 * 1024)
        print(f"    PM adapter:          {adapter_kb:.0f} KB (O(1), set by LoRA rank)")
        print(f"    RAG embedding model: {emb_mb:.1f} MB (O(1), shared)")
        print("  Per-fact variable costs:")
        for scale_str, sr in scale_results.items():
            pm = sr["parametric"]
            rg = sr["rag"]
            n = sr["num_keyed_pairs"]
            pm_var = pm["registry_size_kb"] + pm["keyed_pairs_size_kb"]
            rg_var = rg["total_size_kb"]
            pm_per = (pm_var / n * 1024) if n else 0
            rg_per = (rg_var / n * 1024) if n else 0
            print(
                f"    {scale_str:>3} keys: "
                f"PM {pm_var:.1f}KB ({pm_per:.0f} B/key) vs "
                f"RAG {rg_var:.1f}KB ({rg_per:.0f} B/key)"
            )
        # TODO: Reframe paper storage comparison as:
        #   1. Fixed cost: adapter (35MB) vs embedding model (87MB) — PM wins
        #   2. Per-fact cost: ~0.26KB/key vs ~1.66KB/key — PM wins
        #   3. Total at scale N: PM smaller at every scale
        #   Current summary table lumps fixed + variable, which is misleading.

        # Final save (overwrites partial with complete results)
        results = {
            "experiment": "test6_footprint",
            "model": bench_name,
            "model_id": bench_model_config.model_id,
            "epochs": args.num_epochs,
            "rank": args.rank,
            "character": args.character,
            "data_source": source,
            "scale_results": scale_results,
        }
        save_results(results, output_dir)

        unload_model(model, tokenizer)


if __name__ == "__main__":
    main()
