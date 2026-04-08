"""Test 9: Natural Recall Emergence.

Measures how parametric memory surfaces facts across three probe styles
and how recall emerges with adapter scale.

For each Test 8 cycle checkpoint, three probe passes:
  1. **Keyed retrieval**: "Recall the QA pair stored under key 'graphN'."
     (structured, baseline — should be ~100%)
  2. **Direct question**: The natural question from keyed_pairs without
     the key prefix. (medium difficulty)
  3. **Open-ended**: "What do you know about {entity}?" — one per unique
     entity, scored against all known facts. (hardest)

Output: emergence curve per probe style across all cycles.

Usage:
    python experiments/test9_natural_recall.py --model mistral
    python experiments/test9_natural_recall.py --model mistral --cycles 34,35
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

PAUSE_FILE = Path.home() / ".training_pause"


def is_paused():
    """Check if pause has been requested via tpause."""
    return PAUSE_FILE.exists()


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from peft import PeftModel  # noqa: E402

from experiments.utils.test_harness import (  # noqa: E402
    BENCHMARK_MODELS,
    model_output_dir,
    setup_logging,
)
from paramem.evaluation.recall import generate_answer  # noqa: E402
from paramem.models.loader import load_adapter, load_base_model, switch_adapter  # noqa: E402
from paramem.training.indexed_memory import probe_key  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

OUTPUT_BASE = project_root / "outputs" / "test9_natural_recall"
TEST8_BASE = project_root / "outputs" / "test8_large_scale"
SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely."

# A fact is "recalled" if this fraction of its content words appear in response
FACT_OVERLAP_THRESHOLD = 0.4

STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "shall",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "about",
    "that",
    "this",
    "it",
    "its",
    "and",
    "or",
    "but",
    "not",
    "no",
    "so",
    "if",
    "then",
    "than",
    "who",
    "what",
    "where",
    "when",
    "how",
    "which",
    "their",
    "they",
    "them",
    "he",
    "she",
    "his",
    "her",
    "you",
    "your",
}


def content_tokens(text: str) -> set[str]:
    """Extract content words (lowercase, stop words removed)."""
    return set(text.lower().split()) - STOP_WORDS


def score_fact_overlap(generated: str, expected_answer: str) -> dict:
    """Score whether a single fact appears in the generated response."""
    expected_tokens = content_tokens(expected_answer)
    generated_tokens = content_tokens(generated)

    if not expected_tokens:
        return {"overlap": 0.0, "recalled": False}

    matched = expected_tokens & generated_tokens
    overlap = len(matched) / len(expected_tokens)

    return {
        "overlap": round(overlap, 3),
        "recalled": overlap >= FACT_OVERLAP_THRESHOLD,
    }


def build_entity_facts(keyed_pairs: list[dict]) -> dict[str, list[dict]]:
    """Group facts by entity from keyed_pairs."""
    entities = defaultdict(list)
    for kp in keyed_pairs:
        entity = kp.get("source_subject", "")
        if not entity or len(entity) <= 1:
            continue
        entities[entity].append(
            {
                "key": kp.get("key", ""),
                "question": kp.get("question", ""),
                "answer": kp.get("answer", ""),
            }
        )
    return dict(entities)


def find_test8_cycles(model_name: str) -> list[dict]:
    """Find all Test 8 cycle checkpoints with adapters and keyed_pairs."""
    model_dir = TEST8_BASE / model_name
    if not model_dir.exists():
        logger.error("Test 8 output not found at %s", model_dir)
        return []

    run_dirs = sorted(model_dir.iterdir())
    if not run_dirs:
        return []
    run_dir = run_dirs[-1]

    cycles = []
    for cycle_dir in sorted(run_dir.glob("cycle_*")):
        kp_path = cycle_dir / "keyed_pairs.json"
        adapter_path = cycle_dir / "adapter" / "episodic" / "adapter_config.json"
        registry_path = cycle_dir / "simhash_registry.json"
        if kp_path.exists() and adapter_path.exists():
            cycle_num = int(cycle_dir.name.replace("cycle_", ""))
            with open(kp_path) as f:
                keyed_pairs = json.load(f)
            registry = {}
            if registry_path.exists():
                with open(registry_path) as f:
                    registry = json.load(f)
            cycles.append(
                {
                    "cycle": cycle_num,
                    "cycle_dir": cycle_dir,
                    "adapter_dir": str(cycle_dir / "adapter"),
                    "keyed_pairs": keyed_pairs,
                    "registry": registry,
                    "key_count": len(keyed_pairs),
                }
            )

    return cycles


# --- Pass 1: Keyed retrieval (baseline) ---


def probe_keyed_retrieval(
    model,
    tokenizer,
    keyed_pairs: list[dict],
    registry: dict,
) -> dict:
    """Probe all keys using the standard keyed retrieval prompt."""
    exact_count = 0
    total = len(keyed_pairs)

    for kp in keyed_pairs:
        key = kp.get("key", "")
        result = probe_key(model, tokenizer, key, registry=registry)
        if result and "failure_reason" not in result:
            exact_count += 1

    return {
        "exact_count": exact_count,
        "total": total,
        "recall_rate": round(exact_count / total, 4) if total > 0 else 0.0,
    }


# --- Pass 2: Direct questions ---


def probe_direct_questions(
    model,
    tokenizer,
    keyed_pairs: list[dict],
) -> dict:
    """Probe using the natural questions from keyed_pairs (no key prefix)."""
    results = []

    for kp in keyed_pairs:
        question = kp.get("question", "")
        expected = kp.get("answer", "")
        if not question or not expected:
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        generated = generate_answer(model, tokenizer, prompt, max_new_tokens=150, temperature=0.0)

        score = score_fact_overlap(generated, expected)
        results.append(
            {
                "key": kp.get("key", ""),
                "question": question,
                "expected_answer": expected,
                "generated": generated,
                **score,
            }
        )

    recalled = sum(1 for r in results if r["recalled"])
    total = len(results)
    mean_overlap = sum(r["overlap"] for r in results) / total if total > 0 else 0.0

    return {
        "recalled_count": recalled,
        "total": total,
        "recall_rate": round(recalled / total, 4) if total > 0 else 0.0,
        "mean_overlap": round(mean_overlap, 4),
        "per_key_results": results,
    }


# --- Pass 3: Open-ended per entity ---


def probe_open_ended(
    model,
    tokenizer,
    keyed_pairs: list[dict],
) -> dict:
    """Probe with "What do you know about {entity}?" per unique entity."""
    entity_facts = build_entity_facts(keyed_pairs)
    entity_results = []

    for entity, facts in sorted(entity_facts.items()):
        question = f"What do you know about {entity}?"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        generated = generate_answer(model, tokenizer, prompt, max_new_tokens=300, temperature=0.0)

        fact_scores = []
        for fact in facts:
            score = score_fact_overlap(generated, fact["answer"])
            fact_scores.append(
                {
                    "key": fact["key"],
                    "expected_answer": fact["answer"],
                    **score,
                }
            )

        facts_recalled = sum(1 for f in fact_scores if f["recalled"])
        total_facts = len(fact_scores)
        mean_overlap = (
            sum(f["overlap"] for f in fact_scores) / total_facts if total_facts > 0 else 0.0
        )

        entity_results.append(
            {
                "entity": entity,
                "question": question,
                "generated": generated,
                "total_facts": total_facts,
                "facts_recalled": facts_recalled,
                "recall_rate": round(facts_recalled / total_facts, 4) if total_facts > 0 else 0.0,
                "mean_overlap": round(mean_overlap, 4),
                "fact_results": fact_scores,
            }
        )

    # Aggregate
    total_facts = sum(r["total_facts"] for r in entity_results)
    total_recalled = sum(r["facts_recalled"] for r in entity_results)
    entity_count = len(entity_results)
    entities_with_recall = sum(1 for r in entity_results if r["facts_recalled"] > 0)
    aggregate_overlap = (
        sum(r["mean_overlap"] * r["total_facts"] for r in entity_results) / total_facts
        if total_facts > 0
        else 0.0
    )

    return {
        "entity_count": entity_count,
        "total_facts": total_facts,
        "facts_recalled": total_recalled,
        "fact_recall_rate": round(total_recalled / total_facts, 4) if total_facts > 0 else 0.0,
        "entity_hit_rate": round(entities_with_recall / entity_count, 4)
        if entity_count > 0
        else 0.0,
        "mean_overlap": round(aggregate_overlap, 4),
        "entity_results": entity_results,
    }


# --- Main test runner ---


def _find_resume_dir(model_name: str) -> Path | None:
    """Find the Test 9 output directory with the most completed cycles.

    Results may be spread across multiple timestamped dirs (from separate
    resume runs). Pick the one with the most results so new cycles land
    alongside the majority.
    """
    model_dir = OUTPUT_BASE / model_name
    if not model_dir.exists():
        return None
    run_dirs = sorted(model_dir.iterdir())
    if not run_dirs:
        return None
    best_dir = None
    best_count = -1
    for d in run_dirs:
        count = len(list(d.glob("cycle_*_results.json")))
        if count > best_count:
            best_count = count
            best_dir = d
    return best_dir


def _load_completed_cycles(output_dir: Path) -> dict[int, dict]:
    """Load completed cycle results from ALL run dirs under this model.

    Merges results across timestamped dirs so resume sees everything,
    regardless of which dir results landed in.
    """
    completed = {}
    # Scan all sibling run dirs, not just output_dir
    model_dir = output_dir.parent
    for run_dir in model_dir.iterdir():
        for result_file in run_dir.glob("cycle_*_results.json"):
            try:
                with open(result_file) as f:
                    data = json.load(f)
                cycle_num = data.get("cycle")
                if cycle_num is not None:
                    completed[cycle_num] = data
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def run_test(
    model_name: str,
    cycle_filter: list[int] | None = None,
    resume: bool = False,
):
    """Run Test 9 across all Test 8 cycle checkpoints."""
    cycles = find_test8_cycles(model_name)
    if not cycles:
        logger.error("No Test 8 cycles found for model %s", model_name)
        return

    if cycle_filter:
        cycles = [c for c in cycles if c["cycle"] in cycle_filter]
        logger.info("Filtered to %d cycles: %s", len(cycles), cycle_filter)

    # Resume: reuse existing output dir and skip completed cycles
    completed_results = {}
    if resume:
        resume_dir = _find_resume_dir(model_name)
        if resume_dir:
            completed_results = _load_completed_cycles(resume_dir)
            output_dir = resume_dir
            skippable = [c for c in cycles if c["cycle"] in completed_results]
            cycles = [c for c in cycles if c["cycle"] not in completed_results]
            logger.info(
                "Resuming: %d cycles already done, %d remaining",
                len(skippable),
                len(cycles),
            )
        else:
            logger.info("No previous run found, starting fresh")
            output_dir = model_output_dir(OUTPUT_BASE, model_name)
    else:
        output_dir = model_output_dir(OUTPUT_BASE, model_name)

    if not cycles:
        logger.info("All cycles already completed")
        return

    logger.info("Test 9: %d cycles, model %s", len(cycles), model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base model once
    logger.info("Loading base model: %s", model_name)
    model_config = BENCHMARK_MODELS[model_name]
    model, tokenizer = load_base_model(model_config)
    model.gradient_checkpointing_disable()

    emergence_curve = []
    start_time = time.time()

    for i, cycle_info in enumerate(cycles):
        cycle_num = cycle_info["cycle"]
        key_count = cycle_info["key_count"]
        adapter_dir = cycle_info["adapter_dir"]
        keyed_pairs = cycle_info["keyed_pairs"]
        registry = cycle_info["registry"]

        entity_facts = build_entity_facts(keyed_pairs)
        entity_count = len(entity_facts)

        logger.info(
            "Cycle %d (%d/%d): %d keys, %d entities",
            cycle_num,
            i + 1,
            len(cycles),
            key_count,
            entity_count,
        )

        # Swap adapter
        if isinstance(model, PeftModel):
            model.delete_adapter("episodic")
            if not model.peft_config:
                model = model.base_model.model

        model = load_adapter(model, adapter_dir, "episodic")
        if isinstance(model, PeftModel):
            switch_adapter(model, "episodic")

        cycle_start = time.time()

        # Pass 1: Keyed retrieval
        logger.info("  Pass 1: Keyed retrieval (%d keys)...", key_count)
        keyed_result = probe_keyed_retrieval(model, tokenizer, keyed_pairs, registry)
        logger.info(
            "    Keyed: %d/%d (%.1f%%)",
            keyed_result["exact_count"],
            keyed_result["total"],
            keyed_result["recall_rate"] * 100,
        )

        # Pass 2: Direct questions
        logger.info("  Pass 2: Direct questions (%d keys)...", key_count)
        direct_result = probe_direct_questions(model, tokenizer, keyed_pairs)
        logger.info(
            "    Direct: %d/%d (%.1f%%), overlap %.3f",
            direct_result["recalled_count"],
            direct_result["total"],
            direct_result["recall_rate"] * 100,
            direct_result["mean_overlap"],
        )

        # Pass 3: Open-ended
        logger.info("  Pass 3: Open-ended (%d entities)...", entity_count)
        open_result = probe_open_ended(model, tokenizer, keyed_pairs)
        logger.info(
            "    Open: %d/%d facts (%.1f%%), %d/%d entities hit (%.1f%%)",
            open_result["facts_recalled"],
            open_result["total_facts"],
            open_result["fact_recall_rate"] * 100,
            int(open_result["entity_hit_rate"] * entity_count),
            entity_count,
            open_result["entity_hit_rate"] * 100,
        )

        cycle_elapsed = time.time() - cycle_start

        cycle_result = {
            "cycle": cycle_num,
            "key_count": key_count,
            "entity_count": entity_count,
            "elapsed_seconds": round(cycle_elapsed, 1),
            "keyed_retrieval": keyed_result,
            "direct_question": {k: v for k, v in direct_result.items() if k != "per_key_results"},
            "open_ended": {k: v for k, v in open_result.items() if k != "entity_results"},
            "direct_question_detail": direct_result["per_key_results"],
            "open_ended_detail": open_result["entity_results"],
        }

        emergence_curve.append(
            {
                "cycle": cycle_num,
                "key_count": key_count,
                "entity_count": entity_count,
                "keyed_recall": keyed_result["recall_rate"],
                "direct_recall": direct_result["recall_rate"],
                "open_fact_recall": open_result["fact_recall_rate"],
                "open_entity_hit": open_result["entity_hit_rate"],
            }
        )

        logger.info("  Cycle %d complete in %.1fs", cycle_num, cycle_elapsed)

        # Save per-cycle results incrementally
        cycle_output = output_dir / f"cycle_{cycle_num:03d}_results.json"
        with open(cycle_output, "w") as f:
            json.dump(cycle_result, f, indent=2, ensure_ascii=False)

        # Check for pause signal between cycles
        if is_paused():
            logger.info(
                "\n  Pause requested. Stopping after cycle %d.\n"
                "  Resume with: python experiments/test9_natural_recall.py"
                " --model %s --resume\n",
                cycle_num,
                model_name,
            )
            break

    total_elapsed = time.time() - start_time

    # Merge completed results into emergence curve for full picture
    for cycle_num, data in sorted(completed_results.items()):
        keyed = data.get("keyed_retrieval", {})
        direct = data.get("direct_question", {})
        open_e = data.get("open_ended", {})
        emergence_curve.append(
            {
                "cycle": cycle_num,
                "key_count": data.get("key_count", 0),
                "entity_count": data.get("entity_count", 0),
                "keyed_recall": keyed.get("recall_rate", 0),
                "direct_recall": direct.get("recall_rate", 0),
                "open_fact_recall": open_e.get("fact_recall_rate", 0),
                "open_entity_hit": open_e.get("entity_hit_rate", 0),
            }
        )
    emergence_curve.sort(key=lambda x: x["cycle"])

    # Save summary
    all_cycles_count = len(emergence_curve)
    summary = {
        "test": "test9_natural_recall",
        "model": model_name,
        "probe_styles": [
            "keyed_retrieval (Recall the QA pair stored under key...)",
            "direct_question (natural question, no key prefix)",
            "open_ended (What do you know about {entity}?)",
        ],
        "fact_overlap_threshold": FACT_OVERLAP_THRESHOLD,
        "system_prompt": SYSTEM_PROMPT,
        "total_cycles": all_cycles_count,
        "new_cycles_this_run": len(cycles),
        "total_elapsed_seconds": round(total_elapsed, 1),
        "emergence_curve": emergence_curve,
    }

    summary_path = output_dir / "results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Results saved to %s", output_dir)

    # Print emergence curve
    print("\n=== Natural Recall Emergence ===\n")
    print(
        f"{'Cycle':>5} {'Keys':>5} {'Ent':>4} "
        f"{'Keyed':>7} {'Direct':>7} {'OpenFact':>9} {'OpenEnt':>8}"
    )
    print("-" * 55)
    for p in emergence_curve:
        print(
            f"{p['cycle']:>5} {p['key_count']:>5} {p['entity_count']:>4} "
            f"{p['keyed_recall']:>6.1%} {p['direct_recall']:>6.1%} "
            f"{p['open_fact_recall']:>8.1%} {p['open_entity_hit']:>7.1%}"
        )

    # Unload model
    if isinstance(model, PeftModel):
        model.delete_adapter("episodic")
        if not model.peft_config:
            model = model.base_model.model
    from paramem.models.loader import unload_model

    unload_model(model, tokenizer)


def main():
    parser = argparse.ArgumentParser(description="Test 9: Natural Recall Emergence")
    parser.add_argument(
        "--model",
        type=str,
        default="mistral",
        help="Model name (default: mistral)",
    )
    parser.add_argument(
        "--cycles",
        type=str,
        default=None,
        help="Comma-separated cycle numbers to test (default: all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last run, skipping completed cycles",
    )
    args = parser.parse_args()

    cycle_filter = None
    if args.cycles:
        cycle_filter = [int(c) for c in args.cycles.split(",")]

    from experiments.utils.gpu_guard import acquire_gpu

    with acquire_gpu():
        run_test(args.model, cycle_filter, resume=args.resume)


if __name__ == "__main__":
    main()
