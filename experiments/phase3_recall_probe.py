"""Post-experiment recall probing for Phase 3.

Loads the final model state and measures:
1. Promoted memory retention (semantic adapter)
2. Episodic decay (episodic adapter on unreinforced facts)
3. Semantic drift (semantic adapter stability)

Uses embedding similarity (sentence-transformers) for scoring,
with keyword overlap reported alongside for comparison.

Usage:
    python experiments/phase3_recall_probe.py                           # default: outputs/phase3b
    python experiments/phase3_recall_probe.py --output-dir outputs/phase3  # probe original run
    python experiments/phase3_recall_probe.py --save-as baseline  # recall_probe_baseline.json
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
os.environ["WANDB_MODE"] = "disabled"

from paramem.evaluation.embedding_scorer import compute_similarity  # noqa: E402
from paramem.evaluation.recall import _score_pair, generate_answer  # noqa: E402
from paramem.graph.qa_generator import generate_qa_from_relations  # noqa: E402
from paramem.models.loader import load_base_model, switch_adapter  # noqa: E402
from paramem.training.dataset import _format_inference_prompt  # noqa: E402
from paramem.utils.config import load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Known facts from the synthetic data (ground truth)
PROMOTED_FACTS = [
    {"subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"},
    {"subject": "Alex", "predicate": "works_at", "object": "AutoMate"},
    {
        "subject": "Alex",
        "predicate": "has_pet",
        "object": "Luna the German Shepherd",
    },
    {"subject": "Alex", "predicate": "prefers", "object": "Python"},
    {"subject": "Alex", "predicate": "prefers", "object": "dark mode"},
    {"subject": "Alex", "predicate": "studies_at", "object": "KIT"},
    {"subject": "Alex", "predicate": "knows", "object": "Jonas"},
    {"subject": "Alex", "predicate": "prefers", "object": "black coffee"},
    {
        "subject": "Alex",
        "predicate": "has_hobby",
        "object": "hiking in the Black Forest",
    },
    {
        "subject": "Maria",
        "predicate": "manages",
        "object": "robotics team budget",
    },
]

ONESHOT_FACTS = [
    {
        "subject": "Alex",
        "predicate": "visited",
        "object": "Barcelona last weekend",
    },
    {
        "subject": "Alex",
        "predicate": "read",
        "object": "a paper on transformers",
    },
    {
        "subject": "Alex",
        "predicate": "fixed",
        "object": "a bug in the CI pipeline",
    },
    {"subject": "Alex", "predicate": "attended", "object": "a meetup on Rust"},
    {"subject": "Alex", "predicate": "bought", "object": "new running shoes"},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3 Recall Probe")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/phase3b",
        help="Directory containing cycle outputs (default: outputs/phase3b)",
    )
    parser.add_argument(
        "--save-as",
        type=str,
        default=None,
        help="Save results as recall_probe_<save-as>.json (default: recall_probe.json)",
    )
    return parser.parse_args()


def _probe(model, tokenizer, qa_pairs: list[dict]) -> dict:
    """Probe model recall, returning both embedding and keyword scores."""
    model.gradient_checkpointing_disable()

    results = []
    for qa in qa_pairs:
        prompt = _format_inference_prompt(qa["question"], tokenizer)
        generated = generate_answer(model, tokenizer, prompt, temperature=0.1)
        emb_score = compute_similarity(qa["answer"], generated)
        kw_score = _score_pair(qa["answer"], generated, method="keyword")

        results.append(
            {
                "question": qa["question"],
                "expected": qa["answer"],
                "generated": generated,
                "embedding_score": emb_score,
                "keyword_score": kw_score,
            }
        )

    emb_scores = [r["embedding_score"] for r in results]
    kw_scores = [r["keyword_score"] for r in results]

    return {
        "mean_embedding": sum(emb_scores) / len(emb_scores) if emb_scores else 0.0,
        "mean_keyword": sum(kw_scores) / len(kw_scores) if kw_scores else 0.0,
        "details": results,
    }


def probe_adapter(model, tokenizer, qa_pairs, adapter_name):
    """Probe a specific adapter."""
    switch_adapter(model, adapter_name)
    return _probe(model, tokenizer, qa_pairs)


def probe_base(model, tokenizer, qa_pairs):
    """Probe the base model (adapters disabled)."""
    model.disable_adapter_layers()
    result = _probe(model, tokenizer, qa_pairs)
    model.enable_adapter_layers()
    return result


def main():
    args = parse_args()
    config = load_config()
    output_dir = project_root / args.output_dir

    logger.info("Loading model with final adapters from %s...", output_dir)
    model, tokenizer = load_base_model(config.model)

    # Load final cycle adapters
    cycle_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("cycle_")]
    if not cycle_dirs:
        logger.error("No cycle directories found in %s", output_dir)
        sys.exit(1)

    final_cycle = max(int(d.name.split("_")[1]) for d in cycle_dirs)
    logger.info("Loading adapters from cycle %d", final_cycle)

    from peft import PeftModel

    cycle_dir = output_dir / f"cycle_{final_cycle}"
    model = PeftModel.from_pretrained(
        model,
        str(cycle_dir / "episodic" / "episodic"),
        adapter_name="episodic",
    )
    model.load_adapter(
        str(cycle_dir / "semantic" / "semantic"),
        adapter_name="semantic",
    )

    # Generate QA pairs for probing
    promoted_qa = generate_qa_from_relations(PROMOTED_FACTS)
    oneshot_qa = generate_qa_from_relations(ONESHOT_FACTS)
    logger.info(
        "Probing %d promoted QA, %d oneshot QA",
        len(promoted_qa),
        len(oneshot_qa),
    )

    # 1. Base model recall (control)
    logger.info("=== Base model recall ===")
    base_promoted = probe_base(model, tokenizer, promoted_qa)
    base_oneshot = probe_base(model, tokenizer, oneshot_qa)

    # 2. Semantic adapter
    logger.info("=== Semantic adapter recall ===")
    sem_promoted = probe_adapter(model, tokenizer, promoted_qa, "semantic")
    sem_oneshot = probe_adapter(model, tokenizer, oneshot_qa, "semantic")

    # 3. Episodic adapter
    logger.info("=== Episodic adapter recall ===")
    epi_promoted = probe_adapter(model, tokenizer, promoted_qa, "episodic")
    epi_oneshot = probe_adapter(model, tokenizer, oneshot_qa, "episodic")

    # Report — embedding scores are primary, keyword for comparison
    sem_retention = sem_promoted["mean_embedding"]
    decay_delta = base_oneshot["mean_embedding"] - epi_oneshot["mean_embedding"]
    drift = abs(sem_promoted["mean_embedding"] - epi_promoted["mean_embedding"])
    delta = sem_promoted["mean_embedding"] - base_promoted["mean_embedding"]

    print("\n" + "=" * 60)
    print("Recall Probe (embedding similarity)")
    print(f"Adapters from: {output_dir}")
    print("=" * 60)
    fmt = "{:<32s} {:>8s}  {:>8s}"
    print(fmt.format("Condition", "Embed", "Keyword"))
    print("-" * 52)
    print(
        fmt.format(
            "Base (promoted)",
            f"{base_promoted['mean_embedding']:.1%}",
            f"{base_promoted['mean_keyword']:.1%}",
        )
    )
    print(
        fmt.format(
            "Base (oneshot)",
            f"{base_oneshot['mean_embedding']:.1%}",
            f"{base_oneshot['mean_keyword']:.1%}",
        )
    )
    print(
        fmt.format(
            "Semantic (promoted)",
            f"{sem_promoted['mean_embedding']:.1%}",
            f"{sem_promoted['mean_keyword']:.1%}",
        )
    )
    print(
        fmt.format(
            "Semantic (oneshot)",
            f"{sem_oneshot['mean_embedding']:.1%}",
            f"{sem_oneshot['mean_keyword']:.1%}",
        )
    )
    print(
        fmt.format(
            "Episodic (promoted)",
            f"{epi_promoted['mean_embedding']:.1%}",
            f"{epi_promoted['mean_keyword']:.1%}",
        )
    )
    print(
        fmt.format(
            "Episodic (oneshot)",
            f"{epi_oneshot['mean_embedding']:.1%}",
            f"{epi_oneshot['mean_keyword']:.1%}",
        )
    )
    print()
    print(f"Semantic delta vs base:   {delta:+.1%}")
    print(
        f"Promoted retention:       {sem_retention:.1%}"
        f"  {'PASS' if sem_retention >= 0.80 else 'FAIL'} (target >80%)"
    )
    print(
        f"Episodic decay delta:     {decay_delta:.1%}"
        f"  {'PASS' if decay_delta > 0 else 'NEEDS ANALYSIS'}"
    )
    print(
        f"Semantic drift:           {drift:.1%}  {'PASS' if drift < 0.05 else 'FAIL'} (target <5%)"
    )
    print("=" * 60)

    # Save results
    probe_results = {
        "source_dir": str(output_dir),
        "final_cycle": final_cycle,
        "scoring_method": "embedding (all-MiniLM-L6-v2)",
        "base_promoted_emb": base_promoted["mean_embedding"],
        "base_promoted_kw": base_promoted["mean_keyword"],
        "base_oneshot_emb": base_oneshot["mean_embedding"],
        "base_oneshot_kw": base_oneshot["mean_keyword"],
        "semantic_promoted_emb": sem_promoted["mean_embedding"],
        "semantic_promoted_kw": sem_promoted["mean_keyword"],
        "semantic_oneshot_emb": sem_oneshot["mean_embedding"],
        "semantic_oneshot_kw": sem_oneshot["mean_keyword"],
        "episodic_promoted_emb": epi_promoted["mean_embedding"],
        "episodic_promoted_kw": epi_promoted["mean_keyword"],
        "episodic_oneshot_emb": epi_oneshot["mean_embedding"],
        "episodic_oneshot_kw": epi_oneshot["mean_keyword"],
        "targets": {
            "promoted_retention": sem_retention,
            "decay_delta": decay_delta,
            "semantic_drift": drift,
        },
        "details": {
            "base_promoted": base_promoted["details"],
            "base_oneshot": base_oneshot["details"],
            "semantic_promoted": sem_promoted["details"],
            "semantic_oneshot": sem_oneshot["details"],
            "episodic_promoted": epi_promoted["details"],
            "episodic_oneshot": epi_oneshot["details"],
        },
    }

    if args.save_as:
        save_path = output_dir / f"recall_probe_{args.save_as}.json"
    else:
        save_path = output_dir / "recall_probe.json"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(probe_results, f, indent=2)
    logger.info("Results saved to %s", save_path)


if __name__ == "__main__":
    main()
