"""Phase 4: RAG Baseline — compare retrieval-augmented generation against parametric memory.

Builds a RAG index over the same 55 session transcripts used in Phase 3b,
evaluates on the same promoted/oneshot fact probes, and compares head-to-head
against Phase 3b parametric results (loaded from outputs/phase3b/recall_probe.json).

Usage:
    python experiments/phase4_rag_baseline.py
    python experiments/phase4_rag_baseline.py --top-k 5
    python experiments/phase4_rag_baseline.py --output-dir outputs/phase4
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
from paramem.evaluation.rag import RAGPipeline, format_rag_prompt  # noqa: E402
from paramem.evaluation.recall import generate_answer  # noqa: E402
from paramem.graph.qa_generator import generate_qa_from_relations  # noqa: E402
from paramem.models.loader import load_base_model  # noqa: E402
from paramem.utils.config import load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Same facts as Phase 3 recall probe — ground truth
PROMOTED_FACTS = [
    {"subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"},
    {"subject": "Alex", "predicate": "works_at", "object": "AutoMate"},
    {"subject": "Alex", "predicate": "has_pet", "object": "Luna the German Shepherd"},
    {"subject": "Alex", "predicate": "prefers", "object": "Python"},
    {"subject": "Alex", "predicate": "prefers", "object": "dark mode"},
    {"subject": "Alex", "predicate": "studies_at", "object": "KIT"},
    {"subject": "Alex", "predicate": "knows", "object": "Jonas"},
    {"subject": "Alex", "predicate": "prefers", "object": "black coffee"},
    {"subject": "Alex", "predicate": "has_hobby", "object": "hiking in the Black Forest"},
    {"subject": "Maria", "predicate": "manages", "object": "robotics team budget"},
]

ONESHOT_FACTS = [
    {"subject": "Alex", "predicate": "visited", "object": "Barcelona last weekend"},
    {"subject": "Alex", "predicate": "read", "object": "a paper on transformers"},
    {"subject": "Alex", "predicate": "fixed", "object": "a bug in the CI pipeline"},
    {"subject": "Alex", "predicate": "attended", "object": "a meetup on Rust"},
    {"subject": "Alex", "predicate": "bought", "object": "new running shoes"},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4: RAG Baseline")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve per query (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/phase4",
        help="Output directory (default: outputs/phase4)",
    )
    parser.add_argument(
        "--phase3b-probe",
        type=str,
        default="outputs/phase3b/recall_probe.json",
        help="Path to Phase 3b recall probe results for comparison",
    )
    return parser.parse_args()


def _aggregate_scores(scores):
    """Compute mean of scores, defaulting to 0 if empty."""
    return sum(scores) / len(scores) if scores else 0.0


def _probe_qa_pairs(model, tokenizer, qa_pairs, prompt_fn):
    """Probe QA pairs using given prompt generator function.

    Args:
        model: The model to use for generation.
        tokenizer: The tokenizer.
        qa_pairs: List of {question, answer} dicts.
        prompt_fn: Function(qa_dict) -> prompt_str that generates the prompt.

    Returns dict with mean_embedding and details list.
    """
    model.gradient_checkpointing_disable()

    results = []
    for qa in qa_pairs:
        prompt = prompt_fn(qa)
        generated = generate_answer(model, tokenizer, prompt, temperature=0.1)
        emb_score = compute_similarity(qa["answer"], generated)

        results.append(
            {
                "question": qa["question"],
                "expected": qa["answer"],
                "generated": generated,
                "embedding_score": emb_score,
            }
        )

    scores = [r["embedding_score"] for r in results]
    return {
        "mean_embedding": _aggregate_scores(scores),
        "details": results,
    }


def probe_rag(model, tokenizer, pipeline, qa_pairs, top_k):
    """Probe RAG recall: retrieve context, generate answer, score."""

    def make_rag_prompt(qa):
        contexts = pipeline.retrieve_texts(qa["question"], top_k=top_k)
        return format_rag_prompt(qa["question"], contexts, tokenizer)

    results = _probe_qa_pairs(model, tokenizer, qa_pairs, make_rag_prompt)
    for i, qa in enumerate(qa_pairs):
        results["details"][i]["retrieved_contexts"] = pipeline.retrieve_texts(
            qa["question"], top_k=top_k
        )
    return results


def probe_base_no_context(model, tokenizer, qa_pairs):
    """Probe base model without any context (control)."""
    from paramem.training.dataset import _format_inference_prompt

    def make_base_prompt(qa):
        return _format_inference_prompt(qa["question"], tokenizer)

    return _probe_qa_pairs(model, tokenizer, qa_pairs, make_base_prompt)


def load_phase3b_results(path):
    """Load Phase 3b recall probe results for comparison."""
    probe_path = project_root / path
    if not probe_path.exists():
        logger.warning("Phase 3b probe not found at %s — skipping comparison", probe_path)
        return None
    with open(probe_path) as f:
        return json.load(f)


def main():
    args = parse_args()
    config = load_config()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load session transcripts
    sessions_path = project_root / "data" / "sessions" / "synthetic_sessions.json"
    with open(sessions_path) as f:
        sessions = json.load(f)
    logger.info("Loaded %d session transcripts", len(sessions))

    # Build RAG index
    logger.info("Building RAG index (top_k=%d)...", args.top_k)
    pipeline = RAGPipeline()
    pipeline.build_index(sessions)

    # Load model (base only, no adapters)
    logger.info("Loading base model...")
    model, tokenizer = load_base_model(config.model)

    # Generate QA pairs
    promoted_qa = generate_qa_from_relations(PROMOTED_FACTS)
    oneshot_qa = generate_qa_from_relations(ONESHOT_FACTS)
    logger.info("Probing %d promoted QA, %d oneshot QA", len(promoted_qa), len(oneshot_qa))

    # 1. Base model (no context) — control
    logger.info("=== Base model (no context) ===")
    base_promoted = probe_base_no_context(model, tokenizer, promoted_qa)
    base_oneshot = probe_base_no_context(model, tokenizer, oneshot_qa)

    # 2. RAG (base model + retrieved context)
    logger.info("=== RAG (base model + retrieved context, top_k=%d) ===", args.top_k)
    rag_promoted = probe_rag(model, tokenizer, pipeline, promoted_qa, args.top_k)
    rag_oneshot = probe_rag(model, tokenizer, pipeline, oneshot_qa, args.top_k)

    # Load Phase 3b parametric results
    phase3b = load_phase3b_results(args.phase3b_probe)

    # Print comparison
    print("\n" + "=" * 72)
    print("Phase 4: RAG Baseline vs Parametric Memory")
    print(f"RAG top_k={args.top_k}, {len(pipeline.chunks)} chunks from {len(sessions)} sessions")
    print("=" * 72)

    fmt = "{:<40s} {:>10s}"
    print(fmt.format("Condition", "Embed"))
    print("-" * 52)
    print(fmt.format("Base (promoted, no context)", f"{base_promoted['mean_embedding']:.1%}"))
    print(fmt.format("Base (oneshot, no context)", f"{base_oneshot['mean_embedding']:.1%}"))
    print(fmt.format("RAG (promoted)", f"{rag_promoted['mean_embedding']:.1%}"))
    print(fmt.format("RAG (oneshot)", f"{rag_oneshot['mean_embedding']:.1%}"))

    if phase3b:
        print()
        print(fmt.format("Phase 3b semantic (promoted)", f"{phase3b['semantic_promoted_emb']:.1%}"))
        print(fmt.format("Phase 3b episodic (promoted)", f"{phase3b['episodic_promoted_emb']:.1%}"))

    rag_delta = rag_promoted["mean_embedding"] - base_promoted["mean_embedding"]
    print()
    print(f"RAG delta vs base (promoted):        {rag_delta:+.1%}")
    if phase3b:
        sem_delta = phase3b["semantic_promoted_emb"] - phase3b["base_promoted_emb"]
        epi_delta = phase3b["episodic_promoted_emb"] - phase3b["base_promoted_emb"]
        print(f"Phase 3b semantic delta vs base:     {sem_delta:+.1%}")
        print(f"Phase 3b episodic delta vs base:     {epi_delta:+.1%}")
        rag_vs_sem = rag_promoted["mean_embedding"] - phase3b["semantic_promoted_emb"]
        print(f"RAG vs semantic (promoted):          {rag_vs_sem:+.1%}")

    # Per-fact breakdown
    print()
    print("Per-fact RAG results (promoted):")
    print("-" * 72)
    for detail in rag_promoted["details"]:
        status = "OK" if detail["embedding_score"] >= 0.8 else "  "
        print(f"  [{status}] {detail['question']}")
        print(f"       Expected:  {detail['expected']}")
        print(f"       Generated: {detail['generated'][:80]}")
        print(f"       Score:     {detail['embedding_score']:.3f}")

    print("=" * 72)

    # Save results
    results = {
        "top_k": args.top_k,
        "num_sessions": len(sessions),
        "num_chunks": len(pipeline.chunks),
        "base_promoted_emb": base_promoted["mean_embedding"],
        "base_oneshot_emb": base_oneshot["mean_embedding"],
        "rag_promoted_emb": rag_promoted["mean_embedding"],
        "rag_oneshot_emb": rag_oneshot["mean_embedding"],
        "rag_delta_promoted": rag_delta,
        "details": {
            "base_promoted": base_promoted["details"],
            "base_oneshot": base_oneshot["details"],
            "rag_promoted": rag_promoted["details"],
            "rag_oneshot": rag_oneshot["details"],
        },
    }

    if phase3b:
        results["phase3b_comparison"] = {
            "semantic_promoted_emb": phase3b["semantic_promoted_emb"],
            "episodic_promoted_emb": phase3b["episodic_promoted_emb"],
            "base_promoted_emb": phase3b["base_promoted_emb"],
        }

    results_path = output_dir / "rag_baseline.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
