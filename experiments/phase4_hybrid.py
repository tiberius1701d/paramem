"""Phase 4: Hybrid RAG+LoRA — combine retrieval with adapted model.

Tests whether RAG retrieval + LoRA adaptation is complementary or redundant.
Evaluates all conditions: base, RAG, parametric (from Phase 3b), hybrid.

Usage:
    python experiments/phase4_hybrid.py
    python experiments/phase4_hybrid.py --top-k 5
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
from paramem.models.loader import load_base_model, switch_adapter  # noqa: E402
from paramem.training.dataset import _format_inference_prompt  # noqa: E402
from paramem.utils.config import load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Same facts as Phase 3 recall probe
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
    parser = argparse.ArgumentParser(description="Phase 4: Hybrid RAG+LoRA")
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
        "--phase3b-dir",
        type=str,
        default="outputs/phase3b",
        help="Phase 3b output directory with adapters and recall probe",
    )
    return parser.parse_args()


def _probe(model, tokenizer, qa_pairs, prompt_fn):
    """Probe model recall with a given prompt formatting function.

    prompt_fn: (question) -> formatted prompt string
    """
    model.gradient_checkpointing_disable()

    results = []
    for qa in qa_pairs:
        prompt = prompt_fn(qa["question"])
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
        "mean_embedding": sum(scores) / len(scores) if scores else 0.0,
        "details": results,
    }


def probe_condition(model, tokenizer, qa_pairs, prompt_fn, adapter_name=None):
    """Probe a specific condition (base, adapter, or hybrid)."""
    from peft import PeftModel

    if adapter_name and isinstance(model, PeftModel):
        model.enable_adapter_layers()
        switch_adapter(model, adapter_name)
    elif isinstance(model, PeftModel):
        model.disable_adapter_layers()

    return _probe(model, tokenizer, qa_pairs, prompt_fn)


def main():
    args = parse_args()
    config = load_config()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    phase3b_dir = project_root / args.phase3b_dir

    # Load session transcripts and build RAG index
    sessions_path = project_root / "data" / "sessions" / "synthetic_sessions.json"
    with open(sessions_path) as f:
        sessions = json.load(f)
    logger.info("Loaded %d session transcripts", len(sessions))

    pipeline = RAGPipeline()
    pipeline.build_index(sessions)

    # Load model with Phase 3b adapters
    logger.info("Loading base model...")
    model, tokenizer = load_base_model(config.model)

    # Find final cycle directory for adapter loading
    cycle_dirs = [d for d in phase3b_dir.iterdir() if d.is_dir() and d.name.startswith("cycle_")]
    if not cycle_dirs:
        logger.error("No cycle directories found in %s", phase3b_dir)
        sys.exit(1)

    final_cycle = max(int(d.name.split("_")[1]) for d in cycle_dirs)
    cycle_dir = phase3b_dir / f"cycle_{final_cycle}"
    logger.info("Loading adapters from cycle %d", final_cycle)

    from peft import PeftModel

    model = PeftModel.from_pretrained(
        model,
        str(cycle_dir / "episodic" / "episodic"),
        adapter_name="episodic",
    )
    model.load_adapter(
        str(cycle_dir / "semantic" / "semantic"),
        adapter_name="semantic",
    )

    # Generate QA pairs
    promoted_qa = generate_qa_from_relations(PROMOTED_FACTS)
    oneshot_qa = generate_qa_from_relations(ONESHOT_FACTS)
    logger.info("Probing %d promoted, %d oneshot QA pairs", len(promoted_qa), len(oneshot_qa))

    # Prompt formatters
    def no_context_prompt(question):
        return _format_inference_prompt(question, tokenizer)

    def rag_prompt(question):
        contexts = pipeline.retrieve_texts(question, top_k=args.top_k)
        return format_rag_prompt(question, contexts, tokenizer)

    # === Run all conditions ===

    conditions = {}

    # 1. Base model, no context
    logger.info("=== Base (no context) ===")
    conditions["base"] = {
        "promoted": probe_condition(model, tokenizer, promoted_qa, no_context_prompt),
        "oneshot": probe_condition(model, tokenizer, oneshot_qa, no_context_prompt),
    }

    # 2. RAG (base model + context)
    logger.info("=== RAG (base + context) ===")
    conditions["rag"] = {
        "promoted": probe_condition(model, tokenizer, promoted_qa, rag_prompt),
        "oneshot": probe_condition(model, tokenizer, oneshot_qa, rag_prompt),
    }

    # 3. Parametric only (episodic adapter, no context)
    logger.info("=== Episodic (no context) ===")
    conditions["episodic"] = {
        "promoted": probe_condition(
            model, tokenizer, promoted_qa, no_context_prompt, adapter_name="episodic"
        ),
        "oneshot": probe_condition(
            model, tokenizer, oneshot_qa, no_context_prompt, adapter_name="episodic"
        ),
    }

    # 4. Parametric only (semantic adapter, no context)
    logger.info("=== Semantic (no context) ===")
    conditions["semantic"] = {
        "promoted": probe_condition(
            model, tokenizer, promoted_qa, no_context_prompt, adapter_name="semantic"
        ),
        "oneshot": probe_condition(
            model, tokenizer, oneshot_qa, no_context_prompt, adapter_name="semantic"
        ),
    }

    # 5. Hybrid: RAG + episodic adapter
    logger.info("=== Hybrid RAG + Episodic ===")
    conditions["hybrid_episodic"] = {
        "promoted": probe_condition(
            model, tokenizer, promoted_qa, rag_prompt, adapter_name="episodic"
        ),
        "oneshot": probe_condition(
            model, tokenizer, oneshot_qa, rag_prompt, adapter_name="episodic"
        ),
    }

    # 6. Hybrid: RAG + semantic adapter
    logger.info("=== Hybrid RAG + Semantic ===")
    conditions["hybrid_semantic"] = {
        "promoted": probe_condition(
            model, tokenizer, promoted_qa, rag_prompt, adapter_name="semantic"
        ),
        "oneshot": probe_condition(
            model, tokenizer, oneshot_qa, rag_prompt, adapter_name="semantic"
        ),
    }

    # === Print comparison ===

    print("\n" + "=" * 72)
    print("Phase 4: Hybrid RAG+LoRA — All Conditions")
    print(f"RAG top_k={args.top_k}, adapters from cycle {final_cycle}")
    print("=" * 72)

    fmt = "{:<35s} {:>12s} {:>12s}"
    print(fmt.format("Condition", "Promoted", "Oneshot"))
    print("-" * 62)

    for name, data in conditions.items():
        print(
            fmt.format(
                name,
                f"{data['promoted']['mean_embedding']:.1%}",
                f"{data['oneshot']['mean_embedding']:.1%}",
            )
        )

    # Deltas
    base_p = conditions["base"]["promoted"]["mean_embedding"]
    print()
    print("Deltas vs base (promoted):")
    for name, data in conditions.items():
        if name == "base":
            continue
        delta = data["promoted"]["mean_embedding"] - base_p
        print(f"  {name:<33s} {delta:+.1%}")

    # Complementarity check
    rag_p = conditions["rag"]["promoted"]["mean_embedding"]
    epi_p = conditions["episodic"]["promoted"]["mean_embedding"]
    sem_p = conditions["semantic"]["promoted"]["mean_embedding"]
    hyb_epi_p = conditions["hybrid_episodic"]["promoted"]["mean_embedding"]
    hyb_sem_p = conditions["hybrid_semantic"]["promoted"]["mean_embedding"]

    print()
    print("Complementarity analysis:")
    print(f"  RAG alone:              {rag_p:.1%}")
    print(f"  Episodic alone:         {epi_p:.1%}")
    print(f"  Hybrid RAG+Episodic:    {hyb_epi_p:.1%}  (vs max of parts: {max(rag_p, epi_p):.1%})")
    print(f"  Semantic alone:         {sem_p:.1%}")
    print(f"  Hybrid RAG+Semantic:    {hyb_sem_p:.1%}  (vs max of parts: {max(rag_p, sem_p):.1%})")

    complementary = hyb_epi_p > max(rag_p, epi_p) or hyb_sem_p > max(rag_p, sem_p)
    print(f"  Complementary:          {'YES' if complementary else 'NO'}")
    print("=" * 72)

    # === Save results ===

    results = {
        "top_k": args.top_k,
        "final_cycle": final_cycle,
        "num_sessions": len(sessions),
        "num_chunks": len(pipeline.chunks),
        "conditions": {},
    }
    for name, data in conditions.items():
        results["conditions"][name] = {
            "promoted_emb": data["promoted"]["mean_embedding"],
            "oneshot_emb": data["oneshot"]["mean_embedding"],
            "promoted_details": data["promoted"]["details"],
            "oneshot_details": data["oneshot"]["details"],
        }
    results["complementary"] = complementary

    results_path = output_dir / "hybrid_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
