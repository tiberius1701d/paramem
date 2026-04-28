"""Test 4 resume: load saved adapter, re-evaluate recall, run RAG baseline, save results.

Resumes from a completed Test 4 reinforcement run where the adapter and
registry were saved but results.json was not (e.g., crash in RAG section).

Usage:
    python experiments/test4_resume_rag.py \
        --run-dir outputs/test4_reinforcement/mistral/20260318_074617
"""

import argparse
import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.utils.test_harness import (  # noqa: E402
    evaluate_indexed_recall,
    load_model_and_config,
    save_results,
    setup_logging,
)
from paramem.evaluation.embedding_scorer import compute_similarity  # noqa: E402
from paramem.evaluation.rag_qa import QARAGPipeline  # noqa: E402
from paramem.evaluation.recall import generate_answer  # noqa: E402
from paramem.models.loader import load_adapter, unload_model  # noqa: E402
from paramem.training.indexed_memory import assign_keys  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

DATA_PATH = project_root / "data" / "synthetic" / "reinforcement_sessions.json"


def main():
    parser = argparse.ArgumentParser(description="Test 4 Resume: RAG baseline")
    parser.add_argument(
        "--run-dir", type=str, required=True, help="Path to the incomplete run directory"
    )
    parser.add_argument("--model", type=str, default="mistral")
    parser.add_argument("--rag-only", action="store_true", help="Skip recall re-eval, run RAG only")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    # Find the last session directory
    session_dirs = sorted(run_dir.glob("session_*"), key=lambda p: int(p.name.split("_")[1]))
    if not session_dirs:
        print(f"No session directories found in {run_dir}")
        return
    last_session = session_dirs[-1]
    session_num = int(last_session.name.split("_")[1])
    print(f"Resuming from {last_session} (session {session_num})")

    # Load data
    with open(DATA_PATH) as f:
        data = json.load(f)

    all_facts = {}
    for group_name in ["reinforced", "mentioned_twice", "single_mention"]:
        for fact in data["facts"][group_name]:
            all_facts[fact["id"]] = {
                "question": fact["question"],
                "answer": fact["answer"],
                "group": group_name,
                "sessions": fact.get("sessions", []),
                "mention_count": len(fact.get("sessions", [])),
            }

    # Load distilled keyed_pairs saved during training
    keyed_pairs_path = last_session / "keyed_pairs.json"
    if keyed_pairs_path.exists():
        with open(keyed_pairs_path) as f:
            keyed_pairs = json.load(f)
        print(f"Loaded distilled keyed_pairs: {len(keyed_pairs)} pairs")
    else:
        # Fallback: reconstruct from raw data (will NOT match trained QA — recall will be wrong)
        logger.warning(
            "No keyed_pairs.json found — falling back to raw data reconstruction. "
            "Recall evaluation will be invalid."
        )
        cumulative_qa = {}
        for session_n in range(1, session_num + 1):
            for group_name in ["reinforced", "mentioned_twice", "single_mention"]:
                for fact in data["facts"][group_name]:
                    if session_n in fact.get("sessions", []):
                        cumulative_qa[fact["id"]] = {
                            "question": fact["question"],
                            "answer": fact["answer"],
                        }
        qa_list = list(cumulative_qa.values())
        keyed_pairs = assign_keys(qa_list)
        print(f"WARNING: Using raw QA pairs (not distilled): {len(keyed_pairs)}")

    # Load registry
    registry_path = last_session / "simhash_registry.json"
    with open(registry_path) as f:
        registry = json.load(f)
    print(f"Registry: {len(registry)} keys")

    # Load model + adapter

    # Resolve model config
    from paramem.server.config import MODEL_REGISTRY

    bench_model_config = MODEL_REGISTRY[args.model]
    model, tokenizer = load_model_and_config(bench_model_config)

    # load_adapter expects the parent dir; it appends adapter_name
    adapter_dir = last_session / "adapter"
    adapter_full = adapter_dir / "episodic"
    if not (adapter_full / "adapter_config.json").exists():
        # Last resort: find it
        import glob

        pattern = str(adapter_dir / "**" / "adapter_config.json")
        hits = glob.glob(pattern, recursive=True)
        if hits:
            # Derive parent dir from found path
            adapter_dir = Path(hits[0]).parent.parent
        else:
            print(f"ERROR: Cannot find adapter under {adapter_dir}")
            return
    print(f"Loading adapter from: {adapter_dir}")
    model = load_adapter(model, str(adapter_dir), "episodic")
    print(f"Adapter loaded from {adapter_dir}")

    # Phase 1: Re-evaluate recall (skip if --rag-only)
    recall_result = None
    per_key = []
    exact = 0
    total = len(keyed_pairs)

    if not getattr(args, "rag_only", False):
        print("\n--- Re-evaluating recall ---")
        recall_result = evaluate_indexed_recall(
            model, tokenizer, keyed_pairs, registry, adapter_name="episodic"
        )
        per_key = recall_result["per_key"]
        exact = recall_result["exact_count"]
        total = recall_result["total"]
        print(f"Recall: {exact}/{total}")
        print(f"Mean confidence: {recall_result['mean_confidence']:.3f}")
    else:
        print("\n--- Skipping recall re-evaluation (--rag-only) ---")

    # Phase 2: RAG baseline
    print("\n--- RAG baseline ---")
    all_qa = [{"question": v["question"], "answer": v["answer"]} for v in all_facts.values()]
    rag = QARAGPipeline()
    rag.build_index(all_qa)

    from peft import PeftModel as _PeftModel

    model.gradient_checkpointing_disable()

    rag_per_group = {g: [] for g in ["reinforced", "mentioned_twice", "single_mention"]}

    def _run_rag_probes():
        for fact_id, fact_info in all_facts.items():
            prompt = rag.format_prompt(fact_info["question"], tokenizer, top_k=3)
            generated = generate_answer(
                model,
                tokenizer,
                prompt,
                max_new_tokens=200,
                temperature=0.0,
            )
            similarity = compute_similarity(fact_info["answer"], generated)
            rag_per_group[fact_info["group"]].append(
                {
                    "fact_id": fact_id,
                    "question": fact_info["question"],
                    "expected": fact_info["answer"],
                    "generated": generated,
                    "similarity": similarity,
                }
            )

    if isinstance(model, _PeftModel):
        with model.disable_adapter():
            _run_rag_probes()
    else:
        _run_rag_probes()

    rag_summary = {}
    for group, items in rag_per_group.items():
        if not items:
            continue
        mean_sim = sum(r["similarity"] for r in items) / len(items)
        rag_summary[group] = {"mean_similarity": mean_sim, "per_fact": items}
        print(f"  {group:>20}: mean_sim={mean_sim:.3f} ({len(items)} facts)")

    # Save complete results
    results = {
        "experiment": "test4_reinforcement",
        "model": args.model,
        "model_id": bench_model_config.model_id,
        "epochs_per_cycle": 30,
        "rank": 8,
        "total_time_seconds": None,
        "note": "Resumed from saved adapter — recall re-evaluated, RAG baseline added",
        "final_recall": {
            "session": session_num,
            "cumulative_facts": len(keyed_pairs),
            "exact_count": exact,
            "total": total,
            "mean_confidence": recall_result["mean_confidence"] if recall_result else None,
            "per_key": per_key if per_key else None,
            "source": "re-evaluated" if recall_result else "skipped (--rag-only)",
        },
        "rag_baseline": rag_summary,
        "fact_metadata": {
            k: {"group": v["group"], "mention_count": v["mention_count"]}
            for k, v in all_facts.items()
        },
    }

    save_results(results, run_dir)
    print(f"\nResults saved to {run_dir}")

    unload_model(model, tokenizer)


if __name__ == "__main__":
    main()
