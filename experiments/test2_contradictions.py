"""Test 2: Contradiction Resolution — temporal fact updates.

Tests whether parametric memory naturally resolves contradictions when facts
change over time, compared to RAG which stores all versions and may return
stale facts.

10 fact chains with 3 temporal versions each across 10 sessions.

Usage:
    python experiments/test2_contradictions.py
    python experiments/test2_contradictions.py --num-epochs 20
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.utils.test_harness import (  # noqa: E402
    IndexedDataset,
    load_model_and_config,
    save_results,
    setup_logging,
)

setup_logging()
logger = logging.getLogger(__name__)

DATA_PATH = project_root / "data" / "synthetic" / "contradiction_sessions.json"
OUTPUT_DIR = project_root / "outputs" / "test2_contradictions"


def load_contradiction_data():
    """Load contradiction session data."""
    with open(DATA_PATH) as f:
        return json.load(f)


def get_current_facts(fact_chains, up_to_session):
    """Get the most recent version of each fact at a given session.

    Returns list of {"question", "answer", "chain_id", "version_session"}.
    """
    current = []
    for chain in fact_chains:
        latest = None
        for version in chain["versions"]:
            if version["session"] <= up_to_session:
                latest = version
        if latest is not None:
            current.append(
                {
                    "question": chain["question"],
                    "answer": latest["answer"],
                    "chain_id": chain["id"],
                    "version_session": latest["session"],
                }
            )
    return current


def get_all_versions(fact_chains):
    """Get ALL versions of all facts (for RAG indexing)."""
    all_qa = []
    for chain in fact_chains:
        for version in chain["versions"]:
            all_qa.append(
                {
                    "question": chain["question"],
                    "answer": version["answer"],
                    "chain_id": chain["id"],
                    "session": version["session"],
                }
            )
    return all_qa


def main():
    parser = argparse.ArgumentParser(description="Test 2: Contradiction Resolution")
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--skip-rag", action="store_true")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    data = load_contradiction_data()
    fact_chains = data["fact_chains"]

    model, tokenizer, config = load_model_and_config()

    from paramem.evaluation.embedding_scorer import compute_similarity  # noqa: E402
    from paramem.models.loader import create_adapter, switch_adapter  # noqa: E402
    from paramem.training.indexed_memory import (  # noqa: E402
        assign_keys,
        build_registry,
        format_indexed_training,
        probe_all_keys,
        validate_recall,
    )
    from paramem.training.trainer import train_adapter  # noqa: E402
    from paramem.utils.config import AdapterConfig, TrainingConfig  # noqa: E402

    adapter_config = AdapterConfig(
        rank=args.rank,
        alpha=args.rank * 2,
        learning_rate=1e-4,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        dropout=0.0,
    )
    training_config = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=2,
        max_seq_length=1024,
        num_epochs=args.num_epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=42,
    )

    model = create_adapter(model, adapter_config, "episodic")

    # Sessions where fact changes occur (evaluate after these)
    change_sessions = set()
    for chain in fact_chains:
        for version in chain["versions"]:
            change_sessions.add(version["session"])
    eval_sessions = sorted(change_sessions)

    session_results = []
    total_start = time.time()

    for session_num in range(1, 11):
        # Get current facts at this point in time
        current_facts = get_current_facts(fact_chains, session_num)
        if not current_facts:
            continue

        # Train on current facts only (most recent versions)
        qa_list = [{"question": f["question"], "answer": f["answer"]} for f in current_facts]
        keyed_pairs = assign_keys(qa_list)
        registry = build_registry(keyed_pairs)

        examples = format_indexed_training(keyed_pairs, tokenizer, max_length=1024)
        dataset = IndexedDataset(examples)

        # Reinitialize adapter each session (simulating retrain with updated facts)
        if session_num > 1:
            model.delete_adapter("episodic")
            model = create_adapter(model, adapter_config, "episodic")

        logger.info("Session %d: training on %d current facts", session_num, len(qa_list))

        cycle_start = time.time()
        metrics = train_adapter(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            adapter_name="episodic",
            training_config=training_config,
            adapter_config=adapter_config,
            output_dir=output_dir / f"session_{session_num}" / "adapter",
            run_name=f"contradictions-session-{session_num}",
        )

        # Evaluate: does the adapter return the CURRENT version?
        if session_num in eval_sessions:
            model.gradient_checkpointing_disable()
            switch_adapter(model, "episodic")

            trained_keys = [kp["key"] for kp in keyed_pairs]
            recalled = probe_all_keys(model, tokenizer, trained_keys, registry=registry)

            per_chain = {}
            for i, fact in enumerate(current_facts):
                kp = keyed_pairs[i]
                result = validate_recall(recalled[kp["key"]], kp, registry)

                # Also check semantic similarity to current answer
                recalled_answer = ""
                if result["recalled"] is not None:
                    recalled_answer = result["recalled"].get("answer", "")
                similarity_to_current = (
                    compute_similarity(fact["answer"], recalled_answer) if recalled_answer else 0.0
                )

                per_chain[fact["chain_id"]] = {
                    "exact_match": result["exact_match"],
                    "confidence": result["confidence"],
                    "similarity_to_current": similarity_to_current,
                    "current_answer": fact["answer"],
                    "recalled_answer": recalled_answer,
                    "version_session": fact["version_session"],
                }

            cycle_time = time.time() - cycle_start
            exact = sum(1 for v in per_chain.values() if v["exact_match"])
            mean_sim = (
                sum(v["similarity_to_current"] for v in per_chain.values()) / len(per_chain)
                if per_chain
                else 0.0
            )

            session_results.append(
                {
                    "session": session_num,
                    "facts_count": len(current_facts),
                    "exact_recall": exact,
                    "total": len(per_chain),
                    "mean_similarity_to_current": mean_sim,
                    "train_loss": metrics.get("train_loss"),
                    "wall_clock_seconds": cycle_time,
                    "per_chain": per_chain,
                }
            )

            print(
                f"  Session {session_num}: {exact}/{len(per_chain)} exact, "
                f"sim_to_current={mean_sim:.3f}"
            )

    # RAG comparison: index ALL versions, query with current questions
    rag_result = None
    if not args.skip_rag:
        from paramem.evaluation.rag_qa import QARAGPipeline  # noqa: E402
        from paramem.evaluation.recall import generate_answer  # noqa: E402

        logger.info("Running RAG baseline with all fact versions indexed...")
        all_versions = get_all_versions(fact_chains)
        rag = QARAGPipeline()
        rag.build_index(all_versions)

        # Disable adapters for RAG generation
        model.disable_adapter_layers()
        model.gradient_checkpointing_disable()

        # Query with each chain's question — which version does RAG return?
        final_facts = get_current_facts(fact_chains, 10)
        rag_per_chain = {}

        for fact in final_facts:
            prompt = rag.format_prompt(fact["question"], tokenizer, top_k=3)
            generated = generate_answer(
                model,
                tokenizer,
                prompt,
                max_new_tokens=150,
                temperature=0.1,
                repetition_penalty=1.3,
            )

            # Check similarity to current (latest) vs earliest version
            sim_current = compute_similarity(fact["answer"], generated)

            # Find earliest version for this chain
            chain = next(c for c in fact_chains if c["id"] == fact["chain_id"])
            earliest_answer = chain["versions"][0]["answer"]
            sim_earliest = compute_similarity(earliest_answer, generated)

            rag_per_chain[fact["chain_id"]] = {
                "question": fact["question"],
                "current_answer": fact["answer"],
                "earliest_answer": earliest_answer,
                "rag_generated": generated,
                "similarity_to_current": sim_current,
                "similarity_to_earliest": sim_earliest,
                "returns_current": sim_current > sim_earliest,
            }

        model.enable_adapter_layers()

        returns_current = sum(1 for v in rag_per_chain.values() if v["returns_current"])
        rag_result = {
            "returns_current_count": returns_current,
            "total": len(rag_per_chain),
            "per_chain": rag_per_chain,
        }

        print(f"\n  RAG returns current version: {returns_current}/{len(rag_per_chain)}")

    total_time = time.time() - total_start

    # Summary
    print("\n" + "=" * 72)
    print("CONTRADICTION RESOLUTION SUMMARY")
    print("=" * 72)
    for sr in session_results:
        print(
            f"  Session {sr['session']:>2}: "
            f"{sr['exact_recall']}/{sr['total']} exact, "
            f"sim={sr['mean_similarity_to_current']:.3f}"
        )
    if rag_result:
        print(
            f"\n  RAG current-version accuracy: "
            f"{rag_result['returns_current_count']}/{rag_result['total']}"
        )
    print(f"  Total time: {total_time:.0f}s")
    print("=" * 72)

    results = {
        "experiment": "test2_contradictions",
        "epochs": args.num_epochs,
        "rank": args.rank,
        "total_time_seconds": total_time,
        "parametric_sessions": session_results,
        "rag_baseline": rag_result,
    }

    save_results(results, output_dir)


if __name__ == "__main__":
    main()
