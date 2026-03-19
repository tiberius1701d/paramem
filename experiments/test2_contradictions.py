"""Test 2: Contradiction Resolution — temporal fact updates.

Tests whether the system resolves contradictions when facts change over time.
Two strategies compared:
  2a) Graph-only: predicate normalization catches exact predicate matches
  2b) Graph + model: LLM semantic reasoning catches synonym predicates

Flow per strategy:
  1. Train on initial facts (sessions 1-3)
  2. New sessions arrive with contradicting facts
  3. Extract graph from new session → merge into cumulative graph
     (contradiction resolution happens during merge)
  4. Generate QA from resolved graph → retrain
  5. Verify: model recalls NEW (current) version, not old

Usage:
    python experiments/test2_contradictions.py
    python experiments/test2_contradictions.py --model gemma
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
    add_model_args,
    evaluate_indexed_recall,
    get_benchmark_models,
    load_model_and_config,
    model_output_dir,
    save_results,
    setup_logging,
    train_indexed_keys,
)
from paramem.evaluation.embedding_scorer import compute_similarity  # noqa: E402
from paramem.graph.merger import GraphMerger  # noqa: E402
from paramem.graph.qa_generator import generate_qa_from_relations  # noqa: E402
from paramem.graph.schema import Entity, Relation, SessionGraph  # noqa: E402
from paramem.models.loader import unload_model  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

DATA_PATH = project_root / "data" / "synthetic" / "contradiction_sessions.json"
OUTPUT_DIR = project_root / "outputs" / "test2_contradictions"
MATCH_THRESHOLD = 0.75


def load_contradiction_data():
    """Load contradiction session data."""
    with open(DATA_PATH) as f:
        return json.load(f)


def build_session_graph(fact_chain_version, session_id):
    """Convert a fact chain version into a minimal SessionGraph for merging.

    Each version becomes a triple: (subject, predicate, answer).
    The question field encodes the predicate naturally.
    """
    from datetime import datetime, timezone

    # Parse the chain to extract a triple
    # The chain has question/answer pairs like:
    #   Q: "Where does Alex work?" A: "Alex works at SpaceX"
    # We need (Alex, works_at, SpaceX) as a relation
    entities = []
    relations = []

    subject = fact_chain_version.get("subject", "Alex")
    predicate = fact_chain_version.get("predicate", "related_to")
    obj = fact_chain_version["answer"]

    entities.append(
        Entity(
            name=subject,
            entity_type="person",
            attributes={},
        )
    )

    relations.append(
        Relation(
            subject=subject,
            predicate=predicate,
            object=obj,
            relation_type="factual",
            confidence=1.0,
        )
    )

    return SessionGraph(
        session_id=session_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        entities=entities,
        relations=relations,
        summary=f"{subject} {predicate} {obj}",
    )


def prepare_contradiction_data(fact_chains):
    """Prepare fact chains with explicit subject/predicate for graph merging.

    Returns enriched chains with subject, predicate per version.
    """
    # Map question patterns to predicates
    predicate_map = {
        "Where does Alex work?": "works_at",
        "Where does Alex live?": "lives_in",
        "Is Alex in a relationship?": "relationship_status",
        "Does Alex have a pet?": "has_pets",
        "What project is Alex working on?": "works_on_project",
        "What is Alex's main hobby?": "main_hobby",
        "What car does Alex drive?": "drives",
        "What is Alex's diet?": "dietary_preference",
        "Who is Alex's manager?": "reports_to",
        "What language is Alex learning?": "learning_language",
    }

    enriched = []
    for chain in fact_chains:
        predicate = predicate_map.get(chain["question"], "related_to")
        enriched_chain = {
            "id": chain["id"],
            "question": chain["question"],
            "subject": "Alex",
            "predicate": predicate,
            "versions": [],
        }
        for version in chain["versions"]:
            enriched_chain["versions"].append(
                {
                    **version,
                    "subject": "Alex",
                    "predicate": predicate,
                }
            )
        enriched.append(enriched_chain)

    return enriched


def run_strategy(
    strategy,
    model,
    tokenizer,
    enriched_chains,
    args,
    output_dir,
):
    """Run one contradiction resolution strategy.

    Returns dict with results per evaluation point.
    """
    logger.info("Running strategy: %s", strategy)

    merger = GraphMerger(
        strategy=strategy,
        model=model if strategy == "model" else None,
        tokenizer=tokenizer if strategy == "model" else None,
    )

    # Collect all sessions in order
    all_versions = []
    for chain in enriched_chains:
        for version in chain["versions"]:
            all_versions.append(
                {
                    **version,
                    "chain_id": chain["id"],
                    "question": chain["question"],
                }
            )
    all_versions.sort(key=lambda v: v["session"])

    # Group by session
    sessions = {}
    for v in all_versions:
        sessions.setdefault(v["session"], []).append(v)

    eval_results = []
    total_start = time.time()

    for session_num in sorted(sessions.keys()):
        session_facts = sessions[session_num]
        logger.info(
            "Session %d: merging %d facts (strategy=%s)",
            session_num,
            len(session_facts),
            strategy,
        )

        # Disable gradient checkpointing before any model inference
        # (merge may call detect_contradiction_with_model for "model" strategy,
        # and QA generation runs model inference). Training re-enables it,
        # so this must happen at the top of every session.
        model.gradient_checkpointing_disable()

        # Merge each fact into the cumulative graph
        for fact in session_facts:
            session_graph = build_session_graph(
                fact,
                f"session_{session_num}",
            )
            merger.merge(session_graph)

        # Generate QA from the resolved graph
        triples = merger.get_all_triples()
        qa_pairs = generate_qa_from_relations(
            [{"subject": s, "predicate": p, "object": o} for s, p, o in triples],
            model=model,
            tokenizer=tokenizer,
        )

        if not qa_pairs:
            logger.warning("No QA pairs from graph after session %d", session_num)
            continue

        if len(qa_pairs) != len(triples):
            logger.warning(
                "QA count mismatch session %d: %d triples → %d QA pairs",
                session_num, len(triples), len(qa_pairs),
            )

        # Unwrap to base model before each session's fresh adapter
        adapter_name = f"episodic_s{session_num}_{strategy}"
        from peft import PeftModel as _PeftModel

        if isinstance(model, _PeftModel):
            model = model.base_model.model

        model, keyed_pairs, registry, train_time, metrics = train_indexed_keys(
            model,
            tokenizer,
            qa_pairs,
            epochs=args.num_epochs,
            rank=args.rank,
            adapter_name=adapter_name,
            output_dir=output_dir / f"session_{session_num}",
            run_name=f"contradictions-{strategy}-s{session_num}",
            skip_distill=True,
        )

        # Evaluate recall
        recall_result = evaluate_indexed_recall(
            model,
            tokenizer,
            keyed_pairs,
            registry,
            adapter_name=adapter_name,
        )

        # Check: does each recalled answer match the CURRENT version?
        expected_current = {}
        for chain in enriched_chains:
            latest = None
            for v in chain["versions"]:
                if v["session"] <= session_num:
                    latest = v
            if latest:
                expected_current[chain["id"]] = latest["answer"]

        # Map chains to recalled answers via predicate (deterministic).
        # Each keyed_pair carries source_predicate from QA generation,
        # and each chain has a unique predicate. This avoids threshold-
        # sensitive embedding matching entirely.
        predicate_to_chain = {
            chain["predicate"]: chain["id"] for chain in enriched_chains
        }

        # Build key → recalled answer lookup
        key_to_recalled = {}
        for kr in recall_result["per_key"]:
            if kr.get("recalled") and kr["recalled"].get("answer"):
                key_to_recalled[kr["key"]] = kr["recalled"]["answer"]

        # Build key → predicate lookup from keyed_pairs (in-memory, has metadata)
        key_to_predicate = {}
        for kp in keyed_pairs:
            if "source_predicate" in kp:
                key_to_predicate[kp["key"]] = kp["source_predicate"]

        per_chain = {}
        for chain in enriched_chains:
            chain_id = chain["id"]
            current_answer = expected_current.get(chain_id, "")
            if not current_answer:
                continue

            # Find the recalled answer for this chain's predicate
            recalled_answer = ""
            for key, pred in key_to_predicate.items():
                if predicate_to_chain.get(pred) == chain_id:
                    recalled_answer = key_to_recalled.get(key, "")
                    break

            sim_to_current = (
                compute_similarity(current_answer, recalled_answer)
                if recalled_answer
                else 0.0
            )
            per_chain[chain_id] = {
                "current_answer": current_answer,
                "recalled_answer": recalled_answer,
                "similarity_to_current": sim_to_current,
                "is_current": sim_to_current > MATCH_THRESHOLD,
            }

        returns_current = sum(1 for v in per_chain.values() if v["is_current"])
        total_chains = len(per_chain)

        eval_results.append(
            {
                "session": session_num,
                "graph_nodes": merger.graph.number_of_nodes(),
                "graph_edges": merger.graph.number_of_edges(),
                "qa_pairs": len(qa_pairs),
                "recall": recall_result["exact_count"],
                "total_keys": recall_result["total"],
                "returns_current": returns_current,
                "total_chains": total_chains,
                "train_time": train_time,
                "per_chain": per_chain,
                "per_key_recall": recall_result["per_key"],
            }
        )

        print(
            f"  Session {session_num} ({strategy}): "
            f"recall={recall_result['exact_count']}/{recall_result['total']}, "
            f"current={returns_current}/{total_chains}, "
            f"graph={merger.graph.number_of_nodes()}n/{merger.graph.number_of_edges()}e"
        )

    total_time = time.time() - total_start
    return {
        "strategy": strategy,
        "sessions": eval_results,
        "contradictions_resolved": merger.contradictions_resolved,
        "total_time_seconds": total_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Test 2: Contradiction Resolution")
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["graph", "model"],
        help="Run a single strategy (default: both)",
    )
    add_model_args(parser)
    args = parser.parse_args()

    base_output_dir = Path(args.output_dir)
    data = load_contradiction_data()
    enriched_chains = prepare_contradiction_data(data["fact_chains"])

    for bench_name, bench_model_config in get_benchmark_models(args):
        print(f"\n{'=' * 72}")
        print(f"  Model: {bench_name} ({bench_model_config.model_id})")
        print(f"{'=' * 72}")

        model, tokenizer, config = load_model_and_config(bench_model_config)
        output_dir = model_output_dir(base_output_dir, bench_name)

        all_results = {
            "experiment": "test2_contradictions",
            "model": bench_name,
            "model_id": bench_model_config.model_id,
            "epochs": args.num_epochs,
            "rank": args.rank,
            "strategies": {},
        }

        strategies = [args.strategy] if args.strategy else ["graph", "model"]
        for strategy in strategies:
            print(f"\n--- Strategy: {strategy} ---")
            result = run_strategy(
                strategy,
                model,
                tokenizer,
                enriched_chains,
                args,
                output_dir / strategy,
            )
            all_results["strategies"][strategy] = result

            # Phase save after each strategy
            save_results(all_results, output_dir)

        # Summary comparison
        print(f"\n{'=' * 72}")
        print(f"CONTRADICTION RESOLUTION SUMMARY ({bench_name})")
        print(f"{'=' * 72}")
        for strategy, result in all_results["strategies"].items():
            n_resolved = len(result["contradictions_resolved"])
            graph_resolved = sum(
                1 for c in result["contradictions_resolved"] if c["method"] == "graph"
            )
            model_resolved = sum(
                1 for c in result["contradictions_resolved"] if c["method"] == "model"
            )
            final = result["sessions"][-1] if result["sessions"] else {}
            print(
                f"  {strategy:>5}: "
                f"contradictions={n_resolved} "
                f"(graph={graph_resolved}, model={model_resolved}), "
                f"final current={final.get('returns_current', 0)}"
                f"/{final.get('total_chains', 0)}, "
                f"time={result['total_time_seconds']:.0f}s"
            )
        print("=" * 72)

        save_results(all_results, output_dir)
        unload_model(model, tokenizer)


if __name__ == "__main__":
    main()
