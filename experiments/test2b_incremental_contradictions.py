"""Test 2b: Incremental Contradiction Resolution — forgetting under cumulative training.

Unlike Test 2 (fresh adapter per session), this test uses a SINGLE persistent
adapter across all sessions, matching the production consolidation pattern.

Three phases:
  A) Learning: Train initial facts cumulatively until stable recall.
  B) Contradiction: Introduce fact changes via graph. New keys for updated facts,
     old keys marked stale in registry.
  C) Decay: Continue training on current facts. Measure:
     - New key recall (should improve)
     - Stale key recall (should decay) — diagnostic probe bypassing registry
     - Control fact recall (should remain stable)

The symmetry hypothesis: if it takes N cycles to learn, it takes ~N cycles
for stale weight residue to decay.

Usage:
    python experiments/test2b_incremental_contradictions.py --model gemma
    python experiments/test2b_incremental_contradictions.py --model mistral
    python experiments/test2b_incremental_contradictions.py --model gemma --smoke
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
    add_model_args,
    get_benchmark_models,
    load_model_and_config,
    model_output_dir,
    save_results,
    setup_logging,
)
from paramem.evaluation.embedding_scorer import compute_similarity  # noqa: E402
from paramem.graph.merger import GraphMerger  # noqa: E402
from paramem.graph.qa_generator import generate_qa_from_relations  # noqa: E402
from paramem.graph.schema import Entity, Relation, SessionGraph  # noqa: E402
from paramem.models.loader import (  # noqa: E402
    create_adapter,
    switch_adapter,
    unload_model,
)
from paramem.training.indexed_memory import (  # noqa: E402
    assign_keys,
    build_enriched_registry,
    build_registry,
    format_indexed_training,
    mark_stale,
    probe_key,
    save_registry,
)
from paramem.training.trainer import train_adapter  # noqa: E402
from paramem.utils.config import AdapterConfig, TrainingConfig  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

DATA_PATH = project_root / "data" / "synthetic" / "contradiction_sessions.json"
OUTPUT_DIR = project_root / "outputs" / "test2b_incremental"

# Control facts that never change — baseline for catastrophic forgetting detection
CONTROL_FACTS = [
    {
        "question": "What is Alex's nationality?",
        "answer": "Alex is German.",
        "subject": "Alex",
        "predicate": "nationality",
    },
    {
        "question": "What is Alex's degree?",
        "answer": "Alex has a master's degree in computer science.",
        "subject": "Alex",
        "predicate": "education",
    },
    {
        "question": "What is Alex's favorite color?",
        "answer": "Alex's favorite color is blue.",
        "subject": "Alex",
        "predicate": "favorite_color",
    },
]

# Map questions to predicates (same as Test 2)
PREDICATE_MAP = {
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


def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)


def build_session_graph(subject, predicate, answer, session_id):
    """Build a minimal SessionGraph from a single triple."""
    from datetime import datetime, timezone

    return SessionGraph(
        session_id=session_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        entities=[Entity(name=subject, entity_type="person", attributes={})],
        relations=[
            Relation(
                subject=subject,
                predicate=predicate,
                object=answer,
                relation_type="factual",
                confidence=1.0,
            )
        ],
        summary=f"{subject} {predicate} {answer}",
    )


def get_current_facts(merger, control_facts):
    """Get all current facts from the graph + control facts as QA pairs."""
    triples = merger.get_all_triples()
    qa_pairs = generate_qa_from_relations(
        [{"subject": s, "predicate": p, "object": o} for s, p, o in triples],
    )

    # Add control facts (always present, use template fallback)
    for cf in control_facts:
        qa_pairs.append({"question": cf["question"], "answer": cf["answer"]})

    return qa_pairs


def train_on_current_facts(
    model,
    tokenizer,
    qa_pairs,
    adapter_name,
    epochs,
    rank,
    output_dir,
    cycle_num,
):
    """Train the existing adapter on current facts. Does NOT create a new adapter."""
    keyed_pairs = assign_keys(qa_pairs)
    registry = build_registry(keyed_pairs)

    examples = format_indexed_training(keyed_pairs, tokenizer, max_length=1024)
    dataset = IndexedDataset(examples)

    adapter_config = AdapterConfig(
        rank=rank,
        alpha=rank * 2,
        learning_rate=1e-4,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        dropout=0.0,
    )
    training_config = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=2,
        max_seq_length=1024,
        num_epochs=epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=42,
    )

    start = time.time()
    metrics = train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        adapter_name=adapter_name,
        training_config=training_config,
        adapter_config=adapter_config,
        output_dir=output_dir / f"cycle_{cycle_num}",
        run_name=f"test2b-cycle-{cycle_num}",
    )
    train_time = time.time() - start

    return keyed_pairs, registry, train_time, metrics


def probe_keys_diagnostic(
    model,
    tokenizer,
    keys,
    expected_answers,
    adapter_name,
    registry=None,
    bypass_registry=False,
):
    """Probe a set of keys and return per-key results with raw output.

    If bypass_registry=True, probes without registry check (for stale keys).
    """
    model.gradient_checkpointing_disable()
    switch_adapter(model, adapter_name)

    probe_registry = None if bypass_registry else registry
    results = []

    for key, expected in zip(keys, expected_answers):
        recalled = probe_key(
            model,
            tokenizer,
            key,
            registry=probe_registry,
            confidence_threshold=0.0 if bypass_registry else 0.75,
        )

        if recalled and "failure_reason" not in recalled:
            answer = recalled.get("answer", "")
            raw = recalled.get("raw_output", "")
            sim = compute_similarity(expected, answer) if answer else 0.0
            results.append(
                {
                    "key": key,
                    "expected": expected,
                    "recalled_answer": answer,
                    "raw_output": raw,
                    "similarity": sim,
                    "status": "recalled",
                }
            )
        elif recalled and "failure_reason" in recalled:
            results.append(
                {
                    "key": key,
                    "expected": expected,
                    "recalled_answer": "",
                    "raw_output": recalled.get("raw_output", ""),
                    "similarity": 0.0,
                    "status": f"failure:{recalled['failure_reason']}",
                }
            )
        else:
            results.append(
                {
                    "key": key,
                    "expected": expected,
                    "recalled_answer": "",
                    "raw_output": "",
                    "similarity": 0.0,
                    "status": "no_output",
                }
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Test 2b: Incremental Contradiction Resolution")
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--learning-cycles", type=int, default=3)
    parser.add_argument("--decay-cycles", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: 3 chains, 2 control, 2 learn cycles, 2 decay cycles",
    )
    add_model_args(parser)
    args = parser.parse_args()

    if args.smoke:
        args.learning_cycles = 2
        args.decay_cycles = 2
        args.num_epochs = 10

    data = load_data()
    fact_chains = data["fact_chains"]

    if args.smoke:
        fact_chains = fact_chains[:3]
        control_facts = CONTROL_FACTS[:2]
    else:
        control_facts = CONTROL_FACTS

    # Enrich chains with predicates
    enriched_chains = []
    for chain in fact_chains:
        predicate = PREDICATE_MAP.get(chain["question"], "related_to")
        enriched = {
            "id": chain["id"],
            "question": chain["question"],
            "subject": "Alex",
            "predicate": predicate,
            "versions": [
                {**v, "subject": "Alex", "predicate": predicate} for v in chain["versions"]
            ],
        }
        enriched_chains.append(enriched)

    base_output_dir = Path(args.output_dir)

    for bench_name, bench_model_config in get_benchmark_models(args):
        print(f"\n{'=' * 72}")
        print(f"  Test 2b: Incremental Contradictions — {bench_name}")
        print(f"  {'SMOKE TEST' if args.smoke else 'Full run'}")
        print(f"{'=' * 72}")

        model, tokenizer = load_model_and_config(bench_model_config)
        output_dir = model_output_dir(base_output_dir, bench_name)

        # Create persistent adapter (used throughout all phases)
        adapter_config = AdapterConfig(
            rank=args.rank,
            alpha=args.rank * 2,
            learning_rate=1e-4,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            dropout=0.0,
        )
        adapter_name = "episodic"
        model = create_adapter(model, adapter_config, adapter_name)

        # Initialize graph merger
        merger = GraphMerger(strategy="graph")

        # Add control facts to graph (they stay forever)
        for cf in control_facts:
            graph = build_session_graph(cf["subject"], cf["predicate"], cf["answer"], "control")
            merger.merge(graph)

        all_cycles = []
        enriched_registry = {}
        stale_keys_map = {}  # stale_key -> {"expected_old": str, "staled_at_cycle": int}
        current_keyed_pairs = []

        # ============================================================
        # Phase A: Learning — introduce initial facts
        # ============================================================
        print(f"\n--- Phase A: Learning ({args.learning_cycles} cycles) ---")

        # Add all version-1 facts to graph
        for chain in enriched_chains:
            v1 = chain["versions"][0]
            graph = build_session_graph(
                v1["subject"],
                v1["predicate"],
                v1["answer"],
                f"session_{v1['session']}",
            )
            merger.merge(graph)

        for cycle in range(1, args.learning_cycles + 1):
            print(f"\n  Cycle {cycle} (learning)")
            qa_pairs = get_current_facts(merger, control_facts)

            keyed_pairs, registry, train_time, metrics = train_on_current_facts(
                model,
                tokenizer,
                qa_pairs,
                adapter_name,
                args.num_epochs,
                args.rank,
                output_dir,
                cycle,
            )
            current_keyed_pairs = keyed_pairs
            enriched_registry = build_enriched_registry(
                keyed_pairs,
                session_id=f"learn_{cycle}",
                existing=enriched_registry,
            )

            # Probe current keys
            current_keys = [kp["key"] for kp in keyed_pairs]
            current_answers = [kp["answer"] for kp in keyed_pairs]
            results = probe_keys_diagnostic(
                model,
                tokenizer,
                current_keys,
                current_answers,
                adapter_name,
                registry,
            )

            recalled_count = sum(1 for r in results if r["similarity"] > 0.7)
            mean_sim = sum(r["similarity"] for r in results) / len(results) if results else 0.0

            cycle_data = {
                "cycle": cycle,
                "phase": "learning",
                "train_time": train_time,
                "train_loss": metrics.get("train_loss"),
                "total_keys": len(keyed_pairs),
                "current_recall": f"{recalled_count}/{len(results)}",
                "mean_similarity": mean_sim,
                "per_key": results,
                "stale_probes": [],
                "control_probes": [],
            }

            # Identify control keys for separate tracking
            control_keys = [
                kp["key"]
                for kp in keyed_pairs
                if kp["question"] in [cf["question"] for cf in control_facts]
            ]
            control_results = [r for r in results if r["key"] in control_keys]
            cycle_data["control_probes"] = control_results
            control_sim = (
                sum(r["similarity"] for r in control_results) / len(control_results)
                if control_results
                else 0.0
            )

            print(
                f"    recall={recalled_count}/{len(results)}, "
                f"mean_sim={mean_sim:.3f}, "
                f"control_sim={control_sim:.3f}, "
                f"loss={metrics.get('train_loss', -1):.4f}, "
                f"time={train_time:.0f}s"
            )

            all_cycles.append(cycle_data)

        # ============================================================
        # Phase B: Contradiction — introduce fact changes
        # ============================================================
        print("\n--- Phase B: Contradiction ---")

        # Record old keys before contradiction
        old_key_answers = {}
        for kp in current_keyed_pairs:
            old_key_answers[kp["key"]] = kp["answer"]

        # Apply version 2 of each chain to the graph
        stale_key_list = []
        for chain in enriched_chains:
            if len(chain["versions"]) < 2:
                continue
            v2 = chain["versions"][1]
            graph = build_session_graph(
                v2["subject"],
                v2["predicate"],
                v2["answer"],
                f"session_{v2['session']}",
            )
            merger.merge(graph)

        # Generate new QA pairs from resolved graph
        qa_pairs = get_current_facts(merger, control_facts)
        new_keyed_pairs = assign_keys(qa_pairs)

        # Determine which old keys are now stale (their content changed)
        new_answers = {kp["answer"] for kp in new_keyed_pairs}
        for old_key, old_answer in old_key_answers.items():
            # If old answer is not in any new QA pair, the fact was contradicted
            if not any(compute_similarity(old_answer, na) > 0.8 for na in new_answers):
                stale_key_list.append(old_key)
                stale_keys_map[old_key] = {
                    "expected_old": old_answer,
                    "staled_at_cycle": args.learning_cycles,
                }

        # Mark stale in enriched registry
        mark_stale(enriched_registry, stale_key_list)

        print(f"  Contradictions applied. {len(stale_key_list)} keys marked stale.")
        print(f"  Stale keys: {stale_key_list}")
        print(f"  New QA pairs: {len(qa_pairs)}")

        # ============================================================
        # Phase C: Decay — train on current facts, measure stale decay
        # ============================================================
        print(f"\n--- Phase C: Decay ({args.decay_cycles} cycles) ---")

        for cycle in range(
            args.learning_cycles + 1,
            args.learning_cycles + args.decay_cycles + 1,
        ):
            print(f"\n  Cycle {cycle} (decay)")
            qa_pairs = get_current_facts(merger, control_facts)

            keyed_pairs, registry, train_time, metrics = train_on_current_facts(
                model,
                tokenizer,
                qa_pairs,
                adapter_name,
                args.num_epochs,
                args.rank,
                output_dir,
                cycle,
            )
            current_keyed_pairs = keyed_pairs
            enriched_registry = build_enriched_registry(
                keyed_pairs,
                session_id=f"decay_{cycle}",
                existing=enriched_registry,
            )

            # Probe current (new) keys
            current_keys = [kp["key"] for kp in keyed_pairs]
            current_answers = [kp["answer"] for kp in keyed_pairs]
            current_results = probe_keys_diagnostic(
                model,
                tokenizer,
                current_keys,
                current_answers,
                adapter_name,
                registry,
            )

            recalled_count = sum(1 for r in current_results if r["similarity"] > 0.7)
            mean_sim = (
                sum(r["similarity"] for r in current_results) / len(current_results)
                if current_results
                else 0.0
            )

            # Probe stale keys — bypass registry, use old expected answers
            stale_results = []
            if stale_keys_map:
                stale_keys = list(stale_keys_map.keys())
                stale_expected = [stale_keys_map[k]["expected_old"] for k in stale_keys]
                stale_results = probe_keys_diagnostic(
                    model,
                    tokenizer,
                    stale_keys,
                    stale_expected,
                    adapter_name,
                    registry=None,
                    bypass_registry=True,
                )

            stale_mean = (
                sum(r["similarity"] for r in stale_results) / len(stale_results)
                if stale_results
                else 0.0
            )
            stale_recalled = sum(1 for r in stale_results if r["similarity"] > 0.7)

            # Control facts
            control_keys = [
                kp["key"]
                for kp in keyed_pairs
                if kp["question"] in [cf["question"] for cf in control_facts]
            ]
            control_results = [r for r in current_results if r["key"] in control_keys]
            control_sim = (
                sum(r["similarity"] for r in control_results) / len(control_results)
                if control_results
                else 0.0
            )

            cycle_data = {
                "cycle": cycle,
                "phase": "decay",
                "train_time": train_time,
                "train_loss": metrics.get("train_loss"),
                "total_keys": len(keyed_pairs),
                "current_recall": f"{recalled_count}/{len(current_results)}",
                "mean_similarity": mean_sim,
                "stale_recall": f"{stale_recalled}/{len(stale_results)}",
                "stale_mean_similarity": stale_mean,
                "control_mean_similarity": control_sim,
                "per_key": current_results,
                "stale_probes": stale_results,
                "control_probes": control_results,
            }

            print(
                f"    current={recalled_count}/{len(current_results)} "
                f"(sim={mean_sim:.3f}), "
                f"stale={stale_recalled}/{len(stale_results)} "
                f"(sim={stale_mean:.3f}), "
                f"control={control_sim:.3f}, "
                f"loss={metrics.get('train_loss', -1):.4f}, "
                f"time={train_time:.0f}s"
            )

            all_cycles.append(cycle_data)

        # ============================================================
        # Summary
        # ============================================================
        print(f"\n{'=' * 72}")
        print("TEST 2b SUMMARY")
        print(f"{'=' * 72}")
        print(f"  Model: {bench_name}")
        print(f"  Learning cycles: {args.learning_cycles}")
        print(f"  Decay cycles: {args.decay_cycles}")
        print(f"  Stale keys: {len(stale_keys_map)}")
        print()
        print(f"  {'Cycle':<8} {'Phase':<10} {'Current':<12} {'Stale':<12} {'Control':<10}")
        print(f"  {'-' * 52}")
        for c in all_cycles:
            stale_str = c.get("stale_recall", "n/a")
            control_str = f"{c.get('control_mean_similarity', 0):.3f}"
            print(
                f"  {c['cycle']:<8} {c['phase']:<10} "
                f"{c['current_recall']:<12} {stale_str:<12} {control_str:<10}"
            )
        print(f"{'=' * 72}")

        results = {
            "experiment": "test2b_incremental_contradictions",
            "model": bench_name,
            "model_id": bench_model_config.model_id,
            "epochs": args.num_epochs,
            "rank": args.rank,
            "learning_cycles": args.learning_cycles,
            "decay_cycles": args.decay_cycles,
            "num_chains": len(enriched_chains),
            "num_control": len(control_facts),
            "stale_keys": list(stale_keys_map.keys()),
            "stale_key_details": stale_keys_map,
            "enriched_registry": enriched_registry,
            "cycles": all_cycles,
        }

        save_results(results, output_dir)
        save_registry(enriched_registry, output_dir / "enriched_registry.json")

        unload_model(model, tokenizer)


if __name__ == "__main__":
    main()
