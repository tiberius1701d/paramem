"""Test 10: Grokking — Emergent Generalization Beyond Memorization.

Tests whether extended LoRA training beyond memorization convergence produces
emergent compositional generalization. Trains on atomic facts from a filtered
subgraph, evaluates on 3-hop compositional questions never seen in training.

Data pipeline:
  1. Load cycle N graph from Test 8
  2. Filter to triples participating in ≥3 three-hop paths
  3. Generate atomic QA pairs from filtered triples
  4. Generate 3-hop and 2-hop evaluation questions from graph paths
  5. Train fresh adapter with extended epochs (30→300)
  6. Probe at each epoch checkpoint: keyed, direct, rephrased, 3-hop, 2-hop, open-ended

Controls:
  - Base model (adapter OFF): separates pretraining knowledge
  - Shuffled labels: confirms generalization requires relational structure

Usage:
    python experiments/test10_grokking.py --model mistral
    python experiments/test10_grokking.py --model mistral --base-cycle 50
    python experiments/test10_grokking.py --model mistral --weight-decay 0.1 --resume
    python experiments/test10_grokking.py --model mistral --control-only
"""

import argparse
import copy
import json
import logging
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

PAUSE_FILE = Path.home() / ".training_pause"


def is_paused():
    """Check if pause has been requested via tpause."""
    return PAUSE_FILE.exists()


def wait_for_cooldown(target=45):
    """Block until GPU temperature drops below target.

    Falls back to a fixed 60s sleep if the cooldown script is unavailable.
    """
    try:
        subprocess.run(
            [
                "bash",
                "-c",
                f"source ~/.local/bin/gpu-cooldown.sh && wait_for_cooldown {target}",
            ],
            check=True,
            timeout=600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("Cooldown script failed (%s), falling back to 60s sleep", e)
        time.sleep(60)


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import networkx as nx  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import TrainerCallback  # noqa: E402

from experiments.utils.test_harness import (  # noqa: E402
    BENCHMARK_MODELS,
    IndexedDataset,
    model_output_dir,
    setup_logging,
)
from paramem.evaluation.recall import generate_answer  # noqa: E402
from paramem.graph.qa_generator import generate_qa_from_relations  # noqa: E402
from paramem.models.loader import (  # noqa: E402
    create_adapter,
    load_adapter,
    load_base_model,
    save_adapter,
)
from paramem.training.indexed_memory import (  # noqa: E402
    assign_keys,
    build_registry,
    format_indexed_training,
    probe_key,
    save_registry,
)
from paramem.training.trainer import train_adapter  # noqa: E402
from paramem.utils.config import AdapterConfig, TrainingConfig  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

OUTPUT_BASE = project_root / "outputs" / "test10_grokking"
TEST8_BASE = project_root / "outputs" / "test8_large_scale"
SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely."
MIN_PATH_PARTICIPATION = 3

# Stop words for token overlap scoring (open-ended probe)
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "that",
    "this", "it", "its", "and", "or", "but", "not", "no", "so", "if",
    "then", "than", "who", "what", "where", "when", "how", "which",
    "their", "they", "them", "he", "she", "his", "her", "you", "your",
}  # fmt: skip


# ============================================================================
# Graph filtering and multi-hop question generation
# ============================================================================


def load_test8_graph(model_name: str, base_cycle: int) -> tuple[dict, Path]:
    """Load cumulative graph from a Test 8 cycle."""
    model_dir = TEST8_BASE / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Test 8 output not found: {model_dir}")

    run_dirs = sorted(model_dir.iterdir())
    if not run_dirs:
        raise FileNotFoundError(f"No runs in {model_dir}")
    run_dir = run_dirs[-1]

    cycle_dir = run_dir / f"cycle_{base_cycle:03d}"
    graph_path = cycle_dir / "cumulative_graph.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"No graph at {graph_path}")

    with open(graph_path) as f:
        graph_data = json.load(f)
    return graph_data, cycle_dir


EXCLUDED_ENTITIES = {"Unknown", "unknown"}


def build_nx_graph(graph_data: dict) -> nx.DiGraph:
    """Build NetworkX directed graph from cumulative_graph.json.

    Excludes artifact entities (e.g. 'Unknown') that pollute multi-hop paths.
    """
    G = nx.DiGraph()
    for n in graph_data.get("nodes", []):
        name = n if isinstance(n, str) else n.get("id", n.get("name", ""))
        if name in EXCLUDED_ENTITIES:
            continue
        G.add_node(name)
    for e in graph_data.get("edges", graph_data.get("links", [])):
        src = e.get("source", e.get("from", ""))
        tgt = e.get("target", e.get("to", ""))
        if src in EXCLUDED_ENTITIES or tgt in EXCLUDED_ENTITIES:
            continue
        pred = e.get("predicate", e.get("label", ""))
        G.add_edge(src, tgt, predicate=pred)
    return G


def enumerate_hop_paths(G: nx.DiGraph, hops: int) -> list[tuple]:
    """Enumerate all N-hop paths in the graph. Returns list of node tuples."""
    paths = []
    if hops == 2:
        for a in G.nodes():
            for b in G.successors(a):
                for c in G.successors(b):
                    if c != a:
                        paths.append((a, b, c))
    elif hops == 3:
        for a in G.nodes():
            for b in G.successors(a):
                for c in G.successors(b):
                    if c != a:
                        for d in G.successors(c):
                            if d != a and d != b:
                                paths.append((a, b, c, d))
    return paths


def filter_triples_by_participation(
    G: nx.DiGraph, three_hop_paths: list[tuple], min_count: int = MIN_PATH_PARTICIPATION
) -> tuple[set, list[tuple]]:
    """Filter triples to those participating in ≥min_count 3-hop paths.

    Returns (filtered_triples, surviving_paths).
    """
    triple_count = Counter()
    for path in three_hop_paths:
        a, b, c, d = path
        triple_count[(a, G.edges[a, b]["predicate"], b)] += 1
        triple_count[(b, G.edges[b, c]["predicate"], c)] += 1
        triple_count[(c, G.edges[c, d]["predicate"], d)] += 1

    kept = {t for t, count in triple_count.items() if count >= min_count}

    surviving = []
    for path in three_hop_paths:
        a, b, c, d = path
        t1 = (a, G.edges[a, b]["predicate"], b)
        t2 = (b, G.edges[b, c]["predicate"], c)
        t3 = (c, G.edges[c, d]["predicate"], d)
        if t1 in kept and t2 in kept and t3 in kept:
            surviving.append(path)

    # Narrow to triples actually used in surviving paths
    actual = set()
    for path in surviving:
        a, b, c, d = path
        actual.add((a, G.edges[a, b]["predicate"], b))
        actual.add((b, G.edges[b, c]["predicate"], c))
        actual.add((c, G.edges[c, d]["predicate"], d))

    return actual, surviving


def compute_entity_coverage(triples: set, G: nx.DiGraph) -> dict:
    """Compute entity coverage stats for hub bias analysis."""
    entities = set()
    for s, _, o in triples:
        entities.add(s)
        entities.add(o)

    degree_map = {}
    for e in entities:
        if G.has_node(e):
            degree_map[e] = G.in_degree(e) + G.out_degree(e)
        else:
            degree_map[e] = 0

    hub_threshold = 5
    hubs = {e for e, d in degree_map.items() if d >= hub_threshold}

    return {
        "unique_entities": len(entities),
        "hub_entities": len(hubs),
        "hub_threshold": hub_threshold,
        "hub_names": sorted(hubs),
        "degree_distribution": {
            e: d for e, d in sorted(degree_map.items(), key=lambda x: -x[1])[:20]
        },
    }


def generate_multihop_questions(G: nx.DiGraph, paths: list[tuple], hops: int) -> list[dict]:
    """Generate natural-language multi-hop questions from graph paths.

    For 3-hop A→B→C→D, generates questions requiring all three relations.
    For 2-hop A→B→C, generates questions requiring both relations.
    """
    questions = []
    seen = set()

    for path in paths:
        if hops == 3:
            a, b, c, d = path
            r1 = G.edges[a, b]["predicate"].replace("_", " ")
            r2 = G.edges[b, c]["predicate"].replace("_", " ")
            r3 = G.edges[c, d]["predicate"].replace("_", " ")
            # Deduplicate by terminal entity + path
            key = (a, b, c, d)
            if key in seen:
                continue
            seen.add(key)
            questions.append(
                {
                    "path": [a, b, c, d],
                    "relations": [r1, r2, r3],
                    "question": (
                        f"{a} {r1} someone. That entity {r2} another. That entity {r3} whom?"
                    ),
                    "expected_entity": d,
                    "intermediate_entities": [b, c],
                    "hops": 3,
                }
            )
        elif hops == 2:
            a, b, c = path
            r1 = G.edges[a, b]["predicate"].replace("_", " ")
            r2 = G.edges[b, c]["predicate"].replace("_", " ")
            key = (a, b, c)
            if key in seen:
                continue
            seen.add(key)
            questions.append(
                {
                    "path": [a, b, c],
                    "relations": [r1, r2],
                    "question": (f"{a} {r1} someone. That entity {r2} whom?"),
                    "expected_entity": c,
                    "intermediate_entities": [b],
                    "hops": 2,
                }
            )

    return questions


# ============================================================================
# Rephrased question generation
# ============================================================================


def generate_rephrased_questions(keyed_pairs: list[dict], model, tokenizer) -> list[dict]:
    """Generate rephrased versions of training questions using the base model.

    Structural changes (passive voice, embedded clauses), not just synonyms.
    """
    rephrased = []

    for kp in keyed_pairs:
        original_q = kp["question"]
        original_a = kp["answer"]
        entity = kp.get("source_object", "")

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "Rephrase the following question using different sentence "
                    "structure (passive voice, embedded clauses, or restructured "
                    "phrasing). The rephrased question must have the same answer. "
                    "Output ONLY the rephrased question, nothing else."
                ),
            },
            {"role": "user", "content": original_q},
        ]
        prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        # Disable adapter for base model generation
        if isinstance(model, PeftModel):
            with model.disable_adapter():
                rephrase = generate_answer(
                    model, tokenizer, prompt, max_new_tokens=100, temperature=0.0
                )
        else:
            rephrase = generate_answer(
                model, tokenizer, prompt, max_new_tokens=100, temperature=0.0
            )

        rephrase = rephrase.strip()
        if not rephrase or rephrase == original_q:
            rephrase = original_q  # Fallback: use original if generation fails

        rephrased.append(
            {
                "key": kp["key"],
                "original_question": original_q,
                "rephrased_question": rephrase,
                "expected_answer": original_a,
                "expected_entity": entity,
            }
        )

    logger.info("Generated %d rephrased questions", len(rephrased))
    return rephrased


# ============================================================================
# Evaluation probes
# ============================================================================


def content_tokens(text: str) -> set[str]:
    """Extract content words (lowercase, stop words removed)."""
    return set(text.lower().split()) - STOP_WORDS


def entity_match(generated: str, expected_entity: str) -> bool:
    """Case-insensitive substring match for entity in generated text."""
    if not expected_entity:
        return False
    return expected_entity.lower() in generated.lower()


def probe_keyed_retrieval(model, tokenizer, keyed_pairs, registry) -> dict:
    """Probe all keys using standard keyed retrieval prompt."""
    results = []
    exact = 0
    for kp in keyed_pairs:
        result = probe_key(model, tokenizer, kp["key"], registry=registry)
        success = bool(result and "failure_reason" not in result)
        if success:
            exact += 1
        results.append(
            {
                "key": kp["key"],
                "expected_answer": kp.get("answer", ""),
                "success": success,
                "raw_output": result.get("raw_output", "") if result else "",
                "failure_reason": result.get("failure_reason", "") if result else "no_result",
            }
        )
    total = len(keyed_pairs)
    return {
        "exact_count": exact,
        "total": total,
        "recall_rate": round(exact / total, 4) if total > 0 else 0.0,
        "results": results,
    }


def probe_direct_questions(model, tokenizer, keyed_pairs) -> dict:
    """Probe using natural questions from keyed_pairs (no key prefix)."""
    results = []
    for kp in keyed_pairs:
        question = kp.get("question", "")
        expected = kp.get("answer", "")
        exp_entity = kp.get("source_object", "")
        if not question:
            continue
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        generated = generate_answer(model, tokenizer, prompt, max_new_tokens=150, temperature=0.0)
        match = entity_match(generated, exp_entity)
        results.append(
            {
                "key": kp["key"],
                "question": question,
                "expected_answer": expected,
                "expected_entity": exp_entity,
                "generated": generated,
                "entity_match": match,
            }
        )

    matched = sum(1 for r in results if r["entity_match"])
    return {
        "matched_count": matched,
        "total": len(results),
        "recall_rate": round(matched / len(results), 4) if results else 0.0,
        "results": results,
    }


def probe_rephrased(model, tokenizer, rephrased_questions) -> dict:
    """Probe using rephrased versions of training questions."""
    results = []
    for rq in rephrased_questions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": rq["rephrased_question"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        generated = generate_answer(model, tokenizer, prompt, max_new_tokens=150, temperature=0.0)
        match = entity_match(generated, rq["expected_entity"])
        results.append(
            {
                "key": rq["key"],
                "rephrased_question": rq["rephrased_question"],
                "original_question": rq["original_question"],
                "expected_entity": rq["expected_entity"],
                "generated": generated,
                "entity_match": match,
            }
        )

    matched = sum(1 for r in results if r["entity_match"])
    return {
        "matched_count": matched,
        "total": len(results),
        "recall_rate": round(matched / len(results), 4) if results else 0.0,
        "results": results,
    }


def probe_multihop(model, tokenizer, questions, hub_entities=None) -> dict:
    """Probe multi-hop (2-hop or 3-hop) questions.

    Primary: exact entity match on terminal entity.
    Secondary: reasoning trace (intermediate entities present).
    """
    if hub_entities is None:
        hub_entities = set()

    results = []
    for q in questions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q["question"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        generated = generate_answer(model, tokenizer, prompt, max_new_tokens=200, temperature=0.0)

        match = entity_match(generated, q["expected_entity"])
        intermediates_present = [entity_match(generated, ie) for ie in q["intermediate_entities"]]
        is_hub_terminal = q["expected_entity"] in hub_entities

        results.append(
            {
                "question": q["question"],
                "expected_entity": q["expected_entity"],
                "intermediate_entities": q["intermediate_entities"],
                "generated": generated,
                "entity_match": match,
                "intermediates_present": intermediates_present,
                "is_hub_terminal": is_hub_terminal,
            }
        )

    matched = sum(1 for r in results if r["entity_match"])
    hub_results = [r for r in results if r["is_hub_terminal"]]
    nonhub_results = [r for r in results if not r["is_hub_terminal"]]

    hub_matched = sum(1 for r in hub_results if r["entity_match"])
    nonhub_matched = sum(1 for r in nonhub_results if r["entity_match"])

    # Reasoning trace: how many correct answers also have all intermediates
    full_chain = sum(1 for r in results if r["entity_match"] and all(r["intermediates_present"]))

    return {
        "matched_count": matched,
        "total": len(results),
        "recall_rate": round(matched / len(results), 4) if results else 0.0,
        "full_chain_count": full_chain,
        "hub_matched": hub_matched,
        "hub_total": len(hub_results),
        "nonhub_matched": nonhub_matched,
        "nonhub_total": len(nonhub_results),
        "results": results,
    }


def probe_open_ended(model, tokenizer, keyed_pairs) -> dict:
    """Probe with 'What do you know about {entity}?' per unique entity."""
    entity_facts = defaultdict(list)
    for kp in keyed_pairs:
        entity = kp.get("source_subject", "")
        if entity and len(entity) > 1:
            entity_facts[entity].append(
                {
                    "key": kp["key"],
                    "answer": kp["answer"],
                }
            )

    entity_results = []
    total_facts = 0
    facts_recalled = 0

    for entity_name, facts in entity_facts.items():
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"What do you know about {entity_name}?"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        generated = generate_answer(model, tokenizer, prompt, max_new_tokens=300, temperature=0.0)

        gen_tokens = content_tokens(generated)
        fact_matches = 0
        for fact in facts:
            expected_tokens = content_tokens(fact["answer"])
            if not expected_tokens:
                continue
            overlap = len(expected_tokens & gen_tokens) / len(expected_tokens)
            if overlap >= 0.4:
                fact_matches += 1

        total_facts += len(facts)
        facts_recalled += fact_matches
        entity_results.append(
            {
                "entity": entity_name,
                "fact_count": len(facts),
                "facts_recalled": fact_matches,
                "generated": generated,
            }
        )

    entities_hit = sum(1 for er in entity_results if er["facts_recalled"] > 0)
    return {
        "entity_count": len(entity_results),
        "total_facts": total_facts,
        "facts_recalled": facts_recalled,
        "fact_recall_rate": round(facts_recalled / total_facts, 4) if total_facts > 0 else 0.0,
        "entity_hit_rate": round(entities_hit / len(entity_results), 4) if entity_results else 0.0,
        "entity_results": entity_results,
    }


def probe_relation_shortcut(model, tokenizer, threehop_questions) -> dict:
    """Probe final relation alone (no chain context) to measure shortcut baseline.

    For each unique final relation in 3-hop questions, ask
    "What entity does someone {relation}?" and score with entity_match.
    If shortcut accuracy matches 3-hop accuracy, there is no composition.
    """
    # Group questions by final relation
    rel_questions = defaultdict(list)
    for q in threehop_questions:
        parts = q["question"].split(". That entity ")
        if len(parts) >= 3:
            final_rel = parts[-1].replace(" whom?", "").strip()
        else:
            final_rel = q["question"]
        rel_questions[final_rel].append(q)

    results = []
    for rel, qs in rel_questions.items():
        shortcut_q = f"What entity does someone {rel}?"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": shortcut_q},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        generated = generate_answer(model, tokenizer, prompt, max_new_tokens=150, temperature=0.0)

        for q in qs:
            match = entity_match(generated, q["expected_entity"])
            results.append(
                {
                    "relation": rel,
                    "shortcut_question": shortcut_q,
                    "expected_entity": q["expected_entity"],
                    "generated": generated,
                    "entity_match": match,
                }
            )

    matched = sum(1 for r in results if r["entity_match"])
    return {
        "matched_count": matched,
        "total": len(results),
        "recall_rate": round(matched / len(results), 4) if results else 0.0,
        "unique_relations": len(rel_questions),
        "results": results,
    }


def run_all_probes(
    model,
    tokenizer,
    keyed_pairs,
    registry,
    rephrased_questions,
    threehop_questions,
    twohop_questions,
    hub_entities,
) -> dict:
    """Run all 7 evaluation probes and return combined results."""
    logger.info("    Probe 1: Keyed retrieval (%d keys)...", len(keyed_pairs))
    keyed = probe_keyed_retrieval(model, tokenizer, keyed_pairs, registry)
    logger.info(
        "      Keyed: %d/%d (%.1f%%)",
        keyed["exact_count"],
        keyed["total"],
        keyed["recall_rate"] * 100,
    )

    logger.info("    Probe 2: Direct questions...")
    direct = probe_direct_questions(model, tokenizer, keyed_pairs)
    logger.info(
        "      Direct: %d/%d (%.1f%%)",
        direct["matched_count"],
        direct["total"],
        direct["recall_rate"] * 100,
    )

    logger.info("    Probe 3: Rephrased questions (%d)...", len(rephrased_questions))
    rephrase = probe_rephrased(model, tokenizer, rephrased_questions)
    logger.info(
        "      Rephrased: %d/%d (%.1f%%)",
        rephrase["matched_count"],
        rephrase["total"],
        rephrase["recall_rate"] * 100,
    )

    logger.info("    Probe 4: 3-hop questions (%d)...", len(threehop_questions))
    hop3 = probe_multihop(model, tokenizer, threehop_questions, hub_entities)
    logger.info(
        "      3-hop: %d/%d (%.1f%%), full-chain: %d, hub: %d/%d, non-hub: %d/%d",
        hop3["matched_count"],
        hop3["total"],
        hop3["recall_rate"] * 100,
        hop3["full_chain_count"],
        hop3["hub_matched"],
        hop3["hub_total"],
        hop3["nonhub_matched"],
        hop3["nonhub_total"],
    )

    logger.info("    Probe 5: 2-hop questions (%d)...", len(twohop_questions))
    hop2 = probe_multihop(model, tokenizer, twohop_questions, hub_entities)
    logger.info(
        "      2-hop: %d/%d (%.1f%%)",
        hop2["matched_count"],
        hop2["total"],
        hop2["recall_rate"] * 100,
    )

    logger.info("    Probe 6: Open-ended...")
    open_e = probe_open_ended(model, tokenizer, keyed_pairs)
    logger.info(
        "      Open: %d/%d facts (%.1f%%)",
        open_e["facts_recalled"],
        open_e["total_facts"],
        open_e["fact_recall_rate"] * 100,
    )

    logger.info("    Probe 7: Relation shortcut control...")
    shortcut = probe_relation_shortcut(model, tokenizer, threehop_questions)
    logger.info(
        "      Shortcut: %d/%d (%.1f%%) — %d unique relations",
        shortcut["matched_count"],
        shortcut["total"],
        shortcut["recall_rate"] * 100,
        shortcut["unique_relations"],
    )

    return {
        "keyed_retrieval": {k: v for k, v in keyed.items() if k != "results"},
        "keyed_detail": keyed["results"],
        "direct_questions": {k: v for k, v in direct.items() if k != "results"},
        "direct_detail": direct["results"],
        "rephrased_questions": {k: v for k, v in rephrase.items() if k != "results"},
        "rephrased_detail": rephrase["results"],
        "threehop": {k: v for k, v in hop3.items() if k != "results"},
        "threehop_detail": hop3["results"],
        "twohop": {k: v for k, v in hop2.items() if k != "results"},
        "twohop_detail": hop2["results"],
        "open_ended": {k: v for k, v in open_e.items() if k != "entity_results"},
        "open_ended_detail": open_e["entity_results"],
        "relation_shortcut": {k: v for k, v in shortcut.items() if k != "results"},
        "relation_shortcut_detail": shortcut["results"],
    }


class ProgressCallback(TrainerCallback):
    """Write progress.json at each epoch for tstatus display."""

    def __init__(self, output_dir, epoch_offset, cycle, num_keys, weight_decay):
        self.output_dir = output_dir
        self.epoch_offset = epoch_offset
        self.cycle = cycle
        self.num_keys = num_keys
        self.weight_decay = weight_decay
        self._last_epoch = -1

    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch) + self.epoch_offset
        if current_epoch <= self._last_epoch:
            return
        self._last_epoch = current_epoch
        target_epoch = int(args.num_train_epochs) + self.epoch_offset
        with open(self.output_dir / "progress.json", "w") as f:
            json.dump(
                {
                    "epoch": current_epoch,
                    "target_epoch": target_epoch,
                    "cycle": self.cycle,
                    "keys": self.num_keys,
                    "weight_decay": self.weight_decay,
                },
                f,
            )


# ============================================================================
# Main experiment
# ============================================================================


def prepare_data(
    model_name: str, base_cycle: int, output_dir: Path, model=None, tokenizer=None
) -> dict:
    """Prepare all data: filter graph, generate QA, generate evaluation questions.

    Returns dict with all cached data paths and loaded data.
    """
    # Check for cached data
    cached_keys = [
        "keyed_pairs",
        "filtered_graph",
        "threehop_questions",
        "twohop_questions",
        "rephrased_questions",
        "entity_coverage",
        "base_cycle",
    ]
    all_cached = all((output_dir / f"{k}.json").exists() for k in cached_keys)

    if all_cached:
        logger.info("Loading cached data from %s", output_dir)
        data = {}
        for k in cached_keys:
            with open(output_dir / f"{k}.json") as f:
                data[k] = json.load(f)
        return data

    logger.info("Preparing data from Test 8 cycle %d...", base_cycle)

    # Load graph
    graph_data, cycle_dir = load_test8_graph(model_name, base_cycle)
    G = build_nx_graph(graph_data)
    logger.info("Loaded graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    # Find 3-hop and 2-hop paths
    three_hop_paths = enumerate_hop_paths(G, 3)
    two_hop_paths = enumerate_hop_paths(G, 2)
    logger.info("Found %d 3-hop paths, %d 2-hop paths", len(three_hop_paths), len(two_hop_paths))

    # Filter triples
    filtered_triples, surviving_3hop = filter_triples_by_participation(
        G, three_hop_paths, MIN_PATH_PARTICIPATION
    )
    logger.info(
        "Filtered: %d triples, %d surviving 3-hop paths, ratio %.2f",
        len(filtered_triples),
        len(surviving_3hop),
        len(surviving_3hop) / len(filtered_triples) if filtered_triples else 0,
    )

    # Entity coverage for hub bias analysis
    coverage = compute_entity_coverage(filtered_triples, G)

    # Filter 2-hop paths to those using only filtered triples
    surviving_2hop = []
    for path in two_hop_paths:
        a, b, c = path
        if not (G.has_edge(a, b) and G.has_edge(b, c)):
            continue
        t1 = (a, G.edges[a, b]["predicate"], b)
        t2 = (b, G.edges[b, c]["predicate"], c)
        if t1 in filtered_triples and t2 in filtered_triples:
            surviving_2hop.append(path)

    # Convert triples to relations format for QA generation
    relations = [
        {"subject": s, "predicate": p, "object": o} for s, p, o in sorted(filtered_triples)
    ]

    # Generate QA pairs (requires model)
    if model is None or tokenizer is None:
        raise RuntimeError("Model and tokenizer required for first-time data generation")

    logger.info("Generating QA pairs from %d triples...", len(relations))
    # Disable adapter if present for QA generation
    if isinstance(model, PeftModel):
        with model.disable_adapter():
            qa_pairs = generate_qa_from_relations(relations, model, tokenizer)
    else:
        qa_pairs = generate_qa_from_relations(relations, model, tokenizer)

    # Assign keys
    keyed_pairs = assign_keys(qa_pairs, start_index=1)
    logger.info("Assigned %d keys", len(keyed_pairs))

    # Generate multi-hop questions
    logger.info("Generating 3-hop questions...")
    threehop_qs = generate_multihop_questions(G, surviving_3hop, 3)
    logger.info("Generated %d 3-hop questions", len(threehop_qs))

    logger.info("Generating 2-hop questions...")
    twohop_qs = generate_multihop_questions(G, surviving_2hop, 2)
    logger.info("Generated %d 2-hop questions", len(twohop_qs))

    # Generate rephrased questions
    logger.info("Generating rephrased questions...")
    rephrased_qs = generate_rephrased_questions(keyed_pairs, model, tokenizer)

    # Save filtered graph
    filtered_graph = {
        "triples": [
            {"subject": s, "predicate": p, "object": o} for s, p, o in sorted(filtered_triples)
        ],
        "source_cycle": base_cycle,
        "source_nodes": G.number_of_nodes(),
        "source_edges": G.number_of_edges(),
        "three_hop_paths_total": len(three_hop_paths),
        "three_hop_paths_surviving": len(surviving_3hop),
        "two_hop_paths_surviving": len(surviving_2hop),
        "ratio": round(len(surviving_3hop) / len(filtered_triples), 2) if filtered_triples else 0,
    }

    base_cycle_info = {
        "cycle": base_cycle,
        "source_dir": str(cycle_dir),
        "num_triples": len(filtered_triples),
        "num_keys": len(keyed_pairs),
        "num_3hop_questions": len(threehop_qs),
        "num_2hop_questions": len(twohop_qs),
        "ratio": filtered_graph["ratio"],
    }

    # Cache everything
    for name, obj in [
        ("keyed_pairs", keyed_pairs),
        ("filtered_graph", filtered_graph),
        ("threehop_questions", threehop_qs),
        ("twohop_questions", twohop_qs),
        ("rephrased_questions", rephrased_qs),
        ("entity_coverage", coverage),
        ("base_cycle", base_cycle_info),
    ]:
        with open(output_dir / f"{name}.json", "w") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    return {
        "keyed_pairs": keyed_pairs,
        "filtered_graph": filtered_graph,
        "threehop_questions": threehop_qs,
        "twohop_questions": twohop_qs,
        "rephrased_questions": rephrased_qs,
        "entity_coverage": coverage,
        "base_cycle": base_cycle_info,
    }


def save_json_atomic(data: dict, target: Path):
    """Write JSON atomically via temp file + rename."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        Path(tmp_path).replace(target)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def save_state(state: dict, output_dir: Path):
    """Save resume state atomically."""
    save_json_atomic(state, output_dir / "state.json")


def load_state(output_dir: Path) -> dict | None:
    """Load resume state. Returns None if not found."""
    state_path = output_dir / "state.json"
    if not state_path.exists():
        return None
    with open(state_path) as f:
        return json.load(f)


def run_control_base_model(
    model,
    tokenizer,
    keyed_pairs,
    registry,
    rephrased_qs,
    threehop_qs,
    twohop_qs,
    hub_entities,
    output_dir,
):
    """Run base model control (adapter OFF)."""
    control_dir = output_dir / "controls" / "base_model"
    if (control_dir / "probe_results.json").exists():
        logger.info("Base model control already complete, skipping")
        return
    control_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running base model control (adapter OFF)...")

    if isinstance(model, PeftModel):
        with model.disable_adapter():
            results = run_all_probes(
                model,
                tokenizer,
                keyed_pairs,
                registry,
                rephrased_qs,
                threehop_qs,
                twohop_qs,
                hub_entities,
            )
    else:
        results = run_all_probes(
            model,
            tokenizer,
            keyed_pairs,
            registry,
            rephrased_qs,
            threehop_qs,
            twohop_qs,
            hub_entities,
        )

    save_json_atomic(results, control_dir / "probe_results.json")
    logger.info("Base model control saved to %s", control_dir)


def run_control_shuffled(
    model,
    tokenizer,
    keyed_pairs,
    registry,
    threehop_qs,
    twohop_qs,
    hub_entities,
    output_dir,
    epochs_per_cycle,
    num_cycles,
    weight_decay,
    learning_rate,
    rank,
):
    """Run shuffled-label control: randomized key→answer pairings, cycle-based.

    Same cycle structure as the main experiment. Rephrased questions are
    skipped (generated from original, non-shuffled answers).
    """
    control_dir = output_dir / "controls" / "shuffled_labels"
    control_dir.mkdir(parents=True, exist_ok=True)

    # Check resume state
    state = load_state(control_dir)
    current_epoch = 0
    completed = set()
    if state:
        current_epoch = state.get("last_completed_epoch", 0)
        completed = set(state.get("completed_epochs", []))

    target_total = num_cycles * epochs_per_cycle
    if current_epoch >= target_total:
        logger.info("Shuffled-label control already complete, skipping")
        return

    logger.info(
        "Running shuffled-label control (E%d → E%d, %d epochs/cycle)...",
        current_epoch,
        target_total,
        epochs_per_cycle,
    )

    # Shuffle answers while keeping keys and questions
    shuffled_pairs = copy.deepcopy(keyed_pairs)
    answers = [kp["answer"] for kp in shuffled_pairs]
    random.seed(42)
    random.shuffle(answers)
    for kp, ans in zip(shuffled_pairs, answers):
        kp["answer"] = ans

    shuffled_registry = build_registry(shuffled_pairs)
    adapter_name = "episodic"

    adapter_config = AdapterConfig(
        rank=rank,
        alpha=rank * 2,
        learning_rate=learning_rate,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        dropout=0.0,
    )

    examples = format_indexed_training(shuffled_pairs, tokenizer, max_length=1024)
    dataset = IndexedDataset(examples)

    cycle = current_epoch // epochs_per_cycle
    while current_epoch < target_total:
        cycle += 1
        target_epoch = current_epoch + epochs_per_cycle

        logger.info("  Shuffled cycle %d: E%d → E%d...", cycle, current_epoch, target_epoch)

        # Unwrap to base model
        if isinstance(model, PeftModel):
            model = model.base_model.model

        # Load previous adapter or create fresh
        if current_epoch > 0:
            prev_adapter = control_dir / f"epoch_{current_epoch:03d}" / "adapter"
            if prev_adapter.exists():
                model = load_adapter(model, prev_adapter, adapter_name)
            else:
                model = create_adapter(model, adapter_config, adapter_name)
        else:
            model = create_adapter(model, adapter_config, adapter_name)

        training_config = TrainingConfig(
            batch_size=1,
            gradient_accumulation_steps=2,
            max_seq_length=1024,
            num_epochs=epochs_per_cycle,
            warmup_steps=100 if current_epoch == 0 else 0,
            warmup_ratio=0.0,
            lr_scheduler_type="constant",
            weight_decay=weight_decay,
            gradient_checkpointing=True,
            max_grad_norm=1.0,
            seed=42,
            save_strategy="no",
        )

        progress_cb = ProgressCallback(
            control_dir, current_epoch, cycle, len(shuffled_pairs), weight_decay
        )
        metrics = train_adapter(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            adapter_name=adapter_name,
            training_config=training_config,
            adapter_config=adapter_config,
            output_dir=control_dir / "adapter_tmp",
            run_name=f"shuffled-c{cycle}",
            callbacks_extra=[progress_cb],
        )

        current_epoch = target_epoch
        epoch_dir = control_dir / f"epoch_{current_epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        # Save adapter
        save_adapter(model, epoch_dir / "adapter", adapter_name)

        # Probe (no rephrased for shuffled control)
        model.gradient_checkpointing_disable()
        try:
            probe_results = run_all_probes(
                model,
                tokenizer,
                shuffled_pairs,
                shuffled_registry,
                [],
                threehop_qs,
                twohop_qs,
                hub_entities,
            )
        finally:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        probe_results["epoch"] = current_epoch
        probe_results["cycle"] = cycle
        probe_results["train_loss"] = metrics.get("train_loss", None)
        save_json_atomic(probe_results, epoch_dir / "probe_results.json")

        completed.add(current_epoch)
        save_state(
            {
                "last_completed_epoch": current_epoch,
                "completed_epochs": sorted(completed),
                "control": "shuffled_labels",
                "weight_decay": weight_decay,
            },
            control_dir,
        )

        logger.info(
            "  Shuffled E%d: keyed=%.1f%%, 3-hop=%.1f%%",
            current_epoch,
            probe_results["keyed_retrieval"]["recall_rate"] * 100,
            probe_results["threehop"]["recall_rate"] * 100,
        )

        wait_for_cooldown(45)

        if is_paused():
            logger.info("  Shuffled control paused at E%d", current_epoch)
            break

    # Unwrap back to base
    if isinstance(model, PeftModel):
        model = model.base_model.model

    return model


def run_experiment(
    model_name: str,
    base_cycle: int,
    weight_decay: float,
    learning_rate: float,
    epochs_per_cycle: int,
    rank: int,
    resume: bool,
    control_only: bool,
):
    """Main experiment loop — train in fixed-size cycles indefinitely."""
    model_config = BENCHMARK_MODELS[model_name]
    adapter_name = "episodic"

    # Output directory
    if resume:
        model_dir = OUTPUT_BASE / model_name
        if model_dir.exists():
            subdirs = sorted(
                [d for d in model_dir.iterdir() if d.is_dir()],
                key=lambda d: d.name,
                reverse=True,
            )
            if subdirs and (subdirs[0] / "state.json").exists():
                output_dir = subdirs[0]
                logger.info("Resuming from: %s", output_dir)
            else:
                logger.info("No resumable state, starting fresh")
                output_dir = model_output_dir(OUTPUT_BASE, model_name)
        else:
            output_dir = model_output_dir(OUTPUT_BASE, model_name)
    else:
        output_dir = model_output_dir(OUTPUT_BASE, model_name)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check disk space
    stat = shutil.disk_usage(output_dir)
    free_gb = stat.free / (1024**3)
    if free_gb < 20:
        logger.error("Insufficient disk space: %.1f GB free", free_gb)
        return
    logger.info("Disk: %.1f GB free", free_gb)

    # Load model
    logger.info("Loading model: %s", model_name)
    model, tokenizer = load_base_model(model_config)
    model.gradient_checkpointing_disable()

    # Prepare data (filter graph, generate QA, multi-hop questions)
    data = prepare_data(model_name, base_cycle, output_dir, model, tokenizer)
    keyed_pairs = data["keyed_pairs"]
    threehop_qs = data["threehop_questions"]
    twohop_qs = data["twohop_questions"]
    rephrased_qs = data["rephrased_questions"]
    coverage = data["entity_coverage"]
    hub_entities = set(coverage.get("hub_names", []))

    registry = build_registry(keyed_pairs)
    save_registry(registry, output_dir / "simhash_registry.json")

    logger.info(
        "Data ready: %d keys, %d 3-hop Qs, %d 2-hop Qs, %d rephrased Qs",
        len(keyed_pairs),
        len(threehop_qs),
        len(twohop_qs),
        len(rephrased_qs),
    )

    # Run controls
    run_control_base_model(
        model,
        tokenizer,
        keyed_pairs,
        registry,
        rephrased_qs,
        threehop_qs,
        twohop_qs,
        hub_entities,
        output_dir,
    )

    if control_only:
        model_for_shuffled = run_control_shuffled(
            model,
            tokenizer,
            keyed_pairs,
            registry,
            threehop_qs,
            twohop_qs,
            hub_entities,
            output_dir,
            epochs_per_cycle,
            10,  # 10 cycles for shuffled control (matches main experiment scale)
            weight_decay,
            learning_rate,
            rank,
        )
        if model_for_shuffled is not None:
            model = model_for_shuffled
        logger.info("Control-only mode complete")
        return

    # Resume state
    state = load_state(output_dir) if resume else None
    completed_epochs = set()
    current_epoch = 0
    if state:
        completed_epochs = set(state.get("completed_epochs", []))
        current_epoch = state.get("last_completed_epoch", 0)
        logger.info("Resume from E%d, completed: %s", current_epoch, sorted(completed_epochs))

    # Clear stale pause file
    if resume and PAUSE_FILE.exists():
        PAUSE_FILE.unlink()
        logger.info("Cleared stale pause file")

    # Unwrap to base model if needed
    if isinstance(model, PeftModel):
        model = model.base_model.model

    # Load saved adapter or create fresh
    adapter_config = AdapterConfig(
        rank=rank,
        alpha=rank * 2,
        learning_rate=learning_rate,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        dropout=0.0,
    )

    if current_epoch > 0:
        adapter_dir = output_dir / f"epoch_{current_epoch:03d}" / "adapter"
        if adapter_dir.exists():
            logger.info("Loading adapter from E%d", current_epoch)
            model = load_adapter(model, adapter_dir, adapter_name)
        else:
            logger.warning("No adapter at E%d, starting fresh", current_epoch)
            model = create_adapter(model, adapter_config, adapter_name)
            current_epoch = 0
    else:
        model = create_adapter(model, adapter_config, adapter_name)

    # Training data
    examples = format_indexed_training(keyed_pairs, tokenizer, max_length=1024)
    dataset = IndexedDataset(examples)

    # ---- Cycle loop: train → probe → save → cooldown → repeat ----
    cycle = current_epoch // epochs_per_cycle if current_epoch > 0 else 0
    while True:
        cycle += 1
        target_epoch = current_epoch + epochs_per_cycle

        logger.info(
            "Cycle %d: training E%d → E%d (%d epochs)...",
            cycle,
            current_epoch,
            target_epoch,
            epochs_per_cycle,
        )

        # Write progress for tstatus
        with open(output_dir / "progress.json", "w") as f:
            json.dump(
                {
                    "epoch": current_epoch,
                    "target_epoch": target_epoch,
                    "cycle": cycle,
                    "keys": len(keyed_pairs),
                    "weight_decay": weight_decay,
                },
                f,
            )

        # Unwrap to base model
        if isinstance(model, PeftModel):
            model = model.base_model.model

        # Load saved adapter weights or create fresh
        if current_epoch > 0:
            prev_adapter = output_dir / f"epoch_{current_epoch:03d}" / "adapter"
            if prev_adapter.exists():
                model = load_adapter(model, prev_adapter, adapter_name)
            else:
                model = create_adapter(model, adapter_config, adapter_name)
        else:
            model = create_adapter(model, adapter_config, adapter_name)

        training_config = TrainingConfig(
            batch_size=1,
            gradient_accumulation_steps=2,
            max_seq_length=1024,
            num_epochs=epochs_per_cycle,
            warmup_steps=100 if current_epoch == 0 else 0,
            warmup_ratio=0.0,
            lr_scheduler_type="constant",
            weight_decay=weight_decay,
            gradient_checkpointing=True,
            max_grad_norm=1.0,
            seed=42,
            save_strategy="no",
        )

        progress_cb = ProgressCallback(
            output_dir, current_epoch, cycle, len(keyed_pairs), weight_decay
        )
        metrics = train_adapter(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            adapter_name=adapter_name,
            training_config=training_config,
            adapter_config=adapter_config,
            output_dir=output_dir / "adapter_tmp",
            run_name=f"grok-c{cycle}-wd{weight_decay}",
            callbacks_extra=[progress_cb],
        )
        train_loss = metrics.get("train_loss", None)

        current_epoch = target_epoch
        epoch_dir = output_dir / f"epoch_{current_epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        # Save adapter checkpoint
        logger.info("  Saving adapter at E%d...", current_epoch)
        save_adapter(model, epoch_dir / "adapter", adapter_name)
        with open(epoch_dir / "keyed_pairs.json", "w") as f:
            json.dump(keyed_pairs, f, indent=2, ensure_ascii=False)
        with open(epoch_dir / "train_loss.json", "w") as f:
            json.dump({"epoch": current_epoch, "train_loss": train_loss}, f)

        # Probe — disable gradient checkpointing for generation
        logger.info("  Probing at E%d (loss=%.6f)...", current_epoch, train_loss or 0.0)
        model.gradient_checkpointing_disable()
        try:
            probe_results = run_all_probes(
                model,
                tokenizer,
                keyed_pairs,
                registry,
                rephrased_qs,
                threehop_qs,
                twohop_qs,
                hub_entities,
            )
        finally:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        probe_results["epoch"] = current_epoch
        probe_results["cycle"] = cycle
        probe_results["train_loss"] = train_loss

        save_json_atomic(probe_results, epoch_dir / "probe_results.json")

        # Update state
        completed_epochs.add(current_epoch)
        run_state = {
            "last_completed_epoch": current_epoch,
            "completed_epochs": sorted(completed_epochs),
            "num_keys": len(keyed_pairs),
            "model": model_name,
            "weight_decay": weight_decay,
            "learning_rate": learning_rate,
            "base_cycle": base_cycle,
            "epochs_per_cycle": epochs_per_cycle,
        }
        save_state(run_state, output_dir)

        # Log summary
        sc = probe_results.get("relation_shortcut", {}).get("recall_rate", 0)
        logger.info(
            "  E%d: keyed=%.1f%%, direct=%.1f%%, rephrased=%.1f%%, "
            "3-hop=%.1f%%, shortcut=%.1f%%, 2-hop=%.1f%%, open=%.1f%%",
            current_epoch,
            probe_results["keyed_retrieval"]["recall_rate"] * 100,
            probe_results["direct_questions"]["recall_rate"] * 100,
            probe_results["rephrased_questions"]["recall_rate"] * 100,
            probe_results["threehop"]["recall_rate"] * 100,
            sc * 100,
            probe_results["twohop"]["recall_rate"] * 100,
            probe_results["open_ended"]["fact_recall_rate"] * 100,
        )

        # Aggregate results.json
        all_results = []
        for ep in sorted(completed_epochs):
            ep_path = output_dir / f"epoch_{ep:03d}" / "probe_results.json"
            if ep_path.exists():
                with open(ep_path) as f:
                    all_results.append(json.load(f))
        save_json_atomic(
            {
                "experiment": "test10_grokking",
                "model": model_name,
                "base_cycle": base_cycle,
                "num_keys": len(keyed_pairs),
                "weight_decay": weight_decay,
                "learning_rate": learning_rate,
                "rank": rank,
                "epochs_per_cycle": epochs_per_cycle,
                "completed_epochs": sorted(completed_epochs),
                "epochs": all_results,
            },
            output_dir / "results.json",
        )

        # GPU cooldown
        logger.info("  Cooling down GPU...")
        wait_for_cooldown(45)

        # Check pause
        if is_paused():
            logger.info(
                "\n  Paused at E%d.\n"
                "  Resume: python experiments/test10_grokking.py"
                " --model %s --resume\n",
                current_epoch,
                model_name,
            )
            break

        # Check disk space
        stat = shutil.disk_usage(output_dir)
        free_gb = stat.free / (1024**3)
        if free_gb < 20:
            logger.error("Disk space low (%.1f GB), stopping", free_gb)
            break


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Test 10: Grokking — Emergent Generalization Beyond Memorization"
    )
    parser.add_argument(
        "--model",
        choices=list(BENCHMARK_MODELS.keys()),
        default="mistral",
        help="Model to use (default: mistral)",
    )
    parser.add_argument(
        "--base-cycle",
        type=int,
        default=50,
        help="Test 8 cycle to use as graph source (default: 50)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="Weight decay (default: 0.1)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--epochs-per-cycle",
        type=int,
        default=30,
        help="Epochs per training cycle (default: 30). Probed after each cycle.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help="LoRA rank (default: 8)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last completed checkpoint",
    )
    parser.add_argument(
        "--control-only",
        action="store_true",
        help="Run control conditions only (no extended training)",
    )

    args = parser.parse_args()

    if args.model not in BENCHMARK_MODELS:
        print(f"Unknown model: {args.model}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  Test 10: Grokking — {args.model}")
    print(f"  Base cycle: {args.base_cycle}")
    print(f"  Weight decay: {args.weight_decay}, LR: {args.learning_rate}")
    print(f"  Epochs per cycle: {args.epochs_per_cycle}")
    print(f"  Rank: {args.rank}")
    print(f"{'=' * 60}")

    run_experiment(
        model_name=args.model,
        base_cycle=args.base_cycle,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        epochs_per_cycle=args.epochs_per_cycle,
        rank=args.rank,
        resume=args.resume,
        control_only=args.control_only,
    )


if __name__ == "__main__":
    from experiments.utils.gpu_guard import acquire_gpu

    with acquire_gpu(interactive=False):
        main()
