"""Phase 4: Adapter Rank Comparison — single-variable episodic rank sweep.

Tests whether smaller episodic adapters (rank 4, 2) produce better recall
than rank 8 by forcing compression pressure. Everything else stays identical:
same 20 sessions, same QA pairs, same evaluation probe, same hyperparameters.

The hypothesis: rank-8 has too much capacity for the available training data,
memorizing noise (multilingual garbage, entity confusion) instead of structured
relations. Smaller adapters physically cannot memorize noise — every parameter
must encode actual relational structure.

Usage:
    python experiments/phase4_rank_comparison.py
    python experiments/phase4_rank_comparison.py --ranks 8,4,2 --num-epochs 20
    python experiments/phase4_rank_comparison.py --ranks 4,2 --max-cycles 10 --num-epochs 5  # smoke
"""

import argparse
import json
import logging
import os
import sys
import time
import unicodedata
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(project_root / ".env")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import wandb  # noqa: E402
from paramem.evaluation.consolidation_metrics import (  # noqa: E402
    compute_consolidation_metrics,
)
from paramem.evaluation.embedding_scorer import compute_similarity  # noqa: E402
from paramem.evaluation.key_fidelity import (  # noqa: E402
    measure_entity_fidelity,
)
from paramem.evaluation.recall import generate_answer  # noqa: E402
from paramem.graph.qa_generator import generate_qa_from_relations  # noqa: E402
from paramem.models.loader import (  # noqa: E402
    create_adapter,
    load_base_model,
    switch_adapter,
)
from paramem.training.consolidation import ConsolidationLoop  # noqa: E402
from paramem.training.dataset import _format_inference_prompt  # noqa: E402
from paramem.training.entity_profile import _get_entity_relations  # noqa: E402
from paramem.utils.config import (  # noqa: E402
    AdapterConfig,
    ConsolidationConfig,
    TrainingConfig,
    load_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Same facts as Phase 3b recall probe and phase4_entity_replay.py
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

# All known entity names from the training data for confusion detection
KNOWN_ENTITIES = {
    "Alex",
    "Maria",
    "Jonas",
    "AutoMate",
    "KIT",
    "Heilbronn",
    "Luna",
    "Black Forest",
    "Barcelona",
    "Python",
}

_RECONSTRUCTION_PROMPTS = [
    "What do you know about {entity}?",
    "What else do you know about {entity}?",
    "Is there anything else you know about {entity}?",
]


# ---------------------------------------------------------------------------
# Quality metric helpers
# ---------------------------------------------------------------------------


def count_garbage_chars(text: str) -> int:
    """Count characters in CJK, Thai, Arabic, and other non-Latin/non-ASCII ranges."""
    garbage = 0
    for char in text:
        cat = unicodedata.category(char)
        if cat.startswith("Lo"):
            # "Lo" = Letter, other — covers CJK ideographs, Thai, Arabic letters, etc.
            # Exclude if it's in a Latin/common script that we expect
            try:
                name = unicodedata.name(char, "")
            except ValueError:
                name = ""
            if any(
                script in name
                for script in ["CJK", "THAI", "ARABIC", "HANGUL", "HIRAGANA", "KATAKANA"]
            ):
                garbage += 1
    return garbage


def garbage_rate(text: str) -> float:
    """Fraction of characters that are garbage (CJK/Thai/Arabic/etc)."""
    if not text:
        return 0.0
    return count_garbage_chars(text) / len(text)


def repetition_rate(text: str) -> float:
    """Fraction of words that are part of 3+ consecutive identical word runs."""
    words = text.split()
    if len(words) < 3:
        return 0.0
    repeated_count = 0
    i = 0
    while i < len(words):
        run_length = 1
        while i + run_length < len(words) and words[i + run_length] == words[i]:
            run_length += 1
        if run_length >= 3:
            repeated_count += run_length
        i += run_length
    return repeated_count / len(words)


def entity_confusion(
    text: str,
    expected_entity: str,
    all_entities: set[str],
) -> bool:
    """Check if a response mentions a wrong entity instead of the expected one.

    Returns True if the response contains a different known entity name
    that is NOT the expected entity (suggesting entity swapping).
    """
    text_lower = text.lower()
    expected_lower = expected_entity.lower()
    for entity in all_entities:
        entity_lower = entity.lower()
        if entity_lower == expected_lower:
            continue
        if entity_lower in text_lower:
            return True
    return False


def analyze_response_quality(
    details: list[dict],
    fact_entity_map: dict[str, str],
) -> dict:
    """Analyze quality metrics across a set of recall probe responses.

    Args:
        details: List of dicts with "question", "expected", "generated", "embedding_score"
        fact_entity_map: Maps question text -> expected primary entity name

    Returns:
        Dict with garbage_rate, repetition_rate, confusion_rate (all 0-1 floats)
    """
    if not details:
        return {"garbage_rate": 0.0, "repetition_rate": 0.0, "confusion_rate": 0.0}

    total_garbage = 0.0
    total_repetition = 0.0
    total_confusion = 0
    for item in details:
        generated = item["generated"]
        total_garbage += garbage_rate(generated)
        total_repetition += repetition_rate(generated)

        expected_entity = fact_entity_map.get(item["question"], "")
        if expected_entity and entity_confusion(generated, expected_entity, KNOWN_ENTITIES):
            total_confusion += 1

    n = len(details)
    return {
        "garbage_rate": total_garbage / n,
        "repetition_rate": total_repetition / n,
        "confusion_rate": total_confusion / n,
    }


# ---------------------------------------------------------------------------
# Adapter management
# ---------------------------------------------------------------------------


def reset_adapters(model, episodic_rank: int, episodic_alpha: int, semantic_config: AdapterConfig):
    """Delete existing adapters and recreate with fresh weights.

    Episodic adapter gets the specified rank/alpha.
    Semantic adapter is always recreated at rank 24, alpha 48.
    """
    from peft import PeftModel

    if isinstance(model, PeftModel):
        for name in list(model.peft_config.keys()):
            try:
                model.delete_adapter(name)
            except ValueError:
                pass

    episodic_config = AdapterConfig(
        rank=episodic_rank,
        alpha=episodic_alpha,
        learning_rate=1e-4,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        dropout=0.0,
    )

    model = create_adapter(model, episodic_config, "episodic")
    model = create_adapter(model, semantic_config, "semantic")

    return model, episodic_config


# ---------------------------------------------------------------------------
# Recall probe
# ---------------------------------------------------------------------------


def probe_recall(model, tokenizer, qa_pairs, adapter_name=None):
    """Probe model recall on QA pairs, optionally with a specific adapter."""
    from peft import PeftModel

    model.gradient_checkpointing_disable()

    if adapter_name and isinstance(model, PeftModel):
        model.enable_adapter_layers()
        switch_adapter(model, adapter_name)
    elif isinstance(model, PeftModel):
        model.disable_adapter_layers()

    results = []
    for qa in qa_pairs:
        prompt = _format_inference_prompt(qa["question"], tokenizer)
        generated = generate_answer(model, tokenizer, prompt, temperature=0.1)
        score = compute_similarity(qa["answer"], generated)
        results.append(
            {
                "question": qa["question"],
                "expected": qa["answer"],
                "generated": generated,
                "embedding_score": score,
            }
        )

    scores = [r["embedding_score"] for r in results]
    return {
        "mean_embedding": sum(scores) / len(scores) if scores else 0.0,
        "details": results,
    }


def probe_entity_fidelity(model, tokenizer, graph, entity_registry, num_passes=3):
    """Probe per-entity reconstruction fidelity with multi-pass prompting."""
    model.gradient_checkpointing_disable()
    switch_adapter(model, "episodic")

    num_passes = min(num_passes, len(_RECONSTRUCTION_PROMPTS))

    fidelity_results = {}
    for entity_name in entity_registry.list_active():
        pass_responses = []
        for pass_idx in range(num_passes):
            prompt = _format_inference_prompt(
                _RECONSTRUCTION_PROMPTS[pass_idx].format(entity=entity_name),
                tokenizer,
            )
            response = generate_answer(
                model,
                tokenizer,
                prompt,
                max_new_tokens=150,
                temperature=0.1,
                repetition_penalty=1.3,
            )
            pass_responses.append(response)

        reconstruction = " ".join(pass_responses)

        ground_truth = _get_entity_relations(graph, entity_name)
        if ground_truth:
            fidelity = measure_entity_fidelity(
                entity_name,
                reconstruction,
                ground_truth,
            )
            fidelity_results[entity_name] = {
                "f1": fidelity["f1"],
                "precision": fidelity["precision"],
                "recall": fidelity["recall"],
                "original_count": fidelity["original_count"],
                "reconstructed_count": fidelity["reconstructed_count"],
                "reconstruction_text": reconstruction,
            }

    return fidelity_results


# ---------------------------------------------------------------------------
# Build fact -> entity mapping for confusion detection
# ---------------------------------------------------------------------------


def build_fact_entity_map(facts: list[dict]) -> dict[str, str]:
    """Map QA questions to their primary entity for confusion detection."""
    qa_pairs = generate_qa_from_relations(facts)
    mapping = {}
    for fact, qa in zip(facts, qa_pairs, strict=False):
        mapping[qa["question"]] = fact["subject"]
    return mapping


# ---------------------------------------------------------------------------
# Single rank run
# ---------------------------------------------------------------------------


def run_single_rank(
    model,
    tokenizer,
    config,
    sessions,
    rank: int,
    alpha: int,
    semantic_config: AdapterConfig,
    args,
) -> dict:
    """Run the full entity-replay pipeline for one episodic rank setting.

    Returns a dict with all metrics for this rank.
    """
    run_start = time.time()
    output_dir = project_root / args.output_dir / f"rank_{rank}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info(
        "Starting rank-%d run (%d sessions, %d epochs)",
        rank,
        args.max_cycles,
        args.num_epochs,
    )
    logger.info("=" * 72)

    # Reset adapters with fresh weights
    model, episodic_config = reset_adapters(model, rank, alpha, semantic_config)

    # Initialize wandb for this rank
    if config.wandb.enabled:
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=f"rank-comparison-r{rank}",
            group=args.wandb_group,
            config={
                "model": config.model.model_id,
                "episodic_rank": rank,
                "episodic_alpha": alpha,
                "num_epochs": args.num_epochs,
                "max_cycles": args.max_cycles,
                "entity_replay_enabled": True,
                "reconstruction_interval": args.reconstruction_interval,
                "reconstruction_passes": args.reconstruction_passes,
                "entity_profile_variants": 1,
            },
            resume="allow",
        )

    # Build consolidation config (identical across ranks)
    consolidation_config = ConsolidationConfig(
        promotion_threshold=config.consolidation.promotion_threshold,
        decay_window=config.consolidation.decay_window,
        procedural_detection_window=config.consolidation.procedural_detection_window,
        episodic_new_weight=config.consolidation.episodic_new_weight,
        semantic_replay_weight=config.consolidation.semantic_replay_weight,
        entity_replay_enabled=True,
        max_active_entities=config.consolidation.max_active_entities,
        entity_retirement_threshold=config.consolidation.entity_retirement_threshold,
        entity_retirement_cycles=config.consolidation.entity_retirement_cycles,
        entity_profile_variants=1,
        reconstruction_interval=args.reconstruction_interval,
        reconstruction_passes=args.reconstruction_passes,
    )

    training_config = TrainingConfig(
        batch_size=config.training.batch_size,
        gradient_accumulation_steps=2,
        max_seq_length=config.training.max_seq_length,
        num_epochs=args.num_epochs,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        gradient_checkpointing=config.training.gradient_checkpointing,
        max_grad_norm=config.training.max_grad_norm,
        seed=config.training.seed,
    )

    # Run consolidation loop
    loop = ConsolidationLoop(
        model=model,
        tokenizer=tokenizer,
        consolidation_config=consolidation_config,
        training_config=training_config,
        episodic_adapter_config=episodic_config,
        semantic_adapter_config=semantic_config,
        wandb_config=config.wandb,
        output_dir=output_dir,
        extraction_temperature=config.graph.extraction_temperature,
    )

    cycle_results = []
    for session in sessions:
        result = loop.run_cycle(
            session_transcript=session["transcript"],
            session_id=session["session_id"],
        )
        cycle_results.append(result)

        if config.wandb.enabled:
            wandb.log(
                {
                    "cycle": result.cycle_index,
                    "entities_extracted": result.entities_extracted,
                    "relations_extracted": result.relations_extracted,
                    "nodes_promoted": result.nodes_promoted,
                    "nodes_decayed": result.nodes_decayed,
                    "episodic_loss": result.episodic_train_loss,
                    "semantic_loss": result.semantic_train_loss,
                    "wall_clock_seconds": result.wall_clock_seconds,
                    "active_entities": len(loop.entity_registry),
                }
            )

        logger.info(
            "Rank %d — Cycle %d/%d complete (%.1fs) — %d active entities",
            rank,
            result.cycle_index,
            args.max_cycles,
            result.wall_clock_seconds,
            len(loop.entity_registry),
        )

    # === ENTITY FIDELITY ===
    logger.info("Rank %d — Probing entity fidelity...", rank)
    entity_fidelity = probe_entity_fidelity(
        model,
        tokenizer,
        loop.merger.graph,
        loop.entity_registry,
        num_passes=args.reconstruction_passes,
    )

    mean_fidelity = 0.0
    if entity_fidelity:
        f1_scores = [v["f1"] for v in entity_fidelity.values()]
        mean_fidelity = sum(f1_scores) / len(f1_scores)

    # === RECALL PROBE ===
    logger.info("Rank %d — Probing recall...", rank)
    promoted_qa = generate_qa_from_relations(PROMOTED_FACTS)
    oneshot_qa = generate_qa_from_relations(ONESHOT_FACTS)

    base_promoted = probe_recall(model, tokenizer, promoted_qa)
    base_oneshot = probe_recall(model, tokenizer, oneshot_qa)
    epi_promoted = probe_recall(model, tokenizer, promoted_qa, "episodic")
    epi_oneshot = probe_recall(model, tokenizer, oneshot_qa, "episodic")
    sem_promoted = probe_recall(model, tokenizer, promoted_qa, "semantic")
    sem_oneshot = probe_recall(model, tokenizer, oneshot_qa, "semantic")

    # === QUALITY METRICS ===
    promoted_entity_map = build_fact_entity_map(PROMOTED_FACTS)
    oneshot_entity_map = build_fact_entity_map(ONESHOT_FACTS)

    epi_promoted_quality = analyze_response_quality(epi_promoted["details"], promoted_entity_map)
    epi_oneshot_quality = analyze_response_quality(epi_oneshot["details"], oneshot_entity_map)

    # Aggregate quality across all episodic responses
    all_epi_details = epi_promoted["details"] + epi_oneshot["details"]
    all_entity_map = {**promoted_entity_map, **oneshot_entity_map}
    epi_quality = analyze_response_quality(all_epi_details, all_entity_map)

    run_wall_clock = time.time() - run_start

    if config.wandb.enabled:
        wandb.log(
            {
                "final/base_promoted": base_promoted["mean_embedding"],
                "final/epi_promoted": epi_promoted["mean_embedding"],
                "final/epi_oneshot": epi_oneshot["mean_embedding"],
                "final/sem_promoted": sem_promoted["mean_embedding"],
                "final/mean_fidelity": mean_fidelity,
                "final/garbage_rate": epi_quality["garbage_rate"],
                "final/confusion_rate": epi_quality["confusion_rate"],
                "final/repetition_rate": epi_quality["repetition_rate"],
                "final/wall_clock_total": run_wall_clock,
            }
        )
        wandb.finish()

    # Consolidation metrics
    consol_metrics = compute_consolidation_metrics(cycle_results)

    run_data = {
        "rank": rank,
        "alpha": alpha,
        "recall": {
            "base_promoted": base_promoted["mean_embedding"],
            "base_oneshot": base_oneshot["mean_embedding"],
            "epi_promoted": epi_promoted["mean_embedding"],
            "epi_oneshot": epi_oneshot["mean_embedding"],
            "sem_promoted": sem_promoted["mean_embedding"],
            "sem_oneshot": sem_oneshot["mean_embedding"],
        },
        "quality": {
            "garbage_rate": epi_quality["garbage_rate"],
            "confusion_rate": epi_quality["confusion_rate"],
            "repetition_rate": epi_quality["repetition_rate"],
            "promoted_quality": epi_promoted_quality,
            "oneshot_quality": epi_oneshot_quality,
        },
        "fidelity": {
            "mean_f1": mean_fidelity,
            "per_entity": {
                name: {k: v for k, v in m.items() if k != "reconstruction_text"}
                for name, m in entity_fidelity.items()
            },
        },
        "efficiency": {
            "wall_clock_total": run_wall_clock,
            "mean_cycle_seconds": consol_metrics.mean_wall_clock_seconds,
        },
        "consolidation": {
            "total_cycles": consol_metrics.total_cycles,
            "total_entities_extracted": consol_metrics.total_entities_extracted,
            "total_promotions": consol_metrics.total_promotions,
            "total_decays": consol_metrics.total_decays,
        },
        "entity_registry": {
            name: loop.entity_registry.get(name).to_dict()
            for name in loop.entity_registry.list_active()
        },
        "entity_fidelity_history": {
            name: loop.entity_registry.get_fidelity_history(name)
            for name in loop.entity_registry.list_active()
        },
        "recall_details": {
            "base_promoted": base_promoted["details"],
            "base_oneshot": base_oneshot["details"],
            "epi_promoted": epi_promoted["details"],
            "epi_oneshot": epi_oneshot["details"],
            "sem_promoted": sem_promoted["details"],
            "sem_oneshot": sem_oneshot["details"],
        },
        "entity_reconstructions": {
            name: m.get("reconstruction_text", "") for name, m in entity_fidelity.items()
        },
        "cycle_results": [
            {
                "cycle": r.cycle_index,
                "session_id": r.session_id,
                "entities": r.entities_extracted,
                "relations": r.relations_extracted,
                "promoted": r.nodes_promoted,
                "decayed": r.nodes_decayed,
                "episodic_loss": r.episodic_train_loss,
                "semantic_loss": r.semantic_train_loss,
                "wall_clock": r.wall_clock_seconds,
            }
            for r in cycle_results
        ],
    }

    # Save per-rank results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(run_data, f, indent=2)
    logger.info("Rank %d results saved to %s", rank, results_path)

    return run_data


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def print_comparison_table(all_results: dict[int, dict]):
    """Print a formatted comparison table across all ranks."""
    ranks = sorted(all_results.keys(), reverse=True)

    print("\n" + "=" * 78)
    print("Phase 4: Adapter Rank Comparison — Entity-Keyed NL Replay")
    print("=" * 78)

    # Header
    header = f"{'Metric':<28s}"
    for rank in ranks:
        header += f"{'Rank ' + str(rank):>15s}"
    print(header)
    print("-" * (28 + 15 * len(ranks)))

    # Recall section
    print("RECALL")
    metrics = [
        ("  Epi promoted", lambda r: r["recall"]["epi_promoted"]),
        ("  Epi oneshot", lambda r: r["recall"]["epi_oneshot"]),
        ("  Sem promoted", lambda r: r["recall"]["sem_promoted"]),
        ("  Sem oneshot", lambda r: r["recall"]["sem_oneshot"]),
        ("  Base promoted", lambda r: r["recall"]["base_promoted"]),
        ("  Base oneshot", lambda r: r["recall"]["base_oneshot"]),
    ]
    for label, getter in metrics:
        row = f"{label:<28s}"
        for rank in ranks:
            row += f"{getter(all_results[rank]):>14.1%} "
        print(row)

    # Quality section
    print()
    print("QUALITY (episodic responses)")
    quality_metrics = [
        ("  Garbage rate", lambda r: r["quality"]["garbage_rate"]),
        ("  Confusion rate", lambda r: r["quality"]["confusion_rate"]),
        ("  Repetition rate", lambda r: r["quality"]["repetition_rate"]),
    ]
    for label, getter in quality_metrics:
        row = f"{label:<28s}"
        for rank in ranks:
            row += f"{getter(all_results[rank]):>14.1%} "
        print(row)

    # Fidelity section
    print()
    print("RECONSTRUCTION FIDELITY")
    row = f"{'  Mean entity F1':<28s}"
    for rank in ranks:
        row += f"{all_results[rank]['fidelity']['mean_f1']:>14.3f} "
    print(row)

    # Per-entity fidelity
    all_entity_names = set()
    for r in all_results.values():
        all_entity_names.update(r["fidelity"]["per_entity"].keys())
    for entity in sorted(all_entity_names):
        row = f"{'    ' + entity:<28s}"
        for rank in ranks:
            f1 = all_results[rank]["fidelity"]["per_entity"].get(entity, {}).get("f1", float("nan"))
            row += f"{f1:>14.3f} "
        print(row)

    # Efficiency section
    print()
    print("EFFICIENCY")
    row = f"{'  Mean cycle (s)':<28s}"
    for rank in ranks:
        row += f"{all_results[rank]['efficiency']['mean_cycle_seconds']:>13.1f}s "
    print(row)
    row = f"{'  Total wall clock (s)':<28s}"
    for rank in ranks:
        row += f"{all_results[rank]['efficiency']['wall_clock_total']:>13.1f}s "
    print(row)

    # Deltas vs base
    print()
    print("DELTA VS BASE (promoted)")
    for rank in ranks:
        r_data = all_results[rank]["recall"]
        epi_delta = r_data["epi_promoted"] - r_data["base_promoted"]
        sem_delta = r_data["sem_promoted"] - r_data["base_promoted"]
        print(f"  Rank {rank}: episodic {epi_delta:+.1%}, semantic {sem_delta:+.1%}")

    print("=" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4: Adapter Rank Comparison")
    parser.add_argument(
        "--ranks",
        type=str,
        default="8,4,2",
        help="Comma-separated episodic ranks to test (default: 8,4,2)",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=20,
        help="Number of sessions per rank (default: 20)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Training epochs per cycle (default: 20)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/phase4_rank_comparison",
        help="Output directory",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default="phase4-rank-comparison",
        help="wandb group name",
    )
    parser.add_argument(
        "--reconstruction-interval",
        type=int,
        default=5,
        help="Reconstruct entity fidelity every N cycles (default: 5)",
    )
    parser.add_argument(
        "--reconstruction-passes",
        type=int,
        default=3,
        help="Number of reconstruction passes per entity (default: 3)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ranks = [int(r.strip()) for r in args.ranks.split(",")]

    config = load_config()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base model once — reused across all rank runs
    logger.info("Loading base model (shared across all rank runs)...")
    model, tokenizer = load_base_model(config.model)

    # Semantic config stays fixed for all runs
    semantic_config = config.adapters.get("semantic")
    if semantic_config is None:
        raise ValueError("'semantic' adapter config required in default.yaml")

    # Load sessions once
    sessions_path = project_root / "data" / "sessions" / "synthetic_sessions.json"
    with open(sessions_path) as f:
        sessions = json.load(f)
    logger.info("Loaded %d sessions, using first %d per rank", len(sessions), args.max_cycles)
    session_slice = sessions[: args.max_cycles]

    # Run each rank
    all_results = {}
    for rank in ranks:
        alpha = rank * 2  # maintain 2:1 ratio
        all_results[rank] = run_single_rank(
            model=model,
            tokenizer=tokenizer,
            config=config,
            sessions=session_slice,
            rank=rank,
            alpha=alpha,
            semantic_config=semantic_config,
            args=args,
        )

    # Print comparison table
    print_comparison_table(all_results)

    # Save combined results
    combined_path = output_dir / "results.json"
    with open(combined_path, "w") as f:
        json.dump(
            {
                "experiment": "phase4_rank_comparison",
                "ranks_tested": ranks,
                "num_sessions": args.max_cycles,
                "num_epochs": args.num_epochs,
                "results_by_rank": {str(k): v for k, v in all_results.items()},
            },
            f,
            indent=2,
        )
    logger.info("Combined results saved to %s", combined_path)

    # Load Phase 3b baseline for reference
    phase3b_probe = project_root / "outputs" / "phase3b" / "recall_probe.json"
    if phase3b_probe.exists():
        with open(phase3b_probe) as f:
            baseline = json.load(f)
        print()
        print("Reference: Phase 3b (55 sessions, 20 epochs, pool-based replay):")
        print(f"  Episodic promoted:  {baseline['episodic_promoted_emb']:.1%}")
        print(f"  Semantic promoted:  {baseline['semantic_promoted_emb']:.1%}")

    logger.info("Rank comparison experiment complete.")


if __name__ == "__main__":
    main()
