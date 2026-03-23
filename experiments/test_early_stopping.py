"""Early Stopping Exploration — per-epoch recall vs loss at multiple scales.

Measures recall AND loss after every epoch at 4 scale points (25, 50, 75, 100)
to determine:
  - At which epoch does recall reach 100% for each scale?
  - Does loss convergence predict recall convergence?
  - Is there a safe epoch floor that works across all scales?

Uses distilled QA from PerLTQA dialogues (not eval QA) so findings transfer
to the large-scale incremental test which also uses distilled QA.

Data is distilled once, then nested subsets are used (25 ⊂ 50 ⊂ 75 ⊂ 100).

Extraction source is configurable:
  --extractor local   (default) use the local model for graph extraction
  --extractor claude  use Claude API for extraction, local model for QA generation

Usage:
    python experiments/test_early_stopping.py --model mistral
    python experiments/test_early_stopping.py --model mistral --extractor claude
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import TrainerCallback  # noqa: E402

from experiments.utils.perltqa_loader import is_available as perltqa_available  # noqa: E402
from experiments.utils.perltqa_loader import load_character_dialogues  # noqa: E402
from experiments.utils.test_harness import (  # noqa: E402
    IndexedDataset,
    add_model_args,
    evaluate_indexed_recall,
    get_benchmark_models,
    load_model_and_config,
    model_output_dir,
    save_results,
    setup_logging,
)
from paramem.graph.extractor import (  # noqa: E402
    EXTRACTION_PROMPT,
    EXTRACTION_SYSTEM,
    extract_graph,
)
from paramem.graph.qa_generator import generate_qa_from_relations  # noqa: E402
from paramem.graph.schema import Entity, Relation, SessionGraph  # noqa: E402
from paramem.models.loader import create_adapter  # noqa: E402
from paramem.training.indexed_memory import (  # noqa: E402
    assign_keys,
    build_registry,
    format_indexed_training,
    save_registry,
)
from paramem.training.trainer import train_adapter  # noqa: E402
from paramem.utils.config import AdapterConfig, TrainingConfig  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

SCALE_POINTS = [25, 50, 75, 100]
OUTPUT_BASE = project_root / "outputs" / "test_early_stopping"
DEFAULT_CHARACTER = "Liang Xin"
NUM_EPOCHS = 30


# ============================================================================
# Recall probing callback — runs inside a single Trainer, no optimizer reset
# ============================================================================


class RecallProbingCallback(TrainerCallback):
    """Probe recall at the end of each epoch within a single training run.

    Stores per-epoch results including per-key raw output.
    """

    def __init__(self, model, tokenizer, keyed_pairs, registry, adapter_name):
        self.model = model
        self.tokenizer = tokenizer
        self.keyed_pairs = keyed_pairs
        self.registry = registry
        self.adapter_name = adapter_name
        self.epoch_log = []
        self.epoch_details = []  # per-key results per epoch
        self.first_perfect_epoch = None
        self.consecutive_perfect = 0
        self._last_epoch = 0

    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch)
        if current_epoch <= self._last_epoch:
            return
        self._last_epoch = current_epoch

        # Disable checkpointing for generation
        self.model.gradient_checkpointing_disable()

        recall_result = evaluate_indexed_recall(
            self.model,
            self.tokenizer,
            self.keyed_pairs,
            self.registry,
            adapter_name=self.adapter_name,
        )

        # Re-enable for next epoch
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        exact = recall_result["exact_count"]
        total = recall_result["total"]

        # Confidence stats
        per_key = recall_result.get("per_key", [])
        confidences = [pk.get("confidence", 0.0) for pk in per_key]
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        min_conf = min(confidences) if confidences else 0.0

        if exact == total:
            self.consecutive_perfect += 1
            if self.first_perfect_epoch is None:
                self.first_perfect_epoch = current_epoch
        else:
            self.consecutive_perfect = 0

        # Get loss from trainer state
        train_loss = None
        if state.log_history:
            for entry in reversed(state.log_history):
                if "loss" in entry:
                    train_loss = entry["loss"]
                    break

        epoch_entry = {
            "epoch": current_epoch,
            "train_loss": train_loss,
            "exact_recall": exact,
            "total": total,
            "recall_rate": exact / total if total > 0 else 0.0,
            "mean_confidence": round(mean_conf, 4),
            "min_confidence": round(min_conf, 4),
            "first_perfect_epoch": self.first_perfect_epoch,
            "consecutive_perfect": self.consecutive_perfect,
        }
        self.epoch_log.append(epoch_entry)

        # Per-key details for this epoch
        epoch_key_details = []
        for pk, kp in zip(per_key, self.keyed_pairs):
            epoch_key_details.append(
                {
                    "key": kp["key"],
                    "question": kp["question"],
                    "answer": kp["answer"],
                    "exact_match": pk.get("exact_match", False),
                    "confidence": pk.get("confidence", 0.0),
                    "raw_output": pk.get("raw_output", ""),
                }
            )
        self.epoch_details.append({"epoch": current_epoch, "per_key": epoch_key_details})

        logger.info(
            "  Epoch %2d: loss=%.4f  recall=%d/%d (%.1f%%)  conf=%.3f  min_conf=%.3f",
            current_epoch,
            train_loss or 0.0,
            exact,
            total,
            epoch_entry["recall_rate"] * 100,
            mean_conf,
            min_conf,
        )


# ============================================================================
# Extraction — local or Claude API
# ============================================================================


def extract_with_claude(transcript, session_id):
    """Extract a knowledge graph using Claude API.

    Uses the same prompt as the local extractor for consistency.
    Returns a SessionGraph.
    """
    try:
        import anthropic
    except ImportError:
        logger.error(
            "anthropic package required for --extractor claude. Install: pip install anthropic"
        )
        sys.exit(1)

    from datetime import datetime, timezone

    client = anthropic.Anthropic()
    prompt = EXTRACTION_PROMPT.format(transcript=transcript)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        temperature=0.0,
        system=EXTRACTION_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        # Try progressive extraction
        start = raw_text.find("{")
        if start >= 0:
            for end in range(len(raw_text) - 1, start, -1):
                if raw_text[end] == "}":
                    try:
                        data = json.loads(raw_text[start : end + 1])
                        break
                    except json.JSONDecodeError:
                        continue
            else:
                logger.warning("Claude extraction failed to parse JSON for %s", session_id)
                return SessionGraph(
                    session_id=session_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
        else:
            logger.warning("No JSON found in Claude output for %s", session_id)
            return SessionGraph(
                session_id=session_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    entities = [
        Entity(
            name=e.get("name", ""),
            entity_type=e.get("entity_type", "concept"),
            attributes=e.get("attributes", {}),
        )
        for e in data.get("entities", [])
    ]

    relations = [
        Relation(
            subject=r.get("subject", ""),
            predicate=r.get("predicate", "related_to"),
            object=r.get("object", ""),
            relation_type=r.get("relation_type", "factual"),
            confidence=r.get("confidence", 1.0),
        )
        for r in data.get("relations", [])
    ]

    graph = SessionGraph(
        session_id=session_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        entities=entities,
        relations=relations,
        summary=data.get("summary", ""),
    )

    logger.info(
        "Claude extracted: %d entities, %d relations (session=%s)",
        len(entities),
        len(relations),
        session_id,
    )
    return graph


def distill_qa_from_sessions(model, tokenizer, character, target_count, extractor="local"):
    """Distill QA pairs from PerLTQA dialogues.

    Extraction can use the local model or Claude API. QA generation always
    uses the local model for consistent formatting.

    Returns (qa_pairs, sessions_used).
    """
    from peft import PeftModel

    sessions = load_character_dialogues(character)
    qa_pairs = []
    seen_questions = set()

    for i, session in enumerate(sessions):
        logger.info(
            "Distilling session %d/%d: %s (%d QA pairs so far, need %d) [extractor=%s]",
            i + 1,
            len(sessions),
            session["session_id"],
            len(qa_pairs),
            target_count,
            extractor,
        )

        # Step 1: Extract graph
        if extractor == "claude":
            session_graph = extract_with_claude(session["transcript"], session["session_id"])
        else:
            model.gradient_checkpointing_disable()
            if isinstance(model, PeftModel):
                with model.disable_adapter():
                    session_graph = extract_graph(
                        model,
                        tokenizer,
                        session["transcript"],
                        session["session_id"],
                        temperature=0.0,
                    )
            else:
                session_graph = extract_graph(
                    model,
                    tokenizer,
                    session["transcript"],
                    session["session_id"],
                    temperature=0.0,
                )

        relations = [
            {"subject": r.subject, "predicate": r.predicate, "object": r.object}
            for r in session_graph.relations
        ]

        if not relations:
            logger.warning("No relations extracted from session %s", session["session_id"])
            continue

        # Step 2: Generate QA from relations (always local model)
        model.gradient_checkpointing_disable()
        if isinstance(model, PeftModel):
            with model.disable_adapter():
                session_qa = generate_qa_from_relations(relations, model=model, tokenizer=tokenizer)
        else:
            session_qa = generate_qa_from_relations(relations, model=model, tokenizer=tokenizer)

        # Step 3: Deduplicate
        new_pairs = []
        for qa in session_qa:
            q_norm = qa["question"].lower().strip()
            if q_norm not in seen_questions:
                seen_questions.add(q_norm)
                new_pairs.append(qa)

        qa_pairs.extend(new_pairs)
        logger.info(
            "Session %s -> %d relations -> %d new QA pairs (total seen: %d)",
            session["session_id"],
            len(relations),
            len(new_pairs),
            len(seen_questions),
        )

        if len(qa_pairs) >= target_count:
            logger.info(
                "Reached %d QA pairs after %d sessions (target: %d)",
                len(qa_pairs),
                i + 1,
                target_count,
            )
            return qa_pairs, i + 1

    logger.info(
        "Exhausted all %d sessions, collected %d QA pairs (target: %d)",
        len(sessions),
        len(qa_pairs),
        target_count,
    )
    return qa_pairs, len(sessions)


# ============================================================================
# GPU control
# ============================================================================


def wait_for_cooldown(target=45):
    """Block until GPU temperature drops below target."""
    subprocess.run(
        [
            "bash",
            "-c",
            f"source ~/.local/bin/gpu-cooldown.sh && wait_for_cooldown {target}",
        ],
        check=True,
    )


# ============================================================================
# Per-scale training with recall callback
# ============================================================================


def run_scale_point(model, tokenizer, qa_pairs, scale, rank, num_epochs, output_dir):
    """Train at a single scale point with per-epoch recall probing.

    Uses a single Trainer call with a callback — no optimizer reset between epochs.

    Returns (model, epoch_log, epoch_details, scale_result).
    """
    from peft import PeftModel as _PeftModel

    subset = qa_pairs[:scale]
    actual_scale = len(subset)
    if actual_scale < scale:
        logger.warning("Only %d QA pairs available, requested %d", actual_scale, scale)

    logger.info("=== Scale point: %d keys, %d epochs ===", actual_scale, num_epochs)

    adapter_name = f"episodic_s{actual_scale}"
    scale_dir = output_dir / f"scale_{actual_scale}"
    scale_dir.mkdir(parents=True, exist_ok=True)

    # Unwrap if needed
    if isinstance(model, _PeftModel):
        model = model.base_model.model

    # Prepare adapter and training data
    adapter_config = AdapterConfig(
        rank=rank,
        alpha=rank * 2,
        learning_rate=1e-4,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        dropout=0.0,
    )
    model = create_adapter(model, adapter_config, adapter_name)

    keyed_pairs = assign_keys(subset, start_index=1)
    registry = build_registry(keyed_pairs)
    save_registry(registry, scale_dir / "simhash_registry.json")

    # Save keyed_pairs
    kp_ser = []
    for kp in keyed_pairs:
        entry = {"key": kp["key"], "question": kp["question"], "answer": kp["answer"]}
        for meta_key in ("source_predicate", "source_subject", "source_object"):
            if meta_key in kp:
                entry[meta_key] = kp[meta_key]
        kp_ser.append(entry)
    with open(scale_dir / "keyed_pairs.json", "w") as f:
        json.dump(kp_ser, f, indent=2)

    examples = format_indexed_training(keyed_pairs, tokenizer, max_length=1024)
    dataset = IndexedDataset(examples)

    # Create recall probing callback
    recall_callback = RecallProbingCallback(model, tokenizer, keyed_pairs, registry, adapter_name)

    # Single Trainer call for all epochs — optimizer state preserved
    training_config = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=2,
        max_seq_length=1024,
        num_epochs=num_epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=42,
    )

    t0 = time.time()
    train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        adapter_name=adapter_name,
        training_config=training_config,
        adapter_config=adapter_config,
        output_dir=scale_dir / "adapter",
        run_name=f"es-s{actual_scale}",
        callbacks_extra=[recall_callback],
    )
    train_time = time.time() - t0

    # Extract results from callback
    epoch_log = recall_callback.epoch_log
    epoch_details = recall_callback.epoch_details

    # Save epoch log and details
    with open(scale_dir / "epoch_log.json", "w") as f:
        json.dump(epoch_log, f, indent=2)
    with open(scale_dir / "epoch_details.json", "w") as f:
        json.dump(epoch_details, f, indent=2)

    # Build scale result summary
    final = epoch_log[-1] if epoch_log else {}
    peak_recall = max((e["exact_recall"] for e in epoch_log), default=0)
    scale_result = {
        "scale": actual_scale,
        "epochs_run": num_epochs,
        "train_time_seconds": round(train_time, 1),
        "final_loss": final.get("train_loss"),
        "final_recall": final.get("exact_recall", 0),
        "final_total": final.get("total", actual_scale),
        "final_recall_rate": final.get("recall_rate", 0),
        "final_mean_confidence": final.get("mean_confidence", 0),
        "first_perfect_epoch": recall_callback.first_perfect_epoch,
        "max_consecutive_perfect": max((e["consecutive_perfect"] for e in epoch_log), default=0),
        # Earliest epoch at which peak recall was first achieved
        "peak_recall": peak_recall,
        "peak_recall_epoch": next(
            (e["epoch"] for e in epoch_log if e["exact_recall"] == peak_recall),
            None,
        ),
    }

    logger.info(
        "Scale %d complete: first_perfect=%s, final=%d/%d, time=%.0fs",
        actual_scale,
        recall_callback.first_perfect_epoch,
        final.get("exact_recall", 0),
        final.get("total", actual_scale),
        train_time,
    )

    return model, epoch_log, epoch_details, scale_result


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Early Stopping Exploration — per-epoch recall vs loss"
    )
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--character", type=str, default=DEFAULT_CHARACTER)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_BASE))
    parser.add_argument(
        "--scales",
        type=str,
        default=",".join(str(s) for s in SCALE_POINTS),
        help="Comma-separated scale points (default: 25,50,75,100)",
    )
    parser.add_argument(
        "--extractor",
        type=str,
        choices=["local", "claude"],
        default="local",
        help="Extraction source: local model or Claude API (default: local)",
    )
    add_model_args(parser)
    args = parser.parse_args()

    num_epochs = args.num_epochs
    scales = [int(s) for s in args.scales.split(",")]
    max_scale = max(scales)
    base_output_dir = Path(args.output_dir)

    if not perltqa_available():
        print("ERROR: PerLTQA required for early stopping exploration")
        sys.exit(1)

    for bench_name, bench_model_config in get_benchmark_models(args):
        print(f"\n{'=' * 72}")
        print(f"  Early Stopping Exploration — {bench_name}")
        print(f"  Scales: {scales}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Extractor: {args.extractor}")
        print(f"{'=' * 72}")

        model, tokenizer, config = load_model_and_config(bench_model_config)
        output_dir = model_output_dir(base_output_dir, bench_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Distill QA pairs from dialogues
        print(f"\n--- Phase 1: Distilling QA pairs (target: {max_scale}) ---")
        t_distill = time.time()
        qa_pairs, sessions_used = distill_qa_from_sessions(
            model, tokenizer, args.character, max_scale, extractor=args.extractor
        )
        distill_time = time.time() - t_distill

        print(
            f"Distilled {len(qa_pairs)} QA pairs from {sessions_used} sessions"
            f" in {distill_time:.0f}s (extractor={args.extractor})"
        )

        if len(qa_pairs) < min(scales):
            print(f"ERROR: Only {len(qa_pairs)} QA pairs distilled, need at least {min(scales)}")
            sys.exit(1)

        # Save distilled QA for reference
        with open(output_dir / "distilled_qa.json", "w") as f:
            json.dump(qa_pairs, f, indent=2)

        # Phase 2: Per-epoch training at each scale point
        all_scale_results = {}
        all_epoch_logs = {}

        for scale in scales:
            if len(qa_pairs) < scale:
                logger.warning(
                    "Skipping scale %d: only %d QA pairs available",
                    scale,
                    len(qa_pairs),
                )
                continue

            print(f"\n--- Scale point: {scale} keys ---")
            wait_for_cooldown(45)

            model, epoch_log, epoch_details, scale_result = run_scale_point(
                model, tokenizer, qa_pairs, scale, args.rank, num_epochs, output_dir
            )

            all_scale_results[str(scale)] = scale_result
            all_epoch_logs[str(scale)] = epoch_log

        # Save combined results
        results = {
            "experiment": "early_stopping_exploration",
            "model": bench_name,
            "model_id": bench_model_config.model_id,
            "character": args.character,
            "data_source": f"perltqa_dialogues:{args.character}",
            "extractor": args.extractor,
            "sessions_used": sessions_used,
            "total_qa_distilled": len(qa_pairs),
            "distill_time_seconds": round(distill_time, 1),
            "scales": scales,
            "epochs_per_scale": num_epochs,
            "rank": args.rank,
            "scale_results": all_scale_results,
            "epoch_logs": all_epoch_logs,
        }
        save_results(results, output_dir)

        # Print summary
        print(f"\n{'=' * 72}")
        print(f"  Early Stopping Summary — {bench_name} (extractor={args.extractor})")
        print(f"{'=' * 72}")
        print(f"  {'Scale':>6}  {'First 100%':>12}  {'Peak':>8}  {'Final':>8}  {'Loss':>8}")
        print(f"  {'─' * 50}")
        for scale in scales:
            s = str(scale)
            if s in all_scale_results:
                r = all_scale_results[s]
                fpe = r["first_perfect_epoch"]
                fpe_str = str(fpe) if fpe else "never"
                print(
                    f"  {r['scale']:>6}  {fpe_str:>12}  "
                    f"{r['peak_recall']}/{r['final_total']:>5}  "
                    f"{r['final_recall']}/{r['final_total']:>5}  "
                    f"{r['final_loss']:>8.4f}"
                )
        print(f"\n  Results: {output_dir / 'results.json'}")

        from paramem.models.loader import unload_model

        unload_model(model, tokenizer)


if __name__ == "__main__":
    main()
