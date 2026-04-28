"""Test 8: Large-Scale Incremental Indexed Key Retrieval.

Proves indexed key retrieval scales to 500+ keys using the full consolidation
pipeline: session transcript → graph extraction → graph merge → QA generation
→ indexed key training with full replay.

Multi-character PerLTQA data, Mistral 7B, 30 epochs, rank 8.

Features:
  - Resumable: state.json tracks progress, --resume continues from last cycle
  - Pausable: touch ~/.training_pause to stop after current cycle
  - GPU cooldown between cycles
  - Full QA regeneration from cumulative graph each cycle (no cache)
  - Phase saves after every cycle
  - Per-epoch recall probing every 5 epochs
  - Per-key first_seen_cycle and source_character for cohort analysis
  - Disk space checking before each cycle

Usage:
    python experiments/test8_large_scale.py --model mistral
    python experiments/test8_large_scale.py --model mistral --resume
    python experiments/test8_large_scale.py --model mistral --batch-size 3 --max-cycles 1
"""

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from peft import PeftModel as _PeftModel  # noqa: E402
from transformers import TrainerCallback  # noqa: E402

from experiments.utils.perltqa_loader import is_available as perltqa_available  # noqa: E402
from experiments.utils.perltqa_loader import (  # noqa: E402
    list_characters,
    load_character_dialogues,
)
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
from paramem.graph.extractor import extract_graph  # noqa: E402
from paramem.graph.merger import GraphMerger  # noqa: E402
from paramem.graph.qa_generator import generate_qa_from_relations  # noqa: E402
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

OUTPUT_BASE = project_root / "outputs" / "test8_large_scale"
PAUSE_FILE = Path.home() / ".training_pause"
DEFAULT_BATCH_SIZE = 5
DEFAULT_TARGET_KEYS = 500
NUM_EPOCHS = 30
RANK = 8
PROBE_EVERY_N_EPOCHS = 5
ESTIMATED_QA_PER_SESSION = 1.8

# Milestone thresholds for detailed logging
MILESTONES = {50, 100, 150, 200, 250, 300, 350, 400, 450, 500}

# Minimum free disk space thresholds
DISK_WARN_FREE_GB = 50
DISK_HARD_MIN_FREE_GB = 10

# Estimated per-cycle disk cost (adapter weights ~27MB + tokenizer ~4MB + JSON ~3MB)
ESTIMATED_CYCLE_DISK_MB = 35


# ============================================================================
# GPU control
# ============================================================================


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


def is_paused():
    """Check if training pause has been requested."""
    return PAUSE_FILE.exists()


# ============================================================================
# Disk space management
# ============================================================================


def check_disk_space(output_dir, remaining_cycles=1):
    """Check free disk space before starting a cycle.

    Returns (ok, used_gb, free_gb). Warns at 50GB free, hard-stops at 10GB.
    """
    stat = shutil.disk_usage(output_dir)
    used_gb = stat.used / (1024**3)
    free_gb = stat.free / (1024**3)
    needed_gb = (remaining_cycles * ESTIMATED_CYCLE_DISK_MB) / 1024

    if free_gb < DISK_HARD_MIN_FREE_GB:
        logger.error(
            "Only %.1f GB free (hard minimum: %d GB). Clean up before continuing.",
            free_gb,
            DISK_HARD_MIN_FREE_GB,
        )
        return False, used_gb, free_gb

    if free_gb < DISK_WARN_FREE_GB:
        logger.warning(
            "Only %.1f GB free (warning threshold: %d GB). Consider cleaning up old results.",
            free_gb,
            DISK_WARN_FREE_GB,
        )

    if free_gb < needed_gb:
        logger.error(
            "Only %.1f GB free, need ~%.1f GB for %d remaining cycles.",
            free_gb,
            needed_gb,
            remaining_cycles,
        )
        return False, used_gb, free_gb

    return True, used_gb, free_gb


def estimate_disk_usage(target_keys, batch_size, num_epochs):
    """Estimate total disk usage for the full run."""
    sessions_per_cycle = batch_size
    keys_per_cycle = sessions_per_cycle * ESTIMATED_QA_PER_SESSION
    estimated_cycles = int(target_keys / keys_per_cycle) + 1
    total_mb = estimated_cycles * ESTIMATED_CYCLE_DISK_MB
    return estimated_cycles, total_mb


# ============================================================================
# Character queue — select enough characters to reach target key count
# ============================================================================


def build_character_queue(target_keys, qa_per_session=ESTIMATED_QA_PER_SESSION):
    """Select characters by descending session count until we have enough data.

    Returns list of (character_name, session_count) tuples.
    """
    chars = list_characters()
    # Sort by dialogue count descending
    sorted_chars = sorted(chars.items(), key=lambda x: x[1]["dialogues"], reverse=True)

    queue = []
    estimated_qa = 0
    for name, info in sorted_chars:
        queue.append((name, info["dialogues"]))
        estimated_qa += info["dialogues"] * qa_per_session
        if estimated_qa >= target_keys:
            break

    logger.info(
        "Character queue: %d characters, ~%d estimated QA pairs (target: %d)",
        len(queue),
        int(estimated_qa),
        target_keys,
    )
    for name, count in queue:
        logger.info("  %s: %d sessions", name, count)

    return queue


# ============================================================================
# Session iterator — yields sessions across characters
# ============================================================================


def session_iterator(character_queue, start_char=None, start_session_idx=0):
    """Yield (character_name, session_index, session) across multiple characters.

    If start_char is provided, skip characters before it and skip sessions
    before start_session_idx within the starting character.
    """
    started = start_char is None
    for char_name, _ in character_queue:
        if not started:
            if char_name == start_char:
                started = True
            else:
                continue

        sessions = load_character_dialogues(char_name)
        start_idx = start_session_idx if char_name == start_char else 0

        for i, session in enumerate(sessions):
            if i < start_idx:
                continue
            yield char_name, i, session


# ============================================================================
# State management — save/load for resumability
# ============================================================================


def save_state(state, output_dir):
    """Save resume state to state.json atomically (write-to-temp + rename)."""
    import os
    import tempfile

    state_copy = dict(state)
    if isinstance(state_copy.get("seen_questions"), set):
        state_copy["seen_questions"] = sorted(state_copy["seen_questions"])
    if isinstance(state_copy.get("seen_triples"), set):
        state_copy["seen_triples"] = [list(t) for t in sorted(state_copy["seen_triples"])]
    if isinstance(state_copy.get("qa_first_seen_cycle"), dict):
        state_copy["qa_first_seen_cycle"] = dict(state_copy["qa_first_seen_cycle"])
    if isinstance(state_copy.get("qa_source_character"), dict):
        state_copy["qa_source_character"] = dict(state_copy["qa_source_character"])

    target = output_dir / "state.json"
    fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state_copy, f, indent=2)
        os.replace(tmp_path, target)
    except BaseException:
        os.unlink(tmp_path)
        raise


def load_state(output_dir):
    """Load resume state from state.json. Returns None if not found."""
    state_path = output_dir / "state.json"
    if not state_path.exists():
        return None
    with open(state_path) as f:
        state = json.load(f)
    if "seen_questions" in state:
        state["seen_questions"] = set(state["seen_questions"])
    if "seen_triples" in state:
        state["seen_triples"] = {tuple(t) for t in state["seen_triples"]}
    # Ensure new fields exist for backward compat with old state files
    if "qa_first_seen_cycle" not in state:
        state["qa_first_seen_cycle"] = {}
    if "qa_source_character" not in state:
        state["qa_source_character"] = {}
    if "current_character_session_index" not in state:
        state["current_character_session_index"] = 0
    if "seen_triples" not in state:
        state["seen_triples"] = set()
    if "skipped_cycles" not in state:
        state["skipped_cycles"] = []
    return state


def fresh_state():
    """Create initial state for a new run."""
    return {
        "last_completed_cycle": 0,
        "characters_completed": [],
        "current_character": None,
        "current_character_session_index": 0,
        "total_sessions_processed": 0,
        "total_qa_pairs": 0,
        "seen_questions": set(),
        "seen_triples": set(),
        "skipped_cycles": [],
        "qa_first_seen_cycle": {},  # question_norm -> cycle_num
        "qa_source_character": {},  # question_norm -> character_name
    }


# ============================================================================
# Extraction — graph extraction with adapter disabled
# ============================================================================


def extract_session_graph(model, tokenizer, transcript, session_id):
    """Extract graph from a session transcript using the base model.

    Disables adapter if model is PeftModel for clean base-model extraction.
    """
    model.gradient_checkpointing_disable()
    if isinstance(model, _PeftModel):
        with model.disable_adapter():
            graph = extract_graph(model, tokenizer, transcript, session_id, temperature=0.0)
    else:
        graph = extract_graph(model, tokenizer, transcript, session_id, temperature=0.0)
    return graph


# ============================================================================
# QA generation — from cumulative graph triples
# ============================================================================


def _character_from_session_id(session_id):
    """Extract character name from a PerLTQA session_id.

    Format: perltqa_{character_name}_{dlg_id}
    Character names are space-separated words (no underscores, no digits).
    dlg_ids always start with a digit (e.g. "1_0_0#0").
    Split on the first underscore followed by a digit.
    """
    prefix = "perltqa_"
    if not session_id.startswith(prefix):
        return "unknown"
    rest = session_id[len(prefix) :]
    # Find first _<digit> boundary — character name is before it
    match = re.search(r"_(\d)", rest)
    if match:
        return rest[: match.start()]
    return rest


def _build_triple_character_map(merger):
    """Build a mapping from (subject, predicate, object) to source character.

    Derives character from the first session_id on each graph edge.
    """
    triple_char = {}
    for subject, obj, data in merger.graph.edges(data=True):
        predicate = data.get("predicate", "related_to")
        sessions = data.get("sessions", [])
        char = _character_from_session_id(sessions[0]) if sessions else "unknown"
        triple_char[(subject, predicate, obj)] = char
    return triple_char


def generate_qa_from_graph(
    model,
    tokenizer,
    merger,
    seen_questions,
    seen_triples,
    qa_first_seen_cycle,
    qa_source_character,
    cycle_num,
):
    """Generate QA pairs from all triples in the cumulative graph.

    Full regeneration each cycle — no caching. This verifies pipeline
    integrity as adapter weights change the extraction landscape.
    Deduplicates questions by normalized text for training data quality.
    Counts new facts by triple identity (subject, predicate, object) for
    the skip decision.
    Source character is derived per-triple from graph edge session metadata.

    Returns (qa_pairs, new_count, new_triples) where new_count is new
    triples (not new question strings) and new_triples is the set of
    triple tuples. Caller must commit new_triples to seen_triples after
    training succeeds.
    """
    triples = merger.get_all_triples()
    if not triples:
        return [], 0

    # Derive source character per triple from graph edge sessions
    triple_char = _build_triple_character_map(merger)

    relations = [{"subject": s, "predicate": p, "object": o} for s, p, o in triples]

    model.gradient_checkpointing_disable()
    if isinstance(model, _PeftModel):
        with model.disable_adapter():
            raw_qa = generate_qa_from_relations(relations, model=model, tokenizer=tokenizer)
    else:
        raw_qa = generate_qa_from_relations(relations, model=model, tokenizer=tokenizer)

    # Verify 1:1 alignment between triples and generated QA
    if len(raw_qa) != len(triples):
        logger.warning(
            "QA/triple count mismatch: %d QA from %d triples. "
            "source_character attribution may be inaccurate.",
            len(raw_qa),
            len(triples),
        )

    # Deduplicate and tag metadata.
    # new_triples is returned separately — caller commits to seen_triples
    # only after training succeeds, so a QA-generation or training failure
    # does not permanently skip those triples.
    qa_pairs = []
    new_triples = set()
    for i, qa in enumerate(raw_qa):
        q_norm = qa["question"].lower().strip()

        # Derive character from the triple that generated this QA
        if i < len(triples):
            triple_key = triples[i]
            triple_source_char = triple_char.get(triple_key, "unknown")
        else:
            triple_key = None
            triple_source_char = "unknown"

        # Identify new triples for skip decision
        if triple_key is not None and triple_key not in seen_triples:
            new_triples.add(triple_key)

        # Track question strings for training data dedup
        if q_norm not in seen_questions:
            seen_questions.add(q_norm)
            qa_first_seen_cycle[q_norm] = cycle_num
            qa_source_character[q_norm] = triple_source_char

        # Tag QA with cohort metadata
        qa["first_seen_cycle"] = qa_first_seen_cycle.get(q_norm, cycle_num)
        qa["source_character"] = qa_source_character.get(q_norm, triple_source_char)
        qa_pairs.append(qa)

    return qa_pairs, len(new_triples), new_triples


# ============================================================================
# Per-epoch recall probing callback
# ============================================================================


class ScaleRecallCallback(TrainerCallback):
    """Probe recall at configurable epoch intervals within a single Trainer.

    Lighter than probing every epoch — designed for long training runs.
    """

    def __init__(
        self,
        model,
        tokenizer,
        keyed_pairs,
        registry,
        adapter_name,
        cycle_dir,
        probe_every=PROBE_EVERY_N_EPOCHS,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.keyed_pairs = keyed_pairs
        self.registry = registry
        self.adapter_name = adapter_name
        self.cycle_dir = cycle_dir
        self.probe_every = probe_every
        self.epoch_log = []
        self._last_epoch = -1

    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch)
        if current_epoch <= self._last_epoch:
            return
        self._last_epoch = current_epoch

        # Write progress file every epoch (cheap, enables ts monitoring).
        # Preserve cycle_started_at written at cycle start — tstatus uses it
        # to compute elapsed/ETA. Per-epoch writes must not reset it.
        total_epochs = int(args.num_train_epochs)
        progress_path = self.cycle_dir / "progress.json"
        cycle_started_at = None
        try:
            existing = json.loads(progress_path.read_text())
            cycle_started_at = existing.get("cycle_started_at")
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        progress = {
            "epoch": current_epoch,
            "total_epochs": total_epochs,
            "keys": len(self.keyed_pairs),
            "cycle_started_at": cycle_started_at or time.time(),
        }
        with open(progress_path, "w") as f:
            json.dump(progress, f)

        # Only probe at intervals (and always at the final epoch)
        is_final = current_epoch == total_epochs
        if current_epoch % self.probe_every != 0 and not is_final:
            return

        self.model.gradient_checkpointing_disable()

        try:
            recall_result = evaluate_indexed_recall(
                self.model,
                self.tokenizer,
                self.keyed_pairs,
                self.registry,
                adapter_name=self.adapter_name,
            )
        except Exception:
            logger.exception("Recall probe failed at epoch %d", current_epoch)
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            return

        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        per_key = recall_result.get("per_key", [])
        confidences = [pk.get("confidence", 0.0) for pk in per_key]
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

        # Get loss from trainer state
        train_loss = None
        if state.log_history:
            for entry in reversed(state.log_history):
                if "loss" in entry:
                    train_loss = entry["loss"]
                    break

        entry = {
            "epoch": current_epoch,
            "train_loss": train_loss,
            "exact_recall": recall_result["exact_count"],
            "total": recall_result["total"],
            "recall_rate": recall_result["rate"],
            "mean_confidence": round(mean_conf, 4),
        }
        self.epoch_log.append(entry)

        logger.info(
            "  Epoch %2d: loss=%.4f  recall=%d/%d (%.1f%%)  conf=%.3f",
            current_epoch,
            train_loss or 0.0,
            recall_result["exact_count"],
            recall_result["total"],
            recall_result["rate"] * 100,
            mean_conf,
        )


# ============================================================================
# Training — full replay on all QA pairs
# ============================================================================


def train_cycle(model, tokenizer, qa_pairs, cycle_num, output_dir, num_epochs, rank):
    """Train adapter on all QA pairs with full replay.

    Unwraps to base model, creates fresh adapter, trains with per-epoch
    probing, evaluates.
    Returns (model, keyed_pairs, registry, recall_result, train_time,
             eval_time, train_loss, epoch_log, cycle_dir).
    """
    adapter_name = "episodic"
    cycle_dir = output_dir / f"cycle_{cycle_num:03d}"
    cycle_dir.mkdir(parents=True, exist_ok=True)

    # Anchor tstatus ETA math: per-epoch callback preserves this timestamp so
    # elapsed is measured from cycle start (not from dir mtime, which shifts
    # every time a checkpoint subdir is created).
    with open(cycle_dir / "progress.json", "w") as f:
        json.dump({"cycle": cycle_num, "cycle_started_at": time.time()}, f)

    # Unwrap to base model
    if isinstance(model, _PeftModel):
        model = model.base_model.model

    # Create adapter
    adapter_config = AdapterConfig(
        rank=rank,
        alpha=rank * 2,
        learning_rate=1e-4,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        dropout=0.0,
    )
    from paramem.models.loader import create_adapter

    model = create_adapter(model, adapter_config, adapter_name)

    # Assign keys and build registry
    keyed_pairs = assign_keys(qa_pairs, start_index=1)
    registry = build_registry(keyed_pairs)
    save_registry(registry, cycle_dir / "simhash_registry.json")

    # Save keyed_pairs with cohort metadata
    kp_ser = []
    for kp in keyed_pairs:
        entry = {"key": kp["key"], "question": kp["question"], "answer": kp["answer"]}
        for meta_key in (
            "source_predicate",
            "source_subject",
            "source_object",
            "first_seen_cycle",
            "source_character",
        ):
            if meta_key in kp:
                entry[meta_key] = kp[meta_key]
        kp_ser.append(entry)
    with open(cycle_dir / "keyed_pairs.json", "w") as f:
        json.dump(kp_ser, f, indent=2)

    # Format training data
    examples = format_indexed_training(keyed_pairs, tokenizer, max_length=1024)
    dataset = IndexedDataset(examples)

    # Create per-epoch recall callback
    recall_callback = ScaleRecallCallback(
        model,
        tokenizer,
        keyed_pairs,
        registry,
        adapter_name,
        cycle_dir=cycle_dir,
        probe_every=PROBE_EVERY_N_EPOCHS,
    )

    # Train
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
        save_strategy="no",
    )

    t0 = time.time()
    train_metrics = train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        adapter_name=adapter_name,
        training_config=training_config,
        adapter_config=adapter_config,
        output_dir=cycle_dir / "adapter",
        run_name=f"scale-cycle-{cycle_num:03d}",
        callbacks_extra=[recall_callback],
    )
    train_time = time.time() - t0

    train_loss = train_metrics.get("train_loss") if train_metrics else None

    # Save epoch log
    if recall_callback.epoch_log:
        with open(cycle_dir / "epoch_log.json", "w") as f:
            json.dump(recall_callback.epoch_log, f, indent=2)

    # Final recall evaluation (full per-key detail, separate from probing)
    t_eval = time.time()
    recall_result = evaluate_indexed_recall(
        model, tokenizer, keyed_pairs, registry, adapter_name=adapter_name
    )
    eval_time = time.time() - t_eval

    return (
        model,
        keyed_pairs,
        registry,
        recall_result,
        train_time,
        eval_time,
        train_loss,
        recall_callback.epoch_log,
        cycle_dir,
    )


# ============================================================================
# Cycle result formatting
# ============================================================================


def format_cycle_result(
    cycle_num,
    sessions_this_cycle,
    total_sessions,
    characters_used,
    merger,
    keyed_pairs,
    qa_pairs,
    new_qa,
    recall_result,
    train_time,
    train_loss,
    epoch_log,
    extraction_time,
    qa_gen_time,
    eval_time,
    cycle_time,
    num_epochs,
):
    """Build the result dict for one cycle."""
    per_key = recall_result.get("per_key", [])
    confidences = [pk.get("confidence", 0.0) for pk in per_key]

    # Build per-key results with cohort metadata (matched by key string)
    kp_by_key = {kp["key"]: kp for kp in keyed_pairs}
    per_key_results = []
    for pk in per_key:
        key = pk.get("key", "")
        kp = kp_by_key.get(key, {})
        recalled = pk.get("recalled") or {}
        per_key_results.append(
            {
                "key": key,
                "exact_match": pk.get("exact_match", False),
                "confidence": pk.get("confidence", 0.0),
                "raw_output": recalled.get("raw_output", ""),
                "recalled_answer": recalled.get("answer", ""),
                "failure_reason": recalled.get("failure_reason", ""),
                "first_seen_cycle": kp.get("first_seen_cycle"),
                "source_character": kp.get("source_character"),
            }
        )

    return {
        "cycle": cycle_num,
        "sessions_processed": sessions_this_cycle,
        "total_sessions": total_sessions,
        "characters_used": characters_used,
        "graph_nodes": merger.graph.number_of_nodes(),
        "graph_edges": merger.graph.number_of_edges(),
        "total_qa_pairs": len(qa_pairs),
        "new_qa_this_cycle": new_qa,
        "exact_recall": recall_result["exact_count"],
        "total_keys": recall_result["total"],
        "recall_rate": recall_result["rate"],
        "mean_confidence": recall_result.get("mean_confidence", 0.0),
        "min_confidence": min(confidences) if confidences else 0.0,
        "per_key_results": per_key_results,
        "train_loss": train_loss,
        "epoch_log": epoch_log,
        "train_time_seconds": round(train_time, 1),
        "eval_time_seconds": round(eval_time, 1),
        "extraction_time_seconds": round(extraction_time, 1),
        "qa_generation_time_seconds": round(qa_gen_time, 1),
        "total_cycle_time_seconds": round(cycle_time, 1),
        "epochs_run": num_epochs,
        "qa_yield_rate": (
            round(new_qa / sessions_this_cycle, 2) if sessions_this_cycle > 0 else 0.0
        ),
    }


# ============================================================================
# Main loop
# ============================================================================


def run_scale_test(model, tokenizer, args, output_dir, bench_name):
    """Run the large-scale incremental test."""
    batch_size = args.batch_size
    max_cycles = args.max_cycles
    num_epochs = args.num_epochs
    rank = args.rank
    target_keys = args.target_keys

    # Estimate and report disk usage upfront
    estimated_cycles, estimated_mb = estimate_disk_usage(target_keys, batch_size, num_epochs)
    print(
        f"  Estimated disk usage: ~{estimated_mb / 1024:.1f} GB"
        f" ({estimated_cycles} cycles × {ESTIMATED_CYCLE_DISK_MB} MB/cycle)"
    )

    ok, used_gb, free_gb = check_disk_space(output_dir, remaining_cycles=estimated_cycles)
    print(f"  Disk: {used_gb:.0f} GB used, {free_gb:.0f} GB free")
    if not ok:
        print("ERROR: Insufficient disk space. Clean up and retry.")
        return

    # Build character queue
    character_queue = build_character_queue(target_keys)
    if not character_queue:
        logger.error("No characters available in PerLTQA")
        return

    # Resume or fresh start
    state = None
    merger = GraphMerger(similarity_threshold=85.0, strategy="graph")

    if args.resume:
        # Clear stale pause file so resume doesn't immediately stop
        if PAUSE_FILE.exists():
            PAUSE_FILE.unlink()
            logger.info("Cleared stale pause file from previous run")

        state = load_state(output_dir)
        if state is not None:
            last_cycle = state["last_completed_cycle"]

            # Prefer per-cycle state snapshot for clean rollback.
            # The top-level state.json may have been manually edited;
            # the cycle snapshot is the authoritative state at that boundary.
            cycle_state_path = output_dir / f"cycle_{last_cycle:03d}" / "state.json"
            if cycle_state_path.exists():
                cycle_state = load_state(output_dir / f"cycle_{last_cycle:03d}")
                if cycle_state is not None:
                    state = cycle_state
                    # Re-read last_cycle from snapshot in case it differs
                    last_cycle = state["last_completed_cycle"]
                    logger.info(
                        "Loaded per-cycle state snapshot from cycle %d",
                        last_cycle,
                    )

            logger.info(
                "Resuming from cycle %d (%d sessions, %d QA pairs)",
                state["last_completed_cycle"],
                state["total_sessions_processed"],
                state["total_qa_pairs"],
            )
            # Load cumulative graph from last completed cycle
            if last_cycle > 0:
                graph_path = output_dir / f"cycle_{last_cycle:03d}" / "cumulative_graph.json"
                if graph_path.exists():
                    merger.load_graph(graph_path)
                    logger.info(
                        "Loaded graph: %d nodes, %d edges",
                        merger.graph.number_of_nodes(),
                        merger.graph.number_of_edges(),
                    )
                    # Reconstruct seen_triples from graph if missing (old state files)
                    if not state.get("seen_triples"):
                        state["seen_triples"] = set(merger.get_all_triples())
                        logger.info(
                            "Reconstructed seen_triples from graph: %d triples",
                            len(state["seen_triples"]),
                        )
                else:
                    logger.error("Graph file missing for cycle %d: %s", last_cycle, graph_path)
                    return
        else:
            logger.info("No state.json found, starting fresh")

    if state is None:
        state = fresh_state()

    # Save run config for reproducibility (only on fresh start)
    config_path = output_dir / "run_config.json"
    if not config_path.exists():
        run_config = {
            "model": bench_name,
            "batch_size": batch_size,
            "max_cycles": max_cycles,
            "num_epochs": num_epochs,
            "rank": rank,
            "target_keys": target_keys,
            "probe_every_n_epochs": PROBE_EVERY_N_EPOCHS,
            "character_queue": [(n, c) for n, c in character_queue],
            "estimated_disk_mb": estimated_mb,
        }
        with open(config_path, "w") as f:
            json.dump(run_config, f, indent=2)

    # Load existing cycle results if resuming
    all_cycle_results = []
    results_path = output_dir / "results.json"
    if args.resume and results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)
        all_cycle_results = existing.get("cycles", [])

    # Build session iterator starting from resume point
    sess_iter = session_iterator(
        character_queue,
        start_char=state.get("current_character"),
        start_session_idx=state.get("current_character_session_index", 0),
    )

    cycle_num = state["last_completed_cycle"]
    total_sessions = state["total_sessions_processed"]
    seen_questions = state["seen_questions"]
    seen_triples = state["seen_triples"]
    qa_first_seen_cycle = state.get("qa_first_seen_cycle", {})
    qa_source_character = state.get("qa_source_character", {})
    characters_used = list(state.get("characters_completed", []))

    # Track current character and session index directly
    current_char = state.get("current_character")
    current_char_session_idx = state.get("current_character_session_index", 0)

    print(f"\nStarting from cycle {cycle_num + 1}")
    print(f"  Sessions processed: {total_sessions}")
    print(f"  QA pairs: {state['total_qa_pairs']}")
    print(f"  Target: {target_keys} keys")
    print(f"  Batch size: {batch_size} sessions/cycle")
    print(f"  Max cycles: {max_cycles or 'unlimited'}")
    print(f"  Recall probing: every {PROBE_EVERY_N_EPOCHS} epochs")
    print()

    start_cycle = cycle_num  # snapshot for max_cycles limit

    while True:
        cycle_num += 1

        # Note if target already reached (continues anyway)
        if state["total_qa_pairs"] >= target_keys and cycle_num == start_cycle + 1:
            print(f"\n  Already at target ({state['total_qa_pairs']}/{target_keys} keys).")
            print("  Continuing beyond target. Use 'tpause' to stop when ready.")

        # Check disk space before each cycle (estimate remaining from current QA count)
        current_qa = state["total_qa_pairs"]
        remaining_keys = max(0, target_keys - current_qa)
        keys_per_cycle = batch_size * ESTIMATED_QA_PER_SESSION
        remaining = max(1, int(remaining_keys / keys_per_cycle) + 1)
        ok, used_gb, free_gb = check_disk_space(output_dir, remaining_cycles=remaining)
        if not ok:
            print(f"\n  Disk space insufficient at cycle {cycle_num}. Stopping.")
            break

        cycle_start = time.time()
        print(f"\n{'=' * 60}")
        print(f"  Cycle {cycle_num}")
        print(f"{'=' * 60}")

        # ---- Phase 1: Extract graphs from batch of sessions ----
        t_extract = time.time()
        sessions_this_cycle = 0

        for _ in range(batch_size):
            try:
                char_name, sess_idx, session = next(sess_iter)
            except StopIteration:
                logger.info("All sessions exhausted")
                break

            # Track character transitions directly
            if char_name != current_char:
                if current_char is not None and current_char not in characters_used:
                    characters_used.append(current_char)
                current_char = char_name
                current_char_session_idx = sess_idx
                logger.info("Switched to character: %s", char_name)
            else:
                current_char_session_idx = sess_idx

            logger.info(
                "Extracting session %s (character: %s, idx: %d)",
                session["session_id"],
                char_name,
                sess_idx,
            )
            session_graph = extract_session_graph(
                model, tokenizer, session["transcript"], session["session_id"]
            )

            rel_count = len(session_graph.relations) if session_graph.relations else 0
            logger.info(
                "  -> %d entities, %d relations",
                len(session_graph.entities) if session_graph.entities else 0,
                rel_count,
            )

            if rel_count > 0:
                merger.merge(session_graph)

            sessions_this_cycle += 1
            total_sessions += 1

        extraction_time = time.time() - t_extract

        if sessions_this_cycle == 0:
            logger.info("No sessions processed this cycle, stopping")
            break

        # Update character tracking
        if current_char not in characters_used:
            characters_used.append(current_char)

        print(
            f"  Extracted {sessions_this_cycle} sessions in {extraction_time:.0f}s"
            f" (graph: {merger.graph.number_of_nodes()} nodes,"
            f" {merger.graph.number_of_edges()} edges)"
        )

        # ---- Phase 2: Regenerate QA from full graph ----
        t_qa = time.time()
        qa_pairs, new_qa, new_triples_this_cycle = generate_qa_from_graph(
            model,
            tokenizer,
            merger,
            seen_questions,
            seen_triples,
            qa_first_seen_cycle,
            qa_source_character,
            cycle_num,
        )
        qa_gen_time = time.time() - t_qa

        print(f"  Generated {len(qa_pairs)} QA pairs ({new_qa} new) in {qa_gen_time:.0f}s")

        if not qa_pairs:
            logger.warning("No QA pairs generated this cycle, skipping training")
            # Update session pointer but do NOT advance last_completed_cycle —
            # no cycle directory was created, so resume would fail looking for
            # cycle_N/cumulative_graph.json that doesn't exist.
            state["skipped_cycles"].append({"cycle": cycle_num, "reason": "no_qa_pairs"})
            state["current_character"] = current_char
            state["current_character_session_index"] = current_char_session_idx + 1
            state["total_sessions_processed"] = total_sessions
            state["characters_completed"] = [c for c in characters_used if c != current_char]
            save_state(state, output_dir)
            if is_paused():
                print(f"\n  Pause requested. Stopping after cycle {cycle_num}.")
                break
            continue

        # Skip training if no new triples — avoid wasting GPU time retraining
        # on identical data. The existing adapter already has 100% recall.
        if new_qa == 0:
            print("  No new triples this cycle, skipping training")
            logger.info(
                "Skipping training: 0 new triples (total QA unchanged at %d)",
                len(qa_pairs),
            )
            # Save graph in case new edges were added (even if no new triples
            # survived dedup). Prevents seen_triples drifting ahead of graph
            # on crash-resume.
            cycle_dir = output_dir / f"cycle_{cycle_num:03d}"
            cycle_dir.mkdir(parents=True, exist_ok=True)
            merger.save_graph(cycle_dir / "cumulative_graph.json")
            state["skipped_cycles"].append({"cycle": cycle_num, "reason": "no_new_triples"})
            state["current_character"] = current_char
            state["current_character_session_index"] = current_char_session_idx + 1
            state["total_sessions_processed"] = total_sessions
            state["total_qa_pairs"] = len(qa_pairs)
            state["characters_completed"] = [c for c in characters_used if c != current_char]
            save_state(state, output_dir)
            if is_paused():
                print(f"\n  Pause requested. Stopping after cycle {cycle_num}.")
                break
            continue

        # Warn if yield is low
        yield_rate = new_qa / sessions_this_cycle if sessions_this_cycle > 0 else 0
        if yield_rate < 1.0:
            logger.warning(
                "Low QA yield: %.1f pairs/session (expected ~1.8)",
                yield_rate,
            )

        # Save cumulative graph before training — if training crashes,
        # the graph is already on disk for safe resume
        cycle_dir = output_dir / f"cycle_{cycle_num:03d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        merger.save_graph(cycle_dir / "cumulative_graph.json")

        # ---- Phase 3: Train on all QA pairs (full replay) ----
        print(f"  Training on {len(qa_pairs)} keys, {num_epochs} epochs...")
        (
            model,
            keyed_pairs,
            registry,
            recall_result,
            train_time,
            eval_time,
            train_loss,
            epoch_log,
            cycle_dir,
        ) = train_cycle(model, tokenizer, qa_pairs, cycle_num, output_dir, num_epochs, rank)

        cycle_time = time.time() - cycle_start

        # Build cycle result
        cycle_result = format_cycle_result(
            cycle_num=cycle_num,
            sessions_this_cycle=sessions_this_cycle,
            total_sessions=total_sessions,
            characters_used=list(characters_used),
            merger=merger,
            keyed_pairs=keyed_pairs,
            qa_pairs=qa_pairs,
            new_qa=new_qa,
            recall_result=recall_result,
            train_time=train_time,
            train_loss=train_loss,
            epoch_log=epoch_log,
            extraction_time=extraction_time,
            qa_gen_time=qa_gen_time,
            eval_time=eval_time,
            cycle_time=cycle_time,
            num_epochs=num_epochs,
        )

        # Save cycle result
        with open(cycle_dir / "cycle_results.json", "w") as f:
            json.dump(cycle_result, f, indent=2, default=str)

        all_cycle_results.append(cycle_result)

        # Print cycle summary
        exact = recall_result["exact_count"]
        total = recall_result["total"]
        rate = recall_result["rate"]
        loss_str = f"{train_loss:.4f}" if train_loss is not None else "n/a"
        print(
            f"  Recall: {exact}/{total} ({rate * 100:.1f}%)"
            f"  |  Loss: {loss_str}"
            f"  |  QA yield: {yield_rate:.1f}/session"
            f"  |  Time: {cycle_time:.0f}s"
        )

        # Milestone reporting
        for milestone in sorted(MILESTONES):
            if len(qa_pairs) >= milestone and state["total_qa_pairs"] < milestone:
                print(f"\n  *** MILESTONE: {milestone} keys reached ***")
                print(f"      Cycles: {cycle_num}")
                print(f"      Sessions: {total_sessions}")
                print(f"      Characters: {len(characters_used)}")
                print(f"      Recall: {exact}/{total} ({rate * 100:.1f}%)")
                print(f"      Graph: {merger.graph.number_of_nodes()} nodes")
                break

        # Update state — direct tracking, no arithmetic reconstruction
        state["last_completed_cycle"] = cycle_num
        state["current_character"] = current_char
        state["current_character_session_index"] = current_char_session_idx + 1
        state["characters_completed"] = [
            c
            for c in characters_used
            if c != current_char  # current character may not be fully done
        ]
        state["total_sessions_processed"] = total_sessions
        state["total_qa_pairs"] = len(qa_pairs)
        state["qa_first_seen_cycle"] = qa_first_seen_cycle
        state["qa_source_character"] = qa_source_character
        # Commit new triples only after training succeeds
        seen_triples.update(new_triples_this_cycle)
        save_state(state, output_dir)

        # Per-cycle state snapshot for clean rollback
        save_state(state, cycle_dir)

        # Save cumulative results
        results = {
            "experiment": "test8_large_scale",
            "model": bench_name,
            "target_keys": target_keys,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "rank": rank,
            "probe_every_n_epochs": PROBE_EVERY_N_EPOCHS,
            "total_cycles": cycle_num,
            "total_sessions": total_sessions,
            "total_qa_pairs": len(qa_pairs),
            "characters_used": list(characters_used),
            "final_recall_rate": rate,
            "cycles": all_cycle_results,
        }
        save_results(results, output_dir)

        # ---- Check stopping conditions before cooldown ----
        if state["total_qa_pairs"] >= target_keys:
            print(f"\n  *** TARGET REACHED: {state['total_qa_pairs']}/{target_keys} keys ***")
            print("  Continuing beyond target. Use 'tpause' to stop when ready.")

        if max_cycles and (cycle_num - start_cycle) >= max_cycles:
            logger.info("Reached max_cycles (%d), stopping", max_cycles)
            break

        if is_paused():
            print(f"\n  Pause requested. Stopping after cycle {cycle_num}.")
            print("  Resume with: python experiments/test8_large_scale.py --model mistral --resume")
            break

        # GPU cooldown between cycles (only if continuing)
        print("  Cooling down...")
        wait_for_cooldown(45)

    # Clean up pause file — run is finished regardless of how the loop exited
    if PAUSE_FILE.exists():
        PAUSE_FILE.unlink()

    # Final summary
    if all_cycle_results:
        final = all_cycle_results[-1]
        print(f"\n{'=' * 60}")
        print(f"  Scale Test Summary — {bench_name}")
        print(f"{'=' * 60}")
        print(f"  Cycles completed: {cycle_num}")
        print(f"  Sessions processed: {total_sessions}")
        print(f"  Characters used: {len(characters_used)}")
        print(f"  Total QA pairs: {final['total_qa_pairs']}")
        print(
            f"  Final recall: {final['exact_recall']}/{final['total_keys']}"
            f" ({final['recall_rate'] * 100:.1f}%)"
        )
        print(f"  Graph: {final['graph_nodes']} nodes, {final['graph_edges']} edges")
        print(f"  Results: {output_dir / 'results.json'}")
        print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Test 8: Large-Scale Incremental Indexed Key Retrieval"
    )
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--rank", type=int, default=RANK)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Sessions per cycle (default: 5)",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Maximum number of cycles (default: unlimited)",
    )
    parser.add_argument(
        "--target-keys",
        type=int,
        default=DEFAULT_TARGET_KEYS,
        help="Target number of indexed keys (default: 500)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last completed cycle",
    )
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_BASE))
    add_model_args(parser)
    args = parser.parse_args()

    if not perltqa_available():
        print("ERROR: PerLTQA dataset required. Place it in data/external/PerLTQA/")
        sys.exit(1)

    base_output_dir = Path(args.output_dir)

    for bench_name, bench_model_config in get_benchmark_models(args):
        print(f"\n{'=' * 72}")
        print(f"  Test 8: Large-Scale Incremental — {bench_name}")
        print(f"  Target: {args.target_keys} keys")
        print(f"  Batch: {args.batch_size} sessions/cycle")
        print(f"  Epochs: {args.num_epochs}, Rank: {args.rank}")
        print(f"  Recall probing: every {PROBE_EVERY_N_EPOCHS} epochs")
        print(f"{'=' * 72}")

        model, tokenizer = load_model_and_config(bench_model_config)

        if args.resume:
            # For resume, find the most recent output dir for this model
            model_dir = base_output_dir / bench_name
            if model_dir.exists():
                # Find latest timestamped dir
                subdirs = sorted(
                    [d for d in model_dir.iterdir() if d.is_dir()],
                    key=lambda d: d.name,
                    reverse=True,
                )
                if subdirs and (subdirs[0] / "state.json").exists():
                    output_dir = subdirs[0]
                    print(f"  Resuming from: {output_dir}")
                else:
                    print("  No resumable state found, starting fresh")
                    output_dir = model_output_dir(base_output_dir, bench_name)
            else:
                print("  No previous runs found, starting fresh")
                output_dir = model_output_dir(base_output_dir, bench_name)
        else:
            output_dir = model_output_dir(base_output_dir, bench_name)

        output_dir.mkdir(parents=True, exist_ok=True)

        run_scale_test(model, tokenizer, args, output_dir, bench_name)

        from paramem.models.loader import unload_model

        unload_model(model, tokenizer)


if __name__ == "__main__":
    from experiments.utils.gpu_guard import acquire_gpu

    with acquire_gpu():
        main()
