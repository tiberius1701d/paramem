"""Test 15: Retention Multi-Seed (Scaffold-Fill vs Answer-Swap, Production Early-Stop).

Research question
-----------------
Does the 6.7x retention advantage for the scaffold-then-fill path over the
no-scaffold answer-swap path (Test 13, n=1) hold across multiple seeds under
the production ``RecallEarlyStopCallback`` (``ANALYSIS_POLICY``)?

Decision rule (headline)
------------------------
``ratio_raw = mean(C2 RP2) / mean(B RP2) >= 5.0`` AND bootstrap
``lower_CI >= 2.5`` (B=10 000 resamples over 5 seeds).
Fail: cut Test 14a (the N=500 scale follow-up).

Phases per seed
---------------
A  — Fresh ``episodic`` adapter on N real Q+A pairs.
B  — Continue ``episodic``, overwrite swap_keys with different answers.
     Retention probe on unchanged 80 per epoch.  RP2 = stop-epoch retention.
     Repair loop: up to max_repair_episodes x 1-epoch LR=1e-5 on failing keys.
C1 — Fresh ``journal`` adapter on N keys: (N-swap) real + swap TBD-k.
C2 — Continue ``journal``, replace TBD-k with real answers.
     Retention probe on unchanged 80 per epoch.  RP2 = stop-epoch retention.
     Repair loop: same as B.

All phases use ``RecallEarlyStopCallback`` with ``ANALYSIS_POLICY``.

Pause / resume
--------------
``tpause`` / ``tresume 15`` / ``tstatus`` via training-control.sh registry.
Three pause sites:
  1. Between seeds.
  2. Between phases within a seed.
  3. Mid-phase (callback detects pause file, fires should_training_stop).

GPU prerequisite
----------------
The ParaMem server must release the GPU.  This script uses
``experiments.utils.gpu_guard.acquire_gpu()`` which auto-switches the server
to cloud-only for the duration.

Usage
-----
    python experiments/test15_retention_multiseed.py \
        --model mistral --n-keys 10 --swap-keys 2 --seeds 42 \
        --phase-a-num-epochs 20 --phase-b-num-epochs 15 \
        --phase-c1-num-epochs 20 --phase-c2-num-epochs 15 \
        --max-repair-episodes 2
    python experiments/test15_retention_multiseed.py --resume
    python experiments/test15_retention_multiseed.py --seeds 42 7 1337 1 11
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np  # noqa: E402
from peft import PeftModel  # noqa: E402

from experiments.utils.early_stop import (  # noqa: E402
    ANALYSIS_POLICY,
    EarlyStopPolicy,
    RecallEarlyStopCallback,
    _EarlyStopState,
    _safe_write_json,
)
from experiments.utils.gpu_guard import acquire_gpu  # noqa: E402
from archive.experiments.legacy_harness import evaluate_indexed_recall  # noqa: E402
from experiments.utils.test_harness import (  # noqa: E402
    BENCHMARK_MODELS,
    IndexedDataset,
    load_model_and_config,
    model_output_dir,
    setup_logging,
)
from paramem.adapters import resolve_adapter_slot  # noqa: E402
from paramem.adapters.manifest import build_manifest_for  # noqa: E402
from paramem.models.loader import (  # noqa: E402
    _adapter_slot_for_load,
    create_adapter,
    save_adapter,
    switch_adapter,
    unload_model,
)
from archive.legacy_qa import (  # noqa: E402
    assign_keys,
    build_registry,
    format_indexed_training,
)
from paramem.memory.persistence import save_registry  # noqa: E402
from paramem.training.trainer import train_adapter  # noqa: E402
from paramem.utils.config import AdapterConfig, TrainingConfig  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_BASE = project_root / "outputs" / "test15_retention_multiseed"
PAUSE_FILE = Path.home() / ".training_pause"

DEFAULT_SEEDS = [42, 7, 1337, 1, 11]
DEFAULT_N_KEYS = 100
DEFAULT_SWAP_KEYS = 20
DEFAULT_RANK = 8
DEFAULT_LR = 1e-4
REPAIR_LR = 1e-5

DEFAULT_PHASE_A_EPOCHS = 50
DEFAULT_PHASE_B_EPOCHS = 30
DEFAULT_PHASE_C1_EPOCHS = 50
DEFAULT_PHASE_C2_EPOCHS = 30
DEFAULT_MAX_REPAIR_EPISODES = 5

# Decision thresholds (benchmarking.md §Test 15)
DECISION_THRESHOLD_RATIO = 5.0
DECISION_THRESHOLD_LOWER_CI = 2.5
BOOTSTRAP_RESAMPLES = 10_000

DISK_HEADROOM_BYTES = 15 * 1024**3

# Number of HF Trainer checkpoints retained per phase.  Keeps the most recent
# plus one backup so _find_latest_checkpoint can always resume safely, even if
# a pause arrives mid-save.  Using num_epochs here would retain one checkpoint
# per epoch (e.g. 20 dirs × 5 seeds × ~50 MB = 5 GB just for Phase A).
CHECKPOINT_RETENTION = 2


# ---------------------------------------------------------------------------
# Probe state
# ---------------------------------------------------------------------------


@dataclass
class EpochProbeState:
    """Per-phase training metrics accumulator (mirrors test14 pattern)."""

    first_perfect_epoch: int | None = None
    stable_perfect_epoch: int | None = None
    stop_epoch: int | None = None
    epoch_log: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pause / marker helpers  (verbatim from test14.py)
# ---------------------------------------------------------------------------


def paused_requested() -> bool:
    """Return True if the global pause file exists."""
    return PAUSE_FILE.exists()


def _check_pause(label: str, run_dir: Path | None = None) -> None:
    """Raise SystemExit with a clean message if pause file is present.

    Writes paused.json BEFORE raising so ``tstatus`` can identify the paused
    state on resume.  The write is attempted even if ``run_dir`` is None (in
    which case the marker write is skipped silently).

    Args:
        label: Human-readable phase label for log and paused.json.
        run_dir: Root run directory for writing ``paused.json``.
    """
    if PAUSE_FILE.exists():
        logger.warning("Pause file detected at %s — halting cleanly.", label)
        if run_dir is not None:
            write_paused_marker(run_dir, label)
        raise SystemExit(f"Training paused at {label}")


def marker_exists(run_dir: Path, marker_name: str) -> bool:
    """Return True if <run_dir>/<marker_name>_done.json exists."""
    return (run_dir / f"{marker_name}_done.json").exists()


def _exit_if_paused_mid_phase(
    probe_state: EpochProbeState,
    phase_label: str,
    num_epochs: int,
    run_dir: Path,
) -> None:
    """Halt cleanly when training was stopped mid-phase by the pause flag.

    Inference rule: a phase has completed iff the early-stop callback fired
    (``stop_epoch is not None``) OR the trainer ran the full epoch budget.
    Anything else means ``RecallEarlyStopCallback`` set
    ``should_training_stop=True`` due to the ``~/.training_pause`` flag.

    When this fires, the phase has NOT finished — partial checkpoints exist on
    disk and the next tresume will pick them up via ``_find_latest_checkpoint``.
    Writes ``paused.json`` with a ``during X`` label so ``tstatus`` shows where
    training stopped.  No ``*_done.json`` marker is written.

    Args:
        probe_state: EpochProbeState accumulator from the phase.
        phase_label: Human-readable label for log and paused.json.
        num_epochs: Total epoch budget for this phase.
        run_dir: Root run directory for writing ``paused.json``.
    """
    if probe_state.stop_epoch is not None:
        return  # natural convergence
    if not probe_state.epoch_log:
        return  # nothing started; outer _check_pause will handle it
    last_epoch = probe_state.epoch_log[-1].get("epoch")
    if last_epoch is None:
        return
    if last_epoch >= num_epochs:
        return  # ran the full budget without convergence
    logger.warning(
        "Phase paused mid-training: %s at epoch %d — no done marker written; "
        "checkpoint preserved for tresume.",
        phase_label,
        last_epoch,
    )
    write_paused_marker(run_dir, phase_label, after_epoch=last_epoch)
    raise SystemExit(f"Training paused {phase_label} at epoch {last_epoch}")


def write_paused_marker(run_dir: Path, after_phase: str, after_epoch: int | None = None) -> None:
    """Write paused.json at clean pause exit.

    Args:
        run_dir: Run directory root.
        after_phase: Label of the phase/checkpoint where pause occurred.
        after_epoch: Epoch number if paused mid-phase; None for boundary pauses.
    """
    marker = {
        "stopped_after_phase": after_phase,
        "stopped_after_epoch": after_epoch,
        "timestamp": int(time.time()),
    }
    _safe_write_json(run_dir / "paused.json", marker)
    logger.info("paused.json written (after_phase=%s)", after_phase)


def clear_paused_marker(run_dir: Path) -> None:
    """Remove paused.json if present (called at run start / tresume).

    Args:
        run_dir: Run directory root.
    """
    p = run_dir / "paused.json"
    if p.exists():
        try:
            p.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Checkpoint finder
# ---------------------------------------------------------------------------


def _find_latest_checkpoint(phase_dir: Path) -> Path | None:
    """Return the checkpoint-N dir with the highest step number in phase_dir.

    Args:
        phase_dir: Directory that contains ``checkpoint-*`` subdirectories.

    Returns:
        Path to the highest-numbered checkpoint, or None if none found.
    """
    candidates = [Path(p) for p in glob(str(phase_dir / "checkpoint-*")) if Path(p).is_dir()]
    if not candidates:
        return None

    def _step(p: Path) -> int:
        try:
            return int(p.name.split("-", 1)[1])
        except (IndexError, ValueError):
            return -1

    # Walk in descending step order; skip partial dirs missing trainer_state.json
    # (can occur after a SIGKILL / system bugcheck mid-save).
    for candidate in sorted(candidates, key=_step, reverse=True):
        if (candidate / "trainer_state.json").exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Run-dir helpers
# ---------------------------------------------------------------------------


def find_latest_run_dir(model_name: str) -> Path | None:
    """Return the most recent run dir for the given model.

    The cfg-restore block in ``main()`` reads back persisted parameters from
    the returned directory.

    Args:
        model_name: Model name key (e.g. ``"mistral"``).

    Returns:
        Path to the latest run dir, or None if no run dirs exist.
    """
    parent = OUTPUT_BASE / model_name
    if not parent.is_dir():
        return None
    candidates: list[Path] = [d for d in sorted(parent.iterdir()) if d.is_dir()]
    return candidates[-1] if candidates else None


def load_or_write_run_config(run_dir: Path, args: argparse.Namespace) -> dict:
    """Read run_config.json if it exists; otherwise write it from args and return it.

    Args:
        run_dir: Run directory root (created if absent).
        args: Parsed CLI namespace.

    Returns:
        Config dict (either freshly written or read from disk).
    """
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        logger.info("Loaded run_config.json: %s", cfg)
        return cfg

    # First launch — freeze config.
    n_keys = args.n_keys
    swap_keys = args.swap_keys
    lr_decay_steps = (
        args.lr_decay_steps
        if args.lr_decay_steps is not None
        else decay_steps_for(n_keys, max(args.phase_a_num_epochs, args.phase_b_num_epochs))
    )
    cfg: dict = {
        "model": args.model,
        "seeds": list(args.seeds),
        "n_keys": n_keys,
        "swap_keys": swap_keys,
        "phase_a_num_epochs": args.phase_a_num_epochs,
        "phase_b_num_epochs": args.phase_b_num_epochs,
        "phase_c1_num_epochs": args.phase_c1_num_epochs,
        "phase_c2_num_epochs": args.phase_c2_num_epochs,
        "lr_scheduler_type": args.lr_scheduler_type,
        "lr_decay_steps": lr_decay_steps,
        "weight_decay": args.weight_decay,
        "max_repair_episodes": args.max_repair_episodes,
        "repair_lr": args.repair_lr,
        "rank": DEFAULT_RANK,
        "lr": DEFAULT_LR,
        "created_at": int(time.time()),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2))
    logger.info("Wrote run_config.json: %s", cfg)
    return cfg


# ---------------------------------------------------------------------------
# Decay-steps formula
# ---------------------------------------------------------------------------


def decay_steps_for(n_keys: int, num_epochs: int) -> int:
    """Compute LR decay steps for the apples-to-apples linear scheduler.

    Formula: ``n_keys * num_epochs // 2`` (benchmarking.md §Test 15 provenance).
    Validated by Test 14's multi-seed cells (2026-05-04).

    Args:
        n_keys: Number of indexed keys being trained.
        num_epochs: Epoch budget for the phase.

    Returns:
        Integer number of optimizer steps for LR decay window.
    """
    return max(1, n_keys * num_epochs // 2)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _adapter_config(rank: int = DEFAULT_RANK, lr: float = DEFAULT_LR) -> AdapterConfig:
    """Return a standard AdapterConfig for Test 15.

    Args:
        rank: LoRA rank (default 8).
        lr: Learning rate (default 1e-4).

    Returns:
        AdapterConfig with production settings.
    """
    return AdapterConfig(
        rank=rank,
        alpha=rank * 2,
        learning_rate=lr,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        dropout=0.0,
    )


def _training_config(
    num_epochs: int,
    seed: int = 42,
    lr_scheduler_type: str = "linear",
    lr_decay_steps: int | None = None,
    weight_decay: float = 0.01,
    save_strategy: str = "epoch",
) -> TrainingConfig:
    """Return a standard TrainingConfig for Test 15.

    Mirrors Test 14's apples-to-apples scheduler: linear + warmup_steps=10 +
    lr_decay_steps decoupled from epoch count.

    Args:
        num_epochs: Training epoch budget.
        seed: HF Trainer seed for optimizer / data shuffler.
        lr_scheduler_type: LR scheduler type (default ``"linear"``).
        lr_decay_steps: Decoupled decay window; None disables override.
        weight_decay: L2 regularisation (default 0.01 per benchmarking.md §Test 15).
        save_strategy: Checkpointing strategy (default ``"epoch"``).

    Returns:
        TrainingConfig with production settings.
    """
    cfg_kwargs: dict = dict(
        batch_size=1,
        gradient_accumulation_steps=2,
        max_seq_length=1024,
        num_epochs=num_epochs,
        warmup_ratio=0.0,
        warmup_steps=10,
        weight_decay=weight_decay,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=seed,
        save_strategy=save_strategy,
        save_total_limit=CHECKPOINT_RETENTION,
    )
    if lr_scheduler_type is not None:
        cfg_kwargs["lr_scheduler_type"] = lr_scheduler_type
    if lr_decay_steps is not None and lr_decay_steps > 0:
        cfg_kwargs["lr_decay_steps"] = lr_decay_steps
    return TrainingConfig(**cfg_kwargs)


# ---------------------------------------------------------------------------
# Dataset / phase builders  (parameterised versions of test13's builders)
# ---------------------------------------------------------------------------


def load_qa_pool(total_keys: int, swap_keys: int) -> list[dict]:
    """Load ``total_keys + swap_keys`` unique QA pairs from PerLTQA.

    Deduplicates on both question and answer strings (same logic as
    ``experiments.test13_journal_scaffold.load_qa_pool`` but parameterised for
    arbitrary ``total_keys`` and ``swap_keys``).

    Sources ``Liang Xin`` first, ``Cai Xiuying`` as fallback, then remaining
    PerLTQA characters in deterministic alphabetical order.

    Args:
        total_keys: Number of indexed keys in the main pool.
        swap_keys: Number of additional answer-swap / placeholder keys.

    Returns:
        List of ``total_keys + swap_keys`` unique QA dicts.

    Raises:
        RuntimeError: If PerLTQA dataset is unavailable or the pool is too small.
    """
    from experiments.utils.perltqa_loader import is_available as perltqa_available
    from experiments.utils.perltqa_loader import list_characters, load_character_eval_qa

    needed = total_keys + swap_keys
    if not perltqa_available():
        raise RuntimeError("PerLTQA dataset not available — required for Test 15")

    primary = ["Liang Xin", "Cai Xiuying"]
    try:
        remaining = sorted(c for c in list_characters() if c not in primary)
    except Exception:  # noqa: BLE001
        remaining = []
    source_order = primary + remaining

    seen_q: set[str] = set()
    seen_a: set[str] = set()
    dropped_collisions = 0
    pool: list[dict] = []

    for char in source_order:
        if len(pool) >= needed:
            break
        batch = load_character_eval_qa(char, max_pairs=needed * 2)
        for pair in batch:
            q = pair.get("question", "").strip()
            a = pair.get("answer", "").strip()
            if not q or not a:
                continue
            if q in seen_q or a in seen_a:
                dropped_collisions += 1
                continue
            seen_q.add(q)
            seen_a.add(a)
            pool.append(pair)
            if len(pool) >= needed:
                break

    if len(pool) < needed:
        raise RuntimeError(
            f"Need {needed} unique QA pairs, got {len(pool)} after dedup "
            f"(dropped {dropped_collisions} collisions). Widen sources."
        )

    if dropped_collisions:
        logger.info(
            "load_qa_pool: dropped %d colliding pairs; kept %d unique from %d sources",
            dropped_collisions,
            len(pool),
            len(source_order),
        )
    return pool[:needed]


def build_phase_A_keyed(qa_pool: list[dict], total_keys: int = DEFAULT_N_KEYS) -> list[dict]:
    """Phase A: ``total_keys`` keys, real (Q, A).

    Args:
        qa_pool: Full QA pool (at least ``total_keys`` entries).
        total_keys: Number of indexed keys.

    Returns:
        List of keyed dicts (key, question, answer).
    """
    return assign_keys(qa_pool[:total_keys], start_index=1)


def build_phase_B_swap_keyed(
    base_keyed: list[dict],
    swap_answers: list[dict],
    swap_keys: int = DEFAULT_SWAP_KEYS,
    total_keys: int = DEFAULT_N_KEYS,
) -> list[dict]:
    """Phase B swap set: ``swap_keys`` keys, same key + Q, different A.

    The last ``swap_keys`` slots of ``base_keyed`` are the swap targets.
    ``swap_answers`` supplies the replacement answer text.

    Args:
        base_keyed: Phase A keyed pairs (all ``total_keys``).
        swap_answers: Replacement QA pool (only ``"answer"`` field used).
        swap_keys: Number of keys to swap.
        total_keys: Total keys (used to derive the swap start slot).

    Returns:
        List of ``swap_keys`` keyed dicts with swapped answers.
    """
    swap_start = total_keys - swap_keys
    assert len(swap_answers) >= swap_keys, (
        f"swap_answers has {len(swap_answers)} entries, need >= {swap_keys}"
    )
    swap_keyed = []
    for i, kp in enumerate(base_keyed[swap_start:]):
        replacement = swap_answers[i]["answer"]
        if replacement.strip() == kp["answer"].strip():
            replacement = replacement + " (variant)"
        swap_keyed.append(
            {
                "key": kp["key"],
                "question": kp["question"],
                "answer": replacement,
            }
        )
    return swap_keyed


def build_phase_C1_keyed(
    qa_pool: list[dict],
    total_keys: int = DEFAULT_N_KEYS,
    swap_keys: int = DEFAULT_SWAP_KEYS,
) -> list[dict]:
    """Phase C1: ``(total_keys - swap_keys)`` real + ``swap_keys`` TBD-k placeholder keys.

    Args:
        qa_pool: Full QA pool (at least ``total_keys`` entries).
        total_keys: Total number of indexed keys.
        swap_keys: Number of TBD placeholder slots at the end.

    Returns:
        List of keyed dicts (real Q+A for first slots, TBD-k for last slots).
    """
    swap_start = total_keys - swap_keys
    mixed = []
    for i, qa in enumerate(qa_pool[:total_keys]):
        if i < swap_start:
            mixed.append(qa)
        else:
            slot_k = i - swap_start + 1  # 1-indexed
            mixed.append(
                {
                    "question": qa["question"],
                    "answer": f"TBD-{slot_k}",
                }
            )
    return assign_keys(mixed, start_index=1)


def build_phase_C2_fill_keyed(
    c1_keyed: list[dict],
    qa_pool: list[dict],
    swap_keys: int = DEFAULT_SWAP_KEYS,
    total_keys: int = DEFAULT_N_KEYS,
) -> list[dict]:
    """Phase C2: replace each TBD-k in C1 with the corresponding real answer.

    Args:
        c1_keyed: C1 keyed pairs (the scaffold keys).
        qa_pool: Full QA pool (source of real answers for swap slots).
        swap_keys: Number of placeholder slots.
        total_keys: Total number of indexed keys.

    Returns:
        List of ``swap_keys`` keyed dicts with real answers replacing TBD-k.
    """
    swap_start = total_keys - swap_keys
    fill_keyed = []
    for i, kp in enumerate(c1_keyed[swap_start:]):
        fill_keyed.append(
            {
                "key": kp["key"],
                "question": kp["question"],
                "answer": qa_pool[swap_start + i]["answer"],
            }
        )
    return fill_keyed


# ---------------------------------------------------------------------------
# Safe probe helper
# ---------------------------------------------------------------------------


def _safe_probe(
    model,
    tokenizer,
    keyed: list[dict],
    registry: dict[str, int],
    adapter_name: str,
) -> dict:
    """Call evaluate_indexed_recall with guaranteed gradient_checkpointing re-enable.

    ``evaluate_indexed_recall`` internally disables gradient checkpointing
    before ``model.generate()``.  This helper re-enables in the ``finally``
    clause so every call site is safe regardless of the exception path.

    Args:
        model: Active PeftModel.
        tokenizer: Tokenizer.
        keyed: Key/QA pairs to probe.
        registry: SimHash registry for keyed pairs.
        adapter_name: Active adapter name.

    Returns:
        Result dict from ``evaluate_indexed_recall``.
    """
    try:
        return evaluate_indexed_recall(model, tokenizer, keyed, registry, adapter_name=adapter_name)
    finally:
        try:
            model.gradient_checkpointing_enable()
        except Exception as exc:  # noqa: BLE001
            logger.warning("gradient_checkpointing_enable() failed in _safe_probe: %s", exc)


# ---------------------------------------------------------------------------
# Phase done marker writer
# ---------------------------------------------------------------------------


def _write_phase_done(
    phase_dir: Path,
    marker_name: str,
    keyed: list[dict],
    registry: dict[str, int],
    probe_state: EpochProbeState,
    final_recall: dict,
    extra: dict,
) -> None:
    """Persist quads, registry, and phase-done marker.

    Args:
        phase_dir: Directory for this phase.
        marker_name: Filename for the done marker JSON (e.g. ``"A_done.json"``).
        keyed: Keyed pairs trained in this phase.
        registry: SimHash registry for keyed.
        probe_state: EpochProbeState accumulator.
        final_recall: Final recall probe result dict.
        extra: Additional fields merged into the marker.
    """
    phase_dir.mkdir(parents=True, exist_ok=True)
    _safe_write_json(
        phase_dir / "quads.json",
        [{"key": kp["key"], "question": kp["question"], "answer": kp["answer"]} for kp in keyed],
    )
    save_registry(registry, phase_dir / "simhash_registry.json")

    marker: dict = {
        "n_keys": len(keyed),
        "first_perfect_epoch": probe_state.first_perfect_epoch,
        "stable_perfect_epoch": probe_state.stable_perfect_epoch,
        "stop_epoch": probe_state.stop_epoch,
        "final_recall": {
            "exact_count": final_recall["exact_count"],
            "total": final_recall["total"],
            "rate": final_recall["rate"],
            "mean_confidence": final_recall["mean_confidence"],
        },
        **extra,
    }
    _safe_write_json(phase_dir / marker_name, marker)
    logger.info("Phase marker written: %s", phase_dir / marker_name)


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------


def _run_phase(
    model,
    tokenizer,
    keyed_to_train: list[dict],
    target_keyed: list[dict],
    target_registry: dict[str, int],
    adapter_name: str,
    phase_dir: Path,
    phase_name: str,
    num_epochs: int,
    seed: int,
    lr_scheduler_type: str,
    lr_decay_steps: int | None,
    weight_decay: float,
    early_stop_policy: EarlyStopPolicy,
    run_name: str,
    retention_keyed: list[dict] | None = None,
    retention_registry: dict[str, int] | None = None,
    resume_from_checkpoint: Path | None = None,
) -> tuple[object, dict, EpochProbeState, float]:
    """Run one train_adapter call with RecallEarlyStopCallback.

    This is the single phase runner used by all four phases.  Dual-probe
    (fill + retention) is activated when retention_keyed is provided.

    Args:
        model: PeftModel with the adapter already created and active.
        tokenizer: Tokenizer.
        keyed_to_train: Keyed pairs for training.
        target_keyed: Fill probing target (stop trigger).
        target_registry: SimHash registry for target_keyed.
        adapter_name: Active adapter name.
        phase_dir: Output directory for this phase.
        phase_name: Human-readable phase label.
        num_epochs: Epoch budget.
        seed: HF Trainer seed.
        lr_scheduler_type: LR scheduler type.
        lr_decay_steps: Decoupled decay steps; None disables override.
        weight_decay: L2 regularisation.
        early_stop_policy: EarlyStopPolicy controlling probe schedule.
        run_name: WandB / HF run name.
        retention_keyed: Optional unchanged-key list for dual-probe logging.
        retention_registry: Required if retention_keyed is provided.
        resume_from_checkpoint: Checkpoint path for tresume; None for fresh.

    Returns:
        Tuple of (model, metrics_dict, probe_state, wall_seconds).
    """
    examples = format_indexed_training(keyed_to_train, tokenizer, max_length=1024)
    dataset = IndexedDataset(examples)

    probe_state = EpochProbeState()
    early_state = _EarlyStopState()

    callback = RecallEarlyStopCallback(
        model=model,
        tokenizer=tokenizer,
        target_keyed=target_keyed,
        target_registry=target_registry,
        adapter_name=adapter_name,
        policy=early_stop_policy,
        state_out=early_state,
        progress_path=phase_dir / "progress.json",
        epoch_log_path=phase_dir / "epoch_log.json",
        first_perfect_log_path=phase_dir / "first_perfect_log.json",
        phase_name=phase_name,
        num_epochs=num_epochs,
        pause_file=PAUSE_FILE,
        retention_keyed=retention_keyed if retention_keyed else None,
        retention_registry=retention_registry if retention_keyed else None,
    )

    adapter_cfg = _adapter_config()
    training_cfg = _training_config(
        num_epochs=num_epochs,
        seed=seed,
        lr_scheduler_type=lr_scheduler_type,
        lr_decay_steps=lr_decay_steps,
        weight_decay=weight_decay,
    )

    t0 = time.time()
    metrics = train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        adapter_name=adapter_name,
        training_config=training_cfg,
        adapter_config=adapter_cfg,
        output_dir=phase_dir,
        run_name=run_name,
        callbacks_extra=[callback],
        resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None,
    )
    wall = time.time() - t0

    probe_state.first_perfect_epoch = early_state.first_perfect_epoch
    probe_state.stable_perfect_epoch = early_state.stable_perfect_epoch
    probe_state.stop_epoch = early_state.stop_epoch
    probe_state.epoch_log = early_state.epoch_log

    return model, metrics, probe_state, wall


# ---------------------------------------------------------------------------
# Repair loop
# ---------------------------------------------------------------------------


def _identify_failing_keys_via_probe(
    stop_probe: dict,
    unchanged_keyed: list[dict],
) -> list[dict]:
    """Identify failing unchanged keys from a fresh ``evaluate_indexed_recall`` probe.

    The training callback's ``epoch_log`` stores only a retention summary (no
    ``per_key`` results), so failing-key identification must use a fresh probe
    taken at repair-loop entry — never the epoch_log.

    Args:
        stop_probe: Result dict from ``evaluate_indexed_recall`` (must contain a
            ``per_key`` list with ``key`` and ``exact_match`` fields).
        unchanged_keyed: Full list of unchanged keyed pairs.

    Returns:
        List of keyed dicts for keys whose ``exact_match`` was False in
        ``stop_probe``.
    """
    per_key = stop_probe.get("per_key", [])
    passing_keys: set[str] = {entry["key"] for entry in per_key if entry.get("exact_match")}
    return [kp for kp in unchanged_keyed if kp["key"] not in passing_keys]


def _sha256_dir(adapter_dir: Path) -> str | None:
    """Compute a deterministic SHA-256 fingerprint of all files under adapter_dir.

    Used by ``_verify_phase_integrity`` to confirm the repair loop wrote
    different weights to the repaired adapter directory.

    Args:
        adapter_dir: Root directory to fingerprint.

    Returns:
        Hex digest string, or None if directory does not exist.
    """
    if not adapter_dir.exists():
        return None
    h = hashlib.sha256()
    for p in sorted(adapter_dir.rglob("*")):
        if p.is_file():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _verify_phase_integrity(seed_dir: Path, seed: int) -> None:
    """Verify done markers and adapter integrity for one completed seed.

    Called from ``run_seed`` after both B and C2 complete.  Runs for every
    seed.  The sha256 computation is cheap (~1 second per
    seed for small adapter dirs).

    Checks:
    1. Done-marker existence for all four phases (A, B, C1, C2).
    2. For each of (B/episodic_adapter) and (C2/journal_adapter): when both
       the original and the repaired sibling directory exist, assert their
       sha256 digests differ (repair wrote different weights).  When no
       repaired sibling exists, the repair loop found no failing keys and
       the check is skipped (legitimate skip).

    Args:
        seed_dir: Per-seed directory (``run_dir / f"seed{seed}"``).
        seed: Integer seed value (used only for log messages).

    Raises:
        AssertionError: If any done marker is missing or if the repaired
            adapter is identical to the original (indicating the repair loop
            did not write different weights).
    """
    for phase in ["A", "B", "C1", "C2"]:
        marker = seed_dir / phase / f"{phase}_done.json"
        assert marker.exists(), (
            f"_verify_phase_integrity: {phase}_done.json missing for seed {seed} "
            f"(expected at {marker})"
        )

    for phase, adapter_name in [("B", "episodic_adapter"), ("C2", "journal_adapter")]:
        phase_dir = seed_dir / phase
        original_dir = phase_dir / adapter_name
        repaired_dir = phase_dir / f"{adapter_name}_repaired"

        if original_dir.exists() and repaired_dir.exists():
            sha_orig = _sha256_dir(original_dir)
            sha_repaired = _sha256_dir(repaired_dir)
            assert sha_orig != sha_repaired or sha_orig is None, (
                f"_verify_phase_integrity: repaired adapter identical to original for "
                f"seed {seed} phase {phase} — repair did not run or wrote back to source "
                f"(sha256={sha_orig!r})"
            )
        # If repaired dir absent: repair found no failing keys — legitimate skip.

    logger.info("_verify_phase_integrity OK for seed%d", seed)


def run_repair_loop(
    model,
    tokenizer,
    adapter_name: str,
    unchanged_keyed: list[dict],
    unchanged_registry: dict[str, int],
    phase_dir: Path,
    max_episodes: int,
    repair_lr: float,
    weight_decay: float,
    phase: str = "",
    seed: int = -1,
) -> tuple[object, dict]:
    """Run the post-phase repair loop on failing unchanged keys.

    Takes a fresh ``evaluate_indexed_recall`` probe at entry to determine
    which keys are failing and to record RP2.  The training callback's
    ``epoch_log`` stores only a retention summary (no ``per_key`` results) and
    therefore cannot be used for failing-key identification.

    This implements the Test 13b recovery primitive in a multi-episode loop:
    each episode trains for 1 epoch at ``repair_lr`` on the currently failing
    subset of unchanged keys.  Between episodes the full unchanged set is
    probed; the loop exits when all keys pass or the episode budget is
    exhausted.

    HARD RULE — originals preserved:
      - ``save_strategy="no"`` prevents HF Trainer from writing checkpoints.
      - ``output_dir`` is set to ``<phase_dir>/repair_episodes/episode_N/``
        (never the main adapter dir).
      - Final repaired state is saved to ``<adapter_name>_adapter_repaired/``
        as a sibling to ``<adapter_name>_adapter/``.

    Resume semantics: repair is non-resumable mid-loop.  If called from a
    resume that has ``<phase>_train_done.json`` but not
    ``<phase>_repair_done.json``, the caller reloads the original adapter
    and calls this function from episode 1.

    Args:
        model: Active PeftModel (loaded from original adapter at phase-done).
        tokenizer: Tokenizer.
        adapter_name: Adapter name (``"episodic"`` or ``"journal"``).
        unchanged_keyed: Full list of unchanged keyed pairs.
        unchanged_registry: SimHash registry for unchanged_keyed.
        phase_dir: Phase directory root.
        max_episodes: Maximum repair episodes.
        repair_lr: Learning rate for repair (default 1e-5).
        weight_decay: L2 regularisation.
        phase: Phase label (``"B"`` or ``"C2"``), recorded in repair_log.
        seed: RNG seed for this run, recorded in repair_log.

    Returns:
        Tuple of (model, repair_log_dict).
    """
    # Fresh per-key probe at repair start.  The training callback's epoch_log
    # stores only a retention summary (no per_key field), so we cannot use it
    # for failing-key identification.  This probe also provides the canonical
    # RP2 source — its summary numbers match epoch_log[-1]["retention"] to
    # within probe stochasticity (zero at temperature=0).
    stop_probe = _safe_probe(model, tokenizer, unchanged_keyed, unchanged_registry, adapter_name)
    # _safe_probe re-enables gradient_checkpointing in its finally block, so
    # it is safe to call train_adapter immediately after.

    failing_keyed = _identify_failing_keys_via_probe(stop_probe, unchanged_keyed)
    failing_keys_pre_repair: int = len(failing_keyed)
    repair_curve: list[dict] = []

    # RP2: retention at repair-start (canonical source: fresh stop_probe).
    rp2_rate = stop_probe["rate"]
    rp2_exact = stop_probe["exact_count"]

    # Track which keys were passing before repair (for collateral damage).
    passing_before: set[str] = {
        entry["key"] for entry in stop_probe["per_key"] if entry.get("exact_match")
    }

    if not failing_keyed:
        logger.info("Repair loop: no failing keys — skipping.")
        repair_log = {
            "phase": phase,
            "seed": seed,
            "failing_keys_pre_repair": failing_keys_pre_repair,
            "RP2_rate": round(rp2_rate, 6),
            "RP2_exact_count": rp2_exact,
            "RP3_rate": round(rp2_rate, 6),
            "RP3_exact_count": rp2_exact,
            "RP3_total": len(unchanged_keyed),
            "alignment_delta": 0.0,
            "corruption_residual": round(1.0 - rp2_rate, 6),
            "recovered_count": 0,
            "still_failing_count": 0,
            "collateral_loss_count": 0,
            "stop_reason": "no_failing_keys",
            "episodes_used": 0,
            "max_episodes": max_episodes,
            "lr": repair_lr,
            "curve": [],
        }
        return model, repair_log

    logger.info(
        "Repair loop: %d failing keys, max_episodes=%d, lr=%.1e",
        len(failing_keyed),
        max_episodes,
        repair_lr,
    )

    episodes_used = 0
    stop_reason = "max_episodes"

    for episode in range(1, max_episodes + 1):
        # Honour pause at the start of every episode iteration.
        if paused_requested():
            logger.warning(
                "Pause file detected during repair %s episode %d — exiting cleanly.",
                phase_dir.name,
                episode,
            )
            write_paused_marker(
                phase_dir.parent.parent,  # run_dir
                f"during repair {phase_dir.name} episode {episode}",
            )
            raise SystemExit(f"Training paused during repair {phase_dir.name} episode {episode}")

        ep_dir = phase_dir / "repair_episodes" / f"episode_{episode}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Repair episode %d/%d: %d failing keys", episode, max_episodes, len(failing_keyed)
        )

        examples = format_indexed_training(failing_keyed, tokenizer, max_length=1024)
        dataset = IndexedDataset(examples)

        adapter_cfg = _adapter_config(lr=repair_lr)
        training_cfg = _training_config(
            num_epochs=1,
            weight_decay=weight_decay,
            save_strategy="no",
        )
        training_cfg.save_total_limit = None

        t0 = time.time()
        metrics = train_adapter(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            adapter_name=adapter_name,
            training_config=training_cfg,
            adapter_config=adapter_cfg,
            output_dir=ep_dir,
            run_name=f"test15-repair-{phase_dir.name}-ep{episode}",
        )
        ep_wall = time.time() - t0
        train_loss = metrics.get("train_loss")

        # Probe full unchanged set after this episode.
        probe_result = _safe_probe(
            model, tokenizer, unchanged_keyed, unchanged_registry, adapter_name
        )

        episodes_used = episode
        repair_curve.append(
            {
                "episode": episode,
                "retention": probe_result["rate"],
                "exact_count": probe_result["exact_count"],
                "total": probe_result["total"],
                "wall_seconds": round(ep_wall, 1),
                "train_loss": train_loss,
            }
        )

        logger.info(
            "  Repair ep %d: retention %d/%d (%.3f)",
            episode,
            probe_result["exact_count"],
            probe_result["total"],
            probe_result["rate"],
        )

        # Early exit when full recovery.
        if probe_result["exact_count"] == len(unchanged_keyed):
            stop_reason = "full_recovery"
            logger.info("  Repair: full recovery at episode %d", episode)
            break

        # Update failing keys for next episode.
        failing_keyed = [
            kp
            for kp in unchanged_keyed
            if not any(
                entry["key"] == kp["key"] and entry.get("exact_match")
                for entry in (probe_result.get("per_key") or [])
            )
        ]

    # Final state after repair.
    final_retention = _safe_probe(
        model, tokenizer, unchanged_keyed, unchanged_registry, adapter_name
    )
    rp3_rate = final_retention["rate"]
    rp3_exact = final_retention["exact_count"]

    # Save repaired adapter.
    repaired_dir = phase_dir / f"{adapter_name}_adapter_repaired"
    save_adapter(model, repaired_dir, adapter_name)
    logger.info("Repaired adapter saved to %s", repaired_dir)

    # Compute alignment_delta and corruption_residual.
    alignment_delta = round(rp3_rate - rp2_rate, 6)
    corruption_residual = round(1.0 - rp3_rate, 6)

    # Collateral loss: keys passing before repair that fail after.
    passing_after: set[str] = {
        entry["key"] for entry in (final_retention.get("per_key") or []) if entry.get("exact_match")
    }
    collateral_lost = len(passing_before - passing_after)

    # Recovery count.  Use stop_probe (the fresh probe taken at repair-loop
    # entry) for failing_before_keys — matches what was used to build failing_keyed.
    failing_before_keys = set(
        kp["key"] for kp in _identify_failing_keys_via_probe(stop_probe, unchanged_keyed)
    )
    still_failing_keys = set(
        entry["key"]
        for entry in (final_retention.get("per_key") or [])
        if not entry.get("exact_match")
    )
    recovered_count = len(failing_before_keys - still_failing_keys)
    still_failing_count = len(failing_before_keys & still_failing_keys)

    repair_log = {
        "phase": phase,
        "seed": seed,
        "failing_keys_pre_repair": failing_keys_pre_repair,
        "RP2_rate": round(rp2_rate, 6),
        "RP2_exact_count": rp2_exact,
        "RP3_rate": round(rp3_rate, 6),
        "RP3_exact_count": rp3_exact,
        "RP3_total": len(unchanged_keyed),
        "alignment_delta": alignment_delta,
        "corruption_residual": corruption_residual,
        "recovered_count": recovered_count,
        "still_failing_count": still_failing_count,
        "collateral_loss_count": collateral_lost,
        "stop_reason": stop_reason,
        "episodes_used": episodes_used,
        "max_episodes": max_episodes,
        "lr": repair_lr,
        "curve": repair_curve,
    }
    return model, repair_log


# ---------------------------------------------------------------------------
# Per-seed runner
# ---------------------------------------------------------------------------


def run_seed(
    model,
    tokenizer,
    seed: int,
    run_dir: Path,
    cfg: dict,
    early_stop_policy: EarlyStopPolicy,
) -> dict:
    """Run the full A→B→C1→C2 protocol for one seed.

    Returns a dict with RP2/RP3 rates, stop epochs, and repair stats for
    phases B and C2 (used by the multi-seed aggregator).

    Args:
        model: Loaded base model (must NOT be a PeftModel at entry).
        tokenizer: Tokenizer.
        seed: RNG seed for this run.
        run_dir: Top-level run directory.
        cfg: Run config dict (from load_or_write_run_config).
        early_stop_policy: EarlyStopPolicy instance.

    Returns:
        Per-seed result dict with B and C2 metrics.
    """
    # Cross-seed discipline: on entry the model has been unwrapped to its base
    # in main(). create_adapter("episodic", ...) re-initialises the LoRA weights
    # for that name from scratch — even if "episodic" already exists in
    # peft_config from the previous seed. The accumulating peft_config dict is
    # benign at LoRA rank 8 (kilobytes per entry); forward-path correctness is
    # preserved by the explicit set_adapter/switch_adapter calls inside each phase.

    seed_dir = run_dir / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    n_keys: int = cfg["n_keys"]
    swap_keys: int = cfg["swap_keys"]
    swap_start = n_keys - swap_keys
    weight_decay: float = cfg["weight_decay"]
    lr_scheduler_type: str = cfg["lr_scheduler_type"]
    lr_decay_steps: int | None = cfg["lr_decay_steps"]
    max_repair_episodes: int = cfg["max_repair_episodes"]
    repair_lr: float = cfg["repair_lr"]

    logger.info("=" * 72)
    logger.info("Seed %d: loading QA pool (n_keys=%d, swap_keys=%d)", seed, n_keys, swap_keys)
    logger.info("=" * 72)

    qa_pool = load_qa_pool(n_keys, swap_keys)
    swap_answers = qa_pool[n_keys:]  # extra swap_keys entries

    # -----------------------------------------------------------------------
    # Phase A — Fresh
    # -----------------------------------------------------------------------
    phase_a_dir = seed_dir / "A"

    if (phase_a_dir / "A_done.json").exists():
        logger.info("Seed %d Phase A: already done — loading from disk.", seed)
        a_keyed = json.loads((phase_a_dir / "quads.json").read_text())
        # Reload adapter.
        if isinstance(model, PeftModel):
            model = model.base_model.model
        a_slot = resolve_adapter_slot(phase_a_dir / "episodic_adapter", "episodic", "")
        if a_slot is None:
            raise FileNotFoundError(
                f"Seed {seed}: episodic adapter not found under {phase_a_dir / 'episodic_adapter'}"
            )
        with _adapter_slot_for_load(a_slot) as _load_path:
            model = PeftModel.from_pretrained(model, str(_load_path), adapter_name="episodic")
        switch_adapter(model, "episodic")
    else:
        _check_pause(f"before Phase A seed {seed}", run_dir)

        logger.info(
            "Seed %d Phase A — Fresh: %d keys, up to %d epochs",
            seed,
            n_keys,
            cfg["phase_a_num_epochs"],
        )
        if isinstance(model, PeftModel):
            model = model.base_model.model
        model = create_adapter(model, _adapter_config(), "episodic")
        switch_adapter(model, "episodic")

        a_keyed = build_phase_A_keyed(qa_pool, total_keys=n_keys)
        a_registry = build_registry(a_keyed)

        # Check for checkpoint to resume from mid-phase.
        a_ckpt = _find_latest_checkpoint(phase_a_dir)

        model, metrics_a, probe_a, wall_a = _run_phase(
            model=model,
            tokenizer=tokenizer,
            keyed_to_train=a_keyed,
            target_keyed=a_keyed,
            target_registry=a_registry,
            adapter_name="episodic",
            phase_dir=phase_a_dir,
            phase_name="Phase_A",
            num_epochs=cfg["phase_a_num_epochs"],
            seed=seed,
            lr_scheduler_type=lr_scheduler_type,
            lr_decay_steps=lr_decay_steps,
            weight_decay=weight_decay,
            early_stop_policy=early_stop_policy,
            run_name=f"test15-A-seed{seed}",
            resume_from_checkpoint=a_ckpt,
        )

        _exit_if_paused_mid_phase(
            probe_a, f"during A seed {seed}", cfg["phase_a_num_epochs"], run_dir
        )

        final_a = _safe_probe(model, tokenizer, a_keyed, a_registry, "episodic")
        _manifest_a = build_manifest_for(
            model,
            tokenizer,
            "episodic",
            registry_path=None,
            quads_path=phase_a_dir / "quads.json",
            key_count=len(a_keyed),
        )
        save_adapter(model, phase_a_dir / "episodic_adapter", "episodic", manifest=_manifest_a)
        _write_phase_done(
            phase_a_dir,
            "A_done.json",
            a_keyed,
            a_registry,
            probe_a,
            final_a,
            extra={
                "condition": "A_fresh",
                "seed": seed,
                "train_loss": metrics_a.get("train_loss"),
                "wall_seconds": round(wall_a, 1),
            },
        )

    # -----------------------------------------------------------------------
    # Phase B — Answer-swap
    # -----------------------------------------------------------------------
    phase_b_dir = seed_dir / "B"
    unchanged_keyed = a_keyed[:swap_start]
    unchanged_registry = build_registry(unchanged_keyed)

    b_result: dict = {}

    if (phase_b_dir / "B_done.json").exists():
        logger.info("Seed %d Phase B: already done.", seed)
        b_done = json.loads((phase_b_dir / "B_done.json").read_text())
        b_result = {
            "RP2_rate": b_done.get("rp2_rate", 0.0),
            "RP3_rate": b_done.get("rp3_rate", 0.0),
            "stop_epoch": b_done.get("stop_epoch"),
            "alignment_delta": b_done.get("alignment_delta", 0.0),
            "corruption_residual": b_done.get("corruption_residual", 0.0),
            "episodes_used": b_done.get("episodes_used", 0),
        }
        # Reload adapter for C1 (need unwrapped base).
        if isinstance(model, PeftModel):
            model = model.base_model.model
    else:
        # Check if train_done but repair not done → resume repair.
        b_train_done = (phase_b_dir / "B_train_done.json").exists()
        b_repair_done = (phase_b_dir / "B_repair_done.json").exists()

        if b_train_done and not b_repair_done:
            logger.info("Seed %d Phase B: train done, resuming repair loop.", seed)
            b_train_data = json.loads((phase_b_dir / "B_train_done.json").read_text())
            # Reload original adapter.
            if isinstance(model, PeftModel):
                model = model.base_model.model
            b_slot = resolve_adapter_slot(phase_b_dir / "episodic_adapter", "episodic", "")
            if b_slot is None:
                raise FileNotFoundError(
                    f"Seed {seed}: B episodic adapter not found under "
                    f"{phase_b_dir / 'episodic_adapter'}"
                )
            with _adapter_slot_for_load(b_slot) as _load_path:
                model = PeftModel.from_pretrained(model, str(_load_path), adapter_name="episodic")
            switch_adapter(model, "episodic")
            epoch_log_b = b_train_data.get("epoch_log", [])
            stop_epoch_b = b_train_data.get("stop_epoch")
            probe_b_state = EpochProbeState()
            probe_b_state.stop_epoch = stop_epoch_b
            probe_b_state.epoch_log = epoch_log_b
        else:
            _check_pause(f"before Phase B seed {seed}", run_dir)

            logger.info("Seed %d Phase B — Answer-swap: %d keys", seed, swap_keys)
            switch_adapter(model, "episodic")

            b_swap_keyed = build_phase_B_swap_keyed(
                a_keyed, swap_answers, swap_keys=swap_keys, total_keys=n_keys
            )
            b_swap_registry = build_registry(b_swap_keyed)

            # Check for checkpoint to resume from mid-phase.
            b_ckpt = _find_latest_checkpoint(phase_b_dir)

            model, metrics_b, probe_b_state, wall_b = _run_phase(
                model=model,
                tokenizer=tokenizer,
                keyed_to_train=b_swap_keyed,
                target_keyed=b_swap_keyed,
                target_registry=b_swap_registry,
                adapter_name="episodic",
                phase_dir=phase_b_dir,
                phase_name="Phase_B",
                num_epochs=cfg["phase_b_num_epochs"],
                seed=seed,
                lr_scheduler_type=lr_scheduler_type,
                lr_decay_steps=lr_decay_steps,
                weight_decay=weight_decay,
                early_stop_policy=early_stop_policy,
                run_name=f"test15-B-seed{seed}",
                retention_keyed=unchanged_keyed,
                retention_registry=unchanged_registry,
                resume_from_checkpoint=b_ckpt,
            )

            _exit_if_paused_mid_phase(
                probe_b_state, f"during B seed {seed}", cfg["phase_b_num_epochs"], run_dir
            )

            final_b = _safe_probe(  # noqa: F841
                model, tokenizer, b_swap_keyed, b_swap_registry, "episodic"
            )
            _manifest_b = build_manifest_for(
                model,
                tokenizer,
                "episodic",
                registry_path=None,
                quads_path=phase_b_dir / "quads.json",
                key_count=len(b_swap_keyed),
            )
            save_adapter(model, phase_b_dir / "episodic_adapter", "episodic", manifest=_manifest_b)

            # Write B_train_done.json.
            b_train_marker = {
                "condition": "B_answer_swap",
                "seed": seed,
                "stop_epoch": probe_b_state.stop_epoch,
                "epoch_log": probe_b_state.epoch_log,
                "train_loss": metrics_b.get("train_loss"),
                "wall_seconds": round(wall_b, 1),
                "n_keys": len(b_swap_keyed),
            }
            _safe_write_json(phase_b_dir / "B_train_done.json", b_train_marker)
            epoch_log_b = probe_b_state.epoch_log

            _check_pause(f"before B repair seed {seed}", run_dir)

        # Run repair loop.  epoch_log is not passed — run_repair_loop takes
        # a fresh probe at entry (the callback does not store per_key results).
        model, repair_log_b = run_repair_loop(
            model=model,
            tokenizer=tokenizer,
            adapter_name="episodic",
            unchanged_keyed=unchanged_keyed,
            unchanged_registry=unchanged_registry,
            phase_dir=phase_b_dir,
            max_episodes=max_repair_episodes,
            repair_lr=repair_lr,
            weight_decay=weight_decay,
            phase="B",
            seed=seed,
        )
        _safe_write_json(phase_b_dir / "repair_log.json", repair_log_b)
        _safe_write_json(
            phase_b_dir / "B_repair_done.json",
            {"timestamp": int(time.time()), "stop_reason": repair_log_b["stop_reason"]},
        )

        # Write B_done.json.
        b_done_data = {
            "condition": "B_answer_swap",
            "seed": seed,
            "rp2_rate": repair_log_b["RP2_rate"],
            "rp3_rate": repair_log_b["RP3_rate"],
            "stop_epoch": probe_b_state.stop_epoch,
            "alignment_delta": repair_log_b["alignment_delta"],
            "corruption_residual": repair_log_b["corruption_residual"],
            "episodes_used": repair_log_b["episodes_used"],
            "stop_reason": repair_log_b["stop_reason"],
        }
        _safe_write_json(phase_b_dir / "B_done.json", b_done_data)

        b_result = {
            "RP2_rate": repair_log_b["RP2_rate"],
            "RP3_rate": repair_log_b["RP3_rate"],
            "stop_epoch": probe_b_state.stop_epoch,
            "alignment_delta": repair_log_b["alignment_delta"],
            "corruption_residual": repair_log_b["corruption_residual"],
            "episodes_used": repair_log_b["episodes_used"],
        }
        logger.info(
            "Seed %d Phase B done. RP2=%.3f RP3=%.3f",
            seed,
            b_result["RP2_rate"],
            b_result["RP3_rate"],
        )

        # Unwrap for C1.
        if isinstance(model, PeftModel):
            model = model.base_model.model

    _check_pause(f"after B seed {seed}", run_dir)

    # -----------------------------------------------------------------------
    # Phase C1 — Scaffold
    # -----------------------------------------------------------------------
    phase_c1_dir = seed_dir / "C1"

    if (phase_c1_dir / "C1_done.json").exists():
        logger.info("Seed %d Phase C1: already done — loading from disk.", seed)
        c1_keyed = json.loads((phase_c1_dir / "quads.json").read_text())
        if isinstance(model, PeftModel):
            model = model.base_model.model
        c1_slot = resolve_adapter_slot(phase_c1_dir / "journal_adapter", "journal", "")
        if c1_slot is None:
            raise FileNotFoundError(
                f"Seed {seed}: journal adapter not found under {phase_c1_dir / 'journal_adapter'}"
            )
        with _adapter_slot_for_load(c1_slot) as _load_path:
            model = PeftModel.from_pretrained(model, str(_load_path), adapter_name="journal")
        switch_adapter(model, "journal")
    else:
        _check_pause(f"before Phase C1 seed {seed}", run_dir)

        logger.info(
            "Seed %d Phase C1 — Scaffold: %d keys (%d real + %d TBD-k), up to %d epochs",
            seed,
            n_keys,
            swap_start,
            swap_keys,
            cfg["phase_c1_num_epochs"],
        )
        if isinstance(model, PeftModel):
            model = model.base_model.model
        model = create_adapter(model, _adapter_config(), "journal")
        switch_adapter(model, "journal")

        c1_keyed = build_phase_C1_keyed(qa_pool, total_keys=n_keys, swap_keys=swap_keys)
        c1_registry = build_registry(c1_keyed)

        c1_ckpt = _find_latest_checkpoint(phase_c1_dir)

        model, metrics_c1, probe_c1, wall_c1 = _run_phase(
            model=model,
            tokenizer=tokenizer,
            keyed_to_train=c1_keyed,
            target_keyed=c1_keyed,
            target_registry=c1_registry,
            adapter_name="journal",
            phase_dir=phase_c1_dir,
            phase_name="Phase_C1",
            num_epochs=cfg["phase_c1_num_epochs"],
            seed=seed,
            lr_scheduler_type=lr_scheduler_type,
            lr_decay_steps=lr_decay_steps,
            weight_decay=weight_decay,
            early_stop_policy=early_stop_policy,
            run_name=f"test15-C1-seed{seed}",
            resume_from_checkpoint=c1_ckpt,
        )

        _exit_if_paused_mid_phase(
            probe_c1, f"during C1 seed {seed}", cfg["phase_c1_num_epochs"], run_dir
        )

        final_c1 = _safe_probe(model, tokenizer, c1_keyed, c1_registry, "journal")
        _manifest_c1 = build_manifest_for(
            model,
            tokenizer,
            "journal",
            registry_path=None,
            quads_path=phase_c1_dir / "quads.json",
            key_count=len(c1_keyed),
        )
        save_adapter(model, phase_c1_dir / "journal_adapter", "journal", manifest=_manifest_c1)
        _write_phase_done(
            phase_c1_dir,
            "C1_done.json",
            c1_keyed,
            c1_registry,
            probe_c1,
            final_c1,
            extra={
                "condition": "C1_scaffold",
                "seed": seed,
                "train_loss": metrics_c1.get("train_loss"),
                "wall_seconds": round(wall_c1, 1),
            },
        )

    _check_pause(f"after C1 seed {seed}", run_dir)

    # -----------------------------------------------------------------------
    # Phase C2 — Fill
    # -----------------------------------------------------------------------
    phase_c2_dir = seed_dir / "C2"

    c2_result: dict = {}

    if (phase_c2_dir / "C2_done.json").exists():
        logger.info("Seed %d Phase C2: already done.", seed)
        c2_done = json.loads((phase_c2_dir / "C2_done.json").read_text())
        c2_result = {
            "RP2_rate": c2_done.get("rp2_rate", 0.0),
            "RP3_rate": c2_done.get("rp3_rate", 0.0),
            "stop_epoch": c2_done.get("stop_epoch"),
            "alignment_delta": c2_done.get("alignment_delta", 0.0),
            "corruption_residual": c2_done.get("corruption_residual", 0.0),
            "episodes_used": c2_done.get("episodes_used", 0),
        }
    else:
        c2_train_done = (phase_c2_dir / "C2_train_done.json").exists()
        c2_repair_done = (phase_c2_dir / "C2_repair_done.json").exists()

        if c2_train_done and not c2_repair_done:
            logger.info("Seed %d Phase C2: train done, resuming repair loop.", seed)
            c2_train_data = json.loads((phase_c2_dir / "C2_train_done.json").read_text())
            if isinstance(model, PeftModel):
                model = model.base_model.model
            c2_slot = resolve_adapter_slot(phase_c2_dir / "journal_adapter", "journal", "")
            if c2_slot is None:
                raise FileNotFoundError(
                    f"Seed {seed}: C2 journal adapter not found under "
                    f"{phase_c2_dir / 'journal_adapter'}"
                )
            with _adapter_slot_for_load(c2_slot) as _load_path:
                model = PeftModel.from_pretrained(model, str(_load_path), adapter_name="journal")
            switch_adapter(model, "journal")
            epoch_log_c2 = c2_train_data.get("epoch_log", [])
            stop_epoch_c2 = c2_train_data.get("stop_epoch")
            probe_c2_state = EpochProbeState()
            probe_c2_state.stop_epoch = stop_epoch_c2
            probe_c2_state.epoch_log = epoch_log_c2
        else:
            _check_pause(f"before Phase C2 seed {seed}", run_dir)

            logger.info("Seed %d Phase C2 — Fill: %d keys", seed, swap_keys)
            switch_adapter(model, "journal")

            c2_fill_keyed = build_phase_C2_fill_keyed(
                c1_keyed, qa_pool, swap_keys=swap_keys, total_keys=n_keys
            )
            c2_fill_registry = build_registry(c2_fill_keyed)

            c2_ckpt = _find_latest_checkpoint(phase_c2_dir)

            model, metrics_c2, probe_c2_state, wall_c2 = _run_phase(
                model=model,
                tokenizer=tokenizer,
                keyed_to_train=c2_fill_keyed,
                target_keyed=c2_fill_keyed,
                target_registry=c2_fill_registry,
                adapter_name="journal",
                phase_dir=phase_c2_dir,
                phase_name="Phase_C2",
                num_epochs=cfg["phase_c2_num_epochs"],
                seed=seed,
                lr_scheduler_type=lr_scheduler_type,
                lr_decay_steps=lr_decay_steps,
                weight_decay=weight_decay,
                early_stop_policy=early_stop_policy,
                run_name=f"test15-C2-seed{seed}",
                retention_keyed=unchanged_keyed,
                retention_registry=unchanged_registry,
                resume_from_checkpoint=c2_ckpt,
            )

            _exit_if_paused_mid_phase(
                probe_c2_state, f"during C2 seed {seed}", cfg["phase_c2_num_epochs"], run_dir
            )

            final_c2 = _safe_probe(  # noqa: F841
                model, tokenizer, c2_fill_keyed, c2_fill_registry, "journal"
            )
            _manifest_c2 = build_manifest_for(
                model,
                tokenizer,
                "journal",
                registry_path=None,
                quads_path=phase_c2_dir / "quads.json",
                key_count=len(c2_fill_keyed),
            )
            save_adapter(model, phase_c2_dir / "journal_adapter", "journal", manifest=_manifest_c2)

            c2_train_marker = {
                "condition": "C2_fill",
                "seed": seed,
                "stop_epoch": probe_c2_state.stop_epoch,
                "epoch_log": probe_c2_state.epoch_log,
                "train_loss": metrics_c2.get("train_loss"),
                "wall_seconds": round(wall_c2, 1),
                "n_keys": len(c2_fill_keyed),
            }
            _safe_write_json(phase_c2_dir / "C2_train_done.json", c2_train_marker)
            epoch_log_c2 = probe_c2_state.epoch_log

            _check_pause(f"before C2 repair seed {seed}", run_dir)

        # Run repair loop.  epoch_log is not passed — run_repair_loop takes
        # a fresh probe at entry (the callback does not store per_key results).
        model, repair_log_c2 = run_repair_loop(
            model=model,
            tokenizer=tokenizer,
            adapter_name="journal",
            unchanged_keyed=unchanged_keyed,
            unchanged_registry=unchanged_registry,
            phase_dir=phase_c2_dir,
            max_episodes=max_repair_episodes,
            repair_lr=repair_lr,
            weight_decay=weight_decay,
            phase="C2",
            seed=seed,
        )
        _safe_write_json(phase_c2_dir / "repair_log.json", repair_log_c2)
        _safe_write_json(
            phase_c2_dir / "C2_repair_done.json",
            {"timestamp": int(time.time()), "stop_reason": repair_log_c2["stop_reason"]},
        )

        c2_done_data = {
            "condition": "C2_fill",
            "seed": seed,
            "rp2_rate": repair_log_c2["RP2_rate"],
            "rp3_rate": repair_log_c2["RP3_rate"],
            "stop_epoch": probe_c2_state.stop_epoch,
            "alignment_delta": repair_log_c2["alignment_delta"],
            "corruption_residual": repair_log_c2["corruption_residual"],
            "episodes_used": repair_log_c2["episodes_used"],
            "stop_reason": repair_log_c2["stop_reason"],
        }
        _safe_write_json(phase_c2_dir / "C2_done.json", c2_done_data)

        c2_result = {
            "RP2_rate": repair_log_c2["RP2_rate"],
            "RP3_rate": repair_log_c2["RP3_rate"],
            "stop_epoch": probe_c2_state.stop_epoch,
            "alignment_delta": repair_log_c2["alignment_delta"],
            "corruption_residual": repair_log_c2["corruption_residual"],
            "episodes_used": repair_log_c2["episodes_used"],
        }
        logger.info(
            "Seed %d Phase C2 done. RP2=%.3f RP3=%.3f",
            seed,
            c2_result["RP2_rate"],
            c2_result["RP3_rate"],
        )

        if isinstance(model, PeftModel):
            model = model.base_model.model

    _verify_phase_integrity(seed_dir, seed)

    return {
        "B": b_result,
        "C2": c2_result,
    }


# ---------------------------------------------------------------------------
# Multi-seed aggregator
# ---------------------------------------------------------------------------


def compute_multiseed_aggregate(
    per_seed: dict[str, dict],
    seeds: list[int],
    n_rng_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict:
    """Compute multi-seed aggregate statistics with bootstrap CI.

    Bootstrap: 10 000 resamples over the 5 seeds, percentile method
    (NumPy ``np.random.choice`` with replacement), 95% CI.  Lower CI is the
    2.5th percentile.

    Verdict rules:
        - ``mean_b == 0 and mean_c2 == 0`` → ``"INDETERMINATE"``
          (0/0 is undefined; no claim can be made — e.g. when the run is
          too small to overwrite any unchanged keys).
        - ``mean_b == 0 and mean_c2 > 0`` → ``"HOLDS"``
          (trivial pass: scaffold path retains keys that the swap path loses
          entirely; ``ratio_C2_over_B`` is set to ``None`` in the JSON because
          infinity is not JSON-serialisable).
        - ``mean_b > 0`` → ``"HOLDS"`` iff ``ratio_raw >= DECISION_THRESHOLD_RATIO``
          AND ``ratio_lower_ci >= DECISION_THRESHOLD_LOWER_CI``; else
          ``"DOES NOT HOLD"``.

    Args:
        per_seed: Dict mapping seed (as string) to per-seed result dict with B/C2 metrics.
        seeds: List of integer seeds.
        n_rng_resamples: Number of bootstrap resamples (default 10 000).

    Returns:
        Aggregate dict matching the multiseed_aggregate.json schema.
    """
    b_rp2 = [per_seed[str(s)]["B"]["RP2_rate"] for s in seeds if str(s) in per_seed]
    c2_rp2 = [per_seed[str(s)]["C2"]["RP2_rate"] for s in seeds if str(s) in per_seed]
    b_rp3 = [per_seed[str(s)]["B"]["RP3_rate"] for s in seeds if str(s) in per_seed]
    c2_rp3 = [per_seed[str(s)]["C2"]["RP3_rate"] for s in seeds if str(s) in per_seed]

    mean_b = float(np.mean(b_rp2)) if b_rp2 else 0.0
    mean_c2 = float(np.mean(c2_rp2)) if c2_rp2 else 0.0
    mean_b_repaired = float(np.mean(b_rp3)) if b_rp3 else 0.0
    mean_c2_repaired = float(np.mean(c2_rp3)) if c2_rp3 else 0.0

    # Handle degenerate case: mean_b == 0 → ratio is inf.
    if mean_b == 0.0:
        ratio_raw = float("inf")
        ratio_lower_ci = float("inf")
    else:
        ratio_raw = mean_c2 / mean_b

        # Bootstrap CI for unrepaired ratio.
        rng = np.random.default_rng(seed=0)
        b_arr = np.array(b_rp2)
        c2_arr = np.array(c2_rp2)
        n = len(b_arr)
        ratios = []
        for _ in range(n_rng_resamples):
            idx = rng.choice(n, size=n, replace=True)
            b_boot = float(np.mean(b_arr[idx]))
            c2_boot = float(np.mean(c2_arr[idx]))
            if b_boot == 0.0:
                ratios.append(float("inf"))
            else:
                ratios.append(c2_boot / b_boot)
        finite_ratios = [r for r in ratios if not np.isinf(r)]
        if finite_ratios:
            ratio_lower_ci = float(np.percentile(finite_ratios, 2.5))
        else:
            ratio_lower_ci = float("inf")

    # Repaired ratio.
    if mean_b_repaired == 0.0:
        ratio_repaired = float("inf")
        ratio_lower_ci_repaired = float("inf")
    else:
        ratio_repaired = mean_c2_repaired / mean_b_repaired
        rng2 = np.random.default_rng(seed=1)
        b_arr3 = np.array(b_rp3)
        c2_arr3 = np.array(c2_rp3)
        n2 = len(b_arr3)
        ratios2: list[float] = []
        for _ in range(n_rng_resamples):
            idx2 = rng2.choice(n2, size=n2, replace=True)
            b_boot2 = float(np.mean(b_arr3[idx2]))
            c2_boot2 = float(np.mean(c2_arr3[idx2]))
            if b_boot2 == 0.0:
                ratios2.append(float("inf"))
            else:
                ratios2.append(c2_boot2 / b_boot2)
        finite_ratios2 = [r for r in ratios2 if not np.isinf(r)]
        ratio_lower_ci_repaired = (
            float(np.percentile(finite_ratios2, 2.5)) if finite_ratios2 else float("inf")
        )

    # Verdict.
    # Three cases:
    #   B=0 and C2=0 → INDETERMINATE (0/0 is undefined; no claim can be made).
    #   B=0 and C2>0 → HOLDS (trivial pass: scaffold retains more than the swap path).
    #   B>0          → threshold test on ratio_raw and bootstrap lower CI.
    verdict: str
    if mean_b == 0.0 and mean_c2 == 0.0:
        verdict = "INDETERMINATE"
    elif mean_b == 0.0 and mean_c2 > 0.0:
        verdict = "HOLDS"
    elif ratio_raw >= DECISION_THRESHOLD_RATIO and ratio_lower_ci >= DECISION_THRESHOLD_LOWER_CI:
        verdict = "HOLDS"
    else:
        verdict = "DOES NOT HOLD"

    return {
        "n_completed": len([s for s in seeds if str(s) in per_seed]),
        "seeds": seeds,
        "per_seed": per_seed,
        "mean_retention_B": round(mean_b, 6),
        "mean_retention_C2": round(mean_c2, 6),
        "mean_retention_B_repaired": round(mean_b_repaired, 6),
        "mean_retention_C2_repaired": round(mean_c2_repaired, 6),
        "ratio_C2_over_B": ratio_raw if not np.isinf(ratio_raw) else None,
        "ratio_repaired_C2_over_B": ratio_repaired if not np.isinf(ratio_repaired) else None,
        "ratio_lower_ci": ratio_lower_ci if not np.isinf(ratio_lower_ci) else None,
        "ratio_lower_ci_repaired": (
            ratio_lower_ci_repaired if not np.isinf(ratio_lower_ci_repaired) else None
        ),
        "decision_threshold_ratio": DECISION_THRESHOLD_RATIO,
        "decision_threshold_lower_ci": DECISION_THRESHOLD_LOWER_CI,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Test 15.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Test 15: Retention Multi-Seed (Scaffold-Fill vs Answer-Swap)"
    )
    parser.add_argument(
        "--model",
        default="mistral",
        choices=list(BENCHMARK_MODELS.keys()),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--n-keys", "--n_keys", type=int, default=DEFAULT_N_KEYS, dest="n_keys")
    parser.add_argument(
        "--swap-keys", "--swap_keys", type=int, default=DEFAULT_SWAP_KEYS, dest="swap_keys"
    )
    parser.add_argument(
        "--phase-a-num-epochs",
        "--phase_a_num_epochs",
        type=int,
        default=DEFAULT_PHASE_A_EPOCHS,
        dest="phase_a_num_epochs",
    )
    parser.add_argument(
        "--phase-b-num-epochs",
        "--phase_b_num_epochs",
        type=int,
        default=DEFAULT_PHASE_B_EPOCHS,
        dest="phase_b_num_epochs",
    )
    parser.add_argument(
        "--phase-c1-num-epochs",
        "--phase_c1_num_epochs",
        type=int,
        default=DEFAULT_PHASE_C1_EPOCHS,
        dest="phase_c1_num_epochs",
    )
    parser.add_argument(
        "--phase-c2-num-epochs",
        "--phase_c2_num_epochs",
        type=int,
        default=DEFAULT_PHASE_C2_EPOCHS,
        dest="phase_c2_num_epochs",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        "--lr_scheduler_type",
        choices=["linear", "constant", "constant_with_warmup", "cosine"],
        default="linear",
        dest="lr_scheduler_type",
    )
    parser.add_argument(
        "--lr-decay-steps",
        "--lr_decay_steps",
        type=int,
        default=None,
        dest="lr_decay_steps",
        help="LR decay steps; computed as n_keys * max(phase_epochs) / 2 if None.",
    )
    parser.add_argument(
        "--weight-decay",
        "--weight_decay",
        type=float,
        default=0.01,
        dest="weight_decay",
    )
    parser.add_argument(
        "--max-repair-episodes",
        "--max_repair_episodes",
        type=int,
        default=DEFAULT_MAX_REPAIR_EPISODES,
        dest="max_repair_episodes",
    )
    parser.add_argument(
        "--repair-lr",
        "--repair_lr",
        type=float,
        default=REPAIR_LR,
        dest="repair_lr",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Auto-find latest run dir and continue."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for Test 15."""
    args = parse_args()

    # Pre-flight disk space check.
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(OUTPUT_BASE).free
    if free_bytes <= DISK_HEADROOM_BYTES:
        raise SystemExit(
            f"Insufficient disk space: {free_bytes / 1024**3:.1f} GB free in {OUTPUT_BASE}; "
            f"need > {DISK_HEADROOM_BYTES / 1024**3:.0f} GB."
        )

    # Resolve output dir.
    if args.resume:
        latest = find_latest_run_dir(args.model)
        if latest is None:
            logger.warning("--resume: no prior run — starting fresh")
            run_dir = model_output_dir(OUTPUT_BASE, args.model)
        else:
            run_dir = latest
            logger.info("Resuming from %s", run_dir)
    else:
        run_dir = model_output_dir(OUTPUT_BASE, args.model)

    run_dir.mkdir(parents=True, exist_ok=True)

    if paused_requested():
        logger.warning("Pause file present at launch — clearing before starting work.")
        try:
            PAUSE_FILE.unlink()
        except OSError:
            pass
    clear_paused_marker(run_dir)

    # Load or write run config.
    cfg = load_or_write_run_config(run_dir, args)

    # On resume, read back persisted values (they win over defaults, lose to explicit CLI).
    if args.resume:
        args.n_keys = cfg.get("n_keys", args.n_keys)
        args.swap_keys = cfg.get("swap_keys", args.swap_keys)
        args.phase_a_num_epochs = cfg.get("phase_a_num_epochs", args.phase_a_num_epochs)
        args.phase_b_num_epochs = cfg.get("phase_b_num_epochs", args.phase_b_num_epochs)
        args.phase_c1_num_epochs = cfg.get("phase_c1_num_epochs", args.phase_c1_num_epochs)
        args.phase_c2_num_epochs = cfg.get("phase_c2_num_epochs", args.phase_c2_num_epochs)
        args.lr_scheduler_type = cfg.get("lr_scheduler_type", args.lr_scheduler_type)
        args.lr_decay_steps = cfg.get("lr_decay_steps", args.lr_decay_steps)
        args.weight_decay = cfg.get("weight_decay", args.weight_decay)
        args.max_repair_episodes = cfg.get("max_repair_episodes", args.max_repair_episodes)
        args.repair_lr = cfg.get("repair_lr", args.repair_lr)
        args.seeds = cfg.get("seeds", args.seeds)

    model_config = BENCHMARK_MODELS[args.model]

    with acquire_gpu(interactive=True):
        model, tokenizer = load_model_and_config(model_config)

        per_seed: dict[str, dict] = {}

        for seed in cfg.get("seeds", args.seeds):
            _check_pause(f"before seed {seed}", run_dir)

            logger.info("=" * 72)
            logger.info("Starting seed %d", seed)
            logger.info("=" * 72)

            # Ensure model is unwrapped before each seed.
            if isinstance(model, PeftModel):
                model = model.base_model.model

            seed_result = run_seed(model, tokenizer, seed, run_dir, cfg, ANALYSIS_POLICY)
            per_seed[str(seed)] = seed_result

            # Check if all phases for all seeds so far are done and write aggregate.
            all_seeds = cfg.get("seeds", args.seeds)
            seeds_done = [
                s
                for s in all_seeds
                if all(
                    (run_dir / f"seed{s}" / ph / f"{ph}_done.json").exists()
                    for ph in ["A", "B", "C1", "C2"]
                )
            ]
            if len(seeds_done) == len(all_seeds):
                # Load all results from disk to rebuild full per_seed.
                full_per_seed: dict[str, dict] = {}
                for s in all_seeds:
                    b_path = run_dir / f"seed{s}" / "B" / "B_done.json"
                    c2_path = run_dir / f"seed{s}" / "C2" / "C2_done.json"
                    if b_path.exists() and c2_path.exists():
                        b_data = json.loads(b_path.read_text())
                        c2_data = json.loads(c2_path.read_text())
                        full_per_seed[str(s)] = {
                            "B": {
                                "RP2_rate": b_data.get("rp2_rate", 0.0),
                                "RP3_rate": b_data.get("rp3_rate", 0.0),
                                "stop_epoch": b_data.get("stop_epoch"),
                                "alignment_delta": b_data.get("alignment_delta", 0.0),
                                "corruption_residual": b_data.get("corruption_residual", 0.0),
                                "episodes_used": b_data.get("episodes_used", 0),
                            },
                            "C2": {
                                "RP2_rate": c2_data.get("rp2_rate", 0.0),
                                "RP3_rate": c2_data.get("rp3_rate", 0.0),
                                "stop_epoch": c2_data.get("stop_epoch"),
                                "alignment_delta": c2_data.get("alignment_delta", 0.0),
                                "corruption_residual": c2_data.get("corruption_residual", 0.0),
                                "episodes_used": c2_data.get("episodes_used", 0),
                            },
                        }
                aggregate = compute_multiseed_aggregate(full_per_seed, all_seeds)
                _safe_write_json(run_dir / "multiseed_aggregate.json", aggregate)
                logger.info("multiseed_aggregate.json written. Verdict: %s", aggregate["verdict"])

        unload_model(model, tokenizer)

    logger.info("Test 15 complete.")


if __name__ == "__main__":
    main()
