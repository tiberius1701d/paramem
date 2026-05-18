"""Test 16: Repair-Loop Sensitivity Sweep.

Research questions
------------------
1. Does recovery (episodes-to-full-recovery, final retention RP3, collateral
   loss, overwrite reversion) depend on how deeply the original knowledge was
   trained before the overwrite (encoding depth D)?
2. How sensitive is recovery to repair learning rate (1e-5 / 2e-5 / 5e-5)?
3. How sensitive is recovery to repair epochs-per-episode (1 / 3)?
4. Spot-check: does weight_decay=0.1 vs 0.01 move any metric?

Protocol (B-path only — no journal/scaffold path)
--------------------------------------------------
Per (seed, D):
  base_D    — Fresh ``episodic`` adapter on N≈50 PerLTQA keys, fixed D epochs,
              no early stop (``EarlyStopPolicy(signal_from_epoch=10**9)``).
              Records ``stable_perfect_epoch`` → ``encoding_floor_epoch``.
  corrupted_D — Continue training on K≈12 swap keys (different answers),
              fixed ``overwrite_epochs=20``, no early stop.
              Retains per-epoch unchanged-key retention curve in epoch_log.json.
              Saves ``unchanged_keyed.json`` / ``overwrite_swap_keyed.json`` /
              ``overwritten_keyed.json`` + their registries.
  repair cells — Grid ``repair_lr ∈ {1e-5, 2e-5, 5e-5} × repair_epochs_per_episode
              ∈ {1, 3}`` (6 cells) + spot-check ``(D=30, lr=1e-5, ep=1, wd=0.1)``.
              Each cell: reload corrupted_D fresh → run_repair_loop_v2 → save
              repaired adapter sibling.

New Test-16 metrics
-------------------
  overwrite_recall_after_repair  — does repair revert the intended overwrite?
  original_answer_resurfaced_rate — did original answers resurface?
Both via ``_safe_probe`` only (never raw ``evaluate_indexed_recall``).

Pause / resume
--------------
Three-level ``*_done.json`` markers: seed → depth → repair cell.
``tpause`` / ``tresume 16`` / ``tstatus`` via training-control.sh registry.

Smoke run
---------
``--smoke``: N=50, K=12, seeds=[42], depths=[30, 50], 1 repair cell per
depth; writes to ``outputs/test16_repair_sweep/mistral/_smoke/<ts>/``.
``find_latest_run_dir`` excludes ``*/_smoke/*``.

GPU prerequisite
----------------
The ParaMem server must release the GPU.  This script uses
``experiments.utils.gpu_guard.acquire_gpu()`` which auto-switches the server
to cloud-only for the duration.

Usage
-----
    python experiments/test16_repair_sweep.py --model mistral
    python experiments/test16_repair_sweep.py --model mistral --resume
    python experiments/test16_repair_sweep.py --model mistral --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np  # noqa: E402
from peft import PeftModel  # noqa: E402

from experiments.quadruple_adapter import load_unique_triples  # noqa: E402
from experiments.utils.early_stop import (  # noqa: E402
    EarlyStopPolicy,
    RecallEarlyStopCallback,
    _EarlyStopState,
    _safe_write_json,
)
from experiments.utils.gpu_guard import acquire_gpu  # noqa: E402
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
from paramem.training.entry_memory import (  # noqa: E402
    assign_keys,
    build_registry,
    format_entry_training,
)
from paramem.training.indexed_memory import (  # noqa: E402
    load_registry,
    save_registry,
)
from paramem.training.recall_eval import (  # noqa: E402
    evaluate_indexed_recall as evaluate_entry_recall,
)
from paramem.training.trainer import train_adapter  # noqa: E402
from paramem.utils.config import AdapterConfig, TrainingConfig  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_BASE = project_root / "outputs" / "test16_repair_sweep"
PAUSE_FILE = Path.home() / ".training_pause"

# Triple data source — the LongMemEval graph snapshot built by
# experiments/lme_graph_builder.py.  ``load_unique_triples`` returns
# 563 deduplicated (subject, predicate, object) tuples; test 16 needs
# the first n_keys + swap_keys (default 62).  Same snapshot used by
# the quadruple_adapter experiment (commit a8b329d) — reuse keeps
# triple-format experiments aligned on one canonical source.
DEFAULT_GRAPH_SNAPSHOT = project_root / "outputs" / "lme_graph" / "graph_snapshot.json"

DEFAULT_SEEDS = [42, 7, 1337, 1, 11]
DEFAULT_N_KEYS = 50
DEFAULT_SWAP_KEYS = 12
# "Depth" is epochs trained PAST first_perfect_epoch (the encoding floor).
# Each arm trains base for first_perfect_epoch + depth_past_floor epochs at
# the same fixed LR trajectory — see REFERENCE_EPOCHS.  D=0 means "stop the
# moment encoding is just-perfect"; D=30 means "30 more epochs of refinement
# past the floor".  Replaces the prior D=[30, 50] convention where D was
# total-epochs and the LR scheduler's decay-to-zero point at num_epochs/2
# made D=30 land before encoding had completed.
DEFAULT_DEPTHS_PAST_FLOOR = [0, 10, 30]
DEFAULT_OVERWRITE_EPOCHS = 20
DEFAULT_RANK = 8
DEFAULT_LR = 1e-4
DEFAULT_REPAIR_LRS = [1e-5, 2e-5, 5e-5]
DEFAULT_REPAIR_EPOCHS = [1, 3]
DEFAULT_REPAIR_WEIGHT_DECAY = 0.01
DEFAULT_REPAIR_WD_SPOTCHECK = 0.1
DEFAULT_SPOTCHECK_DEPTH = 0  # spotcheck cell lives at D=0 past floor
DEFAULT_MAX_REPAIR_EPISODES = 5
DEFAULT_PRETRAIN_WEIGHT_DECAY = 0.1  # >=0.1 per extended-training lore

# Reference epoch count used to pin lr_decay_steps independent of the per-arm
# epoch budget.  CLAUDE.md: "linear LR scheduler with fixed lr_decay_steps
# (decoupled from num_train_epochs)".  Set to comfortably exceed
# max(first_perfect_epoch) + max(depths_past_floor) so the LR remains nonzero
# throughout every arm's training.  Observed first_perfect across seeds in
# Test 15 (apples-to-apples scheduler) ranged e22..e26; with max
# depth_past_floor=30, a single shared schedule of decay over 60-epoch worth
# of steps keeps LR nonzero through epoch 60.
REFERENCE_EPOCHS = 60

# Upper bound on the base training epoch budget.  The floor-relative stop
# callback halts at first_perfect_epoch + depth_past_floor; this is the
# hard ceiling that triggers a refuse-to-corrupt failure if the floor was
# never reached.  Equals REFERENCE_EPOCHS so the LR trajectory does not
# decay past the last legitimate step.
MAX_BASE_EPOCHS = REFERENCE_EPOCHS

# Probe batch size for recall evaluation.  Matches server.yaml's
# `consolidation.recall_probe_batch_size=16` (2026-05-18 cutover).  Test 18
# verified 137/137 exact-match parity for Mistral 7B nf4 across b ∈ {1, 2,
# 4, 8, 16, 32, 64, 128}; b=16 yields ~4.75× per-probe speedup, ~346 MiB
# peak VRAM delta on this hardware.  Combined with `probe_every_n_epochs=3`
# in the EarlyStopPolicy this cuts probe wall time ~10-15× per epoch.
# Set to 1 to fall back to the serial probe path (byte-identical results).
RECALL_PROBE_BATCH_SIZE = 16

BOOTSTRAP_RESAMPLES = 10_000

DISK_HEADROOM_BYTES = 15 * 1024**3

# Number of HF Trainer checkpoints retained per phase.
CHECKPOINT_RETENTION = 2


# ---------------------------------------------------------------------------
# Probe state
# ---------------------------------------------------------------------------


@dataclass
class EpochProbeState:
    """Per-phase training metrics accumulator (mirrors test15 pattern)."""

    first_perfect_epoch: int | None = None
    stable_perfect_epoch: int | None = None
    stop_epoch: int | None = None
    epoch_log: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pause / marker helpers  (verbatim from test15_retention_multiseed.py)
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
    """Return True if <run_dir>/<marker_name>_done.json exists.

    Args:
        run_dir: Directory to check.
        marker_name: Stem of the marker file (without ``_done.json``).

    Returns:
        True if the marker file exists.
    """
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
    With the no-stop policy, ``stop_epoch`` is never set, so completion
    reduces to "ran the full epoch budget."

    Args:
        probe_state: EpochProbeState accumulator from the phase.
        phase_label: Human-readable label for log and paused.json.
        num_epochs: Total epoch budget for this phase.
        run_dir: Root run directory for writing ``paused.json``.
    """
    if probe_state.stop_epoch is not None:
        return  # natural convergence (would only fire if policy ever fired)
    if not probe_state.epoch_log:
        return  # nothing started; outer _check_pause will handle it
    last_epoch = probe_state.epoch_log[-1].get("epoch")
    if last_epoch is None:
        return
    if last_epoch >= num_epochs:
        return  # ran the full budget
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
# Checkpoint finder  (verbatim from test15_retention_multiseed.py)
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

    for candidate in sorted(candidates, key=_step, reverse=True):
        if (candidate / "trainer_state.json").exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Run-dir helpers
# ---------------------------------------------------------------------------


def find_latest_run_dir(model_name: str) -> Path | None:
    """Return the most recent non-smoke run dir for the given model.

    Smoke runs write to a ``_smoke/`` subdirectory and are excluded so that
    ``tresume 16`` after a completed smoke never reads the smoke's frozen
    2-cell ``repair_grid``.

    Args:
        model_name: Model name key (e.g. ``"mistral"``).

    Returns:
        Path to the latest non-smoke run dir, or None if none found.
    """
    parent = OUTPUT_BASE / model_name
    if not parent.is_dir():
        return None
    candidates: list[Path] = [
        d for d in sorted(parent.iterdir()) if d.is_dir() and d.name != "_smoke"
    ]
    return candidates[-1] if candidates else None


def load_or_write_run_config(run_dir: Path, args: argparse.Namespace) -> dict:
    """Read run_config.json if it exists; otherwise write it from args and return it.

    On first launch the ``repair_grid`` is materialised and frozen so that
    cell-dir names remain deterministic across resume.

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

    # Materialise the repair grid at first launch.
    repair_grid: list[list] = []
    for lr in args.repair_lrs:
        for ep in args.repair_epochs:
            repair_grid.append([lr, ep, args.repair_weight_decay])
    # Spot-check cell (for the designated depth only).
    spotcheck_cell = {
        "D": args.spotcheck_depth,
        "lr": args.repair_lrs[0],
        "ep": args.repair_epochs[0],
        "wd": args.repair_wd_spotcheck,
    }

    cfg: dict = {
        "model": args.model,
        "seeds": list(args.seeds),
        # "depths_past_floor": the test's D knob, defined as epochs trained
        # past first_perfect_epoch.  Per (seed, D) arm trains base for
        # first_perfect+D epochs at the shared apples-to-apples schedule;
        # see DEFAULT_DEPTHS_PAST_FLOOR docstring.
        "depths_past_floor": list(args.depths_past_floor),
        "n_keys": args.n_keys,
        "swap_keys": args.swap_keys,
        "overwrite_epochs": args.overwrite_epochs,
        "pretrain_weight_decay": args.pretrain_weight_decay,
        "repair_lrs": list(args.repair_lrs),
        "repair_epochs": list(args.repair_epochs),
        "repair_weight_decay": args.repair_weight_decay,
        "repair_wd_spotcheck": args.repair_wd_spotcheck,
        "spotcheck_depth": args.spotcheck_depth,
        "max_repair_episodes": args.max_repair_episodes,
        "lr_scheduler_type": args.lr_scheduler_type,
        "rank": DEFAULT_RANK,
        "lr": DEFAULT_LR,
        "reference_epochs": REFERENCE_EPOCHS,
        "max_base_epochs": MAX_BASE_EPOCHS,
        "repair_grid": repair_grid,
        "spotcheck_cell": spotcheck_cell,
        "created_at": int(time.time()),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2))
    logger.info("Wrote run_config.json: %s", cfg)
    return cfg


# ---------------------------------------------------------------------------
# Decay-steps formula  (verbatim from test15_retention_multiseed.py)
# ---------------------------------------------------------------------------


def decay_steps_for(n_keys: int, reference_epochs: int = REFERENCE_EPOCHS) -> int:
    """LR decay steps for the apples-to-apples linear scheduler, pinned to
    a *reference* epoch count rather than the per-arm ``num_epochs``.

    Per CLAUDE.md "Extended-training config": linear LR scheduler with fixed
    ``lr_decay_steps`` **decoupled from** ``num_train_epochs``.  The earlier
    Test 13/14/15 convention ``n_keys * num_epochs // 2`` coupled the
    scheduler to per-arm epoch budget and silently produced LR=0 from
    ``num_epochs / 2`` onward — see benchmarking.md §Test 16 redesign for
    the diagnostic.  This helper computes a *shared* schedule that is the
    same for every base/corrupted arm regardless of the arm's own epoch
    budget, so all arms see identical LR-vs-step trajectories up to
    whatever step they stop on.

    Formula: ``n_keys * reference_epochs``.  The decay window spans the
    full reference span, so LR is nonzero throughout every arm's training
    budget (``MAX_BASE_EPOCHS = REFERENCE_EPOCHS`` for base; corruption's
    own reference_epochs is its overwrite budget).  The Test 15-era
    ``// 2`` halving has been removed: it produced LR=0 from
    ``reference_epochs / 2`` onward, which silently zeroed the second half
    of every arm's training under the redesigned ``depths_past_floor``
    knob and collapsed D=10 vs D=30 into indistinguishable end states.
    See benchmarking.md §Test 16 redesign 2026-05-14.

    Args:
        n_keys: Number of indexed keys being trained.
        reference_epochs: Reference epoch count (default: REFERENCE_EPOCHS).
            Pass the largest plausible per-arm epoch count for the phase
            family so LR remains nonzero throughout every arm.

    Returns:
        Integer number of optimizer steps for the LR decay window.
    """
    return max(1, n_keys * reference_epochs)


# ---------------------------------------------------------------------------
# Config helpers  (verbatim from test15_retention_multiseed.py)
# ---------------------------------------------------------------------------


def _adapter_config(rank: int = DEFAULT_RANK, lr: float = DEFAULT_LR) -> AdapterConfig:
    """Return a standard AdapterConfig for Test 16.

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
    """Return a standard TrainingConfig for Test 16.

    Mirrors Test 15's apples-to-apples scheduler: linear + warmup_steps=10 +
    lr_decay_steps decoupled from epoch count.

    Args:
        num_epochs: Training epoch budget.
        seed: HF Trainer seed for optimizer / data shuffler.
        lr_scheduler_type: LR scheduler type (default ``"linear"``).
        lr_decay_steps: Decoupled decay window; None disables override.
        weight_decay: L2 regularisation (default 0.01).
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
# Dataset / phase builders  (verbatim from test15_retention_multiseed.py)
# ---------------------------------------------------------------------------


def load_triple_pool(
    total_keys: int,
    swap_keys: int,
    graph_path: Path = DEFAULT_GRAPH_SNAPSHOT,
) -> list[tuple[str, str, str]]:
    """Load the first ``total_keys + swap_keys`` unique ``(subject, predicate,
    object)`` triples from the LongMemEval graph snapshot.

    Replaces the prior PerLTQA QA-pair loader.  Test 16 now uses the
    production triple format ``{key, subject, predicate, object}``; the
    triple source is the same graph snapshot the quadruple_adapter
    experiment uses (a8b329d).  Order is deterministic: the order
    ``load_unique_triples`` returns from the graph's edge traversal.

    Args:
        total_keys: Number of indexed keys in the main pool.
        swap_keys: Number of additional triples reserved for the overwrite
            phase (whole-triple replacement of the last ``swap_keys`` keys).
        graph_path: Path to the graph snapshot JSON.  Defaults to
            ``DEFAULT_GRAPH_SNAPSHOT``.

    Returns:
        List of ``total_keys + swap_keys`` ``(subject, predicate, object)``
        3-tuples.

    Raises:
        FileNotFoundError: If the snapshot is missing.
        RuntimeError: If the snapshot does not yield enough unique triples.
    """
    needed = total_keys + swap_keys
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Graph snapshot not found: {graph_path}.  Build it via "
            f"`tresume lme` (experiments/lme_graph_builder.py)."
        )

    triples = load_unique_triples(graph_path)
    if len(triples) < needed:
        raise RuntimeError(
            f"Need {needed} unique triples, got {len(triples)} from {graph_path}. "
            f"Re-build the graph snapshot at a larger split."
        )

    pool = triples[:needed]
    logger.info(
        "load_triple_pool: kept %d of %d unique triples from %s",
        len(pool),
        len(triples),
        graph_path,
    )
    return pool


def build_phase_A_keyed(
    triple_pool: list[tuple[str, str, str]],
    total_keys: int = DEFAULT_N_KEYS,
) -> list[dict]:
    """Phase A (pretrain base_D): ``total_keys`` keyed triple entries.

    Args:
        triple_pool: Full ``(subject, predicate, object)`` pool (at least
            ``total_keys`` entries).
        total_keys: Number of indexed keys.

    Returns:
        List of entry dicts ``{key, subject, predicate, object}``.
    """
    return assign_keys(triple_pool[:total_keys], start_index=1)


def build_phase_B_swap_keyed(
    base_keyed: list[dict],
    swap_triples: list[tuple[str, str, str]],
    swap_keys: int = DEFAULT_SWAP_KEYS,
    total_keys: int = DEFAULT_N_KEYS,
) -> list[dict]:
    """Build the overwrite (corrupted_D) swap set: ``swap_keys`` keys,
    same ``key``, completely different ``(subject, predicate, object)``.

    The QA-format predecessor kept ``key`` + ``question`` and only swapped
    ``answer``.  Under the triple format, "swapping the answer" has no
    direct analogue — there is no separate question/answer split.  Instead,
    each swap key has its *entire* triple replaced by a triple from the
    reserve pool.  This matches the production overwrite semantics: the
    key still indexes "a fact", just a different fact, with no shared
    field with the original.

    Args:
        base_keyed: Pretrain keyed entries (all ``total_keys``).
        swap_triples: Reserve ``(subject, predicate, object)`` pool.
            Must contain at least ``swap_keys`` entries; the i-th reserve
            triple replaces the i-th of the last ``swap_keys`` base keys.
        swap_keys: Number of keys whose triples get replaced.
        total_keys: Total keys (used to derive the swap start slot).

    Returns:
        List of ``swap_keys`` entry dicts ``{key, subject, predicate,
        object}`` with the original key and the reserve triple's fields.
    """
    swap_start = total_keys - swap_keys
    assert len(swap_triples) >= swap_keys, (
        f"swap_triples has {len(swap_triples)} entries, need >= {swap_keys}"
    )
    return [
        {
            "key": kp["key"],
            "subject": s,
            "predicate": p,
            "object": o,
        }
        for kp, (s, p, o) in zip(base_keyed[swap_start:], swap_triples[:swap_keys])
    ]


# ---------------------------------------------------------------------------
# GPU cooldown helper  (verbatim from test14.py)
# ---------------------------------------------------------------------------


def gpu_cooldown_between(label: str) -> None:
    """Source gpu-cooldown and wait_for_cooldown 52 between GPU-intensive steps.

    Args:
        label: Human-readable label for the upcoming step (used in log messages).
    """
    logger.info("GPU cooldown before %s ...", label)
    try:
        subprocess.run(
            ["bash", "-c", "source ~/.local/bin/gpu-cooldown.sh && wait_for_cooldown 52"],
            check=False,
            timeout=3600,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("gpu_cooldown_between failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Safe probe helper  (verbatim from test15_retention_multiseed.py)
# ---------------------------------------------------------------------------


def _safe_probe(
    model,
    tokenizer,
    keyed: list[dict],
    registry: dict[str, int],
    adapter_name: str,
) -> dict:
    """Call the triple-aware evaluate_indexed_recall with guaranteed
    gradient_checkpointing re-enable.

    ``evaluate_entry_recall`` (= ``paramem.training.recall_eval
    .evaluate_indexed_recall``) internally disables gradient checkpointing
    before ``model.generate()``.  This helper re-enables in the ``finally``
    clause so every call site is safe regardless of the exception path.

    Args:
        model: Active PeftModel.
        tokenizer: Tokenizer.
        keyed: Keyed triple entries (``{key, subject, predicate, object}``)
            to probe.
        registry: SimHash registry for the entries.
        adapter_name: Active adapter name.

    Returns:
        Result dict from ``evaluate_entry_recall``.
    """
    try:
        return evaluate_entry_recall(
            model,
            tokenizer,
            keyed,
            registry,
            adapter_name=adapter_name,
            batch_size=RECALL_PROBE_BATCH_SIZE,
        )
    finally:
        try:
            model.gradient_checkpointing_enable()
        except Exception as exc:  # noqa: BLE001
            logger.warning("gradient_checkpointing_enable() failed in _safe_probe: %s", exc)


# ---------------------------------------------------------------------------
# Phase done marker writer  (adapted from test15_retention_multiseed.py)
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
        marker_name: Filename for the done marker JSON (e.g. ``"base_30_done.json"``).
        keyed: Keyed pairs trained in this phase.
        registry: SimHash registry for keyed.
        probe_state: EpochProbeState accumulator.
        final_recall: Final recall probe result dict.
        extra: Additional fields merged into the marker.
    """
    phase_dir.mkdir(parents=True, exist_ok=True)
    _safe_write_json(
        phase_dir / "quads.json",
        [
            {
                "key": kp["key"],
                "subject": kp["subject"],
                "predicate": kp["predicate"],
                "object": kp["object"],
            }
            for kp in keyed
        ],
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
# Phase runner  (verbatim from test15_retention_multiseed.py)
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
        early_stop_policy: EarlyStopPolicy controlling probe schedule and
            optional floor-relative stop via
            ``extra_epochs_past_first_perfect``.
        run_name: WandB / HF run name.
        retention_keyed: Optional unchanged-key list for dual-probe logging.
        retention_registry: Required if retention_keyed is provided.
        resume_from_checkpoint: Checkpoint path for tresume; None for fresh.

    Returns:
        Tuple of (model, metrics_dict, probe_state, wall_seconds).
    """
    examples = format_entry_training(keyed_to_train, tokenizer, max_length=1024)
    dataset = IndexedDataset(examples)

    probe_state = EpochProbeState()
    early_state = _EarlyStopState()

    # Wire batched recall probe to match server.yaml's
    # `consolidation.recall_probe_batch_size=16` (2026-05-18 cutover).
    # Production path in paramem/training/consolidation.py:4042-4048 does the
    # same `functools.partial` plumbing — Test 18 verified 137/137 exact-match
    # parity across b ∈ {1, 2, 4, 8, 16, 32, 64, 128}.  Falls back to the
    # callback's lazy default when batch_size=1 so existing callers see no
    # change.
    import functools

    _eval_fn: object | None
    if RECALL_PROBE_BATCH_SIZE > 1:
        _eval_fn = functools.partial(evaluate_entry_recall, batch_size=RECALL_PROBE_BATCH_SIZE)
    else:
        _eval_fn = None  # callback lazy-imports default

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
        eval_fn=_eval_fn,
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
# Failing-key identification  (verbatim from test15_retention_multiseed.py)
# ---------------------------------------------------------------------------


def _identify_failing_keys_via_probe(
    stop_probe: dict,
    unchanged_keyed: list[dict],
) -> list[dict]:
    """Identify failing unchanged keys from a fresh ``evaluate_indexed_recall`` probe.

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


# ---------------------------------------------------------------------------
# Directory fingerprint  (verbatim from test15_retention_multiseed.py)
# ---------------------------------------------------------------------------


def _sha256_dir(adapter_dir: Path) -> str | None:
    """Compute a deterministic SHA-256 fingerprint of all files under adapter_dir.

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


# ---------------------------------------------------------------------------
# Cell-dir naming
# ---------------------------------------------------------------------------


def _cell_dirname(D: int, lr: float, ep: int, wd: float, base_wd: float) -> str:
    """Return the canonical cell directory name for a repair cell.

    Format: ``repair_{D}_lr{lr:.0e}_ep{ep}`` for standard cells, with
    ``_wd{wd:g}`` appended for the weight-decay spot-check.

    Args:
        D: Pretrain depth in epochs.
        lr: Repair learning rate.
        ep: Repair epochs per episode.
        wd: Weight decay used for this cell.
        base_wd: The standard (non-spotcheck) weight decay to compare against.

    Returns:
        Directory name string.
    """
    name = f"repair_{D}_lr{lr:.0e}_ep{ep}"
    if wd != base_wd:
        name += f"_wd{wd:g}"
    return name


# ---------------------------------------------------------------------------
# Repair loop v2 (adapted from test15 run_repair_loop — adds repair_epochs_per_episode)
# ---------------------------------------------------------------------------


def run_repair_loop_v2(
    model,
    tokenizer,
    adapter_name: str,
    unchanged_keyed: list[dict],
    unchanged_registry: dict[str, int],
    cell_dir: Path,
    max_episodes: int,
    repair_lr: float,
    repair_epochs_per_episode: int,
    weight_decay: float,
    D: int,
    phase: str = "",
    seed: int = -1,
) -> tuple[object, dict]:
    """Run the post-corruption repair loop on failing unchanged keys.

    Identical to ``run_repair_loop`` in test15 except:
      - ``repair_epochs_per_episode`` replaces the hardcoded ``num_epochs=1``.
      - ``D``, ``repair_epochs_per_episode``, and ``weight_decay`` are added
        to the returned ``repair_log``.
      - ``phase_dir`` arg is renamed ``cell_dir`` for clarity.

    HARD RULE — originals preserved:
      - ``save_strategy="no"`` prevents HF Trainer from writing checkpoints.
      - ``output_dir`` is set to ``<cell_dir>/repair_episodes/episode_N/``
        (never the main adapter dir).
      - Final repaired state is saved to ``episodic_adapter_repaired/``
        as a sibling to ``episodic_adapter/``.

    Resume semantics: repair is non-resumable mid-loop.  If called without a
    done marker, the caller reloads the corrupted_D adapter and calls this
    function from episode 1.

    Args:
        model: Active PeftModel (loaded from corrupted_D adapter).
        tokenizer: Tokenizer.
        adapter_name: Adapter name (``"episodic"``).
        unchanged_keyed: Full list of unchanged keyed pairs.
        unchanged_registry: SimHash registry for unchanged_keyed.
        cell_dir: Repair cell directory root.
        max_episodes: Maximum repair episodes.
        repair_lr: Learning rate for repair.
        repair_epochs_per_episode: Number of epochs per repair episode.
        weight_decay: L2 regularisation for repair.
        D: Pretrain depth (epochs); logged in repair_log for reference.
        phase: Phase label (``"repair"``), recorded in repair_log.
        seed: RNG seed, recorded in repair_log.

    Returns:
        Tuple of (model, repair_log_dict).
    """
    # Fresh per-key probe at repair start (canonical RP2 source for this cell).
    stop_probe = _safe_probe(model, tokenizer, unchanged_keyed, unchanged_registry, adapter_name)

    failing_keyed = _identify_failing_keys_via_probe(stop_probe, unchanged_keyed)
    failing_keys_pre_repair: int = len(failing_keyed)
    repair_curve: list[dict] = []

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
            "D": D,
            "repair_epochs_per_episode": repair_epochs_per_episode,
            "weight_decay": weight_decay,
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
        "Repair loop: %d failing keys, max_episodes=%d, lr=%.1e, ep_per_ep=%d",
        len(failing_keyed),
        max_episodes,
        repair_lr,
        repair_epochs_per_episode,
    )

    episodes_used = 0
    stop_reason = "max_episodes"

    for episode in range(1, max_episodes + 1):
        # Honour pause at the start of every episode iteration.
        if paused_requested():
            logger.warning(
                "Pause file detected during repair %s episode %d — exiting cleanly.",
                cell_dir.name,
                episode,
            )
            write_paused_marker(
                cell_dir.parent.parent,  # run_dir (cell_dir = run_dir/seedN/repair_D_...)
                f"during repair {cell_dir.name} episode {episode}",
            )
            raise SystemExit(f"Training paused during repair {cell_dir.name} episode {episode}")

        ep_dir = cell_dir / "repair_episodes" / f"episode_{episode}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Repair episode %d/%d: %d failing keys", episode, max_episodes, len(failing_keyed)
        )

        examples = format_entry_training(failing_keyed, tokenizer, max_length=1024)
        dataset = IndexedDataset(examples)

        adapter_cfg = _adapter_config(lr=repair_lr)
        training_cfg = _training_config(
            num_epochs=repair_epochs_per_episode,
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
            run_name=f"test16-repair-{cell_dir.name}-ep{episode}",
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
    repaired_dir = cell_dir / "episodic_adapter_repaired"
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

    # Recovery count.
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
        "D": D,
        "repair_epochs_per_episode": repair_epochs_per_episode,
        "weight_decay": weight_decay,
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
        # Per-key raw probes.
        "rp2_entry_per_key": stop_probe.get("per_key", []),
        "rp3_final_per_key": final_retention.get("per_key", []),
    }
    return model, repair_log


# ---------------------------------------------------------------------------
# Integrity check  (adapted from test15 _verify_phase_integrity)
# ---------------------------------------------------------------------------


def _verify_phase_integrity(seed_dir: Path, seed: int, cfg: dict) -> None:
    """Verify done markers and adapter integrity for one completed (seed, D) pair.

    Checks:
    1. ``base_{D}_done.json`` and ``corrupted_{D}_done.json`` exist for all depths.
    2. For each completed repair cell: if ``episodic_adapter_repaired/`` exists,
       its sha256 must differ from ``corrupted_{D}/episodic_adapter/`` unless
       ``stop_reason == "no_failing_keys"``.

    Args:
        seed_dir: Per-seed directory (``run_dir / f"seed{seed}"``).
        seed: Integer seed value (used only for log messages).
        cfg: Run config dict (provides depths and repair_grid).
    """
    depths: list[int] = cfg["depths_past_floor"]
    repair_grid: list[list] = cfg["repair_grid"]
    spotcheck_cell: dict = cfg["spotcheck_cell"]
    base_wd: float = cfg["repair_weight_decay"]

    for D in depths:
        for marker_stem in [f"base_{D}", f"corrupted_{D}"]:
            marker = seed_dir / marker_stem / f"{marker_stem}_done.json"
            if not marker.exists():
                # Not necessarily an error during partial completion.
                logger.debug(
                    "_verify_phase_integrity: %s missing for seed %d (partial run?)",
                    marker,
                    seed,
                )
                continue

        # Check completed repair cells.
        corrupted_dir = seed_dir / f"corrupted_{D}"
        source_adapter_dir = corrupted_dir / "episodic_adapter"

        for lr, ep, wd in repair_grid:
            cell_name = _cell_dirname(D, lr, ep, wd, base_wd)
            cell_dir = seed_dir / cell_name
            done_marker = cell_dir / f"{cell_name}_done.json"
            if not done_marker.exists():
                continue
            repaired_dir = cell_dir / "episodic_adapter_repaired"
            if source_adapter_dir.exists() and repaired_dir.exists():
                sha_src = _sha256_dir(source_adapter_dir)
                sha_rep = _sha256_dir(repaired_dir)
                # Allow identical iff stop_reason == "no_failing_keys".
                if sha_src == sha_rep and sha_src is not None:
                    try:
                        cell_result = json.loads((cell_dir / "cell_result.json").read_text())
                        stop_reason = cell_result.get("repair", {}).get("stop_reason", "")
                    except Exception:  # noqa: BLE001
                        stop_reason = ""
                    if stop_reason != "no_failing_keys":
                        raise AssertionError(
                            f"_verify_phase_integrity: repaired adapter byte-identical to "
                            f"corrupted source for seed {seed} cell {cell_name} "
                            f"(stop_reason={stop_reason!r}) — repair did not write new weights "
                            f"and originals-preserved HARD RULE may be violated."
                        )

        # Spot-check cell.
        if D == spotcheck_cell["D"]:
            sc_lr = spotcheck_cell["lr"]
            sc_ep = spotcheck_cell["ep"]
            sc_wd = spotcheck_cell["wd"]
            sc_name = _cell_dirname(D, sc_lr, sc_ep, sc_wd, base_wd)
            sc_cell_dir = seed_dir / sc_name
            sc_done = sc_cell_dir / f"{sc_name}_done.json"
            if sc_done.exists():
                repaired_dir = sc_cell_dir / "episodic_adapter_repaired"
                if source_adapter_dir.exists() and repaired_dir.exists():
                    sha_src = _sha256_dir(source_adapter_dir)
                    sha_rep = _sha256_dir(repaired_dir)
                    if sha_src == sha_rep and sha_src is not None:
                        try:
                            cell_result = json.loads((sc_cell_dir / "cell_result.json").read_text())
                            stop_reason = cell_result.get("repair", {}).get("stop_reason", "")
                        except Exception:  # noqa: BLE001
                            stop_reason = ""
                        if stop_reason != "no_failing_keys":
                            raise AssertionError(
                                f"_verify_phase_integrity: repaired adapter byte-identical to "
                                f"corrupted source for seed {seed} spot-check cell {sc_name} "
                                f"(stop_reason={stop_reason!r}) — originals-preserved HARD RULE "
                                f"may be violated."
                            )

    logger.info("_verify_phase_integrity OK for seed%d", seed)


# ---------------------------------------------------------------------------
# Per-(seed, D) runner
# ---------------------------------------------------------------------------


def run_cell(
    model,
    tokenizer,
    seed: int,
    D: int,
    run_dir: Path,
    cfg: dict,
) -> dict:
    """Run the pretrain → overwrite → repair-sweep protocol for one (seed, D).

    Returns a dict with per-cell results for all completed repair cells at this
    (seed, D).  Already-done cells (``*_done.json`` present) are skipped.

    Args:
        model: Loaded base model (must NOT be a PeftModel at entry).
        tokenizer: Tokenizer.
        seed: RNG seed for this run.
        D: Pretrain epoch depth.
        run_dir: Top-level run directory.
        cfg: Run config dict (from load_or_write_run_config).

    Returns:
        Dict mapping cell_name → cell_result for all completed cells at this depth.
    """
    seed_dir = run_dir / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    n_keys: int = cfg["n_keys"]
    swap_keys: int = cfg["swap_keys"]
    swap_start = n_keys - swap_keys
    overwrite_epochs: int = cfg["overwrite_epochs"]
    pretrain_wd: float = cfg["pretrain_weight_decay"]
    lr_scheduler_type: str = cfg["lr_scheduler_type"]
    max_repair_episodes: int = cfg["max_repair_episodes"]

    # Two policies share the probe-every-epoch schedule but differ in
    # stop semantics:
    #   - `base_stop_policy` fires at `first_perfect_epoch + D` via the
    #     `extra_epochs_past_first_perfect` field on EarlyStopPolicy (added
    #     2026-05-13 for Test 16's depth-past-floor sweep).  Stable-perfect
    #     path is disabled via signal_from_epoch=10**9 — adopting server.yaml's
    #     `recall_signal_from_epoch=20` here would short-circuit the
    #     depth-past-floor mechanism (the stable-perfect path would fire at
    #     ~e25 regardless of D, collapsing D=10 and D=30 into the same arm).
    #   - `no_stop_policy` for corruption: probe every epoch, never halt;
    #     the fixed `overwrite_epochs` budget is the natural stop.
    #
    # Probe cadence stays at every epoch to preserve apples-to-apples with
    # the n=3 already-completed seeds (42/7/1337).  `recall_probe_every_n_epochs`
    # is NOT adopted from server.yaml — only `recall_probe_batch_size=16` is
    # (wired into eval_fn below).  Batched probe is byte-identical to serial
    # at b=16 per Test 18's empirical parity claim, so seeds 1 + 11 stay
    # comparable to the earlier three.
    base_stop_policy = EarlyStopPolicy(
        probe_from_epoch=1,
        signal_from_epoch=10**9,
        window=3,
        probe_every_n_epochs=1,
        extra_epochs_past_first_perfect=D,
    )
    no_stop_policy = EarlyStopPolicy(
        probe_from_epoch=1,
        signal_from_epoch=10**9,
        window=3,
        probe_every_n_epochs=1,
    )

    logger.info("=" * 72)
    logger.info(
        "Seed %d / D=%d: loading triple pool (n_keys=%d, swap_keys=%d)",
        seed,
        D,
        n_keys,
        swap_keys,
    )
    logger.info("=" * 72)

    triple_pool = load_triple_pool(n_keys, swap_keys)
    # Reserve triples that will overwrite the last `swap_keys` base entries
    # during corruption.  Whole-triple swap: same key, completely different
    # (subject, predicate, object).
    swap_triples = triple_pool[n_keys:]

    # -----------------------------------------------------------------------
    # Pretrain phase — base_D
    # -----------------------------------------------------------------------
    base_dir = seed_dir / f"base_{D}"
    base_done_path = base_dir / f"base_{D}_done.json"

    a_keyed: list[dict]
    a_registry: dict[str, int]
    encoding_floor_epoch: int | None

    if base_done_path.exists():
        logger.info("Seed %d D=%d base: already done — loading from disk.", seed, D)
        base_done = json.loads(base_done_path.read_text())
        a_keyed = json.loads((base_dir / "quads.json").read_text())
        a_registry = load_registry(base_dir / "simhash_registry.json")
        encoding_floor_epoch = base_done.get("encoding_floor_epoch")
        # Reload adapter.
        if isinstance(model, PeftModel):
            model = model.base_model.model
        a_slot = resolve_adapter_slot(base_dir / "episodic_adapter", "episodic", "")
        if a_slot is None:
            raise FileNotFoundError(
                f"Seed {seed} D={D}: episodic adapter not found under "
                f"{base_dir / 'episodic_adapter'}"
            )
        with _adapter_slot_for_load(a_slot) as _load_path:
            model = PeftModel.from_pretrained(model, str(_load_path), adapter_name="episodic")
        switch_adapter(model, "episodic")
    else:
        _check_pause(f"before base_{D} seed {seed}", run_dir)

        logger.info(
            "Seed %d base_%d — Pretrain: %d keys, up to %d epochs, stop at "
            "first_perfect+%d (shared LR decay over %d-epoch reference)",
            seed,
            D,
            n_keys,
            MAX_BASE_EPOCHS,
            D,
            REFERENCE_EPOCHS,
        )
        if isinstance(model, PeftModel):
            model = model.base_model.model
        model = create_adapter(model, _adapter_config(), "episodic")
        switch_adapter(model, "episodic")

        a_keyed = build_phase_A_keyed(triple_pool, total_keys=n_keys)
        a_registry = build_registry(a_keyed)

        a_ckpt = _find_latest_checkpoint(base_dir)
        # lr_decay_steps pinned to a shared reference epoch count — identical
        # across every D arm so each arm sees the same LR-vs-step trajectory
        # up to its (first_perfect + D) stop point.  CLAUDE.md: "decoupled
        # from num_train_epochs".
        pretrain_decay = decay_steps_for(n_keys)

        model, metrics_a, probe_a, wall_a = _run_phase(
            model=model,
            tokenizer=tokenizer,
            keyed_to_train=a_keyed,
            target_keyed=a_keyed,
            target_registry=a_registry,
            adapter_name="episodic",
            phase_dir=base_dir,
            phase_name=f"base_{D}",
            num_epochs=MAX_BASE_EPOCHS,
            seed=seed,
            lr_scheduler_type=lr_scheduler_type,
            lr_decay_steps=pretrain_decay,
            weight_decay=pretrain_wd,
            early_stop_policy=base_stop_policy,
            run_name=f"test16-base{D}-seed{seed}",
            resume_from_checkpoint=a_ckpt,
        )

        _exit_if_paused_mid_phase(probe_a, f"during base_{D} seed {seed}", MAX_BASE_EPOCHS, run_dir)

        # Floor anchor for the redesign: first_perfect_epoch (the first
        # observation of 100% recall on the training set).  Distinct from
        # stable_perfect_epoch (3 consecutive perfect epochs) — the
        # `extra_epochs_past_first_perfect` path in RecallEarlyStopCallback
        # fires on first_perfect to make the "depth past floor" definition
        # unambiguous regardless of late-epoch wobble.  Refuse-to-corrupt:
        # hard-fail if encoding never reached perfect within MAX_BASE_EPOCHS.
        # Silent acceptance of sub-perfect base was the bug Test 16's
        # redesign forbids — see benchmarking.md §Test 16 redesign.
        encoding_floor_epoch = probe_a.first_perfect_epoch
        if encoding_floor_epoch is None:
            raise RuntimeError(
                f"Seed {seed} base_{D}: first_perfect_epoch is None — encoding "
                f"never reached 100% recall within {MAX_BASE_EPOCHS} epochs at "
                f"n_keys={n_keys}, decay_steps={pretrain_decay}.  Refusing to "
                f"corrupt a sub-perfect base.  Inspect epoch_log.json and either "
                f"raise MAX_BASE_EPOCHS or REFERENCE_EPOCHS, or investigate why "
                f"this seed cannot encode under the apples-to-apples schedule."
            )

        final_a = _safe_probe(model, tokenizer, a_keyed, a_registry, "episodic")
        _manifest_a = build_manifest_for(
            model,
            tokenizer,
            "episodic",
            registry_path=None,
            key_count=len(a_keyed),
        )
        save_adapter(model, base_dir / "episodic_adapter", "episodic", manifest=_manifest_a)

        depth_past_floor = D  # by construction — stop callback fires at floor+D
        total_epochs_trained = encoding_floor_epoch + D
        _write_phase_done(
            base_dir,
            f"base_{D}_done.json",
            a_keyed,
            a_registry,
            probe_a,
            final_a,
            extra={
                "condition": f"base_{D}",
                "seed": seed,
                "D": D,
                "depth_past_floor": depth_past_floor,
                "encoding_floor_epoch": encoding_floor_epoch,
                "total_epochs_trained": total_epochs_trained,
                "max_base_epochs": MAX_BASE_EPOCHS,
                "reference_epochs": REFERENCE_EPOCHS,
                "lr_decay_steps": pretrain_decay,
                "train_loss": metrics_a.get("train_loss"),
                "wall_seconds": round(wall_a, 1),
            },
        )

    # -----------------------------------------------------------------------
    # Overwrite phase — corrupted_D
    # -----------------------------------------------------------------------
    corrupted_dir = seed_dir / f"corrupted_{D}"
    corrupted_done_path = corrupted_dir / f"corrupted_{D}_done.json"

    unchanged_keyed: list[dict]
    unchanged_registry: dict[str, int]
    overwrite_swap_keyed: list[dict]
    overwrite_swap_registry: dict[str, int]
    overwritten_keyed: list[dict]  # originals (before swap)
    overwritten_registry: dict[str, int]

    if corrupted_done_path.exists():
        logger.info("Seed %d D=%d corrupted: already done — loading from disk.", seed, D)
        unchanged_keyed = json.loads((corrupted_dir / "unchanged_keyed.json").read_text())
        unchanged_registry = load_registry(corrupted_dir / "unchanged_registry.json")
        overwrite_swap_keyed = json.loads((corrupted_dir / "overwrite_swap_keyed.json").read_text())
        overwrite_swap_registry = load_registry(corrupted_dir / "overwrite_swap_registry.json")
        overwritten_keyed = json.loads((corrupted_dir / "overwritten_keyed.json").read_text())
        overwritten_registry = load_registry(corrupted_dir / "overwritten_registry.json")
        # Reload corrupted adapter for repair cells.
        if isinstance(model, PeftModel):
            model = model.base_model.model
        c_slot = resolve_adapter_slot(corrupted_dir / "episodic_adapter", "episodic", "")
        if c_slot is None:
            raise FileNotFoundError(
                f"Seed {seed} D={D}: corrupted episodic adapter not found under "
                f"{corrupted_dir / 'episodic_adapter'}"
            )
        with _adapter_slot_for_load(c_slot) as _load_path:
            model = PeftModel.from_pretrained(model, str(_load_path), adapter_name="episodic")
        switch_adapter(model, "episodic")
    else:
        # Stopping conditions before cooldown (CLAUDE.md: all stops checked before cooldown).
        _check_pause(f"before corrupted_{D} seed {seed}", run_dir)
        gpu_cooldown_between(f"pretrain→overwrite seed {seed} D={D}")

        logger.info(
            "Seed %d corrupted_%d — Overwrite: %d swap keys, %d epochs (no early stop)",
            seed,
            D,
            swap_keys,
            overwrite_epochs,
        )
        switch_adapter(model, "episodic")

        unchanged_keyed = a_keyed[:swap_start]
        unchanged_registry = build_registry(unchanged_keyed)
        overwrite_swap_keyed = build_phase_B_swap_keyed(
            a_keyed, swap_triples, swap_keys=swap_keys, total_keys=n_keys
        )
        overwrite_swap_registry = build_registry(overwrite_swap_keyed)
        # Originals (for original_answer_resurfaced_rate probe).
        overwritten_keyed = a_keyed[swap_start:]
        overwritten_registry = build_registry(overwritten_keyed)

        c_ckpt = _find_latest_checkpoint(corrupted_dir)
        # lr_decay_steps calibrated to overwrite phase size.
        overwrite_decay = decay_steps_for(swap_keys, overwrite_epochs)

        model, metrics_c, probe_c, wall_c = _run_phase(
            model=model,
            tokenizer=tokenizer,
            keyed_to_train=overwrite_swap_keyed,
            target_keyed=overwrite_swap_keyed,
            target_registry=overwrite_swap_registry,
            adapter_name="episodic",
            phase_dir=corrupted_dir,
            phase_name=f"corrupted_{D}",
            num_epochs=overwrite_epochs,
            seed=seed,
            lr_scheduler_type=lr_scheduler_type,
            lr_decay_steps=overwrite_decay,
            weight_decay=pretrain_wd,
            early_stop_policy=no_stop_policy,
            run_name=f"test16-corrupted{D}-seed{seed}",
            retention_keyed=unchanged_keyed,
            retention_registry=unchanged_registry,
            resume_from_checkpoint=c_ckpt,
        )

        _exit_if_paused_mid_phase(
            probe_c, f"during corrupted_{D} seed {seed}", overwrite_epochs, run_dir
        )

        # Probe RP2 (retention of unchanged keys at end of overwrite).
        rp2_probe = _safe_probe(model, tokenizer, unchanged_keyed, unchanged_registry, "episodic")
        rp2_rate = rp2_probe["rate"]
        rp2_exact_count = rp2_probe["exact_count"]
        failing_keys_pre_repair_list = [
            entry["key"]
            for entry in (rp2_probe.get("per_key") or [])
            if not entry.get("exact_match")
        ]
        # Probe overwrite recall (new answers must be ≥ ~0.95 — overwrite "stuck").
        overwrite_recall_probe = _safe_probe(
            model, tokenizer, overwrite_swap_keyed, overwrite_swap_registry, "episodic"
        )
        overwrite_recall_rate = overwrite_recall_probe["rate"]
        if overwrite_recall_rate < 0.95:
            logger.warning(
                "Seed %d D=%d: overwrite_recall_rate=%.3f < 0.95 — overwrite may not "
                "have stuck; repair cells will run but results may be degenerate.",
                seed,
                D,
                overwrite_recall_rate,
            )

        # Save artifacts (probe used only for manifest build — result not stored separately;
        # overwrite_recall_rate was captured above via overwrite_recall_probe).
        _manifest_c = build_manifest_for(
            model,
            tokenizer,
            "episodic",
            registry_path=None,
            key_count=len(overwrite_swap_keyed),
        )
        save_adapter(model, corrupted_dir / "episodic_adapter", "episodic", manifest=_manifest_c)

        # Persist unchanged / swap / original keyed sets + registries.
        corrupted_dir.mkdir(parents=True, exist_ok=True)
        _safe_write_json(corrupted_dir / "unchanged_keyed.json", unchanged_keyed)
        save_registry(unchanged_registry, corrupted_dir / "unchanged_registry.json")
        _safe_write_json(corrupted_dir / "overwrite_swap_keyed.json", overwrite_swap_keyed)
        save_registry(overwrite_swap_registry, corrupted_dir / "overwrite_swap_registry.json")
        _safe_write_json(corrupted_dir / "overwritten_keyed.json", overwritten_keyed)
        save_registry(overwritten_registry, corrupted_dir / "overwritten_registry.json")
        # quads.json for corrupted phase = overwrite_swap_keyed.
        _safe_write_json(corrupted_dir / "quads.json", overwrite_swap_keyed)
        save_registry(overwrite_swap_registry, corrupted_dir / "simhash_registry.json")

        epoch_log_from_file: list[dict] = []
        if (corrupted_dir / "epoch_log.json").exists():
            try:
                epoch_log_from_file = json.loads((corrupted_dir / "epoch_log.json").read_text())
            except Exception:  # noqa: BLE001
                pass

        corrupted_done_marker = {
            "condition": f"corrupted_{D}",
            "seed": seed,
            "D": D,
            "encoding_floor_epoch": encoding_floor_epoch,
            "rp2_rate": round(rp2_rate, 6),
            "rp2_exact_count": rp2_exact_count,
            "failing_keys_pre_repair": failing_keys_pre_repair_list,
            "unchanged_total": len(unchanged_keyed),
            "overwrite_recall_rate": round(overwrite_recall_rate, 6),
            "train_loss": metrics_c.get("train_loss"),
            "wall_seconds": round(wall_c, 1),
            "epoch_log": epoch_log_from_file,
            "n_keys": len(overwrite_swap_keyed),
            "first_perfect_epoch": probe_c.first_perfect_epoch,
            "stable_perfect_epoch": probe_c.stable_perfect_epoch,
            "stop_epoch": probe_c.stop_epoch,
        }
        _safe_write_json(corrupted_done_path, corrupted_done_marker)
        logger.info(
            "Seed %d corrupted_%d done. RP2=%.3f overwrite_recall=%.3f",
            seed,
            D,
            rp2_rate,
            overwrite_recall_rate,
        )

    # -----------------------------------------------------------------------
    # Repair sweep — forked off corrupted_D
    # -----------------------------------------------------------------------
    # (No cooldown here: each not-yet-done repair cell does its own pause-check
    # then cooldown below, in the correct order.)

    # Load the corrupted done marker for annotation in cell_result.json.
    corrupted_done = json.loads(corrupted_done_path.read_text())
    rp2_for_cells = corrupted_done.get("rp2_rate", 0.0)
    rp2_exact_for_cells = corrupted_done.get("rp2_exact_count", 0)
    failing_keys_for_cells = corrupted_done.get("failing_keys_pre_repair", [])
    unchanged_total = corrupted_done.get("unchanged_total", len(unchanged_keyed))
    overwrite_recall_rate_pre = corrupted_done.get("overwrite_recall_rate", 1.0)
    enc_floor = corrupted_done.get("encoding_floor_epoch")

    # Build the full list of cells to run for this depth.
    cells_for_D: list[tuple] = []
    for lr, ep, wd in cfg["repair_grid"]:
        cells_for_D.append((D, lr, ep, wd))
    # Spot-check cell.
    sc = cfg["spotcheck_cell"]
    if sc["D"] == D:
        cells_for_D.append((D, sc["lr"], sc["ep"], sc["wd"]))

    cell_results: dict = {}
    base_wd = cfg["repair_weight_decay"]

    for D_cell, lr, ep, wd in cells_for_D:
        cell_name = _cell_dirname(D_cell, lr, ep, wd, base_wd)
        cell_dir = seed_dir / cell_name
        cell_done_path = cell_dir / f"{cell_name}_done.json"

        if cell_done_path.exists():
            logger.info("Seed %d cell %s: already done — loading result.", seed, cell_name)
            cell_result = json.loads((cell_dir / "cell_result.json").read_text())
            cell_results[cell_name] = cell_result
            continue

        _check_pause(f"before repair cell {cell_name} seed {seed}", run_dir)
        gpu_cooldown_between(f"repair cell {cell_name} seed {seed}")

        logger.info(
            "Seed %d cell %s: lr=%.1e ep=%d wd=%.3f",
            seed,
            cell_name,
            lr,
            ep,
            wd,
        )

        # Reload corrupted_D adapter fresh for each cell.
        if isinstance(model, PeftModel):
            model = model.base_model.model
        c_slot = resolve_adapter_slot(corrupted_dir / "episodic_adapter", "episodic", "")
        if c_slot is None:
            raise FileNotFoundError(
                f"Seed {seed} D={D}: corrupted adapter not found for cell {cell_name}"
            )
        with _adapter_slot_for_load(c_slot) as _load_path:
            model = PeftModel.from_pretrained(model, str(_load_path), adapter_name="episodic")
        switch_adapter(model, "episodic")

        cell_dir.mkdir(parents=True, exist_ok=True)

        model, repair_log = run_repair_loop_v2(
            model=model,
            tokenizer=tokenizer,
            adapter_name="episodic",
            unchanged_keyed=unchanged_keyed,
            unchanged_registry=unchanged_registry,
            cell_dir=cell_dir,
            max_episodes=max_repair_episodes,
            repair_lr=lr,
            repair_epochs_per_episode=ep,
            weight_decay=wd,
            D=D,
            phase="repair",
            seed=seed,
        )
        _safe_write_json(cell_dir / "repair_log.json", repair_log)

        # Overwrite-integrity probes (both mandatory, both via _safe_probe).
        overwrite_after_probe = _safe_probe(
            model, tokenizer, overwrite_swap_keyed, overwrite_swap_registry, "episodic"
        )
        overwrite_recall_after_repair = overwrite_after_probe["rate"]

        original_resurfaced_probe = _safe_probe(
            model, tokenizer, overwritten_keyed, overwritten_registry, "episodic"
        )
        original_answer_resurfaced_rate = original_resurfaced_probe["rate"]

        if overwrite_recall_after_repair < 0.95:
            logger.warning(
                "Seed %d cell %s: overwrite_recall_after_repair=%.3f < 0.95 "
                "— repair leaked into the swap set.",
                seed,
                cell_name,
                overwrite_recall_after_repair,
            )
        if original_answer_resurfaced_rate > 0.0:
            logger.info(
                "Seed %d cell %s: original_answer_resurfaced_rate=%.3f > 0 "
                "— active reversion to original answers detected.",
                seed,
                cell_name,
                original_answer_resurfaced_rate,
            )

        depth_past_floor = max(0, D - enc_floor) if enc_floor is not None else None

        cell_result: dict = {
            "seed": seed,
            "D": D,
            "encoding_floor_epoch": enc_floor,
            "depth_past_floor": depth_past_floor,
            "repair_lr": lr,
            "repair_epochs_per_episode": ep,
            "repair_weight_decay": wd,
            "max_repair_episodes": max_repair_episodes,
            "n_keys": n_keys,
            "swap_keys": swap_keys,
            "unchanged_total": unchanged_total,
            "overwrite_epochs": overwrite_epochs,
            "corruption": {
                "rp2_rate": rp2_for_cells,
                "rp2_exact_count": rp2_exact_for_cells,
                "failing_keys_pre_repair": failing_keys_for_cells,
                "overwrite_recall_rate_pre_repair": overwrite_recall_rate_pre,
            },
            "repair": {
                "RP2_rate": repair_log["RP2_rate"],
                "RP2_exact_count": repair_log["RP2_exact_count"],
                "RP3_rate": repair_log["RP3_rate"],
                "RP3_exact_count": repair_log["RP3_exact_count"],
                "RP3_total": repair_log["RP3_total"],
                "alignment_delta": repair_log["alignment_delta"],
                "corruption_residual": repair_log["corruption_residual"],
                "recovered_count": repair_log["recovered_count"],
                "still_failing_count": repair_log["still_failing_count"],
                "collateral_loss_count": repair_log["collateral_loss_count"],
                "episodes_used": repair_log["episodes_used"],
                "stop_reason": repair_log["stop_reason"],
                "repair_epochs_per_episode": ep,
                "weight_decay": wd,
                "curve": repair_log["curve"],
            },
            "overwrite_integrity": {
                "overwrite_recall_after_repair": round(overwrite_recall_after_repair, 6),
                "original_answer_resurfaced_rate": round(original_answer_resurfaced_rate, 6),
            },
            "probes": {
                "rp2_entry_per_key": repair_log.get("rp2_entry_per_key", []),
                "rp3_final_per_key": repair_log.get("rp3_final_per_key", []),
                "overwrite_after_repair_per_key": overwrite_after_probe.get("per_key", []),
                "original_resurfaced_per_key": original_resurfaced_probe.get("per_key", []),
            },
            "timestamp": int(time.time()),
        }
        _safe_write_json(cell_dir / "cell_result.json", cell_result)
        _safe_write_json(cell_dir / f"{cell_name}_done.json", {"timestamp": int(time.time())})
        logger.info(
            "Seed %d cell %s done. RP2=%.3f RP3=%.3f episodes=%d",
            seed,
            cell_name,
            repair_log["RP2_rate"],
            repair_log["RP3_rate"],
            repair_log["episodes_used"],
        )

        # Prune repair_episodes/episode_*/ HF scratch after cell completes.
        # Keep: repair_log.json, cell_result.json, *_done.json,
        #       episodic_adapter_repaired/.
        ep_scratch = cell_dir / "repair_episodes"
        if ep_scratch.exists():
            for ep_dir in ep_scratch.iterdir():
                if ep_dir.is_dir() and ep_dir.name.startswith("episode_"):
                    shutil.rmtree(ep_dir, ignore_errors=True)
            logger.info("Pruned repair_episodes scratch for cell %s", cell_name)

        cell_results[cell_name] = cell_result

    return cell_results


# ---------------------------------------------------------------------------
# Multi-seed aggregator
# ---------------------------------------------------------------------------


def compute_test16_aggregate(run_dir: Path, cfg: dict) -> dict:
    """Compute aggregate statistics from all completed repair cells.

    Reads per-cell ``cell_result.json`` files from disk and computes:
      - ``per_cell_rows``: flat list of per-cell metrics.
      - ``means_by_cell``: mean + percentile bootstrap CI over seeds
        (``BOOTSTRAP_RESAMPLES=10_000``) for rp3_rate, episodes_used.
        Also computes ``residual_failing_keys``: keys that fail across ALL
        seeds for a cell (common hard keys).
      - ``per_axis_summary``: depth / repair_lr / repair_epochs / wd
        slices with ``note`` fields flagging CIs.
      - ``encoding_floor_by_depth``: per-D floor statistics.

    Exploratory — no pass/fail gate.

    Args:
        run_dir: Top-level run directory.
        cfg: Run config dict.

    Returns:
        Aggregate dict written to ``test16_aggregate.json``.
    """
    seeds: list[int] = cfg["seeds"]
    depths: list[int] = cfg["depths_past_floor"]
    repair_grid: list[list] = cfg["repair_grid"]
    spotcheck_cell: dict = cfg["spotcheck_cell"]
    base_wd: float = cfg["repair_weight_decay"]

    per_cell_rows: list[dict] = []

    # Collect per-cell rows from disk.
    for seed in seeds:
        seed_dir = run_dir / f"seed{seed}"
        for D in depths:
            cells_for_D: list[tuple] = [(lr, ep, wd) for lr, ep, wd in repair_grid]
            sc = spotcheck_cell
            if sc["D"] == D:
                cells_for_D.append((sc["lr"], sc["ep"], sc["wd"]))
            for lr, ep, wd in cells_for_D:
                cell_name = _cell_dirname(D, lr, ep, wd, base_wd)
                cell_result_path = seed_dir / cell_name / "cell_result.json"
                if not cell_result_path.exists():
                    continue
                try:
                    cr = json.loads(cell_result_path.read_text())
                except Exception:  # noqa: BLE001
                    continue
                row = {
                    "seed": seed,
                    "D": D,
                    "encoding_floor_epoch": cr.get("encoding_floor_epoch"),
                    "repair_lr": lr,
                    "repair_epochs_per_episode": ep,
                    "weight_decay": wd,
                    "rp2_rate": cr.get("corruption", {}).get("rp2_rate"),
                    "rp3_rate": cr.get("repair", {}).get("RP3_rate"),
                    "episodes_used": cr.get("repair", {}).get("episodes_used"),
                    "recovered_count": cr.get("repair", {}).get("recovered_count"),
                    "collateral_loss_count": cr.get("repair", {}).get("collateral_loss_count"),
                    "alignment_delta": cr.get("repair", {}).get("alignment_delta"),
                    "corruption_residual": cr.get("repair", {}).get("corruption_residual"),
                    "overwrite_recall_after_repair": cr.get("overwrite_integrity", {}).get(
                        "overwrite_recall_after_repair"
                    ),
                    "original_answer_resurfaced_rate": cr.get("overwrite_integrity", {}).get(
                        "original_answer_resurfaced_rate"
                    ),
                }
                per_cell_rows.append(row)

    # Group rows by cell key.
    def _cell_key(D: int, lr: float, ep: int, wd: float) -> str:
        # Use a normalised key for means_by_cell dict.
        return f"D{D}_lr{lr:.0e}_ep{ep}_wd{wd:g}"

    from collections import defaultdict

    rows_by_cell: dict[str, list[dict]] = defaultdict(list)
    for row in per_cell_rows:
        key = _cell_key(
            row["D"], row["repair_lr"], row["repair_epochs_per_episode"], row["weight_decay"]
        )
        rows_by_cell[key].append(row)

    def _bootstrap_ci(
        values: list[float], n_resamples: int = BOOTSTRAP_RESAMPLES
    ) -> tuple[float, float, float]:
        """Return (mean, ci_low, ci_high) with 95% percentile bootstrap CI."""
        if not values:
            return (float("nan"), float("nan"), float("nan"))
        arr = np.array(values, dtype=float)
        n = len(arr)
        mean_val = float(np.mean(arr))
        if n == 1:
            return (mean_val, mean_val, mean_val)
        rng = np.random.default_rng(seed=0)
        boot_means = np.array(
            [np.mean(arr[rng.choice(n, size=n, replace=True)]) for _ in range(n_resamples)]
        )
        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))
        return (mean_val, ci_low, ci_high)

    means_by_cell: dict[str, dict] = {}
    for cell_key, rows in rows_by_cell.items():
        rp3_vals = [r["rp3_rate"] for r in rows if r.get("rp3_rate") is not None]
        ep_vals = [r["episodes_used"] for r in rows if r.get("episodes_used") is not None]
        orar_vals = [
            r["overwrite_recall_after_repair"]
            for r in rows
            if r.get("overwrite_recall_after_repair") is not None
        ]
        orig_vals = [
            r["original_answer_resurfaced_rate"]
            for r in rows
            if r.get("original_answer_resurfaced_rate") is not None
        ]
        rp2_vals = [r["rp2_rate"] for r in rows if r.get("rp2_rate") is not None]

        rp3_mean, rp3_lo, rp3_hi = _bootstrap_ci(rp3_vals)
        ep_mean, ep_lo, ep_hi = _bootstrap_ci(ep_vals)

        # Residual failing keys: keys still failing across ALL seeds.
        still_failing_per_seed: list[set] = []
        for r in rows:
            seed_val = r["seed"]
            D_val = r["D"]
            lr_val = r["repair_lr"]
            ep_val = r["repair_epochs_per_episode"]
            wd_val = r["weight_decay"]
            cell_name = _cell_dirname(D_val, lr_val, ep_val, wd_val, base_wd)
            cr_path = run_dir / f"seed{seed_val}" / cell_name / "cell_result.json"
            if not cr_path.exists():
                continue
            try:
                cr = json.loads(cr_path.read_text())
            except Exception:  # noqa: BLE001
                continue
            rp3_per_key = cr.get("probes", {}).get("rp3_final_per_key", [])
            failing = {e["key"] for e in rp3_per_key if not e.get("exact_match")}
            still_failing_per_seed.append(failing)

        if len(still_failing_per_seed) >= 2:
            common_failing = still_failing_per_seed[0].copy()
            for s in still_failing_per_seed[1:]:
                common_failing &= s
            residual_failing_keys = {k: len(still_failing_per_seed) for k in common_failing}
        else:
            # Need ≥2 seeds for a meaningful intersection; a single seed's misses
            # are not "residual across seeds". (Recomputed at end-of-run when all
            # seeds are present.)
            residual_failing_keys = {}

        means_by_cell[cell_key] = {
            "n_seeds": len(rows),
            "mean_rp2": round(float(np.mean(rp2_vals)), 6) if rp2_vals else None,
            "mean_rp3": round(rp3_mean, 6) if not np.isnan(rp3_mean) else None,
            "rp3_ci": (
                [round(rp3_lo, 6), round(rp3_hi, 6)] if not np.isnan(rp3_lo) else [None, None]
            ),
            "mean_episodes_used": round(ep_mean, 6) if not np.isnan(ep_mean) else None,
            "episodes_ci": (
                [round(ep_lo, 6), round(ep_hi, 6)] if not np.isnan(ep_lo) else [None, None]
            ),
            "mean_recovered": round(
                float(
                    np.mean(
                        [r["recovered_count"] for r in rows if r.get("recovered_count") is not None]
                    )
                ),
                3,
            )
            if rows
            else None,
            "mean_collateral": round(
                float(
                    np.mean(
                        [
                            r["collateral_loss_count"]
                            for r in rows
                            if r.get("collateral_loss_count") is not None
                        ]
                    )
                ),
                3,
            )
            if rows
            else None,
            "mean_overwrite_recall_after_repair": round(float(np.mean(orar_vals)), 6)
            if orar_vals
            else None,
            "mean_original_answer_resurfaced_rate": round(float(np.mean(orig_vals)), 6)
            if orig_vals
            else None,
            "residual_failing_keys": residual_failing_keys,
        }

    # Encoding floor by depth.
    encoding_floor_by_depth: dict[str, dict] = {}
    for D in depths:
        per_seed_floors: dict[str, int | None] = {}
        for seed in seeds:
            base_done_path = run_dir / f"seed{seed}" / f"base_{D}" / f"base_{D}_done.json"
            if base_done_path.exists():
                try:
                    bd = json.loads(base_done_path.read_text())
                    per_seed_floors[str(seed)] = bd.get("encoding_floor_epoch")
                except Exception:  # noqa: BLE001
                    pass
        floors = [v for v in per_seed_floors.values() if v is not None]
        encoding_floor_by_depth[str(D)] = {
            "per_seed": per_seed_floors,
            "mean": round(float(np.mean(floors)), 2) if floors else None,
        }

    # Per-axis summaries.
    def _axis_means(
        fixed: dict,
        var_key: str,
        var_values: list,
    ) -> dict[str, dict]:
        """Build a table of mean RP3 / episodes for one axis, others fixed."""
        table: dict[str, dict] = {}
        for v in var_values:
            query = {**fixed, var_key: v}
            matching = [
                r
                for r in per_cell_rows
                if all(
                    (
                        abs(r.get(k, float("nan")) - float(val)) < 1e-12
                        if isinstance(val, float)
                        else r.get(k) == val
                    )
                    for k, val in query.items()
                )
            ]
            rp3_v = [r["rp3_rate"] for r in matching if r.get("rp3_rate") is not None]
            ep_v = [r["episodes_used"] for r in matching if r.get("episodes_used") is not None]
            rp3_m, rp3_lo_v, rp3_hi_v = _bootstrap_ci(rp3_v)
            ep_m, _, _ = _bootstrap_ci(ep_v)
            table[str(v)] = {
                "n_seeds": len(matching),
                "mean_rp3": round(rp3_m, 6) if not np.isnan(rp3_m) else None,
                "rp3_ci": (
                    [round(rp3_lo_v, 6), round(rp3_hi_v, 6)]
                    if not np.isnan(rp3_lo_v)
                    else [None, None]
                ),
                "mean_episodes_used": round(ep_m, 6) if not np.isnan(ep_m) else None,
            }
        return table

    # Depth axis (fixed lr=1e-5, ep=1, wd=base_wd).
    fixed_depth = {"repair_lr": 1e-5, "repair_epochs_per_episode": 1, "weight_decay": base_wd}
    depth_table = _axis_means(fixed_depth, "D", depths)
    # Note: flag if cross-D RP2 difference confounds the depth comparison.
    rp2_by_depth: dict[int, list[float]] = {}
    for D in depths:
        rp2_by_depth[D] = [
            r["rp2_rate"]
            for r in per_cell_rows
            if r["D"] == D
            and abs(r["repair_lr"] - 1e-5) < 1e-12
            and r["repair_epochs_per_episode"] == 1
            and abs(r["weight_decay"] - base_wd) < 1e-12
            and r.get("rp2_rate") is not None
        ]
    depth_note_parts = []
    for D in depths:
        vals = rp2_by_depth.get(D, [])
        if vals:
            _, lo_d, hi_d = _bootstrap_ci(vals)
            mean_rp2_d = float(np.mean(vals))
            depth_note_parts.append(f"D={D} mean_RP2={mean_rp2_d:.3f} CI=[{lo_d:.3f},{hi_d:.3f}]")
    if len(depths) == 2 and all(rp2_by_depth.get(D) for D in depths):
        rp2_d0 = float(np.mean(rp2_by_depth[depths[0]]))
        rp2_d1 = float(np.mean(rp2_by_depth[depths[1]]))
        _, lo_d0, hi_d0 = _bootstrap_ci(rp2_by_depth[depths[0]])
        _, lo_d1, hi_d1 = _bootstrap_ci(rp2_by_depth[depths[1]])
        ci_hw_d0 = (hi_d0 - lo_d0) / 2
        ci_hw_d1 = (hi_d1 - lo_d1) / 2
        rp2_diff = abs(rp2_d1 - rp2_d0)
        max_ci_hw = max(ci_hw_d0, ci_hw_d1)
        if rp2_diff > max_ci_hw:
            depth_note_parts.append(
                f"CAUTION: cross-D RP2 difference ({rp2_diff:.3f}) exceeds "
                f"cross-seed CI half-width ({max_ci_hw:.3f}) — depth comparison is "
                f"confounded; corruption level differs across depths."
            )
        else:
            depth_note_parts.append(
                f"cross-D RP2 difference ({rp2_diff:.3f}) within CI half-width "
                f"({max_ci_hw:.3f}) — depth comparison is tentatively clean."
            )
    depth_note = "; ".join(depth_note_parts) if depth_note_parts else "insufficient data"

    # Repair-LR axis (fixed D=depths[0], ep=1, wd=base_wd).
    lr_vals = sorted({lr for lr, _ep, _wd in repair_grid})
    fixed_lr_axis = {"D": depths[0], "repair_epochs_per_episode": 1, "weight_decay": base_wd}
    lr_table = _axis_means(fixed_lr_axis, "repair_lr", lr_vals)

    # Repair-epochs axis (fixed D=depths[0], lr=1e-5, wd=base_wd).
    ep_vals_axis = sorted({ep for _lr, ep, _wd in repair_grid})
    fixed_ep_axis = {"D": depths[0], "repair_lr": 1e-5, "weight_decay": base_wd}
    ep_table = _axis_means(fixed_ep_axis, "repair_epochs_per_episode", ep_vals_axis)

    # Weight-decay spot-check (D=spotcheck_depth, lr=1e-5, ep=1; wd 0.01 vs spotcheck_wd).
    sc_D = spotcheck_cell["D"]
    sc_lr = spotcheck_cell["lr"]
    sc_ep = spotcheck_cell["ep"]
    sc_wd = spotcheck_cell["wd"]
    wd_fixed = {"D": sc_D, "repair_lr": sc_lr, "repair_epochs_per_episode": sc_ep}
    wd_table = _axis_means(wd_fixed, "weight_decay", [base_wd, sc_wd])
    wd_note = f"single LR ({sc_lr:.0e}) — no LR×wd interaction probed; do not over-read a null"

    per_axis_summary = {
        "depth_axis": {
            "fixed": fixed_depth,
            "table": depth_table,
            "note": depth_note,
        },
        "repair_lr_axis": {
            "fixed": fixed_lr_axis,
            "table": lr_table,
            "note": (
                "LR sweep at fixed D and ep; higher LR may speed recovery or cause instability."
            ),
        },
        "repair_epochs_axis": {
            "fixed": fixed_ep_axis,
            "table": ep_table,
            "note": (
                "Epochs-per-episode sweep at fixed D and LR; ep=3 trains 3x longer per episode."
            ),
        },
        "weight_decay_spotcheck": {
            "fixed": wd_fixed,
            "table": wd_table,
            "note": wd_note,
        },
    }

    aggregate: dict = {
        "model": cfg["model"],
        "seeds": seeds,
        "depths": depths,
        "repair_grid": repair_grid,
        "spotcheck_cell": spotcheck_cell,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "n_completed_cells": len(per_cell_rows),
        "encoding_floor_by_depth": encoding_floor_by_depth,
        "per_cell_rows": per_cell_rows,
        "means_by_cell": means_by_cell,
        "per_axis_summary": per_axis_summary,
        "notes": "exploratory mechanism work — no pass/fail gate",
        "timestamp": int(time.time()),
    }
    _safe_write_json(run_dir / "test16_aggregate.json", aggregate)
    logger.info(
        "test16_aggregate.json written: %d cell rows, %d means_by_cell entries",
        len(per_cell_rows),
        len(means_by_cell),
    )
    return aggregate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Test 16.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Test 16: Repair-Loop Sensitivity Sweep")
    parser.add_argument(
        "--model",
        default="mistral",
        choices=list(BENCHMARK_MODELS.keys()),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--depths-past-floor",
        "--depths_past_floor",
        nargs="+",
        type=int,
        default=DEFAULT_DEPTHS_PAST_FLOOR,
        dest="depths_past_floor",
        help=(
            "Depth-past-floor values to sweep (default: 0 10 30).  Each arm "
            "trains base for first_perfect_epoch + D epochs.  Replaces the "
            "earlier --depths total-epoch knob (see benchmarking.md §Test 16)."
        ),
    )
    parser.add_argument("--n-keys", "--n_keys", type=int, default=DEFAULT_N_KEYS, dest="n_keys")
    parser.add_argument(
        "--swap-keys", "--swap_keys", type=int, default=DEFAULT_SWAP_KEYS, dest="swap_keys"
    )
    parser.add_argument(
        "--overwrite-epochs",
        "--overwrite_epochs",
        type=int,
        default=DEFAULT_OVERWRITE_EPOCHS,
        dest="overwrite_epochs",
        help="Fixed epoch budget for the overwrite phase (all D).",
    )
    parser.add_argument(
        "--pretrain-weight-decay",
        "--pretrain_weight_decay",
        type=float,
        default=DEFAULT_PRETRAIN_WEIGHT_DECAY,
        dest="pretrain_weight_decay",
        help="Weight decay for pretrain and overwrite phases (>=0.1 per extended-training lore).",
    )
    parser.add_argument(
        "--repair-lrs",
        "--repair_lrs",
        nargs="+",
        type=float,
        default=DEFAULT_REPAIR_LRS,
        dest="repair_lrs",
    )
    parser.add_argument(
        "--repair-epochs",
        "--repair_epochs",
        nargs="+",
        type=int,
        default=DEFAULT_REPAIR_EPOCHS,
        dest="repair_epochs",
        help="Repair epochs-per-episode values to sweep.",
    )
    parser.add_argument(
        "--repair-weight-decay",
        "--repair_weight_decay",
        type=float,
        default=DEFAULT_REPAIR_WEIGHT_DECAY,
        dest="repair_weight_decay",
        help="Weight decay pinned for all 6 main repair cells.",
    )
    parser.add_argument(
        "--repair-wd-spotcheck",
        "--repair_wd_spotcheck",
        type=float,
        default=DEFAULT_REPAIR_WD_SPOTCHECK,
        dest="repair_wd_spotcheck",
        help="Weight decay for the single spot-check cell.",
    )
    parser.add_argument(
        "--spotcheck-depth",
        "--spotcheck_depth",
        type=int,
        default=DEFAULT_SPOTCHECK_DEPTH,
        dest="spotcheck_depth",
        help="Which depth gets the weight-decay spot-check cell (default: 30).",
    )
    parser.add_argument(
        "--max-repair-episodes",
        "--max_repair_episodes",
        type=int,
        default=DEFAULT_MAX_REPAIR_EPISODES,
        dest="max_repair_episodes",
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
        help="Override lr_decay_steps; if None, computed per-phase via decay_steps_for().",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke run: N=50, K=12, seeds=[42], depths=[30,50], 1 repair cell per depth. "
            "Writes to outputs/test16_repair_sweep/<model>/_smoke/<ts>/."
        ),
    )
    parser.add_argument(
        "--resume", action="store_true", help="Auto-find latest non-smoke run dir and continue."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for Test 16."""
    args = parse_args()

    # Smoke mode: override key parameters.
    if args.smoke:
        args.seeds = [42]
        args.depths_past_floor = [0, 10]
        args.n_keys = 50
        args.swap_keys = 12
        # 1 repair cell per depth (lr=1e-5, ep=1, wd=base_wd).
        args.repair_lrs = [1e-5]
        args.repair_epochs = [1]
        # No spotcheck cell in smoke (would need different wd list).
        # We suppress it by setting spotcheck_depth to a depth not in depths.
        args.spotcheck_depth = -1

    # Pre-flight disk space check.
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(OUTPUT_BASE).free
    if free_bytes <= DISK_HEADROOM_BYTES:
        raise SystemExit(
            f"Insufficient disk space: {free_bytes / 1024**3:.1f} GB free in {OUTPUT_BASE}; "
            f"need > {DISK_HEADROOM_BYTES / 1024**3:.0f} GB."
        )

    # Resolve output dir.
    if args.smoke:
        # Timestamped dir under _smoke/ — kept out of find_latest_run_dir so a
        # finished smoke never leaks into a `--resume` of a prod run.
        import datetime

        smoke_base = OUTPUT_BASE / args.model / "_smoke"
        smoke_base.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = smoke_base / ts
    elif args.resume:
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

    # On resume, read back persisted values (persisted > defaults; explicit CLI > persisted).
    if args.resume and not args.smoke:
        args.seeds = cfg.get("seeds", args.seeds)
        args.depths_past_floor = cfg.get("depths_past_floor", args.depths_past_floor)
        args.n_keys = cfg.get("n_keys", args.n_keys)
        args.swap_keys = cfg.get("swap_keys", args.swap_keys)
        args.overwrite_epochs = cfg.get("overwrite_epochs", args.overwrite_epochs)
        args.pretrain_weight_decay = cfg.get("pretrain_weight_decay", args.pretrain_weight_decay)
        args.repair_lrs = cfg.get("repair_lrs", args.repair_lrs)
        args.repair_epochs = cfg.get("repair_epochs", args.repair_epochs)
        args.repair_weight_decay = cfg.get("repair_weight_decay", args.repair_weight_decay)
        args.repair_wd_spotcheck = cfg.get("repair_wd_spotcheck", args.repair_wd_spotcheck)
        args.spotcheck_depth = cfg.get("spotcheck_depth", args.spotcheck_depth)
        args.max_repair_episodes = cfg.get("max_repair_episodes", args.max_repair_episodes)
        args.lr_scheduler_type = cfg.get("lr_scheduler_type", args.lr_scheduler_type)

    model_config = BENCHMARK_MODELS[cfg["model"]]

    with acquire_gpu(interactive=True):
        model, tokenizer = load_model_and_config(model_config)

        for seed in cfg.get("seeds", args.seeds):
            _check_pause(f"before seed {seed}", run_dir)

            logger.info("=" * 72)
            logger.info("Starting seed %d", seed)
            logger.info("=" * 72)

            for D in cfg.get("depths_past_floor", args.depths_past_floor):
                _check_pause(f"before D={D} seed {seed}", run_dir)

                logger.info("Starting seed %d D=%d", seed, D)

                # Ensure model is unwrapped before each (seed, D).
                if isinstance(model, PeftModel):
                    model = model.base_model.model

                run_cell(model, tokenizer, seed, D, run_dir, cfg)

                # Verify integrity + recompute aggregate after each (seed, D).
                _verify_phase_integrity(run_dir / f"seed{seed}", seed, cfg)
                compute_test16_aggregate(run_dir, cfg)

        unload_model(model, tokenizer)

    # Final aggregate recompute.
    compute_test16_aggregate(run_dir, cfg)
    logger.info("Test 16 complete.")


if __name__ == "__main__":
    main()
