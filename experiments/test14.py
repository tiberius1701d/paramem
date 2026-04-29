"""Test 14: Content-Free Placeholder Scaffold — Scale Validation + Multi-Round.

Research question
-----------------
Can a content-free placeholder scaffold (V1/V2/V3) replace Test 13's real-Q
scaffold (C1), and does the mechanism survive multi-round answer writes without
accumulating irrecoverable weight-space corruption?

Three modes
-----------
``--mode=pre``  (14a-pre)  — V1/V2/V3 head-to-head at N=100.  Produces
    ``pre_decision.json`` with the winning variant or null.
``--mode=scale``  (14a)  — Winning variant at default N=500 (``--n_keys``).
    Validates scaffold scale-up.
``--mode=multiround``  (14b)  — P0 + 3 rounds of fill/swap with RP1/RP2/RP3
    retention probes and touch-up.  Corruption-residual analysis.

Resume discipline
-----------------
Pass ``--resume`` to auto-find the latest matching run dir and continue from
where it left off.  Phase complete ↔ ``<phase>_done.json`` exists.
For 14a ``--mode=scale``: ``--resume --n_keys=N`` with N > saved value tops up
the data pool and extends training from the latest checkpoint.

GPU prerequisite
----------------
The ParaMem server must release the GPU.  This script uses
``experiments.utils.gpu_guard.acquire_gpu()`` which auto-switches the server
to cloud-only for the duration.

Usage
-----
    python experiments/test14.py --smoke                          # infra check
    python experiments/test14.py --mode=pre --resume              # 14a-pre
    python experiments/test14.py --mode=scale --resume            # 14a
    python experiments/test14.py --mode=multiround --scale-run=<path>  # 14b
    python experiments/test14.py --mode=multiround --resume       # 14b resume
"""

from __future__ import annotations

import argparse
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

from peft import PeftModel  # noqa: E402

from experiments.test13_journal_scaffold import load_qa_pool  # noqa: E402
from experiments.utils.early_stop import (  # noqa: E402
    ANALYSIS_POLICY,
    RecallEarlyStopCallback,
    _EarlyStopState,
    _safe_write_json,
)
from experiments.utils.gpu_guard import acquire_gpu  # noqa: E402
from experiments.utils.recall_diagnostics import (  # noqa: E402
    serialize_confusion_matrix,
)
from experiments.utils.scaffold import (  # noqa: E402
    PLACEHOLDER_STRINGS,
    V3,
    V3_EXTENDED,
    VARIANT_BUILDERS,
    VARIANTS,
    build_fill_keyed,
)
from experiments.utils.test_harness import (  # noqa: E402
    BENCHMARK_MODELS,
    IndexedDataset,
    evaluate_indexed_recall,
    load_model_and_config,
    model_output_dir,
    save_results,
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
from paramem.training.indexed_memory import (  # noqa: E402
    assign_keys,
    build_registry,
    format_indexed_training,
    load_registry,
    save_registry,
)
from paramem.training.trainer import train_adapter  # noqa: E402
from paramem.utils.config import AdapterConfig, TrainingConfig  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRE_N_KEYS = 100  # hard-capped for 14a-pre
PRE_FILL_KEYS = 20  # last 20 slots filled in Phase C (14a-pre)

OUTPUT_BASES = {
    "pre": project_root / "outputs" / "test14_pre",
    "scale": project_root / "outputs" / "test14a",
    "multiround": project_root / "outputs" / "test14b",
}

PAUSE_FILE = Path.home() / ".training_pause"

# Disk headroom required before model load (15 GB).
DISK_HEADROOM_BYTES = 15 * 1024**3

# Default training hyper-parameters (match Test 13 production config).
DEFAULT_NUM_EPOCHS = 30
DEFAULT_RANK = 8
DEFAULT_LR = 1e-4
TOUCHUP_LR = 1e-5
TOUCHUP_EPOCHS = 2

# Phase C fill fraction for 14a (scale mode).
FILL_FRACTION = 0.20  # last 20% of slots


# ---------------------------------------------------------------------------
# Probe state
# ---------------------------------------------------------------------------


@dataclass
class EpochProbeState:
    """Per-phase training metrics accumulator (mirrors test13/13b pattern)."""

    first_perfect_epoch: int | None = None
    stable_perfect_epoch: int | None = None
    stop_epoch: int | None = None
    epoch_log: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers — pause / markers
# ---------------------------------------------------------------------------


def paused_requested() -> bool:
    """Return True if the global pause file exists."""
    return PAUSE_FILE.exists()


def _check_pause(label: str, run_dir: Path | None = None) -> None:
    """Raise SystemExit with a clean message if pause file is present.

    Writes paused.json BEFORE raising so that ``tstatus`` can identify the
    paused state on resume.  The write is attempted even if ``run_dir`` is
    None (in which case the marker is skipped silently).

    Args:
        label: Human-readable phase label for log and paused.json.
        run_dir: Root run directory for writing ``paused.json``.  When
            None the marker write is skipped but the exit still fires.
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
    probe_state: "EpochProbeState",
    phase_label: str,
    num_epochs: int,
    run_dir: Path,
) -> None:
    """Halt cleanly when training was stopped mid-phase by the pause flag.

    Inference rule (no extra state field needed): a phase has completed iff
    the early-stop callback fired (``stop_epoch is not None``) OR the trainer
    ran the full epoch budget.  Anything else means
    ``RecallEarlyStopCallback`` set ``should_training_stop=True`` for a
    reason that is NOT natural convergence — and in test14 the only such
    reason is the ``~/.training_pause`` flag.

    When this fires, the phase has NOT finished — partial checkpoints exist
    on disk and the next tresume will pick them up via
    ``_find_latest_checkpoint``.  Writes ``paused.json`` with a ``during X``
    label (vs the boundary ``after X`` label written by ``_check_pause``)
    so ``tstatus`` can show the user where training stopped.

    No ``*_done.json`` marker is written; the caller's normal phase-end
    code (final probe, save_adapter, _write_phase_done) MUST be skipped
    when this raises.
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
        "Phase paused mid-training: %s at epoch %d — no done marker "
        "written; checkpoint preserved for tresume.",
        phase_label,
        last_epoch,
    )
    write_paused_marker(run_dir, phase_label, after_epoch=last_epoch)
    raise SystemExit(f"Training paused {phase_label} at epoch {last_epoch}")


def write_paused_marker(run_dir: Path, after_phase: str, after_epoch: int | None = None) -> None:
    """Write paused.json at clean pause exit."""
    marker = {
        "stopped_after_phase": after_phase,
        "stopped_after_epoch": after_epoch,
        "timestamp": int(time.time()),
    }
    _safe_write_json(run_dir / "paused.json", marker)
    logger.info("paused.json written (after_phase=%s)", after_phase)


def clear_paused_marker(run_dir: Path) -> None:
    """Remove paused.json if present (called at run start / tresume)."""
    p = run_dir / "paused.json"
    if p.exists():
        try:
            p.unlink()
        except OSError:
            pass


def gpu_cooldown_between(label: str) -> None:
    """Source gpu-cooldown and wait_for_cooldown 52 between GPU-intensive steps."""
    import subprocess

    logger.info("GPU cooldown before %s …", label)
    try:
        subprocess.run(
            ["bash", "-c", "source ~/.local/bin/gpu-cooldown.sh && wait_for_cooldown 52"],
            check=False,
            timeout=3600,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("gpu_cooldown_between failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for test14."""
    parser = argparse.ArgumentParser(description="Test 14: Content-Free Scaffold Validation")
    parser.add_argument(
        "--mode",
        choices=["pre", "scale", "multiround"],
        default="pre",
        help="Test mode: pre=14a-pre (N=100 3-variant), scale=14a (N=500 winner), "
        "multiround=14b (P0+3rounds).",
    )
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS) + ["auto"],
        default="auto",
        help="Scaffold variant. In 'pre' mode: auto runs all variants. In 'scale' mode: "
        "required (or read from run_config.json on resume).",
    )
    parser.add_argument(
        "--reuse-phase-a-from",
        type=str,
        default=None,
        dest="reuse_phase_a_from",
        help="Path to an existing 14a-pre run dir.  When set, Phase A is skipped for "
        "new variants (V3_extended/V4/V5) and the V3 Phase A adapter from that run "
        "is loaded as the baseline reference.  Saves ~8h of redundant compute.",
    )
    parser.add_argument(
        "--scaffold-resume-epochs",
        type=int,
        default=30,
        dest="scaffold_resume_epochs",
        help="Additional epochs to train in V3_extended Phase B (default 30, "
        "giving total_effective_epochs=60 when V3 Phase B ran 30 epochs).",
    )
    parser.add_argument(
        "--no-early-stop-phase-b",
        action="store_true",
        default=False,
        dest="no_early_stop_phase_b",
        help="Disable early-stop signalling during Phase B.  Used for V3_extended "
        "Phase B to deepen binding rather than stopping at the first stable window.",
    )
    parser.add_argument(
        "--n-keys",
        "--n_keys",
        type=int,
        default=None,
        dest="n_keys",
        help="Scale key count for --mode=scale (default 500). Ignored in pre/multiround.",
    )
    parser.add_argument(
        "--model",
        choices=["mistral"],
        default="mistral",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        dest="num_epochs",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=DEFAULT_RANK,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Auto-find the latest matching run dir and continue.",
    )
    parser.add_argument(
        "--scale-run",
        type=str,
        default=None,
        dest="scale_run",
        help="Path to completed 14a scale run dir (required for --mode=multiround "
        "if not using --resume).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke-test mode: N=10, 3 epochs, V1 only. Overrides --mode.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Run-dir helpers
# ---------------------------------------------------------------------------


def find_latest_run_dir(mode: str, model_name: str) -> Path | None:
    """Return the most recent non-smoke run dir for a given mode and model."""
    output_base = OUTPUT_BASES[mode]
    parent = output_base / model_name
    if not parent.is_dir():
        return None
    candidates: list[Path] = []
    for d in sorted(parent.iterdir()):
        if not d.is_dir():
            continue
        cfg_path = d / "run_config.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text())
                if cfg.get("smoke"):
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        candidates.append(d)
    return candidates[-1] if candidates else None


def load_or_write_run_config(run_dir: Path, args: argparse.Namespace) -> dict:
    """On first launch: write run_config.json from args.  On resume: read back.

    For --mode=scale with --resume --n_keys=N (extension): updates n_keys if
    the new value is larger.  Raises SystemExit if new value is smaller.
    """
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        logger.info("Loaded run_config.json: %s", cfg)
        # Extension discipline for scale mode.
        if args.mode == "scale" and args.n_keys is not None:
            saved_n = cfg.get("n_keys", 500)
            new_n = args.n_keys
            if new_n < saved_n:
                raise SystemExit(
                    f"--n_keys cannot shrink across resumes (saved={saved_n}, "
                    f"requested={new_n}); would corrupt saved state."
                )
            if new_n > saved_n:
                logger.info("Extending n_keys from %d to %d", saved_n, new_n)
                cfg["n_keys"] = new_n
                cfg_path.write_text(json.dumps(cfg, indent=2))
        return cfg

    # First launch — freeze config.
    n_keys = (
        args.n_keys if args.n_keys is not None else (500 if args.mode == "scale" else PRE_N_KEYS)
    )
    fill_keys = max(1, int(n_keys * FILL_FRACTION)) if args.mode == "scale" else PRE_FILL_KEYS

    cfg: dict = {
        "model": args.model,
        "mode": args.mode,
        "variant": getattr(args, "_active_variant", args.variant),
        "n_keys": n_keys,
        "total_keys": n_keys,
        "fill_keys": fill_keys,
        "num_epochs": args.num_epochs,
        "rank": args.rank,
        "lr": DEFAULT_LR,
        "seed": 42,
        "scale_run": str(getattr(args, "scale_run", None) or ""),
        "smoke": bool(args.smoke),
        "created_at": int(time.time()),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2))
    logger.info("Wrote run_config.json: %s", cfg)
    return cfg


# ---------------------------------------------------------------------------
# Adapter config helper
# ---------------------------------------------------------------------------


def _adapter_config(rank: int, lr: float = DEFAULT_LR) -> AdapterConfig:
    """Return a standard AdapterConfig for Test 14."""
    return AdapterConfig(
        rank=rank,
        alpha=rank * 2,
        learning_rate=lr,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        dropout=0.0,
    )


def _training_config(num_epochs: int, rank: int = DEFAULT_RANK) -> TrainingConfig:
    """Return a standard TrainingConfig for Test 14.

    Note: learning rate lives on AdapterConfig (see ``_adapter_config``), not
    TrainingConfig — TrainingConfig has no ``learning_rate`` field.
    """
    return TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=2,
        max_seq_length=1024,
        num_epochs=num_epochs,
        warmup_ratio=0.0,
        warmup_steps=10,
        weight_decay=0.1,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=42,
        save_strategy="epoch",
        save_total_limit=num_epochs,
    )


# ---------------------------------------------------------------------------
# Safe probe helper (F1 — §14.3 gradient_checkpointing contract)
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
    before ``model.generate()`` (test_harness.py:403).  If it raises, the
    model stays disabled and any subsequent training step runs without
    checkpointing.  This helper re-enables in the ``finally`` clause so every
    call site is safe regardless of the exception path.

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
# Phase A (fresh real Q+A baseline)
# ---------------------------------------------------------------------------


def run_phase_A_fresh(
    model,
    tokenizer,
    keyed: list[dict],
    registry: dict[str, int],
    adapter_name: str,
    args: argparse.Namespace,
    phase_dir: Path,
    run_name: str = "test14-A-fresh",
) -> tuple[object, dict, EpochProbeState, float]:
    """Run Phase A: fresh real Q+A baseline.

    Trains adapter_name on all n_keys real Q+A pairs for num_epochs.
    Returns (model, metrics, probe_state, wall_seconds).

    Gradient_checkpointing is managed by TrainingConfig; disabled before
    any generate() call in the callback and re-enabled after.
    """
    examples = format_indexed_training(keyed, tokenizer, max_length=1024)
    dataset = IndexedDataset(examples)

    probe_state = EpochProbeState()
    early_state = _EarlyStopState()
    callback = RecallEarlyStopCallback(
        model=model,
        tokenizer=tokenizer,
        target_keyed=keyed,
        target_registry=registry,
        adapter_name=adapter_name,
        policy=ANALYSIS_POLICY,
        state_out=early_state,
        progress_path=phase_dir / "progress.json",
        epoch_log_path=phase_dir / "epoch_log.json",
        first_perfect_log_path=phase_dir / "first_perfect_log.json",
        phase_name="Phase_A",
        num_epochs=args.num_epochs,
        pause_file=PAUSE_FILE,
    )

    adapter_cfg = _adapter_config(args.rank)
    training_cfg = _training_config(args.num_epochs, rank=args.rank)

    t0 = time.time()
    metrics = train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        adapter_name=adapter_name,
        training_config=training_cfg,
        adapter_config=adapter_cfg,
        output_dir=phase_dir / "adapter",
        run_name=run_name,
        callbacks_extra=[callback],
    )
    wall = time.time() - t0

    probe_state.first_perfect_epoch = early_state.first_perfect_epoch
    probe_state.stable_perfect_epoch = early_state.stable_perfect_epoch
    probe_state.stop_epoch = early_state.stop_epoch
    probe_state.epoch_log = early_state.epoch_log

    return model, metrics, probe_state, wall


# ---------------------------------------------------------------------------
# Phase B (scaffold build)
# ---------------------------------------------------------------------------


def run_phase_B_scaffold(
    model,
    tokenizer,
    scaffold_keyed: list[dict],
    registry: dict[str, int],
    args: argparse.Namespace,
    phase_dir: Path,
    run_name: str = "test14-B-scaffold",
) -> tuple[object, dict, EpochProbeState, float]:
    """Run Phase B: scaffold build.

    Trains the 'journal' adapter on the scaffold keyed pairs (placeholder Q+A).
    Returns (model, metrics, probe_state, wall_seconds).
    """
    examples = format_indexed_training(scaffold_keyed, tokenizer, max_length=1024)
    dataset = IndexedDataset(examples)

    probe_state = EpochProbeState()
    early_state = _EarlyStopState()
    callback = RecallEarlyStopCallback(
        model=model,
        tokenizer=tokenizer,
        target_keyed=scaffold_keyed,
        target_registry=registry,
        adapter_name="journal",
        policy=ANALYSIS_POLICY,
        state_out=early_state,
        progress_path=phase_dir / "progress.json",
        epoch_log_path=phase_dir / "epoch_log.json",
        first_perfect_log_path=phase_dir / "first_perfect_log.json",
        phase_name="Phase_B",
        num_epochs=args.num_epochs,
        pause_file=PAUSE_FILE,
    )

    adapter_cfg = _adapter_config(args.rank)
    training_cfg = _training_config(args.num_epochs, rank=args.rank)

    t0 = time.time()
    metrics = train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        adapter_name="journal",
        training_config=training_cfg,
        adapter_config=adapter_cfg,
        output_dir=phase_dir / "adapter",
        run_name=run_name,
        callbacks_extra=[callback],
    )
    wall = time.time() - t0

    probe_state.first_perfect_epoch = early_state.first_perfect_epoch
    probe_state.stable_perfect_epoch = early_state.stable_perfect_epoch
    probe_state.stop_epoch = early_state.stop_epoch
    probe_state.epoch_log = early_state.epoch_log

    return model, metrics, probe_state, wall


# ---------------------------------------------------------------------------
# Phase C (fill)
# ---------------------------------------------------------------------------


def run_phase_C_fill(
    model,
    tokenizer,
    scaffold_keyed: list[dict],
    fill_keyed: list[dict],
    fill_registry: dict[str, int],
    retention_keyed: list[dict],
    retention_registry: dict[str, int],
    args: argparse.Namespace,
    phase_dir: Path,
    run_name: str = "test14-C-fill",
    resume_from_checkpoint: Path | None = None,
) -> tuple[object, dict, EpochProbeState, float]:
    """Run Phase C: fill.

    Trains the 'journal' adapter on fill_keyed (real Q+A) with dual probing
    (fill + retention).  Early-stop fires on 3x 100% aggregate fill recall
    after epoch 10.

    Returns (model, metrics, probe_state, wall_seconds).
    """
    examples = format_indexed_training(fill_keyed, tokenizer, max_length=1024)
    dataset = IndexedDataset(examples)

    probe_state = EpochProbeState()
    early_state = _EarlyStopState()
    callback = RecallEarlyStopCallback(
        model=model,
        tokenizer=tokenizer,
        target_keyed=fill_keyed,
        target_registry=fill_registry,
        adapter_name="journal",
        policy=ANALYSIS_POLICY,
        state_out=early_state,
        progress_path=phase_dir / "progress.json",
        epoch_log_path=phase_dir / "epoch_log.json",
        first_perfect_log_path=phase_dir / "first_perfect_log.json",
        phase_name="Phase_C",
        num_epochs=args.num_epochs,
        pause_file=PAUSE_FILE,
        retention_keyed=retention_keyed if retention_keyed else None,
        retention_registry=retention_registry if retention_keyed else None,
    )

    adapter_cfg = _adapter_config(args.rank)
    training_cfg = _training_config(args.num_epochs, rank=args.rank)

    t0 = time.time()
    metrics = train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        adapter_name="journal",
        training_config=training_cfg,
        adapter_config=adapter_cfg,
        output_dir=phase_dir / "adapter",
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
# Retention probe helper (standalone, with gradient_checkpointing guard)
# ---------------------------------------------------------------------------


def run_round_retention_probe(
    model,
    tokenizer,
    unchanged_keyed: list[dict],
    unchanged_registry: dict[str, int],
    adapter_name: str,
    out_path: Path,
    label: str,
) -> dict:
    """Probe unchanged keys and persist to out_path.

    Implements the §14.3 gradient_checkpointing re-enable contract:
    ``evaluate_indexed_recall`` disables checkpointing internally; this
    helper re-enables it in the ``finally`` clause (always, including on
    exception path) so that any subsequent training step sees the correct
    memory configuration.

    Args:
        model: Active PeftModel.
        tokenizer: Tokenizer.
        unchanged_keyed: Keys to probe.
        unchanged_registry: SimHash registry for unchanged_keyed.
        adapter_name: Active adapter name.
        out_path: Destination JSON path.
        label: One of "RP1", "RP2", "RP3" (for logging).

    Returns:
        Dict with exact_count, total, rate, mean_confidence, per_key.
    """
    was_enabled = getattr(model, "is_gradient_checkpointing", True)
    result: dict = {}
    try:
        result = evaluate_indexed_recall(
            model,
            tokenizer,
            unchanged_keyed,
            unchanged_registry,
            adapter_name=adapter_name,
        )
        _safe_write_json(
            out_path,
            {
                "label": label,
                "exact_count": result["exact_count"],
                "total": result["total"],
                "rate": result["rate"],
                "mean_confidence": result["mean_confidence"],
                "per_key": result["per_key"],
            },
        )
        logger.info(
            "  %s: %d/%d (%.3f) mean_conf=%.3f",
            label,
            result["exact_count"],
            result["total"],
            result["rate"],
            result["mean_confidence"],
        )
    finally:
        # §14.3: always re-enable, including exception path.
        if was_enabled:
            try:
                model.gradient_checkpointing_enable()
            except Exception as exc:  # noqa: BLE001
                logger.warning("gradient_checkpointing_enable() failed in %s: %s", label, exc)
    return result


# ---------------------------------------------------------------------------
# Touch-up step (Test 13b idiom)
# ---------------------------------------------------------------------------


def run_touchup_step(
    model,
    tokenizer,
    failing_keyed: list[dict],
    adapter_name: str,
    output_dir: Path,
    passing_count: int = 0,
) -> dict:
    """Run 2-epoch LR=1e-5 touch-up on failing keys (Test 13b idiom).

    Returns a dict with touchup_train_loss, touchup_wall_seconds,
    touchup_n_failing_keys, touchup_n_recovered, touchup_n_collateral.

    Args:
        model: Active PeftModel.
        tokenizer: Tokenizer.
        failing_keyed: Keys that failed recall after RP2.
        adapter_name: Active adapter name.
        output_dir: ``<round>/touchup_adapter/`` directory.
        passing_count: Number of keys that were passing before touch-up
            (for collateral damage estimation via re-probe after touch-up).
    """
    if not failing_keyed:
        return {
            "touchup_triggered": False,
            "touchup_skipped": True,
            "touchup_n_failing_keys": 0,
            "touchup_n_recovered": 0,
            "touchup_n_collateral": 0,
            "touchup_wall_seconds": 0.0,
            "touchup_train_loss": None,
        }

    examples = format_indexed_training(failing_keyed, tokenizer, max_length=1024)
    dataset = IndexedDataset(examples)

    failing_registry = build_registry(failing_keyed)
    adapter_cfg = _adapter_config(rank=DEFAULT_RANK, lr=TOUCHUP_LR)
    training_cfg = _training_config(
        num_epochs=TOUCHUP_EPOCHS,
        rank=DEFAULT_RANK,
    )
    # Override save_strategy to "no" per Test 13b idiom.
    training_cfg.save_strategy = "no"
    training_cfg.save_total_limit = None

    t0 = time.time()
    metrics = train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        adapter_name=adapter_name,
        training_config=training_cfg,
        adapter_config=adapter_cfg,
        output_dir=output_dir,
        run_name="test14-touchup",
    )
    wall = time.time() - t0

    # Probe the previously-failing keys after touch-up.
    was_enabled = getattr(model, "is_gradient_checkpointing", True)
    try:
        post_touchup = evaluate_indexed_recall(
            model,
            tokenizer,
            failing_keyed,
            failing_registry,
            adapter_name=adapter_name,
        )
    finally:
        if was_enabled:
            try:
                model.gradient_checkpointing_enable()
            except Exception as exc:  # noqa: BLE001
                logger.warning("gradient_checkpointing_enable() failed in touchup probe: %s", exc)

    recovered = post_touchup["exact_count"]

    return {
        "touchup_triggered": True,
        "touchup_skipped": False,
        "touchup_n_failing_keys": len(failing_keyed),
        "touchup_n_recovered": recovered,
        "touchup_n_collateral": 0,  # computed by caller if needed via RP3
        "touchup_wall_seconds": round(wall, 1),
        "touchup_train_loss": metrics.get("train_loss"),
    }


# ---------------------------------------------------------------------------
# Round metrics (14b)
# ---------------------------------------------------------------------------


def compute_round_metrics(
    rp1: dict,
    rp2: dict,
    rp3: dict,
    fill_state: EpochProbeState,
    touchup_meta: dict,
    round_index: str,
    max_epochs: int = DEFAULT_NUM_EPOCHS,
    t13b_reference_stop_epoch: int | None = None,
) -> dict:
    """Compute per-round summary metrics for 14b (corruption-residual analysis).

    Args:
        rp1, rp2, rp3: Retention probe result dicts (rate, exact_count, total, …).
        fill_state: EpochProbeState from the fill training phase.
        touchup_meta: Return value of run_touchup_step.
        round_index: "P1", "P2", or "P3".
        max_epochs: Budget per phase (for hard-FAIL detection).
        t13b_reference_stop_epoch: T13b stable_perfect for drift comparison.

    Returns:
        Dict matching the ``<round>_done.json`` schema (§3 of plan).
    """
    ret_pre = rp1.get("rate", 0.0)
    ret_pretu = rp2.get("rate", 0.0)
    ret_post = rp3.get("rate", 0.0)

    alignment_delta = ret_post - ret_pretu
    corruption_residual = 1.0 - ret_post

    # B3 fix: compute touchup_n_collateral from RP2 vs RP3 per-key set diff.
    # Keys that pass pre-touchup (RP2) but fail post-touchup (RP3) are collateral
    # damage from the touch-up step.  When touchup_skipped the answer is 0 by
    # construction because no training occurred between RP2 and RP3.
    touchup_skipped = touchup_meta.get("touchup_skipped", True)
    if touchup_skipped:
        touchup_n_collateral = 0
    else:
        rp2_pass: set[str] = {
            entry["key"] for entry in rp2.get("per_key", []) if entry.get("exact_match", False)
        }
        rp3_fail: set[str] = {
            entry["key"] for entry in rp3.get("per_key", []) if not entry.get("exact_match", True)
        }
        touchup_n_collateral = len(rp2_pass & rp3_fail)

    # F10 fix: assert the touchup_skipped == (n_failing == 0) invariant.
    n_failing = touchup_meta.get("touchup_n_failing_keys", 0)
    assert touchup_skipped == (n_failing == 0), (
        f"invariant violated: touchup_skipped={touchup_skipped} but "
        f"touchup_n_failing_keys={n_failing}"
    )

    stop_epoch = fill_state.stop_epoch
    early_stop_fired = stop_epoch is not None

    stopped_after_epoch = fill_state.epoch_log[-1]["epoch"] if fill_state.epoch_log else None

    stop_epoch_drift: int | None = None
    if stop_epoch is not None and t13b_reference_stop_epoch is not None:
        stop_epoch_drift = stop_epoch - t13b_reference_stop_epoch

    # Build per-round fill final recall from last epoch_log entry.
    fill_final_recall = {"exact_count": None, "total": None, "rate": None}
    if fill_state.epoch_log:
        last = fill_state.epoch_log[-1]
        fill_entry = last.get("fill", {})
        fill_final_recall = {
            "exact_count": fill_entry.get("exact_count"),
            "total": fill_entry.get("total"),
            "rate": fill_entry.get("rate"),
        }

    return {
        "round_index": round_index,
        "fill_first_perfect_epoch": fill_state.first_perfect_epoch,
        "fill_stable_perfect_epoch": fill_state.stable_perfect_epoch,
        "fill_final_recall": fill_final_recall,
        "early_stop_fired": early_stop_fired,
        "stop_epoch": stop_epoch,
        "stop_epoch_drift_from_t13b": stop_epoch_drift,
        "stopped_after_epoch": stopped_after_epoch,
        "retention_pre_round": {
            "exact_count": rp1.get("exact_count"),
            "total": rp1.get("total"),
            "rate": ret_pre,
            "mean_confidence": rp1.get("mean_confidence"),
        },
        "retention_pre_touchup": {
            "exact_count": rp2.get("exact_count"),
            "total": rp2.get("total"),
            "rate": ret_pretu,
            "mean_confidence": rp2.get("mean_confidence"),
        },
        "retention_post_touchup": {
            "exact_count": rp3.get("exact_count"),
            "total": rp3.get("total"),
            "rate": ret_post,
            "mean_confidence": rp3.get("mean_confidence"),
        },
        "retention_alignment_delta": round(alignment_delta, 6),
        "retention_corruption_residual": round(corruption_residual, 6),
        "touchup_triggered": touchup_meta.get("touchup_triggered", False),
        "touchup_skipped": touchup_skipped,
        "touchup_n_failing_keys": touchup_meta.get("touchup_n_failing_keys", 0),
        "touchup_n_recovered": touchup_meta.get("touchup_n_recovered", 0),
        "touchup_n_collateral": touchup_n_collateral,
        "touchup_wall_seconds": touchup_meta.get("touchup_wall_seconds", 0.0),
        "touchup_train_loss": touchup_meta.get("touchup_train_loss"),
        "fill_wall_seconds": None,
        "fill_train_loss": None,
        "epoch_log": fill_state.epoch_log,
    }


def evaluate_14b_pass_fail(round_results: list[dict], cumulative: dict, inherited_n: int) -> dict:
    """Evaluate Test 14b PASS/CONCERN/FAIL per §5 of plan.

    Args:
        round_results: List of three round dicts (P1, P2, P3) as from compute_round_metrics.
        cumulative: Cumulative retention probe result dict.
        inherited_n: n_keys inherited from 14a.

    Returns:
        Dict with verdict, flags, and reason.
    """
    max_epochs = DEFAULT_NUM_EPOCHS

    concerns: list[str] = []
    fail_reasons: list[str] = []

    # Criterion 1: cumulative retention >= 0.95.
    cumul_rate = cumulative.get("rate", 0.0)
    if cumul_rate < 0.95:
        fail_reasons.append(f"cumulative_retention {cumul_rate:.3f} < 0.95")

    resids = [r.get("retention_corruption_residual", 1.0) for r in round_results]
    post_rates = [r.get("retention_post_touchup", {}).get("rate", 0.0) for r in round_results]
    stop_epochs = [r.get("stop_epoch") for r in round_results]
    alignment_deltas = [r.get("retention_alignment_delta", 0.0) for r in round_results]

    # Criterion 2: per-round post_touchup >= 0.95.
    for i, rate in enumerate(post_rates):
        if rate < 0.95:
            fail_reasons.append(f"Round {i + 1} retention_post_touchup {rate:.3f} < 0.95")

    # Criterion 3: corruption residual must NOT grow monotonically.
    residual_grew = len(resids) >= 3 and resids[0] < resids[1] < resids[2]
    if residual_grew:
        fail_reasons.append("retention_corruption_residual grew monotonically across rounds")

    # Criterion 4: early-stop must fire before budget.
    for i, round_r in enumerate(round_results):
        se = round_r.get("stop_epoch")
        if se is None or se >= max_epochs:
            fail_reasons.append(
                f"Round {i + 1} early_stop did not fire before max_epochs={max_epochs}"
            )

    # Criterion 5: recall at stop >= 0.95 (sanity check — always 100% by trigger construction).
    for i, round_r in enumerate(round_results):
        fr = round_r.get("fill_final_recall", {})
        if fr.get("rate") is not None and fr["rate"] < 0.95:
            fail_reasons.append(f"Round {i + 1} fill_final_recall {fr['rate']:.3f} < 0.95")

    # Criterion 6: recovery rate >= 0.95.
    for i, round_r in enumerate(round_results):
        if round_r.get("touchup_skipped"):
            continue
        n_fail = round_r.get("touchup_n_failing_keys", 0)
        n_rec = round_r.get("touchup_n_recovered", 0)
        if n_fail > 0 and n_rec / n_fail < 0.95:
            fail_reasons.append(
                f"Round {i + 1} recovery rate {n_rec}/{n_fail}={n_rec / n_fail:.3f} < 0.95"
            )

    # Criterion 7: collateral <= 2%.
    for i, round_r in enumerate(round_results):
        n_col = round_r.get("touchup_n_collateral", 0)
        pre_tu_rate = round_r.get("retention_pre_touchup", {}).get("rate", 0.0)
        n_total = round_r.get("retention_pre_touchup", {}).get("total", inherited_n)
        passing_pre = int(pre_tu_rate * n_total)
        if passing_pre > 0 and n_col / passing_pre > 0.02:
            fail_reasons.append(
                f"Round {i + 1} collateral {n_col}/{passing_pre}={n_col / passing_pre:.3f} > 0.02"
            )

    # CONCERN checks.
    # (a) Alignment delta growing substantially.
    if len(alignment_deltas) >= 3 and alignment_deltas[0] > 0:
        ratio = alignment_deltas[2] / alignment_deltas[0]
        if ratio > 1.5 and alignment_deltas[2] > alignment_deltas[1]:
            concerns.append(
                f"alignment_delta grew: {alignment_deltas[0]:.4f}->{alignment_deltas[2]:.4f} "
                f"(ratio={ratio:.2f} > 1.5)"
            )

    # (b/c) stop_epoch outside reference window.
    for i, se in enumerate(stop_epochs):
        if se is not None:
            if se < 12:
                concerns.append(
                    f"Round {i + 1} stop_epoch={se} < 12 (fired earlier than reference)"
                )
            elif se > 25:
                concerns.append(f"Round {i + 1} stop_epoch={se} > 25 (fired later than reference)")

    # Hard FAIL: any corruption residual > 5%.
    for i, resid in enumerate(resids):
        if resid > 0.05:
            fail_reasons.append(f"Round {i + 1} corruption_residual {resid:.3f} > 0.05")
    # Hard FAIL: any post-touchup < 0.90.
    for i, rate in enumerate(post_rates):
        if rate < 0.90:
            fail_reasons.append(f"Round {i + 1} retention_post_touchup {rate:.3f} < 0.90")

    if fail_reasons:
        verdict = "FAIL"
    elif concerns:
        verdict = "CONCERN"
    else:
        verdict = "PASS"

    return {
        "verdict": verdict,
        "fail_reasons": fail_reasons,
        "concerns": concerns,
        "corruption_residual_grew_monotonically": residual_grew,
        "alignment_delta_concern_trigger": len(alignment_deltas) >= 3
        and alignment_deltas[0] > 0
        and alignment_deltas[2] / alignment_deltas[0] > 1.5
        and alignment_deltas[2] > alignment_deltas[1],
    }


# ---------------------------------------------------------------------------
# Checkpoint finder (mirrors test13b pattern)
# ---------------------------------------------------------------------------


def _find_latest_checkpoint(adapter_dir: Path) -> Path | None:
    """Return the checkpoint-N dir with the highest step number in adapter_dir."""
    candidates = [Path(p) for p in glob(str(adapter_dir / "checkpoint-*")) if Path(p).is_dir()]
    if not candidates:
        return None

    def _step(p: Path) -> int:
        try:
            return int(p.name.split("-", 1)[1])
        except (IndexError, ValueError):
            return -1

    return max(candidates, key=_step)


# ---------------------------------------------------------------------------
# Leakage check (mirrors test13 leakage_count)
# ---------------------------------------------------------------------------


def _leakage_count(per_key_results: list[dict]) -> int:
    """Count keys whose recalled answer contains a placeholder substring."""
    count = 0
    for entry in per_key_results:
        recalled = entry.get("recalled") or {}
        recalled_a = recalled.get("answer", "") if isinstance(recalled, dict) else ""
        if any(ph in recalled_a for ph in PLACEHOLDER_STRINGS):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Phase A reuse helper (for V3_extended / V4 / V5)
# ---------------------------------------------------------------------------


def load_phase_a_from_existing(
    model,
    source_run_dir: Path,
    source_variant: str,
    dest_variant_dir: Path,
) -> tuple[object, dict]:
    """Load Phase A adapter from an existing run dir without retraining.

    Used when ``--reuse-phase-a-from`` is passed.  Reads the Phase A
    done marker from ``<source_run_dir>/<source_variant>/A/A_done.json``,
    loads the adapter via ``PeftModel.from_pretrained`` using the existing
    ``resolve_adapter_slot`` pattern, and writes a ``phase_a_reused.json``
    marker to ``<dest_variant_dir>/A/`` recording provenance.

    This follows the PEFT adapter-load pattern from test14.py Phase C resume
    (lines 1424-1438) — no ``delete_adapter`` / ``create_adapter`` cycle.

    Args:
        model: Base model (must NOT be a PeftModel; unwrap before calling).
        source_run_dir: Path to the 14a-pre run dir that contains the source
            variant directory.
        source_variant: Variant name in source_run_dir (typically "V3").
        dest_variant_dir: Destination variant directory under the current
            run dir (e.g. ``<run_dir>/V4``).  The ``A/`` subdirectory is
            created and ``phase_a_reused.json`` written here.

    Returns:
        Tuple of (model, a_done_data) where model is the loaded PeftModel
        and a_done_data is the dict loaded from A_done.json.

    Raises:
        FileNotFoundError: If the source A_done.json or adapter is missing.
    """
    a_done_path = source_run_dir / source_variant / "A" / "A_done.json"
    if not a_done_path.exists():
        raise FileNotFoundError(f"Phase A reuse: A_done.json not found at {a_done_path}")
    a_done = json.loads(a_done_path.read_text())

    adapter_name_a = f"episodic_{source_variant.lower()}"
    a_adapter_dir = source_run_dir / source_variant / "A" / "adapter"
    a_slot = resolve_adapter_slot(a_adapter_dir, adapter_name_a, "")
    if a_slot is None:
        raise FileNotFoundError(f"Phase A reuse: adapter slot not found under {a_adapter_dir}")

    # Load adapter; model must be the bare base model here (not PeftModel).
    # Use the module-level PeftModel import so test mocks of
    # `experiments.test14.PeftModel` take effect.  _adapter_slot_for_load
    # transparently decrypts age-encrypted safetensors via memfd, no-op on
    # plaintext slots.
    with _adapter_slot_for_load(a_slot) as _load_path:
        model = PeftModel.from_pretrained(model, str(_load_path), adapter_name=adapter_name_a)
    logger.info(
        "Phase A reuse: loaded %s adapter from %s",
        adapter_name_a,
        a_slot,
    )

    # Write provenance marker.
    dest_a_dir = dest_variant_dir / "A"
    dest_a_dir.mkdir(parents=True, exist_ok=True)
    reuse_marker = {
        "source_run_dir": str(source_run_dir),
        "source_variant": source_variant,
        "source_adapter_slot": str(a_slot),
        "source_final_recall_rate": a_done.get("final_recall", {}).get("rate"),
        "source_stop_epoch": a_done.get("stop_epoch"),
        "source_wall_seconds": a_done.get("wall_seconds"),
        "timestamp": int(time.time()),
    }
    _safe_write_json(dest_a_dir / "phase_a_reused.json", reuse_marker)
    logger.info("phase_a_reused.json written to %s", dest_a_dir)

    return model, a_done


# ---------------------------------------------------------------------------
# Phase B scaffold-resume helper (for V3_extended)
# ---------------------------------------------------------------------------


def run_phase_B_scaffold_resume(
    model,
    tokenizer,
    scaffold_keyed: list[dict],
    registry: dict[str, int],
    args: argparse.Namespace,
    phase_dir: Path,
    run_name: str = "test14-B-scaffold-resume",
    disable_early_stop: bool = True,
) -> tuple[object, dict, EpochProbeState, float]:
    """Run Phase B scaffold-resume for V3_extended.

    Continues training from loaded adapter weights (V3's Phase B adapter must
    already be active in the model).  Unlike ``run_phase_B_scaffold``, this
    function does NOT call ``create_adapter`` — the adapter is already loaded.

    HF Trainer optimizer/scheduler resume:
      The model carries the trained LoRA weights from V3 Phase B.  The HF
      Trainer starts a *fresh* optimizer and LR schedule for the additional
      epochs — resuming optimizer state from V3's Phase B checkpoint directory
      is deliberately NOT done here because (a) the checkpoint dir belongs to
      a different variant and (b) fresh optimizer state is the safer default
      for a deepening-not-diverging extension run.  This choice is documented
      here so it is visible at review.

    Early-stop policy:
      When ``disable_early_stop=True`` (default), the stop signal from
      ``EarlyStopPolicy`` is suppressed by setting ``signal_from_epoch`` to a
      value beyond the training budget.  The goal of V3_extended Phase B is to
      deepen binding (run all ``args.scaffold_resume_epochs``), not to stop at
      the first stable window.

    Args:
        model: Active PeftModel with "journal" adapter already loaded.
        tokenizer: Tokenizer.
        scaffold_keyed: V3 scaffold keyed pairs (uniform "pending"/"pending").
        registry: SimHash registry for scaffold_keyed.
        args: Parsed CLI namespace; uses ``args.scaffold_resume_epochs`` as
            epoch budget and ``args.rank``.
        phase_dir: Output directory for this Phase B extension.
        run_name: W&B / HF run name string.
        disable_early_stop: When True, the stop signal cannot fire during this
            phase (signal_from_epoch > num_epochs).  Pass False only for
            testing or if the caller wants early-stop behaviour.

    Returns:
        Tuple of (model, metrics, probe_state, wall_seconds).
    """
    from experiments.utils.early_stop import EarlyStopPolicy

    num_epochs = getattr(args, "scaffold_resume_epochs", 30)

    if disable_early_stop:
        # Suppress stop signal by moving signal_from_epoch beyond the budget.
        # probe_from_epoch=1 still lets us capture epoch-level diagnostics.
        policy = EarlyStopPolicy(
            probe_from_epoch=1,
            signal_from_epoch=num_epochs + 1,
            window=3,
            probe_every_n_epochs=1,
        )
    else:
        policy = ANALYSIS_POLICY

    examples = format_indexed_training(scaffold_keyed, tokenizer, max_length=1024)
    dataset = IndexedDataset(examples)

    probe_state = EpochProbeState()
    early_state = _EarlyStopState()
    callback = RecallEarlyStopCallback(
        model=model,
        tokenizer=tokenizer,
        target_keyed=scaffold_keyed,
        target_registry=registry,
        adapter_name="journal",
        policy=policy,
        state_out=early_state,
        progress_path=phase_dir / "progress.json",
        epoch_log_path=phase_dir / "epoch_log.json",
        first_perfect_log_path=phase_dir / "first_perfect_log.json",
        phase_name="Phase_B_extended",
        num_epochs=num_epochs,
        pause_file=PAUSE_FILE,
    )

    # Use _adapter_config and _training_config, matching the existing scaffold
    # build pattern in run_phase_B_scaffold.
    adapter_cfg = _adapter_config(args.rank)
    training_cfg = _training_config(num_epochs, rank=args.rank)

    t0 = time.time()
    metrics = train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        adapter_name="journal",
        training_config=training_cfg,
        adapter_config=adapter_cfg,
        output_dir=phase_dir / "adapter",
        run_name=run_name,
        callbacks_extra=[callback],
    )
    wall = time.time() - t0

    probe_state.first_perfect_epoch = early_state.first_perfect_epoch
    probe_state.stable_perfect_epoch = early_state.stable_perfect_epoch
    probe_state.stop_epoch = early_state.stop_epoch
    probe_state.epoch_log = early_state.epoch_log

    return model, metrics, probe_state, wall


# ---------------------------------------------------------------------------
# Phase-done marker writer
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
    """Persist keyed_pairs, registry, and phase-done marker."""
    phase_dir.mkdir(parents=True, exist_ok=True)
    _safe_write_json(
        phase_dir / "keyed_pairs.json",
        [{"key": kp["key"], "question": kp["question"], "answer": kp["answer"]} for kp in keyed],
    )
    save_registry(registry, phase_dir / "simhash_registry.json")

    # Serialize confusion matrix if first_perfect_log.json exists.
    confusion_summary: dict = {}
    fpl_path = phase_dir / "first_perfect_log.json"
    if fpl_path.exists():
        try:
            fpl = json.loads(fpl_path.read_text())
            expected_by_key = {kp["key"]: kp["answer"] for kp in keyed}
            confusion_summary = serialize_confusion_matrix(fpl, expected_by_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("confusion matrix serialization failed: %s", exc)

    # Q/A split at final epoch (last epoch_log entry with q_a_split).
    qa_split_final: dict = {}
    for entry in reversed(probe_state.epoch_log):
        if "q_a_split" in entry:
            qa_split_final = entry["q_a_split"]
            break

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
        "epoch_log": probe_state.epoch_log,
        "q_a_split_at_final": qa_split_final,
        "confusion_matrix": confusion_summary,
        **extra,
    }
    _safe_write_json(phase_dir / marker_name, marker)
    logger.info("Phase marker written: %s", phase_dir / marker_name)


# ---------------------------------------------------------------------------
# Pre-mode winner decision
# ---------------------------------------------------------------------------


def decide_pre_winner(run_dir: Path, existing_winner: str | None = None) -> str | None:
    """Read per-variant C_done.json and apply winner rules.

    Scans ALL variant subdirectories present under ``run_dir`` — not just the
    original three (V1/V2/V3) — so that V3_extended/V4/V5 are automatically
    included when their phase markers exist.

    Extended-run override rule
    --------------------------
    When ``existing_winner`` is provided (the winner from the original V1/V2/V3
    pass), a new variant replaces it only if its Phase C ``stop_epoch`` is
    strictly ``<= 14``.  Otherwise the existing winner is preserved.  This
    threshold is documented here and logged with the verdict.

    Base pass criteria (same as original V1/V2/V3 rules)
    -----------------------------------------------------
    - b_final_rate >= 0.99
    - final_fill_rate >= 0.95
    - stable_c <= 22 (or None)
    - leakage == 0
    - q_only_frac <= 0.05
    - q_only_trend < 2

    Args:
        run_dir: Root run directory containing per-variant subdirectories.
        existing_winner: Optional previously-decided winner variant.  When
            provided, new variants must have stop_epoch <= 14 to override it.

    Returns:
        Winner variant string, or None if no variant passes.
        Writes ``pre_decision.json``.
    """
    NEW_VARIANT_OVERRIDE_THRESHOLD = 14

    passing: list[dict] = []
    per_variant: dict = {}

    # Discover all variant dirs dynamically: any subdir that has a B/ and C/
    # structure.  This includes V1/V2/V3 and any newly run V3_extended/V4/V5.
    candidate_dirs = sorted(
        d for d in run_dir.iterdir() if d.is_dir() and (d / "C" / "C_done.json").exists()
    )

    for v_dir in candidate_dirs:
        v = v_dir.name
        c_done_path = v_dir / "C" / "C_done.json"
        try:
            c = json.loads(c_done_path.read_text())
        except Exception:  # noqa: BLE001
            per_variant[v] = {"status": "unreadable"}
            continue

        b_done_path = v_dir / "B" / "B_done.json"
        b = {}
        if b_done_path.exists():
            try:
                b = json.loads(b_done_path.read_text())
            except Exception:  # noqa: BLE001
                pass

        final_fill_rate = c.get("final_recall", {}).get("rate", 0.0)
        b_final_rate = b.get("final_recall", {}).get("rate", 0.0)
        stable_c = c.get("stable_perfect_epoch")
        stop_epoch = c.get("stop_epoch")
        stable_b = b.get("stable_perfect_epoch")
        leakage = c.get("placeholder_leakage_count", 0)
        qa_split = c.get("q_a_split_at_final", {})
        q_only_frac = qa_split.get("q_only", 0) / max(qa_split.get("total", 1), 1)

        # Discriminator-collapse check: trend test (2 probes above 0.05)
        epoch_log = c.get("epoch_log", [])
        q_only_trend = sum(
            1
            for e in epoch_log
            if e.get("q_a_split", {}).get("q_only", 0)
            / max(e.get("q_a_split", {}).get("total", 1), 1)
            > 0.02
        )

        passes = (
            b_final_rate >= 0.99
            and final_fill_rate >= 0.95
            and (stable_c is None or stable_c <= 22)
            and leakage == 0
            and q_only_frac <= 0.05
            and q_only_trend < 2
        )

        per_variant[v] = {
            "status": "pass" if passes else "fail",
            "fill_final_rate": final_fill_rate,
            "b_final_rate": b_final_rate,
            "stable_c": stable_c,
            "stop_epoch": stop_epoch,
            "stable_b": stable_b,
            "leakage": leakage,
            "q_only_frac": q_only_frac,
            "q_only_trend_count": q_only_trend,
        }
        if passes:
            passing.append(
                {
                    "variant": v,
                    "stable_c": stable_c,
                    "stop_epoch": stop_epoch,
                    "stable_b": stable_b,
                }
            )

    winner: str | None = None
    reason: str = ""
    override_threshold_applied: bool = False

    if len(passing) >= 1:
        # Sort by stable_c (ascending), tiebreak stable_b (ascending).
        passing_sorted = sorted(
            passing,
            key=lambda x: (
                x["stable_c"] if x["stable_c"] is not None else 99,
                x["stable_b"] if x["stable_b"] is not None else 99,
            ),
        )
        best = passing_sorted[0]
        best_variant = best["variant"]

        # Extended-run override rule: if there was an existing winner from the
        # original three-variant pass, a new variant (not in the original set)
        # must have stop_epoch <= NEW_VARIANT_OVERRIDE_THRESHOLD to displace it.
        original_variants = {"V1", "V2", "V3"}
        if existing_winner is not None and best_variant not in original_variants:
            best_stop = best.get("stop_epoch")
            if best_stop is not None and best_stop <= NEW_VARIANT_OVERRIDE_THRESHOLD:
                winner = best_variant
                reason = (
                    f"new variant {best_variant} overrides existing winner "
                    f"{existing_winner}: stop_epoch={best_stop} <= "
                    f"{NEW_VARIANT_OVERRIDE_THRESHOLD}"
                )
                override_threshold_applied = True
                logger.info(
                    "Extended-run override: %s replaces %s (stop_epoch=%s <= threshold=%d)",
                    best_variant,
                    existing_winner,
                    best_stop,
                    NEW_VARIANT_OVERRIDE_THRESHOLD,
                )
            else:
                # New variant does not beat the threshold; keep existing winner.
                winner = existing_winner
                reason = (
                    f"existing winner {existing_winner} retained: best new variant "
                    f"{best_variant} stop_epoch={best_stop} > {NEW_VARIANT_OVERRIDE_THRESHOLD}"
                )
                override_threshold_applied = True
                logger.info(
                    "Extended-run: %s does not beat threshold=%d (stop_epoch=%s) — retaining %s",
                    best_variant,
                    NEW_VARIANT_OVERRIDE_THRESHOLD,
                    best_stop,
                    existing_winner,
                )
        else:
            winner = best_variant
            reason = f"smallest stable_c={best['stable_c']}"
    else:
        if existing_winner is not None:
            # All new variants failed; preserve the existing winner.
            winner = existing_winner
            reason = f"all new variants failed; retaining existing winner {existing_winner}"
            logger.info(
                "All extended variants failed criteria — retaining existing winner %s",
                existing_winner,
            )
        else:
            reason = "all-variants-failed-or-ambiguous"

    decision = {
        "winner": winner,
        "reason": reason,
        "per_variant_summary": per_variant,
        "override_threshold_applied": override_threshold_applied,
        "override_threshold_value": NEW_VARIANT_OVERRIDE_THRESHOLD,
    }
    _safe_write_json(run_dir / "pre_decision.json", decision)
    logger.info("pre_decision.json written: winner=%s reason=%s", winner, reason)
    return winner


# ---------------------------------------------------------------------------
# Mode: pre (14a-pre)
# ---------------------------------------------------------------------------


def run_mode_pre(model, tokenizer, run_dir: Path, args: argparse.Namespace) -> None:
    """Run 14a-pre: V1/V2/V3 head-to-head at N=100; optionally V3_extended/V4/V5.

    When ``--reuse-phase-a-from PATH`` is set, Phase A is skipped for new
    variants (V3_extended/V4/V5) and the V3 Phase A adapter from PATH is
    loaded as the baseline.  V1/V2/V3 always run full Phase A when not yet
    complete.

    V3_extended Phase B loads V3's existing Phase B adapter from the source
    run dir and trains for ``args.scaffold_resume_epochs`` additional epochs
    (default 30) rather than creating a fresh adapter.

    After all variants complete, ``decide_pre_winner`` re-evaluates all
    variant results and updates ``pre_decision.json`` and ``results.json``.
    """
    n_keys = PRE_N_KEYS
    fill_count = PRE_FILL_KEYS
    fill_start = n_keys - fill_count  # 80
    fill_indices = list(range(fill_start, n_keys))

    qa_pool = load_qa_pool(n_keys + 20)  # +20 == SWAP_KEYS in load_qa_pool
    assert len(qa_pool) >= n_keys + 20, (
        f"load_qa_pool returned {len(qa_pool)} pairs; 14a-pre requires >= {n_keys + 20}"
    )

    # Resolve the Phase A reuse source dir (if any).
    reuse_phase_a_from: Path | None = None
    if getattr(args, "reuse_phase_a_from", None):
        reuse_phase_a_from = Path(args.reuse_phase_a_from)
        if not reuse_phase_a_from.is_dir():
            raise SystemExit(
                f"--reuse-phase-a-from: directory does not exist: {reuse_phase_a_from}"
            )
        logger.info("Phase A reuse source: %s", reuse_phase_a_from)

    # Original three variants always included; additional ones only when the
    # reuse source is provided (since they skip Phase A and rely on V3's adapter).
    original_variants = ("V1", "V2", "V3")
    extended_variants = (V3_EXTENDED, "V4", "V5")
    if reuse_phase_a_from is not None:
        variants_to_run = original_variants + extended_variants
    else:
        variants_to_run = original_variants

    for v_idx, variant in enumerate(variants_to_run):
        v_dir = run_dir / variant

        if (
            (
                (v_dir / "A" / "A_done.json").exists()
                or (v_dir / "A" / "phase_a_reused.json").exists()
            )
            and (v_dir / "B" / "B_done.json").exists()
            and (v_dir / "C" / "C_done.json").exists()
        ):
            logger.info("Variant %s: all phases complete — skipping", variant)
            continue

        if v_idx > 0:
            gpu_cooldown_between(f"variant {variant}")
        _check_pause(f"before variant {variant}", run_dir)

        logger.info("=" * 72)
        logger.info("14a-pre: variant %s (%d/%d)", variant, v_idx + 1, len(variants_to_run))
        logger.info("=" * 72)

        v_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------------------
        # Phase A
        # ----------------------------------------------------------------
        phase_a_done = (v_dir / "A" / "A_done.json").exists()
        phase_a_reused = (v_dir / "A" / "phase_a_reused.json").exists()

        if not phase_a_done and not phase_a_reused:
            if variant in extended_variants and reuse_phase_a_from is not None:
                # Reuse V3's Phase A adapter — no fresh training.
                if isinstance(model, PeftModel):
                    model = model.base_model.model
                model, _ = load_phase_a_from_existing(
                    model,
                    source_run_dir=reuse_phase_a_from,
                    source_variant=V3,
                    dest_variant_dir=v_dir,
                )
                # keyed_a / registry_a come from V3's Phase A artefacts.
                # simhash_registry.json is encrypted (via save_registry →
                # write_infra_bytes) when the daily age identity is loaded;
                # use load_registry which transparently handles both age and
                # plaintext envelopes.  keyed_pairs.json is plaintext.
                v3_a_keyed_path = reuse_phase_a_from / V3 / "A" / "keyed_pairs.json"
                v3_a_reg_path = reuse_phase_a_from / V3 / "A" / "simhash_registry.json"
                keyed_a = json.loads(v3_a_keyed_path.read_text())
                registry_a = {k: int(vv) for k, vv in load_registry(v3_a_reg_path).items()}
                adapter_name_a = f"episodic_{V3.lower()}"
            else:
                # Standard fresh Phase A.
                if isinstance(model, PeftModel):
                    model = model.base_model.model
                adapter_name_a = f"episodic_{variant.lower()}"
                model = create_adapter(model, _adapter_config(args.rank), adapter_name_a)
                switch_adapter(model, adapter_name_a)

                keyed_a = assign_keys(qa_pool[:n_keys], start_index=1)
                registry_a = build_registry(keyed_a)

                try:
                    model, metrics_a, probe_a, wall_a = run_phase_A_fresh(
                        model,
                        tokenizer,
                        keyed_a,
                        registry_a,
                        adapter_name=adapter_name_a,
                        args=args,
                        phase_dir=v_dir / "A",
                        run_name=f"test14-pre-A-{variant}",
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        _safe_write_json(
                            v_dir / "A" / "phase_A_oom.json",
                            {
                                "phase": "A",
                                "variant": variant,
                                "reason": "cuda_oom",
                                "error_text": str(e),
                                "timestamp": int(time.time()),
                            },
                        )
                        raise SystemExit(f"Phase A OOM in variant {variant} — halting")
                    raise

                _exit_if_paused_mid_phase(
                    probe_a, f"during A, variant {variant}", args.num_epochs, run_dir
                )

                final_a = _safe_probe(model, tokenizer, keyed_a, registry_a, adapter_name_a)
                _manifest_a = build_manifest_for(
                    model,
                    tokenizer,
                    adapter_name_a,
                    registry_path=None,
                    keyed_pairs_path=v_dir / "A" / "keyed_pairs.json",
                    key_count=len(keyed_a),
                )
                save_adapter(model, v_dir / "A" / "adapter", adapter_name_a, manifest=_manifest_a)
                _write_phase_done(
                    v_dir / "A",
                    "A_done.json",
                    keyed_a,
                    registry_a,
                    probe_a,
                    final_a,
                    extra={
                        "condition": f"A_fresh_{variant}",
                        "train_loss": metrics_a.get("train_loss"),
                        "wall_seconds": round(wall_a, 1),
                    },
                )
        else:
            logger.info("Variant %s Phase A: already done / reused", variant)
            if phase_a_reused:
                # keyed_a / registry_a come from V3's Phase A artefacts in source dir.
                v3_a_keyed_path = reuse_phase_a_from / V3 / "A" / "keyed_pairs.json"
                v3_a_reg_path = reuse_phase_a_from / V3 / "A" / "simhash_registry.json"
                keyed_a = json.loads(v3_a_keyed_path.read_text())
                registry_a = {k: int(vv) for k, vv in load_registry(v3_a_reg_path).items()}
                adapter_name_a = f"episodic_{V3.lower()}"
            else:
                keyed_a = json.loads((v_dir / "A" / "keyed_pairs.json").read_text())
                registry_a = {
                    k: int(vv)
                    for k, vv in load_registry(v_dir / "A" / "simhash_registry.json").items()
                }
                adapter_name_a = f"episodic_{variant.lower()}"

        _check_pause(f"after A, variant {variant}", run_dir)

        # ----------------------------------------------------------------
        # Phase B
        # ----------------------------------------------------------------
        if not (v_dir / "B" / "B_done.json").exists():
            if variant == V3_EXTENDED:
                # V3_extended Phase B: load V3's existing Phase B adapter and
                # continue training for scaffold_resume_epochs additional epochs.
                # Unwrap to base model first (never delete_adapter + create_adapter).
                if isinstance(model, PeftModel):
                    model = model.base_model.model

                v3_b_adapter_dir = reuse_phase_a_from / V3 / "B" / "adapter"
                v3_b_slot = resolve_adapter_slot(v3_b_adapter_dir, "journal", "")
                if v3_b_slot is None:
                    raise FileNotFoundError(
                        f"V3_extended Phase B: journal adapter not found under {v3_b_adapter_dir}"
                    )
                with _adapter_slot_for_load(v3_b_slot) as _load_path:
                    model = PeftModel.from_pretrained(
                        model, str(_load_path), adapter_name="journal"
                    )
                logger.info("V3_extended Phase B: loaded journal adapter from %s", v3_b_slot)
                switch_adapter(model, "journal")

                # Build V3 scaffold (uniform "pending"/"pending") — same as V3.
                scaffold_keyed = VARIANT_BUILDERS[V3_EXTENDED](n_keys, start_index=1)
                scaffold_registry = build_registry(scaffold_keyed)

                # V3_extended starts from V3's already-converged adapter (recall =
                # 1.0 at e10).  EarlyStopPolicy would fire at e10 immediately and
                # the deepening goal would be silently defeated.  Force-disable
                # early-stop on Phase B so all `--scaffold-resume-epochs` train
                # through.  This is hard-wired — there is no use case for
                # V3_extended Phase B with early-stop active.
                disable_early_stop_b = True
                model, metrics_b, probe_b, wall_b = run_phase_B_scaffold_resume(
                    model,
                    tokenizer,
                    scaffold_keyed,
                    scaffold_registry,
                    args=args,
                    phase_dir=v_dir / "B",
                    run_name=f"test14-pre-B-{variant}",
                    disable_early_stop=disable_early_stop_b,
                )

                _exit_if_paused_mid_phase(
                    probe_b,
                    f"during B (extended), variant {variant}",
                    getattr(args, "scaffold_resume_epochs", 30),
                    run_dir,
                )

                final_b = _safe_probe(
                    model, tokenizer, scaffold_keyed, scaffold_registry, "journal"
                )
                _manifest_b = build_manifest_for(
                    model,
                    tokenizer,
                    "journal",
                    registry_path=None,
                    keyed_pairs_path=v_dir / "B" / "keyed_pairs.json",
                    key_count=len(scaffold_keyed),
                )
                save_adapter(model, v_dir / "B" / "adapter", "journal", manifest=_manifest_b)
                # Done marker includes epochs_added and total_effective_epochs fields.
                _write_phase_done(
                    v_dir / "B",
                    "B_done.json",
                    scaffold_keyed,
                    scaffold_registry,
                    probe_b,
                    final_b,
                    extra={
                        "condition": f"B_scaffold_{variant}",
                        "train_loss": metrics_b.get("train_loss"),
                        "wall_seconds": round(wall_b, 1),
                        "epochs_added": getattr(args, "scaffold_resume_epochs", 30),
                        "total_effective_epochs": (
                            args.num_epochs + getattr(args, "scaffold_resume_epochs", 30)
                        ),
                    },
                )
            else:
                # Standard fresh scaffold build (V1/V2/V3/V4/V5).
                if isinstance(model, PeftModel):
                    model = model.base_model.model
                model = create_adapter(model, _adapter_config(args.rank), "journal")
                switch_adapter(model, "journal")

                scaffold_keyed = VARIANT_BUILDERS[variant](n_keys, start_index=1)
                scaffold_registry = build_registry(scaffold_keyed)

                model, metrics_b, probe_b, wall_b = run_phase_B_scaffold(
                    model,
                    tokenizer,
                    scaffold_keyed,
                    scaffold_registry,
                    args=args,
                    phase_dir=v_dir / "B",
                    run_name=f"test14-pre-B-{variant}",
                )

                _exit_if_paused_mid_phase(
                    probe_b, f"during B, variant {variant}", args.num_epochs, run_dir
                )

                final_b = _safe_probe(
                    model, tokenizer, scaffold_keyed, scaffold_registry, "journal"
                )
                _manifest_b = build_manifest_for(
                    model,
                    tokenizer,
                    "journal",
                    registry_path=None,
                    keyed_pairs_path=v_dir / "B" / "keyed_pairs.json",
                    key_count=len(scaffold_keyed),
                )
                save_adapter(model, v_dir / "B" / "adapter", "journal", manifest=_manifest_b)
                _write_phase_done(
                    v_dir / "B",
                    "B_done.json",
                    scaffold_keyed,
                    scaffold_registry,
                    probe_b,
                    final_b,
                    extra={
                        "condition": f"B_scaffold_{variant}",
                        "train_loss": metrics_b.get("train_loss"),
                        "wall_seconds": round(wall_b, 1),
                    },
                )
        else:
            logger.info("Variant %s Phase B: already done", variant)
            scaffold_keyed = json.loads((v_dir / "B" / "keyed_pairs.json").read_text())
            scaffold_registry = {
                k: int(vv) for k, vv in load_registry(v_dir / "B" / "simhash_registry.json").items()
            }

        _check_pause(f"after B, variant {variant}", run_dir)

        # ----------------------------------------------------------------
        # Phase C (fill)
        # ----------------------------------------------------------------
        if not (v_dir / "C" / "C_done.json").exists():
            # B1 fix: when Phase B is already done (B_done.json exists but C is
            # not yet done) the journal adapter was saved to disk but has NOT
            # been loaded in the current Python process.  switch_adapter() on a
            # bare base model raises because there is no PeftModel wrapper.
            # Reload from the saved B/adapter directory before entering Phase C.
            if not isinstance(model, PeftModel):
                b_journal_slot = resolve_adapter_slot(v_dir / "B" / "adapter", "journal", "")
                if b_journal_slot is None:
                    raise FileNotFoundError(
                        f"Variant {variant}: journal adapter not found under "
                        f"{v_dir / 'B' / 'adapter'}; cannot resume Phase C."
                    )
                with _adapter_slot_for_load(b_journal_slot) as _load_path:
                    model = PeftModel.from_pretrained(
                        model, str(_load_path), adapter_name="journal"
                    )
                logger.info(
                    "Variant %s Phase C resume: loaded journal adapter from %s",
                    variant,
                    b_journal_slot,
                )
            switch_adapter(model, "journal")

            fill_keyed = build_fill_keyed(scaffold_keyed, qa_pool, fill_indices)
            fill_registry = build_registry(fill_keyed)
            retention_keyed = [scaffold_keyed[i] for i in range(fill_start)]
            retention_registry = build_registry(retention_keyed)

            # Resume from checkpoint if partial C run exists.
            ckpt = _find_latest_checkpoint(v_dir / "C" / "adapter")

            model, metrics_c, probe_c, wall_c = run_phase_C_fill(
                model,
                tokenizer,
                scaffold_keyed,
                fill_keyed,
                fill_registry,
                retention_keyed,
                retention_registry,
                args=args,
                phase_dir=v_dir / "C",
                run_name=f"test14-pre-C-{variant}",
                resume_from_checkpoint=ckpt,
            )

            _exit_if_paused_mid_phase(
                probe_c, f"during C, variant {variant}", args.num_epochs, run_dir
            )

            final_c = _safe_probe(model, tokenizer, fill_keyed, fill_registry, "journal")
            leaks = _leakage_count(final_c["per_key"])
            _manifest_c = build_manifest_for(
                model,
                tokenizer,
                "journal",
                registry_path=None,
                keyed_pairs_path=v_dir / "C" / "keyed_pairs.json",
                key_count=len(fill_keyed),
            )
            save_adapter(model, v_dir / "C" / "adapter", "journal", manifest=_manifest_c)
            _write_phase_done(
                v_dir / "C",
                "C_done.json",
                fill_keyed,
                fill_registry,
                probe_c,
                final_c,
                extra={
                    "condition": f"C_fill_{variant}",
                    "train_loss": metrics_c.get("train_loss"),
                    "wall_seconds": round(wall_c, 1),
                    "placeholder_leakage_count": leaks,
                },
            )
        else:
            logger.info("Variant %s Phase C: already done", variant)

        _check_pause(f"after C, variant {variant}", run_dir)

    # All variants done — decide winner.  Pass existing_winner (from original
    # V1/V2/V3 pass) so the extended-run override rule can be applied.
    existing_results_path = run_dir / "results.json"
    existing_winner: str | None = None
    if existing_results_path.exists():
        try:
            existing_results = json.loads(existing_results_path.read_text())
            existing_winner = existing_results.get("winner")
        except Exception:  # noqa: BLE001
            pass

    winner = decide_pre_winner(run_dir, existing_winner=existing_winner)
    logger.info("14a-pre complete. winner=%s", winner)
    save_results(
        {"mode": "pre", "winner": winner, "run_dir": str(run_dir)},
        run_dir,
        filename="results.json",
    )


# ---------------------------------------------------------------------------
# Mode: scale (14a)
# ---------------------------------------------------------------------------


def run_mode_scale(
    model, tokenizer, run_dir: Path, args: argparse.Namespace, n_keys: int, variant: str
) -> None:
    """Run 14a (scale mode): single winning variant at n_keys scale."""
    fill_count = max(1, int(n_keys * FILL_FRACTION))
    fill_start = n_keys - fill_count
    fill_indices = list(range(fill_start, n_keys))

    qa_pool = load_qa_pool(n_keys + 50)
    assert len(qa_pool) >= n_keys + 50, (
        f"load_qa_pool({n_keys + 50}) returned {len(qa_pool)} pairs; 14a requires >= {n_keys + 50}"
    )

    logger.info("14a: variant=%s n_keys=%d fill_count=%d", variant, n_keys, fill_count)

    # --- Phase A ---
    if not (run_dir / "A" / "A_done.json").exists():
        if isinstance(model, PeftModel):
            model = model.base_model.model
        model = create_adapter(model, _adapter_config(args.rank), "episodic")
        switch_adapter(model, "episodic")

        keyed_a = assign_keys(qa_pool[:n_keys], start_index=1)
        registry_a = build_registry(keyed_a)

        try:
            model, metrics_a, probe_a, wall_a = run_phase_A_fresh(
                model,
                tokenizer,
                keyed_a,
                registry_a,
                adapter_name="episodic",
                args=args,
                phase_dir=run_dir / "A",
                run_name="test14-scale-A",
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                _safe_write_json(
                    run_dir / "A" / "phase_A_oom.json",
                    {
                        "phase": "A",
                        "reason": "cuda_oom",
                        "error_text": str(e),
                        "timestamp": int(time.time()),
                    },
                )
                raise SystemExit("Phase A OOM — halting; do not cascade into Phase B")
            raise

        _exit_if_paused_mid_phase(probe_a, "during A (scale)", args.num_epochs, run_dir)

        final_a = _safe_probe(model, tokenizer, keyed_a, registry_a, "episodic")
        _manifest_a = build_manifest_for(
            model,
            tokenizer,
            "episodic",
            registry_path=None,
            keyed_pairs_path=run_dir / "A" / "keyed_pairs.json",
            key_count=len(keyed_a),
        )
        save_adapter(model, run_dir / "A" / "adapter", "episodic", manifest=_manifest_a)
        _write_phase_done(
            run_dir / "A",
            "A_done.json",
            keyed_a,
            registry_a,
            probe_a,
            final_a,
            extra={
                "condition": "A_fresh_scale",
                "train_loss": metrics_a.get("train_loss"),
                "wall_seconds": round(wall_a, 1),
            },
        )
    else:
        logger.info("Phase A: already done")
        keyed_a = json.loads((run_dir / "A" / "keyed_pairs.json").read_text())
        registry_a = {
            k: int(v) for k, v in load_registry(run_dir / "A" / "simhash_registry.json").items()
        }

    _check_pause("after Phase A (scale)", run_dir)

    # --- Phase B ---
    if not (run_dir / "B" / "B_done.json").exists():
        if isinstance(model, PeftModel):
            model = model.base_model.model
        model = create_adapter(model, _adapter_config(args.rank), "journal")
        switch_adapter(model, "journal")

        scaffold_keyed = VARIANT_BUILDERS[variant](n_keys, start_index=1)
        scaffold_registry = build_registry(scaffold_keyed)

        model, metrics_b, probe_b, wall_b = run_phase_B_scaffold(
            model,
            tokenizer,
            scaffold_keyed,
            scaffold_registry,
            args=args,
            phase_dir=run_dir / "B",
            run_name="test14-scale-B",
        )

        _exit_if_paused_mid_phase(probe_b, "during B (scale)", args.num_epochs, run_dir)

        final_b = _safe_probe(model, tokenizer, scaffold_keyed, scaffold_registry, "journal")
        _manifest_b = build_manifest_for(
            model,
            tokenizer,
            "journal",
            registry_path=None,
            keyed_pairs_path=run_dir / "B" / "keyed_pairs.json",
            key_count=len(scaffold_keyed),
        )
        save_adapter(model, run_dir / "B" / "adapter", "journal", manifest=_manifest_b)
        _write_phase_done(
            run_dir / "B",
            "B_done.json",
            scaffold_keyed,
            scaffold_registry,
            probe_b,
            final_b,
            extra={
                "condition": "B_scaffold_scale",
                "train_loss": metrics_b.get("train_loss"),
                "wall_seconds": round(wall_b, 1),
            },
        )
    else:
        logger.info("Phase B: already done")
        scaffold_keyed = json.loads((run_dir / "B" / "keyed_pairs.json").read_text())
        scaffold_registry = {
            k: int(v) for k, v in load_registry(run_dir / "B" / "simhash_registry.json").items()
        }

    _check_pause("after Phase B (scale)", run_dir)

    # --- Phase C ---
    if not (run_dir / "C" / "C_done.json").exists():
        # B1 fix: when Phase B is already done (B_done.json exists but C is not
        # yet done) the journal adapter was saved to disk but has NOT been loaded
        # in the current Python process.  switch_adapter() on a bare base model
        # raises because there is no PeftModel wrapper.
        if not isinstance(model, PeftModel):
            b_journal_slot = resolve_adapter_slot(run_dir / "B" / "adapter", "journal", "")
            if b_journal_slot is None:
                raise FileNotFoundError(
                    f"Scale: journal adapter not found under {run_dir / 'B' / 'adapter'}; "
                    "cannot resume Phase C."
                )
            with _adapter_slot_for_load(b_journal_slot) as _load_path:
                model = PeftModel.from_pretrained(model, str(_load_path), adapter_name="journal")
            logger.info("Scale Phase C resume: loaded journal adapter from %s", b_journal_slot)
        switch_adapter(model, "journal")

        fill_keyed = build_fill_keyed(scaffold_keyed, qa_pool, fill_indices)
        fill_registry = build_registry(fill_keyed)
        # Retention for scale: unused keys below fill_start.
        retention_keyed = [scaffold_keyed[i] for i in range(fill_start)]
        retention_registry = build_registry(retention_keyed)

        ckpt = _find_latest_checkpoint(run_dir / "C" / "adapter")

        model, metrics_c, probe_c, wall_c = run_phase_C_fill(
            model,
            tokenizer,
            scaffold_keyed,
            fill_keyed,
            fill_registry,
            retention_keyed,
            retention_registry,
            args=args,
            phase_dir=run_dir / "C",
            run_name="test14-scale-C",
            resume_from_checkpoint=ckpt,
        )

        _exit_if_paused_mid_phase(probe_c, "during C (scale)", args.num_epochs, run_dir)

        final_c = _safe_probe(model, tokenizer, fill_keyed, fill_registry, "journal")
        leaks = _leakage_count(final_c["per_key"])
        _manifest_c = build_manifest_for(
            model,
            tokenizer,
            "journal",
            registry_path=None,
            keyed_pairs_path=run_dir / "C" / "keyed_pairs.json",
            key_count=len(fill_keyed),
        )
        save_adapter(model, run_dir / "C" / "adapter", "journal", manifest=_manifest_c)
        _write_phase_done(
            run_dir / "C",
            "C_done.json",
            fill_keyed,
            fill_registry,
            probe_c,
            final_c,
            extra={
                "condition": "C_fill_scale",
                "train_loss": metrics_c.get("train_loss"),
                "wall_seconds": round(wall_c, 1),
                "placeholder_leakage_count": leaks,
            },
        )
    else:
        logger.info("Phase C: already done")

    save_results(
        {"mode": "scale", "variant": variant, "n_keys": n_keys, "run_dir": str(run_dir)},
        run_dir,
        filename="results.json",
    )
    logger.info("14a scale complete.")


# ---------------------------------------------------------------------------
# Mode: multiround (14b)
# ---------------------------------------------------------------------------


def run_mode_multiround(
    model, tokenizer, run_dir: Path, args: argparse.Namespace, scale_run: Path
) -> None:
    """Run 14b: P0 verify + 3 rounds with RP1/RP2/RP3 retention probes."""
    # --- P0: Load & Verify ---
    scale_cfg = json.loads((scale_run / "run_config.json").read_text())
    n_keys = int(scale_cfg["n_keys"])
    logger.info("14b P0: inherited n_keys=%d from %s", n_keys, scale_run)

    qa_pool = load_qa_pool(n_keys + 50)
    # §14.6 mandatory assertion.
    assert len(qa_pool) >= n_keys + 40, (
        f"load_qa_pool({n_keys + 50}) returned {len(qa_pool)} pairs; "
        f"14b requires >= {n_keys + 40} for swap-set indices {n_keys}..{n_keys + 39}. "
        f"Re-check PerLTQA dedup behavior."
    )

    # Build all-key keyed pairs from PerLTQA (self-contained: not loaded from 14a saved JSON).
    all_keyed = assign_keys(qa_pool[:n_keys], start_index=1)
    all_registry = build_registry(all_keyed)

    # Load 14a's filled adapter (C/adapter/journal).
    c_adapter_dir = scale_run / "C" / "adapter"
    journal_slot = resolve_adapter_slot(c_adapter_dir, "journal", "")
    if journal_slot is None:
        raise FileNotFoundError(
            f"Could not find journal adapter under {c_adapter_dir}. "
            "Ensure 14a Phase C completed cleanly."
        )

    if not (run_dir / "P0_done.json").exists():
        if isinstance(model, PeftModel):
            model = model.base_model.model
        with _adapter_slot_for_load(journal_slot) as _load_path:
            model = PeftModel.from_pretrained(model, str(_load_path), adapter_name="journal")
        switch_adapter(model, "journal")

        p0_probe = _safe_probe(model, tokenizer, all_keyed, all_registry, "journal")
        _safe_write_json(
            run_dir / "P0_done.json",
            {
                "phase": "P0",
                "n_keys": n_keys,
                "baseline_recall": {
                    "exact_count": p0_probe["exact_count"],
                    "total": p0_probe["total"],
                    "rate": p0_probe["rate"],
                    "mean_confidence": p0_probe["mean_confidence"],
                },
            },
        )
        logger.info(
            "P0: baseline %d/%d (%.3f)",
            p0_probe["exact_count"],
            p0_probe["total"],
            p0_probe["rate"],
        )
    else:
        logger.info("P0: already done")
        if not isinstance(model, PeftModel):
            with _adapter_slot_for_load(journal_slot) as _load_path:
                model = PeftModel.from_pretrained(model, str(_load_path), adapter_name="journal")
        switch_adapter(model, "journal")

    # --- Rounds plan (frozen at first launch for deterministic resume) ---
    rounds_plan_path = run_dir / "rounds_plan.json"
    if not rounds_plan_path.exists():
        swap_r2 = list(range(0, 40))  # slots 0..39
        swap_r3 = list(range(40, 80))  # slots 40..79
        rounds_plan = {
            "round_1": {"swap_indices": list(range(n_keys))},  # P1: all keys
            "round_2": {"swap_indices": swap_r2},
            "round_3": {"swap_indices": swap_r3},
            "latest_answer_by_slot": {},
            "inherited_n_keys": n_keys,
        }
        _safe_write_json(rounds_plan_path, rounds_plan)
    else:
        rounds_plan = json.loads(rounds_plan_path.read_text())

    latest_answer_by_slot: dict = rounds_plan.get("latest_answer_by_slot", {})

    round_results_list: list[dict] = []

    for round_idx, round_label in enumerate(("P1", "P2", "P3")):
        round_dir = run_dir / round_label
        round_done_path = round_dir / f"{round_label}_done.json"

        if round_done_path.exists():
            logger.info("Round %s: already done — loading", round_label)
            round_result_saved = json.loads(round_done_path.read_text())
            round_results_list.append(round_result_saved)
            # Update latest_answer_by_slot from saved state.
            plan_key = f"round_{round_idx + 1}"
            for slot_idx in rounds_plan[plan_key]["swap_indices"]:
                latest_answer_by_slot[str(slot_idx)] = round_label
            # B2 fix: load this round's final adapter so subsequent rounds
            # carry forward the correct weight state.  Without this, when
            # resuming past a completed round the model still holds a prior
            # round's (or 14a's) adapter, silently corrupting the
            # corruption-residual signal for all later rounds.
            touchup_skipped = round_result_saved.get("touchup_skipped", True)
            if touchup_skipped:
                carry_dir = round_dir / "adapter"
            else:
                carry_dir = round_dir / "touchup_adapter"
            carry_slot = resolve_adapter_slot(carry_dir, "journal", "")
            if carry_slot is not None:
                if isinstance(model, PeftModel):
                    model = model.base_model.model
                with _adapter_slot_for_load(carry_slot) as _load_path:
                    model = PeftModel.from_pretrained(
                        model, str(_load_path), adapter_name="journal"
                    )
                logger.info(
                    "Round %s carry-forward: loaded adapter from %s", round_label, carry_slot
                )
            else:
                logger.warning(
                    "Round %s carry-forward: adapter not found under %s — "
                    "subsequent rounds may use stale weights",
                    round_label,
                    carry_dir,
                )
            continue

        _check_pause(f"before round {round_label}", run_dir)
        round_dir.mkdir(parents=True, exist_ok=True)

        swap_indices = rounds_plan[f"round_{round_idx + 1}"]["swap_indices"]

        # Build swap keyed pairs.
        if round_label == "P1":
            # P1: re-train all n_keys with real Q+A (from qa_pool).
            swap_keyed = assign_keys(qa_pool[:n_keys], start_index=1)
        else:
            # P2/P3: answer-swap on 40 disjoint slots.
            swap_keyed = []
            for k, slot_idx in enumerate(swap_indices):
                orig = all_keyed[slot_idx]
                new_answer = qa_pool[n_keys + k]["answer"]
                swap_keyed.append(
                    {"key": orig["key"], "question": orig["question"], "answer": new_answer}
                )
            # Update ground truth for these slots.
            for k, slot_idx in enumerate(swap_indices):
                latest_answer_by_slot[str(slot_idx)] = round_label
            # Persist updated plan.
            rounds_plan["latest_answer_by_slot"] = latest_answer_by_slot
            _safe_write_json(rounds_plan_path, rounds_plan)

        swap_registry = build_registry(swap_keyed)
        _safe_write_json(round_dir / "swap_keyed.json", swap_keyed)
        save_registry(swap_registry, round_dir / "swap_simhash_registry.json")

        # Build unchanged-set (keys NOT in swap for P2/P3; for P1 = empty since all are swapped).
        if round_label == "P1":
            unchanged_keyed = []
            unchanged_registry = {}
        else:
            swap_set = set(swap_indices)
            # Latest answers for unchanged slots.
            unchanged_keyed = []
            for slot_idx, kp in enumerate(all_keyed):
                if slot_idx not in swap_set:
                    latest_round = latest_answer_by_slot.get(str(slot_idx))
                    if latest_round in ("P2",):
                        # Slot was swapped in P2 — use that answer.
                        k_in_r2 = rounds_plan["round_2"]["swap_indices"].index(slot_idx)
                        answer = qa_pool[n_keys + k_in_r2]["answer"]
                    elif latest_round in ("P3",):
                        k_in_r3 = rounds_plan["round_3"]["swap_indices"].index(slot_idx)
                        answer = qa_pool[n_keys + k_in_r3]["answer"]
                    else:
                        answer = kp["answer"]
                    unchanged_keyed.append(
                        {"key": kp["key"], "question": kp["question"], "answer": answer}
                    )
            unchanged_registry = build_registry(unchanged_keyed) if unchanged_keyed else {}

        _safe_write_json(round_dir / "unchanged_keyed.json", unchanged_keyed)
        if unchanged_registry:
            save_registry(unchanged_registry, round_dir / "unchanged_simhash_registry.json")

        # RP1 (round-start probe).
        rp1_path = round_dir / "retention_probe_rp1.json"
        if rp1_path.exists():
            rp1 = json.loads(rp1_path.read_text())
        else:
            if unchanged_keyed:
                rp1 = run_round_retention_probe(
                    model,
                    tokenizer,
                    unchanged_keyed,
                    unchanged_registry,
                    "journal",
                    rp1_path,
                    "RP1",
                )
            else:
                rp1 = {
                    "exact_count": 0,
                    "total": 0,
                    "rate": 1.0,
                    "mean_confidence": 1.0,
                    "per_key": [],
                }
                _safe_write_json(rp1_path, rp1)

        _check_pause(f"after RP1, round {round_label}", run_dir)

        # Fill phase.
        fill_state = EpochProbeState()
        ckpt = _find_latest_checkpoint(round_dir / "adapter")

        early_state = _EarlyStopState()
        callback = RecallEarlyStopCallback(
            model=model,
            tokenizer=tokenizer,
            target_keyed=swap_keyed,
            target_registry=swap_registry,
            adapter_name="journal",
            policy=ANALYSIS_POLICY,
            state_out=early_state,
            progress_path=round_dir / "progress.json",
            epoch_log_path=round_dir / "epoch_log.json",
            first_perfect_log_path=round_dir / "first_perfect_log.json",
            phase_name=f"fill_{round_label}",
            num_epochs=args.num_epochs,
            pause_file=PAUSE_FILE,
            retention_keyed=unchanged_keyed if unchanged_keyed else None,
            retention_registry=unchanged_registry if unchanged_keyed else None,
        )

        examples = format_indexed_training(swap_keyed, tokenizer, max_length=1024)
        dataset = IndexedDataset(examples)
        adapter_cfg = _adapter_config(args.rank)
        training_cfg = _training_config(args.num_epochs, rank=args.rank)

        t0 = time.time()
        fill_metrics = train_adapter(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            adapter_name="journal",
            training_config=training_cfg,
            adapter_config=adapter_cfg,
            output_dir=round_dir / "adapter",
            run_name=f"test14-14b-{round_label}",
            callbacks_extra=[callback],
            resume_from_checkpoint=str(ckpt) if ckpt else None,
        )
        fill_wall = time.time() - t0

        fill_state.first_perfect_epoch = early_state.first_perfect_epoch
        fill_state.stable_perfect_epoch = early_state.stable_perfect_epoch
        fill_state.stop_epoch = early_state.stop_epoch
        fill_state.epoch_log = early_state.epoch_log

        _exit_if_paused_mid_phase(
            fill_state, f"during fill, round {round_label}", args.num_epochs, run_dir
        )

        # Save post-fill adapter.
        _manifest_fill = build_manifest_for(
            model,
            tokenizer,
            "journal",
            registry_path=None,
            keyed_pairs_path=round_dir / "swap_keyed.json",
            key_count=len(swap_keyed),
        )
        save_adapter(model, round_dir / "adapter", "journal", manifest=_manifest_fill)

        _check_pause(f"after fill, round {round_label}", run_dir)

        # RP2 (post-fill, pre-touchup).
        rp2_path = round_dir / "retention_probe_rp2.json"
        if rp2_path.exists():
            rp2 = json.loads(rp2_path.read_text())
        else:
            if unchanged_keyed:
                rp2 = run_round_retention_probe(
                    model,
                    tokenizer,
                    unchanged_keyed,
                    unchanged_registry,
                    "journal",
                    rp2_path,
                    "RP2",
                )
            else:
                rp2 = {
                    "exact_count": 0,
                    "total": 0,
                    "rate": 1.0,
                    "mean_confidence": 1.0,
                    "per_key": [],
                }
                _safe_write_json(rp2_path, rp2)

        _check_pause(f"after RP2, round {round_label}", run_dir)

        # Touch-up.
        failing_keys = [r["key"] for r in rp2.get("per_key", []) if not r.get("exact_match")]
        failing_keyed_list = []
        if failing_keys and unchanged_keyed:
            kp_map = {kp["key"]: kp for kp in unchanged_keyed}
            failing_keyed_list = [kp_map[k] for k in failing_keys if k in kp_map]

        touchup_meta = run_touchup_step(
            model,
            tokenizer,
            failing_keyed_list,
            "journal",
            round_dir / "touchup_adapter",
        )

        # Save touchup adapter (even if skipped — to mark the slot exists).
        if touchup_meta.get("touchup_triggered"):
            _manifest_tu = build_manifest_for(
                model,
                tokenizer,
                "journal",
                registry_path=None,
                keyed_pairs_path=round_dir / "unchanged_keyed.json",
                key_count=len(unchanged_keyed),
            )
            save_adapter(model, round_dir / "touchup_adapter", "journal", manifest=_manifest_tu)

        _check_pause(f"after touchup, round {round_label}", run_dir)

        # RP3 (post-touchup).
        rp3_path = round_dir / "retention_probe_rp3.json"
        if rp3_path.exists():
            rp3 = json.loads(rp3_path.read_text())
        else:
            if unchanged_keyed:
                rp3 = run_round_retention_probe(
                    model,
                    tokenizer,
                    unchanged_keyed,
                    unchanged_registry,
                    "journal",
                    rp3_path,
                    "RP3",
                )
            else:
                rp3 = {
                    "exact_count": 0,
                    "total": 0,
                    "rate": 1.0,
                    "mean_confidence": 1.0,
                    "per_key": [],
                }
                _safe_write_json(rp3_path, rp3)

        # T13b reference stop epochs: P1 reference=11 (fill), P2/P3 reference=18 (swap).
        t13b_ref = 11 if round_label == "P1" else 18
        round_result = compute_round_metrics(
            rp1,
            rp2,
            rp3,
            fill_state,
            touchup_meta,
            round_index=round_label,
            max_epochs=args.num_epochs,
            t13b_reference_stop_epoch=t13b_ref,
        )
        round_result["fill_wall_seconds"] = round(fill_wall, 1)
        round_result["fill_train_loss"] = fill_metrics.get("train_loss")

        _safe_write_json(round_done_path, round_result)
        round_results_list.append(round_result)
        logger.info(
            "Round %s done: corruption_residual=%.4f alignment_delta=%.4f",
            round_label,
            round_result.get("retention_corruption_residual", -1),
            round_result.get("retention_alignment_delta", -1),
        )

        _check_pause(f"after round {round_label}", run_dir)

    # --- Cumulative retention probe (end of P3) ---
    cumul_path = run_dir / "cumulative_retention.json"
    if not cumul_path.exists():
        # Build latest-answer ground truth.
        cumul_keyed = []
        for slot_idx, kp in enumerate(all_keyed):
            lr = latest_answer_by_slot.get(str(slot_idx))
            if lr == "P2":
                k_in_r2 = rounds_plan["round_2"]["swap_indices"].index(slot_idx)
                answer = qa_pool[n_keys + k_in_r2]["answer"]
            elif lr == "P3":
                k_in_r3 = rounds_plan["round_3"]["swap_indices"].index(slot_idx)
                answer = qa_pool[n_keys + k_in_r3]["answer"]
            else:
                answer = kp["answer"]
            cumul_keyed.append({"key": kp["key"], "question": kp["question"], "answer": answer})
        cumul_registry = build_registry(cumul_keyed)
        cumulative = run_round_retention_probe(
            model,
            tokenizer,
            cumul_keyed,
            cumul_registry,
            "journal",
            cumul_path,
            "cumulative",
        )
    else:
        cumulative = json.loads(cumul_path.read_text())

    # Trend + pass/fail.
    resids = [r.get("retention_corruption_residual", 1.0) for r in round_results_list]
    pre_rates = [r.get("retention_pre_round", {}).get("rate", 0.0) for r in round_results_list]
    pretu_rates = [r.get("retention_pre_touchup", {}).get("rate", 0.0) for r in round_results_list]
    post_rates = [r.get("retention_post_touchup", {}).get("rate", 0.0) for r in round_results_list]
    deltas = [r.get("retention_alignment_delta", 0.0) for r in round_results_list]

    pass_fail = evaluate_14b_pass_fail(round_results_list, cumulative, n_keys)

    trend = {
        "retention_pre_round_rates": pre_rates,
        "retention_pre_touchup_rates": pretu_rates,
        "retention_post_touchup_rates": post_rates,
        "retention_alignment_deltas": deltas,
        "retention_corruption_residuals": resids,
        "corruption_residual_grew_monotonically": (
            len(resids) == 3 and resids[0] < resids[1] < resids[2]
        ),
        "alignment_delta_grew_monotonically": (
            len(deltas) == 3 and deltas[0] < deltas[1] < deltas[2]
        ),
        "max_corruption_residual": max(resids) if resids else None,
        "min_post_touchup_retention": min(post_rates) if post_rates else None,
        "alignment_delta_concern_trigger": pass_fail.get("alignment_delta_concern_trigger", False),
    }

    save_results(
        {
            "mode": "multiround",
            "n_keys": n_keys,
            "rounds": round_results_list,
            "trend": trend,
            "cumulative_retention": {
                "exact_count": cumulative.get("exact_count"),
                "total": n_keys,
                "rate": cumulative.get("rate"),
            },
            "pass_fail": pass_fail,
        },
        run_dir,
        filename="results.json",
    )
    logger.info("14b multiround complete. Verdict: %s", pass_fail["verdict"])


# ---------------------------------------------------------------------------
# Smoke mode
# ---------------------------------------------------------------------------


def run_smoke(model, tokenizer, run_dir: Path, args: argparse.Namespace) -> None:
    """Smoke test: N=10, 3 epochs, V1 only (§14.5 specification).

    Verifies in order:
    1. Model load and adapter creation (fresh + add_adapter paths).
    2. build_v1_scaffold(10) produces 10 entries with correct keys.
    3. RecallEarlyStopCallback instantiates with ANALYSIS_POLICY and
       on_epoch_end fires.
    4. per_field_split_counts computed and serialized.
    5. <phase>_done.json markers written at phase exit.
    6. Pause-file handling honored.
    7. Artifacts land under smoke_v1/ dir.
    8. training-control.sh dispatch (tested via run_config.json presence).
    """
    from experiments.utils.scaffold import build_v1_scaffold

    n_keys = 10
    fill_count = 2
    fill_start = n_keys - fill_count
    fill_indices = list(range(fill_start, n_keys))

    logger.info("SMOKE: N=%d, epochs=%d, variant=V1", n_keys, args.num_epochs)

    smoke_dir = run_dir / "smoke_v1"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke_cfg = {
        "mode": "smoke",
        "variant": "V1",
        "n_keys": n_keys,
        "num_epochs": args.num_epochs,
        "smoke": True,
        "created_at": int(time.time()),
    }
    # F7 fix: write run_config.json at the canonical top-level <ts>/ location
    # so find_latest_run_dir can detect it (and skip it via the smoke=True field).
    # Also write under smoke_v1/ for backward compatibility with any direct reads.
    _safe_write_json(run_dir / "run_config.json", smoke_cfg)
    _safe_write_json(smoke_dir / "run_config.json", smoke_cfg)

    qa_pool = load_qa_pool(n_keys + 20)

    # Verify scaffold shape.
    scaffold = build_v1_scaffold(n_keys)
    assert len(scaffold) == n_keys, f"scaffold length {len(scaffold)} != {n_keys}"
    for i, entry in enumerate(scaffold):
        assert entry["key"] == f"graph{i + 1}", f"key mismatch at slot {i}"
    keys_seen: set[str] = set()
    for entry in scaffold:
        assert entry["key"] not in keys_seen, f"duplicate key {entry['key']}"
        keys_seen.add(entry["key"])
    logger.info("SMOKE: scaffold shape OK")

    # Phase A.
    if not (smoke_dir / "A" / "A_done.json").exists():
        if isinstance(model, PeftModel):
            model = model.base_model.model
        model = create_adapter(model, _adapter_config(args.rank), "episodic")
        switch_adapter(model, "episodic")

        keyed_a = assign_keys(qa_pool[:n_keys], start_index=1)
        registry_a = build_registry(keyed_a)

        try:
            model, metrics_a, probe_a, wall_a = run_phase_A_fresh(
                model,
                tokenizer,
                keyed_a,
                registry_a,
                adapter_name="episodic",
                args=args,
                phase_dir=smoke_dir / "A",
                run_name="test14-smoke-A",
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                _safe_write_json(
                    smoke_dir / "A" / "phase_A_oom.json",
                    {"phase": "A", "reason": "cuda_oom", "error_text": str(e)},
                )
                raise SystemExit("SMOKE Phase A OOM")
            raise

        _exit_if_paused_mid_phase(probe_a, "during A (smoke)", args.num_epochs, smoke_dir)

        final_a = _safe_probe(model, tokenizer, keyed_a, registry_a, "episodic")
        save_adapter(model, smoke_dir / "A" / "adapter", "episodic")
        _write_phase_done(
            smoke_dir / "A",
            "A_done.json",
            keyed_a,
            registry_a,
            probe_a,
            final_a,
            extra={"condition": "smoke_A"},
        )

    # Phase B (scaffold build).
    if not (smoke_dir / "B" / "B_done.json").exists():
        if isinstance(model, PeftModel):
            model = model.base_model.model
        model = create_adapter(model, _adapter_config(args.rank), "journal")
        switch_adapter(model, "journal")

        scaffold_keyed = build_v1_scaffold(n_keys)
        scaffold_registry = build_registry(scaffold_keyed)

        model, metrics_b, probe_b, wall_b = run_phase_B_scaffold(
            model,
            tokenizer,
            scaffold_keyed,
            scaffold_registry,
            args=args,
            phase_dir=smoke_dir / "B",
            run_name="test14-smoke-B",
        )

        _exit_if_paused_mid_phase(probe_b, "during B (smoke)", args.num_epochs, smoke_dir)

        final_b = _safe_probe(model, tokenizer, scaffold_keyed, scaffold_registry, "journal")
        save_adapter(model, smoke_dir / "B" / "adapter", "journal")
        _write_phase_done(
            smoke_dir / "B",
            "B_done.json",
            scaffold_keyed,
            scaffold_registry,
            probe_b,
            final_b,
            extra={"condition": "smoke_B"},
        )
        logger.info("SMOKE: B_done.json written")

    # Phase C (fill).
    if not (smoke_dir / "C" / "C_done.json").exists():
        scaffold_keyed = json.loads((smoke_dir / "B" / "keyed_pairs.json").read_text())
        scaffold_registry = {
            k: int(v) for k, v in load_registry(smoke_dir / "B" / "simhash_registry.json").items()
        }

        switch_adapter(model, "journal")

        fill_keyed = build_fill_keyed(scaffold_keyed, qa_pool, fill_indices)
        fill_registry = build_registry(fill_keyed)
        retention_keyed = [scaffold_keyed[i] for i in range(fill_start)]
        retention_registry = build_registry(retention_keyed)

        model, metrics_c, probe_c, wall_c = run_phase_C_fill(
            model,
            tokenizer,
            scaffold_keyed,
            fill_keyed,
            fill_registry,
            retention_keyed,
            retention_registry,
            args=args,
            phase_dir=smoke_dir / "C",
            run_name="test14-smoke-C",
        )

        _exit_if_paused_mid_phase(probe_c, "during C (smoke)", args.num_epochs, smoke_dir)

        final_c = _safe_probe(model, tokenizer, fill_keyed, fill_registry, "journal")
        save_adapter(model, smoke_dir / "C" / "adapter", "journal")
        leaks = _leakage_count(final_c["per_key"])
        _write_phase_done(
            smoke_dir / "C",
            "C_done.json",
            fill_keyed,
            fill_registry,
            probe_c,
            final_c,
            extra={"condition": "smoke_C", "placeholder_leakage_count": leaks},
        )
        logger.info("SMOKE: C_done.json written")

    # F2: Resume-extension drill (smoke spec item 9).
    # Verify load_or_write_run_config handles n_keys extension correctly:
    # simulate a scale run_config.json with n_keys=10, then re-read it with
    # n_keys=15 and confirm the config is updated and the pool tops up.
    if not (smoke_dir / "smoke_extension_done.json").exists():
        ext_dir = smoke_dir / "ext_drill"
        ext_dir.mkdir(parents=True, exist_ok=True)
        _safe_write_json(
            ext_dir / "run_config.json",
            {
                "mode": "scale",
                "variant": "V1",
                "n_keys": 10,
                "fill_keys": 2,
                "num_epochs": 3,
                "rank": args.rank,
                "lr": DEFAULT_LR,
                "seed": 42,
                "scale_run": "",
                "smoke": True,
                "created_at": int(time.time()),
            },
        )
        # Simulate --resume --n_keys=15: load_or_write_run_config should update.
        import types

        ext_args = types.SimpleNamespace(
            mode="scale",
            variant="V1",
            n_keys=15,
            num_epochs=3,
            rank=args.rank,
            smoke=True,
        )
        ext_cfg = load_or_write_run_config(ext_dir, ext_args)
        assert ext_cfg["n_keys"] == 15, (
            f"Extension drill: expected n_keys=15, got {ext_cfg['n_keys']}"
        )
        # Verify qa_pool tops up (load_qa_pool can supply >= 15 + buffer).
        ext_pool = load_qa_pool(ext_cfg["n_keys"] + 20)
        assert len(ext_pool) >= ext_cfg["n_keys"], (
            f"Extension drill: pool only has {len(ext_pool)} items for n_keys={ext_cfg['n_keys']}"
        )
        _safe_write_json(
            smoke_dir / "smoke_extension_done.json",
            {
                "drill": "resume_extension",
                "initial_n_keys": 10,
                "extended_n_keys": 15,
                "pool_size": len(ext_pool),
                "status": "pass",
                "timestamp": int(time.time()),
            },
        )
        logger.info("SMOKE extension drill: OK (n_keys 10→15, pool=%d items)", len(ext_pool))

    # Pause-file handling check.
    if PAUSE_FILE.exists():
        logger.info("SMOKE: pause file present — writing paused.json and exiting cleanly")
        write_paused_marker(smoke_dir, "C")
        return

    save_results({"mode": "smoke", "status": "complete"}, smoke_dir, filename="results.json")
    logger.info("SMOKE complete. All checks passed.")


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for Test 14."""
    args = parse_args()

    # Smoke mode overrides everything.
    if args.smoke:
        args.mode = "pre"  # use pre output base for smoke dir
        args.num_epochs = 3
        args.n_keys = 10

    # F11 fix: enforce minimum 30 epochs for production modes per CLAUDE.md
    # "Minimum 30 epochs for indexed keys" rule.  Smoke mode is exempt (it sets
    # num_epochs=3 above for infrastructure-only verification).
    if not args.smoke and args.num_epochs < 30:
        raise SystemExit(
            f"--num-epochs={args.num_epochs} is below the minimum of 30 required for "
            "production indexed-key training (CLAUDE.md: 'Minimum 30 epochs for indexed "
            "keys').  Use --smoke for infrastructure checks with fewer epochs."
        )

    # Pre-flight disk space check (§14.7).
    output_base = OUTPUT_BASES.get(args.mode, OUTPUT_BASES["pre"])
    output_base.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_base).free
    if free_bytes <= DISK_HEADROOM_BYTES:
        raise SystemExit(
            f"Insufficient disk space: {free_bytes / 1024**3:.1f} GB free in {output_base}; "
            f"need > {DISK_HEADROOM_BYTES / 1024**3:.0f} GB."
        )

    # Resolve output dir.
    if args.smoke:
        run_dir = model_output_dir(output_base, args.model)
    elif args.resume:
        latest = find_latest_run_dir(args.mode, args.model)
        if latest is None:
            logger.warning(
                "--resume: no prior run for mode=%s model=%s — starting fresh",
                args.mode,
                args.model,
            )
            run_dir = model_output_dir(output_base, args.model)
        else:
            run_dir = latest
            logger.info("Resuming from %s", run_dir)
    else:
        run_dir = model_output_dir(output_base, args.model)

    run_dir.mkdir(parents=True, exist_ok=True)

    if paused_requested():
        logger.warning("Pause file present at launch — clearing before starting work")
        try:
            PAUSE_FILE.unlink()
        except OSError:
            pass
    clear_paused_marker(run_dir)

    # Load or write run_config (for non-smoke modes).
    if not args.smoke:
        if args.mode == "pre":
            # 14a-pre: n_keys hard-capped at 100.
            args._active_variant = "auto"
            cfg = load_or_write_run_config(run_dir, args)
        elif args.mode == "scale":
            if args.n_keys is None:
                args.n_keys = 500
            cfg = load_or_write_run_config(run_dir, args)
            args.n_keys = cfg["n_keys"]
            # Validate variant.
            if args.variant == "auto":
                # Check pre_decision.json from the latest pre run.
                pre_latest = find_latest_run_dir("pre", args.model)
                if pre_latest is not None:
                    pre_decision_path = pre_latest / "pre_decision.json"
                    if pre_decision_path.exists():
                        pd = json.loads(pre_decision_path.read_text())
                        args.variant = pd.get("winner") or "V1"
                        logger.info("Using winner variant from pre_decision.json: %s", args.variant)
                    else:
                        args.variant = cfg.get("variant", "V1")
                else:
                    args.variant = cfg.get("variant", "V1")
            cfg["variant"] = args.variant
        elif args.mode == "multiround":
            cfg = load_or_write_run_config(run_dir, args)
        args.num_epochs = cfg.get("num_epochs", args.num_epochs)
        args.rank = cfg.get("rank", args.rank)
    else:
        cfg = {}

    model_config = BENCHMARK_MODELS[args.model]

    with acquire_gpu(interactive=True):
        model, tokenizer = load_model_and_config(model_config)

        if args.smoke:
            run_smoke(model, tokenizer, run_dir, args)
        elif args.mode == "pre":
            run_mode_pre(model, tokenizer, run_dir, args)
        elif args.mode == "scale":
            n_keys = args.n_keys or 500
            variant = args.variant if args.variant != "auto" else "V1"
            run_mode_scale(model, tokenizer, run_dir, args, n_keys=n_keys, variant=variant)
        elif args.mode == "multiround":
            # Resolve scale_run.
            if args.scale_run:
                scale_run = Path(args.scale_run)
            else:
                scale_run = find_latest_run_dir("scale", args.model)
                if scale_run is None:
                    raise SystemExit(
                        "--mode=multiround requires a completed 14a scale run. "
                        "Pass --scale-run=<path> or run --mode=scale first."
                    )
            run_mode_multiround(model, tokenizer, run_dir, args, scale_run=scale_run)
        else:
            raise SystemExit(f"Unknown mode: {args.mode}")

        unload_model(model, tokenizer)

    logger.info("Test 14 complete.")


if __name__ == "__main__":
    main()
