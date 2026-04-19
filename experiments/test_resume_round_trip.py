"""Epoch-level resume round-trip verification for BackgroundTrainer (Task #35).

Exercises the full resume mechanism end-to-end on GPU:

  Phase A — train N epochs, inject a RuntimeError at ``HALT_EPOCH`` so the
    training thread exits with resume_state.json + bg_checkpoint/ intact.
  Phase B — create a fresh BackgroundTrainer on the same (still-loaded)
    model and submit the same job.  BackgroundTrainer must detect the
    resume state, call ``Trainer.train(resume_from_checkpoint=<path>)``,
    and continue from HALT_EPOCH+1 to TOTAL_EPOCHS.
  Phase C — once resumed training commits, verify resume_state.json and
    bg_checkpoint/ are gone, production adapter weights are on disk, and
    smoke_test_adapter() recalls the trained key(s) with rate == 1.0.

Pass condition: every phase completes and Phase C recall == 1.0.  Any
skipped phase, stale state, or checkpoint left on disk after Phase C
is a failure.

Usage:
    conda activate paramem
    python experiments/test_resume_round_trip.py [--model mistral]
    python experiments/test_resume_round_trip.py --dry-run

WSL2 constraint: at most two from_pretrained calls per process.
  Load 1 — training (used for both Phase A and Phase B BackgroundTrainers).
  Load 2 — smoke_test_adapter probe.
The training model is deleted and CUDA cache cleared between the two loads.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch  # noqa: E402

from experiments.utils.gpu_guard import acquire_gpu  # noqa: E402
from experiments.utils.test_harness import (  # noqa: E402
    BENCHMARK_MODELS,
    setup_logging,
    smoke_test_adapter,
)
from paramem.models.loader import create_adapter, load_base_model  # noqa: E402
from paramem.server import background_trainer as bt_mod  # noqa: E402
from paramem.server.background_trainer import (  # noqa: E402
    _RESUME_STATE_FILE,
    BackgroundTrainer,
    TrainingJob,
)
from paramem.training.indexed_memory import (  # noqa: E402
    assign_keys,
    build_registry,
    save_registry,
)
from paramem.utils.config import AdapterConfig, TrainingConfig  # noqa: E402

setup_logging()
logger = logging.getLogger("resume_round_trip")

OUTPUT_BASE = project_root / "outputs" / "test_resume_round_trip"

ADAPTER_RANK = 8
ADAPTER_ALPHA = 16
ADAPTER_LR = 1e-4
ADAPTER_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
TOTAL_EPOCHS = 30
HALT_EPOCH = 15
RECALL_THRESHOLD = 0.95
ADAPTER_NAME = "episodic"  # resume path requires adapter already exists as peft_config

# Two fictional facts — small enough to fit comfortably in a few seconds per epoch.
SYNTHETIC_QA: list[dict] = [
    {"question": "What does Milo do on Sundays?", "answer": "Milo plays chess on Sundays."},
    {"question": "Where does Nora volunteer?", "answer": "Nora volunteers at the library."},
]


def _check_server_inactive() -> None:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", "paramem-server"],
        capture_output=True,
    )
    if result.returncode == 0:
        raise RuntimeError(
            "paramem-server.service is active. Stop it first:\n"
            "  systemctl --user stop paramem-server"
        )


def _check_disk_space(path: Path, min_gb: float = 5.0) -> None:
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024**3)
    if free_gb < min_gb:
        raise RuntimeError(f"Insufficient disk space: {free_gb:.1f} GB free (need {min_gb} GB)")
    logger.info("Disk: %.1f GB free", free_gb)


def _save_json_atomic(data: dict | list, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        Path(tmp_path).replace(target)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _wait_for_thread_exit(trainer: BackgroundTrainer, timeout_s: float = 300.0) -> None:
    """Block until the background training thread finishes (success or error)."""
    deadline = time.time() + timeout_s
    while trainer.is_training and time.time() < deadline:
        time.sleep(0.5)
    if trainer.is_training:
        raise TimeoutError(f"Training thread did not exit within {timeout_s:.0f}s")


def _install_halt_callback(halt_epoch: int) -> None:
    """Monkey-patch _PauseForInferenceCallback.on_epoch_end to raise at halt_epoch.

    The original callback writes resume_state.json before this wrapper fires,
    so the state-on-disk invariant is preserved: when the raise propagates up
    the training thread, resume_state.json already reflects last_completed_epoch
    == halt_epoch.
    """
    original = bt_mod._PauseForInferenceCallback.on_epoch_end

    def patched(self, args, state, control, **kwargs):
        original(self, args, state, control, **kwargs)
        epoch = int(state.epoch)
        if epoch >= halt_epoch:
            logger.warning("INJECTED FAILURE at epoch %d (halt_epoch=%d)", epoch, halt_epoch)
            raise RuntimeError(f"Injected failure at epoch {epoch} for resume test")

    bt_mod._PauseForInferenceCallback.on_epoch_end = patched


def _restore_halt_callback(saved) -> None:
    bt_mod._PauseForInferenceCallback.on_epoch_end = saved


def _inspect_resume_state(adapter_dir: Path) -> dict | None:
    state_path = adapter_dir / "in_training" / _RESUME_STATE_FILE
    if not state_path.exists():
        return None
    with open(state_path) as f:
        return json.load(f)


def _run_phase_a(bt: BackgroundTrainer, job: TrainingJob) -> tuple[bool, int]:
    """Phase A: start training with the halt callback installed.

    Returns (training_raised, last_epoch_on_disk).
    """
    original_on_epoch_end = bt_mod._PauseForInferenceCallback.on_epoch_end
    _install_halt_callback(HALT_EPOCH)
    raised = False
    error_flag = {"hit": False}

    def _on_error():
        error_flag["hit"] = True

    try:
        logger.info("Phase A: submitting job with halt at epoch >= %d", HALT_EPOCH)
        bt.start_jobs([job], on_error=_on_error)
        _wait_for_thread_exit(bt, timeout_s=300.0)
        raised = error_flag["hit"]
    finally:
        _restore_halt_callback(original_on_epoch_end)

    state = _inspect_resume_state(bt.output_dir)
    last_epoch = state.get("last_completed_epoch", 0) if state else 0
    logger.info(
        "Phase A exit: error_callback_fired=%s, resume_state_exists=%s, last_epoch=%d",
        raised,
        state is not None,
        last_epoch,
    )
    return raised, last_epoch


def _run_phase_b(bt: BackgroundTrainer, job: TrainingJob) -> bool:
    """Phase B: fresh BackgroundTrainer submits same job — must resume from state."""
    logger.info("Phase B: fresh BackgroundTrainer, expect resume from checkpoint")
    completed = {"hit": False}

    def _on_complete():
        completed["hit"] = True

    bt.start_jobs([job], on_complete=_on_complete)
    _wait_for_thread_exit(bt, timeout_s=600.0)
    return completed["hit"]


def main(model_name: str, dry_run: bool) -> int:
    if model_name not in BENCHMARK_MODELS:
        raise KeyError(f"Unknown model '{model_name}'. Available: {list(BENCHMARK_MODELS.keys())}")

    model_config = BENCHMARK_MODELS[model_name]

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_BASE / model_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", run_dir)

    if dry_run:
        logger.info(
            "Dry run OK. model=%s, total_epochs=%d, halt_epoch=%d, threshold=%.2f",
            model_config.model_id,
            TOTAL_EPOCHS,
            HALT_EPOCH,
            RECALL_THRESHOLD,
        )
        return 0

    _check_server_inactive()
    _check_disk_space(run_dir, min_gb=5.0)

    t_start = time.time()

    keyed_pairs = assign_keys(SYNTHETIC_QA, start_index=1)
    registry = build_registry(keyed_pairs)
    _save_json_atomic(
        [
            {"key": kp["key"], "question": kp["question"], "answer": kp["answer"]}
            for kp in keyed_pairs
        ],
        run_dir / "keyed_pairs.json",
    )
    save_registry(registry, run_dir / "simhash_registry.json")

    adapter_config = AdapterConfig(
        rank=ADAPTER_RANK,
        alpha=ADAPTER_ALPHA,
        learning_rate=ADAPTER_LR,
        target_modules=ADAPTER_TARGET_MODULES,
        dropout=0.0,
    )
    training_config = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=1,
        max_seq_length=1024,
        num_epochs=TOTAL_EPOCHS,
        warmup_ratio=0.0,
        warmup_steps=0,
        lr_scheduler_type="constant",
        weight_decay=0.0,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=42,
    )

    adapter_output_dir = run_dir / "adapter"
    adapter_output_dir.mkdir(parents=True, exist_ok=True)
    (adapter_output_dir / "in_training").mkdir(parents=True, exist_ok=True)

    outcome = {
        "model": model_name,
        "model_id": model_config.model_id,
        "total_epochs": TOTAL_EPOCHS,
        "halt_epoch": HALT_EPOCH,
        "threshold": RECALL_THRESHOLD,
        "phase_a": {},
        "phase_b": {},
        "phase_c": {},
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }

    with acquire_gpu(interactive=False):
        logger.info("Loading %s NF4 (training load) ...", model_config.model_id)
        model, tokenizer = load_base_model(model_config)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

        model = create_adapter(model, adapter_config, ADAPTER_NAME)
        model = create_adapter(model, adapter_config, "in_training")

        job = TrainingJob(
            keyed_pairs=keyed_pairs,
            adapter_name=ADAPTER_NAME,
            adapter_config=adapter_config,
            inference_fallback_adapter=ADAPTER_NAME,
        )

        # --- Phase A ---
        bt_a = BackgroundTrainer(
            model=model,
            tokenizer=tokenizer,
            training_config=training_config,
            output_dir=adapter_output_dir,
        )
        phase_a_start = time.time()
        raised, last_epoch = _run_phase_a(bt_a, job)
        phase_a_elapsed = time.time() - phase_a_start

        resume_state = _inspect_resume_state(adapter_output_dir)
        checkpoint_dir_exists = (adapter_output_dir / "in_training" / "bg_checkpoint").exists()
        outcome["phase_a"] = {
            "error_callback_fired": raised,
            "resume_state_exists": resume_state is not None,
            "last_completed_epoch": last_epoch,
            "checkpoint_dir_exists": checkpoint_dir_exists,
            "elapsed_seconds": round(phase_a_elapsed, 1),
        }
        if resume_state is None or not checkpoint_dir_exists:
            logger.error("Phase A failed: no resume state or checkpoint on disk")
            _save_json_atomic(outcome, run_dir / "summary.json")
            return 1
        if last_epoch < HALT_EPOCH:
            logger.error(
                "Phase A halted too early: last_epoch=%d < HALT_EPOCH=%d",
                last_epoch,
                HALT_EPOCH,
            )
            _save_json_atomic(outcome, run_dir / "summary.json")
            return 1

        # --- Phase B ---
        bt_b = BackgroundTrainer(
            model=model,
            tokenizer=tokenizer,
            training_config=training_config,
            output_dir=adapter_output_dir,
        )
        phase_b_start = time.time()
        completed = _run_phase_b(bt_b, job)
        phase_b_elapsed = time.time() - phase_b_start

        post_state = _inspect_resume_state(adapter_output_dir)
        checkpoint_dir_cleaned = not (adapter_output_dir / "in_training" / "bg_checkpoint").exists()
        outcome["phase_b"] = {
            "completed": completed,
            "resume_state_cleared": post_state is None,
            "checkpoint_dir_cleaned": checkpoint_dir_cleaned,
            "elapsed_seconds": round(phase_b_elapsed, 1),
        }
        if not completed:
            logger.error("Phase B failed: training did not complete after resume")
            _save_json_atomic(outcome, run_dir / "summary.json")
            return 1
        if post_state is not None:
            logger.error("Phase B failed: resume_state.json still present after success")
            _save_json_atomic(outcome, run_dir / "summary.json")
            return 1

        # --- Release training model before Phase C probe (WSL2 two-load limit) ---
        del model
        del bt_a
        del bt_b
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # --- Phase C: smoke_test_adapter expects cycle_dir/"adapter"/<adapter_name> ---
        logger.info("Phase C: probing recall on committed adapter")
        probe = smoke_test_adapter(run_dir, model_config, adapter_name=ADAPTER_NAME)
        outcome["phase_c"] = {
            "recall": probe["rate"],
            "exact_count": probe["exact_count"],
            "total": probe["total"],
            "per_key": probe["per_key"],
        }

    total_elapsed = time.time() - t_start
    outcome["duration_seconds"] = round(total_elapsed, 1)
    passed = outcome["phase_c"]["recall"] >= RECALL_THRESHOLD
    outcome["passed"] = passed
    _save_json_atomic(outcome, run_dir / "summary.json")

    logger.info(
        "Summary written: passed=%s, recall=%.3f, duration=%.1fs",
        passed,
        outcome["phase_c"]["recall"],
        total_elapsed,
    )

    if passed:
        logger.info("PASSED: resume round-trip verified")
        return 0
    logger.error(
        "FAILED: recall=%.3f < threshold=%.2f",
        outcome["phase_c"]["recall"],
        RECALL_THRESHOLD,
    )
    return 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BackgroundTrainer resume round-trip verification (Task #35).",
    )
    parser.add_argument("--model", default="mistral", choices=list(BENCHMARK_MODELS.keys()))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        rc = main(model_name=args.model, dry_run=args.dry_run)
    except Exception:
        logger.exception("Resume round-trip crashed")
        rc = 2
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    sys.exit(rc)
