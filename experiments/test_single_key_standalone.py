"""Single-key floor verification for indexed-key training.

Trains one synthetic QA pair onto a fresh LoRA adapter using production
defaults (30 epochs, rank 8, alpha 16, threshold 0.95) and verifies
exact recall via smoke_test_adapter().

Pass condition: recall rate == 1.0 on the single trained key.

Complements benchmarking.md §5.1 (10 keys), §5.2 (20 keys), §5.3
(+5 incremental), and the Step 6(B) live GPU run (2 keys via
post_session_train, commit 156ff44). This is the explicit N=1 floor.

Usage:
    conda activate paramem
    python experiments/test_single_key_standalone.py [--model mistral]
    python experiments/test_single_key_standalone.py --dry-run

The script stops only on exit; it does NOT auto-restart
paramem-server.service. Run `systemctl --user start paramem-server`
manually afterwards.

WSL2 constraint: at most two from_pretrained calls per process.
  Load 1 — training (main).
  Load 2 — smoke_test_adapter probe.
The training model is deleted and CUDA cache cleared between the two loads.

Design note: ConsolidationLoop.post_session_train is intentionally bypassed.
post_session_train saves adapters via save_adapter(model, output_dir, name)
which nests files at output_dir/name/, while smoke_test_adapter expects
cycle_dir/"adapter"/name.  Reconciling these layouts would require either
mutating output_dir conventions or post-training symlinks — both are worse
than the direct train_adapter path used by test12, which produces exactly
the layout smoke_test_adapter requires.  GPU guard (acquire_gpu) and
notify_ml discipline are preserved via the acquire_gpu context manager,
which calls notify_ml(ML_STARTED) on enter and notify_ml(ML_FINISHED) on
exit.
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
    IndexedDataset,
    setup_logging,
    smoke_test_adapter,
)
from paramem.models.loader import create_adapter, load_base_model  # noqa: E402
from paramem.training.indexed_memory import (  # noqa: E402
    assign_keys,
    build_registry,
    format_indexed_training,
    save_registry,
)
from paramem.training.trainer import train_adapter  # noqa: E402
from paramem.utils.config import AdapterConfig, TrainingConfig  # noqa: E402

setup_logging()
logger = logging.getLogger("single_key_standalone")

OUTPUT_BASE = project_root / "outputs" / "test_single_key_standalone"

# Production adapter defaults — must match configs/default.yaml adapters.episodic.*
ADAPTER_RANK = 8
ADAPTER_ALPHA = 16  # 2 * rank
ADAPTER_LR = 1e-4
ADAPTER_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
EPOCHS = 30
RECALL_THRESHOLD = 0.95  # same as recall_sanity_threshold in post_session_train

# One synthetic session: fictional entity, single unambiguous fact.
# "Milo" and "chess" are fully fictional — no real personal data.
TRANSCRIPT = (
    "User: My friend Milo plays chess on Sundays.\n"
    "Assistant: That sounds like a fun weekend ritual.\n"
)

# Pre-built synthetic QA pair derived from the transcript triple
# (Milo, plays, chess on Sundays).  Provided explicitly so the test
# does not depend on extraction quality — the point is to verify
# parametric training at N=1, not extraction.
SYNTHETIC_QA: list[dict] = [
    {
        "question": "What does Milo do on Sundays?",
        "answer": "Milo plays chess on Sundays.",
    }
]

ADAPTER_NAME = "single_key_floor"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_server_inactive() -> None:
    """Raise RuntimeError if paramem-server.service is active.

    Uses systemctl --user is-active which exits 0 when active, non-zero
    otherwise.  Refusal prevents VRAM contention and GPU state corruption.
    """
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
    """Raise RuntimeError if free disk space is below min_gb."""
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024**3)
    if free_gb < min_gb:
        raise RuntimeError(f"Insufficient disk space: {free_gb:.1f} GB free (need {min_gb} GB)")
    logger.info("Disk: %.1f GB free", free_gb)


def _save_json_atomic(data: dict | list, target: Path) -> None:
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


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main(model_name: str, dry_run: bool) -> int:
    """Run single-key floor verification.

    Args:
        model_name: Key into BENCHMARK_MODELS (e.g. "mistral").
        dry_run: If True, validate imports and config wiring only — no GPU
            work.  Exits 0 on success, non-zero on import/config failure.

    Returns:
        Exit code: 0 on pass, 1 on test failure, 2 on unexpected error.
    """
    if model_name not in BENCHMARK_MODELS:
        raise KeyError(f"Unknown model '{model_name}'. Available: {list(BENCHMARK_MODELS.keys())}")

    model_config = BENCHMARK_MODELS[model_name]

    # Timestamped, model-specific run directory so re-runs never overwrite.
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_BASE / model_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Output dir: %s", run_dir)

    if dry_run:
        logger.info(
            "Dry run: imports OK. model=%s, epochs=%d, rank=%d, threshold=%.2f, target_modules=%s",
            model_config.model_id,
            EPOCHS,
            ADAPTER_RANK,
            RECALL_THRESHOLD,
            ADAPTER_TARGET_MODULES,
        )
        return 0

    # --- Preflight ---
    _check_server_inactive()
    _check_disk_space(run_dir, min_gb=5.0)

    t_start = time.time()

    # --- Phase 1: Assign keys and persist artefacts ---
    # Artefacts are saved before training so a crash during training
    # does not leave a loadable adapter without its registry/keyed_pairs.
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

    logger.info(
        "Phase 1 done: %d key(s) assigned, artefacts saved to %s",
        len(keyed_pairs),
        run_dir,
    )

    # --- Phase 2: Train (first from_pretrained) ---
    adapter_config = AdapterConfig(
        rank=ADAPTER_RANK,
        alpha=ADAPTER_ALPHA,
        learning_rate=ADAPTER_LR,
        target_modules=ADAPTER_TARGET_MODULES,
        dropout=0.0,
    )

    training_config = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=2,
        max_seq_length=1024,
        num_epochs=EPOCHS,
        warmup_ratio=0.0,
        warmup_steps=0,
        lr_scheduler_type="constant",
        weight_decay=0.0,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=42,
        save_strategy="no",
        save_total_limit=0,
        early_stopping=False,
    )

    adapter_output_dir = run_dir / "adapter"
    adapter_output_dir.mkdir(parents=True, exist_ok=True)

    with acquire_gpu(interactive=False):
        logger.info("Loading %s NF4 (training load) ...", model_config.model_id)
        model, tokenizer = load_base_model(model_config)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

        model = create_adapter(model, adapter_config, ADAPTER_NAME)

        examples = format_indexed_training(keyed_pairs, tokenizer, max_length=1024)
        dataset = IndexedDataset(examples)

        logger.info(
            "Training %d key(s), %d example(s), %d epochs (adapter=%s, rank=%d) ...",
            len(keyed_pairs),
            len(dataset),
            EPOCHS,
            ADAPTER_NAME,
            ADAPTER_RANK,
        )
        t_train = time.time()
        metrics = train_adapter(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            adapter_name=ADAPTER_NAME,
            training_config=training_config,
            adapter_config=adapter_config,
            output_dir=adapter_output_dir,
            run_name=f"single-key-floor-{model_name}",
        )
        train_elapsed = time.time() - t_train
        logger.info(
            "Phase 2 done: trained in %.1fs, loss=%.4f",
            train_elapsed,
            metrics.get("train_loss", -1.0),
        )

        # Release training model before probe load (WSL2 two-load limit).
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # --- Phase 3: Probe recall (second from_pretrained via smoke_test_adapter) ---
        logger.info("Phase 3: probing recall via smoke_test_adapter ...")
        probe_result = smoke_test_adapter(run_dir, model_config, adapter_name=ADAPTER_NAME)

    total_elapsed = time.time() - t_start
    recall = probe_result["rate"]
    passed = recall >= RECALL_THRESHOLD

    summary = {
        "keys_trained": len(keyed_pairs),
        "recall": recall,
        "passed": passed,
        "epochs": EPOCHS,
        "threshold": RECALL_THRESHOLD,
        "adapter_name": ADAPTER_NAME,
        "duration_seconds": round(total_elapsed, 1),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "model": model_name,
        "model_id": model_config.model_id,
        "train_loss": metrics.get("train_loss"),
        "train_time_seconds": round(train_elapsed, 1),
        "exact_count": probe_result["exact_count"],
        "total": probe_result["total"],
        "mean_confidence": probe_result["mean_confidence"],
        "per_key": probe_result["per_key"],
    }
    _save_json_atomic(summary, run_dir / "summary.json")
    logger.info(
        "Summary: recall=%.3f passed=%s (%s) — written to %s",
        recall,
        passed,
        run_dir / "summary.json",
        run_dir,
    )

    if passed:
        logger.info(
            "PASSED: single-key floor verified (recall=%.3f >= %.2f)", recall, RECALL_THRESHOLD
        )
        return 0
    else:
        logger.error("FAILED: recall=%.3f < threshold=%.2f", recall, RECALL_THRESHOLD)
        return 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Single-key floor verification for indexed-key training.",
    )
    parser.add_argument(
        "--model",
        default="mistral",
        choices=list(BENCHMARK_MODELS.keys()),
        help="Model to benchmark (default: mistral).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate imports and config wiring without GPU work.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        rc = main(model_name=args.model, dry_run=args.dry_run)
    except Exception:
        logger.exception("Single-key floor verification crashed.")
        rc = 2
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    sys.exit(rc)
