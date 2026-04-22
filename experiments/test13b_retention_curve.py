"""Test 13b: Retention-Curve Re-run — When Does Retention Degrade?

Research question
-----------------
Test 13's C2 phase (scaffold→fill) reached stable-perfect recall on 40 fill
keys at epoch 11 out of 30 and ended with 37.5% retention on the 160 unchanged
keys (measured ONCE at epoch 30).  This test answers: at what epoch does
retention start degrading, and how does the retention-curve shape relate to the
fill-convergence epoch?

All the data exists to make this cheap: the C1 adapter is saved on disk and is
loaded directly, so no C1 re-training is required.

Scope
-----
- Loads the saved C1 adapter from the production Test 13 run dir.
- Runs the C2 fill phase (40 keys only) against the full 160-key unchanged set.
- Probes BOTH sets at every epoch boundary via ``DualRecallProbeCallback``.
- Writes per-epoch fill + retention metrics to ``epoch_log.json``.
- Derives ``retention_knee_epoch`` (first epoch where retention drops >5pp from
  the running maximum) and ``epochs_between_fill_converged_and_retention_decay``.
- Honors ``~/.training_pause`` at every epoch boundary; resumes from HF
  Trainer checkpoint via ``train_adapter(..., resume_from_checkpoint=...)``.

Non-goals
---------
- Does NOT re-run C1 training.
- Does NOT add baseline B-phase retention probe.
- Does NOT use early stopping — ``stable_perfect_epoch`` is recorded
  observationally only; training runs to full ``num_epochs`` budget.
- Does NOT modify ``experiments/test13_journal_scaffold.py``.

Key metrics in results.json
---------------------------
``fill_stable_perfect`` — epoch where fill recall was stable-perfect (2 consecutive
perfect epochs).  ``retention_at_fill_stable_perfect`` — retention rate at that
epoch.  ``retention_knee_epoch`` — first epoch where retention drops > 5pp from
running max.  ``epochs_between_fill_converged_and_retention_decay`` — knee minus
stable_perfect (may be negative; negative means retention started dropping
BEFORE fill converged).

Resume
------
Pass ``--resume`` to auto-find the latest run under
``outputs/test13b_retention_curve/<model>/``.  The script reads
``run_config.json``, reloads ``fill_keyed.json`` / ``unchanged_keyed.json``
and their registries, locates the latest HF checkpoint in ``adapter/``,
and calls ``train_adapter(..., resume_from_checkpoint=<ckpt>)`` to continue
from the last completed epoch.  Pass ``~/.training_pause`` via ``tpause``
to stop cleanly at the next epoch boundary.

Source run (Test 13 C2 numbers being replicated / extended)
------------------------------------------------------------
  outputs/test13_journal_scaffold/mistral/20260420_231031/
  C2: fill stable_perfect=11, fill 40/40 at epoch 30, retention 37.5% at epoch 30.

Usage
-----
    python experiments/test13b_retention_curve.py             # production run
    python experiments/test13b_retention_curve.py --smoke     # 3-epoch infra check
    python experiments/test13b_retention_curve.py --resume    # continue from last run
    python experiments/test13b_retention_curve.py \\
        --c1-source-run outputs/test13_journal_scaffold/mistral/20260420_231031

GPU prerequisite
----------------
Stop the ParaMem server before running::

    systemctl --user stop paramem-server
    source ~/.local/bin/gpu-cooldown.sh && wait_for_cooldown

The ParaMem server must release the GPU.  This script uses
``experiments.utils.gpu_guard.acquire_gpu()`` which auto-switches the server
to cloud-only for the duration.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from peft import PeftModel  # noqa: E402
from transformers import TrainerCallback  # noqa: E402

# Imported from test13 — the data helpers are importable without running main()
from experiments.test13_journal_scaffold import (  # noqa: E402
    build_phase_C2_fill_keyed,
    load_qa_pool,
    read_keyed,
)
from experiments.utils.gpu_guard import acquire_gpu  # noqa: E402
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
    load_adapter,
    save_adapter,
    switch_adapter,
    unload_model,
)
from paramem.training.indexed_memory import (  # noqa: E402
    build_registry,
    format_indexed_training,
    save_registry,
)
from paramem.training.trainer import train_adapter  # noqa: E402
from paramem.utils.config import AdapterConfig, TrainingConfig  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = project_root / "outputs" / "test13b_retention_curve"
DEFAULT_C1_SOURCE_RUN = (
    project_root / "outputs" / "test13_journal_scaffold" / "mistral" / "20260420_231031"
)

# C1 training had total_keys=200, swap_keys=40 → unchanged = first 160.
# These are the production defaults; smoke mode truncates AFTER slicing.
TOTAL_KEYS = 200
SWAP_KEYS = 40
SWAP_START_SLOT = TOTAL_KEYS - SWAP_KEYS  # 160

STABLE_EPOCH_WINDOW = 2  # consecutive perfect epochs for "stable" (matches Test 13)
RETENTION_KNEE_THRESHOLD = 0.05  # 5pp drop from running max counts as knee

PAUSE_FILE = Path.home() / ".training_pause"


# ---------------------------------------------------------------------------
# Probe state
# ---------------------------------------------------------------------------


@dataclass
class EpochProbeState:
    """Dual fill+retention probe state accumulated across epochs."""

    first_perfect_epoch: int | None = None
    stable_perfect_epoch: int | None = None
    epoch_log: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DualRecallProbeCallback — probes fill + unchanged at every epoch boundary
# ---------------------------------------------------------------------------


class DualRecallProbeCallback(TrainerCallback):
    """Probe fill keys and unchanged keys after every training epoch.

    Writes a combined entry per epoch to ``state_out.epoch_log``::

        {
            "epoch": N,
            "fill":      {"recall": int, "total": int, "rate": float, "mean_confidence": float},
            "retention": {"recall": int, "total": int, "rate": float, "mean_confidence": float},
        }

    Tracks ``first_perfect_epoch`` / ``stable_perfect_epoch`` on the FILL set
    only (matches Test 13 C2 reporting semantics).  ``STABLE_EPOCH_WINDOW=2``
    for comparability.

    Also increments ``epoch_log.json`` and ``progress.json`` after every epoch
    so that mid-run state survives a crash.

    Pause semantics: after writing progress, checks ``pause_file.exists()``.
    If the pause file is present, sets ``control.should_training_stop = True``
    and logs the event.  The caller's main() detects the incomplete log and
    writes ``paused.json``.
    """

    def __init__(
        self,
        model,
        tokenizer,
        fill_keyed: list[dict],
        fill_registry: dict[str, int],
        unchanged_keyed: list[dict],
        unchanged_registry: dict[str, int],
        adapter_name: str,
        state_out: EpochProbeState,
        progress_path: Path,
        epoch_log_path: Path,
        phase_name: str,
        num_epochs: int,
        pause_file: Path,
    ):
        """Initialise the dual-probe callback.

        Args:
            model: The PeftModel being trained.
            tokenizer: Tokenizer for generation.
            fill_keyed: The 40 fill key–value pairs (subject of training).
            fill_registry: SimHash registry for fill keys.
            unchanged_keyed: The 160 unchanged key–value pairs (retention probe).
            unchanged_registry: SimHash registry for unchanged keys.
            adapter_name: Active adapter name (``"journal"``).
            state_out: Mutable state shared with the caller.
            progress_path: Path to write ``progress.json`` after each epoch.
            epoch_log_path: Path to write/extend ``epoch_log.json`` after each epoch.
            phase_name: String label used in progress.json (``"fill_retention_curve"``).
            num_epochs: Total training epoch budget.
            pause_file: Path of the global pause semaphore (``~/.training_pause``).
        """
        self._model = model
        self._tokenizer = tokenizer
        self._fill_keyed = fill_keyed
        self._fill_registry = fill_registry
        self._unchanged_keyed = unchanged_keyed
        self._unchanged_registry = unchanged_registry
        self._adapter_name = adapter_name
        self._state = state_out
        self._progress_path = progress_path
        self._epoch_log_path = epoch_log_path
        self._phase_name = phase_name
        self._num_epochs = num_epochs
        self._pause_file = pause_file
        self._last_epoch = -1
        self._cycle_started_at = int(time.time())

    def on_epoch_end(self, args, state, control, **kwargs):
        """Probe fill + unchanged sets, update logs, check pause."""
        epoch = int(round(state.epoch))
        if epoch <= self._last_epoch:
            return
        self._last_epoch = epoch

        # --- Fill probe ---
        fill_recall = evaluate_indexed_recall(
            self._model,
            self._tokenizer,
            self._fill_keyed,
            self._fill_registry,
            adapter_name=self._adapter_name,
        )

        # --- Unchanged (retention) probe ---
        retention_recall = evaluate_indexed_recall(
            self._model,
            self._tokenizer,
            self._unchanged_keyed,
            self._unchanged_registry,
            adapter_name=self._adapter_name,
        )

        # Re-enable gradient checkpointing once after both probes (evaluate
        # calls disable it; one re-enable per on_epoch_end is sufficient).
        self._model.gradient_checkpointing_enable()

        entry = {
            "epoch": epoch,
            "fill": {
                "recall": fill_recall["exact_count"],
                "total": fill_recall["total"],
                "rate": fill_recall["rate"],
                "mean_confidence": fill_recall["mean_confidence"],
            },
            "retention": {
                "recall": retention_recall["exact_count"],
                "total": retention_recall["total"],
                "rate": retention_recall["rate"],
                "mean_confidence": retention_recall["mean_confidence"],
            },
        }
        self._state.epoch_log.append(entry)

        logger.info(
            "  epoch %d: fill %d/%d (%.3f) | retention %d/%d (%.3f)",
            epoch,
            fill_recall["exact_count"],
            fill_recall["total"],
            fill_recall["rate"],
            retention_recall["exact_count"],
            retention_recall["total"],
            retention_recall["rate"],
        )

        # --- Fill stability tracking ---
        fill_is_perfect = (
            fill_recall["exact_count"] == fill_recall["total"] and fill_recall["total"] > 0
        )
        if fill_is_perfect and self._state.first_perfect_epoch is None:
            self._state.first_perfect_epoch = epoch

        if self._state.stable_perfect_epoch is None and fill_is_perfect:
            window = self._state.epoch_log[-STABLE_EPOCH_WINDOW:]
            if len(window) >= STABLE_EPOCH_WINDOW and all(
                e["fill"]["recall"] == e["fill"]["total"] and e["fill"]["total"] > 0 for e in window
            ):
                self._state.stable_perfect_epoch = epoch

        # --- Write epoch_log.json incrementally (crash safety) ---
        try:
            self._epoch_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._epoch_log_path.write_text(json.dumps(self._state.epoch_log, indent=2))
        except OSError as exc:
            logger.warning("epoch_log.json write failed: %s", exc)

        # --- Write progress.json (for tstatus / training-control.sh) ---
        progress = {
            "phase": self._phase_name,
            "keys": len(self._fill_keyed) + len(self._unchanged_keyed),
            "epoch": epoch,
            "total_epochs": self._num_epochs,
            "epoch_offset": 0,
            "cycle_started_at": self._cycle_started_at,
            "fill_recall": fill_recall["exact_count"],
            "fill_total": fill_recall["total"],
            "fill_rate": fill_recall["rate"],
            "retention_recall": retention_recall["exact_count"],
            "retention_total": retention_recall["total"],
            "retention_rate": retention_recall["rate"],
        }
        try:
            self._progress_path.parent.mkdir(parents=True, exist_ok=True)
            self._progress_path.write_text(json.dumps(progress, indent=2))
        except OSError as exc:
            logger.warning("progress.json write failed: %s", exc)

        # --- Pause check (after writing progress) ---
        if self._pause_file.exists():
            logger.warning(
                "Pause file detected at epoch %d — signalling stop. Use tresume 13b to continue.",
                epoch,
            )
            control.should_training_stop = True


# ---------------------------------------------------------------------------
# CLI / run_config helpers
# ---------------------------------------------------------------------------


def _find_latest_run_dir(model_name: str) -> Path | None:
    """Return the most recent timestamped prod run dir for the given model.

    Skips dirs whose ``run_config.json`` marks the run as smoke — otherwise
    a leftover smoke dir with ``fill_done.json`` would short-circuit a
    subsequent ``--resume`` prod launch.
    """
    parent = OUTPUT_DIR / model_name
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


def load_or_write_run_config(output_dir: Path, args) -> dict:
    """On first launch: write ``run_config.json`` from args.  On resume: read and apply.

    Ensures that ``--num-epochs``, ``--rank``, etc. are frozen for the
    lifetime of a run so that ``--resume`` always restores the original budget.
    """
    cfg_path = output_dir / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        logger.info("Loaded run_config.json: %s", cfg)
        return cfg
    cfg = {
        "model": args.model,
        "num_epochs": args.num_epochs,
        "rank": args.rank,
        "c1_source_run": str(args.c1_source_run),
        "smoke": bool(getattr(args, "smoke", False)),
        "created_at": int(time.time()),
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))
    logger.info("Wrote run_config.json: %s", cfg)
    return cfg


# ---------------------------------------------------------------------------
# Step-number extractor for sorting checkpoint dirs
# ---------------------------------------------------------------------------


def _step_num_from_checkpoint(ckpt_path: str | Path) -> int:
    """Return the step number encoded in a checkpoint dir name.

    ``adapter/checkpoint-120`` → 120.  Returns -1 if not parseable.
    """
    name = Path(ckpt_path).name
    if name.startswith("checkpoint-"):
        try:
            return int(name.split("-", 1)[1])
        except (IndexError, ValueError):
            pass
    return -1


def _find_latest_checkpoint(adapter_dir: Path) -> Path | None:
    """Return the checkpoint-N dir with the highest step number inside adapter_dir."""
    candidates = [Path(p) for p in glob(str(adapter_dir / "checkpoint-*")) if Path(p).is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: _step_num_from_checkpoint(p))


# ---------------------------------------------------------------------------
# Derived analytics
# ---------------------------------------------------------------------------


def _compute_results(
    probe_state: EpochProbeState,
    c1_source_run: Path,
    output_dir: Path,
    fill_keyed: list[dict],
    unchanged_keyed: list[dict],
    metrics: dict,
    wall: float,
) -> dict:
    """Derive summary analytics from the epoch log and write results.json."""
    epoch_log = probe_state.epoch_log

    # Final-epoch entries
    final_entry = epoch_log[-1] if epoch_log else None
    final_fill = final_entry["fill"] if final_entry else {}
    final_retention = final_entry["retention"] if final_entry else {}

    # Retention at fill stable_perfect epoch
    retention_at_stable: float | None = None
    if probe_state.stable_perfect_epoch is not None:
        # epoch_log is 1-indexed by epoch number; find matching entry
        for e in epoch_log:
            if e["epoch"] == probe_state.stable_perfect_epoch:
                retention_at_stable = e["retention"]["rate"]
                break

    # Retention knee: first epoch where retention drops > RETENTION_KNEE_THRESHOLD
    # from the running maximum up to that point
    retention_knee_epoch: int | None = None
    retention_peak_epoch: int | None = None
    retention_peak_rate: float = 0.0
    running_max = 0.0
    for e in epoch_log:
        r = e["retention"]["rate"]
        if r > running_max:
            running_max = r
            retention_peak_rate = r
            retention_peak_epoch = e["epoch"]
        elif running_max > 0.0 and (running_max - r) > RETENTION_KNEE_THRESHOLD:
            if retention_knee_epoch is None:
                retention_knee_epoch = e["epoch"]

    # Epochs between fill converging and retention starting to decay
    epochs_between: int | None = None
    if probe_state.stable_perfect_epoch is not None and retention_knee_epoch is not None:
        epochs_between = retention_knee_epoch - probe_state.stable_perfect_epoch

    results = {
        "output_dir": str(output_dir),
        "c1_source_run": str(c1_source_run),
        "fill_first_perfect": probe_state.first_perfect_epoch,
        "fill_stable_perfect": probe_state.stable_perfect_epoch,
        "fill_final_recall": final_fill,
        "retention_at_fill_stable_perfect": retention_at_stable,
        "retention_at_final": final_retention,
        "retention_peak_epoch": retention_peak_epoch,
        "retention_peak_rate": retention_peak_rate,
        "retention_knee_epoch": retention_knee_epoch,
        "epochs_between_fill_converged_and_retention_decay": epochs_between,
        "wall_seconds": round(wall, 1),
        "train_loss": metrics.get("train_loss"),
        "n_fill": len(fill_keyed),
        "n_unchanged": len(unchanged_keyed),
        "epoch_log": epoch_log,
    }

    save_results(results, output_dir, "results.json")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():  # noqa: C901  (complexity acceptable for experiment orchestration)
    """Entry point for Test 13b: retention curve re-run."""
    parser = argparse.ArgumentParser(description="Test 13b: Retention-Curve Re-run")
    parser.add_argument("--model", choices=["mistral"], default="mistral")
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument(
        "--c1-source-run",
        type=Path,
        default=DEFAULT_C1_SOURCE_RUN,
        help="Path to a Test 13 run dir whose C1 adapter + keyed_pairs.json will be loaded.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Explicit output directory. If omitted (and --resume not set), a fresh "
        "timestamped dir is created under outputs/test13b_retention_curve/<model>/.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Auto-find the latest run dir for --model and continue.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast infra smoke: fill=5, unchanged=10, num_epochs=3, save_total_limit=3.",
    )
    args = parser.parse_args()

    # Smoke defaults (applied before output_dir resolution so they persist in run_config)
    if args.smoke:
        args.num_epochs = 3
        logger.info("SMOKE mode: num_epochs=3, fill=5, unchanged=10")

    # --- Output dir resolution ---
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.resume:
        latest = _find_latest_run_dir(args.model)
        if latest is None:
            logger.warning(
                "--resume requested but no prior run for %s — starting fresh", args.model
            )
            output_dir = model_output_dir(OUTPUT_DIR, args.model)
        else:
            output_dir = latest
            logger.info("Resuming from %s", output_dir)
    else:
        output_dir = model_output_dir(OUTPUT_DIR, args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", output_dir)

    # --- Run config ---
    cfg = load_or_write_run_config(output_dir, args)
    # On resume, restore frozen params from config
    args.num_epochs = cfg["num_epochs"]
    args.rank = cfg["rank"]
    args.c1_source_run = Path(cfg["c1_source_run"])
    smoke = cfg.get("smoke", False)

    # --- Short-circuit if already complete ---
    fill_done_path = output_dir / "fill_done.json"
    if fill_done_path.exists():
        logger.info("fill_done.json exists — run already complete. Exiting.")
        return

    # --- Clear stale pause semaphore and paused.json on fresh launch ---
    if PAUSE_FILE.exists():
        logger.warning("Pause file present at launch — clearing before starting work")
        try:
            PAUSE_FILE.unlink()
        except OSError:
            pass

    stale_paused = output_dir / "paused.json"
    if stale_paused.exists():
        stale_paused.unlink()

    # --- Load or rebuild data ---
    fill_keyed_path = output_dir / "fill_keyed.json"
    unchanged_keyed_path = output_dir / "unchanged_keyed.json"
    fill_registry_path = output_dir / "fill_simhash_registry.json"
    unchanged_registry_path = output_dir / "unchanged_simhash_registry.json"

    epoch_log_path = output_dir / "epoch_log.json"

    if fill_keyed_path.exists() and unchanged_keyed_path.exists():
        # Resume path: reload from disk
        logger.info("Loading keyed data from %s (resume)", output_dir)
        with open(fill_keyed_path) as f:
            fill_keyed = json.load(f)
        with open(unchanged_keyed_path) as f:
            unchanged_keyed = json.load(f)
        with open(fill_registry_path) as f:
            fill_registry = {k: int(v) for k, v in json.load(f).items()}
        with open(unchanged_registry_path) as f:
            unchanged_registry = {k: int(v) for k, v in json.load(f).items()}
        logger.info("Loaded %d fill / %d unchanged keys", len(fill_keyed), len(unchanged_keyed))
    else:
        # Fresh build
        logger.info("Loading C1 keyed data from %s", args.c1_source_run)
        c1_keyed, _ = read_keyed(args.c1_source_run, "C1")

        # Determine production swap boundary from the source run config
        c1_cfg_path = args.c1_source_run / "run_config.json"
        if c1_cfg_path.exists():
            c1_cfg = json.loads(c1_cfg_path.read_text())
            c1_total = c1_cfg.get("total_keys", TOTAL_KEYS)
            c1_swap = c1_cfg.get("swap_keys", SWAP_KEYS)
        else:
            c1_total = TOTAL_KEYS
            c1_swap = SWAP_KEYS
        c1_swap_start = c1_total - c1_swap

        # Build QA pool (needed to resolve real answers for fill set)
        qa_pool = load_qa_pool(c1_total)

        # Override SWAP_START_SLOT in imported module scope so
        # build_phase_C2_fill_keyed uses the right boundary. Restore
        # unconditionally so an exception inside the helper cannot leak
        # the patched value to any later caller in this process.
        import experiments.test13_journal_scaffold as t13

        orig_swap_start = t13.SWAP_START_SLOT
        t13.SWAP_START_SLOT = c1_swap_start
        try:
            fill_keyed = build_phase_C2_fill_keyed(c1_keyed, qa_pool)
        finally:
            t13.SWAP_START_SLOT = orig_swap_start

        # Unchanged keys: everything before the swap boundary
        unchanged_keyed = c1_keyed[:c1_swap_start]

        # Smoke: truncate both sets
        if smoke:
            fill_keyed = fill_keyed[:5]
            unchanged_keyed = unchanged_keyed[:10]
            logger.info(
                "SMOKE: truncated to %d fill / %d unchanged keys",
                len(fill_keyed),
                len(unchanged_keyed),
            )

        fill_registry = build_registry(fill_keyed)
        unchanged_registry = build_registry(unchanged_keyed)

        # Persist for crash-safe resume
        with open(fill_keyed_path, "w") as f:
            json.dump(
                [
                    {"key": kp["key"], "question": kp["question"], "answer": kp["answer"]}
                    for kp in fill_keyed
                ],
                f,
                indent=2,
            )
        with open(unchanged_keyed_path, "w") as f:
            json.dump(
                [
                    {"key": kp["key"], "question": kp["question"], "answer": kp["answer"]}
                    for kp in unchanged_keyed
                ],
                f,
                indent=2,
            )
        save_registry(fill_registry, fill_registry_path)
        save_registry(unchanged_registry, unchanged_registry_path)
        logger.info(
            "Built %d fill keys + %d unchanged keys; persisted to %s",
            len(fill_keyed),
            len(unchanged_keyed),
            output_dir,
        )

    # --- Reload partial epoch_log for resume ---
    probe_state = EpochProbeState()
    if epoch_log_path.exists():
        try:
            saved_log = json.loads(epoch_log_path.read_text())
            probe_state.epoch_log = saved_log
            # Recompute fill stability from the loaded log
            for e in saved_log:
                fill_count = e["fill"]["recall"]
                fill_total = e["fill"]["total"]
                is_perfect = fill_count == fill_total and fill_total > 0
                epoch = e["epoch"]
                if is_perfect and probe_state.first_perfect_epoch is None:
                    probe_state.first_perfect_epoch = epoch
                if probe_state.stable_perfect_epoch is None and is_perfect:
                    window_start = max(0, len(probe_state.epoch_log) - STABLE_EPOCH_WINDOW)
                    window = probe_state.epoch_log[window_start:]
                    if len(window) >= STABLE_EPOCH_WINDOW and all(
                        w["fill"]["recall"] == w["fill"]["total"] and w["fill"]["total"] > 0
                        for w in window
                    ):
                        probe_state.stable_perfect_epoch = epoch
            logger.info(
                "Restored %d epoch entries from epoch_log.json "
                "(first_perfect=%s, stable_perfect=%s)",
                len(saved_log),
                probe_state.first_perfect_epoch,
                probe_state.stable_perfect_epoch,
            )
        except Exception as exc:
            logger.warning("Failed to reload epoch_log.json: %s — starting fresh", exc)
            probe_state = EpochProbeState()

    # --- Locate C1 adapter ---
    c1_adapter_dir = args.c1_source_run / "C1" / "adapter"

    # --- TrainingConfig / AdapterConfig ---
    save_total_limit = 3 if smoke else args.num_epochs
    training_config = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=2,
        max_seq_length=1024,
        num_epochs=args.num_epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=42,
        save_strategy="epoch",
        save_total_limit=save_total_limit,
    )
    adapter_config = AdapterConfig(
        rank=args.rank,
        alpha=args.rank * 2,
        learning_rate=1e-4,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        dropout=0.0,
    )
    adapter_name = "journal"
    adapter_output_dir = output_dir / "adapter"
    model_config = BENCHMARK_MODELS[args.model]

    # --- Locate latest HF checkpoint for resume ---
    resume_checkpoint: Path | None = None
    if probe_state.epoch_log:
        latest_ckpt = _find_latest_checkpoint(adapter_output_dir)
        if latest_ckpt is not None:
            resume_checkpoint = latest_ckpt
            logger.info("Will resume from HF checkpoint: %s", resume_checkpoint)
        else:
            logger.warning(
                "epoch_log has %d entries but no checkpoint found in %s — "
                "restarting from C1 adapter",
                len(probe_state.epoch_log),
                adapter_output_dir,
            )
            probe_state = EpochProbeState()  # discard stale log

    # --- Dataset ---
    model_config_ref = model_config  # capture for acquire_gpu scope
    with acquire_gpu(interactive=True):
        model, tokenizer, _ = load_model_and_config(model_config_ref)

        # Load C1 adapter weights into the model
        logger.info("Loading C1 adapter from %s", c1_adapter_dir)
        slot = resolve_adapter_slot(c1_adapter_dir, adapter_name, "")
        if slot is not None:
            logger.info("Resolved C1 slot: %s", slot)
            model = PeftModel.from_pretrained(model, str(slot), adapter_name=adapter_name)
        else:
            logger.info("No slot found; using load_adapter for %s", c1_adapter_dir)
            model = load_adapter(model, c1_adapter_dir, adapter_name)
        switch_adapter(model, adapter_name)

        # Epoch-0 baseline probe: measure fill + retention on the loaded
        # C1 adapter BEFORE any fill training. Gives retention_knee_epoch
        # a pre-training anchor — without it, the running-max starts at
        # epoch 1 and the knee can be mis-assigned when retention drops
        # between e0 and e1 by > RETENTION_KNEE_THRESHOLD. Skipped on
        # resume because epoch_log already contains e0 from the first run.
        if not probe_state.epoch_log:
            logger.info("Running epoch-0 baseline probe (fill + retention)...")
            fill_pre = evaluate_indexed_recall(
                model, tokenizer, fill_keyed, fill_registry, adapter_name=adapter_name
            )
            retention_pre = evaluate_indexed_recall(
                model,
                tokenizer,
                unchanged_keyed,
                unchanged_registry,
                adapter_name=adapter_name,
            )
            model.gradient_checkpointing_enable()
            entry_0 = {
                "epoch": 0,
                "fill": {
                    "recall": fill_pre["exact_count"],
                    "total": fill_pre["total"],
                    "rate": fill_pre["rate"],
                    "mean_confidence": fill_pre["mean_confidence"],
                },
                "retention": {
                    "recall": retention_pre["exact_count"],
                    "total": retention_pre["total"],
                    "rate": retention_pre["rate"],
                    "mean_confidence": retention_pre["mean_confidence"],
                },
            }
            probe_state.epoch_log.append(entry_0)
            epoch_log_path.write_text(json.dumps(probe_state.epoch_log, indent=2))
            logger.info(
                "  epoch 0: fill=%d/%d (conf=%.3f)  retention=%d/%d (conf=%.3f)",
                fill_pre["exact_count"],
                fill_pre["total"],
                fill_pre["mean_confidence"],
                retention_pre["exact_count"],
                retention_pre["total"],
                retention_pre["mean_confidence"],
            )

        # Build training dataset from fill keys
        examples = format_indexed_training(fill_keyed, tokenizer, max_length=1024)
        dataset = IndexedDataset(examples)

        progress_path = output_dir / "progress.json"
        callback = DualRecallProbeCallback(
            model=model,
            tokenizer=tokenizer,
            fill_keyed=fill_keyed,
            fill_registry=fill_registry,
            unchanged_keyed=unchanged_keyed,
            unchanged_registry=unchanged_registry,
            adapter_name=adapter_name,
            state_out=probe_state,
            progress_path=progress_path,
            epoch_log_path=epoch_log_path,
            phase_name="fill_retention_curve",
            num_epochs=args.num_epochs,
            pause_file=PAUSE_FILE,
        )

        logger.info(
            "Starting fill training: %d fill keys, %d unchanged probed, %d epochs",
            len(fill_keyed),
            len(unchanged_keyed),
            args.num_epochs,
        )

        t0 = time.time()
        metrics = train_adapter(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            adapter_name=adapter_name,
            training_config=training_config,
            adapter_config=adapter_config,
            output_dir=adapter_output_dir,
            run_name="test13b-fill-retention",
            callbacks_extra=[callback],
            resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None,
        )
        wall = time.time() - t0

        # --- Detect pause / incomplete run ---
        completed_epoch = probe_state.epoch_log[-1]["epoch"] if probe_state.epoch_log else 0
        if completed_epoch < args.num_epochs:
            # Stopped early — write paused.json and exit without fill_done.json
            latest_ckpt_after = _find_latest_checkpoint(adapter_output_dir)
            latest_step = (
                _step_num_from_checkpoint(latest_ckpt_after) if latest_ckpt_after else None
            )
            paused_data = {
                "stopped_after_epoch": completed_epoch,
                "latest_checkpoint_step": latest_step,
                "c1_source_run": str(args.c1_source_run),
                "timestamp": int(time.time()),
            }
            (output_dir / "paused.json").write_text(json.dumps(paused_data, indent=2))
            logger.warning(
                "Run paused after epoch %d / %d. Use tresume 13b to continue.",
                completed_epoch,
                args.num_epochs,
            )
            unload_model(model, tokenizer)
            return

        # --- Post-training: save adapter slot ---
        manifest = build_manifest_for(
            model,
            tokenizer,
            adapter_name,
            registry_path=None,
            keyed_pairs_path=output_dir / "fill_keyed.json",
            key_count=len(fill_keyed),
        )
        save_adapter(model, adapter_output_dir, adapter_name, manifest=manifest)

        # --- Compute and write results.json ---
        results = _compute_results(
            probe_state=probe_state,
            c1_source_run=args.c1_source_run,
            output_dir=output_dir,
            fill_keyed=fill_keyed,
            unchanged_keyed=unchanged_keyed,
            metrics=metrics,
            wall=wall,
        )

        # --- Write fill_done.json marker ---
        fill_done = {
            "first_perfect_epoch": probe_state.first_perfect_epoch,
            "stable_perfect_epoch": probe_state.stable_perfect_epoch,
            "condition": "fill_retention_curve",
            "n_fill": len(fill_keyed),
            "n_unchanged": len(unchanged_keyed),
            "train_loss": metrics.get("train_loss"),
            "wall_seconds": round(wall, 1),
            "final_fill": results["fill_final_recall"],
            "final_retention": results["retention_at_final"],
            "retention_knee_epoch": results["retention_knee_epoch"],
            "epoch_log": probe_state.epoch_log,
        }
        fill_done_path.write_text(json.dumps(fill_done, indent=2))
        logger.info("fill_done.json written: %s", fill_done_path)

        # --- Summary log ---
        logger.info("=" * 72)
        logger.info("Test 13b Summary")
        logger.info(
            "  fill:  first_perfect=%s  stable_perfect=%s  final=%s/%s",
            probe_state.first_perfect_epoch,
            probe_state.stable_perfect_epoch,
            results["fill_final_recall"].get("recall"),
            results["fill_final_recall"].get("total"),
        )
        logger.info(
            "  retention:  at_stable_perfect=%s  peak=%s@E%s  knee=E%s  final=%s/%s",
            (
                f"{results['retention_at_fill_stable_perfect']:.3f}"
                if results["retention_at_fill_stable_perfect"] is not None
                else "N/A"
            ),
            f"{results['retention_peak_rate']:.3f}" if results["retention_peak_rate"] else "?",
            results["retention_peak_epoch"],
            results["retention_knee_epoch"],
            results["retention_at_final"].get("recall"),
            results["retention_at_final"].get("total"),
        )
        logger.info(
            "  epochs fill→knee: %s  wall=%.1fs",
            results["epochs_between_fill_converged_and_retention_decay"],
            wall,
        )
        logger.info("=" * 72)

        unload_model(model, tokenizer)


if __name__ == "__main__":
    main()
