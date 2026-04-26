"""Early-stop policy and callback for Test 14.

``EarlyStopPolicy`` — dataclass encoding when probing starts, when the stop
signal can fire, and how many consecutive perfect probes are required.

``RecallEarlyStopCallback`` — HF ``TrainerCallback`` that probes fill and
(optionally) retention sets after each epoch, tracks the first-perfect log,
writes ``epoch_log.json`` incrementally, and fires
``control.should_training_stop = True`` when the aggregate-recall stop
condition is met.

Stop-trigger semantics (lock-in §13.3 of plan-test14.md)
---------------------------------------------------------
The callback fires when ``fill["exact_count"] == fill["total"]`` (aggregate
100% recall) for ``window`` consecutive probes AND ``epoch >=
signal_from_epoch``.  Per-key streak tracking is NOT used; 100% aggregate
at a probe epoch means every key passed at that probe, which is equivalent
in effect but does not require per-key state.

``gradient_checkpointing`` re-enable (§14.3 of plan-test14.md)
---------------------------------------------------------------
``evaluate_indexed_recall`` (test_harness.py:403) calls
``model.gradient_checkpointing_disable()`` internally.  The in-loop
callback re-enables it inline after both probes complete and before
returning from ``on_epoch_end``.  This is the "callback is exempt from the
standalone helper rule" mentioned in §14.3 — HF Trainer manages
checkpointing state across ``on_epoch_end`` boundaries, so the next epoch's
training step sees the re-enabled state.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from transformers import TrainerCallback

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Default policy used throughout Test 14 (plan §6, §8).
# probe_from_epoch=1: start probing at the first epoch boundary.
# signal_from_epoch=10: only fire the stop signal after epoch 10.
# window=3: require 3 consecutive probes with aggregate 100% recall.
# probe_every_n_epochs=1: probe at every epoch boundary.
ANALYSIS_POLICY_KWARGS = {
    "probe_from_epoch": 1,
    "signal_from_epoch": 10,
    "window": 3,
    "probe_every_n_epochs": 1,
}


@dataclass
class EarlyStopPolicy:
    """Parameters controlling when probing starts and when to stop training.

    Attributes:
        probe_from_epoch: Earliest epoch at which a probe is run.
        signal_from_epoch: Earliest epoch at which the stop signal can fire.
            Must be >= probe_from_epoch.
        window: Number of consecutive perfect probes required to fire the stop
            signal.
        probe_every_n_epochs: Run a probe every N epochs (1 = every epoch).

    Raises:
        ValueError: If ``signal_from_epoch < probe_from_epoch``, or if
            ``window < 1``, or if ``probe_every_n_epochs < 1``.
    """

    probe_from_epoch: int = 10
    signal_from_epoch: int = 10
    window: int = 3
    probe_every_n_epochs: int = 1

    def __post_init__(self) -> None:
        """Validate invariants."""
        if self.signal_from_epoch < self.probe_from_epoch:
            raise ValueError(
                f"signal_from_epoch ({self.signal_from_epoch}) must be >= "
                f"probe_from_epoch ({self.probe_from_epoch})"
            )
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")
        if self.probe_every_n_epochs < 1:
            raise ValueError(f"probe_every_n_epochs must be >= 1, got {self.probe_every_n_epochs}")


# Singleton policy for Test 14 phases (plan §6 ANALYSIS_POLICY).
ANALYSIS_POLICY = EarlyStopPolicy(**ANALYSIS_POLICY_KWARGS)


@dataclass
class _EarlyStopState:
    """Mutable accumulator written back to the caller via state_out."""

    first_perfect_epoch: int | None = None
    stable_perfect_epoch: int | None = None
    stop_epoch: int | None = None
    epoch_log: list[dict] = field(default_factory=list)


class RecallEarlyStopCallback(TrainerCallback):
    """Per-epoch recall probe with aggregate-recall-based early stop.

    Probes the fill set (and optionally a retention set) after each epoch
    boundary.  Fires ``control.should_training_stop = True`` when aggregate
    100% fill-recall holds for ``policy.window`` consecutive probes AND the
    epoch has passed ``policy.signal_from_epoch``.

    Writes ``epoch_log.json`` and ``progress.json`` incrementally so
    mid-run state survives a crash.

    Parameters
    ----------
    model:
        The PeftModel being trained.
    tokenizer:
        Tokenizer for generation (passed through to
        ``evaluate_indexed_recall``).
    target_keyed:
        Fill key–value pairs (the training target set; probed every epoch).
    target_registry:
        SimHash registry for ``target_keyed``.
    adapter_name:
        Active adapter name (e.g. ``"journal"``).
    policy:
        ``EarlyStopPolicy`` instance controlling probe schedule and stop gate.
    state_out:
        Mutable ``_EarlyStopState`` shared with the caller; accumulates
        ``first_perfect_epoch``, ``stable_perfect_epoch``, ``stop_epoch``,
        and ``epoch_log``.
    progress_path:
        Path to write ``progress.json`` after each epoch.  May be None.
    epoch_log_path:
        Path to write ``epoch_log.json`` after each epoch (incrementally).
    first_perfect_log_path:
        Path to write ``first_perfect_log.json`` after each probe.
    phase_name:
        String label for ``progress.json`` (e.g. ``"Phase_C"``).
    num_epochs:
        Total training epoch budget (for progress display).
    pause_file:
        Path of the global pause semaphore (``~/.training_pause``).
    retention_keyed:
        Optional unchanged-key list for dual-probe logging.
    retention_registry:
        Required if ``retention_keyed`` is provided.
    """

    def __init__(
        self,
        *,
        model,
        tokenizer,
        target_keyed: list[dict],
        target_registry: dict[str, int],
        adapter_name: str,
        policy: EarlyStopPolicy,
        state_out: _EarlyStopState,
        progress_path: Path | None,
        epoch_log_path: Path,
        first_perfect_log_path: Path,
        phase_name: str,
        num_epochs: int,
        pause_file: Path,
        retention_keyed: list[dict] | None = None,
        retention_registry: dict[str, int] | None = None,
    ) -> None:
        """Initialise callback state."""
        self._model = model
        self._tokenizer = tokenizer
        self._target_keyed = target_keyed
        self._target_registry = target_registry
        self._adapter_name = adapter_name
        self._policy = policy
        self._state = state_out
        self._progress_path = progress_path
        self._epoch_log_path = epoch_log_path
        self._first_perfect_log_path = first_perfect_log_path
        self._phase_name = phase_name
        self._num_epochs = num_epochs
        self._pause_file = pause_file
        self._retention_keyed = retention_keyed
        self._retention_registry = retention_registry

        # Internal state (§6 pseudocode variable names).
        self._last_epoch: int = -1
        self._consecutive_perfect: int = 0
        self._first_perfect_log: dict[str, dict] = {}
        self._signaled_stop: bool = False
        self._cycle_started_at: int = int(time.time())

    def on_epoch_end(self, args, state, control, **kwargs) -> None:  # noqa: ARG002
        """Probe fill (and optionally retention), update logs, check stop.

        Implements the pseudocode from plan §6 exactly:
        - _last_epoch guard prevents double-fire for the same epoch.
        - Only probes on ``probe_every_n_epochs`` cadence.
        - Aggregate-recall stop trigger: ``_consecutive_perfect >= window``.
        - ``gradient_checkpointing_enable()`` called after probes and before
          returning (callback-exempt re-enable per §14.3).
        - Pause file checked AFTER probe completes and epoch_log is persisted.
        """
        # Lazy import here to allow unit-testing without GPU.
        from experiments.utils.recall_diagnostics import (
            per_field_split_counts,
            update_first_perfect_log,
        )
        from experiments.utils.test_harness import evaluate_indexed_recall

        epoch = int(round(state.epoch))

        # Guard: HF Trainer may fire on_epoch_end twice for the same epoch.
        if epoch <= self._last_epoch:
            return
        self._last_epoch = epoch

        policy = self._policy

        # Determine whether to probe this epoch.
        should_probe = epoch >= policy.probe_from_epoch and (
            (epoch - policy.probe_from_epoch) % policy.probe_every_n_epochs == 0
        )

        if not should_probe:
            _write_progress(
                self._progress_path,
                phase=self._phase_name,
                keys=len(self._target_keyed),
                epoch=epoch,
                total_epochs=self._num_epochs,
                cycle_started_at=self._cycle_started_at,
                label="non-probe-epoch",
            )
            if self._pause_file.exists():
                logger.warning(
                    "Pause file detected at epoch %d (non-probe) — signalling stop.",
                    epoch,
                )
                control.should_training_stop = True
            return

        # --- Fill probe ---
        fill = evaluate_indexed_recall(
            self._model,
            self._tokenizer,
            self._target_keyed,
            self._target_registry,
            adapter_name=self._adapter_name,
        )

        # --- Optional retention probe ---
        retention = None
        if self._retention_keyed is not None and self._retention_registry is not None:
            retention = evaluate_indexed_recall(
                self._model,
                self._tokenizer,
                self._retention_keyed,
                self._retention_registry,
                adapter_name=self._adapter_name,
            )

        # Re-enable gradient checkpointing after ALL probes (§14.3 callback exemption).
        # evaluate_indexed_recall disables it; we re-enable once here before returning.
        self._model.gradient_checkpointing_enable()

        # --- Diagnostics ---
        split = per_field_split_counts(fill["per_key"])
        update_first_perfect_log(fill["per_key"], self._first_perfect_log, epoch)

        # --- Epoch log entry ---
        entry: dict = {
            "epoch": epoch,
            "fill": {
                "exact_count": fill["exact_count"],
                "total": fill["total"],
                "rate": fill["rate"],
                "mean_confidence": fill["mean_confidence"],
            },
            "q_a_split": split,
        }
        if retention is not None:
            entry["retention"] = {
                "exact_count": retention["exact_count"],
                "total": retention["total"],
                "rate": retention["rate"],
                "mean_confidence": retention["mean_confidence"],
            }
        self._state.epoch_log.append(entry)

        logger.info(
            "  epoch %d: fill %d/%d (%.3f) split={both=%d q_only=%d a_only=%d neither=%d}%s",
            epoch,
            fill["exact_count"],
            fill["total"],
            fill["rate"],
            split["both"],
            split["q_only"],
            split["a_only"],
            split["neither"],
            (
                f" | retention {retention['exact_count']}/{retention['total']}"
                f" ({retention['rate']:.3f})"
                if retention is not None
                else ""
            ),
        )

        # --- Write epoch_log.json incrementally ---
        _safe_write_json(self._epoch_log_path, self._state.epoch_log)

        # --- Write first_perfect_log.json incrementally ---
        _safe_write_json(self._first_perfect_log_path, self._first_perfect_log)

        # --- Progress.json ---
        _write_progress(
            self._progress_path,
            phase=self._phase_name,
            keys=len(self._target_keyed),
            epoch=epoch,
            total_epochs=self._num_epochs,
            cycle_started_at=self._cycle_started_at,
            fill_recall=fill["exact_count"],
            fill_total=fill["total"],
            fill_rate=fill["rate"],
        )

        # --- First-perfect / stable-perfect tracking (on fill set) ---
        is_perfect = fill["exact_count"] == fill["total"] and fill["total"] > 0

        if is_perfect and self._state.first_perfect_epoch is None:
            self._state.first_perfect_epoch = epoch
            logger.info("  first_perfect_epoch = %d", epoch)

        if is_perfect:
            self._consecutive_perfect += 1
        else:
            self._consecutive_perfect = 0

        if self._state.stable_perfect_epoch is None and self._consecutive_perfect >= policy.window:
            self._state.stable_perfect_epoch = epoch
            logger.info("  stable_perfect_epoch = %d", epoch)

        # --- Aggregate-recall early-stop trigger (lock-in §13.3) ---
        if (
            not self._signaled_stop
            and epoch >= policy.signal_from_epoch
            and self._consecutive_perfect >= policy.window
        ):
            control.should_training_stop = True
            self._signaled_stop = True
            self._state.stop_epoch = epoch
            logger.info(
                "  RecallEarlyStopCallback: stop triggered at epoch %d "
                "(consecutive_perfect=%d >= window=%d, signal_from_epoch=%d)",
                epoch,
                self._consecutive_perfect,
                policy.window,
                policy.signal_from_epoch,
            )

        # --- Pause check (after log is persisted) ---
        if self._pause_file.exists():
            logger.warning(
                "Pause file detected at epoch %d — signalling stop. Use tresume 14 to continue.",
                epoch,
            )
            control.should_training_stop = True

    def get_first_perfect_log(self) -> dict:
        """Return the accumulated first-perfect log dict.

        Keys are ``key`` strings; values are dicts with
        ``epoch_first_perfect`` (int or None) and related fields.
        """
        return self._first_perfect_log


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _safe_write_json(path: Path, data: object) -> None:
    """Write ``data`` as JSON to ``path``, creating parent dirs as needed.

    Failures are logged as warnings; they do not abort training.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
    except OSError as exc:
        logger.warning("JSON write failed (%s): %s", path, exc)


def _write_progress(
    path: Path | None,
    *,
    phase: str,
    keys: int,
    epoch: int,
    total_epochs: int,
    cycle_started_at: int,
    label: str | None = None,
    fill_recall: int | None = None,
    fill_total: int | None = None,
    fill_rate: float | None = None,
) -> None:
    """Write a ``progress.json`` snapshot (for tstatus / training-control.sh).

    Args:
        path: Destination path.  If None, skip silently.
        phase: Phase label.
        keys: Number of fill keys being trained.
        epoch: Current epoch.
        total_epochs: Total budget.
        cycle_started_at: Unix timestamp when training started.
        label: Optional non-probe-epoch label.
        fill_recall / fill_total / fill_rate: Fill recall snapshot if probed.
    """
    if path is None:
        return
    progress: dict = {
        "phase": phase,
        "keys": keys,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "epoch_offset": 0,
        "cycle_started_at": cycle_started_at,
    }
    if label is not None:
        progress["label"] = label
    if fill_recall is not None:
        progress["fill_recall"] = fill_recall
        progress["fill_total"] = fill_total
        progress["fill_rate"] = fill_rate
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(progress, indent=2))
    except OSError as exc:
        logger.warning("progress.json write failed (%s): %s", path, exc)
