"""Recall-based early-stop policy and callback (production + experiments).

Originally implemented for Test 14 in ``experiments/utils/early_stop.py``;
lifted to ``paramem.training.early_stop`` 2026-05-06 so production
``BackgroundTrainer`` can use the same gate.  Experiment scripts continue
to import from ``experiments.utils.early_stop`` via a re-export shim.

``EarlyStopPolicy`` — dataclass encoding when probing starts, when the stop
signal can fire, and how many consecutive perfect probes are required.

``RecallEarlyStopCallback`` — HF ``TrainerCallback`` that probes fill and
(optionally) retention sets after each epoch, tracks the first-perfect log,
writes ``epoch_log.json`` incrementally, and fires
``control.should_training_stop = True`` when the aggregate-recall stop
condition is met.

``pause_file`` and ``first_perfect_log_path`` accept ``None`` (production
default; production has its own pause flow + does not need per-key logs).
``progress_path`` already accepts ``None``.  All three are guarded at every
``.exists()`` / write site.

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
    from typing import Callable

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
        extra_epochs_past_first_perfect: When set, fire the stop signal at
            ``first_perfect_epoch + extra_epochs_past_first_perfect`` instead
            of waiting for ``consecutive_perfect >= window``.  Independent of
            ``signal_from_epoch``.  None disables this alternate stop path —
            existing callers see no change.  Used by Test 16's
            depth-past-floor sweep where the first observation of 100% recall
            is the anchor and an additional fixed number of epochs is trained
            past it.

    Raises:
        ValueError: If ``signal_from_epoch < probe_from_epoch``, or if
            ``window < 1``, or if ``probe_every_n_epochs < 1``, or if
            ``extra_epochs_past_first_perfect`` is set and negative.
    """

    probe_from_epoch: int = 10
    signal_from_epoch: int = 10
    window: int = 3
    probe_every_n_epochs: int = 1
    extra_epochs_past_first_perfect: int | None = None

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
        if (
            self.extra_epochs_past_first_perfect is not None
            and self.extra_epochs_past_first_perfect < 0
        ):
            raise ValueError(
                f"extra_epochs_past_first_perfect must be >= 0, got "
                f"{self.extra_epochs_past_first_perfect}"
            )


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
        epoch_log_path: Path | None,
        first_perfect_log_path: Path | None,
        phase_name: str,
        num_epochs: int,
        pause_file: Path | None,
        retention_keyed: list[dict] | None = None,
        retention_registry: dict[str, int] | None = None,
        eval_fn: "Callable | None" = None,
    ) -> None:
        """Initialise callback state.

        Args:
            eval_fn: Probe function called for both the fill probe and the
                optional retention probe at each epoch boundary.  ``None``
                (default) selects
                :func:`paramem.training.recall_eval.evaluate_indexed_recall`
                via a lazy import at each ``on_epoch_end`` call — preserving
                patchability at ``paramem.training.recall_eval.evaluate_indexed_recall``.
        """
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
        self._eval_fn = eval_fn

        # Internal state (§6 pseudocode variable names).
        self._last_epoch: int = -1
        self._consecutive_perfect: int = 0
        self._first_perfect_log: dict[str, dict] = {}
        self._signaled_stop: bool = False
        self._cycle_started_at: int = int(time.time())

    def _rehydrate_from_disk(self) -> None:
        """Restore in-memory state from the previous run's persisted artifacts.

        Reads:
          - ``progress_path`` for ``cycle_started_at`` (anchors wall-time math
            across resumes).
          - ``epoch_log_path`` for ``state.epoch_log`` and the derived
            counters (``_last_epoch``, ``_consecutive_perfect``,
            ``state.first_perfect_epoch``, ``state.stable_perfect_epoch``).
          - ``first_perfect_log_path`` for the per-key first-perfect map.

        Failures are swallowed: a missing or malformed file is treated as a
        cold start.  ``stop_epoch`` is intentionally NOT rehydrated — if the
        prior run had stopped, we would not be resuming.
        """
        if self._progress_path is not None and self._progress_path.exists():
            try:
                existing = json.loads(self._progress_path.read_text())
                saved = existing.get("cycle_started_at")
                if isinstance(saved, (int, float)):
                    self._cycle_started_at = int(saved)
            except (OSError, json.JSONDecodeError):
                pass

        if self._epoch_log_path is not None and self._epoch_log_path.exists():
            try:
                saved_log = json.loads(self._epoch_log_path.read_text())
            except (OSError, json.JSONDecodeError):
                saved_log = None
            if isinstance(saved_log, list) and saved_log:
                self._state.epoch_log = list(saved_log)
                try:
                    self._last_epoch = int(saved_log[-1].get("epoch", -1))
                except (TypeError, ValueError):
                    self._last_epoch = -1
                consecutive = 0
                window = self._policy.window
                for entry in saved_log:
                    fill = entry.get("fill") or {}
                    exact = fill.get("exact_count", 0)
                    total = fill.get("total", 0)
                    is_perfect = total > 0 and exact == total
                    consecutive = consecutive + 1 if is_perfect else 0
                    if is_perfect and self._state.first_perfect_epoch is None:
                        try:
                            self._state.first_perfect_epoch = int(entry.get("epoch"))
                        except (TypeError, ValueError):
                            pass
                    if self._state.stable_perfect_epoch is None and consecutive >= window:
                        try:
                            self._state.stable_perfect_epoch = int(entry.get("epoch"))
                        except (TypeError, ValueError):
                            pass
                self._consecutive_perfect = consecutive

        if self._first_perfect_log_path is not None and self._first_perfect_log_path.exists():
            try:
                saved_fp = json.loads(self._first_perfect_log_path.read_text())
                if isinstance(saved_fp, dict):
                    self._first_perfect_log = saved_fp
            except (OSError, json.JSONDecodeError):
                pass

    def set_probe_adapter(self, adapter_name: str) -> None:
        """Bind the recall probe to a specific adapter slot (explicit handoff).

        Called by the staging owner (``train_adapter``) under the AD-20
        staging+promote contract: HF trains the transient ``in_training``
        slot, so the probe must measure that slot, not the caller-supplied
        production name (which holds un-promoted weights until the post-train
        promote).  The callback never infers the trained slot — the owner
        states it.  For compose/direct training the owner does not call this,
        so the constructor's production ``adapter_name`` stands.
        """
        self._adapter_name = adapter_name

    def on_train_begin(self, args, state, control, **kwargs) -> None:  # noqa: ARG002
        """(Re)initialise early-stop accumulators for this training run.

        The probe target is set explicitly by the staging owner via
        :meth:`set_probe_adapter` before training starts; this hook only
        handles accumulator state.

        On a checkpoint resume (``state.global_step > 0``), restore the
        accumulators from disk so the probe history, ``_last_epoch``, and the
        perfect-streak counters carry forward across the interruption.  On a
        fresh run (``state.global_step == 0``), start clean and remove any
        ``epoch_log.json`` / ``first_perfect_log.json`` left in this output
        dir by a prior run, so stale state is picked up neither here nor by a
        later resume of this run.  ``progress.json`` is rewritten each epoch
        and needs no cleanup.
        """
        if state.global_step > 0:
            # Genuine checkpoint resume — carry accumulators forward.
            self._rehydrate_from_disk()
        else:
            # Fresh run — unlink any stale artifacts this callback owns.
            for p in (self._epoch_log_path, self._first_perfect_log_path):
                if p is not None and p.exists():
                    p.unlink()

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
        # Lazy import for diagnostics helpers (GPU-free; no startup penalty).

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
            if self._pause_file is not None and self._pause_file.exists():
                logger.warning(
                    "Pause file detected at epoch %d (non-probe) — signalling stop.",
                    epoch,
                )
                control.should_training_stop = True
            return

        # Resolve the probe function.  ``None`` → lazy import so the default
        # production probe is looked up at call time, preserving test-suite
        # patchability of the module-level name.
        if self._eval_fn is None:
            from paramem.training.recall_eval import evaluate_indexed_recall as _probe_fn
        else:
            _probe_fn = self._eval_fn

        # --- Fill probe ---
        fill = _probe_fn(
            self._model,
            self._tokenizer,
            self._target_keyed,
            self._target_registry,
            adapter_name=self._adapter_name,
        )

        # --- Optional retention probe ---
        retention = None
        if self._retention_keyed is not None and self._retention_registry is not None:
            retention = _probe_fn(
                self._model,
                self._tokenizer,
                self._retention_keyed,
                self._retention_registry,
                adapter_name=self._adapter_name,
            )

        # Re-enable gradient checkpointing after ALL probes (§14.3 callback
        # exemption) — but only if the training run had it enabled to begin
        # with.  evaluate_indexed_recall disables it unconditionally; we
        # restore the pre-probe state from ``args.gradient_checkpointing``
        # (TrainingArguments forwards the TrainingConfig setting verbatim,
        # see paramem/training/trainer.py:203 and paramem/server/background_trainer.py:1007).
        # If the trainer was constructed with checkpointing OFF, leaving it
        # OFF after the probe matches the original training-step contract
        # and avoids the silent KV-cache breakage flagged in CLAUDE.md
        # ("HF Transformers silently disables KV cache when checkpointing
        # is active").
        if getattr(args, "gradient_checkpointing", False):
            self._model.gradient_checkpointing_enable()

        # --- Diagnostics ---
        # ``per_field_split_counts`` and ``update_first_perfect_log`` consume
        # ``question_match``/``answer_match``/``recalled`` keys that are absent
        # from entry ``per_key`` dicts (which carry
        # ``recalled_subject``/``recalled_predicate``/``recalled_object``
        # instead).  The early-stop decision (``fill["rate"]`` +
        # ``_consecutive_perfect``) is driven by ``fill["exact_count"]`` and
        # ``fill["total"]`` which are format-agnostic.  Emit the constant zero
        # q/a split shape so the epoch log schema stays stable for historical
        # consumers.
        split = {
            "both": 0,
            "q_only": 0,
            "a_only": 0,
            "neither": 0,
            "total": 0,
            "q_correct": 0,
            "a_correct": 0,
        }

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
        if self._epoch_log_path is not None:
            safe_write_json(self._epoch_log_path, self._state.epoch_log)

        # --- Write first_perfect_log.json incrementally ---
        if self._first_perfect_log_path is not None:
            safe_write_json(self._first_perfect_log_path, self._first_perfect_log)

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
        # --- Floor-relative stop trigger (extra_epochs_past_first_perfect) ---
        # Independent of the stable-perfect path above.  Fires once the first
        # observation of 100% recall is N epochs in the past, where N is the
        # policy's extra_epochs_past_first_perfect.  Stop is stamped on the
        # same state.stop_epoch field so downstream consumers (paused-marker
        # guards, done-marker writers) treat it identically to the stable-
        # perfect path.
        elif (
            not self._signaled_stop
            and policy.extra_epochs_past_first_perfect is not None
            and self._state.first_perfect_epoch is not None
            and epoch >= self._state.first_perfect_epoch + policy.extra_epochs_past_first_perfect
        ):
            control.should_training_stop = True
            self._signaled_stop = True
            self._state.stop_epoch = epoch
            logger.info(
                "  RecallEarlyStopCallback: stop triggered at epoch %d "
                "(first_perfect_epoch=%d + extra=%d)",
                epoch,
                self._state.first_perfect_epoch,
                policy.extra_epochs_past_first_perfect,
            )

        # --- Pause check (after log is persisted) ---
        if self._pause_file is not None and self._pause_file.exists():
            logger.warning(
                "Pause file detected at epoch %d — signalling stop. Use tresume to continue.",
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


def safe_write_json(path: Path, data: object) -> None:
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
