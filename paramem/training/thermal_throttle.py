"""Thermal throttle as a config-driven HF TrainerCallback.

Pauses sustained GPU activity during the configured quiet-hours window for
fan-noise control. Sleeps in place so total GPU utilization actually drops
— does NOT touch the PA inference reservation lock (heat depends on
activity, not on lock ownership). Exits early when GPU cools below the
limit, the quiet-hours window ends, a shutdown signal arrives, or a PA
conversation is in progress (latency protection).

Independent of training driver: ``paramem.training.trainer.train_adapter``
installs the callback when a ``ThermalPolicy`` is supplied.

Default-OFF discipline. ``ThermalPolicy.from_consolidation_config`` returns
``None`` when ``training_temp_limit <= 0`` (the default for
``ConsolidationConfig``). Callers that don't pass a non-zero limit get
``thermal_policy=None`` and the callback is never installed — experiments and
overnight runs are unaffected by construction. Live-server-only.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Callable

from transformers import TrainerCallback

if TYPE_CHECKING:
    from paramem.server.config import ConsolidationConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThermalPolicy:
    """Quiet-hours-gated thermal throttle policy.

    ``temp_limit`` is the GPU °C ceiling above which training pauses;
    ``check_interval`` is the step cadence at which the throttle inspects
    temperature; the three quiet-hours fields gate when the throttle is
    allowed to fire (mode semantics in ``ConsolidationScheduleConfig``).
    """

    temp_limit: int
    check_interval: int
    quiet_hours_mode: str
    quiet_hours_start: str
    quiet_hours_end: str

    @classmethod
    def from_consolidation_config(cls, cfg: "ConsolidationConfig") -> "ThermalPolicy | None":
        """Build a policy from a ``ConsolidationConfig``, or ``None`` when disabled.

        Returns ``None`` when ``cfg.training_temp_limit <= 0`` (the dataclass
        default). Live-server deployments set a positive limit in
        ``server.yaml``; experiment scripts and tests that don't override the
        default get ``None`` — and ``train_adapter`` skips the callback install.
        """
        if cfg.training_temp_limit <= 0:
            return None
        return cls(
            temp_limit=cfg.training_temp_limit,
            check_interval=cfg.training_temp_check_interval,
            quiet_hours_mode=cfg.quiet_hours_mode,
            quiet_hours_start=cfg.quiet_hours_start,
            quiet_hours_end=cfg.quiet_hours_end,
        )


def is_thermal_policy_active(
    mode: str,
    start: str,
    end: str,
    now: datetime | None = None,
) -> bool:
    """Pure predicate: is the thermal throttle active under this quiet-hours policy?

    Re-exported by ``paramem.server.background_trainer`` so the ``/status``
    endpoint can report policy state without a live trainer. Mode semantics in
    ``ConsolidationScheduleConfig``. Invalid windows in ``auto`` mode fall
    back to ``True`` (prefer-silence default).
    """
    if mode == "always_off":
        return False
    if mode == "always_on":
        return True
    # mode == "auto"
    try:
        sh, sm = (int(x) for x in start.split(":"))
        eh, em = (int(x) for x in end.split(":"))
    except Exception:
        return True
    t = (now or datetime.now()).time()
    cur = t.hour * 60 + t.minute
    s = sh * 60 + sm
    e = eh * 60 + em
    if s == e:
        return True
    if s < e:
        return s <= cur < e
    return cur >= s or cur < e


def _gpu_temp() -> int | None:
    """Read current GPU temperature via nvidia-smi. Returns None on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return None


def wait_for_cooldown(
    threshold_c: int,
    max_wait_s: int,
    poll_s: int = 5,
    *,
    label: str = "",
) -> int | None:
    """Block until GPU temp <= threshold_c, bounded by max_wait_s.

    Mirrors gpu-cooldown.sh:wait_for_cooldown (gpu-cooldown.sh:218) using the
    same _gpu_temp() reader the fold throttle uses.  Distinct from the
    quiet-hours ThermalThrottleCallback (fan-noise control) — this is a
    TDR-prevention pre-task gate.

    No-op (returns immediately) when:

    * ``PARAMEM_COOLDOWN_DISABLED=1`` is set in the environment (explicit
      gate-disable knob — also usable at runtime on a host with a broken GPU
      sensor, and set by ``tests/conftest.py`` for non-gpu test runs).
    * ``threshold_c <= 0`` (gate disabled).
    * ``_gpu_temp()`` returns ``None`` (sensor unavailable — never block GPU
      work on a missing sensor; silently degrades to today's no-gate
      behaviour).
    * GPU is already at or below ``threshold_c``.

    Bounded: on timeout it logs a WARNING and returns the still-hot temp so
    the caller proceeds rather than hanging (boot SIGKILL / user-request
    latency).  Returns the last observed temp (or ``None``).

    Args:
        threshold_c: Target temperature ceiling in °C.  0 disables the gate.
        max_wait_s: Hard cap on how long to block in seconds.
        poll_s: Seconds between successive temperature reads.
        label: Optional site label for log messages (e.g. ``"preload"``,
            ``"fold"``, ``"inference"``).
    """
    if os.environ.get("PARAMEM_COOLDOWN_DISABLED") == "1":
        return None
    if threshold_c <= 0:
        return None
    temp = _gpu_temp()
    if temp is None or temp <= threshold_c:
        return temp
    waited = 0
    _tag = f" [{label}]" if label else ""
    logger.info(
        "cooldown gate%s: GPU %d°C > %d°C — waiting (cap %ds)",
        _tag,
        temp,
        threshold_c,
        max_wait_s,
    )
    while temp is not None and temp > threshold_c and waited < max_wait_s:
        time.sleep(poll_s)
        waited += poll_s
        temp = _gpu_temp()
    if temp is not None and temp > threshold_c:
        logger.warning(
            "cooldown gate%s: GPU still %d°C after %ds — proceeding",
            _tag,
            temp,
            waited,
        )
    else:
        logger.info("cooldown gate%s: GPU at %s°C — proceeding", _tag, temp)
    return temp


def _should_throttle_now(policy: ThermalPolicy, now: datetime | None = None) -> bool:
    """Whether the throttle is active right now under the policy's quiet-hours."""
    return is_thermal_policy_active(
        policy.quiet_hours_mode,
        policy.quiet_hours_start,
        policy.quiet_hours_end,
        now,
    )


class ThermalThrottleCallback(TrainerCallback):
    """Pause training when GPU temperature exceeds the policy limit.

    Sleeps in place during a quiet-hours window for fan-noise control. Does NOT
    touch the PA inference reservation lock — heat depends on actual GPU
    activity, not on lock ownership.

    The wait loop exits on any of three conditions: GPU cools below
    ``policy.temp_limit``, the quiet-hours window ends mid-wait, or a shutdown
    signal arrives (the shutdown predicate ORs the BG abort event, so /chat
    arrivals during the hot-wait break the loop cleanly).

    ``shutdown_fn`` defaults to a constant ``False`` so non-server callers
    (experiments via ``train_adapter``) don't depend on a BG-trainer-specific
    signal.  When hooks are supplied via ``TrainingHooks.on_shutdown_check``,
    that predicate already ORs the BG abort event (via
    ``BackgroundTrainer.training_hooks_for_job``), so a mid-wait abort breaks
    the loop cleanly without a separate field.
    """

    def __init__(
        self,
        policy: ThermalPolicy,
        shutdown_fn: Callable[[], bool] = lambda: False,
    ):
        self._policy = policy
        self._shutdown_fn = shutdown_fn

    def on_step_end(self, args, state, control, **kwargs):
        self._maybe_throttle(state.global_step)

    def on_epoch_end(self, args, state, control, **kwargs):
        self._maybe_throttle(state.global_step)

    def _maybe_throttle(self, global_step: int) -> None:
        if self._policy.temp_limit <= 0:
            return
        if global_step % self._policy.check_interval != 0:
            return
        if not _should_throttle_now(self._policy):
            return

        temp = _gpu_temp()
        if temp is None or temp <= self._policy.temp_limit:
            return

        logger.info(
            "Thermal throttle: GPU at %d°C (limit %d°C) — pausing at step %d for fan-noise control",
            temp,
            self._policy.temp_limit,
            global_step,
        )

        while temp is not None and temp > self._policy.temp_limit:
            if self._shutdown_fn():
                # shutdown_fn already ORs the abort event via training_hooks_for_job,
                # so an abort during a hot-wait breaks the loop cleanly here.
                return
            # Quiet-hours window may end mid-wait (e.g. 07:00 arrives) — in
            # that case we no longer care about fan noise and resume even if
            # still hot.
            if not _should_throttle_now(self._policy):
                logger.info(
                    "Thermal throttle: quiet-hours window ended at %d°C — resuming",
                    temp,
                )
                return
            time.sleep(5)
            temp = _gpu_temp()

        logger.info("Thermal throttle: GPU cooled to %d°C — resuming", temp)
