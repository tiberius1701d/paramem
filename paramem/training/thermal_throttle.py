"""Thermal throttle as a config-driven HF TrainerCallback.

Pauses training when the GPU exceeds a configured temperature limit during
the configured quiet-hours window. Releases the GPU lock while waiting so
inference / STT / TTS can proceed; re-acquires when the GPU cools or the
window ends.

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

    Releases the GPU lock while waiting so other consumers (STT/TTS/inference)
    can proceed; re-acquires when the GPU cools below ``policy.temp_limit`` or
    the quiet-hours window ends mid-wait.

    The mid-wait shutdown signal flows through ``shutdown_fn`` — defaults to a
    constant ``False`` so non-server callers (experiments via ``train_adapter``)
    don't depend on a BG-trainer-specific signal. The server-side
    ``BackgroundTrainer`` passes ``lambda: self._shutdown_requested`` so a
    shutdown during a hot-wait breaks the loop cleanly.
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

        from paramem.server.gpu_lock import acquire_gpu, release_gpu

        logger.info(
            "Thermal throttle: GPU at %d°C (limit %d°C) — releasing GPU at step %d",
            temp,
            self._policy.temp_limit,
            global_step,
        )
        release_gpu()

        while temp is not None and temp > self._policy.temp_limit:
            if self._shutdown_fn():
                acquire_gpu()
                return
            # Quiet-hours window may end mid-wait (e.g. 07:00 arrives) — in
            # that case we no longer care about fan noise and resume even if
            # still hot.
            if not _should_throttle_now(self._policy):
                logger.info(
                    "Thermal throttle: quiet-hours window ended at %d°C — resuming",
                    temp,
                )
                acquire_gpu()
                return
            time.sleep(5)
            temp = _gpu_temp()

        acquire_gpu()
        logger.info("Thermal throttle: GPU cooled to %d°C — resuming training", temp)
