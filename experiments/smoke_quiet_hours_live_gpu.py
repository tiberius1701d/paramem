"""Live GPU smoke: quiet-hours policy shapes GPU power draw.

Proves the `quiet_hours_mode` gate produces a measurable wattage delta
on real hardware. Uses GPU power.draw (via nvidia-smi) as the observable
— no model load required.

Design:
  Two back-to-back phases of sustained matmul load, each ~20s. Between
  matmul steps the smoke inlines `BackgroundTrainer._thermal_throttle(step)`
  at every step boundary (mirroring what the HF Trainer callback does
  during real training). The workload runs on the same thread that
  invokes the throttle, so when the throttle sleeps the matmul pauses
  too — and GPU power falls to idle. Between phases we cool down so
  phase B doesn't inherit phase A's thermal state.

  * Phase A — quiet_hours_mode="always_off":
        Gate short-circuits → throttle is a no-op → sustained high draw.
  * Phase B — quiet_hours_mode="always_on", temp_limit slightly above
    idle so the throttle fires within seconds:
        Matmul heats past the limit → throttle releases GPU + sleeps
        → power drops to idle → re-acquire → heat climbs → throttle
        fires again. Net: lower mean draw over the phase.

Assertion: mean_off - mean_on ≥ 10W. Observed on RTX 5070 Laptop (60W TGP):
sustained 4096×4096 fp32 matmul pushes draw to ~65-75W (up to ~82W peak);
thermal throttling drops it to ~8-12W idle during the sleep windows, so the
delta is large and stable (measured ≈57W on first pass).

Pre-conditions:
  * `paramem-server.service` STOPPED — we need the GPU.

Usage:
    conda activate paramem
    python experiments/smoke_quiet_hours_live_gpu.py
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch  # noqa: E402

from experiments.utils.gpu_guard import acquire_gpu as reserve_gpu  # noqa: E402
from experiments.utils.test_harness import setup_logging  # noqa: E402
from paramem.server.background_trainer import BackgroundTrainer  # noqa: E402
from paramem.server.gpu_lock import acquire_gpu, release_gpu  # noqa: E402

setup_logging()
logger = logging.getLogger("smoke_quiet_hours_live_gpu")

OUTPUT_DIR = project_root / "outputs" / "smoke_quiet_hours_live_gpu"

# Tunables.
MATMUL_DIM = 4096  # 4096×4096 fp32 → ~15ms/op on RTX 5070, ~70W sustained
PHASE_SECONDS = 20  # sustained load per phase
COOLDOWN_SECONDS = 25  # between phases: give the GPU time to shed heat
THROTTLE_CHECK_EVERY = 5  # invoke _thermal_throttle every N matmul steps
POLL_INTERVAL = 0.25  # wattage sample period
TEMP_LIMIT = 52  # just above baseline idle — throttle fires within seconds of load
MIN_DELTA_WATTS = 10.0  # assertion margin


def _gpu_power() -> float | None:
    """Current power draw in watts via nvidia-smi, or None on failure."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            timeout=2,
        )
        return float(out.decode().strip().split("\n")[0])
    except Exception:
        return None


def _gpu_temp() -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=2,
        )
        return int(out.decode().strip().split("\n")[0])
    except Exception:
        return None


class _PowerPoller(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.samples: list[tuple[float, float]] = []  # (t_since_start, watts)
        self._stop_event = threading.Event()
        self._started_at = time.time()

    def run(self) -> None:
        while not self._stop_event.is_set():
            w = _gpu_power()
            if w is not None:
                self.samples.append((time.time() - self._started_at, w))
            time.sleep(POLL_INTERVAL)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=2)


def _run_phase(label: str, mode: str) -> list[float]:
    """Run sustained matmul load with inline throttle callbacks for PHASE_SECONDS.

    Returns the list of power.draw samples captured during this phase.
    """
    bt = BackgroundTrainer(
        model=MagicMock(),
        tokenizer=MagicMock(),
        training_config=MagicMock(),
        temp_limit=TEMP_LIMIT,
        temp_check_interval=THROTTLE_CHECK_EVERY,
        quiet_hours_mode=mode,
    )

    poller = _PowerPoller()
    poller.start()

    logger.info(
        "Phase %s: mode=%s temp_limit=%d°C duration=%ds — temp before=%s°C",
        label,
        mode,
        TEMP_LIMIT,
        PHASE_SECONDS,
        _gpu_temp(),
    )

    acquire_gpu()
    x = torch.randn(MATMUL_DIM, MATMUL_DIM, device="cuda", dtype=torch.float32)
    deadline = time.time() + PHASE_SECONDS
    step = 0
    try:
        while time.time() < deadline:
            # Busy matmul → drives power draw ~40-55W.
            x = x @ x.T
            x = x / (x.abs().max() + 1e-6)  # keep values bounded
            torch.cuda.synchronize()
            step += 1
            # Inline throttle — under always_on it will pause us when hot.
            bt._thermal_throttle(step)
    finally:
        release_gpu()
        poller.stop()

    logger.info(
        "Phase %s: steps=%d samples=%d temp after=%s°C",
        label,
        step,
        len(poller.samples),
        _gpu_temp(),
    )
    # Return just the wattage values. Timestamps kept in the poller if needed.
    return [w for _, w in poller.samples]


def _cooldown() -> None:
    target = 50
    deadline = time.time() + COOLDOWN_SECONDS
    while time.time() < deadline:
        t = _gpu_temp()
        if t is not None and t <= target:
            logger.info("Cooldown: GPU at %d°C ≤ %d°C — proceeding.", t, target)
            return
        time.sleep(1)
    logger.info("Cooldown: hit %ds budget, temp=%s°C — proceeding anyway.", COOLDOWN_SECONDS, t)


def main() -> int:
    reserve_gpu(interactive=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        logger.error("CUDA not available — aborting.")
        return 3

    logger.info("Baseline GPU power: %.1fW temp=%s°C", _gpu_power() or 0.0, _gpu_temp())

    samples_off = _run_phase("A", "always_off")
    _cooldown()
    samples_on = _run_phase("B", "always_on")

    mean_off = statistics.mean(samples_off) if samples_off else 0.0
    mean_on = statistics.mean(samples_on) if samples_on else 0.0
    median_off = statistics.median(samples_off) if samples_off else 0.0
    median_on = statistics.median(samples_on) if samples_on else 0.0
    delta = mean_off - mean_on

    summary = {
        "phase_A_always_off": {
            "samples": len(samples_off),
            "mean_watts": round(mean_off, 2),
            "median_watts": round(median_off, 2),
            "min_watts": round(min(samples_off), 2) if samples_off else 0.0,
            "max_watts": round(max(samples_off), 2) if samples_off else 0.0,
        },
        "phase_B_always_on": {
            "samples": len(samples_on),
            "mean_watts": round(mean_on, 2),
            "median_watts": round(median_on, 2),
            "min_watts": round(min(samples_on), 2) if samples_on else 0.0,
            "max_watts": round(max(samples_on), 2) if samples_on else 0.0,
        },
        "delta_mean_watts": round(delta, 2),
        "min_delta_required": MIN_DELTA_WATTS,
        "passed": delta >= MIN_DELTA_WATTS,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Summary:\n%s", json.dumps(summary, indent=2))

    if not summary["passed"]:
        logger.error(
            "FAIL: delta=%.1fW below required %.1fW — throttle gate not shaping power draw.",
            delta,
            MIN_DELTA_WATTS,
        )
        return 1
    logger.info(
        "PASS: always_off=%.1fW vs always_on=%.1fW (Δ=%.1fW) — quiet-hours gate works live.",
        mean_off,
        mean_on,
        delta,
    )
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        logger.exception("Quiet-hours GPU smoke crashed.")
        rc = 2
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    sys.exit(rc)
