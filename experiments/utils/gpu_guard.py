"""Paramem-side gpu_guard registration.

Imports gpu_guard core, registers the env-stamp adapter and the paramem
notifier as defaults, and re-exports the public API for backward-compat
imports.

All detection / release / idle / describe logic for the paramem-server
process lives in ``~/.config/gpu-guard/config.toml`` under
``[consumers.paramem-server]``.
"""

from __future__ import annotations

from gpu_guard import (  # noqa: F401 — re-exported for legacy imports
    GPUAcquireError,
    GPUConfigMissing,
    acquire_gpu,
    add_default_consumer,
    check_gpu,
    clear_default_consumers,
    release_consumer_gpu_by_name,
    set_default_inhibitor,
    set_default_notifier,
)

from paramem.gpu_consumer import adapter as _paramem_env_stamp_adapter
from paramem.utils.notify import (  # noqa: F401
    ML_FINISHED,
    ML_PAUSED,
    ML_RESUMED,
    ML_STARTED,
    notify_ml,
)


class _ParamemNotifier:
    """Notifier that fires Windows toast notifications via paramem.utils.notify."""

    def started(self) -> None:
        """Send ML_STARTED notification."""
        notify_ml(ML_STARTED)

    def finished(self) -> None:
        """Send ML_FINISHED notification."""
        notify_ml(ML_FINISHED)

    def paused(self) -> None:
        """Send ML_PAUSED notification."""
        notify_ml(ML_PAUSED)

    def resumed(self) -> None:
        """Send ML_RESUMED notification."""
        notify_ml(ML_RESUMED)


add_default_consumer(_paramem_env_stamp_adapter)
set_default_notifier(_ParamemNotifier())


def release_server_gpu(port: int = 8420, timeout: int = 30) -> bool:
    """Ask the paramem server to release the GPU and wait for it to switch to cloud-only.

    Thin wrapper around ``release_consumer_gpu_by_name`` for backward
    compatibility with V1 callers.

    The ``port`` argument is intentionally unused — the port lives in
    ``~/.config/gpu-guard/config.toml`` under ``[consumers.paramem-server]``.
    It is kept in the signature for source compatibility with callers that
    pass it explicitly (e.g. ``paramem/utils/gpu_hold.py``).

    Args:
        port: Ignored.  Port is read from the TOML config.
        timeout: Maximum seconds to wait for the server to become idle.

    Returns:
        True if the server released the GPU, False on timeout.

    Raises:
        GPUConfigMissing: when ``~/.config/gpu-guard/config.toml`` lacks a
            ``[consumers.paramem-server]`` section.  Loud failure surfaces a
            misconfigured workstation rather than masquerading as a release
            timeout.
    """
    return release_consumer_gpu_by_name("paramem-server", timeout=timeout)


def notify_paused() -> None:
    """Notify that an ML workload has paused (e.g. tpause)."""
    notify_ml(ML_PAUSED)


def notify_resumed() -> None:
    """Notify that an ML workload has resumed (e.g. tresume)."""
    notify_ml(ML_RESUMED)
