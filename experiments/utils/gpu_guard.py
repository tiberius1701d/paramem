"""Paramem-side gpu_guard registration.

Imports gpu_guard core, registers the env-stamp adapter and the paramem
notifier as defaults, and re-exports the public API for backward-compat
imports.

All detection / release / idle / describe logic for the paramem-server
process lives in ``~/.config/gpu-guard/config.toml`` under
``[consumers.paramem-server]``.

Bearer-token wiring
-------------------
gpu_guard reads ``GPU_GUARD_HTTP_BEARER`` at request time and sends it as
``Authorization: Bearer <token>`` on HTTP release/idle calls.  This module
resolves the token from ``PARAMEM_API_TOKEN`` (ambient env or repo ``.env``)
and writes it into ``GPU_GUARD_HTTP_BEARER`` at import time so every
experiment that imports :func:`acquire_gpu` from this wrapper automatically
authenticates against the paramem server — no manual ``export`` required.

Resolution precedence (see :func:`_resolve_http_bearer`):

1. ``GPU_GUARD_HTTP_BEARER`` already set → left untouched.
2. ``PARAMEM_API_TOKEN`` in ambient env → copied.
3. ``PARAMEM_API_TOKEN`` found in repo ``.env`` → used.
4. Nothing found → ``GPU_GUARD_HTTP_BEARER`` not set (unauthenticated path).

``PARAMEM_API_TOKEN`` is **never** written to ``os.environ`` by this module.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values
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

# PROJECT_ROOT: two levels up from experiments/utils/gpu_guard.py
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _resolve_http_bearer(
    env: dict[str, str],
    dotenv_path: Path,
) -> str | None:
    """Resolve the bearer token for gpu_guard HTTP calls.

    Does NOT read or modify ``os.environ`` — callers pass a snapshot of the
    environment as *env* and receive the resolved token (or ``None``).  This
    keeps the function pure and trivially testable without import-side-effect
    hazards.

    Resolution order:

    1. ``GPU_GUARD_HTTP_BEARER`` already in *env* → return it unchanged
       (caller / launcher wins; preserves backward-compat for non-paramem use).
    2. ``PARAMEM_API_TOKEN`` in *env* → return its value.
    3. ``PARAMEM_API_TOKEN`` in *dotenv_path* (single-key extraction via
       :func:`dotenv.dotenv_values`) → return its value.
    4. Nothing found → return ``None`` (gpu_guard omits the header).

    ``PARAMEM_API_TOKEN`` is never returned; only the resolved bearer string is.

    Args:
        env: Mapping that represents the process environment to inspect
            (typically ``os.environ``, passed as a snapshot so tests can
            substitute freely without touching the real env).
        dotenv_path: Path to the ``.env`` file to consult as a last resort.
            Need not exist; a missing file is treated as empty.

    Returns:
        The bearer token string, or ``None`` when no token is available.
    """
    if "GPU_GUARD_HTTP_BEARER" in env:
        return env["GPU_GUARD_HTTP_BEARER"]

    token = env.get("PARAMEM_API_TOKEN")
    if token:
        return token

    if dotenv_path.is_file():
        token = dotenv_values(dotenv_path).get("PARAMEM_API_TOKEN")
        if token:
            return token

    return None


# Resolve and inject GPU_GUARD_HTTP_BEARER before registering consumers so
# that gpu_guard's HTTP primitives (which read the var lazily at request time)
# already see the token when the first acquire_gpu() fires.
_bearer = _resolve_http_bearer(dict(os.environ), _PROJECT_ROOT / ".env")
if _bearer is not None:
    os.environ["GPU_GUARD_HTTP_BEARER"] = _bearer
del _bearer


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
    compatibility with V1 callers — the actual release primitive (HTTP
    POST ``/gpu/release`` by default) is configured in
    ``~/.config/gpu-guard/config.toml`` under ``[consumers.paramem-server]``.

    Args:
        port: Ignored.  Kept in the signature so legacy V1 call sites
            that passed it positionally still type-check.  The port now
            lives in the TOML config.
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
