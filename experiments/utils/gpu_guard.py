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
resolves the token via :func:`paramem.cli.http_client.resolve_token` — the
one credential resolver the CLI itself uses (ambient env
``PARAMEM_API_TOKEN``, then the per-secret file, then the repo ``.env``) —
and writes it into ``GPU_GUARD_HTTP_BEARER`` at import time so every
experiment that imports :func:`acquire_gpu` from this wrapper automatically
authenticates against the paramem server — no manual ``export`` required.

``resolve_token()`` returns ``None`` when no token is available anywhere,
in which case ``GPU_GUARD_HTTP_BEARER`` is left unset (gpu_guard omits the
``Authorization`` header on that path). ``PARAMEM_API_TOKEN`` is **never**
written to ``os.environ`` by this module.
"""

from __future__ import annotations

import os

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

from paramem.cli.http_client import resolve_token
from paramem.utils.gpu_consumer import adapter as _paramem_env_stamp_adapter
from paramem.utils.notify import (  # noqa: F401
    ML_FINISHED,
    ML_PAUSED,
    ML_RESUMED,
    ML_STARTED,
    notify_ml,
)

# Resolve and inject GPU_GUARD_HTTP_BEARER before registering consumers so
# that gpu_guard's HTTP primitives (which read the var lazily at request time)
# already see the token when the first acquire_gpu() fires. The one
# credential resolver — paramem.cli.http_client.resolve_token — is also the
# CLI's own outbound bearer resolution, so there is no second resolution
# order to keep in sync.
_bearer = resolve_token()
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
