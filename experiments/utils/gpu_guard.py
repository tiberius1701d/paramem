"""GPU guard shim — backward-compatible wrapper around gpu_guard.

All real logic lives in ``gpu_guard`` (lab-tools).  Paramem-specific logic
lives in ``paramem.gpu_consumer``.

This module:
  1. Registers the ParaMem server consumer and notifier as module-level defaults.
  2. Re-exports the public API so callers need no changes.
  3. Re-exports AND locally shadows test-patched internal names so that
     ``patch.object(gpu_guard, "_systemd_main_pid", ...)`` in existing tests
     still takes effect.

Test-patching discipline: the existing tests do::

    patch.object(gpu_guard, "_systemd_main_pid", ...)

``patch.object`` replaces the attribute on *this* module object.  For that
replacement to be visible to the functions that use these names, those
functions must resolve the names through this module's namespace.  The local
shim functions defined below (``_identify_server_pids``, ``_SleepInhibitor``)
use module-level lookups so the patches work.

Patched internals exposed here (8 names — tracked by
tests/test_gpu_guard_shim_completeness.py):
  _systemd_main_pid          (from paramem.gpu_consumer, shadowed locally)
  _listener_pid              (from paramem.gpu_consumer, shadowed locally)
  _pid_cmdline               (from paramem.gpu_consumer, shadowed locally)
  _is_paramem_server_cmdline (from paramem.gpu_consumer, shadowed locally)
  _identify_server_pids      (shim-local wrapper using shim-level names)
  _SleepInhibitor            (shim-local subclass using shim-level names)
  notify_ml                  (from paramem.utils.notify)
  _get_gpu_pids              (from gpu_guard._core)
"""

from __future__ import annotations

import shutil
import subprocess

# ---------------------------------------------------------------------------
# Core public API from gpu_guard
# ---------------------------------------------------------------------------
from gpu_guard import (  # noqa: F401
    GPUAcquireError,
    _get_gpu_pids,
    acquire_gpu,
    add_default_consumer,
    check_gpu,
    clear_default_consumers,
    release_consumer_gpu,
    set_default_inhibitor,
    set_default_notifier,
)
from gpu_guard._core import _GPUGuard  # noqa: F401  (imported by test mocking)
from gpu_guard.inhibitor import _is_wsl2  # noqa: F401 (patched by test_sleep_inhibitor)

# ---------------------------------------------------------------------------
# Paramem-specific internals — imported AND locally shadowed for patch support
# ---------------------------------------------------------------------------
from paramem.gpu_consumer import (
    _is_paramem_server_cmdline,
    _listener_pid,
    _pid_cmdline,
    _systemd_main_pid,
)
from paramem.gpu_consumer import (
    consumer as _paramem_consumer,
)
from paramem.utils.notify import (  # noqa: F401
    ML_FINISHED,
    ML_PAUSED,
    ML_RESUMED,
    ML_STARTED,
    notify_ml,
)

# Re-export the paramem-specific names at shim level so patch.object works.
_is_paramem_server_cmdline = _is_paramem_server_cmdline  # noqa: F811
_listener_pid = _listener_pid  # noqa: F811
_pid_cmdline = _pid_cmdline  # noqa: F811
_systemd_main_pid = _systemd_main_pid  # noqa: F811


def _identify_server_pids(external_pids: list[int], port: int) -> list[int]:
    """Shim-local wrapper so patch.object(gpu_guard, '_systemd_main_pid') works.

    Calls ``_systemd_main_pid``, ``_listener_pid``, and ``_pid_cmdline``
    through this module's namespace so tests that patch those names at the
    shim level see the patched versions.
    """
    candidates: set[int] = set()
    pid = _systemd_main_pid()
    if pid is not None:
        candidates.add(pid)
    pid = _listener_pid(port)
    if pid is not None:
        candidates.add(pid)
    for p in external_pids:
        if p in candidates:
            continue
        if _is_paramem_server_cmdline(_pid_cmdline(p)):
            candidates.add(p)
    return sorted(candidates & set(external_pids))


# ---------------------------------------------------------------------------
# _SleepInhibitor shim-local subclass
#
# Tests patch ``experiments.utils.gpu_guard._is_wsl2``,
# ``experiments.utils.gpu_guard.subprocess.Popen``, and
# ``experiments.utils.gpu_guard.atexit.register``.  For those patches to
# affect _SleepInhibitor, its methods must reference the shim-level names
# (not gpu_guard.inhibitor's namespace).  The subclass below overrides
# start() and stop() to use shim-level lookups.
# ---------------------------------------------------------------------------


class _SleepInhibitor:
    """Shim-local sleep inhibitor.

    Mirrors the original _SleepInhibitor API so test patches on the shim
    module's ``_is_wsl2``, ``subprocess.Popen``, and ``atexit`` names work
    without any edits to the existing tests.
    """

    _PS_CMD = (
        "[System.Threading.Thread]::CurrentThread"
        ".SetApartmentState([System.Threading.ApartmentState]::STA); "
        "Add-Type -MemberDefinition '"
        '[DllImport("kernel32.dll")]'
        " public static extern uint SetThreadExecutionState(uint esFlags);"
        "' -Name 'Kernel32' -Namespace 'Win32' 2>$null; "
        "[Win32.Kernel32]::SetThreadExecutionState(0x80000001) | Out-Null; "
        "[System.Threading.Thread]::Sleep([System.Threading.Timeout]::Infinite)"
    )

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._atexit_registered: bool = False

    def start(self) -> None:
        """Start the sleep inhibitor, resolving names through the shim module."""
        if self._proc is not None:
            return

        # Use shim-level _is_wsl2 so tests can patch it.
        import experiments.utils.gpu_guard as _shim

        if not _shim._is_wsl2():
            return

        ps_exe = shutil.which("powershell.exe") or "powershell.exe"
        try:
            # Use shim-level subprocess.Popen so tests can patch it.
            self._proc = _shim.subprocess.Popen(
                [ps_exe, "-NoProfile", "-NonInteractive", "-Command", self._PS_CMD],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, OSError):
            self._proc = None
            return

        if not self._atexit_registered:
            # Use shim-level atexit so tests can patch it.
            _shim.atexit.register(self.stop)
            self._atexit_registered = True

    def stop(self) -> None:
        """Stop the sleep inhibitor."""
        proc = self._proc
        if proc is None:
            return

        self._proc = None

        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        except (ChildProcessError, OSError):
            pass

        if self._atexit_registered:
            try:
                import experiments.utils.gpu_guard as _shim

                _shim.atexit.unregister(self.stop)
            except Exception:
                pass
            self._atexit_registered = False


# ---------------------------------------------------------------------------
# Paramem notifier — wraps notify_ml for gpu_guard's Notifier protocol
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Module-level registration — runs at import time
# ---------------------------------------------------------------------------
add_default_consumer(_paramem_consumer)
set_default_notifier(_ParamemNotifier())

# ---------------------------------------------------------------------------
# Thin wrappers preserving original shim-facing API
# ---------------------------------------------------------------------------


def release_server_gpu(
    port: int = 8420,
    timeout: int = 30,
) -> bool:
    """Send SIGUSR1 to the server and wait for VRAM to clear.

    Thin wrapper around ``release_consumer_gpu`` for backward compatibility.

    Args:
        port: Server port (unused — consumer is pre-configured at port 8420).
        timeout: Maximum seconds to wait for VRAM to clear.

    Returns:
        True if GPU was released, False on timeout.
    """
    return release_consumer_gpu(_paramem_consumer, timeout=timeout)


def notify_paused() -> None:
    """Notify that an ML workload has paused (e.g. tpause)."""
    notify_ml(ML_PAUSED)


def notify_resumed() -> None:
    """Notify that an ML workload has resumed (e.g. tresume)."""
    notify_ml(ML_RESUMED)
