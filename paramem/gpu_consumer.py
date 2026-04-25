"""ParaMem-specific GPU consumer for the gpu_guard arbitration layer.

Owns ALL paramem-specific GPU negotiation logic:

- Identifying the server process via systemd MainPID, TCP listener, or
  /proc cmdline markers.
- Sending SIGUSR1 to switch the server to cloud-only mode.
- Polling /status to detect when the server is idle (cloud-only shortcut).
- Stamping / clearing PARAMEM_HOLD_* systemd env vars so pstatus and
  auto-reclaim can track the active ML hold.
- Human-readable prompt strings for the interactive approval flow.

The ``httpx`` import is here, not in ``gpu_guard`` core.  The core is
paramem-agnostic; keeping httpx out of it avoids pulling the server stack
into unrelated GPU consumers.

Module-level ``consumer = ParamemServerConsumer(port=8420)`` is the
pre-instantiated object used by the paramem shim and by ``with-gpu
--consumer paramem.gpu_consumer:consumer``.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (mirrors original gpu_guard.py)
# ---------------------------------------------------------------------------

_SERVER_UNIT = "paramem-server"
_SERVER_CMDLINE_MARKERS = ("paramem.server", "paramem/server/app")

_LSOF = (
    shutil.which("lsof")
    or shutil.which("lsof", path="/usr/bin:/usr/sbin:/bin:/usr/local/bin")
    or "lsof"
)

VRAM_CLEAR_TIMEOUT = 30
VRAM_POLL_INTERVAL = 2


# ---------------------------------------------------------------------------
# Detection helpers (paramem-specific; re-exported for test patching)
# ---------------------------------------------------------------------------


def _systemd_main_pid(unit: str = _SERVER_UNIT) -> int | None:
    """Return the MainPID reported by ``systemctl --user`` for ``unit``.

    Returns None when the unit is stopped, when systemctl is unavailable,
    or when the output cannot be parsed.
    """
    try:
        result = subprocess.run(
            ["systemctl", "--user", "show", "-p", "MainPID", "--value", unit],
            capture_output=True,
            text=True,
            timeout=5,
        )
        pid = int(result.stdout.strip())
        return pid if pid > 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return None


def _listener_pid(port: int) -> int | None:
    """Return the PID listening on ``port`` (TCP LISTEN only), or None.

    ``-sTCP:LISTEN`` filters established client-side sockets from the result.
    """
    try:
        result = subprocess.run(
            [_LSOF, "-i", f"TCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        pids = [int(p.strip()) for p in result.stdout.strip().split("\n") if p.strip()]
        return pids[0] if pids else None
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return None


def _pid_cmdline(pid: int) -> str:
    """Return the command line of ``pid`` as a space-separated string, or ''."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            raw = fh.read()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def _is_paramem_server_cmdline(cmdline: str) -> bool:
    """Return True if ``cmdline`` matches any paramem server marker."""
    return any(marker in cmdline for marker in _SERVER_CMDLINE_MARKERS)


def _identify_server_pids(external_pids: list[int], port: int) -> list[int]:
    """Classify ``external_pids`` → sorted PIDs recognised as the paramem server.

    Union of three sources (priority order):
      1. systemd --user MainPID for ``paramem-server``.
      2. TCP listener on ``port`` (listening socket only, via lsof).
      3. /proc/<pid>/cmdline matches any of ``_SERVER_CMDLINE_MARKERS``.

    The cmdline backstop catches the server whenever (1) and (2) both miss —
    startup race before port bind, lsof missing from PATH, unmanaged systemd,
    or a SIGUSR1 transition window.

    Returns only PIDs that are in ``external_pids``.
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


def _server_is_cloud_only(port: int) -> bool:
    """Return True if the server /status reports mode=='cloud-only'.

    Transient unreachability (e.g. during a SIGUSR1 transition) is logged at
    debug level and returns False; the polling caller retries.
    """
    try:
        import httpx

        resp = httpx.get(f"http://localhost:{port}/status", timeout=3)
        return resp.json().get("mode") == "cloud-only"
    except Exception as exc:
        logger.debug("server /status check failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Consumer implementation
# ---------------------------------------------------------------------------


class ParamemServerConsumer:
    """Consumer adaptor for the ParaMem server process.

    Implements the ``gpu_guard.Consumer`` protocol so the gpu_guard arbitrator
    can negotiate with the server without knowing any paramem internals.

    Detection uses three sources in priority order (systemd MainPID, TCP
    listener, /proc cmdline) so classification is robust across start-up
    races and unmanaged deployments.

    Args:
        port: The TCP port the server listens on (default 8420).
    """

    name: str = "paramem-server"
    default_priority: int = 5
    non_evictable_without_confirm: bool = False

    def __init__(self, port: int = 8420) -> None:
        self._port = port

    def find_pids(self, candidate_pids: list[int]) -> list[int]:
        """Return candidate PIDs that belong to the ParaMem server.

        Args:
            candidate_pids: PIDs currently using the GPU.

        Returns:
            Sorted list of PIDs recognised as the server.
        """
        return _identify_server_pids(candidate_pids, self._port)

    def is_idle(self) -> bool:
        """Return True if the server is already in cloud-only mode."""
        return _server_is_cloud_only(self._port)

    def request_release(self, pid: int) -> None:
        """Send SIGUSR1 to the server, asking it to unload the model.

        Args:
            pid: The server's primary process PID.
        """
        logger.debug("Sending SIGUSR1 to ParaMem server (PID %d)", pid)
        try:
            os.kill(pid, signal.SIGUSR1)
        except ProcessLookupError:
            logger.debug("Server PID %d already gone when sending SIGUSR1", pid)

    def wait_for_idle(self, pid: int, timeout: int) -> bool:
        """Poll /status until cloud-only or timeout.

        A server that has unloaded its model keeps a CUDA context in
        nvidia-smi --query-compute-apps but uses no real VRAM.  Checking
        /status mode=='cloud-only' is the authoritative idle signal.

        Args:
            pid: The server's primary PID (unused; idle check is via HTTP).
            timeout: Maximum seconds to wait.

        Returns:
            True if the server became cloud-only within ``timeout``.
        """
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if _server_is_cloud_only(self._port):
                return True
            time.sleep(VRAM_POLL_INTERVAL)
        return False

    def describe(self, pid: int) -> str:
        """Return a human-readable prompt string for the server process.

        Args:
            pid: The server's primary PID.
        """
        return (
            f"ParaMem server (PID {pid}) is using the GPU."
            "\n  It will switch to cloud-only mode during this workload."
        )

    def on_acquired(self, own_pid: int, argv: list[str]) -> None:
        """Stamp PARAMEM_HOLD_* systemd env vars so pstatus and auto-reclaim work.

        PARAMEM_EXTRA_ARGS=--defer-model is set by training-control.sh before
        this process launches; we add the identity fields (PID, timestamp, hint).
        Cleared in ``on_released``.

        Args:
            own_pid: PID of the acquiring ML process.
            argv: sys.argv of the acquiring process.
        """
        from paramem.utils.gpu_hold import format_cmd_hint

        hint = format_cmd_hint(argv)
        stamp_args = [
            "systemctl",
            "--user",
            "set-environment",
            f"PARAMEM_HOLD_PID={own_pid}",
            f"PARAMEM_HOLD_STARTED_AT={int(time.time())}",
        ]
        if hint:
            stamp_args.append(f"PARAMEM_HOLD_CMD={hint}")
        try:
            subprocess.run(stamp_args, check=False, capture_output=True, timeout=5)
        except Exception:
            pass

    def on_released(self) -> None:
        """Clear PARAMEM_HOLD_* and PARAMEM_EXTRA_ARGS from the systemd environment.

        Training has stopped.  Drop the defer-model flag before the next
        auto-reclaim tick loads the model in local mode.  SIGKILL skips this
        hook; auto-reclaim's GPU-free check handles that case independently.
        """
        try:
            subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "unset-environment",
                    "PARAMEM_EXTRA_ARGS",
                    "PARAMEM_HOLD_PID",
                    "PARAMEM_HOLD_STARTED_AT",
                    "PARAMEM_HOLD_CMD",
                ],
                check=False,
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass


# Module-level pre-instantiated consumer for ``with-gpu --consumer`` and the shim.
consumer = ParamemServerConsumer(port=8420)
