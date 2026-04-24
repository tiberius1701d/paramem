"""GPU guard — safe GPU acquisition for ML workloads.

Ensures only one GPU consumer runs at a time. If the ParaMem server
holds the GPU, sends SIGUSR1 to switch it to cloud-only mode. Refuses
to start if an unknown ML process is occupying the GPU.

Usage:
    from experiments.utils.gpu_guard import acquire_gpu

    with acquire_gpu():
        # ML work here — server is in cloud-only mode
    # on exit: server auto-reclaims GPU via timer
"""

import atexit
import logging
import os
import shutil
import signal
import subprocess
import sys
import time

from paramem.utils.gpu_hold import format_cmd_hint
from paramem.utils.notify import ML_FINISHED, ML_PAUSED, ML_RESUMED, ML_STARTED, notify_ml

logger = logging.getLogger(__name__)

# Default server port — read from config if available
DEFAULT_SERVER_PORT = 8420
VRAM_CLEAR_TIMEOUT = 30  # seconds to wait for VRAM to clear
VRAM_POLL_INTERVAL = 2  # seconds between polls

# Resolve nvidia-smi at import time. On WSL2 it lives in /usr/lib/wsl/lib
# which systemd may not have on PATH; on native Linux it's typically in
# /usr/bin.  Falls back to bare name (caught by the caller's exception handler).
_NVIDIA_SMI = (
    shutil.which("nvidia-smi")
    or shutil.which("nvidia-smi", path="/usr/lib/wsl/lib:/usr/bin:/usr/local/bin")
    or "nvidia-smi"
)

# Resolve lsof the same way. systemd user services can ship a minimal PATH;
# without an explicit resolution a bare "lsof" call raises FileNotFoundError
# and silently downgrades server classification to "unknown" — the root of
# the misclassification incident that killed a pytest run.
_LSOF = (
    shutil.which("lsof")
    or shutil.which("lsof", path="/usr/bin:/usr/sbin:/bin:/usr/local/bin")
    or "lsof"
)

# Paramem server identity sources (in priority order):
#   1. systemd --user MainPID for this unit (ground truth when managed by systemd).
#   2. TCP listener on the configured port (lsof).
#   3. /proc/<pid>/cmdline matches any of these markers (backstop: catches
#      the server when systemd is unmanaged and the port hasn't been bound
#      yet / transient lsof error).
_SERVER_UNIT = "paramem-server"
_SERVER_CMDLINE_MARKERS = ("paramem.server", "paramem/server/app")


def _is_wsl2() -> bool:
    """Return True if running inside WSL2 (Windows Subsystem for Linux).

    Detection is based on the kernel version string in ``/proc/version``,
    which on WSL2 contains the substring ``microsoft`` (case-insensitive) or
    ``WSL``.  On native Linux neither substring is present, so the function
    returns False without any side effects.
    """
    try:
        with open("/proc/version") as fh:
            version_str = fh.read().lower()
        return "microsoft" in version_str or "wsl" in version_str
    except OSError:
        return False


class _SleepInhibitor:
    """Prevent Windows Modern Standby from engaging during GPU workloads.

    On WSL2, spawns a background ``powershell.exe`` child process that calls
    ``SetThreadExecutionState`` with ``ES_CONTINUOUS | ES_SYSTEM_REQUIRED``
    (``0x80000001``).  The child process blocks indefinitely so the execution
    state remains set for its lifetime.  Killing the child releases the
    inhibitor automatically — Windows resets the state when no thread holds it.

    On native Linux the class is a complete no-op: ``start()`` returns
    immediately and ``stop()`` is safe to call at any time.

    The class is *not* thread-safe — callers must ensure that ``start`` and
    ``stop`` are called from the same thread (or with external locking).
    """

    # Single-line PowerShell that sets the execution state and then sleeps
    # until killed.  ES_CONTINUOUS (0x80000000) | ES_SYSTEM_REQUIRED (0x1).
    _PS_CMD = (
        # Set STA apartment state so Add-Type works reliably.
        "[System.Threading.Thread]::CurrentThread"
        ".SetApartmentState([System.Threading.ApartmentState]::STA); "
        # P/Invoke shim for SetThreadExecutionState.
        "Add-Type -MemberDefinition '"
        '[DllImport("kernel32.dll")]'
        " public static extern uint SetThreadExecutionState(uint esFlags);"
        "' -Name 'Kernel32' -Namespace 'Win32' 2>$null; "
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED — prevent sleep while alive.
        "[Win32.Kernel32]::SetThreadExecutionState(0x80000001) | Out-Null; "
        "[System.Threading.Thread]::Sleep([System.Threading.Timeout]::Infinite)"
    )

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._atexit_registered: bool = False

    def start(self) -> None:
        """Start the sleep inhibitor.

        On WSL2, spawns a background ``powershell.exe`` process that holds
        ``ES_CONTINUOUS | ES_SYSTEM_REQUIRED`` for its lifetime.  Calling
        ``start()`` when already started is idempotent — the existing process
        is reused and no second process is spawned.

        On non-WSL2 systems the method returns immediately without any
        side effects.
        """
        if self._proc is not None:
            # Already running — idempotent.
            return

        if not _is_wsl2():
            return

        ps_exe = shutil.which("powershell.exe") or "powershell.exe"
        try:
            self._proc = subprocess.Popen(
                [ps_exe, "-NoProfile", "-NonInteractive", "-Command", self._PS_CMD],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.debug("Sleep inhibitor started (PID %d)", self._proc.pid)
        except (FileNotFoundError, OSError) as exc:
            # powershell.exe unavailable — log and continue rather than
            # blocking the GPU workload.
            logger.warning("Sleep inhibitor unavailable: %s", exc)
            self._proc = None
            return

        if not self._atexit_registered:
            atexit.register(self.stop)
            self._atexit_registered = True

    def stop(self) -> None:
        """Stop the sleep inhibitor and reap the child process.

        Safe to call even if ``start()`` was never called or returned without
        spawning a process.  After ``stop()`` returns the execution state
        inhibitor is released (Windows resets it automatically when the child
        exits).
        """
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
            logger.warning("Sleep inhibitor process did not exit within 5 s")
        except (ChildProcessError, OSError):
            pass

        if self._atexit_registered:
            try:
                atexit.unregister(self.stop)
            except Exception:
                pass
            self._atexit_registered = False

        logger.debug("Sleep inhibitor stopped")


class GPUAcquireError(RuntimeError):
    """Raised when the GPU cannot be acquired."""


def _get_gpu_pids() -> list[int]:
    """Return PIDs of processes using the GPU for compute."""
    try:
        result = subprocess.run(
            [_NVIDIA_SMI, "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return [int(p.strip()) for p in result.stdout.strip().split("\n") if p.strip()]
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return []


def _systemd_main_pid(unit: str = _SERVER_UNIT) -> int | None:
    """Return the MainPID reported by ``systemctl --user`` for ``unit``.

    systemd prints ``0`` when the unit is not running; that case returns None.
    Any lookup error (systemctl missing, timeout, non-integer output) also
    returns None so callers fall through to the next detection source.
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
    """Return the PID of the process *listening* on ``port``, or None.

    ``-sTCP:LISTEN`` filters to listeners only; without it ``lsof -i :port``
    also reports client-side established sockets on machines that happen to
    have a symmetric port number, and ``pids[0]`` would be arbitrary.
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
    """Return ``pid``'s command line as a space-separated string (or '')."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            raw = fh.read()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def _is_paramem_server_cmdline(cmdline: str) -> bool:
    """True if ``cmdline`` looks like the paramem server process."""
    return any(marker in cmdline for marker in _SERVER_CMDLINE_MARKERS)


def _get_server_pid(port: int = DEFAULT_SERVER_PORT) -> int | None:
    """Find the paramem server PID (systemd MainPID preferred, lsof fallback)."""
    pid = _systemd_main_pid()
    if pid is not None:
        return pid
    return _listener_pid(port)


def _identify_server_pids(external_pids: list[int], port: int) -> list[int]:
    """Classify ``external_pids`` → sorted PIDs recognized as the paramem server.

    Union of three sources:
      1. systemd --user MainPID for ``paramem-server``.
      2. TCP listener on ``port`` (listening socket only).
      3. /proc/<pid>/cmdline matches against ``_SERVER_CMDLINE_MARKERS``.

    The cmdline backstop catches the server whenever (1) and (2) both miss —
    startup race before port bind, lsof missing from PATH, unmanaged systemd,
    or a SIGUSR1 transition window where ``/status`` briefly fails.
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


def _server_is_cloud_only(port: int = DEFAULT_SERVER_PORT) -> bool:
    """Check if the server is already in cloud-only mode.

    Transient unreachability (connect-error during a SIGUSR1 transition) is
    logged at debug level and reported as False; the polling caller
    (_wait_for_vram_clear) retries.
    """
    try:
        import httpx

        resp = httpx.get(f"http://localhost:{port}/status", timeout=3)
        return resp.json().get("mode") == "cloud-only"
    except Exception as exc:
        logger.debug("server /status check failed: %s", exc)
        return False


def _send_release_signal(server_pid: int) -> None:
    """Send SIGUSR1 to the server to release the GPU."""
    os.kill(server_pid, signal.SIGUSR1)


def _wait_for_vram_clear(
    own_pid: int,
    timeout: int = VRAM_CLEAR_TIMEOUT,
    server_pid: int | None = None,
    port: int = DEFAULT_SERVER_PORT,
) -> bool:
    """Wait until no other process is using GPU VRAM.

    A server process that has unloaded its model keeps a CUDA context
    (stays in nvidia-smi --query-compute-apps) but uses no real VRAM.
    If server_pid is given, treat it as cleared once /status shows cloud-only.
    """
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        pids = _get_gpu_pids()
        external = [p for p in pids if p != own_pid]
        # Server with empty CUDA context is harmless — check status API
        if server_pid and external == [server_pid]:
            if _server_is_cloud_only(port):
                return True
        elif not external:
            return True
        time.sleep(VRAM_POLL_INTERVAL)
    return False


def check_gpu(port: int = DEFAULT_SERVER_PORT) -> dict:
    """Check GPU state without prompting or taking action.

    Returns a dict with:
        status: "free" | "server" | "server_cloud_only" | "unknown" | "mixed"
        server_pid: int | None
        unknown_pids: list[int]
        unknown_info: list[str]  — process details for display
        message: str  — human-readable summary
    """
    own_pid = os.getpid()
    gpu_pids = _get_gpu_pids()
    external_pids = [p for p in gpu_pids if p != own_pid]

    if not external_pids:
        return {
            "status": "free",
            "server_pid": None,
            "unknown_pids": [],
            "unknown_info": [],
            "message": "GPU is free.",
        }

    server_pids = _identify_server_pids(external_pids, port)
    server_pid = server_pids[0] if server_pids else None
    unknown_pids = [p for p in external_pids if p not in server_pids]

    unknown_info = []
    for pid in unknown_pids:
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "pid,etime,cmd", "--no-headers"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            unknown_info.append(result.stdout.strip())
        except Exception:
            unknown_info.append(f"PID {pid} (unknown)")

    if unknown_pids and not server_pids:
        return {
            "status": "unknown",
            "server_pid": None,
            "unknown_pids": unknown_pids,
            "unknown_info": unknown_info,
            "message": f"GPU occupied by unknown process(es): {unknown_info}",
        }

    if unknown_pids and server_pids:
        cloud = _server_is_cloud_only(port)
        return {
            "status": "mixed",
            "server_pid": server_pid,
            "unknown_pids": unknown_pids,
            "unknown_info": unknown_info,
            "message": f"Server (PID {server_pid}, {'cloud-only' if cloud else 'active'}) "
            f"and unknown processes on GPU: {unknown_info}",
        }

    # Only server
    if _server_is_cloud_only(port):
        return {
            "status": "server_cloud_only",
            "server_pid": server_pid,
            "unknown_pids": [],
            "unknown_info": [],
            "message": f"Server (PID {server_pid}) in cloud-only mode — GPU is available.",
        }

    return {
        "status": "server",
        "server_pid": server_pid,
        "unknown_pids": [],
        "unknown_info": [],
        "message": f"ParaMem server (PID {server_pid}) is using the GPU. "
        "It will switch to cloud-only mode if released.",
    }


def release_server_gpu(port: int = DEFAULT_SERVER_PORT, timeout: int = VRAM_CLEAR_TIMEOUT) -> bool:
    """Send SIGUSR1 to the server and wait for VRAM to clear.

    Returns True if GPU was released, False on timeout.
    """
    server_pid = _get_server_pid(port)
    if not server_pid:
        return False
    if _server_is_cloud_only(port):
        return True

    _send_release_signal(server_pid)
    return _wait_for_vram_clear(os.getpid(), timeout=timeout, server_pid=server_pid, port=port)


def acquire_gpu(port: int = DEFAULT_SERVER_PORT, interactive: bool = True):
    """Context manager for safe GPU acquisition.

    Args:
        port: ParaMem server port (for identification and status checks).
        interactive: If True, prompts user for confirmation. Set False
            for non-interactive scripts.

    Returns:
        A context manager. On enter: ensures GPU is free. On exit:
        sends ML_FINISHED notification (server auto-reclaims via timer).

    Raises:
        GPUAcquireError: If GPU cannot be acquired.
    """
    return _GPUGuard(port, interactive)


class _GPUGuard:
    def __init__(self, port: int, interactive: bool):
        self._port = port
        self._interactive = interactive
        self._released_server = False
        self._sleep_inhibitor = _SleepInhibitor()

    def __enter__(self):
        own_pid = os.getpid()
        gpu_pids = _get_gpu_pids()
        external_pids = [p for p in gpu_pids if p != own_pid]

        if not external_pids:
            # GPU is free
            notify_ml(ML_STARTED)
            self._sleep_inhibitor.start()
            return self

        # Classify GPU processes: server vs unknown. Uses three detection
        # sources (systemd MainPID, port listener, /proc cmdline) so a
        # server that escapes port detection still routes through the
        # defer-to-cloud path, never the kill path.
        server_pids = _identify_server_pids(external_pids, self._port)
        primary_server_pid = server_pids[0] if server_pids else None
        unknown_pids = [p for p in external_pids if p not in server_pids]

        # Unknown processes on GPU — offer to kill them
        if unknown_pids:
            process_info = []
            for pid in unknown_pids:
                try:
                    result = subprocess.run(
                        ["ps", "-p", str(pid), "-o", "pid,etime,cmd", "--no-headers"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    process_info.append(result.stdout.strip())
                except Exception:
                    process_info.append(f"PID {pid} (unknown)")

            print("\n  GPU is occupied by unknown process(es):")
            for info in process_info:
                print(f"    {info}")
            if primary_server_pid:
                print(
                    f"  (ParaMem server PID {primary_server_pid} also on GPU — "
                    "will be deferred to cloud-only after unknown process(es) clear.)"
                )

            # Fail-safe: every path that kills must pass through operator approval.
            if not self._interactive:
                raise GPUAcquireError(
                    "GPU occupied by unknown process(es); non-interactive mode — "
                    f"cannot obtain approval to kill. PIDs={unknown_pids}"
                )
            if not sys.stdin.isatty():
                raise GPUAcquireError(
                    "GPU occupied by unknown process(es); cannot prompt for approval "
                    f"(no TTY). Resolve manually. PIDs={unknown_pids}"
                )
            answer = input("  Kill these processes? [Y/n] ").strip().lower()
            if answer not in ("", "y", "yes"):
                raise GPUAcquireError("User declined — GPU not acquired")

            # Approved: SIGTERM → 3s → SIGKILL escalation. Without the
            # escalation a hung process holds the GPU until the 30 s VRAM
            # timeout elapses and we abort.
            for pid in unknown_pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                    print(f"  Sent SIGTERM to PID {pid}")
                except ProcessLookupError:
                    pass

            time.sleep(3)
            still_alive: list[int] = []
            for pid in unknown_pids:
                try:
                    os.kill(pid, 0)
                    still_alive.append(pid)
                except ProcessLookupError:
                    pass
            for pid in still_alive:
                try:
                    os.kill(pid, signal.SIGKILL)
                    print(f"  Sent SIGKILL to PID {pid} (SIGTERM ignored)")
                except ProcessLookupError:
                    pass

            # In the mixed case fall through to the server block below —
            # its own _wait_for_vram_clear handles the combined condition
            # (unknown PIDs gone AND server in cloud-only) and already
            # short-circuits on server_pid. Calling _wait_for_vram_clear
            # here would time out because the server is still active.
            if not primary_server_pid:
                if not _wait_for_vram_clear(own_pid):
                    raise GPUAcquireError("Timeout waiting for processes to release VRAM")
                print("  GPU cleared.")

        # Server on the GPU — defer to cloud-only (handles both server-only
        # and mixed cases; external_pids was computed pre-kill but the
        # classification of ``primary_server_pid`` stays valid).
        if primary_server_pid:
            server_pid = primary_server_pid
            if _server_is_cloud_only(self._port):
                notify_ml(ML_STARTED)
                self._sleep_inhibitor.start()
                return self

            if self._interactive and sys.stdin.isatty():
                print(
                    f"\n  ParaMem server (PID {server_pid}) is using the GPU."
                    "\n  It will switch to cloud-only mode during this workload."
                )
                answer = input("  Continue? [Y/n] ").strip().lower()
                if answer not in ("", "y", "yes"):
                    raise GPUAcquireError("User declined GPU acquisition")
            elif self._interactive:
                print(f"  ParaMem server (PID {server_pid}) on GPU — proceeding (no TTY).")

            print("  Sending SIGUSR1 to release GPU...")
            _send_release_signal(server_pid)

            if not _wait_for_vram_clear(own_pid, server_pid=server_pid, port=self._port):
                raise GPUAcquireError(
                    f"Timeout waiting for server (PID {server_pid}) to release VRAM"
                )

            self._released_server = True
            print("  GPU released by server.")

        # Stamp ownership of the PARAMEM_EXTRA_ARGS=--defer-model hold so the
        # server can distinguish a legitimate mid-training hold from an
        # orphaned env var left behind by a SIGKILLed test.  PARAMEM_EXTRA_ARGS
        # itself is set by ``_release_server_gpu`` in training-control.sh
        # before this process launches; we only add the identity fields.
        # PARAMEM_HOLD_CMD is a short hint ("python / paramem.experiments.testN")
        # that survives holder death so pstatus can still name the orphan.
        # /status surfaces these for pstatus and auto-reclaim uses owner_alive
        # to decide whether to reclaim.  Cleared in __exit__.
        hint = format_cmd_hint(sys.argv)
        stamp_args = [
            "systemctl",
            "--user",
            "set-environment",
            f"PARAMEM_HOLD_PID={os.getpid()}",
            f"PARAMEM_HOLD_STARTED_AT={int(time.time())}",
        ]
        if hint:
            stamp_args.append(f"PARAMEM_HOLD_CMD={hint}")
        try:
            subprocess.run(
                stamp_args,
                check=False,
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass

        notify_ml(ML_STARTED)
        self._sleep_inhibitor.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop sleep inhibitor first so Windows standby can resume as soon as
        # the GPU workload is done, before any other cleanup.
        self._sleep_inhibitor.stop()

        # Training actually stopped now — drop the defer-model flag before
        # anything else so the next auto-reclaim tick (or any racing
        # restart) loads the model in local mode. tpause only signals a
        # cycle-boundary stop; the script keeps running until the cycle
        # finishes, so the env var must not drop at tpause time — only
        # here, when the test process is exiting. Crash-exit (SIGKILL)
        # skips this hook, but auto-reclaim's GPU-free check combined with
        # main()'s fallback in app.py handle that case independently.
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
        notify_ml(ML_FINISHED)
        return False


def notify_paused():
    """Call when an ML workload pauses (e.g. tpause)."""
    notify_ml(ML_PAUSED)


def notify_resumed():
    """Call when an ML workload resumes (e.g. tresume)."""
    notify_ml(ML_RESUMED)
