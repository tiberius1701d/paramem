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

import os
import shutil
import signal
import subprocess
import sys
import time

from paramem.utils.notify import ML_FINISHED, ML_PAUSED, ML_RESUMED, ML_STARTED, notify_ml

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


def _get_server_pid(port: int = DEFAULT_SERVER_PORT) -> int | None:
    """Find the PID of the process listening on the server port."""
    try:
        result = subprocess.run(
            ["lsof", "-i", f":{port}", "-t"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        pids = [int(p.strip()) for p in result.stdout.strip().split("\n") if p.strip()]
        return pids[0] if pids else None
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return None


def _server_is_cloud_only(port: int = DEFAULT_SERVER_PORT) -> bool:
    """Check if the server is already in cloud-only mode."""
    try:
        import httpx

        resp = httpx.get(f"http://localhost:{port}/status", timeout=3)
        return resp.json().get("mode") == "cloud-only"
    except Exception:
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

    server_pid = _get_server_pid(port)
    unknown_pids = [p for p in external_pids if p != server_pid]

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

    if unknown_pids and not (server_pid and server_pid in external_pids):
        return {
            "status": "unknown",
            "server_pid": None,
            "unknown_pids": unknown_pids,
            "unknown_info": unknown_info,
            "message": f"GPU occupied by unknown process(es): {unknown_info}",
        }

    if unknown_pids and server_pid and server_pid in external_pids:
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

    def __enter__(self):
        own_pid = os.getpid()
        gpu_pids = _get_gpu_pids()
        external_pids = [p for p in gpu_pids if p != own_pid]

        if not external_pids:
            # GPU is free
            notify_ml(ML_STARTED)
            return self

        # Classify GPU processes: server vs unknown
        server_pid = _get_server_pid(self._port)
        unknown_pids = [p for p in external_pids if p != server_pid]

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

            if self._interactive and sys.stdin.isatty():
                answer = input("  Kill these processes? [Y/n] ").strip().lower()
                if answer not in ("", "y", "yes"):
                    raise GPUAcquireError("User declined — GPU not acquired")
            elif not self._interactive:
                raise GPUAcquireError("GPU occupied by unknown processes (non-interactive mode)")

            for pid in unknown_pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                    print(f"  Sent SIGTERM to PID {pid}")
                except ProcessLookupError:
                    pass

            if not _wait_for_vram_clear(own_pid):
                raise GPUAcquireError("Timeout waiting for processes to release VRAM")
            print("  GPU cleared.")

        # Only the server is on the GPU
        if server_pid and server_pid in external_pids:
            if _server_is_cloud_only(self._port):
                notify_ml(ML_STARTED)
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

        notify_ml(ML_STARTED)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        notify_ml(ML_FINISHED)
        # Server auto-reclaims via timer — no action needed
        return False


def notify_paused():
    """Call when an ML workload pauses (e.g. tpause)."""
    notify_ml(ML_PAUSED)


def notify_resumed():
    """Call when an ML workload resumes (e.g. tresume)."""
    notify_ml(ML_RESUMED)
