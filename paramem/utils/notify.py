"""Windows toast notifications from WSL2 via powershell.exe.

Fire-and-forget: spawns a background process, never blocks the caller.
Silently no-ops if powershell.exe is not available (non-WSL environments).
"""

import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)

_POWERSHELL = shutil.which("powershell.exe")

# Notification categories
SERVER_LOCAL = "Local model active"
SERVER_CLOUD_ONLY = "GPU released, running in cloud-only mode"
SERVER_RECLAIMED = "GPU reclaimed, local model active"
ML_STARTED = "ML workload started — GPU acquired"
ML_FINISHED = "ML workload finished — GPU free"
ML_PAUSED = "ML workload paused — GPU free"
ML_RESUMED = "ML workload resumed — GPU acquired"


def notify(title: str, message: str) -> None:
    """Send a Windows toast notification from WSL2.

    Non-blocking: launches powershell.exe in background.
    No-ops silently if not in WSL or powershell.exe unavailable.
    """
    if not _POWERSHELL:
        logger.debug("powershell.exe not found, skipping notification")
        return

    script = (
        "Add-Type -AssemblyName System.Windows.Forms;"
        "$n = New-Object System.Windows.Forms.NotifyIcon;"
        "$n.Icon = [System.Drawing.SystemIcons]::Information;"
        "$n.Visible = $true;"
        f"$n.ShowBalloonTip(5000, '{_escape(title)}', '{_escape(message)}', 'Info');"
        "Start-Sleep 6; $n.Dispose()"
    )

    try:
        subprocess.Popen(
            [_POWERSHELL, "-NoProfile", "-Command", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as e:
        logger.debug("Notification failed: %s", e)


def notify_server(message: str) -> None:
    """Send a ParaMem server notification."""
    notify("ParaMem Server", message)


def notify_ml(message: str) -> None:
    """Send an ML workload notification."""
    notify("ParaMem ML", message)


def _escape(text: str) -> str:
    """Escape single quotes for PowerShell string literals."""
    return text.replace("'", "''")
