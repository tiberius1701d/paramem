#!/bin/bash
# Forward port 8420 from LAN to WSL using a Python TCP forwarder.
# Runs on Windows Python — netsh portproxy is unreliable on WSL2 NAT mode.
#
# Called from start-server.sh. Launches the forwarder in the background
# and ensures the Windows firewall rule exists.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8420
WSL_IP=$(hostname -I | awk '{print $1}')

if [ -z "$WSL_IP" ]; then
    echo "ERROR: Could not determine WSL IP"
    exit 1
fi

# Ensure firewall rule exists (idempotent, needs elevation)
powershell.exe -Command "Start-Process powershell -ArgumentList '-Command \"if (-not (Get-NetFirewallRule -DisplayName \\\"ParaMem Server\\\" -ErrorAction SilentlyContinue)) { New-NetFirewallRule -DisplayName \\\"ParaMem Server\\\" -Direction Inbound -LocalPort $PORT -Protocol TCP -Action Allow | Out-Null; Write-Host \\\"Firewall rule created\\\" } else { Write-Host \\\"Firewall rule exists\\\" }\"' -Verb RunAs -Wait" 2>/dev/null

# Kill any existing forwarder on this port
powershell.exe -Command "Get-Process python* -ErrorAction SilentlyContinue | Where-Object { \$_.CommandLine -like '*tcp-forward*$PORT*' } | Stop-Process -Force -ErrorAction SilentlyContinue" 2>/dev/null || true

# Launch TCP forwarder on Windows Python (background)
FORWARD_SCRIPT=$(wslpath -w "$SCRIPT_DIR/tcp-forward.py")
powershell.exe -Command "Start-Process python -ArgumentList '$FORWARD_SCRIPT $PORT $WSL_IP' -WindowStyle Hidden" 2>/dev/null

echo "Port $PORT forwarded to WSL at $WSL_IP (TCP forwarder)"
