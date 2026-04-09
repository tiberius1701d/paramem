#!/bin/bash
# Ensure netsh portproxy rules forward LAN traffic to WSL for ParaMem ports.
# Rules persist across reboots. Only adds missing rules; no-op if already set.
#
# Requires elevation (netsh portproxy needs admin). Run from WSL:
#   bash scripts/win-port-forward.sh

set -euo pipefail

PORTS=(8420 10300)  # REST API + Wyoming STT
WSL_IP=$(hostname -I | awk '{print $1}')

if [ -z "$WSL_IP" ]; then
    echo "ERROR: Could not determine WSL IP"
    exit 1
fi

echo "WSL IP: $WSL_IP"

EXISTING=$(powershell.exe -Command "netsh interface portproxy show v4tov4" 2>/dev/null)

for PORT in "${PORTS[@]}"; do
    if echo "$EXISTING" | grep -q "$PORT"; then
        # Rule exists — check if IP matches
        if echo "$EXISTING" | grep "$PORT" | grep -q "$WSL_IP"; then
            echo "Port $PORT: already forwarded to $WSL_IP"
        else
            # IP changed — update the rule
            powershell.exe -Command "Start-Process powershell -ArgumentList '-Command \"netsh interface portproxy delete v4tov4 listenport=$PORT listenaddress=0.0.0.0; netsh interface portproxy add v4tov4 listenport=$PORT listenaddress=0.0.0.0 connectport=$PORT connectaddress=$WSL_IP\"' -Verb RunAs -Wait" 2>/dev/null
            echo "Port $PORT: updated forward to $WSL_IP"
        fi
    else
        # No rule — add it
        powershell.exe -Command "Start-Process powershell -ArgumentList '-Command \"netsh interface portproxy add v4tov4 listenport=$PORT listenaddress=0.0.0.0 connectport=$PORT connectaddress=$WSL_IP\"' -Verb RunAs -Wait" 2>/dev/null
        echo "Port $PORT: added forward to $WSL_IP"
    fi

    # Ensure firewall rule exists
    powershell.exe -Command "Start-Process powershell -ArgumentList '-Command \"if (-not (Get-NetFirewallRule -DisplayName \\\"ParaMem Port $PORT\\\" -ErrorAction SilentlyContinue)) { New-NetFirewallRule -DisplayName \\\"ParaMem Port $PORT\\\" -Direction Inbound -LocalPort $PORT -Protocol TCP -Action Allow | Out-Null; Write-Host \\\"Firewall rule created for port $PORT\\\" } else { Write-Host \\\"Firewall rule exists for port $PORT\\\" }\"' -Verb RunAs -Wait" 2>/dev/null
done

echo "Done. Current rules:"
powershell.exe -Command "netsh interface portproxy show v4tov4" 2>/dev/null
