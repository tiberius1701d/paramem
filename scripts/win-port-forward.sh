#!/bin/bash
# Ensure netsh portproxy rules forward LAN traffic to WSL for ParaMem ports.
# Rules persist across reboots. Only adds missing rules; no-op if already set.
# Single elevation prompt for all ports.
#
# Requires elevation (netsh portproxy needs admin). Run from WSL:
#   bash scripts/win-port-forward.sh

set -euo pipefail

PORTS=(8420 10300 10301)  # REST API + Wyoming STT + Wyoming TTS
WSL_IP=$(hostname -I | awk '{print $1}')

if [ -z "$WSL_IP" ]; then
    echo "ERROR: Could not determine WSL IP"
    exit 1
fi

echo "WSL IP: $WSL_IP"

EXISTING=$(powershell.exe -Command "netsh interface portproxy show v4tov4" 2>/dev/null)

# Build a single PowerShell script for all port/firewall changes
PS_COMMANDS=""
NEEDS_ELEVATION=false

for PORT in "${PORTS[@]}"; do
    if echo "$EXISTING" | grep -q "$PORT"; then
        if echo "$EXISTING" | grep "$PORT" | grep -q "$WSL_IP"; then
            echo "Port $PORT: already forwarded to $WSL_IP"
        else
            echo "Port $PORT: IP changed, will update"
            PS_COMMANDS+="netsh interface portproxy delete v4tov4 listenport=$PORT listenaddress=0.0.0.0; "
            PS_COMMANDS+="netsh interface portproxy add v4tov4 listenport=$PORT listenaddress=0.0.0.0 connectport=$PORT connectaddress=$WSL_IP; "
            NEEDS_ELEVATION=true
        fi
    else
        echo "Port $PORT: will add forward"
        PS_COMMANDS+="netsh interface portproxy add v4tov4 listenport=$PORT listenaddress=0.0.0.0 connectport=$PORT connectaddress=$WSL_IP; "
        NEEDS_ELEVATION=true
    fi

    # Firewall rules
    PS_COMMANDS+="if (-not (Get-NetFirewallRule -DisplayName 'ParaMem Port $PORT' -ErrorAction SilentlyContinue)) { New-NetFirewallRule -DisplayName 'ParaMem Port $PORT' -Direction Inbound -LocalPort $PORT -Protocol TCP -Action Allow | Out-Null; Write-Host 'Firewall rule created for port $PORT' } else { Write-Host 'Firewall rule exists for port $PORT' }; "
done

# Execute all changes in a single elevated session
if [ "$NEEDS_ELEVATION" = true ]; then
    powershell.exe -Command "Start-Process powershell -ArgumentList '-Command \"$PS_COMMANDS\"' -Verb RunAs -Wait" 2>/dev/null
else
    # Still check firewall rules (no elevation needed if they exist)
    powershell.exe -Command "$PS_COMMANDS" 2>/dev/null || \
        powershell.exe -Command "Start-Process powershell -ArgumentList '-Command \"$PS_COMMANDS\"' -Verb RunAs -Wait" 2>/dev/null
fi

echo "Done. Current rules:"
powershell.exe -Command "netsh interface portproxy show v4tov4" 2>/dev/null
