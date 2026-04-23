#!/bin/bash
# Ensure netsh portproxy rules forward LAN traffic to WSL for ParaMem ports.
# Rules persist across reboots. Only adds missing rules; no-op if already set.
# Single elevation prompt for all ports.
#
# Requires elevation (netsh portproxy needs admin). Run from WSL:
#   bash scripts/net/win-port-forward.sh
#
# Security scoping (opt-in via env vars — WP5 in plan_security_hardening.md):
#   PARAMEM_LISTEN_IP  — Windows-side IP to bind the portproxy listener on (e.g. the
#                         Ethernet interface address). When unset, listens on 0.0.0.0
#                         (every interface, including Wi-Fi). Strongly recommend setting.
#   PARAMEM_NAS_IP     — IP of the Home Assistant NAS. When set, the Windows Firewall
#                         rule is scoped to accept inbound only from that source IP.
#                         When unset, the rule accepts any source on the LAN.
#
# Both env vars are independent: you can scope the listener without scoping the
# firewall, or vice versa. Security-ON requires both; open mode (neither set) prints
# a WARN and preserves the existing wide-open behavior.

set -euo pipefail

PORTS=(8420 10300 10301)  # REST API + Wyoming STT + Wyoming TTS
WSL_IP=$(hostname -I | awk '{print $1}')

if [ -z "$WSL_IP" ]; then
    echo "ERROR: Could not determine WSL IP"
    exit 1
fi

# Listener scoping: default 0.0.0.0 unless PARAMEM_LISTEN_IP is set.
LISTEN_IP="${PARAMEM_LISTEN_IP:-0.0.0.0}"

# Firewall source scoping: unset by default, pass-through to PowerShell when set.
NAS_IP="${PARAMEM_NAS_IP:-}"

echo "WSL IP:         $WSL_IP"
echo "Listen IP:      $LISTEN_IP"
if [ -n "$NAS_IP" ]; then
    echo "Firewall scope: inbound from $NAS_IP only"
else
    echo "Firewall scope: any LAN source (no PARAMEM_NAS_IP set)"
fi

# Loud WARN if neither scoping knob is set — matches the documented security posture.
if [ "$LISTEN_IP" = "0.0.0.0" ] && [ -z "$NAS_IP" ]; then
    echo ""
    echo "WARN: ParaMem ports will be reachable from every LAN interface, unscoped."
    echo "WARN: Set PARAMEM_LISTEN_IP and PARAMEM_NAS_IP to scope exposure."
    echo "WARN: See docs/plan_security_hardening.md (WP5) for rationale."
    echo ""
fi

EXISTING=$(powershell.exe -Command "netsh interface portproxy show v4tov4" 2>/dev/null)

# Build a single PowerShell script for all port/firewall changes
PS_COMMANDS=""
NEEDS_ELEVATION=false

for PORT in "${PORTS[@]}"; do
    if echo "$EXISTING" | grep -q "$PORT"; then
        # Existing rule — check if it matches the current (LISTEN_IP, WSL_IP) pair.
        # The netsh output format is:  Listen on ipv4:  <listen>  Connect to ipv4:  <connect>
        EXISTING_ROW=$(echo "$EXISTING" | grep -E "[[:space:]]$PORT[[:space:]]" || true)
        if echo "$EXISTING_ROW" | grep -q "$WSL_IP" && echo "$EXISTING_ROW" | grep -q "$LISTEN_IP"; then
            echo "Port $PORT: already forwarded ($LISTEN_IP → $WSL_IP)"
        else
            echo "Port $PORT: config drift detected, will update"
            # Delete any rule on this listenport regardless of address to avoid duplicates.
            PS_COMMANDS+="netsh interface portproxy delete v4tov4 listenport=$PORT listenaddress=0.0.0.0 2>\$null; "
            PS_COMMANDS+="netsh interface portproxy delete v4tov4 listenport=$PORT listenaddress=$LISTEN_IP 2>\$null; "
            PS_COMMANDS+="netsh interface portproxy add v4tov4 listenport=$PORT listenaddress=$LISTEN_IP connectport=$PORT connectaddress=$WSL_IP; "
            NEEDS_ELEVATION=true
        fi
    else
        echo "Port $PORT: will add forward"
        PS_COMMANDS+="netsh interface portproxy add v4tov4 listenport=$PORT listenaddress=$LISTEN_IP connectport=$PORT connectaddress=$WSL_IP; "
        NEEDS_ELEVATION=true
    fi

    # Firewall rules — scope RemoteAddress when PARAMEM_NAS_IP is set.
    if [ -n "$NAS_IP" ]; then
        # Scoped rule: remove any pre-existing unscoped rule first, then create scoped.
        PS_COMMANDS+="Get-NetFirewallRule -DisplayName 'ParaMem Port $PORT' -ErrorAction SilentlyContinue | Remove-NetFirewallRule -ErrorAction SilentlyContinue; "
        PS_COMMANDS+="New-NetFirewallRule -DisplayName 'ParaMem Port $PORT' -Direction Inbound -LocalPort $PORT -Protocol TCP -RemoteAddress $NAS_IP -Action Allow | Out-Null; "
        PS_COMMANDS+="Write-Host 'Firewall rule (scoped to $NAS_IP) created for port $PORT'; "
    else
        PS_COMMANDS+="if (-not (Get-NetFirewallRule -DisplayName 'ParaMem Port $PORT' -ErrorAction SilentlyContinue)) { New-NetFirewallRule -DisplayName 'ParaMem Port $PORT' -Direction Inbound -LocalPort $PORT -Protocol TCP -Action Allow | Out-Null; Write-Host 'Firewall rule (unscoped) created for port $PORT' } else { Write-Host 'Firewall rule exists for port $PORT' }; "
    fi
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
