#!/bin/bash
# Ensure netsh portproxy rules forward LAN traffic to WSL for ParaMem ports.
# Rules persist across reboots. Only adds missing rules; no-op if already set.
# Single elevation prompt for all ports.
#
# Elevation strategy: when the `ParaMem-Port-Forward` scheduled task is present
# (registered by scripts/setup/headless-boot.sh with RunLevel Highest), it is
# triggered prompt-free via `schtasks /Run`.  If the task is absent (headless_boot
# disabled, or first run before headless-boot registered it), falls back to an
# interactive one-time Start-Process -Verb RunAs prompt.  The actual rule-building
# logic lives entirely in win-port-forward.ps1 — this script only decides HOW to
# invoke that worker (mirrors the WU-Lock pattern in training-control.sh).
#
# Security scoping (opt-in via env vars):
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

# Listener scoping: default 0.0.0.0 unless PARAMEM_LISTEN_IP is set.
LISTEN_IP="${PARAMEM_LISTEN_IP:-0.0.0.0}"

# Firewall source scoping: unset by default, passed through to win-port-forward.ps1.
NAS_IP="${PARAMEM_NAS_IP:-}"

# Loud WARN if neither scoping knob is set — matches the documented security posture.
if [ "$LISTEN_IP" = "0.0.0.0" ] && [ -z "$NAS_IP" ]; then
    echo ""
    echo "WARN: ParaMem ports will be reachable from every LAN interface, unscoped."
    echo "WARN: Set PARAMEM_LISTEN_IP and PARAMEM_NAS_IP to scope exposure."
    echo ""
fi

# Windows path to the PowerShell worker (single source of truth for rule logic).
PS1_WIN=$(wslpath -w "$(dirname "$0")/win-port-forward.ps1")

# Distro name for the ps1 worker's WSL IP lookup.
DISTRO="${WSL_DISTRO_NAME:-Ubuntu-24.04}"

# Trigger method: task present -> prompt-free; absent -> interactive UAC fallback.
if schtasks.exe /Query /TN 'ParaMem-Port-Forward' >/dev/null 2>&1; then
    echo "Triggering port-forward via scheduled task (no UAC)..."
    schtasks.exe /Run /TN 'ParaMem-Port-Forward'
else
    echo "ParaMem-Port-Forward task not found; requesting one-time elevation..."
    echo "(Run 'bash scripts/setup/headless-boot.sh' once to register the task and avoid future prompts.)"
    powershell.exe -NoProfile -Command \
        "Start-Process powershell -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File \"$PS1_WIN\" -Distro $DISTRO -ListenIp $LISTEN_IP -NasIp \"$NAS_IP\"' -Verb RunAs -Wait" \
        2>/dev/null
fi

echo "Done. Current rules:"
powershell.exe -Command "netsh interface portproxy show v4tov4" 2>/dev/null
