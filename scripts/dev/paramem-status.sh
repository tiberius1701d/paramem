#!/bin/bash
# paramem-status.sh — Show ParaMem server status
#
# Usage:
#   pstatus          # show server status
#
# Alias (add to ~/.bashrc):
#   alias pstatus='bash ~/.local/bin/paramem-status.sh'

set -euo pipefail

PARAMEM_SERVER_PORT=8420

# Colors
BOLD="\033[1m"
RESET="\033[0m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
CYAN="\033[0;36m"
DIM="\033[2m"

# Check if server process is running
server_pid=$(lsof -i :"$PARAMEM_SERVER_PORT" -t 2>/dev/null | head -1)

echo -e "${BOLD}ParaMem Server${RESET}"
echo "  ────────────────────────────────────────"

if [[ -z "$server_pid" ]]; then
    echo -e "  Status:   ${RED}NOT RUNNING${RESET}"
    echo -e "  Service:  $(systemctl --user is-active paramem-server 2>/dev/null || echo "unknown")"
    exit 0
fi

# Query /status endpoint
status_json=$(curl -s --max-time 3 "http://localhost:${PARAMEM_SERVER_PORT}/status" 2>/dev/null)
if [[ -z "$status_json" ]]; then
    echo -e "  Status:   ${YELLOW}UNREACHABLE${RESET} (PID ${server_pid})"
    exit 1
fi

# Parse fields
# All status parsing happens in a single python call; everything we render comes from this JSON.
# JSON passed via env var to avoid heredoc-vs-herestring stdin conflict.
parsed=$(STATUS_JSON="$status_json" python3 <<'PY'
import json, os
d = json.loads(os.environ["STATUS_JSON"])

def fmt_duration(seconds):
    if seconds is None:
        return "-"
    seconds = int(seconds)
    if seconds < 0:
        return "now"
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    if h < 48:
        return f"{h}h{m:02d}m"
    return f"{h // 24}d{h % 24:02d}h"

def fmt_result(r):
    if not r:
        return "-"
    status = r.get("status", "?")
    if status == "simulated":
        return f"simulated {r.get('sessions', 0)}s → {r.get('episodic_qa', 0)}ep, {r.get('procedural_rels', 0)}pr"
    if status == "trained":
        return f"trained {r.get('total_keys', 0)} keys ({','.join(r.get('jobs', [])) or '?'})"
    if status == "no_facts":
        return f"no facts from {r.get('sessions', 0)} sessions"
    return status

fields = {
    "mode": d.get("mode", "?"),
    "cloud_only_reason": d.get("cloud_only_reason") or "none",
    "model": d.get("model", "?"),
    "adapter_loaded": d.get("adapter_loaded", False),
    "keys_count": d.get("keys_count", 0),
    "pending_sessions": d.get("pending_sessions", 0),
    "consolidating": d.get("consolidating", False),
    "last_consolidation": d.get("last_consolidation") or "never",
    "last_result": fmt_result(d.get("last_consolidation_result")),
    "schedule": d.get("schedule", "") or "-",
    "mode_config": d.get("mode_config", "") or "-",
    "next_run": fmt_duration(d.get("next_run_seconds")),
    "scheduler_started": d.get("scheduler_started", False),
    "orphaned": d.get("orphaned_pending", 0),
    "oldest_age": fmt_duration(d.get("oldest_pending_seconds")),
    "speaker_profiles": d.get("speaker_profiles", 0),
    "pending_enrollments": d.get("pending_enrollments", 0),
    "stt_loaded": d.get("stt_loaded", False),
    "stt_model": d.get("stt_model") or "-",
    "bg_trainer_active": d.get("bg_trainer_active", False),
    "bg_trainer_adapter": d.get("bg_trainer_adapter") or "-",
}
# Scalar line
print("|".join(str(fields[k]) for k in [
    "mode", "cloud_only_reason", "model", "adapter_loaded", "keys_count",
    "pending_sessions", "consolidating", "last_consolidation", "last_result",
    "schedule", "mode_config", "next_run", "scheduler_started",
    "orphaned", "oldest_age", "speaker_profiles", "pending_enrollments",
    "stt_loaded", "stt_model",
    "bg_trainer_active", "bg_trainer_adapter",
]))
# Per-speaker lines: id<TAB>name<TAB>embeddings<TAB>pending<TAB>method
for s in d.get("speakers", []):
    print("SPK\t{}\t{}\t{}\t{}\t{}".format(
        s.get("id", "?"), s.get("name", "?"),
        s.get("embeddings", 0), s.get("pending", 0),
        s.get("enroll_method", "unknown"),
    ))
PY
)

IFS='|' read -r mode cloud_only_reason model adapter_loaded keys_count pending_sessions \
    consolidating last_consolidation last_result schedule mode_config next_run \
    scheduler_started orphaned oldest_age speaker_profiles pending_enrollments \
    stt_loaded stt_model bg_trainer_active bg_trainer_adapter \
    <<< "$(echo "$parsed" | head -1)"
speaker_lines=$(echo "$parsed" | awk '/^SPK\t/')

# Mode display
if [[ "$mode" == "local" ]]; then
    mode_display="${GREEN}LOCAL${RESET} (GPU active)"
elif [[ "$mode" == "cloud-only" ]]; then
    reason_text=""
    case "$cloud_only_reason" in
        explicit)     reason_text="explicit --cloud-only, auto-reclaim disabled" ;;
        training)     reason_text="deferred for training, auto-reclaim enabled" ;;
        gpu_conflict) reason_text="GPU occupied at startup, auto-reclaim enabled" ;;
        *)            reason_text="unknown" ;;
    esac
    mode_display="${YELLOW}CLOUD-ONLY${RESET} (${reason_text})"
else
    mode_display="${RED}${mode}${RESET}"
fi

echo -e "  Server:   ${mode_display}"

# GPU status
if command -v nvidia-smi &>/dev/null; then
    temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | xargs)
    if [[ -n "$temp" ]]; then
        power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | xargs)
        mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | xargs)
        mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | xargs)
        # Color temperature
        if (( temp >= 80 )); then
            temp_color="${RED}"
        elif (( temp >= 60 )); then
            temp_color="${YELLOW}"
        else
            temp_color="${CYAN}"
        fi
        # Server on GPU?
        server_on_gpu=""
        if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q "^${server_pid}$"; then
            server_on_gpu=" | server allocated"
        fi
        echo -e "  GPU:      ${temp_color}${temp}°C${RESET} | ${power}W | ${mem_used}/${mem_total} MiB${server_on_gpu}"
    fi
fi

echo -e "  PID:      ${server_pid}"
echo -e "  Model:    ${CYAN}${model}${RESET}"

# Adapter
if [[ "$adapter_loaded" == "True" ]]; then
    echo -e "  Adapter:  ${GREEN}loaded${RESET} (${keys_count} keys)"
else
    echo -e "  Adapter:  ${DIM}not loaded${RESET}"
fi

# Sessions
pending_line="${pending_sessions} sessions"
if (( orphaned > 0 )); then
    pending_line+=" (${YELLOW}${orphaned} orphaned${RESET})"
fi
if [[ "$oldest_age" != "-" ]]; then
    pending_line+=" | oldest ${DIM}${oldest_age}${RESET}"
fi
echo -e "  Pending:  ${pending_line}"

# Consolidation
if [[ "$consolidating" == "True" ]]; then
    echo -e "  Consol:   ${YELLOW}IN PROGRESS${RESET}"
else
    consol_line="last ${DIM}${last_consolidation}${RESET}"
    if [[ "$last_result" != "-" ]]; then
        consol_line+=" | ${CYAN}${last_result}${RESET}"
    fi
    echo -e "  Consol:   ${consol_line}"
fi

# Background trainer
if [[ "$bg_trainer_active" == "True" ]]; then
    echo -e "  BG Train: ${GREEN}training${RESET} (${CYAN}${bg_trainer_adapter}${RESET})"
fi

# Scheduler
if [[ "$schedule" == "-" || -z "$schedule" ]]; then
    echo -e "  Schedule: ${DIM}manual only${RESET} (mode: ${mode_config})"
else
    sched_line="${CYAN}${schedule}${RESET} (mode: ${mode_config})"
    if [[ "$scheduler_started" != "True" ]]; then
        sched_line+=" | ${DIM}first run pending${RESET}"
    elif [[ "$next_run" != "-" ]]; then
        sched_line+=" | next in ${next_run}"
    fi
    echo -e "  Schedule: ${sched_line}"
fi

# Pending enrollments
if (( pending_enrollments > 0 )); then
    echo -e "  Enroll:   ${YELLOW}${pending_enrollments} awaiting name extraction${RESET}"
fi

# Speakers (name + method + embeddings + pending)
if (( speaker_profiles > 0 )); then
    echo -e "  Speakers: ${speaker_profiles} enrolled"
    while IFS=$'\t' read -r _marker sid sname semb spending smethod; do
        [[ -z "$sid" ]] && continue
        pending_tag=""
        if [[ "$spending" != "0" ]]; then
            pending_tag=" — ${CYAN}${spending} pending${RESET}"
        fi
        method_tag="${DIM}${smethod}${RESET}"
        echo -e "    ${DIM}${sid}${RESET}  ${sname}  [${method_tag}]  (${semb} emb)${pending_tag}"
    done <<< "$speaker_lines"
else
    echo -e "  Speakers: ${DIM}none enrolled${RESET}"
fi

# STT
if [[ "$stt_loaded" == "True" ]]; then
    echo -e "  STT:      ${GREEN}loaded${RESET} (${stt_model})"
else
    echo -e "  STT:      ${DIM}not loaded${RESET}"
fi

# Windows Update lock (read-only check — no elevation)
if command -v powershell.exe &>/dev/null; then
    wu_info=$(powershell.exe -NoProfile -Command "
        \$v=(Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings' PauseUpdatesExpiryTime -ErrorAction SilentlyContinue).PauseUpdatesExpiryTime
        if (\$v -and ([datetime]\$v).ToUniversalTime() -gt (Get-Date).ToUniversalTime()) {
            ([datetime]\$v).ToLocalTime().ToString('yyyy-MM-dd HH:mm')
        }
    " 2>/dev/null | tr -d '\r\n')
    if [[ -n "$wu_info" ]]; then
        echo -e "  WU Lock:  ${YELLOW}paused until ${wu_info}${RESET}"
    fi
fi

# Systemd service info
service_status=$(systemctl --user is-active paramem-server 2>/dev/null || echo "unknown")
uptime_info=$(systemctl --user show paramem-server --property=ActiveEnterTimestamp 2>/dev/null | cut -d= -f2)
if [[ -n "$uptime_info" && "$uptime_info" != "" ]]; then
    echo -e "  Service:  ${service_status}, since ${DIM}${uptime_info}${RESET}"
else
    echo -e "  Service:  ${service_status}"
fi
