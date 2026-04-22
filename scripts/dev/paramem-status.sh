#!/bin/bash
# paramem-status.sh — Show ParaMem server status
#
# Usage:
#   pstatus                # show server status
#   pstatus --force-local  # clear any PARAMEM_EXTRA_ARGS=--defer-model hold
#                            and restart the server so it boots in local mode
#
# Alias (add to ~/.bashrc):
#   alias pstatus='bash ~/.local/bin/paramem-status.sh'

set -euo pipefail

: "${PARAMEM_SERVER_PORT:=8420}"

# --force-local: clear the deferred-mode hold and restart into local mode.
# Intended for operator use when auto-reclaim has flagged an orphaned hold
# (holder PID dead or no PID registered) and stopped looping.  Hits
# POST /gpu/force-local on the running server.
if [[ "${1:-}" == "--force-local" ]]; then
    resp=$(curl -s --max-time 10 -X POST \
        "http://localhost:${PARAMEM_SERVER_PORT}/gpu/force-local" 2>/dev/null || true)
    if [[ -z "$resp" ]]; then
        echo "Server unreachable — clearing environment directly."
        systemctl --user unset-environment \
            PARAMEM_EXTRA_ARGS PARAMEM_HOLD_PID PARAMEM_HOLD_STARTED_AT || true
        systemctl --user restart paramem-server || true
        echo "Done. Run pstatus to verify mode."
        exit 0
    fi
    echo "$resp"
    exit 0
fi

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

def short_model_id(mid):
    # Strip HF org prefix so "mistralai/Mistral-7B-Instruct-v0.3" renders as
    # "Mistral-7B-Instruct-v0.3" — the org is noise for a status line.
    if not mid:
        return ""
    return mid.rsplit("/", 1)[-1]

fields = {
    "mode": d.get("mode", "?"),
    "cloud_only_reason": d.get("cloud_only_reason") or "none",
    "model": d.get("model", "?"),
    "model_id_short": short_model_id(d.get("model_id")),
    "model_device": d.get("model_device") or "-",
    "episodic_rank": d.get("episodic_rank") if d.get("episodic_rank") is not None else "-",
    "adapter_loaded": d.get("adapter_loaded", False),
    # Render adapter inventory as a comma-separated "kind(count)" list so
    # bash can echo it straight to the terminal without further parsing.
    "adapter_config": ", ".join(
        f"{k}({v})" for k, v in (d.get("adapter_config") or {}).items() if v
    ) or "-",
    "active_adapter": d.get("active_adapter") or "-",
    "keys_count": d.get("keys_count", 0),
    "pending_sessions": d.get("pending_sessions", 0),
    "consolidating": d.get("consolidating", False),
    "last_consolidation": d.get("last_consolidation") or "never",
    "last_result": fmt_result(d.get("last_consolidation_result")),
    "refresh_cadence": d.get("refresh_cadence", "") or "-",
    "consolidation_period": d.get("consolidation_period", "") or "-",
    "max_interim_count": d.get("max_interim_count", 0),
    "mode_config": d.get("mode_config", "") or "-",
    "next_run": fmt_duration(d.get("next_run_seconds")),
    "next_interim": fmt_duration(d.get("next_interim_seconds")),
    "scheduler_started": d.get("scheduler_started", False),
    "orphaned": d.get("orphaned_pending", 0),
    "oldest_age": fmt_duration(d.get("oldest_pending_seconds")),
    "speaker_profiles": d.get("speaker_profiles", 0),
    "pending_enrollments": d.get("pending_enrollments", 0),
    "speaker_embedding_backend": d.get("speaker_embedding_backend") or "-",
    "speaker_embedding_model": d.get("speaker_embedding_model") or "-",
    "speaker_embedding_device": d.get("speaker_embedding_device") or "-",
    "stt_loaded": d.get("stt_loaded", False),
    "stt_engine": d.get("stt_engine") or "-",
    "stt_model": d.get("stt_model") or "-",
    "stt_device": d.get("stt_device") or "-",
    "tts_loaded": d.get("tts_loaded", False),
    "tts_engine": d.get("tts_engine") or "-",
    "tts_languages": ",".join(d.get("tts_languages") or []) or "-",
    "tts_device": d.get("tts_device") or "-",
    "bg_trainer_active": d.get("bg_trainer_active", False),
    "bg_trainer_adapter": d.get("bg_trainer_adapter") or "-",
    # Thermal throttle policy (quiet-hours). mode: on|off|hours in the UI,
    # corresponds to always_on / always_off / auto on the server side.
    "throttle_mode": (d.get("thermal_policy") or {}).get("mode") or "-",
    "throttle_start": (d.get("thermal_policy") or {}).get("start") or "-",
    "throttle_end": (d.get("thermal_policy") or {}).get("end") or "-",
    "throttle_temp_limit": (d.get("thermal_policy") or {}).get("temp_limit") if (d.get("thermal_policy") or {}).get("temp_limit") is not None else "-",
    "throttle_active": (d.get("thermal_policy") or {}).get("currently_throttling", False),
    # adapter_health: count degenerated so bash can render a warning banner
    # without re-parsing the whole map.
    "degenerated_count": sum(
        1 for h in (d.get("adapter_health") or {}).values()
        if h.get("status") == "degenerated"
    ),
    "adapter_health_count": len(d.get("adapter_health") or {}),
    # Deferred-mode hold — owner_alive is "yes" / "no" / "-" (unknown) so
    # bash can render a coloured status tag without re-parsing the JSON.
    # owner_hint is the short cmd tag ("python / paramem.server.app") so
    # operators recognise the holder even after SIGKILL — "-" when unstamped.
    # Strip "|" to keep the bash scalar line (pipe-delimited) parsable.
    "hold_active": (d.get("hold") or {}).get("hold_active", False),
    "hold_owner_pid": (d.get("hold") or {}).get("owner_pid") if (d.get("hold") or {}).get("owner_pid") is not None else "-",
    "hold_owner_alive": (
        "yes" if (d.get("hold") or {}).get("owner_alive") is True
        else "no" if (d.get("hold") or {}).get("owner_alive") is False
        else "-"
    ),
    "hold_age": fmt_duration((d.get("hold") or {}).get("age_seconds")),
    "hold_owner_hint": ((d.get("hold") or {}).get("owner_hint") or "-").replace("|", "/"),
}
# Scalar line
print("|".join(str(fields[k]) for k in [
    "mode", "cloud_only_reason", "model", "model_id_short", "model_device",
    "episodic_rank", "adapter_loaded",
    "adapter_config", "active_adapter", "keys_count",
    "pending_sessions", "consolidating", "last_consolidation", "last_result",
    "refresh_cadence", "consolidation_period", "max_interim_count",
    "mode_config", "next_run", "next_interim", "scheduler_started",
    "orphaned", "oldest_age", "speaker_profiles", "pending_enrollments",
    "speaker_embedding_backend", "speaker_embedding_model",
    "speaker_embedding_device",
    "stt_loaded", "stt_engine", "stt_model", "stt_device",
    "tts_loaded", "tts_engine", "tts_languages", "tts_device",
    "bg_trainer_active", "bg_trainer_adapter",
    "throttle_mode", "throttle_start", "throttle_end",
    "throttle_temp_limit", "throttle_active",
    "degenerated_count", "adapter_health_count",
    "hold_active", "hold_owner_pid", "hold_owner_alive", "hold_age",
    "hold_owner_hint",
]))
# Per-adapter spec lines: kind<TAB>rank<TAB>alpha<TAB>lr<TAB>target_kind
for _kind, _spec in (d.get("adapter_specs") or {}).items():
    print("ADPT\t{}\t{}\t{}\t{}\t{}".format(
        _kind,
        _spec.get("rank", "?"),
        _spec.get("alpha", "?"),
        _spec.get("learning_rate", "?"),
        _spec.get("target_kind", "?"),
    ))
# Per-speaker lines: id<TAB>name<TAB>embeddings<TAB>pending<TAB>method
for s in d.get("speakers", []):
    print("SPK\t{}\t{}\t{}\t{}\t{}".format(
        s.get("id", "?"), s.get("name", "?"),
        s.get("embeddings", 0), s.get("pending", 0),
        s.get("enroll_method", "unknown"),
    ))
# Per-adapter health lines: adapter<TAB>status<TAB>reason<TAB>keys_at_mark<TAB>updated_at
# Sorted: degenerated first so the most important rows surface at the top.
health_items = list((d.get("adapter_health") or {}).items())
health_items.sort(key=lambda kv: (kv[1].get("status") != "degenerated", kv[0]))
for adapter_id, h in health_items:
    print("HLT\t{}\t{}\t{}\t{}\t{}".format(
        adapter_id,
        h.get("status", "?"),
        (h.get("reason") or "").replace("\t", " ").replace("\n", " "),
        h.get("keys_at_mark", 0),
        h.get("updated_at", ""),
    ))
# Slice 5a — Attention block lines: ATTN<TAB>kind<TAB>level<TAB>summary<TAB>action_hint<TAB>age_seconds
for item in (d.get("attention") or {}).get("items", []) or []:
    print("ATTN\t{}\t{}\t{}\t{}\t{}".format(
        item.get("kind", "?"),
        item.get("level", "info"),
        (item.get("summary") or "").replace("\t", " ").replace("\n", " "),
        (item.get("action_hint") or "").replace("\t", " ").replace("\n", " "),
        item.get("age_seconds") if item.get("age_seconds") is not None else "",
    ))
# Slice 5a — Migrate footer: MIGRATE<TAB>state<TAB>config_rev<TAB>applied_date
_mig = d.get("migration") or {}
_loaded_hash = (d.get("config_drift") or {}).get("loaded_hash", "")
_config_rev = _mig.get("config_rev") or (_loaded_hash[:8] if _loaded_hash else "")
# server_started_at is now on /status (Fix 2); take the YYYY-MM-DD slice.
_started = d.get("server_started_at") or ""
_applied_date = _started[:10] if _started else ""
print("MIGRATE\t{}\t{}\t{}".format(
    _mig.get("state", "live"),
    _config_rev,
    _applied_date,
))
PY
)

IFS='|' read -r mode cloud_only_reason model model_id_short model_device \
    episodic_rank adapter_loaded \
    adapter_config_str active_adapter keys_count pending_sessions \
    consolidating last_consolidation last_result refresh_cadence consolidation_period \
    max_interim_count mode_config next_run next_interim \
    scheduler_started orphaned oldest_age speaker_profiles pending_enrollments \
    spk_emb_backend spk_emb_model spk_emb_device \
    stt_loaded stt_engine stt_model stt_device \
    tts_loaded tts_engine tts_languages tts_device \
    bg_trainer_active bg_trainer_adapter \
    throttle_mode throttle_start throttle_end \
    throttle_temp_limit throttle_active \
    degenerated_count adapter_health_count \
    hold_active hold_owner_pid hold_owner_alive hold_age \
    hold_owner_hint \
    <<< "$(echo "$parsed" | head -1)"
speaker_lines=$(echo "$parsed" | awk '/^SPK\t/')
health_lines=$(echo "$parsed" | awk '/^HLT\t/')
adapter_spec_lines=$(echo "$parsed" | awk '/^ADPT\t/')
# Slice 5a — attention items and migrate footer.
attention_lines=$(echo "$parsed" | awk '/^ATTN\t/')
migrate_line=$(echo "$parsed" | awk '/^MIGRATE\t/' | head -1)

# Render a duration in seconds as a short human-readable string.
# Used for the age tag in the Attention block.
fmt_duration_inline() {
    local secs="$1"
    if [[ -z "$secs" || "$secs" == "0" ]]; then
        echo "0s"
        return
    fi
    local s=$((secs % 60))
    local m=$((secs / 60 % 60))
    local h=$((secs / 3600 % 24))
    local d=$((secs / 86400))
    if (( d > 0 )); then
        echo "${d}d${h}h"
    elif (( h > 0 )); then
        echo "${h}h${m}m"
    elif (( m > 0 )); then
        echo "${m}m"
    else
        echo "${s}s"
    fi
}

# Render a device string as a coloured tag. "cuda" and anything else
# non-cpu map to "GPU" so the label stays hardware-agnostic — the
# underlying string is still cuda today, but ROCm / MPS / future GPU
# backends would land on the same green tag without a display change.
fmt_device() {
    case "$1" in
        cpu)   echo -e "${YELLOW}CPU${RESET}" ;;
        cuda)  echo -e "${GREEN}GPU${RESET}" ;;
        mixed) echo -e "${YELLOW}mixed${RESET}" ;;
        ""|-)  echo -e "${DIM}-${RESET}" ;;
        *)     echo -e "${GREEN}GPU${RESET}" ;;
    esac
}

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

# PID line — append hold annotation when a deferred-mode hold is active.
# The server PID stays the primary number; the annotation names the holder
# (test process) by its stamped cmd hint.  Orphaned holds get a yellow
# tag + the reclaim hint so the row itself is self-documenting.
pid_line="  PID:      ${server_pid}"
if [[ "$hold_active" == "True" ]]; then
    cmd_tag=""
    if [[ "$hold_owner_hint" != "-" && -n "$hold_owner_hint" ]]; then
        cmd_tag=" [${hold_owner_hint}]"
    fi
    age_tag=""
    if [[ "$hold_age" != "-" ]]; then
        age_tag=" (age ${hold_age})"
    fi
    case "$hold_owner_alive" in
        yes)
            pid_line+=" ${DIM}(held by${cmd_tag}${age_tag})${RESET}"
            ;;
        no)
            pid_line+=" ${YELLOW}(orphaned hold by${cmd_tag}${age_tag} — pstatus --force-local)${RESET}"
            ;;
        *)
            pid_line+=" ${YELLOW}(orphaned hold, no holder registered — pstatus --force-local)${RESET}"
            ;;
    esac
fi
echo -e "$pid_line"

# Slice 5a — Attention block. Omitted entirely when attention_lines is empty.
if [[ -n "$attention_lines" ]]; then
    # Determine banner color: red (✗) if any item is level="failed", else yellow (⚠).
    banner_color="$YELLOW"
    banner_glyph="⚠"
    while IFS=$'\t' read -r _attn_marker _akind alevel _asummary _ahint _aage; do
        [[ -z "$alevel" ]] && continue
        if [[ "$alevel" == "failed" ]]; then
            banner_color="$RED"
            banner_glyph="✗"
            break
        fi
    done <<< "$attention_lines"

    echo "  ────────────────────────────────────────"
    echo -e "  ${banner_color}${banner_glyph}  ATTENTION — USER ACTION REQUIRED${RESET}"
    echo "  ────────────────────────────────────────"

    while IFS=$'\t' read -r _attn_marker akind alevel asummary ahint aage; do
        [[ -z "$akind" ]] && continue
        case "$alevel" in
            failed)          row_color="$RED"    ;;
            action_required) row_color="$YELLOW" ;;
            info|*)          row_color="$CYAN"   ;;
        esac
        # Map kind to display label.
        case "$akind" in
            migration_*)             label="Migration:" ;;
            consolidation_blocked)   label="Consol:   " ;;
            sweeper_held)            label="Sweeper:  " ;;
            config_drift)            label="Config:   " ;;
            adapter_fingerprint_*)   label="Adapter:  " ;;
            *)                       label="${akind}: " ;;
        esac
        # Age tag (when present): " (age 42m)"
        age_tag=""
        if [[ -n "$aage" && "$aage" != "" ]]; then
            age_tag=" (age $(fmt_duration_inline "$aage"))"
        fi
        echo -e "  ${row_color}${label}${RESET} ${asummary}${age_tag}"
        if [[ -n "$ahint" && "$ahint" != "" ]]; then
            echo -e "             ${DIM}→ ${ahint}${RESET}"
        fi
    done <<< "$attention_lines"
    echo "  ────────────────────────────────────────"
fi

# Model line: short name + configured variant + live device tag. The device
# tag is hardware-agnostic — cuda/rocm/mps all map to "GPU" via fmt_device.
model_line="${CYAN}${model}${RESET}"
if [[ "$model_id_short" != "-" && -n "$model_id_short" ]]; then
    model_line+=" (${DIM}${model_id_short}${RESET})"
fi
if [[ "$model_device" != "-" && -n "$model_device" ]]; then
    model_line+=" on $(fmt_device "$model_device")"
fi
echo -e "  Model:    ${model_line}"

# Adapters: configured inventory + per-kind spec. Configuration comes
# from yaml; active is whichever adapter `set_adapter()` last selected
# (PEFT only keeps one active at a time). "loaded" would be misleading —
# multiple adapters can be RESIDENT while only one is ACTIVE.
echo -e "  Adapters: ${CYAN}${adapter_config_str}${RESET}"
# Per-kind spec rows: episodic / semantic / procedural differ in
# learning rate and target_kind (procedural adds MLP targets). Skip the
# row when no kind is enabled.
if [[ -n "$adapter_spec_lines" ]]; then
    while IFS=$'\t' read -r _marker akind arank aalpha alr atgt; do
        [[ -z "$akind" ]] && continue
        echo -e "    ${DIM}${akind}${RESET}  r=${arank} α=${aalpha} lr=${alr} ${DIM}${atgt}${RESET}"
    done <<< "$adapter_spec_lines"
fi
if [[ "$active_adapter" != "-" && -n "$active_adapter" ]]; then
    echo -e "  Active:   ${GREEN}${active_adapter}${RESET} (${keys_count} keys)"
else
    echo -e "  Active:   ${DIM}none${RESET}"
fi

# Adapter health — warn if any adapter is degenerated, list all tracked entries.
# Health lines are pre-sorted so degenerated adapters appear first.
if (( adapter_health_count > 0 )); then
    if (( degenerated_count > 0 )); then
        echo -e "  Health:   ${RED}${degenerated_count} degenerated${RESET} of ${adapter_health_count} tracked — new facts queued until next full consolidation"
    else
        echo -e "  Health:   ${GREEN}all ${adapter_health_count} healthy${RESET}"
    fi
    while IFS=$'\t' read -r _marker hname hstatus hreason hkeys hupdated; do
        [[ -z "$hname" ]] && continue
        if [[ "$hstatus" == "degenerated" ]]; then
            status_tag="${RED}${hstatus}${RESET}"
        else
            status_tag="${DIM}${hstatus}${RESET}"
        fi
        row="    ${DIM}${hname}${RESET}  [${status_tag}]  (${hkeys} keys)"
        if [[ -n "$hreason" ]]; then
            row+="  ${DIM}${hreason}${RESET}"
        fi
        echo -e "$row"
    done <<< "$health_lines"
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

# Thermal throttle (quiet-hours). Three UI modes: on | off | hours.
#   always_on  → on     (always throttled — server room noisy, always silent)
#   always_off → off    (never throttled — cellar/server room, noise fine)
#   auto       → hours  (silent during [start, end), loud otherwise)
if [[ "$throttle_mode" != "-" && -n "$throttle_mode" ]]; then
    throttle_temp_tag=""
    if [[ "$throttle_temp_limit" != "-" && -n "$throttle_temp_limit" ]]; then
        throttle_temp_tag=" ≤${throttle_temp_limit}°C"
    fi
    case "$throttle_mode" in
        always_on)
            echo -e "  Throttle: ${CYAN}on${RESET} (${GREEN}always silent${RESET}${throttle_temp_tag})"
            ;;
        always_off)
            echo -e "  Throttle: ${CYAN}off${RESET} (${DIM}never throttles${RESET})"
            ;;
        auto)
            if [[ "$throttle_active" == "True" ]]; then
                state_tag="${GREEN}silent now${RESET}"
            else
                state_tag="${DIM}loud now${RESET}"
            fi
            echo -e "  Throttle: ${CYAN}hours${RESET} ${throttle_start}–${throttle_end}${throttle_temp_tag} | ${state_tag}"
            ;;
        *)
            echo -e "  Throttle: ${DIM}${throttle_mode}${RESET}"
            ;;
    esac
fi

# Scheduler — show interim cadence + derived full-cycle period.
# Two separate "next" markers:
#   - next interim: deterministic cadence boundary from midnight (when
#     post_session_train rolls over to a new interim adapter stamp).
#   - next full:    wall-clock time of the next systemd timer tick (the
#     full-consolidation cycle = refresh_cadence × max_interim_count).
if [[ "$refresh_cadence" == "-" || -z "$refresh_cadence" ]]; then
    echo -e "  Schedule: ${DIM}manual only${RESET} (mode: ${mode_config})"
else
    sched_line="refresh ${CYAN}${refresh_cadence}${RESET} × ${max_interim_count} = full cycle ${CYAN}${consolidation_period}${RESET} (mode: ${mode_config})"
    echo -e "  Schedule: ${sched_line}"
    # Interim line: next cadence boundary (event-driven per-session training
    # rolls over here).
    if [[ "$next_interim" != "-" ]]; then
        echo -e "            next interim in ${CYAN}${next_interim}${RESET}"
    fi
    # Full line: systemd timer wall-clock tick.
    if [[ "$scheduler_started" != "True" ]]; then
        echo -e "            next full   ${DIM}first run pending${RESET}"
    elif [[ "$next_run" != "-" ]]; then
        echo -e "            next full   in ${CYAN}${next_run}${RESET}"
    fi
fi

# Pending enrollments
if (( pending_enrollments > 0 )); then
    echo -e "  Enroll:   ${YELLOW}${pending_enrollments} awaiting name extraction${RESET}"
fi

# Slice 5a — Migrate footer (always rendered when data is present).
if [[ -n "$migrate_line" ]]; then
    IFS=$'\t' read -r _mig_marker mig_state mig_rev mig_applied <<< "$migrate_line"
    case "$mig_state" in
        live)    state_tag="${GREEN}LIVE${RESET}"        ;;
        staging) state_tag="${YELLOW}STAGING${RESET}"   ;;
        trial)   state_tag="${YELLOW}TRIAL${RESET}"     ;;
        failed)  state_tag="${RED}FAILED${RESET}"       ;;
        *)       state_tag="${DIM}${mig_state}${RESET}" ;;
    esac
    rev_tag=""
    if [[ -n "$mig_rev" ]]; then
        rev_tag=" (config rev ${DIM}${mig_rev}${RESET}"
        if [[ -n "$mig_applied" ]]; then
            rev_tag+=" applied ${DIM}${mig_applied}${RESET}"
        fi
        rev_tag+=")"
    fi
    echo -e "  Migrate:  ${state_tag}${rev_tag}"
fi

# Speaker embedding backend (pyannote wespeaker on CPU by default).
# Surfaced so a disabled or failed-load backend is visible alongside
# STT/TTS. Skip entirely when none of the fields are populated.
if [[ "$spk_emb_backend" != "-" && -n "$spk_emb_backend" ]]; then
    spk_dev_tag=$(fmt_device "$spk_emb_device")
    echo -e "  SpkEmbed: ${GREEN}${spk_emb_backend}${RESET} (${DIM}${spk_emb_model}${RESET} on ${spk_dev_tag})"
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

# STT — prepend engine family (whisper) so the backend is visible even when
# only one exists today; when more STT backends land the label flexes.
if [[ "$stt_loaded" == "True" ]]; then
    stt_dev_tag=$(fmt_device "$stt_device")
    stt_engine_prefix=""
    if [[ "$stt_engine" != "-" && -n "$stt_engine" ]]; then
        stt_engine_prefix="${stt_engine} "
    fi
    echo -e "  STT:      ${GREEN}loaded${RESET} (${stt_engine_prefix}${stt_model} on ${stt_dev_tag})"
else
    echo -e "  STT:      ${DIM}not loaded${RESET}"
fi

# TTS — mirror STT so legal CPU fallback paths are visible. Engine family
# (piper / mms_tts / piper+mms) makes mixed-backend fleets readable.
if [[ "$tts_loaded" == "True" ]]; then
    tts_dev_tag=$(fmt_device "$tts_device")
    tts_engine_prefix=""
    if [[ "$tts_engine" != "-" && -n "$tts_engine" ]]; then
        tts_engine_prefix="${tts_engine} "
    fi
    echo -e "  TTS:      ${GREEN}loaded${RESET} (${tts_engine_prefix}${tts_languages} on ${tts_dev_tag})"
else
    echo -e "  TTS:      ${DIM}not loaded${RESET}"
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
