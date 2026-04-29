#!/bin/bash
# ============================================================================
# Training Control — Pause/Resume/Status for long-running GPU experiments
# ============================================================================
#
# Extends gpu-cooldown.sh with training lifecycle management.
# Designed for multi-day experiments that run in cycles.
#
# Usage (source for functions):
#   source ~/.local/bin/training-control.sh
#
#   training_pause       # Signal: stop after current cycle/checkpoint
#   training_resume [N]  # Launch test N (8, 9, 10, 10b, 11, 13). Default: 8.
#   training_status      # Show pause state + GPU + progress for all tests
#
# Recommended aliases (add to ~/.bashrc):
#   alias tpause='training_pause'
#   alias tresume='training_resume'
#   alias tstatus='training_status'
#
# GPU lifecycle (service-level):
#   tresume stops the ParaMem service, restarts it with --defer-model
#   (cloud-only, no GPU). Training runs with exclusive GPU access.
#   When training finishes cleanly, gpu_guard.__exit__ clears the
#   PARAMEM_EXTRA_ARGS / PARAMEM_HOLD_* systemd env vars and the server
#   auto-reclaims on the next 10-min tick.
#
# Orphan recovery (SIGKILLed test, env vars left behind):
#   Auto-reclaim detects the orphan (hold set + holder PID dead) and
#   stops looping after one WARN. Run `pstatus --force-local` (or POST
#   /gpu/force-local) to clear the stamps and restart into local mode.
#
# ============================================================================

# Source GPU cooldown for gpu_temp, gpu_status, wait_for_cooldown
source ~/.local/bin/gpu-cooldown.sh

# --- Configuration ---
PAUSE_FILE="$HOME/.training_pause"
PROJECT_DIR="$HOME/projects/paramem"
PARAMEM_SERVER_PORT=8420
PYTHON_BIN="$HOME/miniforge3/envs/paramem/bin/python"

# Test registry: script name, output dir, pgrep pattern
declare -A TEST_SCRIPTS TEST_OUTPUT_DIRS TEST_PGREP
TEST_SCRIPTS[8]="experiments/test8_large_scale.py"
TEST_SCRIPTS[9]="experiments/test9_natural_recall.py"
TEST_SCRIPTS[10]="experiments/test10_grokking.py"
TEST_SCRIPTS["10b"]="experiments/test10b_diverse_rephrase.py"
TEST_SCRIPTS[11]="experiments/test11_adapter_extraction.py"
TEST_SCRIPTS[13]="experiments/test13_journal_scaffold.py"
TEST_SCRIPTS["13b"]="experiments/test13b_retention_curve.py"
TEST_SCRIPTS[14]="experiments/test14.py"

TEST_OUTPUT_DIRS[8]="outputs/test8_large_scale"
TEST_OUTPUT_DIRS[9]="outputs/test9_natural_recall"
TEST_OUTPUT_DIRS[10]="outputs/test10_grokking"
TEST_OUTPUT_DIRS["10b"]="outputs/test10b_diverse_rephrase"
TEST_OUTPUT_DIRS[11]="outputs/test11_adapter_extraction"
TEST_OUTPUT_DIRS[13]="outputs/test13_journal_scaffold"
TEST_OUTPUT_DIRS["13b"]="outputs/test13b_retention_curve"
TEST_OUTPUT_DIRS[14]="outputs/test14_pre"

TEST_PGREP[8]="test8_large_scale"
TEST_PGREP[9]="test9_natural_recall"
TEST_PGREP[10]="test10_grokking"
TEST_PGREP["10b"]="test10b_diverse_rephrase"
TEST_PGREP[11]="test11_adapter_extraction"
TEST_PGREP[13]="test13_journal_scaffold"
TEST_PGREP["13b"]="test13b_retention_curve"
TEST_PGREP[14]="test14"

# --- Colors (inherit from gpu-cooldown.sh, add extras) ---
BLUE='\033[0;34m'

# ============================================================================
# Internal helpers
# ============================================================================

_server_pid() {
    lsof -i :"$PARAMEM_SERVER_PORT" -t 2>/dev/null | head -1
}

# ---------------------------------------------------------------------------
# Windows Update lock (see scripts/wu-lock.ps1, scripts/setup-wu-lock.sh)
# ---------------------------------------------------------------------------
_WU_LOCK_PS1_WIN=""
_wu_lock_path() {
    [[ -n "$_WU_LOCK_PS1_WIN" ]] && { echo "$_WU_LOCK_PS1_WIN"; return; }
    local p="$PROJECT_DIR/scripts/wu-lock.ps1"
    [[ -f "$p" ]] || return 1
    _WU_LOCK_PS1_WIN=$(wslpath -w "$p")
    echo "$_WU_LOCK_PS1_WIN"
}

# Prints NONE / PENDING_INSTALL / PENDING_REBOOT / IN_PROGRESS (or empty on error).
_wu_check() {
    local unc
    unc=$(_wu_lock_path) || return
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$unc" -Check 2>/dev/null | tr -d '\r\n'
}

_wu_tasks_registered() {
    schtasks.exe /Query /TN 'WU-Lock-Acquire' >/dev/null 2>&1 \
        && schtasks.exe /Query /TN 'WU-Lock-Release' >/dev/null 2>&1
}

_wu_ensure_registered() {
    _wu_tasks_registered && return 0
    local setup="$PROJECT_DIR/scripts/setup-wu-lock.sh"
    if [[ ! -f "$setup" ]]; then
        echo -e "  ${RED}WU-Lock tasks missing and $setup not found${RESET}" >&2
        return 1
    fi
    echo -e "  ${YELLOW}WU-Lock scheduled tasks not registered — running one-time setup (UAC prompt will appear)${RESET}" >&2
    # Note: Start-Process -Verb RunAs does not propagate exit codes, so we can't
    # trust the return code of setup-wu-lock.sh. Verify by re-querying instead.
    bash "$setup" >&2 || true
    if ! _wu_tasks_registered; then
        echo -e "  ${RED}WU-Lock tasks still missing after setup — check UAC was accepted${RESET}" >&2
        return 1
    fi
    echo -e "  ${GREEN}WU-Lock tasks registered${RESET}" >&2
    return 0
}

_wu_acquire() {
    _wu_ensure_registered || return 1
    if ! schtasks.exe /Run /TN 'WU-Lock-Acquire' >/dev/null 2>&1; then
        echo -e "  ${RED}WU-Lock-Acquire task failed to run${RESET}" >&2
        return 1
    fi
    # Give the task a moment to write the registry, then verify expiry is in the future.
    sleep 1
    local until_utc
    until_utc=$(powershell.exe -NoProfile -Command "
        \$v=(Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings' PauseUpdatesExpiryTime -ErrorAction SilentlyContinue).PauseUpdatesExpiryTime
        if (\$v -and ([datetime]\$v).ToUniversalTime() -gt (Get-Date).ToUniversalTime()) {
            ([datetime]\$v).ToLocalTime().ToString('yyyy-MM-dd HH:mm')
        }
    " 2>/dev/null | tr -d '\r\n')
    if [[ -z "$until_utc" ]]; then
        echo -e "  ${RED}WU lock task ran but expiry not in the future — task script failed${RESET}" >&2
        return 1
    fi
    echo -e "  ${CYAN}Windows Update paused until ${until_utc} (local)${RESET}"
}

_wu_release() {
    if ! schtasks.exe /Run /TN 'WU-Lock-Release' >/dev/null 2>&1; then
        echo -e "  ${YELLOW}WU-Lock-Release task failed${RESET}" >&2
        return 1
    fi
    echo -e "  ${CYAN}Windows Update re-enabled${RESET}"
}

# True iff WU is currently paused (PauseUpdatesExpiryTime still in the future).
_wu_is_locked() {
    powershell.exe -NoProfile -Command "
        \$k='HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings'
        \$v=(Get-ItemProperty -Path \$k -Name PauseUpdatesExpiryTime -ErrorAction SilentlyContinue).PauseUpdatesExpiryTime
        if (\$v -and ([datetime]\$v).ToUniversalTime() -gt (Get-Date).ToUniversalTime()) { 'LOCKED' } else { 'OPEN' }
    " 2>/dev/null | tr -d '\r\n' | grep -q LOCKED
}

_server_mode() {
    # Returns "local", "cloud-only", or "unreachable"
    local mode
    mode=$(curl -s --max-time 2 "http://localhost:${PARAMEM_SERVER_PORT}/status" \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('mode','?'))" 2>/dev/null)
    if [[ -z "$mode" || "$mode" == "?" ]]; then
        echo "unreachable"
    else
        echo "$mode"
    fi
}

_release_server_gpu() {
    # Restart the server in cloud-only mode to fully free CUDA context.
    # SIGUSR1 unloads the model but the CUDA context retains VRAM.
    # A full process restart is the only way to reclaim all GPU memory.
    local pid=$(_server_pid)
    if [[ -z "$pid" ]]; then
        return 1
    fi

    # Restart with --defer-model: skips model loading (clean CUDA context)
    # but enables auto-reclaim when training finishes.
    systemctl --user stop paramem-server
    local i
    for i in $(seq 1 10); do
        sleep 1
        if [[ -z "$(_server_pid)" ]]; then
            break
        fi
    done

    # Start with --defer-model via environment override
    systemctl --user set-environment PARAMEM_EXTRA_ARGS="--defer-model"
    systemctl --user start paramem-server

    # Wait for server to come back (up to 30s)
    for i in $(seq 1 15); do
        sleep 2
        if [[ "$(_server_mode)" == "cloud-only" ]]; then
            return 0
        fi
    done
    return 1
}

_find_running_test() {
    # Returns the test number of the currently running test, or empty.
    # Pattern requires "python" in front of the script name so stray shells
    # with the string in a heredoc/argv (e.g. inline analysis scripts, or
    # zombie wrappers) don't register as a running test.
    for t in 8 9 10 10b 11 13 13b 14; do
        if pgrep -f "python[^ ]* .*${TEST_PGREP[$t]}" >/dev/null 2>&1; then
            echo "$t"
            return
        fi
    done
}

_find_latest_state() {
    # Find most recent state.json in a test's output dir
    local output_dir="$1"
    if [[ -d "$PROJECT_DIR/$output_dir" ]]; then
        find "$PROJECT_DIR/$output_dir" -name "state.json" -type f 2>/dev/null | sort | tail -1
    fi
}

# ============================================================================
# Public API
# ============================================================================

# acquire_gpu [--yes]
#
# Ensure the GPU is available for a new ML workload. Checks for any process
# currently using the GPU and offers resolution options:
#   - ParaMem server: restart in cloud-only mode (defer to API)
#   - Known training: warn and offer to kill or abort
#   - Unknown process: show details and offer to kill or abort
#
# Returns 0 if GPU is available, 1 if user aborted.
#
acquire_gpu() {
    local auto_yes=false
    if [[ "${1:-}" == "--yes" ]]; then
        auto_yes=true
    fi

    # Get all PIDs on the GPU
    local gpu_pids
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    if [[ -z "$gpu_pids" ]]; then
        echo -e "  ${GREEN}GPU is free.${RESET}"
        return 0
    fi

    local server_pid=$(_server_pid)

    for pid in $gpu_pids; do
        local proc_name
        proc_name=$(ps -p "$pid" -o comm= 2>/dev/null || echo "unknown")
        local proc_cmd
        proc_cmd=$(ps -p "$pid" -o args= 2>/dev/null || echo "unknown")

        if [[ "$pid" == "$server_pid" ]] || echo "$proc_cmd" | grep -q "paramem.server"; then
            # Case a: ParaMem server
            local mode=$(_server_mode)
            if [[ "$mode" == "local" || "$mode" == "cloud-only" ]]; then
                echo -e "  ${YELLOW}ParaMem server (PID ${pid}) is using the GPU.${RESET}"
                if [[ "$auto_yes" == true ]]; then
                    echo -e "  ${DIM}--yes: restarting in cloud-only mode${RESET}"
                else
                    echo -e "  Options:"
                    echo -e "    ${BOLD}d${RESET} = defer to cloud-only mode (recommended)"
                    echo -e "    ${BOLD}k${RESET} = kill the server"
                    echo -e "    ${BOLD}a${RESET} = abort"
                    read -rp "  Choice [d/k/a]: " answer
                    case "${answer,,}" in
                        ""|d)  ;;
                        k)
                            echo -e "  Killing ParaMem server..."
                            systemctl --user stop paramem-server
                            sleep 2
                            if [[ -n "$(_server_pid)" ]]; then
                                kill -9 "$pid" 2>/dev/null
                            fi
                            echo -e "  ${GREEN}Server stopped.${RESET}"
                            continue
                            ;;
                        *)
                            echo -e "  ${YELLOW}Aborted.${RESET}"
                            return 1
                            ;;
                    esac
                fi
                echo -e "  Restarting server in cloud-only mode..."
                if _release_server_gpu; then
                    echo -e "  ${GREEN}Server restarted — GPU fully released.${RESET}"
                else
                    echo -e "  ${RED}Failed to release server GPU. Aborting.${RESET}"
                    return 1
                fi
            fi
        else
            # Case b/c: another ML workload or unknown process
            local running_test=$(_find_running_test)
            if [[ -n "$running_test" ]]; then
                echo -e "  ${YELLOW}ParaMem test ${running_test} is running (PID ${pid}).${RESET}"
            else
                echo -e "  ${YELLOW}GPU occupied by: ${proc_name} (PID ${pid})${RESET}"
                echo -e "  ${DIM}  ${proc_cmd}${RESET}"
            fi

            if [[ "$auto_yes" == true ]]; then
                echo -e "  ${RED}Cannot auto-resolve unknown GPU process. Aborting.${RESET}"
                return 1
            fi

            echo -e "  Options:"
            echo -e "    ${BOLD}k${RESET} = kill process ${pid}"
            echo -e "    ${BOLD}a${RESET} = abort"
            read -rp "  Choice [k/a]: " answer
            case "${answer,,}" in
                k)
                    echo -e "  Killing PID ${pid}..."
                    kill "$pid" 2>/dev/null
                    sleep 3
                    if kill -0 "$pid" 2>/dev/null; then
                        kill -9 "$pid" 2>/dev/null
                        sleep 1
                    fi
                    if kill -0 "$pid" 2>/dev/null; then
                        echo -e "  ${RED}Failed to kill PID ${pid}.${RESET}"
                        return 1
                    fi
                    echo -e "  ${GREEN}Process killed.${RESET}"
                    ;;
                *)
                    echo -e "  ${YELLOW}Aborted.${RESET}"
                    return 1
                    ;;
            esac
        fi
    done

    # Final check
    local remaining
    remaining=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    if [[ -n "$remaining" ]]; then
        echo -e "  ${RED}GPU still occupied after cleanup. Aborting.${RESET}"
        return 1
    fi

    echo -e "  ${GREEN}GPU is ready.${RESET}"
    return 0
}

# training_pause
#
# Signal training to stop after the current cycle/checkpoint completes.
# The Python scripts check for this file between cycles/checkpoints.
#
training_pause() {
    touch "$PAUSE_FILE"
    echo -e "  ${YELLOW}Pause signal set.${RESET} Training will stop after the current cycle/checkpoint completes."
    echo -e "  ${DIM}This may take a while if a training cycle is in progress.${RESET}"
    echo -e "  ${DIM}Use 'tstatus' to check status.${RESET}"
}

# training_resume [--yes] [test_number]
#
# Clear the pause signal and launch the specified test.
# Default: test 8. Usage: tresume, tresume 13, tresume --yes 10b
# Registered tests: 8, 9, 10, 10b, 11, 13
#
training_resume() {
    local auto_yes=false
    if [[ "${1:-}" == "--yes" ]]; then
        auto_yes=true
        shift
    fi
    local test_num="${1:-8}"

    # Validate test number
    if [[ -z "${TEST_SCRIPTS[$test_num]}" ]]; then
        echo -e "  ${RED}Unknown test: ${test_num}${RESET}. Valid: 8, 9, 10, 10b, 11, 13, 13b, 14."
        return 1
    fi

    # Clear pause signal if present
    if [[ -f "$PAUSE_FILE" ]]; then
        rm "$PAUSE_FILE"
    fi

    # Check if any test is already running
    local running=$(_find_running_test)
    if [[ -n "$running" ]]; then
        if [[ "$running" == "$test_num" ]]; then
            echo -e "  ${YELLOW}Test ${test_num} is already running.${RESET} Pause signal cleared."
        else
            echo -e "  ${RED}Test ${running} is currently running.${RESET} Only one test at a time (8GB VRAM)."
        fi
        return
    fi

    # Windows Update pre-flight: if updates are pending, let user choose.
    local wu_state
    wu_state=$(_wu_check)
    case "$wu_state" in
        PENDING_INSTALL|PENDING_REBOOT|IN_PROGRESS)
            echo -e "  ${YELLOW}Windows Updates: ${wu_state}${RESET}"
            local ans=""
            if [[ "$auto_yes" != true ]]; then
                read -t 15 -rp "  Install updates first? [y/N] (default N, 15s): " ans || ans=n
            fi
            if [[ "${ans,,}" == y* ]]; then
                echo "  Aborting. Install updates, reboot, then retry tresume."
                return 1
            fi
            echo "  Proceeding — WU will be paused for this run."
            ;;
        NONE|"") ;;  # silent
        *) echo -e "  ${DIM}WU check returned: ${wu_state}${RESET}" ;;
    esac
    if ! _wu_acquire; then
        echo -e "  ${RED}Aborting tresume — Windows Update lock not engaged.${RESET}"
        return 1
    fi

    # Ensure GPU is available
    if [[ "$auto_yes" == true ]]; then
        acquire_gpu --yes || { _wu_release; return 1; }
    else
        acquire_gpu || { _wu_release; return 1; }
    fi

    local script="${TEST_SCRIPTS[$test_num]}"

    # Verify the test script exists
    if [[ ! -f "$PROJECT_DIR/$script" ]]; then
        echo -e "  ${RED}Script not found: ${script}${RESET}"
        return 1
    fi

    local log_file="$PROJECT_DIR/${TEST_OUTPUT_DIRS[$test_num]}/training.log"
    local resume_flag="--resume"
    local extra_flags=""

    # Test-specific overrides
    if [[ "$test_num" == "10b" ]]; then
        resume_flag="--resume"
        log_file="$PROJECT_DIR/${TEST_OUTPUT_DIRS[$test_num]}/training.log"
    fi

    # Test 14: detect mode from latest run_config.json so tresume 14 after a
    # pause resumes the correct mode (pre/scale/multiround) rather than
    # defaulting to --mode=pre regardless of where the run stopped.
    #
    # Also auto-detect Phase A reuse: when --mode=pre and the run dir
    # contains V1/V2/V3 done markers, the same dir is the source for the
    # extended-run variants (V3_extended/V4/V5).  Without this, V4/V5
    # would fresh-train Phase A on resume because their phase_a_reused.json
    # markers haven't been written yet — costs ~3 h per variant.
    if [[ "$test_num" == "14" ]]; then
        # run_config.json sits at <output_base>/<model>/<ts>/run_config.json
        # → depth 3 from each output_base.  Earlier -maxdepth 2 silently
        # missed everything.
        local latest_dir=$(find "$PROJECT_DIR/outputs/test14_pre" "$PROJECT_DIR/outputs/test14a" "$PROJECT_DIR/outputs/test14b" -maxdepth 3 -name "run_config.json" 2>/dev/null | sort | tail -1)
        if [[ -n "$latest_dir" ]]; then
            local run_dir=$(dirname "$latest_dir")
            local mode_value=$(python3 -c "import json; print(json.load(open('$latest_dir')).get('mode','pre'))" 2>/dev/null)
            extra_flags="--mode=$mode_value"
            # Phase A reuse pass-through: --mode=pre + V3 baseline already
            # in this run dir = extended-run pattern.  Pass the run dir
            # back through --reuse-phase-a-from so V4/V5 skip Phase A.
            if [[ "$mode_value" == "pre" && -f "$run_dir/V3/A/A_done.json" ]]; then
                extra_flags="$extra_flags --reuse-phase-a-from $run_dir"
            fi
            # 14b multiround binds to a specific 14a (scale) run via
            # scale_run.  The path is persisted in run_config.json on
            # first launch.  Pass it back through --scale-run so resume
            # binds to the same 14a instance even when later 14a runs
            # exist (find_latest_run_dir would otherwise pick the newest).
            if [[ "$mode_value" == "multiround" ]]; then
                local scale_run=$(python3 -c "import json; v=json.load(open('$latest_dir')).get('scale_run',''); print(v if v else '')" 2>/dev/null)
                if [[ -n "$scale_run" ]]; then
                    extra_flags="$extra_flags --scale-run $scale_run"
                fi
            fi
        fi
    fi

    # Test-specific model defaults
    local model_flag="--model mistral"

    mkdir -p "$(dirname "$log_file")"

    # Machine-level GPU env (PYTORCH_CUDA_ALLOC_CONF, HF_DEACTIVATE_ASYNC_LOAD,
    # …) comes from `gpu-guard env` so test scripts inherit the same allocator
    # contract as the server.  Fail loud here — this is the controlled lab
    # launch path; silently dropping the contract is exactly the failure mode
    # that produced the V4 BSOD on 2026-04-27.  bashrc keeps a soft fallback
    # so a fresh shell on a host without lab-tools still works.
    if ! command -v gpu-guard >/dev/null 2>&1; then
        echo -e "  ${RED:-}ERROR: gpu-guard not on PATH — install lab-tools first${RESET:-}" >&2
        echo -e "  See: ~/projects/lab-tools/gpu_guard/README.md" >&2
        return 1
    fi
    # Capture gpu-guard env vars as KEY=value lines.  Injected as `env` args
    # AFTER the .env injection so the machine GPU contract wins on conflict
    # — `env` processes its args left-to-right and later assignments override
    # earlier ones, so any future drift in paramem's .env can't silently
    # clobber the contract.
    local gpu_guard_env
    gpu_guard_env=$(gpu-guard env)

    echo -e "  ${GREEN}Resuming test ${test_num}...${RESET}"
    cd "$PROJECT_DIR" && \
        env $(grep -v '^#' .env | xargs) $gpu_guard_env \
        nohup "$PYTHON_BIN" "$script" $model_flag $resume_flag $extra_flags \
        >> "$log_file" 2>&1 &

    local pid=$!
    echo -e "  PID: ${CYAN}${pid}${RESET}"
    echo -e "  Log: ${DIM}${log_file}${RESET}"

    # Auto-release watcher: polls training PID, releases WU lock on exit.
    # Event-driven (not cron); dies with WSL — 2-day expiry is the safety net.
    ( while kill -0 "$pid" 2>/dev/null; do sleep 30; done; _wu_release ) &
    disown

    echo -e "  ${DIM}Use 'tstatus' to monitor, 'tpause' to stop.${RESET}"
}

# training_status
#
# Show current training state: pause signal, server mode, GPU status,
# progress for all tests.
#
training_status() {
    echo -e "${BOLD}Training Status${RESET}"
    echo "  ────────────────────────────────────────"

    # Pause state
    local running=$(_find_running_test)
    if [[ -f "$PAUSE_FILE" ]]; then
        echo -e "  State:  ${YELLOW}PAUSED${RESET} (waiting for cycle/checkpoint to finish)"
    elif [[ -n "$running" ]]; then
        echo -e "  State:  ${GREEN}RUNNING${RESET} (test ${running})"
    else
        echo -e "  State:  ${DIM}idle${RESET}"
        # Idle + WU still paused → auto-release (idempotent)
        if _wu_is_locked 2>/dev/null; then
            _wu_release
        fi
    fi

    # ParaMem server status
    local server_pid=$(_server_pid)
    if [[ -n "$server_pid" ]]; then
        local server_json
        server_json=$(curl -s --max-time 2 "http://localhost:${PARAMEM_SERVER_PORT}/status" 2>/dev/null)
        local mode=$(_server_mode)
        if [[ "$mode" == "local" ]]; then
            echo -e "  ParaMem: ${GREEN}LOCAL${RESET} (PID ${server_pid}, GPU active)"
        elif [[ "$mode" == "cloud-only" ]]; then
            local reason=""
            if [[ -n "$server_json" ]]; then
                reason=$(python3 -c "
import json, sys
d = json.loads('$server_json')
r = d.get('cloud_only_reason', '')
labels = {
    'explicit': 'explicit, auto-reclaim disabled',
    'training': 'deferred for training, auto-reclaim enabled',
    'gpu_conflict': 'GPU occupied, auto-reclaim enabled',
}
print(labels.get(r, r or ''))
" 2>/dev/null)
            fi
            if [[ -n "$reason" ]]; then
                echo -e "  ParaMem: ${YELLOW}CLOUD-ONLY${RESET} (${reason})"
            else
                echo -e "  ParaMem: ${YELLOW}CLOUD-ONLY${RESET} (PID ${server_pid})"
            fi
        else
            echo -e "  ParaMem: ${RED}UNREACHABLE${RESET} (PID ${server_pid})"
        fi
    else
        echo -e "  ParaMem: ${DIM}stopped${RESET}"
    fi

    # GPU status — show TRAINING when compute is active, cooldown labels otherwise
    if command -v nvidia-smi &>/dev/null; then
        local temp=$(gpu_temp 2>/dev/null)
        if [[ -n "$temp" ]]; then
            local power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | xargs)
            local mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | xargs)
            local mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | xargs)
            local label
            if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
                label="TRAINING"
            else
                label=$(_status_label "$temp")
            fi
            echo -e "  GPU:    $(_temp_color $temp)${temp}°C${RESET} | ${power}W | ${mem_used}/${mem_total} MiB | ${label}"
        else
            echo -e "  GPU:    ${RED}unavailable${RESET}"
        fi
    else
        echo -e "  GPU:    ${DIM}nvidia-smi not found${RESET}"
    fi

    # Show status for each test (reuses $running from above)
    for test_num in 8 9 10 10b 11 13 13b 14; do
        _show_test_status "$test_num" "$running"
    done

    echo ""
}

_show_test_status() {
    local test_num="$1"
    local running_test="$2"
    local output_dir="${TEST_OUTPUT_DIRS[$test_num]}"

    # Test 13 uses per-phase *_done.json markers, not state.json — dispatch early.
    if [[ "$test_num" == "13" ]]; then
        local latest_run_dir
        latest_run_dir=$(find "$PROJECT_DIR/$output_dir" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | sort | tail -1)
        if [[ -z "$latest_run_dir" ]]; then
            if [[ "$running_test" == "$test_num" ]]; then
                echo ""
                echo -e "  ${BOLD}Test 13${RESET} ${GREEN}RUNNING${RESET}"
                echo "  ────────────────────────────────────────"
                echo -e "  ${DIM}Preparing run directory...${RESET}"
            fi
            return
        fi
        local is_running=""
        if [[ "$running_test" == "$test_num" ]]; then
            is_running=" ${GREEN}RUNNING${RESET}"
        fi
        echo ""
        echo -e "  ${BOLD}Test 13${RESET}${is_running}"
        echo "  ────────────────────────────────────────"
        _show_test13_status "$latest_run_dir"
        return
    fi

    # Test 14 uses per-phase *_done.json markers across multiple output dirs.
    if [[ "$test_num" == "14" ]]; then
        # Find most recent run across all three test14 output dirs.
        local latest_run_dir
        latest_run_dir=$(find \
            "$PROJECT_DIR/outputs/test14_pre" \
            "$PROJECT_DIR/outputs/test14a" \
            "$PROJECT_DIR/outputs/test14b" \
            -mindepth 2 -maxdepth 2 -type d 2>/dev/null | sort | tail -1)
        if [[ -z "$latest_run_dir" ]]; then
            if [[ "$running_test" == "$test_num" ]]; then
                echo ""
                echo -e "  ${BOLD}Test 14${RESET} ${GREEN}RUNNING${RESET}"
                echo "  ────────────────────────────────────────"
                echo -e "  ${DIM}Preparing run directory...${RESET}"
            fi
            return
        fi
        local is_running=""
        local running_flag=""
        if [[ "$running_test" == "$test_num" ]]; then
            is_running=" ${GREEN}RUNNING${RESET}"
            running_flag="1"
        fi
        echo ""
        echo -e "  ${BOLD}Test 14${RESET}${is_running}"
        echo "  ────────────────────────────────────────"
        _show_test14_status "$latest_run_dir" "$running_flag"
        return
    fi

    # Test 13b uses fill_done.json marker and progress.json — dispatch early.
    # Exclude _smoke/ subtree so smoke-test run dirs never appear as the prod
    # "latest" (they sort after mistral/ in en_US.UTF-8 collation).
    if [[ "$test_num" == "13b" ]]; then
        local latest_run_dir
        latest_run_dir=$(find "$PROJECT_DIR/$output_dir" -mindepth 2 -maxdepth 2 -type d -not -path "*/_smoke/*" 2>/dev/null | sort | tail -1)
        if [[ -z "$latest_run_dir" ]]; then
            if [[ "$running_test" == "$test_num" ]]; then
                echo ""
                echo -e "  ${BOLD}Test 13b${RESET} ${GREEN}RUNNING${RESET}"
                echo "  ────────────────────────────────────────"
                echo -e "  ${DIM}Preparing run directory...${RESET}"
            fi
            return
        fi
        local is_running=""
        if [[ "$running_test" == "$test_num" ]]; then
            is_running=" ${GREEN}RUNNING${RESET}"
        fi
        echo ""
        echo -e "  ${BOLD}Test 13b${RESET}${is_running}"
        echo "  ────────────────────────────────────────"
        _show_test13b_status "$latest_run_dir"
        return
    fi

    local state_file=$(_find_latest_state "$output_dir")

    # Skip tests with no state
    if [[ -z "$state_file" || ! -f "$state_file" ]]; then
        # Still show if running — check for progress.json
        if [[ "$running_test" == "$test_num" ]]; then
            echo ""
            echo -e "  ${BOLD}Test ${test_num}${RESET} ${GREEN}RUNNING${RESET}"
            echo "  ────────────────────────────────────────"
            # Find progress.json in latest run dir
            local latest_run=$(find "$PROJECT_DIR/$output_dir" -name "progress.json" 2>/dev/null | sort | tail -1)
            if [[ -n "$latest_run" ]]; then
                local run_dir=$(dirname "$latest_run")
                read -r cur_epoch target_epoch cur_cycle num_keys wd < <(python3 -c "
import json
p=json.load(open('$latest_run'))
print(p.get('epoch','?'), p.get('target_epoch','?'), p.get('cycle','?'), p.get('keys','?'), p.get('weight_decay','?'))
" 2>/dev/null)
                echo -e "  Keys:       ${CYAN}${num_keys}${RESET}"
                echo -e "  Wt decay:   ${wd}"
                echo -e "  Training:   ${YELLOW}cycle ${cur_cycle}, E${cur_epoch} → E${target_epoch}${RESET}"
            else
                echo -e "  ${DIM}Preparing data...${RESET}"
            fi
        fi
        return
    fi

    local is_running=""
    if [[ "$running_test" == "$test_num" ]]; then
        is_running=" ${GREEN}RUNNING${RESET}"
    fi

    echo ""
    echo -e "  ${BOLD}Test ${test_num}${RESET}${is_running}"
    echo "  ────────────────────────────────────────"

    if [[ "$test_num" == "8" ]]; then
        _show_test8_status "$state_file"
    elif [[ "$test_num" == "9" ]]; then
        _show_test9_status "$state_file"
    elif [[ "$test_num" == "10" ]]; then
        _show_test10_status "$state_file"
    elif [[ "$test_num" == "10b" ]]; then
        _show_test10b_status "$state_file"
    elif [[ "$test_num" == "11" ]]; then
        _show_test11_status "$state_file"
    fi
}

_show_test8_status() {
    local state_file="$1"
    local cycle=$(python3 -c "import json; d=json.load(open('$state_file')); print(d.get('last_completed_cycle', '?'))" 2>/dev/null)
    local qa_count=$(python3 -c "import json; d=json.load(open('$state_file')); print(d.get('total_qa_pairs', '?'))" 2>/dev/null)
    local sessions=$(python3 -c "import json; d=json.load(open('$state_file')); print(d.get('total_sessions_processed', '?'))" 2>/dev/null)
    local character=$(python3 -c "import json; d=json.load(open('$state_file')); print(d.get('current_character', '?'))" 2>/dev/null)
    local last_time=$(date -r "$state_file" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "?")

    local target=$(python3 -c "
import json, glob, os
d = os.path.dirname('$state_file')
cfg = glob.glob(os.path.join(d, 'run_config.json'))
if cfg:
    print(json.load(open(cfg[0])).get('target_keys', 500))
else:
    print(500)
" 2>/dev/null)
    local pct="?"
    if [[ "$qa_count" != "?" && "$target" != "?" && "$target" -gt 0 ]]; then
        pct=$(( qa_count * 100 / target ))
    fi

    local skipped=$(python3 -c "
import json
d=json.load(open('$state_file'))
sc = d.get('skipped_cycles', [])
print(len(sc))
" 2>/dev/null)

    echo -e "  QA pairs:   ${CYAN}${qa_count} / ${target}${RESET} (${pct}%)"
    echo -e "  Cycles:     ${cycle}"
    if [[ -n "$skipped" && "$skipped" != "0" ]]; then
        echo -e "  Skipped:    ${YELLOW}${skipped}${RESET}"
    fi
    echo -e "  Sessions:   ${sessions}"
    echo -e "  Character:  ${character}"

    # Show active cycle progress if running
    local run_dir=$(dirname "$state_file")
    local active_cycle_dir=""
    local active_cycle_num=""
    for cdir in $(find "$run_dir" -maxdepth 1 -type d -name "cycle_*" | sort -V); do
        local cnum=$(basename "$cdir" | sed 's/cycle_0*//')
        if [[ "$cnum" -gt "$cycle" ]]; then
            active_cycle_dir="$cdir"
            active_cycle_num="$cnum"
        fi
    done
    if [[ -n "$active_cycle_dir" && -d "$active_cycle_dir" ]]; then
        _show_epoch_progress "$active_cycle_dir" "$active_cycle_num"
    fi

    echo -e "  Last saved: ${DIM}${last_time}${RESET}"
}

_show_test9_status() {
    local state_file="$1"
    local run_dir=$(dirname "$state_file")

    # Count completed cycles from result files
    local completed=$(find "$run_dir" -name "cycle_*_results.json" 2>/dev/null | wc -l)
    local last_time=$(date -r "$state_file" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "?")

    # Get latest cycle info
    local latest_cycle=""
    local latest_keys=""
    for result_file in $(find "$run_dir" -name "cycle_*_results.json" | sort -V | tail -1); do
        latest_cycle=$(python3 -c "import json; print(json.load(open('$result_file')).get('cycle','?'))" 2>/dev/null)
        latest_keys=$(python3 -c "import json; print(json.load(open('$result_file')).get('key_count','?'))" 2>/dev/null)
    done

    echo -e "  Cycles:     ${CYAN}${completed}${RESET} evaluated"
    if [[ -n "$latest_cycle" ]]; then
        echo -e "  Latest:     cycle ${latest_cycle} (${latest_keys} keys)"
    fi
    echo -e "  Last saved: ${DIM}${last_time}${RESET}"
}

_show_test10_status() {
    local state_file="$1"
    local run_dir=$(dirname "$state_file")

    read -r last_checkpoint num_keys weight_decay learning_rate epochs_per_cycle < <(python3 -c "
import json
d=json.load(open('$state_file'))
print(d.get('last_completed_epoch','?'), d.get('num_keys','?'), d.get('weight_decay','?'), d.get('learning_rate','?'), d.get('epochs_per_cycle','?'))
" 2>/dev/null)
    local completed_count=$(python3 -c "import json; print(len(json.load(open('$state_file')).get('completed_epochs',[])))" 2>/dev/null)
    local last_time=$(date -r "$state_file" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "?")

    echo -e "  Keys:       ${CYAN}${num_keys}${RESET}"
    echo -e "  Epoch:      ${CYAN}${last_checkpoint}${RESET} (${completed_count} cycles × ${epochs_per_cycle:-?} epochs)"
    echo -e "  Wt decay:   ${weight_decay}, LR: ${learning_rate:-?}, scheduler: constant"

    # Show latest probe results (last completed epoch)
    if [[ "$last_checkpoint" != "?" && "$last_checkpoint" != "0" ]]; then
        local probe_file="$run_dir/epoch_$(printf '%03d' $last_checkpoint)/probe_results.json"
        if [[ -f "$probe_file" ]]; then
            python3 -c "
import json
r = json.load(open('$probe_file'))
keyed = r.get('keyed_retrieval', {}).get('recall_rate', 0) * 100
direct = r.get('direct_questions', {}).get('recall_rate', 0) * 100
rephr = r.get('rephrased_questions', {}).get('recall_rate', 0) * 100
hop3 = r.get('threehop', {}).get('recall_rate', 0) * 100
hop3_n = r.get('threehop', {}).get('matched_count', 0)
hop3_t = r.get('threehop', {}).get('total', 0)
shortcut = r.get('relation_shortcut', {}).get('recall_rate', 0) * 100
hop2 = r.get('twohop', {}).get('recall_rate', 0) * 100
open_e = r.get('open_ended', {}).get('fact_recall_rate', 0) * 100
loss = r.get('train_loss', 0) or 0
print(f'  E{r[\"epoch\"]:>3}:      keyed {keyed:.0f}% | direct {direct:.0f}% | rephr {rephr:.0f}%')
print(f'            3-hop {hop3:.1f}% ({hop3_n}/{hop3_t}) | shortcut {shortcut:.1f}% | 2-hop {hop2:.1f}%')
print(f'            open {open_e:.0f}% | loss {loss:.6f}')
" 2>/dev/null
        fi

        # Show trend if multiple checkpoints exist
        local epoch_count=$(python3 -c "import json; print(len(json.load(open('$state_file')).get('completed_epochs',[])))" 2>/dev/null)
        if [[ "$epoch_count" -ge 3 ]]; then
            python3 -c "
import json, os
d = json.load(open('$state_file'))
epochs = sorted(d.get('completed_epochs', []))
# Show last 5 epochs as a trend line
recent = epochs[-5:]
hop3_trend = []
for ep in recent:
    pf = os.path.join('$run_dir', f'epoch_{ep:03d}', 'probe_results.json')
    if os.path.exists(pf):
        r = json.load(open(pf))
        hop3_trend.append(f'E{ep}:{r.get(\"threehop\",{}).get(\"recall_rate\",0)*100:.1f}%')
if hop3_trend:
    print(f'  3-hop trend: {\" → \".join(hop3_trend)}')
" 2>/dev/null
        fi
    fi

    # Show active cycle progress with bar (reuse shared function)
    if [[ -f "$run_dir/progress.json" ]]; then
        local cur_cycle=$(python3 -c "import json; print(json.load(open('$run_dir/progress.json')).get('cycle','?'))" 2>/dev/null)
        _show_epoch_progress "$run_dir" "$cur_cycle"
    fi

    echo -e "  Last saved: ${DIM}${last_time}${RESET}"
}

_show_test10b_status() {
    local state_file="$1"
    local run_dir=$(dirname "$state_file")
    local last_time=$(date -r "$state_file" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "?")

    # Read state
    local total_cp=$(python3 -c "import json; print(json.load(open('$state_file')).get('total_checkpoints','?'))" 2>/dev/null)
    local done_cp=$(python3 -c "import json; print(json.load(open('$state_file')).get('completed_checkpoints','?'))" 2>/dev/null)

    echo -e "  Checkpoints: ${CYAN}${done_cp} / ${total_cp}${RESET}"

    # Show latest result
    local summary_file="$run_dir/diverse_rephrase_summary.json"
    if [[ -f "$summary_file" ]]; then
        python3 -c "
import json
s = json.load(open('$summary_file'))
cps = s.get('checkpoints', [])
if cps:
    latest = cps[-1]
    print(f'  Latest E{latest[\"epoch\"]}: entity={latest[\"entity_match_rate\"]*100:.1f}% judge={latest[\"judge_match_rate\"]*100:.1f}%')
" 2>/dev/null
    fi

    echo -e "  Last saved: ${DIM}${last_time}${RESET}"
}

_show_test11_status() {
    local state_file="$1"
    local run_dir=$(dirname "$state_file")
    local last_time=$(date -r "$state_file" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "?")

    # Count completed session results
    local session_dir="$run_dir/session_results"
    local completed=0
    local total="?"
    if [[ -d "$session_dir" ]]; then
        completed=$(find "$session_dir" -name "session_*.json" 2>/dev/null | wc -l)
    fi

    # Get total from run_config
    if [[ -f "$run_dir/run_config.json" ]]; then
        total=$(python3 -c "import json; print(json.load(open('$run_dir/run_config.json')).get('num_sessions','?'))" 2>/dev/null)
    fi

    echo -e "  Sessions:   ${CYAN}${completed} / ${total}${RESET}"

    # Show results if complete
    if [[ -f "$run_dir/results.json" ]]; then
        python3 -c "
import json
r = json.load(open('$run_dir/results.json'))
a = r.get('adapter_off', {})
b = r.get('adapter_on', {})
print(f'  Parse:      A={a.get(\"parse_success_rate\",0)*100:.0f}% B={b.get(\"parse_success_rate\",0)*100:.0f}%')
print(f'  Triples:    A={a.get(\"mean_triples\",0):.1f} B={b.get(\"mean_triples\",0):.1f}')
fc = r.get('fact_coverage', {})
if fc:
    fa = fc.get('adapter_off', {}).get('rate', 0)
    fb = fc.get('adapter_on', {}).get('rate', 0)
    print(f'  Coverage:   A={fa*100:.1f}% B={fb*100:.1f}%')
" 2>/dev/null
    fi

    echo -e "  Last saved: ${DIM}${last_time}${RESET}"
}

_show_test13_status() {
    # Argument: path to a specific run dir, e.g.
    #   outputs/test13_journal_scaffold/mistral/20260420_215617
    local run_dir="$1"
    local model_name=$(basename "$(dirname "$run_dir")")
    local run_name=$(basename "$run_dir")

    echo -e "  Model:      ${CYAN}${model_name}${RESET}"
    echo -e "  Run:        ${DIM}${run_name}${RESET}"

    # Per-phase summary read directly from *_done.json markers.
    python3 - "$run_dir" <<'PYEOF' 2>/dev/null
import json, sys, os
run_dir = sys.argv[1]
phases = ("A", "B", "C1", "C2")
any_done = False
for p in phases:
    marker = os.path.join(run_dir, p, f"{p}_done.json")
    if not os.path.exists(marker):
        continue
    any_done = True
    try:
        d = json.load(open(marker))
    except Exception:
        continue
    first = d.get("first_perfect_epoch")
    stable = d.get("stable_perfect_epoch")
    n = d.get("n_keys")
    wall = d.get("wall_seconds")
    retention = d.get("retention_unchanged_80", {}).get("rate")
    leak = d.get("placeholder_leakage_on_fills")
    parts = [f"n={n}" if n is not None else None,
             f"first={first}" if first is not None else None,
             f"stable={stable}" if stable is not None else None,
             f"wall={wall:.0f}s" if isinstance(wall, (int, float)) else None,
             f"retention={retention:.2f}" if retention is not None else None,
             f"leakage={leak}" if leak is not None else None]
    summary = "  ".join(s for s in parts if s)
    print(f"  {p}:        \x1b[32m\u2713\x1b[0m {summary}")
if not any_done:
    print("  \x1b[2mNo phase markers yet — Phase A likely in progress.\x1b[0m")
PYEOF

    # Detect the in-progress phase by checking which marker is missing.
    local current_phase=""
    for p in A B C1 C2; do
        if [[ ! -f "$run_dir/$p/${p}_done.json" ]]; then
            current_phase="$p"
            break
        fi
    done

    # Paused marker written by test13 after a clean pause exit.
    # Takes precedence over the "Current: Phase X" line — the run isn't
    # in-progress, it's waiting for tresume.
    if [[ -f "$run_dir/paused.json" ]]; then
        local after
        after=$(python3 -c "import json; print(json.load(open('$run_dir/paused.json')).get('stopped_after_phase','?'))" 2>/dev/null)
        echo -e "  State:      ${YELLOW}PAUSED${RESET} ${DIM}(stopped after Phase ${after} — tresume to continue)${RESET}"
    elif [[ -n "$current_phase" ]]; then
        local phase_dir="$run_dir/$current_phase"
        echo -e "  Current:    ${YELLOW}Phase ${current_phase}${RESET}"
        if [[ -f "$phase_dir/progress.json" ]]; then
            _show_epoch_progress "$phase_dir" "$current_phase"
        elif [[ -d "$phase_dir/adapter" ]]; then
            local newest_slot
            newest_slot=$(ls -td "$phase_dir/adapter"/*/ 2>/dev/null | head -1)
            local mtime
            if [[ -n "$newest_slot" ]]; then
                mtime=$(date -r "$newest_slot" "+%Y-%m-%d %H:%M:%S" 2>/dev/null)
            else
                mtime=""
            fi
            echo -e "  Training:   ${DIM}starting (adapter mtime: ${mtime})${RESET}"
        else
            echo -e "  Training:   ${DIM}starting (no progress yet)${RESET}"
        fi
    fi

    # Final results.json if all phases done.
    if [[ -f "$run_dir/results.json" ]]; then
        echo -e "  Final:      ${GREEN}all phases complete${RESET}"
    fi
}

_show_test13b_status() {
    # Argument: path to a specific run dir, e.g.
    #   outputs/test13b_retention_curve/mistral/20260422_120000
    local run_dir="$1"
    local model_name=$(basename "$(dirname "$run_dir")")
    local run_name=$(basename "$run_dir")

    echo -e "  Model:      ${CYAN}${model_name}${RESET}"
    echo -e "  Run:        ${DIM}${run_name}${RESET}"

    if [[ -f "$run_dir/fill_done.json" ]]; then
        python3 - "$run_dir/fill_done.json" <<'PYEOF' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
first = d.get("first_perfect_epoch")
stable = d.get("stable_perfect_epoch")
knee = d.get("retention_knee_epoch")
n_fill = d.get("n_fill")
n_unch = d.get("n_unchanged")
final_fill = d.get("final_fill", {})
final_ret = d.get("final_retention", {})
wall = d.get("wall_seconds")
print(f"  fill:       \x1b[32m✓\x1b[0m n={n_fill}  first={first}  stable={stable}  final={final_fill.get('recall','?')}/{final_fill.get('total','?')}")
print(f"  retention:  knee=E{knee}  final={final_ret.get('recall','?')}/{final_ret.get('total','?')}  n_probed={n_unch}")
if isinstance(wall, (int, float)):
    print(f"  Wall:       {wall:.0f}s")
PYEOF
        echo -e "  Final:      ${GREEN}fill_done — complete${RESET}"
        return
    fi

    if [[ -f "$run_dir/paused.json" ]]; then
        local stopped_epoch
        stopped_epoch=$(python3 -c "import json; print(json.load(open('$run_dir/paused.json')).get('stopped_after_epoch','?'))" 2>/dev/null)
        echo -e "  State:      ${YELLOW}PAUSED${RESET} ${DIM}(stopped after epoch ${stopped_epoch} — tresume 13b to continue)${RESET}"
    elif [[ -f "$run_dir/progress.json" ]]; then
        _show_epoch_progress "$run_dir" "fill_retention_curve"
    elif [[ -f "$run_dir/fill_keyed.json" ]]; then
        echo -e "  State:      ${DIM}starting (data prepared, training not yet begun)${RESET}"
    else
        echo -e "  State:      ${DIM}no progress yet${RESET}"
    fi
}

_show_test14_status() {
    # Args:
    #   $1: path to the most recent test14 run dir
    #   $2: optional "running" flag (any non-empty value = test14 process
    #       active; suppresses the "Final: complete" line which is misleading
    #       while extended/scale/multiround phases are mid-flight on top of an
    #       already-complete phase set).
    local run_dir="$1"
    local running_flag="${2:-}"
    local model_name=$(basename "$(dirname "$run_dir")")
    local run_name=$(basename "$run_dir")

    echo -e "  Model:      ${CYAN}${model_name}${RESET}"
    echo -e "  Run:        ${DIM}${run_name}${RESET}"

    # Read mode from run_config.json
    local mode="unknown"
    if [[ -f "$run_dir/run_config.json" ]]; then
        mode=$(python3 -c "import json; print(json.load(open('$run_dir/run_config.json')).get('mode','unknown'))" 2>/dev/null)
        local n_keys=$(python3 -c "import json; print(json.load(open('$run_dir/run_config.json')).get('n_keys','?'))" 2>/dev/null)
        echo -e "  Mode:       ${CYAN}${mode}${RESET}  n_keys=${n_keys}"
    fi

    if [[ "$mode" == "pre" ]]; then
        # Show per-variant phase summary.  A *_done.json marker is the truth
        # signal for completion — pause-during-phase no longer writes the
        # marker, so its absence reliably means the phase did not converge.
        # When progress.json exists without a *_done.json, render an
        # in-flight indicator: ▶ running (process alive), ⏸ paused
        # (paused.json names this phase), or ✗ stopped (no process and no
        # paused marker — a crash).
        python3 - "$run_dir" "$running_flag" <<'PYEOF' 2>/dev/null
import json, sys, os
run_dir = sys.argv[1]
is_running = bool(sys.argv[2]) if len(sys.argv) > 2 else False

# Read top-level paused.json once so we can attribute the pause to the
# specific (variant, phase) it stopped during.  ``stopped_after_phase``
# is e.g. "during C, variant V4" or "after C, variant V4" depending on
# whether the halt was mid-phase or at a phase boundary.
paused_label = ""
paused_path = os.path.join(run_dir, "paused.json")
if os.path.exists(paused_path):
    try:
        paused_label = json.load(open(paused_path)).get("stopped_after_phase", "") or ""
    except Exception:
        paused_label = ""

def _phase_paused_here(variant: str, phase: str) -> bool:
    """True iff the run-level paused.json names this exact (variant, phase)."""
    if not paused_label.startswith("during "):
        return False
    # Format: "during {phase}[ (extended|scale|...)], variant {variant}"
    return f"during {phase}" in paused_label and f"variant {variant}" in paused_label

# Render in deterministic order; include extended-run variants
# (V3_extended / V4 / V5) when their dirs exist.  Phase A may be
# either a fresh A_done.json or a phase_a_reused.json marker.
for variant in ("V1", "V2", "V3", "V3_extended", "V4", "V5"):
    v_dir = os.path.join(run_dir, variant)
    if not os.path.isdir(v_dir):
        continue
    any_state = False
    for phase in ("A", "B", "C"):
        phase_dir = os.path.join(v_dir, phase)
        if phase == "A":
            done_marker = os.path.join(v_dir, "A", "A_done.json")
            reuse_marker = os.path.join(v_dir, "A", "phase_a_reused.json")
            if os.path.exists(done_marker):
                marker = done_marker
                tag = ""
            elif os.path.exists(reuse_marker):
                marker = reuse_marker
                tag = " (reused)"
            else:
                marker = None
        else:
            done_path = os.path.join(phase_dir, f"{phase}_done.json")
            marker = done_path if os.path.exists(done_path) else None
            tag = ""
        if marker is None:
            # No done marker: classify the in-flight state.
            progress_path = os.path.join(phase_dir, "progress.json")
            if os.path.exists(progress_path):
                try:
                    pg = json.load(open(progress_path))
                    cur = pg.get("epoch", "?")
                    total = pg.get("total_epochs") or pg.get("target_epoch") or "?"
                    keys = pg.get("keys", "?")
                    fr = pg.get("fill_rate")
                    fr_str = f", fill={fr:.2f}" if isinstance(fr, (int, float)) else ""
                    if _phase_paused_here(variant, phase):
                        label = "\x1b[33m⏸ paused\x1b[0m"
                        suffix = "\x1b[2m(no done marker — incomplete)\x1b[0m"
                    elif is_running:
                        label = "\x1b[36m▶ running\x1b[0m"
                        suffix = ""
                    else:
                        label = "\x1b[31m✗ stopped\x1b[0m"
                        suffix = "\x1b[2m(no done marker, no pause flag — likely crashed)\x1b[0m"
                    any_state = True
                    line = (
                        f"  {variant}/{phase}:    {label}  "
                        f"epoch {cur}/{total}, n={keys}{fr_str}"
                    )
                    if suffix:
                        line += f"  {suffix}"
                    print(line)
                except Exception:
                    pass
            continue
        any_state = True
        try:
            d = json.load(open(marker))
        except Exception:
            continue
        first = d.get("first_perfect_epoch")
        stable = d.get("stable_perfect_epoch")
        n = d.get("n_keys")
        wall = d.get("wall_seconds")
        parts = [f"n={n}" if n is not None else None,
                 f"first={first}" if first is not None else None,
                 f"stable={stable}" if stable is not None else None,
                 f"wall={wall:.0f}s" if isinstance(wall, (int, float)) else None]
        summary = "  ".join(s for s in parts if s)
        print(f"  {variant}/{phase}:    \x1b[32m✓\x1b[0m {summary}{tag}")
    if not any_state:
        print(f"  {variant}:      \x1b[2mnot started\x1b[0m")
if os.path.exists(os.path.join(run_dir, "pre_decision.json")):
    try:
        d = json.load(open(os.path.join(run_dir, "pre_decision.json")))
        winner = d.get("winner", "null")
        reason = d.get("reason", "")
        print(f"  Decision:   winner={winner} reason={reason}")
    except Exception:
        pass
PYEOF

        # In-flight phase: detect (variant, phase) pair with progress.json
        # but no *_done.json, and render an epoch progress bar with ETA.
        # This is the "tstatus is no longer misleading" fix — the user can
        # see exactly where training was when it paused (or where it is
        # currently working if the script is still running).
        local current_var="" current_phase=""
        for variant in V1 V2 V3 V3_extended V4 V5; do
            local v_dir="$run_dir/$variant"
            [[ -d "$v_dir" ]] || continue
            for phase in A B C; do
                local p_dir="$v_dir/$phase"
                if [[ -f "$p_dir/progress.json" && ! -f "$p_dir/${phase}_done.json" ]]; then
                    current_var="$variant"
                    current_phase="$phase"
                    break 2
                fi
            done
        done
        if [[ -n "$current_var" ]]; then
            echo -e "  Current:    ${YELLOW}${current_var}/Phase ${current_phase}${RESET}"
            _show_epoch_progress "$run_dir/$current_var/$current_phase" "${current_var}/${current_phase}"
        fi

    elif [[ "$mode" == "scale" ]]; then
        python3 - "$run_dir" "$running_flag" <<'PYEOF' 2>/dev/null
import json, sys, os
run_dir = sys.argv[1]
is_running = bool(sys.argv[2]) if len(sys.argv) > 2 else False
paused_label = ""
paused_path = os.path.join(run_dir, "paused.json")
if os.path.exists(paused_path):
    try:
        paused_label = json.load(open(paused_path)).get("stopped_after_phase", "") or ""
    except Exception:
        paused_label = ""
for phase in ("A", "B", "C"):
    phase_dir = os.path.join(run_dir, phase)
    marker = os.path.join(phase_dir, f"{phase}_done.json")
    if not os.path.exists(marker):
        # No done marker: classify the in-flight state (running / paused / stopped).
        progress_path = os.path.join(phase_dir, "progress.json")
        if os.path.exists(progress_path):
            try:
                pg = json.load(open(progress_path))
                cur = pg.get("epoch", "?")
                total = pg.get("total_epochs") or pg.get("target_epoch") or "?"
                keys = pg.get("keys", "?")
                if paused_label.startswith("during ") and f"during {phase}" in paused_label:
                    label = "\x1b[33m⏸ paused\x1b[0m"
                    suffix = "\x1b[2m(no done marker — incomplete)\x1b[0m"
                elif is_running:
                    label = "\x1b[36m▶ running\x1b[0m"
                    suffix = ""
                else:
                    label = "\x1b[31m✗ stopped\x1b[0m"
                    suffix = "\x1b[2m(no done marker, no pause flag — likely crashed)\x1b[0m"
                line = f"  {phase}:        {label}  epoch {cur}/{total}, n={keys}"
                if suffix:
                    line += f"  {suffix}"
                print(line)
            except Exception:
                pass
        continue
    try:
        d = json.load(open(marker))
    except Exception:
        continue
    first = d.get("first_perfect_epoch")
    stable = d.get("stable_perfect_epoch")
    n = d.get("n_keys")
    wall = d.get("wall_seconds")
    parts = [f"n={n}" if n is not None else None,
             f"first={first}" if first is not None else None,
             f"stable={stable}" if stable is not None else None,
             f"wall={wall:.0f}s" if isinstance(wall, (int, float)) else None]
    summary = "  ".join(s for s in parts if s)
    print(f"  {phase}:        \x1b[32m✓\x1b[0m {summary}")
PYEOF

        # In-flight epoch progress bar for scale (single A/B/C tree).
        local current_phase=""
        for phase in A B C; do
            local p_dir="$run_dir/$phase"
            if [[ -f "$p_dir/progress.json" && ! -f "$p_dir/${phase}_done.json" ]]; then
                current_phase="$phase"
                break
            fi
        done
        if [[ -n "$current_phase" ]]; then
            echo -e "  Current:    ${YELLOW}Phase ${current_phase}${RESET}"
            _show_epoch_progress "$run_dir/$current_phase" "$current_phase"
        fi

    elif [[ "$mode" == "multiround" ]]; then
        python3 - "$run_dir" <<'PYEOF' 2>/dev/null
import json, sys, os
run_dir = sys.argv[1]
# P0 marker is at run_dir top level; P1/P2/P3 markers live under
# their per-round subdirs.  Earlier code searched all four at the
# top level — P1/P2/P3 always rendered as not started.
for phase in ("P0", "P1", "P2", "P3"):
    if phase == "P0":
        marker = os.path.join(run_dir, "P0_done.json")
    else:
        marker = os.path.join(run_dir, phase, f"{phase}_done.json")
    if not os.path.exists(marker):
        continue
    try:
        d = json.load(open(marker))
    except Exception:
        continue
    if phase == "P0":
        n = d.get("n_keys")
        print(f"  P0:        \x1b[32m✓\x1b[0m n_keys={n}")
    else:
        ret_pre = d.get("retention_pre_round", {}).get("rate")
        ret_post = d.get("retention_post_touchup", {}).get("rate")
        resid = d.get("retention_corruption_residual")
        stop_ep = d.get("stop_epoch")
        parts = [f"ret_pre={ret_pre:.3f}" if ret_pre is not None else None,
                 f"ret_post={ret_post:.3f}" if ret_post is not None else None,
                 f"resid={resid:.3f}" if resid is not None else None,
                 f"stop_ep={stop_ep}" if stop_ep is not None else None]
        summary = "  ".join(s for s in parts if s)
        print(f"  {phase}:       \x1b[32m✓\x1b[0m {summary}")
PYEOF
    fi

    # Paused marker.  `stopped_after_phase` may be a phase-boundary label
    # ("after C, variant V4") OR a mid-phase label ("during C, variant V4")
    # written by `_exit_if_paused_mid_phase`; both render naturally.
    if [[ -f "$run_dir/paused.json" ]]; then
        local after epoch_at
        after=$(python3 -c "import json; print(json.load(open('$run_dir/paused.json')).get('stopped_after_phase','?'))" 2>/dev/null)
        epoch_at=$(python3 -c "import json; v=json.load(open('$run_dir/paused.json')).get('stopped_after_epoch'); print('' if v is None else f' (epoch {v})')" 2>/dev/null)
        echo -e "  State:      ${YELLOW}PAUSED${RESET} ${DIM}(stopped ${after}${epoch_at} — tresume 14 to continue)${RESET}"
    fi

    # Only show "Final: complete" when no test14 process is running.
    # results.json may exist (from an earlier 14a-pre completion) while a
    # subsequent extended / scale / multiround run is mid-flight; in that
    # case "complete" is misleading.
    if [[ -f "$run_dir/results.json" && -z "$running_flag" ]]; then
        echo -e "  Final:      ${GREEN}complete${RESET}"
    fi
}

_show_epoch_progress() {
    local dir="$1"
    local label="$2"

    local training_keys="?"
    local epoch_cur=""
    local epoch_total=""
    local epoch_offset="0"
    local cycle_started_at=""
    if [[ -f "$dir/progress.json" ]]; then
        read -r training_keys epoch_cur epoch_total epoch_offset cycle_started_at < <(python3 -c "
import json
p=json.load(open('$dir/progress.json'))
print(p.get('keys','?'), p.get('epoch','?'), p.get('total_epochs', p.get('target_epoch', '?')), p.get('epoch_offset', 0), int(p.get('cycle_started_at') or 0))
" 2>/dev/null)
    fi
    if [[ "$training_keys" == "?" && -f "$dir/keyed_pairs.json" ]]; then
        training_keys=$(python3 -c "import json; print(len(json.load(open('$dir/keyed_pairs.json'))))" 2>/dev/null)
    fi

    # Prefer cycle_started_at (authoritative) over dir mtime (noisy — shifts
    # whenever a checkpoint subdir or file is added inside $dir).
    local started=""
    if [[ -n "$cycle_started_at" && "$cycle_started_at" != "0" ]]; then
        started=$cycle_started_at
    else
        started=$(stat -c %Y "$dir" 2>/dev/null)
    fi
    local now=$(date +%s)
    local elapsed=""
    local elapsed_secs=0
    if [[ -n "$started" ]]; then
        elapsed_secs=$((now - started))
        elapsed=$(printf "%d:%02d:%02d" $((elapsed_secs/3600)) $(((elapsed_secs%3600)/60)) $((elapsed_secs%60)))
    fi

    local epoch_info=""
    if [[ -n "$epoch_cur" && "$epoch_cur" != "?" && -n "$epoch_total" && "$epoch_total" != "?" && "$epoch_total" -gt 0 ]]; then
        # Use within-cycle progress when epoch_offset is available
        local cycle_cur=$((epoch_cur - epoch_offset))
        local cycle_total=$((epoch_total - epoch_offset))
        if [[ "$cycle_total" -le 0 ]]; then
            cycle_cur=$epoch_cur
            cycle_total=$epoch_total
        fi
        local pct_done=$((cycle_cur * 100 / cycle_total))
        local bar_width=20
        local filled=$((pct_done * bar_width / 100))
        local empty=$((bar_width - filled))
        local bar=$(printf '%0.s█' $(seq 1 $filled 2>/dev/null))$(printf '%0.s░' $(seq 1 $empty 2>/dev/null))
        epoch_info="epoch ${epoch_cur}/${epoch_total} [${bar}] ${pct_done}%"

        if [[ "$cycle_cur" -gt 0 && "$elapsed_secs" -gt 0 ]]; then
            local remaining_secs=$(( elapsed_secs * (cycle_total - cycle_cur) / cycle_cur ))
            local eta=$(printf "%d:%02d:%02d" $((remaining_secs/3600)) $(((remaining_secs%3600)/60)) $((remaining_secs%60)))
            epoch_info="${epoch_info}, ETA ${eta}"
        fi
    fi

    local info=""
    if [[ -n "$label" ]]; then
        info="cycle ${label}, "
    fi
    if [[ "$training_keys" != "?" ]]; then
        info="${info}${training_keys} keys, "
    fi
    if [[ -n "$epoch_info" ]]; then
        info="${info}${epoch_info}, "
    fi
    info="${info}elapsed ${elapsed:-?}"

    echo -e "  Training:   ${YELLOW}${info}${RESET}"
}

# ============================================================================
# Main — standalone execution (show status)
# ============================================================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    training_status
fi
