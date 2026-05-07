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
#   training_resume [N]  # Launch test N (8, 9, 10, 10b, 11, 13, 13b, 14, 14s, 15).
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

# ============================================================================
# AUTHORING GUIDE — adding a new test or a probe within an existing test set
# ============================================================================
#
# Naming convention
# -----------------
#   - **Bare number** (8, 13, 14):  a top-level "test set".  Owns its own
#     output dir and script; runs phases sequentially; finalized result is
#     a multi-phase artifact in `outputs/<test_name>/<model>/<run_ts>/`.
#   - **Suffixed letter** (10b, 13b, 14s):  a peer test that shares scope
#     with a bare-number sibling.  Two flavors:
#       * "b" suffix (10b, 13b):  separate experiment with its own script
#         and output dir, but methodologically related to the sibling.
#       * "s" suffix (14s):       a probe / smoke that lives INSIDE the
#         sibling's run dir — reuses Phase A/B artifacts, runs only a
#         restricted slice of Phase C.
#
# The two cases are equally valid; pick by whether the new work needs its
# own run dir (fresh experiment) or wants to ride an existing one (probe).
#
# Invariants every entry MUST honor (this is the user-facing contract)
# --------------------------------------------------------------------
# 1. `tpause` writes ~/.training_pause.  Every test script MUST check this
#    at every epoch boundary (and at phase boundaries) and exit cleanly
#    after writing `paused.json` to its run dir.  No mid-step kills.
# 2. `tresume <N>` clears the pause flag, loads the latest checkpoint, and
#    continues from where the script stopped.  The script MUST accept
#    `--resume` and reconstruct state from on-disk markers — never from
#    persisted CLI snapshots that can drift across re-launches.
# 3. Finalized results are preserved on re-launch.  The script SHOULD
#    skip-on-done at every level (phase done-marker, per-(variant, seed)
#    `*_done.json`).  Inserting a new test/probe MUST NOT cause a re-run
#    of finalized data — only the explicitly-named new scope executes.
# 4. Run-config drift is auto-migrated, not silently overwritten.  When
#    a script-level config (e.g., LR scheduler) differs from a finalized
#    result's persisted config, the old result is renamed aside (a tag
#    suffix preserving provenance), not deleted; the new run lands fresh
#    in the canonical name.
#
# Adding a brand-new test set (own script + own output dir)
# ---------------------------------------------------------
# 1. Drop the script at `experiments/<test_name>.py`.
# 2. Add four registry rows below:
#       TEST_SCRIPTS[N]      = "experiments/<test_name>.py"
#       TEST_OUTPUT_DIRS[N]  = "outputs/<test_name>"
#       TEST_PGREP[N]        = "<test_name>"        # specific enough to
#                                                    match only this script
#       (TEST_EXTRA_FLAGS[N] = ""                   # leave unset; the
#                                                    script reads its own
#                                                    run_config.json)
# 3. Add `N` to `_find_running_test`'s iteration order (broader patterns
#    later — see "ordering rule" below).
# 4. Add `N` to `training_status`'s per-test loop.
# 5. Add `N` to the "Valid:" message in `training_resume` and to this
#    file-header `Usage` block.
# 6. The script itself must implement the four invariants above.
#
# Inserting a probe / smoke within an existing test set (peer entry)
# ------------------------------------------------------------------
# A probe is a single-(variant, seed) experiment that reuses an existing
# run dir's Phase A/B and runs only a restricted Phase C scope.  Pattern:
#       TEST_SCRIPTS["Ns"]      = TEST_SCRIPTS[N]            # same script
#       TEST_OUTPUT_DIRS["Ns"]  = TEST_OUTPUT_DIRS[N]        # same dir
#       TEST_PGREP["Ns"]        = "<test_name>.*<distinguisher>"
#       TEST_EXTRA_FLAGS["Ns"]  = "--mode=... --variant ... --phase-c-seeds ..."
#
# The probe's scope flags live in TEST_EXTRA_FLAGS, NOT in the run_dir's
# `run_config.json` — the persisted config belongs to the broader test set
# and must not be contaminated by a probe's narrower scope.  When
# TEST_EXTRA_FLAGS is set, `training_resume` uses those flags verbatim and
# skips the run_config-derived passthrough; the script still receives
# `--resume`, so tpause/tresume cycles continue from preserved checkpoints.
#
# Distinguisher pattern
# ---------------------
# The peer's TEST_PGREP must match a substring of argv that the broader
# sibling does not pass (e.g., `--variant V3` for 14s, where bare 14
# never passes `--variant`).  This is what lets `_find_running_test`
# attribute the live PID to the right entry.
#
# Ordering rule for `_find_running_test`
# --------------------------------------
# Iterate with more-specific patterns FIRST.  When a peer test is running,
# its PID's argv matches BOTH the peer's narrow pgrep AND the broader
# sibling's loose pgrep; the iteration must hit the peer first or the
# attribution is wrong.  Convention: suffixed entries (Ns, Nb, ...) come
# before the bare number N in the iteration order.
#
# Status display
# --------------
# A peer test sharing a run dir does NOT need its own per-test status
# block — the per-(variant, seed) progress is already visible in the
# sibling's status block.  The top-of-output "RUNNING (test Ns)" header
# (driven by `_find_running_test`) tells the user which entry is active.
# Add a separate status block only when the peer has its own run dir.
#
# Concurrency
# -----------
# Only one test runs at a time (8 GB VRAM).  `training_resume` checks
# `_find_running_test` and refuses to launch if anything is active —
# whether bare or peer.  This is enforced at the registry level; the
# script doesn't need its own mutex.
#
# ============================================================================

# Test registry: script name, output dir, pgrep pattern, optional fixed flags.
declare -A TEST_SCRIPTS TEST_OUTPUT_DIRS TEST_PGREP TEST_EXTRA_FLAGS
TEST_SCRIPTS[8]="experiments/test8_large_scale.py"
TEST_SCRIPTS[9]="experiments/test9_natural_recall.py"
TEST_SCRIPTS[10]="experiments/test10_grokking.py"
TEST_SCRIPTS["10b"]="experiments/test10b_diverse_rephrase.py"
TEST_SCRIPTS[11]="experiments/test11_adapter_extraction.py"
TEST_SCRIPTS[13]="experiments/test13_journal_scaffold.py"
TEST_SCRIPTS["13b"]="experiments/test13b_retention_curve.py"
TEST_SCRIPTS[14]="experiments/test14.py"
# 14s = single-(variant, seed) smoke that probes a Phase C config without
# touching the broader multi-seed sequence.  Shares run dir with test 14
# so it can reuse Phase A and Phase B from V3.  Currently configured for
# the V3/seed42 apples-to-apples validation at linear+B50+decay=600.
TEST_SCRIPTS["14s"]="experiments/test14.py"
TEST_SCRIPTS[15]="experiments/test15_retention_multiseed.py"

TEST_OUTPUT_DIRS[8]="outputs/test8_large_scale"
TEST_OUTPUT_DIRS[9]="outputs/test9_natural_recall"
TEST_OUTPUT_DIRS[10]="outputs/test10_grokking"
TEST_OUTPUT_DIRS["10b"]="outputs/test10b_diverse_rephrase"
TEST_OUTPUT_DIRS[11]="outputs/test11_adapter_extraction"
TEST_OUTPUT_DIRS[13]="outputs/test13_journal_scaffold"
TEST_OUTPUT_DIRS["13b"]="outputs/test13b_retention_curve"
TEST_OUTPUT_DIRS[14]="outputs/test14_pre"
TEST_OUTPUT_DIRS["14s"]="outputs/test14_pre"
TEST_OUTPUT_DIRS[15]="outputs/test15_retention_multiseed"

TEST_PGREP[8]="test8_large_scale"
TEST_PGREP[9]="test9_natural_recall"
TEST_PGREP[10]="test10_grokking"
TEST_PGREP["10b"]="test10b_diverse_rephrase"
TEST_PGREP[11]="test11_adapter_extraction"
TEST_PGREP[13]="test13_journal_scaffold"
TEST_PGREP["13b"]="test13b_retention_curve"
TEST_PGREP[14]="test14"
# 14s is distinguished from 14 by its --variant flag — the broader
# multi-seed never passes --variant.  Checked before 14 in the iteration
# order so the more-specific match wins.
TEST_PGREP["14s"]="test14.*--variant V3"
TEST_PGREP[15]="test15_retention_multiseed"

# Smoke flags for 14s.  Hardcoded here rather than persisted to
# run_config.json so the smoke remains an explicit, repeatable probe and
# does not contaminate the persisted multi-seed config.
TEST_EXTRA_FLAGS["14s"]="--mode=pre --variant V3 --phase-c-seeds 42 --phase-c-num-epochs 50 --lr-scheduler-type linear --phase-c-decay-steps 600"

# TODO(probe): register experiments/dataset_probe.py for long runs (>60 min).
# Planned registry values:
#   TEST_SCRIPTS["probe"]="experiments/dataset_probe.py"
#   TEST_OUTPUT_DIRS["probe"]="outputs/dataset_probe"
#   TEST_PGREP["probe"]="dataset_probe"
# Blocked on: probe's state.json shape differs from epoch-based tests
# (uses processed_session_ids list + per-session diagnostics dir, not
# last_completed_epoch). tstatus/tresume helpers branch on epoch fields
# and would need a probe-aware code path: tresume must replay the
# original CLI args (dataset, split, sample_strategy, sample_size, seed,
# model, no_train, debug) from the run's state
# args_snapshot, not derive from registry. tstatus must report
# "N/30 sessions done" instead of "epoch X/Y". Until that lands, run the
# probe directly (`python experiments/dataset_probe.py …`) and rely on
# its built-in --resume.

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
    # We deliberately full-restart instead of using the in-process
    # POST /gpu/release path (gpu_guard's default release primitive):
    # the restart is what lets us set PARAMEM_EXTRA_ARGS=--defer-model
    # before the server re-launches, so it comes back without loading
    # the model at all.
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
    # ``pgrep -f`` matches against the full argv, so unrelated shells whose
    # argv contains the script name as a literal string (e.g. a watcher
    # running ``until ! pgrep -f "python.*test14"``) would falsely register.
    # Filter the candidate PIDs by ``ps -o comm=`` and accept only those
    # whose executable name starts with ``python`` — that's the actual
    # training process, never a shell wrapper.
    #
    # Iteration order: more-specific patterns first.  Variant tests
    # (14s, 13b, 10b) share their script with a broader sibling (14, 13,
    # 10) and must be checked first so a running variant doesn't get
    # mis-attributed to the broader entry.
    for t in 8 9 10b 10 11 13b 13 14s 14 15; do
        [[ -z "${TEST_PGREP[$t]:-}" ]] && continue
        local pids
        pids=$(pgrep -f "${TEST_PGREP[$t]}" 2>/dev/null)
        for pid in $pids; do
            local comm
            comm=$(ps -p "$pid" -o comm= 2>/dev/null)
            if [[ "$comm" == python* ]]; then
                echo "$t"
                return
            fi
        done
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
        echo -e "  ${RED}Unknown test: ${test_num}${RESET}. Valid: 8, 9, 10, 10b, 11, 13, 13b, 14, 14s, 15."
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

    # Test 14 family (14, 14s, future probe variants): all run
    # experiments/test14.py.  When TEST_EXTRA_FLAGS[$test_num] is set, the
    # entry is a variant probe (e.g., 14s smoke) — use those flags
    # verbatim and skip the run_config-derived passthrough so the variant
    # doesn't inherit the broader sibling's scope.  --resume still applies,
    # so checkpoint continuation works through tpause/tresume cycles.
    if [[ "$test_num" == "14" || "$test_num" == "14s" ]] && [[ -n "${TEST_EXTRA_FLAGS[$test_num]:-}" ]]; then
        extra_flags="${TEST_EXTRA_FLAGS[$test_num]}"
    elif [[ "$test_num" == "14" ]]; then
        # Bare test 14: detect mode from latest run_config.json so tresume 14
        # after a pause resumes the correct mode (pre/scale/multiround)
        # rather than defaulting to --mode=pre regardless of where the run
        # stopped.
        # run_config.json sits at <output_base>/<model>/<ts>/run_config.json
        # → depth 3 from each output_base.  Earlier -maxdepth 2 silently
        # missed everything.
        local latest_dir=$(find "$PROJECT_DIR/outputs/test14_pre" "$PROJECT_DIR/outputs/test14a" "$PROJECT_DIR/outputs/test14b" -maxdepth 3 -name "run_config.json" 2>/dev/null | sort | tail -1)
        if [[ -n "$latest_dir" ]]; then
            local run_dir=$(dirname "$latest_dir")
            local mode_value=$(python3 -c "import json; print(json.load(open('$latest_dir')).get('mode','pre'))" 2>/dev/null)
            extra_flags="--mode=$mode_value"
            # Phase A reuse pass-through: read from run_config.json, do NOT
            # infer from disk state.  Inferring (e.g., "V3/A/A_done.json
            # exists") silently expanded the variant set to V3_extended/
            # V4-V8 on every tresume, which violates "tresume continues
            # exactly the paused work."  When the field is null/missing
            # in run_config.json, original variants (V1/V2/V3) run only;
            # the user can pass --reuse-phase-a-from on an explicit later
            # launch to add extended variants on top.
            if [[ "$mode_value" == "pre" ]]; then
                local reuse_phase_a_from=$(python3 -c "
import json
v = json.load(open('$latest_dir')).get('reuse_phase_a_from')
print(v if v else '')
" 2>/dev/null)
                if [[ -n "$reuse_phase_a_from" ]]; then
                    extra_flags="$extra_flags --reuse-phase-a-from $reuse_phase_a_from"
                fi
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
            # Multi-seed pass-through: when phase_c_seeds is set in
            # run_config.json, propagate the same list to every resume so
            # the per-(variant, seed) sub-dir layout stays consistent and
            # resume picks up where each (variant, seed) unit left off.
            local phase_c_seeds=$(python3 -c "
import json
v = json.load(open('$latest_dir')).get('phase_c_seeds')
if v:
    print(' '.join(str(s) for s in v))
" 2>/dev/null)
            if [[ -n "$phase_c_seeds" ]]; then
                extra_flags="$extra_flags --phase-c-seeds $phase_c_seeds"
            fi
            local phase_c_num_epochs=$(python3 -c "
import json
v = json.load(open('$latest_dir')).get('phase_c_num_epochs')
print(v if v is not None else '')
" 2>/dev/null)
            if [[ -n "$phase_c_num_epochs" ]]; then
                extra_flags="$extra_flags --phase-c-num-epochs $phase_c_num_epochs"
            fi
            # lr_scheduler_type pass-through: when set in run_config.json,
            # keep it across every resume so apples-to-apples conditions are
            # preserved.
            local lr_scheduler_type=$(python3 -c "
import json
v = json.load(open('$latest_dir')).get('lr_scheduler_type')
print(v if v else '')
" 2>/dev/null)
            if [[ -n "$lr_scheduler_type" ]]; then
                extra_flags="$extra_flags --lr-scheduler-type $lr_scheduler_type"
            fi
            # phase_c_decay_steps pass-through: 300 (default) decouples LR
            # decay from num_train_epochs; 0 falls back to budget-coupled
            # decay (the original bug, retained for reproducing the
            # historical numbers).
            local phase_c_decay_steps=$(python3 -c "
import json
v = json.load(open('$latest_dir')).get('phase_c_decay_steps')
print(v if v is not None else '')
" 2>/dev/null)
            if [[ -n "$phase_c_decay_steps" ]]; then
                extra_flags="$extra_flags --phase-c-decay-steps $phase_c_decay_steps"
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
    for test_num in 8 9 10 10b 11 13 13b 14 15; do
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
        # A peer test (e.g., 14s smoke) shares the run dir with bare test 14.
        # When the peer is running, the running seed's progress is visible
        # inside this block — so treat the peer as "running for test 14"
        # for status-display purposes.  Detected via shared TEST_OUTPUT_DIRS.
        local is_running=""
        local running_flag=""
        local same_run_dir="0"
        if [[ -n "$running_test" ]] && [[ "${TEST_OUTPUT_DIRS[$running_test]:-}" == "${TEST_OUTPUT_DIRS[$test_num]:-}" ]]; then
            same_run_dir="1"
        fi
        if [[ "$running_test" == "$test_num" ]] || [[ "$same_run_dir" == "1" ]]; then
            if [[ "$running_test" == "$test_num" ]]; then
                is_running=" ${GREEN}RUNNING${RESET}"
            else
                is_running=" ${DIM}(peer ${running_test} active)${RESET} ${GREEN}RUNNING${RESET}"
            fi
            running_flag="1"
        fi
        echo ""
        echo -e "  ${BOLD}Test 14${RESET}${is_running}"
        echo "  ────────────────────────────────────────"
        _show_test14_status "$latest_run_dir" "$running_flag"
        return
    fi

    # Test 15 uses per-seed *_done.json markers under seedN/{A,B,C1,C2}/.
    if [[ "$test_num" == "15" ]]; then
        local latest_run_dir
        latest_run_dir=$(find "$PROJECT_DIR/$output_dir" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | sort | tail -1)
        if [[ -z "$latest_run_dir" ]]; then
            if [[ "$running_test" == "$test_num" ]]; then
                echo ""
                echo -e "  ${BOLD}Test 15${RESET} ${GREEN}RUNNING${RESET}"
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
        echo -e "  ${BOLD}Test 15${RESET}${is_running}"
        echo "  ────────────────────────────────────────"
        _show_test15_status "$latest_run_dir"
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

_show_test15_status() {
    # Argument: path to a specific run dir, e.g.
    #   outputs/test15_retention_multiseed/mistral/20260507_HHMMSS
    #
    # Test 15 runs A→B→C1→C2 per seed under seedN/.  Status reads each
    # seed's *_done.json markers + the in-progress phase's progress.json
    # (rendered as a progress bar via _show_epoch_progress; the per-phase
    # progress.json is written by RecallEarlyStopCallback).
    # multiseed_aggregate.json lands when all seeds complete.
    local run_dir="$1"
    local model_name=$(basename "$(dirname "$run_dir")")
    local run_name=$(basename "$run_dir")

    echo -e "  Model:      ${CYAN}${model_name}${RESET}"
    echo -e "  Run:        ${DIM}${run_name}${RESET}"

    # Read seeds list from run_config.json (falls back to default seeds).
    local seeds_csv
    seeds_csv=$(python3 -c "
import json, sys
try:
    cfg = json.load(open('$run_dir/run_config.json'))
    seeds = cfg.get('seeds', [42, 7, 1337, 1, 11])
except Exception:
    seeds = [42, 7, 1337, 1, 11]
print(' '.join(str(s) for s in seeds))
" 2>/dev/null)
    if [[ -z "$seeds_csv" ]]; then
        seeds_csv="42 7 1337 1 11"
    fi

    # Per-seed phase summary.  For B and C2 we surface retention because
    # that's the experiment's primary observable.
    python3 - "$run_dir" $seeds_csv <<'PYEOF' 2>/dev/null
import json, sys, os
run_dir = sys.argv[1]
seeds = [int(s) for s in sys.argv[2:]]
GREEN = "\x1b[32m"
DIM = "\x1b[2m"
RESET = "\x1b[0m"
for s in seeds:
    seed_dir = os.path.join(run_dir, f"seed{s}")
    parts = []
    for p in ("A", "B", "C1", "C2"):
        marker = os.path.join(seed_dir, p, f"{p}_done.json")
        if not os.path.exists(marker):
            parts.append(f"{p}={DIM}-{RESET}")
            continue
        try:
            d = json.load(open(marker))
        except Exception:
            parts.append(f"{p}={DIM}?{RESET}")
            continue
        if p in ("B", "C2"):
            ret = d.get("retention_unchanged_80", {}).get("rate")
            ret_s = f"{ret:.2f}" if isinstance(ret, (int, float)) else "-"
            stable = d.get("stable_perfect_epoch")
            stable_s = f"e{stable}" if stable is not None else "-"
            parts.append(f"{GREEN}{p}{RESET}={ret_s}/{stable_s}")
        else:
            stable = d.get("stable_perfect_epoch")
            stable_s = f"e{stable}" if stable is not None else "-"
            parts.append(f"{GREEN}{p}{RESET}={stable_s}")
    print(f"  seed{s:>4}:    " + "  ".join(parts))
print(f"  {DIM}(B / C2 fields show retention/stable_epoch — retention is unchanged-80 rate.){RESET}")
PYEOF

    # Detect in-progress (seed, phase) — first seed × phase whose marker is
    # missing.  Iterate seeds in run_config order.
    local current_seed=""
    local current_phase=""
    for s in $seeds_csv; do
        for p in A B C1 C2; do
            if [[ ! -f "$run_dir/seed${s}/${p}/${p}_done.json" ]]; then
                current_seed="$s"
                current_phase="$p"
                break 2
            fi
        done
    done

    # Paused marker takes precedence — script writes paused.json at run-dir
    # level (not per-seed) on clean tpause exit.
    if [[ -f "$run_dir/paused.json" ]]; then
        local after
        after=$(python3 -c "import json; print(json.load(open('$run_dir/paused.json')).get('stopped_after_phase','?'))" 2>/dev/null)
        echo -e "  State:      ${YELLOW}PAUSED${RESET} ${DIM}(stopped after ${after} — tresume 15 to continue)${RESET}"
    elif [[ -n "$current_phase" ]]; then
        local phase_dir="$run_dir/seed${current_seed}/${current_phase}"
        echo -e "  Current:    ${YELLOW}seed${current_seed} Phase ${current_phase}${RESET}"
        if [[ -f "$phase_dir/progress.json" ]]; then
            _show_epoch_progress "$phase_dir" "$current_phase"
        else
            echo -e "  Training:   ${DIM}starting (no progress yet)${RESET}"
        fi
    fi

    # Aggregate verdict, if all seeds × all phases done.
    if [[ -f "$run_dir/multiseed_aggregate.json" ]]; then
        python3 - "$run_dir/multiseed_aggregate.json" <<'PYEOF' 2>/dev/null
import json, sys
a = json.load(open(sys.argv[1]))
n = a.get("n_completed", 0)
mb = a.get("mean_retention_B")
mc = a.get("mean_retention_C2")
ratio = a.get("ratio_C2_over_B")
thr = a.get("decision_threshold_ratio", 5.0)
GREEN = "\x1b[32m"
RED = "\x1b[31m"
RESET = "\x1b[0m"
verdict_color = GREEN if (ratio is not None and ratio >= thr) else RED
verdict = "HOLDS" if (ratio is not None and ratio >= thr) else "DOES NOT HOLD"
mb_s = f"{mb:.3f}" if isinstance(mb, (int, float)) else "?"
mc_s = f"{mc:.3f}" if isinstance(mc, (int, float)) else "?"
ratio_s = f"{ratio:.2f}" if isinstance(ratio, (int, float)) else "?"
print(f"  Aggregate:  n={n}  retB={mb_s}  retC2={mc_s}  ratio={ratio_s}  threshold={thr:.1f}× → {verdict_color}{verdict}{RESET}")
PYEOF
        echo -e "  Final:      ${GREEN}all seeds complete${RESET}"
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
import json, re, sys, os
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

# Multi-seed mode detection: if run_config.json declares phase_c_seeds,
# Phase C renders per-seed sub-dirs (<variant>/C/seed<N>/) and aggregates
# the completed ones.  Backward-compat: when phase_c_seeds is unset OR no
# seed sub-dirs exist, fall back to the legacy <variant>/C/C_done.json
# layout.
configured_seeds = []
cfg_path = os.path.join(run_dir, "run_config.json")
if os.path.exists(cfg_path):
    try:
        cfg = json.load(open(cfg_path))
        configured_seeds = cfg.get("phase_c_seeds") or []
    except Exception:
        configured_seeds = []

# A seed dir is canonical only when its name is exactly ``seed<N>``.  Migrated
# dirs (``seed42_linearLR``, ``seed42_b50``, ``seed42_linear_dbudget``) are
# legacy artifacts kept for provenance — they must not count toward the
# done/in-flight tallies for the configured seed list.
_SEED_NAME_RE = re.compile(r"^seed(\d+)$")
_configured_seed_set = set(int(s) for s in configured_seeds) if configured_seeds else None

def _is_canonical_seed_dir(name: str) -> bool:
    m = _SEED_NAME_RE.match(name)
    if not m:
        return False
    if _configured_seed_set is None:
        return True
    return int(m.group(1)) in _configured_seed_set

# Identify the single seed/phase that is actually being trained.  ``is_running``
# (test-level PID flag) tells us a python.test14 process exists, but the per-
# seed rendering needs to know WHICH seed.  We pick the progress.json with the
# most recent mtime — HF Trainer's epoch-end probe rewrites it every cycle, so
# the active seed always has the freshest mtime by a wide margin.  Stale
# progress.json files from prior sessions (paused, abandoned, crashed) lose.
def _active_progress_path() -> str:
    best_path, best_mtime = "", -1.0
    for dirpath, _dirs, files in os.walk(run_dir):
        if "progress.json" not in files:
            continue
        p = os.path.join(dirpath, "progress.json")
        try:
            m = os.path.getmtime(p)
        except OSError:
            continue
        if m > best_mtime:
            best_path, best_mtime = p, m
    return best_path

active_progress = _active_progress_path() if is_running else ""

import statistics

def _phase_c_render_multiseed(variant, c_dir):
    """Render Phase C as a multi-seed aggregate row.

    Returns True if any seed state was found (done or in-flight), False
    if no seed sub-dirs exist (caller should fall back to legacy).
    """
    if not os.path.isdir(c_dir):
        return False
    seed_subdirs = sorted(
        d for d in os.listdir(c_dir)
        if _is_canonical_seed_dir(d) and os.path.isdir(os.path.join(c_dir, d))
    )
    if not seed_subdirs:
        return False
    completed = []
    in_flight = []
    starting = []  # seed dir created but no probe has fired yet
    for sd_name in seed_subdirs:
        sd = os.path.join(c_dir, sd_name)
        done_p = os.path.join(sd, "C_done.json")
        if os.path.exists(done_p):
            try:
                d = json.load(open(done_p))
                completed.append({
                    "seed": sd_name.replace("seed", ""),
                    "first": d.get("first_perfect_epoch"),
                    "stable": d.get("stable_perfect_epoch"),
                    "wall": d.get("wall_seconds"),
                })
            except Exception:
                pass
        else:
            prog_p = os.path.join(sd, "progress.json")
            if os.path.exists(prog_p):
                try:
                    pg = json.load(open(prog_p))
                    in_flight.append({
                        "seed": sd_name.replace("seed", ""),
                        "epoch": pg.get("epoch"),
                        "total": pg.get("total_epochs") or pg.get("target_epoch"),
                        "fill_rate": pg.get("fill_rate"),
                    })
                except Exception:
                    pass
            elif os.path.isdir(os.path.join(sd, "adapter")):
                # Phase C just started for this seed but the first epoch's
                # probe hasn't completed yet (5-7 min window).  HF Trainer
                # has set up its output_dir; no progress.json written yet.
                starting.append({"seed": sd_name.replace("seed", "")})

    n_target = len(configured_seeds) if configured_seeds else len(seed_subdirs)
    n_done = len(completed)

    # Build the variant-level header line.
    if completed:
        stables = [s["stable"] for s in completed if s["stable"] is not None]
        firsts = [s["first"] for s in completed if s["first"] is not None]
        stable_summary = ""
        first_summary = ""
        if stables:
            mean_s = statistics.mean(stables)
            std_s = statistics.pstdev(stables) if len(stables) > 1 else 0.0
            stable_summary = (
                f"stable=[{','.join(str(s) for s in stables)}] "
                f"mean={mean_s:.1f}"
                + (f" ±{std_s:.1f}" if len(stables) > 1 else "")
            )
        if firsts:
            mean_f = statistics.mean(firsts)
            first_summary = f"first mean={mean_f:.1f}"
        head_label = "\x1b[32m✓\x1b[0m" if n_done >= n_target else "\x1b[33m◐\x1b[0m"
        parts = [f"{n_done}/{n_target} seeds", stable_summary, first_summary]
        summary = "  ".join(p for p in parts if p)
        print(f"  {variant}/C:    {head_label} {summary}")
    elif in_flight or starting:
        print(f"  {variant}/C:    \x1b[33m◐\x1b[0m 0/{n_target} seeds done")
    else:
        # Sub-dirs exist but nothing useful in them — show as starting
        print(f"  {variant}/C:    \x1b[2m{n_target} seeds queued\x1b[0m")

    # Per-seed in-flight detail line: classify each as paused/running/stopped.
    # "running" is awarded only to the seed whose progress.json has the most
    # recent mtime (computed once as ``active_progress``).  Other in-flight
    # seeds are stale artifacts from prior sessions — render as ✗ stopped.
    for s in in_flight:
        seed_num = s["seed"]
        cur = s["epoch"]
        total = s["total"]
        fr = s["fill_rate"]
        fr_str = f", fill={fr:.2f}" if isinstance(fr, (int, float)) else ""
        seed_progress_path = os.path.join(c_dir, f"seed{seed_num}", "progress.json")
        is_paused_here = False
        if paused_label.startswith("during "):
            is_paused_here = (
                "during C" in paused_label
                and f"variant {variant}" in paused_label
                and f"seed {seed_num}" in paused_label
            )
        if is_paused_here:
            lbl = "\x1b[33m⏸ paused\x1b[0m"
        elif is_running and seed_progress_path == active_progress:
            lbl = "\x1b[36m▶ running\x1b[0m"
        else:
            lbl = "\x1b[31m✗ stopped\x1b[0m"
        print(f"  {variant}/C/seed{seed_num}:    {lbl}  e{cur}/{total}{fr_str}")

    # Per-seed "starting" detail: Phase C just entered, no probe yet.
    # In-training train_adapter writes step-progress only into stdout
    # (not into a file we can read), so we can't show step counts here.
    # Without progress.json we can't compare mtimes, so attribute via the
    # adapter dir's mtime instead.
    for s in starting:
        seed_num = s["seed"]
        adapter_dir = os.path.join(c_dir, f"seed{seed_num}", "adapter")
        try:
            adapter_mtime = os.path.getmtime(adapter_dir)
        except OSError:
            adapter_mtime = -1.0
        active_mtime = -1.0
        if active_progress and os.path.exists(active_progress):
            try:
                active_mtime = os.path.getmtime(active_progress)
            except OSError:
                pass
        # If no progress.json exists anywhere yet, the only signal is "is_running"
        # plus a recent adapter dir.  Otherwise, only mark "starting" when this
        # seed's adapter mtime is newer than the active progress.json (i.e.,
        # we're between Phase C entry and first probe).
        is_starting_here = is_running and (
            not active_progress or adapter_mtime >= active_mtime
        )
        if is_starting_here:
            lbl = "\x1b[36m▶ starting\x1b[0m"
            suffix = "\x1b[2m(first probe pending; ~5-7 min from Phase C entry)\x1b[0m"
        else:
            lbl = "\x1b[31m✗ stopped\x1b[0m"
            suffix = "\x1b[2m(adapter dir created but no probe yet — likely crashed during setup)\x1b[0m"
        print(f"  {variant}/C/seed{seed_num}:    {lbl}  {suffix}")
    return True


# Render in deterministic order; include extended-run variants
# (V3_extended / V4-V8) when their dirs exist.  Phase A may be
# either a fresh A_done.json or a phase_a_reused.json marker.
for variant in ("V1", "V2", "V3", "V3_extended", "V4", "V5", "V6", "V7", "V8"):
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
        elif phase == "C":
            # Multi-seed-aware Phase C rendering.  Tries seed sub-dirs first;
            # falls back to legacy single-seed C_done.json when no seed
            # sub-dirs exist.
            if _phase_c_render_multiseed(variant, phase_dir):
                any_state = True
                continue
            done_path = os.path.join(phase_dir, "C_done.json")
            marker = done_path if os.path.exists(done_path) else None
            tag = ""
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
                    elif is_running and progress_path == active_progress:
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
if os.path.exists(os.path.join(run_dir, "multiseed_aggregate.json")):
    try:
        agg = json.load(open(os.path.join(run_dir, "multiseed_aggregate.json")))
        seeds = agg.get("phase_c_seeds", [])
        std = agg.get("pooled_stable_perfect_std")
        std_str = f" pooled_std={std:.2f}" if std is not None else ""
        print(f"  Aggregate:  seeds={seeds}{std_str}")
    except Exception:
        pass
PYEOF

        # In-flight phase: locate the (variant, phase[, seed]) currently being
        # trained by picking the progress.json with the most recent mtime.
        # The training process rewrites that file on every epoch-end probe,
        # so its mtime is the freshest by a wide margin.  Stale progress.json
        # files from prior sessions (paused, abandoned, crashed) lose this
        # comparison automatically.  Skipped entirely when the test isn't
        # running — without a live process, there is no "current" cycle.
        if [[ -n "$running_flag" ]]; then
            local active_progress
            active_progress=$(find "$run_dir" -name progress.json -printf '%T@ %p\n' 2>/dev/null \
                | sort -rn | head -1 | awk '{print $2}')
            if [[ -n "$active_progress" ]]; then
                # Strip $run_dir/ prefix and /progress.json suffix to recover
                # the relative path, then split into variant/phase[/seed].
                local rel="${active_progress#$run_dir/}"
                rel="${rel%/progress.json}"
                local current_var="${rel%%/*}"
                local rest="${rel#*/}"
                local current_phase="${rest%%/*}"
                local current_seed=""
                if [[ "$rest" == */* ]]; then
                    current_seed="${rest#*/}"
                fi
                # Skip rendering if the phase already has a *_done.json marker
                # (e.g., legacy C/C_done.json plus a fresh seed42/progress.json
                # from a re-run — the seed dir is what's active, not the phase).
                local progress_dir="$run_dir/$current_var/$current_phase"
                [[ -n "$current_seed" ]] && progress_dir="$progress_dir/$current_seed"
                if [[ ! -f "$progress_dir/${current_phase}_done.json" ]]; then
                    local label_path="${current_var}/Phase ${current_phase}"
                    [[ -n "$current_seed" ]] && label_path="${label_path}/${current_seed}"
                    echo -e "  Current:    ${YELLOW}${label_path}${RESET}"
                    _show_epoch_progress "$progress_dir" "${label_path}"
                fi
            fi
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

# Identify the active progress.json by mtime (see pre-mode helper for rationale).
def _active_progress_path() -> str:
    best_path, best_mtime = "", -1.0
    for dirpath, _dirs, files in os.walk(run_dir):
        if "progress.json" not in files:
            continue
        p = os.path.join(dirpath, "progress.json")
        try:
            m = os.path.getmtime(p)
        except OSError:
            continue
        if m > best_mtime:
            best_path, best_mtime = p, m
    return best_path

active_progress = _active_progress_path() if is_running else ""

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
                elif is_running and progress_path == active_progress:
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

    # "Final: complete" suppression rules:
    # 1. results.json may exist from an earlier single-seed completion while
    #    a subsequent extended/scale/multiround run is mid-flight; in that
    #    case "complete" is misleading.
    # 2. In multi-seed mode (run_config.json declares phase_c_seeds), the
    #    legacy results.json reflects only the single-seed result, not the
    #    multi-seed batch.  Suppress unless multiseed_aggregate.json shows
    #    every variant has every requested seed done.
    if [[ -f "$run_dir/results.json" && -z "$running_flag" ]]; then
        local has_multiseed=$(python3 -c "
import json
cfg=json.load(open('$run_dir/run_config.json'))
print('1' if cfg.get('phase_c_seeds') else '')
" 2>/dev/null)
        if [[ -z "$has_multiseed" ]]; then
            echo -e "  Final:      ${GREEN}complete${RESET}"
        else
            # Multi-seed: complete iff every variant×seed has C_done.json.
            local multiseed_complete=$(python3 -c "
import json, os
cfg=json.load(open('$run_dir/run_config.json'))
seeds=cfg.get('phase_c_seeds') or []
for v in os.listdir('$run_dir'):
    vdir=os.path.join('$run_dir', v)
    if not os.path.isdir(vdir):
        continue
    cdir=os.path.join(vdir, 'C')
    if not os.path.isdir(cdir):
        continue
    for s in seeds:
        if not os.path.exists(os.path.join(cdir, f'seed{s}', 'C_done.json')):
            print('')  # incomplete
            break
    else:
        continue
    break
else:
    print('1')
" 2>/dev/null)
            if [[ -n "$multiseed_complete" ]]; then
                echo -e "  Final:      ${GREEN}multi-seed batch complete${RESET}"
            fi
        fi
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
