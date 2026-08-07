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
#   training_resume [N]  # Launch test N (10b, 11, 16, quad, lme).
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
#   stops looping after one WARN. Run `pstatus --acquire` (or POST
#   /gpu/acquire) to clear the stamps and reload into local mode in-process.
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
# sibling does not pass (e.g., `--variant V3` for `Ns`, where bare `N`
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
TEST_SCRIPTS["10b"]="experiments/test10b_diverse_rephrase.py"
TEST_SCRIPTS[11]="experiments/test11_adapter_extraction.py"
TEST_SCRIPTS[16]="experiments/test16_repair_sweep.py"
# quad = Quadruple-encoded adapter scaling probe (replaces the pending Test 8
# continuation as the new path to the recall ceiling). Single-phase per
# cycle: train-then-probe over the source graph's quadruples. Extends via
# --resume --n-keys N (N > saved). Pause-aware via TrainingHooks +
# per-25-key probe checkpoints.
TEST_SCRIPTS["quad"]="experiments/quadruple_adapter.py"
# LME = LongMemEval → graph_snapshot.json builder (graph source for the quad-adapter probe).
# Incrementally extracts LME sessions and accumulates triples into a
# canonical graph_snapshot.json used as --graph-snapshot for the quad probe.
# Pause-aware at every session boundary. Resume via tresume lme.
TEST_SCRIPTS["lme"]="experiments/lme_graph_builder.py"

TEST_OUTPUT_DIRS["10b"]="outputs/test10b_diverse_rephrase"
TEST_OUTPUT_DIRS[11]="outputs/test11_adapter_extraction"
TEST_OUTPUT_DIRS[16]="outputs/test16_repair_sweep"
TEST_OUTPUT_DIRS["quad"]="outputs/quad_scale"
TEST_OUTPUT_DIRS["lme"]="outputs/lme_graph"

TEST_PGREP["10b"]="test10b_diverse_rephrase"
TEST_PGREP[11]="test11_adapter_extraction"
TEST_PGREP[16]="test16_repair_sweep"
TEST_PGREP["quad"]="quadruple_adapter"
TEST_PGREP["lme"]="lme_graph_builder"

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
    # Iteration order: more-specific patterns first.  A variant test that
    # shares its script with a broader bare-number sibling must be checked
    # before that sibling so a running variant doesn't get mis-attributed
    # to the broader entry.
    for t in 10b 11 16 quad lme; do
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
# Default: quad. Usage: tresume, tresume 16, tresume --yes 10b
# Registered tests: 10b, 11, 16, quad, lme
#
training_resume() {
    local auto_yes=false
    if [[ "${1:-}" == "--yes" ]]; then
        auto_yes=true
        shift
    fi
    local test_num="${1:-quad}"

    # Validate test number
    if [[ -z "${TEST_SCRIPTS[$test_num]}" ]]; then
        echo -e "  ${RED}Unknown test: ${test_num}${RESET}. Valid: 10b, 11, 16, quad, lme."
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

    # When TEST_EXTRA_FLAGS is set for this slot (a peer/probe entry — see
    # the AUTHORING GUIDE above), use those flags verbatim.  --resume still
    # applies, so checkpoint continuation works through tpause/tresume
    # cycles.
    if [[ -n "${TEST_EXTRA_FLAGS[$test_num]:-}" ]]; then
        extra_flags="${TEST_EXTRA_FLAGS[$test_num]}"
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
    for test_num in 10b 11 16 quad lme; do
        _show_test_status "$test_num" "$running"
    done

    echo ""
}

_show_test_status() {
    local test_num="$1"
    local running_test="$2"
    local output_dir="${TEST_OUTPUT_DIRS[$test_num]}"

    # Test 16 uses per-seed *_done.json markers under seedN/{base_D,corrupted_D,repair_*}/.
    # Exclude the _smoke/ dir itself (it sorts after YYYYMMDD_* timestamps in ASCII) AND its
    # subtree, so smoke-test run dirs never appear as the prod "latest".
    if [[ "$test_num" == "16" ]]; then
        local latest_run_dir
        latest_run_dir=$(find "$PROJECT_DIR/$output_dir" -mindepth 2 -maxdepth 2 -type d -not -name "_smoke" -not -path "*/_smoke/*" 2>/dev/null | sort | tail -1)
        if [[ -z "$latest_run_dir" ]]; then
            if [[ "$running_test" == "$test_num" ]]; then
                echo ""
                echo -e "  ${BOLD}Test 16${RESET} ${GREEN}RUNNING${RESET}"
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
        echo -e "  ${BOLD}Test 16${RESET}${is_running}"
        echo "  ────────────────────────────────────────"
        _show_test16_status "$latest_run_dir"
        return
    fi

    # quad uses per-phase *_done.json markers, not state.json — dispatch early.
    if [[ "$test_num" == "quad" ]]; then
        local latest_run_dir
        latest_run_dir=$(find "$PROJECT_DIR/$output_dir" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | sort | tail -1)
        local is_running=""
        if [[ "$running_test" == "$test_num" ]]; then
            is_running=" ${GREEN}RUNNING${RESET}"
        fi
        echo ""
        echo -e "  ${BOLD}Quadruple Adapter${RESET}${is_running}"
        echo "  ────────────────────────────────────────"
        if [[ -z "$latest_run_dir" ]]; then
            echo -e "  ${DIM}not started${RESET}"
        else
            _show_test_quad_status "$latest_run_dir"
        fi
        return
    fi

    # LME graph builder uses build_state.json / graph_done.json, not state.json.
    if [[ "$test_num" == "lme" ]]; then
        local is_running=""
        if [[ "$running_test" == "$test_num" ]]; then
            is_running=" ${GREEN}RUNNING${RESET}"
        fi
        echo ""
        echo -e "  ${BOLD}LME Graph Builder${RESET}${is_running}"
        echo "  ────────────────────────────────────────"
        _show_test_lme_status "$PROJECT_DIR/$output_dir"
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

    if [[ "$test_num" == "10b" ]]; then
        _show_test10b_status "$state_file"
    elif [[ "$test_num" == "11" ]]; then
        _show_test11_status "$state_file"
    fi
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

_show_test16_status() {
    # Argument: path to a specific run dir, e.g.
    #   outputs/test16_repair_sweep/mistral/20260511_HHMMSS
    #
    # Test 16 runs base_D → corrupted_D → repair_D_* per seed.
    # Status reads *_done.json markers + the most-recent progress.json.
    local run_dir="$1"
    local model_name=$(basename "$(dirname "$run_dir")")
    local run_name=$(basename "$run_dir")

    echo -e "  Model:      ${CYAN}${model_name}${RESET}"
    echo -e "  Run:        ${DIM}${run_name}${RESET}"

    # Overall cells progress bar + test-level ETA from trailing-window mean
    # per-cell wall time.  "Cells" counts repair + spotcheck cells (matches the
    # aggregate semantics).  ETA window is 5; underestimates when the next
    # phase is a slow base_D rather than a fast repair cell — caveat surfaced
    # in the line.
    python3 - "$run_dir" <<'PYEOF' 2>/dev/null
import json, os, glob, sys
run_dir = sys.argv[1]
try:
    cfg = json.load(open(os.path.join(run_dir, "run_config.json")))
except Exception:
    cfg = {}
seeds = cfg.get("seeds", [42])
depths = cfg.get("depths_past_floor") or cfg.get("depths") or [0, 10, 30]
repair_grid = cfg.get("repair_grid", [])
spotcheck_depth = cfg.get("spotcheck_depth", -1)
sc_per_seed = sum(1 for d in depths if d == spotcheck_depth)
total = len(seeds) * (len(repair_grid) * len(depths) + sc_per_seed)
markers = sorted(
    glob.glob(os.path.join(run_dir, "seed*", "repair_*", "repair_*_done.json")),
    key=os.path.getmtime,
)
done = len(markers)
CYAN, YELLOW, DIM, RESET = "\x1b[36m", "\x1b[33m", "\x1b[2m", "\x1b[0m"
if total > 0:
    pct = 100 * done // total
    width = 20
    filled = pct * width // 100
    bar = "█" * filled + "░" * (width - filled)
    print(f"  Cells:      {CYAN}{done}/{total}{RESET}  [{bar}]  {pct}%")
if done >= 2 and total > done:
    window = min(done, 5)
    recent = markers[-window:]
    times = [os.path.getmtime(p) for p in recent]
    intervals = [t2 - t1 for t1, t2 in zip(times, times[1:])]
    if intervals:
        mean = sum(intervals) / len(intervals)
        remaining = total - done
        eta = int(mean * remaining)
        eh, er = divmod(eta, 3600)
        em, es = divmod(er, 60)
        am, asec = divmod(int(mean), 60)
        print(
            f"  ETA:        {YELLOW}{eh:d}:{em:02d}:{es:02d}{RESET}  "
            f"{DIM}(avg {am}:{asec:02d}/cell × {remaining} cells, window={window}){RESET}"
        )
PYEOF

    # Read seeds from run_config.json for the per-seed summary.
    local seeds_csv
    seeds_csv=$(python3 -c "
import json, sys
try:
    cfg = json.load(open('$run_dir/run_config.json'))
    seeds = cfg.get('seeds', [42])
except Exception:
    seeds = [42]
print(' '.join(str(s) for s in seeds))
" 2>/dev/null)
    [[ -z "$seeds_csv" ]] && seeds_csv="42"

    # Per-seed summary: counts as fractions against per-seed scope, with a
    # ✓ DONE marker when every base_D, corrupted_D, and repair cell for that
    # seed has its *_done.json marker.  Totals derive from run_config.json:
    # base/corrupted = len(depths); repair_cells = len(repair_grid)*len(depths)
    # plus the spotcheck cell when spotcheck_depth ∈ depths.
    python3 - "$run_dir" $seeds_csv <<PYEOF 2>/dev/null
import json, sys, os, glob
run_dir = sys.argv[1]
seeds = [int(s) for s in sys.argv[2:]]
try:
    cfg = json.load(open(os.path.join(run_dir, "run_config.json")))
except Exception:
    cfg = {}
depths = cfg.get("depths_past_floor") or cfg.get("depths") or [0, 10, 30]
repair_grid = cfg.get("repair_grid", [])
spotcheck_depth = cfg.get("spotcheck_depth", -1)
sc_per_seed = sum(1 for d in depths if d == spotcheck_depth)
base_total = len(depths)
corrupted_total = len(depths)
repair_total = len(repair_grid) * len(depths) + sc_per_seed
GREEN = "\x1b[32m"
DIM = "\x1b[2m"
YELLOW = "\x1b[33m"
RESET = "\x1b[0m"
for s in seeds:
    seed_dir = os.path.join(run_dir, f"seed{s}")
    base_done = len(glob.glob(os.path.join(seed_dir, "base_*", "base_*_done.json")))
    corrupted_done = len(glob.glob(os.path.join(seed_dir, "corrupted_*", "corrupted_*_done.json")))
    repair_done = len(glob.glob(os.path.join(seed_dir, "repair_*", "repair_*_done.json")))
    status = (
        f"base={base_done}/{base_total}  "
        f"corrupted={corrupted_done}/{corrupted_total}  "
        f"repair_cells={repair_done}/{repair_total}"
    )
    all_done = (
        base_done >= base_total
        and corrupted_done >= corrupted_total
        and repair_done >= repair_total
    )
    if all_done:
        status += f"  {GREEN}✓ DONE{RESET}"
    latest_rp2 = None
    for cd_path in sorted(glob.glob(os.path.join(seed_dir, "corrupted_*", "corrupted_*_done.json"))):
        try:
            d = json.load(open(cd_path))
            latest_rp2 = d.get("rp2_rate")
        except Exception:
            pass
    if latest_rp2 is not None:
        status += f"  latest_RP2={latest_rp2:.3f}"
    print(f"  seed{s:>4}:    {status}")
PYEOF

    # In-flight phase bar (via shared helper).  Pick the newest progress.json
    # by mtime whose phase_dir does NOT yet have its *_done.json marker — that
    # is the currently-training phase.  Mtime-sort, not lexicographic — the
    # latter renders a stale phase when an older phase dir sorts after a newer
    # one (e.g. repair_50_lr5e-05_ep3 > repair_50_lr5e-05_ep1 lexicographically
    # but the older was done first).
    local latest_progress
    latest_progress=$(find "$run_dir" -name "progress.json" -not -path "*/_smoke/*" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | awk '{print $2}')
    if [[ -n "$latest_progress" ]]; then
        local phase_dir=$(dirname "$latest_progress")
        local phase_name=$(basename "$phase_dir")
        if [[ ! -f "$phase_dir/${phase_name}_done.json" ]]; then
            _show_epoch_progress "$phase_dir" "$phase_name"
        fi
    fi

    # Paused marker.
    if [[ -f "$run_dir/paused.json" ]]; then
        local after
        after=$(python3 -c "import json; print(json.load(open('$run_dir/paused.json')).get('stopped_after_phase','?'))" 2>/dev/null)
        echo -e "  State:      ${YELLOW}PAUSED${RESET} ${DIM}(stopped after ${after} — tresume 16 to continue)${RESET}"
    fi

    # Aggregate if present.
    if [[ -f "$run_dir/test16_aggregate.json" ]]; then
        python3 - "$run_dir/test16_aggregate.json" <<'PYEOF' 2>/dev/null
import json, sys
a = json.load(open(sys.argv[1]))
n = a.get("n_completed_cells", 0)
CYAN = "\x1b[36m"
RESET = "\x1b[0m"
print(f"  Aggregate:  {CYAN}{n}{RESET} cell rows completed")
PYEOF
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

_show_test_quad_status() {
    # Argument: path to the latest quad-adapter run dir, e.g.
    #   outputs/quad_scale/mistral/20260511_120000
    #
    # Reads: run_config.json, train_done.json, probe_done.json,
    #        probe_results.json, epoch_log.json, paused.json, metrics.json.
    local run_dir="$1"
    local model_name
    model_name=$(basename "$(dirname "$run_dir")")
    local run_name
    run_name=$(basename "$run_dir")

    echo -e "  Model:      ${CYAN}${model_name}${RESET}"
    echo -e "  Run:        ${DIM}${run_name}${RESET}"

    # Read run_config.json.
    if [[ -f "$run_dir/run_config.json" ]]; then
        python3 - "$run_dir/run_config.json" <<'PYEOF' 2>/dev/null
import json, sys, os
cfg = json.load(open(sys.argv[1]))
snap = cfg.get("graph_snapshot", "?")
# Shorten snapshot path to basename if long.
if len(snap) > 60:
    snap = "..." + snap[-57:]
n_keys = cfg.get("n_keys", "?")
num_epochs = cfg.get("num_epochs", "?")
rank = cfg.get("rank", "?")
es_from = cfg.get("es_from_epoch", "?")
es_win = cfg.get("es_window", "?")
CYAN = "\x1b[36m"
DIM = "\x1b[2m"
RESET = "\x1b[0m"
print(f"  Source:     {DIM}{snap}{RESET}")
print(f"  Config:     n_keys={CYAN}{n_keys}{RESET}  epochs={num_epochs}  rank={rank}  es_from={es_from}  es_window={es_win}")
PYEOF
    fi

    # Phase summary line.
    python3 - "$run_dir" <<'PYEOF' 2>/dev/null
import json, sys, os, glob

run_dir = sys.argv[1]
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
DIM = "\x1b[2m"
RESET = "\x1b[0m"
CYAN = "\x1b[36m"

# Train phase.
train_done = os.path.exists(os.path.join(run_dir, "train_done.json"))
if train_done:
    try:
        td = json.load(open(os.path.join(run_dir, "train_done.json")))
        stop_e = td.get("stop_epoch")
        req_e = td.get("n_epochs_requested", "?")
        run_e = td.get("n_epochs_run", stop_e or req_e)
        first = td.get("first_perfect_epoch")
        stable = td.get("stable_perfect_epoch")
        train_s = f"{GREEN}✓{RESET} e{run_e}/{req_e}"
        if first is not None:
            train_s += f"  first={first}"
        if stable is not None:
            train_s += f"  stable={stable}"
    except Exception:
        train_s = f"{GREEN}✓{RESET}"
else:
    # Check epoch_log.json for in-progress epoch.
    el_path = os.path.join(run_dir, "epoch_log.json")
    if os.path.exists(el_path):
        try:
            el = json.load(open(el_path))
            if el:
                last = el[-1]
                ep = last.get("epoch", "?")
                sr = last.get("strict_rate")
                sr_s = f"{sr:.3f}" if isinstance(sr, float) else "?"
                train_s = f"{YELLOW}◐{RESET} e{ep}  strict={sr_s}"
            else:
                train_s = f"{DIM}-{RESET}"
        except Exception:
            train_s = f"{DIM}-{RESET}"
    else:
        # Check for HF checkpoint dirs as a heartbeat.
        ckpts = glob.glob(os.path.join(run_dir, "adapter", "checkpoint-*"))
        if ckpts:
            # Numeric sort to get latest.
            nums = []
            for c in ckpts:
                try:
                    nums.append(int(os.path.basename(c).split("-")[-1]))
                except ValueError:
                    pass
            latest_step = max(nums) if nums else "?"
            train_s = f"{YELLOW}◐{RESET} step {latest_step}"
        else:
            train_s = f"{DIM}-{RESET}"

# Probe phase.
probe_done = os.path.exists(os.path.join(run_dir, "probe_done.json"))
if probe_done:
    probe_s = f"{GREEN}✓{RESET}"
else:
    pr_path = os.path.join(run_dir, "probe_results.json")
    if os.path.exists(pr_path):
        try:
            pr = json.load(open(pr_path))
            n_done = len(pr)
            cfg = json.load(open(os.path.join(run_dir, "run_config.json")))
            n_total = cfg.get("n_keys") or n_done
            probe_s = f"{YELLOW}◐{RESET} {n_done}/{n_total}"
        except Exception:
            probe_s = f"{YELLOW}◐{RESET}"
    else:
        probe_s = f"{DIM}-{RESET}"

print(f"  Phases:     train: [{train_s}]  probe: [{probe_s}]")
PYEOF

    # Metrics if complete.
    if [[ -f "$run_dir/metrics.json" ]]; then
        python3 - "$run_dir/metrics.json" <<'PYEOF' 2>/dev/null
import json, sys
m = json.load(open(sys.argv[1])).get("overall", {})
strict = m.get("source_triple_recovered_rate")
so = m.get("subject_object_match_rate")
n = m.get("n", "?")
GREEN = "\x1b[32m"
RESET = "\x1b[0m"
strict_s = f"{strict:.1%}" if isinstance(strict, float) else "?"
so_s = f"{so:.1%}" if isinstance(so, float) else "?"
print(f"  Results:    {GREEN}strict={strict_s}  s+o={so_s}  n={n}{RESET}")
PYEOF
    fi

    # Paused marker.
    if [[ -f "$run_dir/paused.json" ]]; then
        python3 - "$run_dir/paused.json" <<'PYEOF' 2>/dev/null
import json, sys
p = json.load(open(sys.argv[1]))
after = p.get("stopped_after", "?")
ckpt = p.get("latest_checkpoint")
YELLOW = "\x1b[33m"
DIM = "\x1b[2m"
RESET = "\x1b[0m"
msg = f"  State:      {YELLOW}PAUSED{RESET} {DIM}(stopped after {after}"
if ckpt:
    import os
    msg += f", checkpoint: {os.path.basename(ckpt)}"
msg += f" — tresume quad to continue){RESET}"
print(msg)
PYEOF
    fi

    # Last-updated timestamp from run directory mtime.
    local last_time
    last_time=$(date -r "$run_dir" "+%Y-%m-%d %H:%M" 2>/dev/null || echo "?")
    echo -e "  Updated:    ${DIM}${last_time}${RESET}"
}

_show_test_lme_status() {
    # Argument: path to the canonical LME output dir (outputs/lme_graph/).
    # Reads: build_state.json, graph_done.json, paused.json.
    local output_dir="$1"

    if [[ ! -d "$output_dir" ]]; then
        echo -e "  ${DIM}not started${RESET}"
        return
    fi

    python3 - "$output_dir" <<'PYEOF' 2>/dev/null
import json, sys, os, time

output_dir = sys.argv[1]
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
DIM = "\x1b[2m"
CYAN = "\x1b[36m"
RESET = "\x1b[0m"

state_path = os.path.join(output_dir, "build_state.json")
done_path = os.path.join(output_dir, "graph_done.json")
paused_path = os.path.join(output_dir, "paused.json")

state = {}
done = {}
paused = {}

if os.path.exists(state_path):
    try:
        state = json.load(open(state_path))
    except Exception:
        pass

if os.path.exists(done_path):
    try:
        done = json.load(open(done_path))
    except Exception:
        pass

if os.path.exists(paused_path):
    try:
        paused = json.load(open(paused_path))
    except Exception:
        pass

# Key fields.
n_sessions = state.get("n_unique_triples") and len(state.get("sessions_done", []))
n_sessions_done = len(state.get("sessions_done", []))
n_triples = state.get("n_unique_triples", 0)
target = state.get("target_keys")
lme_split = state.get("lme_split", "?")
lme_seed = state.get("lme_seed", "?")
updated_at = state.get("updated_at")

print(f"  Split:      {CYAN}{lme_split}{RESET}  seed={lme_seed}")

if target is not None:
    triple_s = f"{n_triples}/{target}"
    pct = int(n_triples * 100 / target) if target > 0 else 0
    print(f"  Triples:    {CYAN}{triple_s}{RESET} ({pct}%)")
elif n_triples > 0:
    print(f"  Triples:    {CYAN}{n_triples}{RESET}")

if n_sessions_done > 0:
    print(f"  Sessions:   {n_sessions_done} extracted")

# State.
if done:
    final_n = done.get("n_triples", n_triples)
    final_s = done.get("n_sessions_extracted", n_sessions_done)
    print(f"  State:      {GREEN}DONE{RESET} ({final_n} triples, {final_s} sessions)")
elif paused:
    after = paused.get("stopped_after_session", "?")
    p_n = paused.get("n_sessions_extracted", "?")
    p_t = paused.get("n_unique_triples", "?")
    ts = paused.get("timestamp")
    ts_s = ""
    if ts:
        import datetime
        ts_s = " @ " + datetime.datetime.fromtimestamp(ts).strftime("%H:%M")
    print(f"  State:      {YELLOW}PAUSED{RESET} {DIM}(after session {after}, {p_t} triples{ts_s} — tresume lme to continue){RESET}")
elif n_triples > 0:
    print(f"  State:      {YELLOW}RUNNING{RESET}")
else:
    print(f"  State:      {DIM}no progress yet{RESET}")

# Last-updated time.
if updated_at:
    import datetime
    dt = datetime.datetime.fromtimestamp(updated_at).strftime("%Y-%m-%d %H:%M")
    print(f"  Updated:    {DIM}{dt}{RESET}")
PYEOF
}

# ============================================================================
# Main — standalone execution (show status)
# ============================================================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    training_status
fi
