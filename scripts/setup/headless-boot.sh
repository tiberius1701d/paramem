#!/bin/bash
# Reconcile OS-level headless-boot state with the `headless_boot` flag in
# server.yaml. See configs/server.yaml for the rationale.
#
# When enabled:
#   - systemd user linger for $USER (boot-time user manager)
#   - Windows scheduled task `ParaMem-Start-WSL-Boot` at system startup
#     (WSL hosts only; launches the WSL2 VM pre-login)
#
# When disabled: both are removed.
#
# Idempotent. Non-fatal on elevation failure — WARNs and continues so the
# server start path is not blocked.
#
# Runs on every start via start-server.sh, matching win-port-forward.sh.

set -uo pipefail
cd "$(dirname "$0")/../.."

CONFIG="${PARAMEM_CONFIG:-configs/server.yaml}"
USER_NAME="$(id -un)"
TASK_NAME="ParaMem-Start-WSL-Boot"

IS_WSL=false
grep -qi microsoft /proc/version 2>/dev/null && IS_WSL=true

# Desired state from config (empty/missing -> false)
PYTHON="${PARAMEM_PYTHON:-python}"
if ! DESIRED=$("$PYTHON" -c "
import sys, yaml
try:
    with open('$CONFIG') as f:
        cfg = yaml.safe_load(f) or {}
    print('true' if bool(cfg.get('headless_boot', False)) else 'false')
except Exception as e:
    sys.stderr.write(f'headless-boot: cannot read $CONFIG: {e}\n')
    sys.exit(1)
" 2>&1); then
    echo "headless-boot: skipping (config read failed: $DESIRED)"
    exit 0
fi

# Current Linux linger state
LINGER_NOW=false
[ -f "/var/lib/systemd/linger/$USER_NAME" ] && LINGER_NOW=true

# Current Windows task state (WSL only).
#
# The task runs as SYSTEM with HIGHEST privilege; its default ACL hides it
# from non-admin queries ("Get-ScheduledTask" returns not-found).  schtasks.exe
# distinguishes cleanly between the two absence-vs-hidden cases:
#   - "ERROR: The system cannot find the file specified."  -> task absent
#   - "ERROR: Access is denied."                           -> task present, ACL-hidden
# Use this distinction to avoid re-registering on every boot.
TASK_NOW=false
if $IS_WSL; then
    OUT=$(powershell.exe -NoProfile -Command "schtasks.exe /Query /TN '$TASK_NAME' /FO LIST" 2>&1 | tr -d '\r')
    if echo "$OUT" | grep -qi "Access is denied"; then
        TASK_NOW=true
    elif echo "$OUT" | grep -q "^TaskName:"; then
        TASK_NOW=true
    fi
fi

# Early exit if already in sync
LINGER_TARGET=$DESIRED
TASK_TARGET=$DESIRED
$IS_WSL || TASK_TARGET=$TASK_NOW  # non-WSL: ignore task half
if [ "$LINGER_NOW" = "$LINGER_TARGET" ] && [ "$TASK_NOW" = "$TASK_TARGET" ]; then
    echo "headless-boot: in sync (headless_boot=$DESIRED)"
    exit 0
fi

echo "headless-boot: reconciling (desired=$DESIRED, linger=$LINGER_NOW, task=$TASK_NOW, wsl=$IS_WSL)"

# --- Linux side: linger ---
#
# Elevation fallback order:
#   1. sudo -n  (cached credentials or NOPASSWD rule — silent)
#   2. interactive sudo prompt on the current TTY (manual script runs)
#   3. WSL-only: spawn a new Windows-console WSL shell so the user can
#      approve sudo there (covers the systemd-launched, no-TTY path, as
#      long as a user is logged in to Windows)
#   4. WARN with the exact command to run manually (truly headless boots)
_spawn_wsl_sudo_window() {
    # Opens a new WSL console window running `sudo loginctl <action>-linger`
    # and blocks until it closes.  Returns 0 if linger state ends up matching
    # the desired action, non-zero otherwise (spawn failed or user declined).
    local action=$1
    local distro="${WSL_DISTRO_NAME:-}"
    [ -n "$distro" ] || return 1
    local tmp
    tmp=$(mktemp --suffix=.sh) || return 1
    local what why
    if [ "$action" = "enable" ]; then
        what="enable systemd user linger"
        why="so paramem-server starts at boot without waiting for a login."
    else
        what="disable systemd user linger"
        why="so paramem-server no longer starts before you log in."
    fi
    cat > "$tmp" <<EOF
#!/bin/bash
cat <<'BANNER'
────────────────────────────────────────────────────────────────
 ParaMem — headless-boot reconciler
────────────────────────────────────────────────────────────────
BANNER
cat <<EXPLAIN

 Triggered by: configs/server.yaml -> headless_boot: $DESIRED
 Needs to:     $what
 Reason:       $why

 About to run: sudo loginctl ${action}-linger $USER_NAME

EXPLAIN
if sudo -p "[ParaMem] sudo password for %p: " loginctl ${action}-linger "$USER_NAME"; then
    echo
    echo "Done."
else
    echo
    echo "FAILED — see message above."
fi
echo
read -t 60 -rp "Auto-closes in 60s, or press Enter now. "
EOF
    chmod +x "$tmp"
    echo "  No TTY here — opening a new WSL console window for sudo approval..."
    # PowerShell array-form ArgumentList is clean here (no -Verb RunAs → no
    # ambiguous parameter set).  2>/dev/null hides "no interactive session"
    # errors on truly headless boots; we re-check linger state to decide
    # whether it actually worked.
    powershell.exe -NoProfile -Command \
        "Start-Process wsl.exe -ArgumentList '-d','$distro','--','bash','$tmp' -Wait" 2>/dev/null
    rm -f "$tmp"
    if [ "$action" = "enable" ]; then
        [ -f "/var/lib/systemd/linger/$USER_NAME" ]
    else
        [ ! -f "/var/lib/systemd/linger/$USER_NAME" ]
    fi
}

_run_sudo() {
    # Usage: _run_sudo enable|disable
    local action=$1
    local cmd="loginctl $action-linger $USER_NAME"
    if sudo -n true 2>/dev/null; then
        sudo $cmd && echo "  $cmd: done"
    elif [ -t 0 ]; then
        echo "  sudo required for: $cmd"
        sudo $cmd && echo "  $cmd: done"
    elif $IS_WSL && _spawn_wsl_sudo_window "$action"; then
        echo "  $cmd: done (via WSL console)"
    else
        echo "  WARN: no TTY and no cached sudo credentials." >&2
        echo "  Run manually: sudo $cmd" >&2
        return 1
    fi
}

if [ "$LINGER_TARGET" = "true" ] && [ "$LINGER_NOW" = "false" ]; then
    _run_sudo enable || true
elif [ "$LINGER_TARGET" = "false" ] && [ "$LINGER_NOW" = "true" ]; then
    _run_sudo disable || true
fi

# --- Windows side: scheduled task (WSL only) ---
#
# Quoting note: the elevated child receives its script via -EncodedCommand
# (base64 UTF-16LE). This sidesteps the single-quote aliasing that happens
# when a -Command string containing embedded single-quoted tokens is parsed
# twice (outer powershell.exe + inner Start-Process). With -EncodedCommand
# there is no outer quoting at all — just a base64 blob.
_run_elevated_ps() {
    local script=$1
    local encoded
    encoded=$(printf '%s' "$script" | iconv -t UTF-16LE | base64 -w0)
    powershell.exe -NoProfile -Command \
        "Start-Process powershell -ArgumentList '-NoProfile -EncodedCommand $encoded' -Verb RunAs -Wait" 2>/dev/null
}

if $IS_WSL; then
    if [ "$TASK_TARGET" = "true" ] && [ "$TASK_NOW" = "false" ]; then
        DISTRO="${WSL_DISTRO_NAME:-}"
        if [ -z "$DISTRO" ]; then
            echo "  WARN: WSL_DISTRO_NAME not set — cannot register task automatically." >&2
            echo "  Run from an interactive WSL shell: bash $0" >&2
        else
            echo "  Registering Windows scheduled task '$TASK_NAME' for distro '$DISTRO'..."
            PS="\$a=New-ScheduledTaskAction -Execute 'wsl.exe' -Argument '-d $DISTRO -u $USER_NAME --exec /bin/true';"
            PS+="\$t=New-ScheduledTaskTrigger -AtStartup;"
            PS+="\$p=New-ScheduledTaskPrincipal -UserId 'SYSTEM' -RunLevel Highest;"
            PS+="\$s=New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries;"
            PS+="Register-ScheduledTask -TaskName '$TASK_NAME' -Action \$a -Trigger \$t -Principal \$p -Settings \$s -Force | Out-Null;"
            PS+="Write-Host 'Task $TASK_NAME registered.'"
            _run_elevated_ps "$PS" \
                || echo "  WARN: failed to register task (UAC declined or no desktop?)" >&2
        fi
    elif [ "$TASK_TARGET" = "false" ] && [ "$TASK_NOW" = "true" ]; then
        echo "  Unregistering Windows scheduled task '$TASK_NAME'..."
        PS="Unregister-ScheduledTask -TaskName '$TASK_NAME' -Confirm:\$false;"
        PS+="Write-Host 'Task $TASK_NAME removed.'"
        _run_elevated_ps "$PS" \
            || echo "  WARN: failed to unregister task (UAC declined?)" >&2
    fi
fi

echo "headless-boot: done."
