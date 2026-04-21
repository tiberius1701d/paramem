#!/bin/bash
# Register the WU-Lock-Acquire / WU-Lock-Release Windows scheduled tasks.
#
# Tasks run as the current Windows user with RunLevel=Highest — needed because
# the PauseUpdates HKLM keys require elevation to write. Registration itself
# also needs elevation, so we launch an elevated PowerShell via
# Start-Process -Verb RunAs (UAC prompt).
#
# Idempotent: re-registration overwrites via -Force.
#
# Called by training-control.sh::_wu_ensure_registered on first tresume when
# the tasks are missing. Can also be invoked manually:
#     bash scripts/setup-wu-lock.sh

set -uo pipefail
cd "$(dirname "$0")/.."

PS1_UNIX="$PWD/scripts/wu-lock.ps1"
if [[ ! -f "$PS1_UNIX" ]]; then
    echo "ERROR: $PS1_UNIX missing" >&2
    exit 1
fi
PS1_WIN=$(wslpath -w "$PS1_UNIX")

if ! grep -qi microsoft /proc/version 2>/dev/null; then
    echo "setup-wu-lock: not a WSL host — skipping." >&2
    exit 0
fi

# Same encoded-command pattern as scripts/setup/headless-boot.sh — avoids the
# quote-aliasing that bites inner Start-Process calls.
_run_elevated_ps() {
    local script=$1
    local encoded
    encoded=$(printf '%s' "$script" | iconv -t UTF-16LE | base64 -w0)
    powershell.exe -NoProfile -Command \
        "Start-Process powershell -ArgumentList '-NoProfile -EncodedCommand $encoded' -Verb RunAs -Wait" 2>/dev/null
}

# `$s` Settings disable flaky "only if idle" / "stop on battery" defaults.
# Register-ScheduledTask needs at least one trigger, so we attach a far-future
# one-time trigger the task will never fire — /Run is the only invocation path.
read -r -d '' PS_BODY <<'PS' || true
$ErrorActionPreference = 'Stop'
$path = '__PS1_WIN__'
$user = "$env:USERDOMAIN\$env:USERNAME"
$never = (Get-Date '2099-01-01 00:00:00')
$set = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
$trig = New-ScheduledTaskTrigger -Once -At $never
$prin = New-ScheduledTaskPrincipal -UserId $user -RunLevel Highest -LogonType InteractiveToken

$actAcq = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ('-NoProfile -ExecutionPolicy Bypass -File "' + $path + '" -Acquire -Days 2')
$actRel = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ('-NoProfile -ExecutionPolicy Bypass -File "' + $path + '" -Release')

Register-ScheduledTask -TaskName 'WU-Lock-Acquire' -Action $actAcq -Trigger $trig -Principal $prin -Settings $set -Force | Out-Null
Register-ScheduledTask -TaskName 'WU-Lock-Release' -Action $actRel -Trigger $trig -Principal $prin -Settings $set -Force | Out-Null
Write-Host 'WU-Lock tasks registered.'
PS
PS_BODY=${PS_BODY//__PS1_WIN__/$PS1_WIN}

echo "  Registering WU-Lock scheduled tasks (UAC prompt will appear)..."
_run_elevated_ps "$PS_BODY"

# Verify — the elevated child's exit code doesn't propagate through RunAs.
if schtasks.exe /Query /TN 'WU-Lock-Acquire' >/dev/null 2>&1 \
   && schtasks.exe /Query /TN 'WU-Lock-Release' >/dev/null 2>&1; then
    echo "  WU-Lock tasks present."
    exit 0
else
    echo "  WARN: WU-Lock tasks still missing after registration attempt." >&2
    echo "  (UAC declined, or running user is not local admin?)" >&2
    exit 1
fi
