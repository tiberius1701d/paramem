# Pause / resume / check Windows Update via the "Pause Updates" registry keys.
#
# Called by the scheduled tasks registered by scripts/setup-wu-lock.sh.
# training-control.sh invokes it indirectly (Acquire/Release go through
# schtasks.exe /Run because HKLM writes require admin; Check is a read and runs
# from any user context).
#
# Usage:
#   wu-lock.ps1 -Check                -> NONE | PENDING_REBOOT
#   wu-lock.ps1 -Acquire [-Days N]    -> pause for N days (default 2). HKLM write -> admin.
#   wu-lock.ps1 -Release              -> clear the pause. HKLM write -> admin.
[CmdletBinding()]
param(
    [switch]$Acquire,
    [switch]$Release,
    [switch]$Check,
    [int]$Days = 2
)

$ErrorActionPreference = 'Stop'
$RegPath = 'HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings'
$PairNames = @(
    @('PauseUpdatesStartTime',        'PauseUpdatesExpiryTime'),
    @('PauseFeatureUpdatesStartTime', 'PauseFeatureUpdatesEndTime'),
    @('PauseQualityUpdatesStartTime', 'PauseQualityUpdatesEndTime')
)
# UX/Settings pause keys above only block new update scans/downloads; they do
# NOT stop Update Orchestrator (UsoSvc / MoUsoCoreWorker.exe) from rebooting
# for already-installed updates during the Active Hours off-window.  The AU
# policy NoAutoRebootWithLoggedOnUsers=1 is the kill-switch: USO honours it
# regardless of Active Hours as long as a user is signed in.
$AuPolicyPath = 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU'
$AuPolicyName = 'NoAutoRebootWithLoggedOnUsers'

function Set-PauseWindow {
    param([int]$Days)
    if (-not (Test-Path $RegPath)) { New-Item -Path $RegPath -Force | Out-Null }
    $start = (Get-Date).ToUniversalTime()
    $end   = $start.AddDays($Days)
    $fmt   = 'yyyy-MM-ddTHH:mm:ssZ'
    foreach ($pair in $PairNames) {
        Set-ItemProperty -Path $RegPath -Name $pair[0] -Value $start.ToString($fmt) -Type String
        Set-ItemProperty -Path $RegPath -Name $pair[1] -Value $end.ToString($fmt)   -Type String
    }
    if (-not (Test-Path $AuPolicyPath)) { New-Item -Path $AuPolicyPath -Force | Out-Null }
    Set-ItemProperty -Path $AuPolicyPath -Name $AuPolicyName -Value 1 -Type DWord
    Write-Output ("WU paused until {0}; auto-reboot blocked while signed in" -f $end.ToLocalTime().ToString('yyyy-MM-dd HH:mm'))
}

function Clear-PauseWindow {
    if (Test-Path $RegPath) {
        foreach ($pair in $PairNames) {
            foreach ($name in $pair) {
                Remove-ItemProperty -Path $RegPath -Name $name -ErrorAction SilentlyContinue
            }
        }
    }
    if (Test-Path $AuPolicyPath) {
        Remove-ItemProperty -Path $AuPolicyPath -Name $AuPolicyName -ErrorAction SilentlyContinue
    }
    Write-Output 'WU pause cleared'
}

function Get-PendingState {
    $rebootKeys = @(
        'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired',
        'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending'
    )
    foreach ($k in $rebootKeys) {
        if (Test-Path $k) { return 'PENDING_REBOOT' }
    }
    return 'NONE'
}

if ($Check)   { Write-Output (Get-PendingState); exit 0 }
if ($Acquire) { Set-PauseWindow -Days $Days;     exit 0 }
if ($Release) { Clear-PauseWindow;               exit 0 }

Write-Error 'Must specify -Acquire, -Release, or -Check'
exit 1
