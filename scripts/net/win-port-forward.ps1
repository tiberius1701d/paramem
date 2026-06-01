# Apply netsh portproxy and Windows Firewall rules for ParaMem ports.
#
# Runs elevated (as a scheduled-task action or via Start-Process -Verb RunAs).
# This is the SINGLE applier of portproxy/firewall rules — the bash wrapper
# (win-port-forward.sh) only decides HOW to invoke this worker.
#
# Parameters:
#   -Distro    WSL distro name (e.g. "Ubuntu-24.04")
#   -ListenIp  Windows-side listener address (default: "0.0.0.0")
#   -NasIp     Firewall source-scope IP (default: "" = unscoped)
#   -Ports     Array of ports to forward (default: 8420, 10300, 10301, 2222)
#
# Idempotent: rebuilds a portproxy rule only when its listen/connect address
# differs from the current values.  No-op when rules already match.

param(
    [string]   $Distro   = 'Ubuntu-24.04',
    [string]   $ListenIp = '0.0.0.0',
    [string]   $NasIp    = '',
    [int[]]    $Ports    = @(8420, 10300, 10301, 2222)
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# 1. Ensure WSL distro is up, then resolve WSL IP from the Windows side.
#    AtStartup may fire before LxssManager is fully ready, so we retry.
# ---------------------------------------------------------------------------
Write-Host "Ensuring WSL distro '$Distro' is up..."
try {
    & wsl.exe -d $Distro -e true 2>$null
} catch {
    # non-fatal — hostname -I below will retry
}

$WslIp = ''
for ($i = 1; $i -le 5; $i++) {
    $raw = & wsl.exe -d $Distro hostname -I 2>$null
    if ($raw) {
        $token = ($raw.Trim() -split '\s+')[0]
        if ($token -match '^\d+\.\d+\.\d+\.\d+$') {
            $WslIp = $token
            break
        }
    }
    if ($i -lt 5) {
        Write-Host "  Attempt $i/5: WSL IP not ready, retrying in 3s..."
        Start-Sleep -Seconds 3
    }
}

if (-not $WslIp) {
    Write-Error "ERROR: Could not determine WSL IP for distro '$Distro' after 5 attempts."
    exit 1
}

Write-Host "WSL IP:         $WslIp"
Write-Host "Listen IP:      $ListenIp"
if ($NasIp) {
    Write-Host "Firewall scope: inbound from $NasIp only"
} else {
    Write-Host "Firewall scope: any LAN source (NasIp not set)"
}

# ---------------------------------------------------------------------------
# 2. Idempotent portproxy + firewall rules — port the drift logic from
#    win-port-forward.sh lines 60-95 exactly.
# ---------------------------------------------------------------------------
$existing = & netsh interface portproxy show v4tov4 2>$null

foreach ($Port in $Ports) {
    # --- portproxy ---
    $existingRow = $existing | Where-Object { $_ -match "\b$Port\b" }
    if ($existingRow) {
        # Rule present — check listen/connect addresses match current values.
        $hasWslIp    = ($existingRow | Select-String -SimpleMatch $WslIp)    -ne $null
        $hasListenIp = ($existingRow | Select-String -SimpleMatch $ListenIp) -ne $null
        if ($hasWslIp -and $hasListenIp) {
            Write-Host "Port ${Port}: already forwarded ($ListenIp -> $WslIp)"
        } else {
            Write-Host "Port ${Port}: config drift detected, updating..."
            # Delete on both possible listen addresses to avoid duplicates.
            & netsh interface portproxy delete v4tov4 listenport=$Port listenaddress=0.0.0.0        2>$null
            & netsh interface portproxy delete v4tov4 listenport=$Port listenaddress=$ListenIp     2>$null
            & netsh interface portproxy add    v4tov4 listenport=$Port listenaddress=$ListenIp `
                connectport=$Port connectaddress=$WslIp
            Write-Host "Port ${Port}: updated ($ListenIp -> $WslIp)"
        }
    } else {
        Write-Host "Port ${Port}: adding forward ($ListenIp -> $WslIp)..."
        & netsh interface portproxy add v4tov4 listenport=$Port listenaddress=$ListenIp `
            connectport=$Port connectaddress=$WslIp
        Write-Host "Port ${Port}: added"
    }

    # --- firewall rule ---
    # New-NetFirewallRule MUST carry -ErrorAction Stop: rule creation needs
    # elevation, and a non-elevated access-denied is a *non-terminating* error
    # by default. Without Stop, powershell.exe exits 0 even on failure, making
    # a creation failure invisible.
    if ($NasIp) {
        # Scoped rule: remove any pre-existing unscoped rule first, then create scoped.
        Get-NetFirewallRule -DisplayName "ParaMem Port $Port" -ErrorAction SilentlyContinue |
            Remove-NetFirewallRule -ErrorAction SilentlyContinue
        New-NetFirewallRule -DisplayName "ParaMem Port $Port" -Direction Inbound `
            -LocalPort $Port -Protocol TCP -RemoteAddress $NasIp -Action Allow `
            -ErrorAction Stop | Out-Null
        Write-Host "Firewall rule (scoped to $NasIp) created for port $Port"
    } else {
        if (-not (Get-NetFirewallRule -DisplayName "ParaMem Port $Port" -ErrorAction SilentlyContinue)) {
            New-NetFirewallRule -DisplayName "ParaMem Port $Port" -Direction Inbound `
                -LocalPort $Port -Protocol TCP -Action Allow -ErrorAction Stop | Out-Null
            Write-Host "Firewall rule (unscoped) created for port $Port"
        } else {
            Write-Host "Firewall rule exists for port $Port"
        }
    }
}

Write-Host "Done. Current rules:"
& netsh interface portproxy show v4tov4
