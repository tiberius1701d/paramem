# Run as Administrator on Windows startup
# Forwards LAN ports to WSL: 8420 (ParaMem server) and 2222 (SSH access)
$ports = @(8420, 2222)

# Get current WSL IP
$wslIp = (wsl hostname -I).Trim().Split(' ')[0]
if (-not $wslIp) {
    Write-Error "Could not determine WSL IP"
    exit 1
}

# Forward each port from LAN to WSL
foreach ($port in $ports) {
    # Remove existing proxy if any
    netsh interface portproxy delete v4tov4 listenport=$port listenaddress=0.0.0.0 2>$null
    # Add new proxy with current WSL IP
    netsh interface portproxy add v4tov4 listenport=$port listenaddress=0.0.0.0 connectport=$port connectaddress=$wslIp
    Write-Host "Port $port forwarded to WSL at $wslIp"
}

# Firewall rule name per port
$rules = @{ "ParaMem Server" = 8420; "WSL SSH" = 2222 }

# Ensure a firewall rule exists for each port
foreach ($name in $rules.Keys) {
    if (-not (Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -DisplayName $name -Direction Inbound -LocalPort $rules[$name] -Protocol TCP -Action Allow | Out-Null
        Write-Host "Firewall rule created: $name (port $($rules[$name]))"
    }
}

