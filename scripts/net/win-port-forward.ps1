# Run as Administrator on Windows startup
# Forwards port 8420 from LAN to WSL for ParaMem server access

$port = 8420

# Get current WSL IP
$wslIp = (wsl hostname -I).Trim().Split(' ')[0]

if (-not $wslIp) {
    Write-Error "Could not determine WSL IP"
    exit 1
}

# Remove existing proxy if any
netsh interface portproxy delete v4tov4 listenport=$port listenaddress=0.0.0.0 2>$null

# Add new proxy with current WSL IP
netsh interface portproxy add v4tov4 listenport=$port listenaddress=0.0.0.0 connectport=$port connectaddress=$wslIp

# Ensure firewall rule exists
if (-not (Get-NetFirewallRule -DisplayName "ParaMem Server" -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName "ParaMem Server" -Direction Inbound -LocalPort $port -Protocol TCP -Action Allow | Out-Null
    Write-Host "Firewall rule created for port $port"
}

Write-Host "Port $port forwarded to WSL at $wslIp"
