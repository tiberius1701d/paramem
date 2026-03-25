#!/bin/bash
# Start the ParaMem server.
#
# Usage:
#   bash scripts/start-server.sh                    # foreground
#   bash scripts/start-server.sh --background       # background with log
#
# Environment:
#   PARAMEM_PYTHON  — Python executable (default: python from PATH)
#   PARAMEM_CONFIG  — Server config path (default: configs/server.yaml)
#
# For persistent deployment, install the systemd service instead:
#   sudo cp scripts/paramem-server.service /etc/systemd/system/
#   sudo systemctl daemon-reload
#   sudo systemctl enable --now paramem-server

set -euo pipefail
cd "$(dirname "$0")/.."

source .env 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_DEACTIVATE_ASYNC_LOAD=${HF_DEACTIVATE_ASYNC_LOAD:-1}

PYTHON=${PARAMEM_PYTHON:-python}
CONFIG=${PARAMEM_CONFIG:-configs/server.yaml}
LOG_DIR=logs

# Update Windows port proxy if running in WSL2
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "Updating Windows port forward..."
    bash scripts/win-port-forward.sh || echo "  Warning: port forward failed (non-fatal)"
fi

if [ "${1:-}" = "--background" ]; then
    mkdir -p "$LOG_DIR"
    LOG="$LOG_DIR/paramem-server-$(date +%Y%m%d_%H%M%S).log"
    echo "Starting ParaMem server in background..."
    echo "  Config: $CONFIG"
    echo "  Log:    $LOG"
    nohup $PYTHON -m paramem.server.app --config "$CONFIG" > "$LOG" 2>&1 &
    PID=$!
    echo "  PID:    $PID"
    echo "$PID" > "$LOG_DIR/paramem-server.pid"
else
    echo "Starting ParaMem server..."
    echo "  Config: $CONFIG"
    exec $PYTHON -m paramem.server.app --config "$CONFIG"
fi
