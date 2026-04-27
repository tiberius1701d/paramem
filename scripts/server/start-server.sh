#!/bin/bash
# Start the ParaMem server.
#
# Usage:
#   bash scripts/server/start-server.sh                    # foreground
#   bash scripts/server/start-server.sh --background       # background with log
#
# Environment:
#   PARAMEM_PYTHON  — Python executable (default: python from PATH)
#   PARAMEM_CONFIG  — Server config path (default: configs/server.yaml)
#
# For persistent deployment, install the systemd service instead:
#   sudo cp scripts/server/paramem-server.service /etc/systemd/system/
#   sudo systemctl daemon-reload
#   sudo systemctl enable --now paramem-server

set -euo pipefail
cd "$(dirname "$0")/../.."

source .env 2>/dev/null || true
# Machine-level GPU env (PYTORCH_CUDA_ALLOC_CONF, HF_DEACTIVATE_ASYNC_LOAD,
# …) lives in ~/.config/gpu-guard/config.toml [env].  Soft fallback when
# gpu-guard is not installed; the systemd path is the production deployment
# and uses the [env] file directly via ExecStartPre + EnvironmentFile.
if command -v gpu-guard >/dev/null 2>&1; then
    eval "$(gpu-guard env --export)"
fi

PYTHON=${PARAMEM_PYTHON:-python}
CONFIG=${PARAMEM_CONFIG:-configs/server.yaml}
LOG_DIR=logs

# Update Windows port proxy if running in WSL2
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "Updating Windows port forward..."
    bash scripts/net/win-port-forward.sh || echo "  Warning: port forward failed (non-fatal)"
fi

# Reconcile headless-boot state (linger + Windows scheduled task) with
# configs/server.yaml. Non-fatal: WARNs and continues if elevation is absent.
bash scripts/setup/headless-boot.sh || echo "  Warning: headless-boot reconcile failed (non-fatal)"

EXTRA_ARGS="${PARAMEM_EXTRA_ARGS:-}"

if [ "${1:-}" = "--background" ]; then
    mkdir -p "$LOG_DIR"
    LOG="$LOG_DIR/paramem-server-$(date +%Y%m%d_%H%M%S).log"
    echo "Starting ParaMem server in background..."
    echo "  Config: $CONFIG"
    echo "  Log:    $LOG"
    nohup $PYTHON -m paramem.server.app --config "$CONFIG" $EXTRA_ARGS > "$LOG" 2>&1 &
    PID=$!
    echo "  PID:    $PID"
    echo "$PID" > "$LOG_DIR/paramem-server.pid"
else
    echo "Starting ParaMem server..."
    echo "  Config: $CONFIG"
    exec $PYTHON -m paramem.server.app --config "$CONFIG" $EXTRA_ARGS
fi
