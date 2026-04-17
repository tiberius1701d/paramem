#!/bin/bash
set -euo pipefail

PROJECT_DIR="$HOME/projects/paramem"
PYTHON="$HOME/miniforge3/envs/paramem/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

cd "$PROJECT_DIR"
export $(grep -v '^#' .env | xargs)

TS=$(date +%Y%m%d_%H%M%S)
echo "=== Probe sweep chain — $TS ==="

source ~/.local/bin/gpu-cooldown.sh

# --- 1. LongMemEval 100 stratified ---
echo "[$(date +%H:%M)] Starting LongMemEval 100 stratified..."
wait_for_cooldown 52
$PYTHON experiments/dataset_probe.py \
  --dataset longmemeval \
  --model mistral \
  --sample-strategy stratified \
  --sample-size 100 \
  --sample-seed 42 \
  --no-train \
  2>&1 | tee "$LOG_DIR/probe_longmemeval_100_$TS.log"
echo "[$(date +%H:%M)] LongMemEval done."

# --- 2. PerLTQA — Deng Yu (31 dialogues) ---
echo "[$(date +%H:%M)] Starting PerLTQA — Deng Yu..."
wait_for_cooldown 52
$PYTHON experiments/dataset_probe.py \
  --dataset perltqa \
  --model mistral \
  --character "Deng Yu" \
  --no-train \
  2>&1 | tee "$LOG_DIR/probe_perltqa_deng_yu_$TS.log"
echo "[$(date +%H:%M)] Deng Yu done."

# --- 3. PerLTQA — Liang Xin (30 dialogues) ---
echo "[$(date +%H:%M)] Starting PerLTQA — Liang Xin..."
wait_for_cooldown 52
$PYTHON experiments/dataset_probe.py \
  --dataset perltqa \
  --model mistral \
  --character "Liang Xin" \
  --no-train \
  2>&1 | tee "$LOG_DIR/probe_perltqa_liang_xin_$TS.log"
echo "[$(date +%H:%M)] Liang Xin done."

# --- 4. PerLTQA — Xia Yu (28 dialogues) ---
echo "[$(date +%H:%M)] Starting PerLTQA — Xia Yu..."
wait_for_cooldown 52
$PYTHON experiments/dataset_probe.py \
  --dataset perltqa \
  --model mistral \
  --character "Xia Yu" \
  --no-train \
  2>&1 | tee "$LOG_DIR/probe_perltqa_xia_yu_$TS.log"
echo "[$(date +%H:%M)] Xia Yu done."

echo "=== All sweeps complete at $(date +%H:%M) ==="
