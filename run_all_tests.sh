#!/bin/bash
# Run all benchmark tests (1-7) for both models sequentially.
# Designed for unattended overnight execution.
#
# Usage: bash run_all_tests.sh 2>&1 | tee run_all_tests.log
# Or:    nohup bash run_all_tests.sh > run_all_tests.log 2>&1 &

set -o pipefail
cd "$(dirname "$0")"

source .env 2>/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DEACTIVATE_ASYNC_LOAD=1

PYTHON=${PARAMEM_PYTHON:-python}
LOG_DIR="outputs/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

SUMMARY="$LOG_DIR/summary.txt"
echo "Test run started: $(date)" > "$SUMMARY"
echo "========================================" >> "$SUMMARY"

# Disk space check (CLAUDE.md: safety threshold 800 GB used)
USED_GB=$(df / --output=used -BG | tail -1 | tr -d ' G')
if [ "$USED_GB" -gt 800 ]; then
    echo "ABORT: Disk usage ${USED_GB}GB exceeds 800GB safety threshold." | tee -a "$SUMMARY"
    exit 1
fi
echo "Disk check passed: ${USED_GB}GB used" | tee -a "$SUMMARY"

# GPU cooldown — source the system cooldown tool
source ~/.local/bin/gpu-cooldown.sh

run_test() {
    local name="$1"
    shift
    local log_file="$LOG_DIR/${name}.log"

    echo ""
    echo "========================================"
    echo "  Starting: $name"
    echo "  Time: $(date)"
    echo "  GPU temp: $(gpu_temp)°C"
    echo "========================================"

    local start_time=$(date +%s)
    $PYTHON "$@" > "$log_file" 2>&1
    local exit_code=$?
    local end_time=$(date +%s)
    local elapsed=$(( (end_time - start_time) / 60 ))

    if [ $exit_code -eq 0 ]; then
        echo "  PASSED: $name (${elapsed} min)" | tee -a "$SUMMARY"
    else
        echo "  FAILED: $name (exit code $exit_code, ${elapsed} min)" | tee -a "$SUMMARY"
    fi

    # Dynamic GPU cooldown — returns instantly if already cool
    wait_for_cooldown

    # Per-test disk check
    USED_GB=$(df / --output=used -BG | tail -1 | tr -d ' G')
    if [ "$USED_GB" -gt 800 ]; then
        echo "  ABORT: Disk usage ${USED_GB}GB exceeds threshold." | tee -a "$SUMMARY"
        echo "Test run aborted: $(date)" >> "$SUMMARY"
        cat "$SUMMARY"
        exit 1
    fi

    return 0  # Always continue to next test
}

# COMPLETED (batch 1+2, 2026-03-19/20):
# test7_gemma:    PASSED — 50/50 both personas, zero contamination
# test7_mistral:  PASSED — 50/50 both personas, zero contamination
# test7b_mistral: PASSED — merge 0/50 (negative result, expected)
# test1_gemma:    PASSED — 100/100 all scales, conf=1.000, sim=1.000
# test1_mistral:  PASSED — 54/54 (yield-capped), conf=1.000, sim=1.000
# test2_gemma:    PASSED — 20/20 contradictions, current=3/10 (QA gen quality)
# test2_mistral:  PASSED — 20/20 contradictions, current=8/10
# test2b_gemma:   PASSED — 16/16, zero forgetting
# test2b_mistral: PASSED — 16/16, zero forgetting
# test3_gemma:    PASSED — PM 0.687 ≈ RAG 0.679 (parity)
# test3_mistral:  PASSED — PM 0.566 ≈ RAG 0.525 (parity)
# test4_gemma:    PASSED — 30/30, all tiers 10/10
# test4_mistral:  PASSED — 30/30, all tiers 10/10

# Test 2b: COMPLETED (see above)
# run_test "test2b_gemma" experiments/test2b_incremental_contradictions.py --model gemma
# run_test "test2b_mistral" experiments/test2b_incremental_contradictions.py --model mistral

# Test 3: COMPLETED (see above)
# run_test "test3_gemma" experiments/test3_inference.py --model gemma
# run_test "test3_mistral" experiments/test3_inference.py --model mistral

# Test 4: COMPLETED (see above)
# run_test "test4_gemma" experiments/test4_reinforcement.py --model gemma
# run_test "test4_mistral" experiments/test4_reinforcement.py --model mistral

# Test 5: Natural Recall
run_test "test5_gemma" experiments/test5_natural_recall.py --model gemma
run_test "test5_mistral" experiments/test5_natural_recall.py --model mistral

# Test 6: Storage Footprint
run_test "test6_gemma" experiments/test6_footprint.py --model gemma
run_test "test6_mistral" experiments/test6_footprint.py --model mistral

# Test 4b: Incremental Learning Without Full Replay
run_test "test4b_gemma" experiments/test4b_incremental_no_replay.py --model gemma
run_test "test4b_mistral" experiments/test4b_incremental_no_replay.py --model mistral

echo "" >> "$SUMMARY"
echo "========================================" >> "$SUMMARY"
echo "Test run completed: $(date)" >> "$SUMMARY"

echo ""
echo "========================================"
echo "  ALL TESTS COMPLETE"
echo "  Summary: $SUMMARY"
echo "========================================"
cat "$SUMMARY"
