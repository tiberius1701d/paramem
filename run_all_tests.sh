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

GPU_COOLDOWN=1200  # 20 minutes between tests — thermal accumulation caused BSOD at 3 min cooldown

run_test() {
    local name="$1"
    shift
    local log_file="$LOG_DIR/${name}.log"

    echo ""
    echo "========================================"
    echo "  Starting: $name"
    echo "  Time: $(date)"
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

    # GPU cooldown — let WSL2 CUDA driver stabilize after model unload
    echo "  GPU cooldown: ${GPU_COOLDOWN}s..."
    sleep $GPU_COOLDOWN

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

# Test 7: Second Persona (Mistral first — Gemma already completed 2026-03-19)
run_test "test7_mistral" experiments/test7_second_persona.py --model mistral

# Test 7b: Merged Persona Adapters (Mistral — uses Test 7 Mistral adapters)
run_test "test7b_mistral" experiments/test7b_merged_personas.py --model mistral

# Test 1: Scale Expansion
run_test "test1_gemma" experiments/test1_scale_expansion.py --model gemma
run_test "test1_mistral" experiments/test1_scale_expansion.py --model mistral

# Test 2: Contradiction Resolution
run_test "test2_gemma" experiments/test2_contradictions.py --model gemma
run_test "test2_mistral" experiments/test2_contradictions.py --model mistral

# Test 2b: Incremental Contradictions
run_test "test2b_gemma" experiments/test2b_incremental_contradictions.py --model gemma
run_test "test2b_mistral" experiments/test2b_incremental_contradictions.py --model mistral

# Test 3: Reasoning Quality Parity
run_test "test3_gemma" experiments/test3_inference.py --model gemma
run_test "test3_mistral" experiments/test3_inference.py --model mistral

# Test 4: Pipeline Robustness
run_test "test4_gemma" experiments/test4_reinforcement.py --model gemma
run_test "test4_mistral" experiments/test4_reinforcement.py --model mistral

# Test 5: Natural Recall
run_test "test5_gemma" experiments/test5_natural_recall.py --model gemma
run_test "test5_mistral" experiments/test5_natural_recall.py --model mistral

# Test 6: Storage Footprint
run_test "test6_gemma" experiments/test6_footprint.py --model gemma
run_test "test6_mistral" experiments/test6_footprint.py --model mistral

echo "" >> "$SUMMARY"
echo "========================================" >> "$SUMMARY"
echo "Test run completed: $(date)" >> "$SUMMARY"

echo ""
echo "========================================"
echo "  ALL TESTS COMPLETE"
echo "  Summary: $SUMMARY"
echo "========================================"
cat "$SUMMARY"
