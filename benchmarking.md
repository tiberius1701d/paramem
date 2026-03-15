# ParaMem Benchmarking Suite

Comprehensive evaluation of parametric memory across realistic scenarios.
All tests run with both distillation models (Gemma 2 9B, Mistral 7B v0.3)
for direct comparison. Results are saved per-model.

## Distillation Models

| Model | Accuracy | Coverage | GPU Time | Config Key |
|---|---|---|---|---|
| Gemma 2 9B Instruct | 0.902 | 88% | 357s | `--distillation-model gemma` |
| Mistral 7B Instruct v0.3 | 0.894 | 84% | 426s | `--distillation-model mistral` |

Without `--distillation-model`, tests fall back to base model graph extraction
(lower quality, but no extra VRAM needed).

## Running the Suite

Each test runs independently. To run with both distillation models:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Example: Test 1 with both models
python experiments/test1_scale_expansion.py --distillation-model gemma
python experiments/test1_scale_expansion.py --distillation-model mistral

# Example: Test 2 with Gemma only
python experiments/test2_contradictions.py --distillation-model gemma
```

Results go to `outputs/testN_*/results.json`.

---

## Test 1: Scale Expansion

**Script:** `experiments/test1_scale_expansion.py`
**Status:** IMPLEMENTED
**GPU time:** ~100 min (full sweep)

Trains indexed keys at 10, 25, 50, 75, 100 keys. Measures recall degradation
as fact count grows. Compares against QA-RAG baseline at each scale point.

**Metrics:** Exact recall rate, SimHash confidence, training time, RAG similarity.

```bash
python experiments/test1_scale_expansion.py --distillation-model gemma
python experiments/test1_scale_expansion.py --scale 50 --distillation-model mistral
```

## Test 2: Contradiction Resolution

**Script:** `experiments/test2_contradictions.py`
**Status:** IMPLEMENTED
**GPU time:** ~50 min

10 fact chains with 3 temporal versions across 10 sessions. Tests whether
parametric memory naturally resolves contradictions (returns current fact)
vs RAG which stores all versions and may return stale facts.

**Metrics:** Current-version recall, similarity to current vs stale, per-chain breakdown.

```bash
python experiments/test2_contradictions.py --distillation-model gemma
```

## Test 3: Associative Inference

**Script:** `experiments/test3_inference.py`
**Status:** IMPLEMENTED
**GPU time:** ~15 min

Trains base facts, then queries with inference questions requiring combination
of 2-3 facts. Honest expectation: indexed keys require explicit lookup, so
inference may fail. A valid negative result.

**Metrics:** Base fact recall (sanity check), inference similarity (parametric vs RAG).

```bash
python experiments/test3_inference.py --distillation-model gemma
```

## Test 4: Multi-Session Reinforcement

**Script:** `experiments/test4_reinforcement.py`
**Status:** IMPLEMENTED
**GPU time:** ~50 min

30 facts at controlled frequencies (reinforced 3-4x, mentioned 2x, single mention)
across 10 sessions. Tests biological consolidation property: reinforced facts
should recall better.

**Metrics:** Per-group recall rate, mean confidence by reinforcement level.

**Known gap:** No RAG baseline (to be added).

```bash
python experiments/test4_reinforcement.py --distillation-model gemma
```

## Test 5: Privacy Preservation

**Script:** `experiments/test5_privacy.py`
**Status:** IMPLEMENTED
**GPU time:** ~5 min

Trains 50 facts, then probes with 10 adversarial extraction prompts. Measures
whether adapter weights leak readable data vs RAG text storage.

**Metrics:** Leaked fact count (parametric vs RAG), similarity scores per probe.

```bash
python experiments/test5_privacy.py --distillation-model gemma
```

## Test 6: Edge Deployment Footprint

**Script:** `experiments/test6_footprint.py`
**Status:** IMPLEMENTED
**GPU time:** ~10 min

Measures adapter file sizes, registry sizes, inference latency at 10/25/50 keys.
Compares total footprint and latency against RAG (embeddings + text store).

**Metrics:** Storage bytes, inference latency (ms), cold start time.

```bash
python experiments/test6_footprint.py --distillation-model gemma
```

## Test 7: Second Persona

**Script:** `experiments/test7_second_persona.py`
**Status:** IMPLEMENTED
**GPU time:** ~25 min

Trains two separate personas on the same base model. Tests generalization
beyond one user and measures cross-contamination between adapters.

**Metrics:** Per-persona recall, cross-contamination rate.

```bash
python experiments/test7_second_persona.py --distillation-model gemma
```

---

## Test 8: Distillation Quality

### 8a: Model Comparison

**Script:** `experiments/test_distillation_models.py`
**Status:** COMPLETE (30 configurations tested)

Benchmarks which models produce the best QA pairs from raw input.
Source of truth for model selection.

**Result:** Gemma 2 9B (best overall), Mistral 7B v0.3 (best pure-GPU).

### 8b: Distillation Quality to Recall Quality

**Script:** `experiments/test8b_distillation_recall.py`
**Status:** NOT IMPLEMENTED

Trains adapters on QA pairs distilled by each model (Gemma, Mistral, base model
graph extraction). Same training config. Compares indexed key recall.

**Purpose:** Proves that better distillation quality translates to better recall.

**Design:**
- Distill 50 QA pairs with each of 3 sources (Gemma, Mistral, graph extraction)
- Train indexed keys on each set (rank 8, 30 epochs)
- Evaluate recall and SimHash confidence
- Report comparison table

### 8c: Cloud Distillation (Claude API)

**Script:** `experiments/test8c_cloud_distillation.py`
**Status:** NOT IMPLEMENTED

Uses Claude API for distillation (quality ceiling), local model for storage.
Quantifies cloud vs local gap.

**Design:**
- Distill same 50 QA pairs via Claude API
- Train adapter, measure recall
- Compare against local distillation results from 8b

---

## Test 9: Base Model Inference Quality

### 9a: Storage Benchmark

**Script:** `experiments/test9a_storage_benchmark.py`
**Status:** NOT IMPLEMENTED

Trains same 20 QA pairs on different base models (Qwen 3B, Llama 3B, Mistral 7B).
Same rank, epochs, config. Measures which model memorizes indexed facts best.

**Metrics:** Exact recall per model, SimHash confidence, training time, VRAM.

### 9b: Inference Benchmark

**Script:** `experiments/test9b_inference_quality.py`
**Status:** NOT IMPLEMENTED

After training indexed facts, tests natural language generation quality with
conversational prompts (not key-based retrieval).

**Metrics:** Factual accuracy, hallucination rate, response latency.

### 9c: Combined Scorecard

**Script:** `experiments/test9c_scorecard.py`
**Status:** NOT IMPLEMENTED

Aggregates 8a + 9a + 9b into a single model selection table:
distillation quality, storage recall, inference quality, resource usage.

---

## Known Gaps

| Gap | Priority | Effort |
|---|---|---|
| Test 8b: distillation → recall correlation | High | Medium |
| Answer word-count validation in all tests | Medium | Low |
| Test 4: add RAG baseline | Medium | Low |
| Test 9a/9b/9c: base model inference | Medium | High |
| Test 8c: cloud distillation | Low | Medium |

## Execution Plan

### Phase A: Run Tests 1-7 with distillation models

Run each test twice — once with Gemma, once with Mistral. Compare results.

```bash
# Full suite with Gemma (~255 min total)
for test in test1_scale_expansion test2_contradictions test3_inference \
            test4_reinforcement test5_privacy test6_footprint test7_second_persona; do
    python experiments/${test}.py --distillation-model gemma
done

# Repeat with Mistral (~255 min total)
for test in test1_scale_expansion test2_contradictions test3_inference \
            test4_reinforcement test5_privacy test6_footprint test7_second_persona; do
    python experiments/${test}.py --distillation-model mistral
done
```

### Phase B: Fill gaps

1. Implement Test 8b (distillation → recall)
2. Add word-count tracking to test_harness.py
3. Add RAG baseline to Test 4
4. Implement Test 9 (if time permits)

### Phase C: Paper updates

Incorporate results into paper tables and analysis.

## Output Structure

```
outputs/
├── test1_scale/          # Scale expansion results
├── test2_contradictions/ # Contradiction resolution
├── test3_inference/      # Associative inference
├── test4_reinforcement/  # Multi-session reinforcement
├── test5_privacy/        # Privacy preservation
├── test6_footprint/      # Edge deployment footprint
├── test7_second_persona/ # Second persona
├── distillation_benchmark/ # Test 8a model comparison
├── prompt_engineering/   # Prompt strategy results
└── smoke_distillation/   # Pipeline smoke test
```
