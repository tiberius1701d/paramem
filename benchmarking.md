# ParaMem Benchmarking Suite

Comprehensive evaluation of parametric memory across realistic scenarios.
Each model owns the full pipeline end-to-end: graph extraction → QA generation →
indexed key training → recall evaluation.

## Benchmark Models

| Model | Params | Quantization | CPU Offload | Config Key |
|---|---|---|---|---|
| Gemma 2 9B Instruct | 9B | NF4 4-bit | Yes (7GiB GPU + 20GiB CPU) | `--model gemma` |
| Mistral 7B Instruct v0.3 | 7B | NF4 4-bit | No | `--model mistral` |

Without `--model`, tests run both models sequentially for direct comparison.

## Running the Suite

Each test runs independently. One model at a time (8GB VRAM constraint).
Results are timestamped — no run can overwrite another.

### Environment setup

All tests load `.env` via `test_harness.py` → `load_dotenv()`. Create a `.env` file
in the project root with platform-specific settings:

```bash
# .env — loaded automatically by test_harness.py
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WANDB_API_KEY=<your key>

# WSL2 only: disable threaded weight loading to avoid CUDA driver races.
# Transformers 5.3+ uses a ThreadPoolExecutor for weight loading which races
# the WSL2 dxg paravirt memory mapper on models >= 4B params.
# Not needed on native Linux. See README.md for details.
HF_DEACTIVATE_ASYNC_LOAD=1
```

### Running tests

```bash
python experiments/test1_scale_expansion.py --model gemma
python experiments/test2_contradictions.py --model mistral
```

Results go to `outputs/testN_*/{model}/{timestamp}/results.json`.

## Test Suite Overview

| Test | Name | Key Question | Expected Outcome | Data |
|------|------|-------------|------------------|------|
| 1 | Scale Expansion | Does recall degrade at 10→100 keys? | ≥95% recall, linear time scaling | PerLTQA dialogues (on-the-fly distillation) |
| 2 | Contradiction Resolution | Can temporal fact updates be detected and resolved? | Graph normalization detects all same-predicate contradictions | Synthetic fact chains (3 temporal versions each) |
| 2b | Incremental Contradictions | Does forgetting work with a persistent adapter? | Immediate overwrite, zero catastrophic forgetting | Same data as Test 2, single persistent adapter |
| 3 | Reasoning Quality Parity | Does PM match RAG for multi-hop reasoning? | Quality parity; PM offers latency/privacy/compression benefits | ~50 PerLTQA facts + LLM-generated inference questions |
| 4 | Pipeline Robustness | Does cumulative train-delete-retrain stay reliable? | Stable recall as facts grow from 3 to 30 across 10 sessions | 30 synthetic facts across 10 sessions |
| 5 | Natural Recall | How much knowledge is accessible without indexed keys? | Limited — motivates the key mechanism | 50 QA pairs + 10 open-ended probes |
| 6 | Storage Footprint | What is the storage cost vs. RAG at scale? | Adapter is O(1); RAG index scales linearly | Adapters at 10/25/50/100 keys |
| 7 | Second Persona | Does the architecture generalize beyond one user? | Similar recall across personas with minimal cross-contamination | 2 personas × 50 QA pairs |

### Design principles

- Each test is standalone: `python experiments/testN_*.py [--model gemma|mistral]`
- Shared infrastructure in `experiments/utils/test_harness.py`
- All tests use the same indexed key pipeline — no test-specific training code
- RAG baselines included where comparison is meaningful (Tests 3, 4, 6)
- Honest expectations stated upfront — negative results are valuable

---

## Qwen 2.5 3B Development Experiments (§5.1-5.5)

**Model:** Qwen/Qwen2.5-3B (base model, NF4 4-bit, no CPU offload)
**Re-verified:** 2026-03-21, all scripts updated with timestamped output + raw capture, temperature=0.0

These use pre-defined synthetic QA pairs (no distillation pipeline). Qwen is a base model, not instruct — unsuitable for the LLM-based graph extraction required by Tests 1-7b.

| Experiment | Script | Result | Loss | Time | Output |
|---|---|---|---|---|---|
| §5.1 Per-fact recall (10 keys) | `phase4_indexed_keys_smoke.py` | 10/10 (100%) | 1.268 | 225s | `outputs/phase4_indexed_keys/qwen2.5-3b/20260321_211658/` |
| §5.2 Capacity (20 keys) | `f4_9c_test1_capacity.py` | 20/20 (100%) | 1.051 | 438s | `outputs/f4_9c_test1_capacity/qwen2.5-3b/20260321_214101/` |
| §5.3 Incremental (10+5) | `f4_9c_test2_incremental.py` | 15/15 (100%) | 1.271→0.817 | 227s+164s | `outputs/f4_9c_test2_incremental/qwen2.5-3b/20260321_221343/` |
| §5.4 Two-adapter (30ep retrain) | `f4_9c_test3_two_adapter.py` | 10/10 ep + 5/5 sem | 1.268/1.319/0.809 | 617s | `outputs/f4_9c_test3_two_adapter/qwen2.5-3b/20260321_225957/` |
| §5.4 Two-adapter (15ep retrain) | `f4_9c_test3_two_adapter.py --retrain-epochs 15` | 7/10 ep + 5/5 sem | — | 482s | `outputs/f4_9c_test3_two_adapter/qwen2.5-3b/20260321_223824/` |
| §5.5 Consolidation (10 cycles) | `f4_10_indexed_consolidation.py` | 8/8 ep + 2/2 sem (100%) | — | 42.7 min | `outputs/f4_10_indexed_consolidation/qwen2.5-3b/20260321_233723/` |

Key findings:
- All experiments achieve 100% recall at their respective scales
- The 15-epoch retrain failure (7/10) confirms the epoch budget requirement: retrain epochs must match initial training budget
- Individual QA recall (0.452 mean embedding similarity) is substantially lower than indexed key recall, validating the key mechanism
- Losses are nearly identical across reruns, confirming reproducibility

---

## Test 1: Scale Expansion

**Script:** `experiments/test1_scale_expansion.py`
**Status:** COMPLETE — both models (rerun 2026-03-19 with methodology fixes)

### What it tests

Can parametric memory maintain recall quality as the number of stored facts grows
from 10 to 100? Facts are extracted on the fly from real dialogue transcripts
(PerLTQA dataset), one session at a time, mimicking idle-time background learning.

### Data pipeline

1. Load dialogue sessions for character "Liang Xin" (30 available)
2. Process sessions one at a time: transcript → graph extraction → QA generation
3. Accumulate QA pairs until target count reached
4. Train indexed key adapter → evaluate recall

### Results (2026-03-20, final run)

| Scale | Gemma 2 9B Recall | Gemma Conf | Gemma Time | Mistral 7B Recall | Mistral Conf | Mistral Time |
|-------|-------------------|------------|------------|-------------------|--------------|--------------|
| 10    | 10/10 (100%)      | 1.000      | 7 min      | 10/10 (100%)      | 1.000        | 5 min        |
| 25    | 25/25 (100%)      | 1.000      | 18 min     | 25/25 (100%)      | 1.000        | 12 min       |
| 50    | 50/50 (100%)      | 1.000      | 35 min     | 50/50 (100%)      | 1.000        | 24 min       |
| 75    | 75/75 (100%)      | 1.000      | 53 min     | 54/54 (100%)*     | 1.000        | 27 min       |
| 100   | 100/100 (100%)    | 1.000      | 70 min     | 54/54 (100%)*     | 1.000        | 27 min       |

*Mistral extracted 54 QA pairs (improved from 41 with QA generator fixes).
Scales 75/100 capped at 54 keys.

Both models achieve **100% recall with 1.000 confidence and 1.000 embedding
similarity at every scale point.** Perfect scores across all three metrics.

**Config:** rank=8, alpha=16, 30 epochs, lr=1e-4, batch=1, grad_accum=2,
temperature=0.0, repetition_penalty=1.1, keyed_pairs persisted.

### Data statistics

| Metric | Gemma 2 9B | Mistral 7B |
|--------|------------|------------|
| QA pairs extracted | 104 | 54 |
| Sessions used | 23/30 | 30/30 |
| QA pairs per session | ~4.5 | ~1.8 |

Mistral extracts selectively at temperature=0.0 — lower yield but higher
precision (80% map to ground truth vs Gemma's 55%). Mistral's 54-key cap
means scale points 75/100 are capped at 54 keys.

### Training time scaling

| Metric | Gemma 2 9B | Mistral 7B |
|--------|------------|------------|
| Time per key (scale 10) | 44.4 s | 31.2 s |
| Time per key (scale 25) | 43.0 s | 30.1 s |
| Time per key (scale 50) | 45.9 s | 30.0 s* |
| Time per key (scale 100) | 42.4 s | 30.9 s* |
| Scaling behavior | Linear | Linear |

*Mistral 50/100 computed at 54 keys (capped).

Training time scales linearly with key count — no superlinear growth.
Mistral is ~30% faster, roughly proportional to parameter count (7B vs 9B).

### Loss convergence

| Scale | Gemma 2 9B | Mistral 7B |
|-------|------------|------------|
| 10    | 0.309      | 0.253      |
| 25    | 0.228      | 0.211      |
| 50    | 0.206      | 0.202*     |
| 75    | 0.192      | 0.198*     |
| 100   | 0.193      | 0.198*     |

*Mistral 50/75/100 trained on 54 keys (capped).

Loss decreases with scale on both models.

### Epoch convergence analysis

Both models reach loss < 0.01 by epoch 4-6 across all scales:

| Loss threshold | Scale 10 | Scale 25 | Scale 50 | Scale 75 | Scale 100 |
|----------------|----------|----------|----------|----------|-----------|
| < 0.1          | epoch 5  | epoch 4  | epoch 3  | epoch 3  | epoch 2   |
| < 0.01         | epoch 6  | epoch 5  | epoch 4  | epoch 4  | epoch 4   |
| < 0.001        | epoch 10 | epoch 7  | epoch 7  | epoch 5  | epoch 5   |

Values are the slower of the two models at each scale point.
The last 10-20 epochs contribute negligible *loss* reduction, but recall
requires the full training window (see Early Stopping Exploration below).

**Note:** Loss convergence is a training diagnostic, not a recall guarantee.
Low loss is necessary but not sufficient for exact key recall. The Early
Stopping Exploration (2026-03-23) quantified this gap: loss reaches < 0.01
by epoch 5-10, but recall requires 19-25 epochs to reach 100%. The 10-15
epoch gap between loss convergence and recall convergence means loss-based
early stopping is fundamentally unsafe for indexed key training.

### Failure analysis

**Gemma failures:** 2 total across all scales. Both are graph13 at scale 25/50
(parse failure, null JSON output). Self-corrected at scale 75/100 with more
training data. No recall failures — only generation format issues.

**Mistral failures:** 6 total, all at scale 50-100. All are generation quality
issues — garbled text (missing spaces, negated facts, truncated words), not
convergence failures. SimHash confidence 0.81-0.91 indicates the model memorized
the content but produces imperfect output.

### Distillation quality — ground truth analysis (2026-03-19)

PerLTQA provides 394 ground-truth QA pairs for Liang Xin across four categories.
The distillation pipeline only sees dialogue text, so the extractable ceiling
depends on what's actually stated in conversations:

| Category | GT QA pairs | Extractable from dialogue text? |
|----------|-------------|--------------------------------|
| Profile (metadata) | 14 | No — gender, age, etc. not stated |
| Social relationships | 50 | Partially — names/roles mentioned |
| Events (narratives) | 198 | Partially — discussed but narratives richer |
| Dialogues (direct) | 132 | Yes — directly in conversation text |

Semantic matching (embedding similarity ≥ 0.65) of distilled pairs against
the 330 dialogue + event GT pairs:

| Metric | Gemma | Mistral |
|--------|-------|---------|
| Distilled pairs | 100 | 41 |
| Match GT (≥0.65) | 55 (55%) | 33 (80%) |
| Unique GT covered | 26 | 20 |
| Novel (no GT match) | 45 | 8 |

**Mistral is more precise, Gemma has higher recall.** Mistral's lower yield at
temperature=0.0 reflects selective extraction — 80% of what it produces maps to
ground truth. Gemma extracts more aggressively but 45% are low-information graph
artifacts ("Training boosted morale", "Who implemented Changes?") that are
technically present in dialogue text but not meaningful personal facts.

Both models miss core profile facts (gender, age, occupation) and use only the
nickname "Xinxin" — never "Liang Xin." Profile facts aren't stated in dialogues.

Combined coverage: 38/330 GT pairs (12%). Neither model approaches the ceiling
from dialogue text alone. Improving extraction density is a model optimization
task — outside the scope of the first paper, which validates the memory
mechanism itself.

### Methodology update (2026-03-18)

Critical design review identified and fixed the following issues:

1. **keyed_pairs now persisted alongside adapters.** `train_indexed_keys` saves
   `keyed_pairs.json` in the output directory for every training run. Enables
   post-hoc re-evaluation and independent verification of results.

2. **Unified scoring metric for PM vs RAG comparison.** Both parametric and RAG
   recall are now scored by embedding similarity (`all-MiniLM-L6-v2` cosine) with
   a 0.75 match threshold, in addition to SimHash confidence for parametric.
   This enables apples-to-apples comparison in the summary table.

3. **Unified max_new_tokens.** RAG evaluation now uses `max_new_tokens=200`
   (previously hardcoded at 150 in the library function). Parametric uses 256
   due to JSON wrapper overhead — noted in results metadata.

4. **RAG indexes distilled keyed_pairs**, not original QA pairs. Same data as
   parametric training, ensuring the only variable is the storage mechanism.

5. **Design note:** Scale points use nested subsets (scale N = first N pairs from
   single distillation). This mirrors real usage (accumulating facts over time)
   but means scale points are not independent observations.

**Rerun completed 2026-03-19. Results above supersede previous run.**

### Key takeaways

1. **Gemma achieves 100% recall across all scales (10-100 keys).** The indexed
   key mechanism is fully reliable on Gemma at this scale.

2. **Mistral's lower yield is higher precision, not a defect.** 54 QA pairs from
   30 sessions, but 80% map to ground truth vs Gemma's 55%. Temperature=0.0
   made Mistral selective — it refuses to extract low-information fragments.
   Mistral achieves 100% recall at all reachable scale points (up to 54 keys).
   Improving extraction density is model optimization work for future papers.

3. **Training time is linear and predictable.** ~43s/key (Gemma) and ~31s/key
   (Mistral), constant across all scale points. No superlinear growth. At scale
   100, training takes 49-71 min — feasible for idle-time background processing.

4. **Distillation coverage is low but sufficient for mechanism validation.**
   Combined coverage is 12% of ground-truth dialogue+event facts. Both models
   miss profile metadata entirely. Improving extraction density (prompt tuning,
   multi-pass extraction, model-specific temperature) is future work — the first
   paper validates the memory mechanism on whatever the pipeline produces.

5. **30 epochs is sufficient with ~10 epochs margin.** Loss converges by epoch
   5-10 but recall requires 19-25 epochs to reach 100% (see Early Stopping
   Exploration below). Loss convergence does NOT predict recall convergence.
   Early stopping saves at most 5-11 epochs (~15-35% time) — a minor
   optimization, not a game changer. Stick with 30 epochs for all experiments.

---

## Early Stopping

**Status:** IMPLEMENTED, NOT ACTIVE — **not recommended** based on exploration results

### Implementation

Loss-based early stopping using epoch-average loss (not per-step):
- **Threshold:** avg epoch loss < 0.01
- **Floor:** 10 epochs minimum
- **Patience:** 2 consecutive epochs below threshold
- **Hard cap:** 30 epochs

Implemented in `LossEarlyStoppingCallback` (`paramem/training/trainer.py`).
Configurable via `TrainingConfig`. Defaults to off (`early_stopping=False`).

### Early validation (scale=25, PerLTQA data, loss-based)

| Metric | 30-epoch baseline | Early stopping | Delta |
|--------|-------------------|----------------|-------|
| Gemma recall | 24/25 (96%) | 24/25 (96%) | No change |
| Gemma epoch reached | 30 | 19 | -37% |
| Gemma training time | 17.4 min | 11.0 min | -37% |
| Mistral recall | 25/25 (100%) | 22/25 (88%) | **-3 keys** |
| Mistral epoch reached | 30 | 22 | -27% |
| Mistral training time | 13.9 min | 8.9 min | -36% |

Note: The smoke tests ran with the distillation pipeline fixes (extractor null
handling, QA generator `A:` format instruction, markdown cleaning) that were
applied after the Test 1 full runs. Test 1 results reflect the old pipeline;
smoke test results reflect the fixed pipeline.

### Early Stopping Exploration (2026-03-23)

**Script:** `experiments/test_early_stopping.py`
**Status:** COMPLETE — Mistral, both local and Claude extraction

Per-epoch recall probing at scales 25 and 50 using a single HF Trainer call
with a custom `RecallProbingCallback` (preserves optimizer state across epochs).

**Critical bug found and fixed:** An earlier version used separate Trainer
instances per epoch, which reset optimizer state (Adam momentum/variance) and
learning rate schedule. This caused recall to plateau at 72-80% with the
identical data that reaches 100% in a single Trainer call. The fix uses
`TrainerCallback.on_epoch_end` within a single training run.

#### Results: Local extraction (Mistral 7B, rank 8, 30 epochs)

Data: 54 QA pairs from 25 sessions (Liang Xin), yield 2.2 QA/session.

**Scale 25:**

| Epoch | Loss | Recall | Rate | Confidence |
|-------|------|--------|------|------------|
| 1 | 2.388 | 0/25 | 0% | 0.000 |
| 5 | 0.382 | 0/25 | 0% | 0.031 |
| 10 | 0.175 | 2/25 | 8% | 0.110 |
| 15 | 0.086 | 15/25 | 60% | 0.630 |
| 19 | 0.001 | 24/25 | 96% | 0.960 |
| **20** | **0.001** | **25/25** | **100%** | **1.000** |
| 25 | 0.000 | 25/25 | 100% | 1.000 |
| 30 | 0.000 | 25/25 | 100% | 1.000 |

First 100%: epoch 20. Stable through epoch 30 (11 consecutive perfect).

**Scale 50:**

| Epoch | Loss | Recall | Rate | Confidence |
|-------|------|--------|------|------------|
| 1 | 1.426 | 0/50 | 0% | 0.000 |
| 5 | 0.256 | 0/50 | 0% | 0.030 |
| 10 | 0.084 | 4/50 | 8% | 0.112 |
| 15 | 0.000 | 35/50 | 70% | 0.731 |
| 18 | 0.000 | 48/50 | 96% | 0.976 |
| **19** | **0.001** | **50/50** | **100%** | **1.000** |
| 25 | 0.000 | 50/50 | 100% | 1.000 |
| 30 | 0.000 | 50/50 | 100% | 1.000 |

First 100%: epoch 19. Stable through epoch 30 (12 consecutive perfect).

#### Results: Claude Sonnet extraction (Mistral 7B training, rank 8, 30 epochs)

Data: 56 QA pairs from 5 sessions (Liang Xin), yield 11.2 QA/session.
Claude Sonnet extracts triples via API; Mistral generates QA from those triples.

**Scale 25:** First 100% at epoch 21. Stable through epoch 30 (10 consecutive).
**Scale 50:** First 100% at epoch 25. Stable through epoch 30 (6 consecutive).

#### Extraction yield comparison

| Metric | Mistral (local) | Claude Sonnet (API) |
|--------|-----------------|---------------------|
| Sessions needed for 50+ QA | 25 | 5 |
| QA yield per session | 2.2 | 11.2 |
| Sessions with zero extraction | 16/25 (64%) | 0/5 (0%) |
| Distillation time | 1032s (17 min) | 134s (2 min) |
| Final recall (both scales) | 100% | 100% |

#### Key findings

1. **Loss does NOT predict recall.** Loss converges by epoch 5-10 but recall
   requires 19-25 epochs. The gap is 10-15 epochs. Loss-based early stopping
   at any threshold would terminate too early.

2. **Recall convergence is a phase transition.** Keys are either 1.000 confidence
   or 0.000 — no gradual improvement. The transition from 0% to 100% happens
   in a ~5 epoch window (epochs 15-20), not gradually across all 30 epochs.

3. **30 epochs provides adequate margin.** Recall reaches 100% at epoch 19-25
   depending on extraction source and scale. 30 epochs gives 5-11 epochs of
   stability margin. Early stopping saves at most ~35% time — a minor
   optimization that adds complexity without meaningful benefit.

4. **Extraction quality is the real bottleneck.** Mistral fails to extract from
   64% of sessions. Claude extracts from 100% with 5x higher yield. Both
   produce QA that trains to 100% recall — the storage mechanism is not the
   constraint. This quantitatively confirms the paper's extraction bottleneck
   limitation.

5. **Optimizer state preservation is critical.** Separate Trainer instances per
   epoch (resetting Adam momentum) causes recall to plateau at 72-80%.
   A single Trainer call with callback-based probing reaches 100%. This is a
   training methodology issue, not an adapter capacity issue.

---

## Test 2: Contradiction Resolution

**Script:** `experiments/test2_contradictions.py`
**Status:** COMPLETE — both models (2026-03-20)

### Methodology updates (2026-03-19)

1. **Predicate-based evaluation matching** replaces question-text similarity.
   Each keyed pair carries `source_predicate` from QA generation, mapped
   deterministically to contradiction chains. No threshold-sensitive embedding
   matching for chain assignment.
2. **`is_current` threshold 0.75** (lowered from 0.85) for answer quality scoring.
3. **QA generator prompt fixed** for verbose narrative objects — focus on
   current state, not transitions.
4. **`gradient_checkpointing_disable` moved** to top of session loop (was after
   merge, causing garbage output for model strategy on Gemma).

**Objective:** Validate that the system detects and resolves contradictions when
facts change over time (e.g. "Alex works at AutoMate" → "Alex works at SpaceX").

**Design:** 10 fact chains with 3 temporal versions each across 10 sessions.
Two strategies compared: graph-only (predicate normalization) and model-assisted
(LLM semantic reasoning). Uses pre-defined session graphs for determinism.
Fresh adapter per session trained on the resolved graph state.

### Results (2026-03-20)

| Metric | Gemma 2 9B | Mistral 7B |
|--------|-----------|------------|
| Contradictions detected | 20/20 | 20/20 |
| Graph — final key recall | 5/10 | 10/10 |
| Graph — final current-fact | 3/10 | 8/10 |
| Model — final key recall | 5/10 | 10/10 |
| Model — final current-fact | 3/10 | 8/10 |
| Graph — peak key recall | 10/10 (sessions 3-7, 9) | 10/10 |
| Time | 211 min | 162 min |

**Config:** rank=8, alpha=16, 30 epochs, lr=1e-4, temperature=0.0,
repetition_penalty=1.1.

### Key findings

1. **Contradiction detection is perfect on both models (20/20).** Graph
   predicate normalization reliably detects same-predicate contradictions.
   The graph is correctly updated with current-version triples.

2. **Key recall diverges between models on model strategy.** Gemma 5/10 vs
   Mistral 10/10. The model-assisted strategy requires clean LLM generation
   during the merge step — Gemma's verbose output degrades the QA pairs.
   Graph strategy achieves 10/10 key recall on both models.

3. **Current-fact accuracy reflects QA generation quality.** Gemma's 3/10
   vs Mistral's 8/10 current-fact score is not a contradiction detection
   failure — the graph has the correct facts. The QA generator sometimes
   produces answers that rephrase the current fact below the 0.75 similarity
   threshold. Mistral's cleaner generation scores higher.

4. **Graph strategy matches or exceeds model strategy.** The LLM-assisted
   approach adds no value for same-predicate contradictions and degrades
   quality on Gemma. Graph normalization is sufficient, faster, and
   model-agnostic.

5. **The mechanism is sound; QA generation is the bottleneck.** Same finding
   as Test 1 — the memory architecture works reliably, the upstream
   distillation pipeline determines quality.

### Key takeaways

1. **The contradiction mechanism works end-to-end.** Detection (graph predicate
   normalization), resolution (triple replacement), training (fresh adapter on
   resolved graph), and recall (indexed keys) all function correctly.

2. **Failures trace to QA generation, not the architecture.** Both models'
   single misses are dot-answers from the QA generator. The knowledge is in the
   graph; the pipeline occasionally fails to express it as a trainable QA pair.

3. **Test design note:** Fresh adapter per session doesn't match the production
   pattern. Test 2b (below) validates incremental training with a persistent adapter.

### Infrastructure improvements from Test 2

1. **Raw output preservation in `probe_key()`:** On failure, returns
   `{"raw_output": ..., "failure_reason": ...}` instead of discarding.
   Per-key recall data saved in results for diagnostics.

2. **`--strategy` flag:** Run `--strategy graph` or `--strategy model`
   individually. Default: both.

3. **Enriched registry with temporal metadata:** `build_enriched_registry()`
   adds per-key: `created_at`, `last_seen_at`, `session_id`, `status`
   (active/stale), `stale_since`, `stale_cycles`. Backward compatible.

4. **Rolling key reclamation:** `mark_stale()`, `get_reclaimable_keys()`,
   `get_active_keys()`. Stale keys are deregistered (Layer 1 rejects them).
   After a configurable number of cycles, key IDs can be recycled. Mechanism
   in place, not yet wired into the training loop.

---

## Test 2b: Incremental Contradiction Resolution

**Script:** `experiments/test2b_incremental_contradictions.py`
**Status:** COMPLETE — both models (confirmed 2026-03-20, identical results)

**Objective:** Test contradiction resolution with a single persistent adapter
trained cumulatively across sessions — the production consolidation pattern.
Measure whether old facts are overwritten, whether unrelated facts survive, and
how many cycles forgetting takes.

**Design:** Three phases on a single adapter (never reinitialized):
- **Phase A (learning):** 3 cycles training initial facts (10 chains + 3 control = 16 keys)
- **Phase B (contradiction):** Graph resolves 10 fact changes. New keys for updated
  facts. Old keys marked stale in enriched registry.
- **Phase C (decay):** 5 cycles training on current facts only. Diagnostic probes
  on stale keys (bypassing registry) to measure old content residue. Control fact
  probes to detect catastrophic forgetting.

### Results (2026-03-17)

| Metric | Gemma 2 9B | Mistral 7B |
|--------|-----------|------------|
| Current fact recall (all cycles) | 16/16 (100%) | 16/16 (100%) |
| Control fact stability (decay phase) | 1.000 | 1.000 |
| Stale keys returning old content | 0/5 | 0/5 |
| Stale keys returning new content | 5/5 | 5/5 |
| Cycles to overwrite stale content | 1 | 1 |
| Catastrophic forgetting observed | None | None |

**Config:** rank=8, alpha=16, 30 epochs, lr=1e-4, single persistent adapter.

### Per-cycle summary (identical for both models)

| Cycle | Phase | Current | Stale | Control |
|-------|-------|---------|-------|---------|
| 1 | learning | 16/16 | n/a | 0.000 |
| 2 | learning | 16/16 | n/a | 0.000 |
| 3 | learning | 16/16 | n/a | 0.000 |
| 4 | decay | 16/16 | 3/5 (sim=0.700) | 1.000 |
| 5 | decay | 16/16 | 3/5 (sim=0.700) | 1.000 |
| 6 | decay | 16/16 | 3/5 (sim=0.700) | 1.000 |
| 7 | decay | 16/16 | 3/5 (sim=0.700) | 1.000 |
| 8 | decay | 16/16 | 3/5 (sim=0.700) | 1.000 |

Note: "3/5" stale similarity is a false positive from shared predicate prefixes
(e.g. "Alex works at..." appears in both old and new answers). Inspection of
raw output confirms all 5 stale keys return the NEW content, not the old.
Control facts show 0.000 in learning phase due to marginal SimHash confidence
in early cycles (0.688); they reach 1.000 once the adapter stabilizes.

### Key findings

1. **Forgetting is immediate.** One retraining cycle on the resolved graph
   overwrites stale content. The symmetry hypothesis (N cycles to learn ≈ N
   cycles to forget) is wrong in our favor.

2. **Zero catastrophic forgetting.** Control facts and non-contradicted facts
   maintain perfect recall throughout. Adding, updating, and overwriting facts
   does not degrade unrelated knowledge.

3. **Model-agnostic.** Both Gemma 2 9B and Mistral 7B produce identical results.
   The findings are architectural, not model-specific.

4. **Scale caveat.** Tested at 16 keys. Test 1 showed 100/100 at 100 keys in
   batch mode. Incremental contradiction resolution at 100+ keys is still open.

### Open items

- **Key reuse vs fresh assignment:** Test 2b showed that stale key positions
  are immediately overwritten with current content. This happened because
  `assign_keys` re-keyed all QA pairs from scratch each cycle, and some key
  IDs happened to be reused. The deliberate version — explicitly training old
  keys against the updated fact — could reinforce the new content more
  reliably. Open question: does deliberate key reuse improve recall over fresh
  assignment? Could eliminate the need for stale key tracking entirely.
- **Scale:** Incremental contradiction resolution at 100+ keys.

---

## Test 3: Reasoning Quality Parity with RAG

**Script:** `experiments/test3_inference.py`
**Status:** COMPLETE — both models (2026-03-20)

**Objective:** Verify that reasoning over parametrically recalled facts achieves
comparable quality to RAG with equivalent context. Reasoning is a base model
capability — the claim is quality parity with better operational properties.

**Design:** ~50 QA pairs distilled from PerLTQA dialogues, trained as indexed
keys. The LLM generates 14-15 inference questions requiring 2+ facts. Three
evaluation conditions:

- **PM Recall+Reason:** Adapter active. Enumerate all keys → reconstruct all
  facts → feed as context → answer.
- **PM Adapter-Only:** Adapter active. Ask directly, no explicit retrieval.
  Diagnostic baseline — different system prompt, not directly comparable.
- **RAG all facts:** Adapter disabled (base model). All facts loaded from
  store → feed as context → answer. Fair comparison to PM Recall+Reason.

**Key comparison:** PM Recall+Reason vs RAG all facts — both have all facts in
context, identical prompt format and system prompt. The only difference is the
source: parametric recall from adapter weights vs loaded from external store.

### Results (2026-03-20)

| Condition | Gemma OK | Gemma sim | Gemma gen/q | Mistral OK | Mistral sim | Mistral gen/q |
|-----------|----------|-----------|-------------|------------|-------------|---------------|
| PM Recall+Reason | 13/14 | 0.687 | 2.40s | 9/14 | 0.566 | 1.76s |
| PM Adapter-Only | 12/14 | 0.633 | 2.98s | 7/14 | 0.446 | 1.12s |
| RAG all facts | 12/14 | 0.679 | 2.26s | 9/14 | 0.525 | 1.71s |

**One-time reconstruction overhead:** Gemma 162s (11.56s amortized over 14
questions), Mistral 121s (8.64s amortized). Reconstruction happens once per
session, not per query.

**Config:** rank=8, alpha=16, 30 epochs, temperature=0.0.

### Key findings

1. **PM matches or exceeds RAG.** PM Recall+Reason vs RAG all facts:
   Gemma 0.687 vs 0.679 (1.2% gap), Mistral 0.566 vs 0.525 (7.8% gap).
   PM is at least as good as RAG on both models, slightly better on Mistral.
   The adapter-tuned model reasoning over recalled facts performs at least as
   well as the base model reasoning over loaded facts.

2. **Per-query generation is equivalent.** PM 2.40s vs RAG 2.26s (Gemma),
   PM 1.76s vs RAG 1.71s (Mistral). Once context is built, both approaches
   have the same generation cost.

3. **PM has a one-time reconstruction overhead.** 162s (Gemma) to enumerate
   and probe all keys. This amortizes across queries in a session. RAG's
   equivalent cost (loading facts from store) is negligible at 50 facts but
   grows with scale.

4. **RAG latency excludes retrieval overhead.** The measured 2.26s is
   generation only — embedding the query, searching the index, and ranking
   results are not included. At 50 facts this is milliseconds; at production
   scale it becomes significant. PM's retrieval mechanism (key enumeration +
   probing) is a different cost structure — one-time reconstruction amortized
   across queries, rather than per-query embedding + search.

5. **Adapter-Only is the production PM path for direct queries.** No
   reconstruction, no context — the model answers from weights. 2.98s (Gemma),
   1.12s (Mistral). This is how PM serves simple factual questions.

### Scale caveat

At 50 facts (~300 tokens of context), exhaustive recall is cheap. At 1000+
facts, injecting all facts would be expensive. Selective enumeration by entity,
topic, or recency would be needed at larger scales.

---

## Test 4: Multi-Session Pipeline Robustness

**Script:** `experiments/test4_reinforcement.py`
**Status:** COMPLETE — both models (2026-03-20)

### Methodology update (2026-03-18)

1. **Reframed from "reinforcement" to "pipeline robustness."** The cumulative
   pool overwrites by fact_id — all facts get one QA pair regardless of mention
   frequency. The test validates train-delete-retrain cycle reliability, not
   frequency-dependent consolidation.
2. **Added `skip_distill=True`** since QA pairs are pre-formed synthetic data.
   Avoids unnecessary distillation round-trip.
3. **RAG max_new_tokens unified to 200** (was 150).

**Rerun completed 2026-03-20. Results below supersede previous run.**

**Objective:** Test whether cumulative fact recall remains reliable across 10
train-delete-retrain cycles as facts accumulate from 3 to 30.

**Design:** 30 facts in three frequency tiers:
- 10 reinforced (appear in 3-4 of 10 sessions)
- 10 mentioned twice
- 10 single mention

Each session adds new facts to a cumulative pool (overwrite-by-fact_id).
Uses `skip_distill=True` — clean synthetic QA pairs bypass graph extraction.
Fresh adapter per session trained on all cumulative facts.

### Results (2026-03-20, final run)

| Metric | Gemma 2 9B | Mistral 7B |
|--------|-----------|------------|
| Final recall (session 10) | 30/30 (100%) | 30/30 (100%) |
| Reinforced (3-4 mentions) | 10/10, conf=1.000 | 10/10, conf=1.000 |
| Mentioned twice | 10/10, conf=1.000 | 10/10, conf=1.000 |
| Single mention | 10/10, conf=1.000 | 10/10, conf=1.000 |
| RAG similarity (reinforced) | 1.000 | 1.000 |
| RAG similarity (mentioned twice) | 0.988 | 0.993 |
| RAG similarity (single mention) | 1.000 | 1.000 |
| Time | 173 min | 132 min |

**Config:** rank=8, alpha=16, 30 epochs, lr=1e-4, temperature=0.0,
repetition_penalty=1.1.

### Key findings

1. **Perfect recall on both models.** 30/30 across all frequency tiers, all
   10 sessions. The adapter unwrap fix (replacing `delete_adapter` with base
   model unwrap) eliminated the cross-key contamination failures from the
   previous run (Gemma was 28/30).

2. **No frequency effect.** All tiers (reinforced, mentioned twice, single
   mention) achieve identical 100% recall and 1.000 confidence. The cumulative
   pool overwrites by fact_id — frequency has no effect on training data.

3. **The mechanism is model-agnostic.** Both models produce identical results.

4. **Cumulative training is robust.** 10 train-delete-retrain cycles with
   facts accumulating from 7 to 30 — no degradation at any point.

---

## Test 4b: Incremental Learning Without Full Replay

**Script:** `experiments/test4b_incremental_no_replay.py`
**Status:** COMPLETE — both models (2026-03-21)

**Objective:** Test whether facts survive when only new keys are trained each
cycle without replaying existing keys. The "daily incremental" regime.

**Design:** Train 20 initial keys (baseline), then 5 incremental cycles adding
5 new keys each — training ONLY the new keys on the existing adapter. Per-epoch
recall probing tracks convergence speed. Final full-replay retrain on all 45
keys validates recovery.

### Results (2026-03-21)

| Cycle | Keys before | New | New recall | Old recall | Recall@ | Gemma time | Mistral time |
|-------|-------------|-----|------------|------------|---------|------------|-------------|
| 0 (baseline) | 0 | 20 | 20/20 | n/a | n/a | 854s | 604s |
| 1 | 20 | 5 | 5/5 | 4/20 (G) 0/20 (M) | e9 / e15 | 278s | 321s |
| 2 | 25 | 5 | 5/5 | 0/25 (G) 0/25 (M) | e13 / e19 | 559s | 565s |
| 3 | 30 | 5 | 5/5 | 3/30 (G) 0/30 (M) | e4 / e7 | 139s | 166s |
| 4 | 35 | 5 | 5/5 | 0/35 (G) 0/35 (M) | e9 / e16 | 360s | 367s |
| 5 | 40 | 5 | 5/5 | 1/40 (G) 0/40 (M) | e6 / e5 | 207s | 209s |
| Full retrain | — | 45 | 44/45 (G) 45/45 (M) | — | — | 2162s | 1360s |

### Key findings

1. **New keys are learned perfectly every cycle (5/5).** The adapter can
   absorb new facts regardless of how many existing facts are in the weights.
   Convergence takes 4-19 epochs with no clear slowdown as keys accumulate.

2. **Old keys suffer catastrophic forgetting.** Training 5 new keys for 30
   epochs at lr=1e-4 destroys nearly all old keys. Gemma retains 0-4 of 20-40;
   Mistral retains 0. This is consistent across both models.

3. **Full replay restores near-perfect recall.** Mistral recovers 45/45
   (100%). Gemma recovers 44/45 (97.8%) — the one miss is a SimHash
   paraphrase threshold issue (confidence 0.734 < 0.75, embedding similarity
   0.994), consistent with the graph23 finding in Test 5.

4. **Incremental training without replay is not viable at rank 8.** The LoRA
   subspace is too constrained — new training overwrites old patterns. Full
   replay is mandatory for a single adapter.

5. **Implication for production:** nightly full-replay consolidation is
   required. The multi-adapter session routing architecture (train per-session
   adapters independently, consolidate with full replay overnight) is the
   correct approach. See ROADMAP_v2.md.

---

## Test 5: Natural Recall

**Script:** `experiments/test5_natural_recall.py`
**Status:** COMPLETE — both models (2026-03-21)

### Methodology update (2026-03-18)

1. **Added per-question natural recall.** Each training question asked directly
   (one at a time, adapter active, no key) — fair comparison with keyed recall.
2. **Unified scoring metric.** Both keyed and natural recall scored by embedding
   similarity (`compute_similarity`, 0.75 threshold). SimHash retained as
   secondary diagnostic for keyed recall.
3. **Scored against keyed_pairs** (post-distillation), not original `qa_pairs`.
   Eliminates ground-truth asymmetry.
4. **Uses PerLTQA data** (50 QA pairs, was 20 synthetic).
5. **Broad natural probes retained** as supplementary data alongside per-question.

**Rerun completed 2026-03-21. Results below supersede previous run.**

**Objective:** Quantify how much stored knowledge is recoverable through
natural-language prompts alone (without indexed keys), motivating the key
mechanism.

**Design:** Train adapter on 50 QA pairs with indexed keys (rank 8, 30 epochs).
Verify keyed recall (control), then probe with 10 open-ended natural-language
prompts — broad ("List everything you know about the user") to narrow ("What do
you know about the user's daily routine?"). Count unique facts surfaced across
all probes (deduplicated). No RAG comparison — test focuses on the keyed vs.
natural recall gap.

**Key metrics:** Keyed recall rate, natural recall rate (unique facts / total
facts), per-prompt fact count.

### Results (2026-03-21, final run)

| Metric | Gemma 2 9B | Mistral 7B |
|---|---|---|
| Keyed recall (SimHash) | 49/50 (conf=0.980) | 49/50 (conf=0.980) |
| Keyed recall (embedding) | 49/50 (sim=0.980) | 49/50 (sim=0.980) |
| Per-question natural | 50/50 (sim=1.000) | 50/50 (sim=1.000) |
| Broad natural (unique) | 22/50 | 32/50 |

**Per-question natural recall** asks each training question directly (without
the key wrapper). 50/50 on both models means the adapter encodes facts well
enough for direct question-answer recall — the key mechanism adds
*addressability and verification*, not the ability to recall.

**Broad natural probes** ("Tell me everything you know") surface only 22-32
of 50 facts. The model selectively activates a subset of stored knowledge.
Mistral surfaces more (32 vs 22) — more informative in open-ended responses.

**One keyed recall miss (graph23):** SimHash confidence 0.734, below the 0.75
threshold. The model recalled the correct content but paraphrased it (different
word order). Embedding similarity was 0.994 — a SimHash strictness issue, not
a recall failure. Identical on both models.

**Indexed keys close the gap** by providing deterministic, exhaustive, verifiable
access to all stored facts — addressing the 56-78% of facts that broad probes
miss.

---

## Test 6: Parametric vs RAG Head-to-Head

**Script:** `experiments/test6_footprint.py`
**Status:** COMPLETE — both models (2026-03-21)

**Objective:** Head-to-head comparison of parametric retrieval (indexed keys)
vs RAG on the same fact set at multiple scales. Three dimensions: storage
footprint, inference latency, and recall quality.

**Design (redesigned):**
- **Data:** PerLTQA eval QA pairs (character "Liang Xin"), 2x input buffer
  for distillation loss. Distill once at max scale, subset for smaller scales.
- **Scales:** 10, 25, 50, 100 keys (default).
- **Storage:** Final adapter weights only (via `selected_adapters`), not
  training checkpoints. RAG total includes embedding model (~87 MB) for fair
  comparison. Breakdown separates fixed costs (adapter, embedding model) from
  per-fact variable costs (registry, index).
- **Latency:** Three conditions (bare model, parametric, RAG), all with
  `max_new_tokens=200`, 3 warm-up queries discarded before timing. Query count
  capped to available keys minus warm-up.
- **Recall:** Both systems scored by embedding similarity (`compute_similarity`,
  `all-MiniLM-L6-v2` cosine, 0.75 threshold). SimHash confidence retained as
  secondary diagnostic for parametric. Both evaluated on identical keyed_pairs.
- **RAG note:** Indexes pre-extracted QA pairs (upper bound on RAG performance).

**Methodology fixes applied:**
1. PerLTQA data with 2x buffer (was: 20-fact synthetic, scales capped)
2. Final adapter only for storage (was: all checkpoints, 94x overestimate)
3. Same questions for PM and RAG (was: distilled vs original)
4. Same scoring metric (was: SimHash vs embedding similarity)
5. Same max_new_tokens (was: 256 vs 150)
6. Warm-up queries (was: none)
7. Single distillation, nested subsets (was: per-scale distillation)
8. `selected_adapters` in save_pretrained (was: saving all accumulated adapters)

### Results (2026-03-21, final run)

**Gemma 2 9B:**

| Scale | PM Recall | RAG Recall | PM Size | RAG Size | PM Latency | RAG Latency |
|-------|-----------|------------|---------|----------|------------|-------------|
| 10 | 10/10 | 6/10 | 35 MB | 89 MB | 4819ms | 1805ms |
| 25 | 25/25 | 16/25 | 35 MB | 89 MB | 4341ms | 1515ms |
| 50 | 49/50 | 36/50 | 35 MB | 89 MB | 4380ms | 1546ms |
| 100 | 99/100 | 78/100 | 35 MB | 89 MB | 4437ms | 1546ms |

**Mistral 7B:**

| Scale | PM Recall | RAG Recall | PM Size | RAG Size | PM Latency | RAG Latency |
|-------|-----------|------------|---------|----------|------------|-------------|
| 10 | 10/10 | 10/10 | 27 MB | 89 MB | 3532ms | 1932ms |
| 25 | 25/25 | 25/25 | 27 MB | 89 MB | 3160ms | 1271ms |
| 50 | 49/50 | 50/50 | 27 MB | 89 MB | 2732ms | 1242ms |
| 100 | 99/100 | 96/100 | 27 MB | 89 MB | 2931ms | 1218ms |

### Key findings

1. **PM storage is constant (O(1)).** 35 MB (Gemma) / 27 MB (Mistral)
   regardless of fact count. RAG is 89 MB (includes embedding model). PM is
   smaller at all scales tested; RAG's per-fact variable cost would overtake
   PM at ~20K facts.

2. **PM recall exceeds RAG on Gemma.** 99/100 vs 78/100 at scale 100. The
   RAG pipeline (top-3 retrieval + generation) loses recall quality as scale
   grows. Mistral's RAG is stronger (96/100) due to cleaner generation.

3. **RAG is faster per query.** 1.2-1.9s vs 2.7-4.8s for PM. RAG benefits
   from disabled adapter (simpler forward pass) and focused context (top-3
   facts). PM generates from full adapter weights with no context assistance.

4. **PM latency is faster than bare model.** The adapter produces more
   confident, concise answers than the unassisted base model (4.4s vs 6.7s
   on Gemma). The adapter adds knowledge, not overhead.

5. **RAG latency excludes retrieval overhead.** The measured times are
   generation only. Embedding the query and searching the index add
   milliseconds at 100 facts but grow linearly at production scale.

---

## Test 7: Second Persona

**Script:** `experiments/test7_second_persona.py`
**Status:** COMPLETE — both models (2026-03-20)

**Objective:** Validate that the architecture generalizes beyond a single user
persona, and that separate adapters maintain isolation on the same base model.

**Design (redesigned):**
- **Data:** Two PerLTQA characters (selected by eval QA count, excluding
  "Liang Xin" for independence from Tests 1-5). 50 QA pairs each.
- **Key namespaces:** Non-overlapping. Persona A: `graph1..graphN`,
  Persona B: `graph1001..graph1000+N`. Ensures cross-contamination probes
  test true isolation, not namespace collision.
- **Training:** `skip_distill=True` — eval QA pairs used directly as indexed
  keys to isolate persona generalization from distillation quality.
- **Evaluations:**
  1. Per-persona recall (should be comparable)
  2. Re-evaluation of persona A after persona B training (isolation check)
  3. Cross-contamination: probe persona A keys with persona B adapter
     (and vice versa) — should fail on untrained keys
- **Fallback:** Synthetic persona B (20 facts), both personas capped equally.

**Methodology fixes applied:**
1. Non-overlapping key namespaces (was: both used graph1..N)
2. skip_distill=True (was: lossy distillation round-trip on eval QA)
3. Character selection filters by eval QA count (was: dialogue count)
4. Re-evaluation of persona A after persona B (was: missing)
5. Excludes "Liang Xin" (was: likely selected as persona A)
6. Equal persona sizes in fallback mode (was: 50 vs 20)

### Results (2026-03-19, Gemma)

| Metric | Gemma 2 9B |
|--------|-----------|
| Persona A (Cai Xiuying) recall | 50/50 (1.000) |
| Persona B (Xiong Fei) recall | 50/50 (1.000) |
| Persona A after B training | 50/50 (zero degradation) |
| Cross-contamination A→B | 0/50 leaked |
| Cross-contamination B→A | 0/50 leaked |
| Training time per persona | ~35 min |

**Config:** rank=8, alpha=16, 30 epochs, skip_distill=True, non-overlapping
key namespaces (graph1..50 vs graph1001..1050).

### Key findings

1. **Perfect isolation.** Both adapters achieve 100% recall. Training persona B
   does not degrade persona A. Cross-contamination is zero in both directions.
2. **Architecture validated.** Multiple independent LoRA adapters on a single
   base model work as designed. The mechanism is model-agnostic — confirmed
   on both Gemma 2 9B and Mistral 7B with identical results.

---

## Test 7b: Multi-Adapter Composition (Exploratory)

**Script:** `experiments/test7b_merged_personas.py`
**Status:** COMPLETE — both models (2026-03-21) — negative result

**Objective:** Test whether two independently trained LoRA adapters can serve
both personas simultaneously without adapter switching. Two approaches tested:
additive composition (both adapters active) and weight merging.

**Design:** Load persona A and persona B adapters from Test 7. Test:
1. **Additive composition:** `set_adapter(["persona_a", "persona_b"])` — both
   LoRA deltas applied in each forward pass, outputs summed.
2. **Weight merge:** `add_weighted_adapter([0.5, 0.5])` — combine weights
   into a single adapter (negative control, known failure).

### Results (2026-03-21, both models)

| Approach | Gemma A | Gemma B | Mistral A | Mistral B |
|----------|---------|---------|-----------|-----------|
| Individual (switching) | 50/50 | 50/50 | 50/50 | 50/50 |
| Composition (additive) | 0/50 | 1/50 | 0/50 | 1/50 |
| Merge [0.5, 0.5] | 0/50 | 1/50 | 0/50 | 1/50 |

### Analysis

Both composition and merging fail for indexed key recall. The combined LoRA
deltas (whether summed in the forward pass or averaged in the weights) destroy
the structured
key→QA mapping that each adapter learned independently.

Root cause: indexed key retrieval requires precise token-level generation
(exact JSON format with specific key-question-answer triples). Even small
perturbations to the LoRA weight matrices disrupt this precision. This contrasts
with task-level merging (e.g., translation + summarization) where approximate
outputs are acceptable.

### Literature context

This is a known limitation. Recent work confirms that linear weight averaging
causes destructive interference for LoRA adapters on structured tasks:

- **"Unraveling LoRA Interference"** (ACL 2025) shows that aligned LoRA weight
  vectors interfere when summed and proposes OSRM (orthogonal subspace
  training) as a mitigation — requires retraining from scratch.
- **"Understanding LoRA as Knowledge Memory"** (2026) systematically studies
  LoRA as a factual knowledge store and finds that combining multiple LoRAs
  for knowledge does not compose well.
- **"Position: Pause Recycling LoRAs"** (ICML 2025) argues that adaptive
  merging relies on shallow pattern matching, not genuine cross-task transfer.

The core insight: merging works for tasks with shared "solution templates"
(reasoning patterns, code structures) where approximate outputs are acceptable.
It fails for tasks requiring distinct memorized mappings — which is exactly
what indexed key recall does.

### Chained adapter composition (2026-03-26)

Follow-up experiment: train adapters sequentially with previous adapters frozen
but active in the forward pass (compose mode). Each adapter learns its residual
delta against `base + Δ₁ + ... + Δₙ₋₁`. Unlike Test 7b's independent training,
this makes each adapter complementary by construction.

**PEFT forward pass verified correct** via `verify_peft_forward.py`: adapter
active via PEFT = 15/15, same adapter merged into NF4 base = 0/15. PEFT merge
code properly dequantizes → adds in float → re-quantizes. The 0/15 is NF4
precision loss, not a broken addition.

| Condition | Session 1 | Session 3 |
|---|---|---|
| Single adapter (baseline) | 15/15 (100%) | — |
| Compose r8 (both PEFT active) | 1/15 (6.7%) | 11/11 (100%) |
| Compose r2,4,8 progressive | 0/15 (0%) | 11/11 (100%) |
| Merge r8 (s1 merged, s3 PEFT) | 0/15 (0%) | 9/9 (100%) |
| Merge r8 (both merged) | 0/15 (0%) | 0/9 (0%) |

**Conclusions:**
- Additive composition causes ~93-100% interference on earlier adapters even
  with frozen weights. The later adapter's delta acts as noise on all inputs.
- Lower-rank adapters are more fragile (r2 = 0% vs r8 = 6.7%).
- NF4 merge always destroys recall. The LoRA delta is too small relative to
  base weights to survive re-quantization.
- Adapter switching remains the only viable multi-adapter approach on NF4.

### Alternative composition approaches (from literature)

Other approaches from literature:
- **TIES-Merging** (Yadav et al. 2023): trim, elect signs, merge — resolves
  sign conflicts. Available in PEFT (`combination_type="ties"`).
- **DARE** (Yu et al. 2024): drop and rescale — prunes redundant parameters
  before merging. Available as `combination_type="dare_ties"`.
- **LoRI** (COLM 2025): freezes A matrices as random projections, trains
  sparse B with task-specific masks. Orthogonal by construction.
- **MoLoRA** (2026): per-token routing across adapters — conceptually ideal
  for key-based routing but requires a trained router.

### Implications

1. **Both composition and merging fail for indexed key recall.** Additive
   composition (both adapters active, deltas summed) produces the same
   near-zero recall as weight merging. Confirmed on both models and by
   recent literature. Not a hyperparameter issue.
2. **Adapter switching is the only viable multi-adapter approach.** `set_adapter`
   (single adapter active) achieves 50/50 on both personas. Switching is fast
   (metadata flip, no weight loading) and guaranteed to work.
3. **Future exploration:** TIES-Merging, DARE, orthogonal subspace training
   (LoRI, OSRM) may preserve structure — but would require retraining with
   orthogonality constraints, not post-hoc merging.

### Bug fixes discovered during Test 7b

1. **PEFT multi-adapter save/reload:** `get_peft_model` on an existing PeftModel
   re-wraps it, causing nested tensor names (`base_model.model.base_model.model.model.`).
   Fix: use `model.add_adapter()` instead for second+ adapters.
2. **PEFT `base_model_name_or_path`:** second adapter's config gets `None`,
   breaking reload. Fix: patch after creation from base model config.
3. **Adapter delete crash:** `delete_adapter` on sole adapter leaves PeftModel
   with empty config. Fix: unwrap to base model via `model.base_model.model`.

---

## Test 8: Large-Scale Incremental (500-Key Target)

**Script:** `experiments/test8_large_scale.py`
**Status:** IN PROGRESS — 56 cycles complete, **536 keys**, 100% recall (2026-04-08). Running. No ceiling found.

**Critical finding (2026-03-25):** Outlines constrained generation never succeeded in any Test 8 cycle — all 25 extraction attempts failed with `max_tokens` kwarg bug. Every successful extraction came from the unconstrained prompt-parse fallback, which itself only succeeded 3/25 times (12%). The 168 keys accumulated from the minority of sessions where fallback extraction worked. Fix: Outlines removed entirely, generate-once parse-once pipeline. **Validated at scale:** cycles 22-23 produced 46 new keys (12+34), QA yield jumped from 1.6 to 6.8/session.

### What it tests

Can indexed key retrieval scale to 500+ keys using the full automated consolidation
pipeline? This is the paper's primary scaling claim — everything beyond 100 keys is
new territory. Uses multi-character PerLTQA data processed through the complete
pipeline: session transcript → graph extraction → graph merge → QA generation →
indexed key training with full replay.

### Design

- **Model:** Mistral 7B Instruct v0.3, NF4 4-bit, rank 8
- **Cycle structure:** 5 sessions per cycle, full replay (retrain all keys from scratch each cycle)
- **Training:** 30 epochs per cycle, batch_size=1, gradient_accumulation=2
- **QA regeneration:** Full regeneration from cumulative graph each cycle (no cache — verifies pipeline integrity as adapter weights change the extraction landscape)
- **Data source:** Multi-character PerLTQA (~11 characters queued for 500 keys)
- **Pause/resume:** `tpause`/`tresume` commands, state.json + cumulative graph persisted at cycle boundaries
- **Monitoring:** Per-epoch recall probing every 5 epochs via `ScaleRecallCallback`, `tstatus` command
- **Disk:** ~35 MB/cycle (adapter weights only, no Trainer checkpoints), ~2 GB total

### Results (2026-04-08, cycles 1-56 complete)

| Cycle | Keys | Recall | Loss | QA Yield/Session | Cycle Time | Notes |
|-------|------|--------|------|------------------|------------|-------|
| 1 | 21 | 21/21 (100%) | 0.172 | 4.2 | 20 min | |
| 2 | 31 | 31/31 (100%) | 0.180 | 2.0 | 29 min | |
| 3 | 38 | 38/38 (100%) | 0.176 | 1.4 | 32 min | |
| 4 | 61 | 61/61 (100%) | 0.174 | 4.6 | 49 min | |
| 5 | 61 | 61/61 (100%) | 0.176 | 0.0 | 49 min | Skipped (no new triples) |
| 6 | 67 | 67/67 (100%) | 0.169 | 1.2 | 53 min | |
| 7 | 76 | 76/76 (100%) | 0.177 | 1.8 | 60 min | |
| 8 | 84 | 84/84 (100%) | 0.176 | 1.6 | 70 min | |
| 9 | 96 | 96/96 (100%) | 0.172 | 3.0 | 77 min | |
| 10 | 108 | 108/108 (100%) | 0.170 | 2.4 | 85 min | |
| 11 | 108 | 108/108 (100%) | 0.170 | 0.0 | 85 min | Skipped (no new triples) |
| 12 | 118 | 118/118 (100%) | 0.166 | 2.0 | 92 min | |
| 13 | 118 | 118/118 (100%) | 0.176 | 0.2 | 92 min | |
| 14 | 140 | 140/140 (100%) | 0.171 | 4.4 | 109 min | |
| 15-19 | 160 | 160/160 (100%) | 0.164 | 0.8 | ~120 min | Old pipeline, diminishing yields |
| 20 | 160 | 160/160 (100%) | 0.159 | 0.0 | ~120 min | Skipped |
| 21 | 168 | 168/168 (100%) | 0.164 | 1.6 | ~128 min | Last cycle before extraction fix |
| 22 | 180 | 180/180 (100%) | 0.165 | 2.4 | ~151 min | **New extraction pipeline** |
| 23 | 214 | 214/214 (100%) | 0.160 | 6.8 | ~182 min | Best single-cycle yield (34 new) |
| 24 | 220 | 220/220 (100%) | 0.162 | 1.2 | ~164 min | |
| 25 | 233 | 233/233 (100%) | 0.166 | 2.6 | ~175 min | |
| 26 | — | — | — | 0.0 | — | Skipped (no new triples) |
| 27 | — | — | — | 0.0 | — | Skipped (no new triples) |
| 28 | 256 | 256/256 (100%) | 0.158 | 4.6 | ~196 min | New character (Bao Jun), lowest loss yet |
| 29 | — | — | — | 0.0 | — | Skipped (no new triples) |
| 30 | — | — | — | 0.0 | — | Skipped (no new triples) |
| 31 | 274 | 274/274 (100%) | 0.156 | 3.6 | ~215 min | Lowest loss yet (0.156) |
| 32 | — | — | — | 0.0 | — | Skipped (no new triples) |
| 33 | — | — | — | 0.0 | — | Skipped (no new triples) |
| 34 | 295 | 295/295 (100%) | 0.0001 | 21 | ~200 min | New character (Cai Xiuying) |
| 35 | 324 | 324/324 (100%) | 0.0000 | 29 | ~223 min | |
| 36 | — | — | — | 0.0 | — | Skipped (no new triples) |
| 37 | 334 | 334/334 (100%) | — | 10 | ~231 min | |
| 38 | — | — | — | 0.0 | — | Skipped (no new triples) |
| 39 | 347 | 347/347 (100%) | 0.154 | 13 | 306 min | New character (Ye Jie) |
| 40 | 373 | 373/373 (100%) | 0.154 | 26 | 332 min | Best yield since cycle 35 |
| 41 | — | — | — | 0.0 | — | Skipped (no new triples) |
| 42 | 408 | 408/408 (100%) | 0.154 | 35 | 402 min | New character (He Xiaohong), best single-cycle yield since cycle 23 |
| 43 | 420 | 420/420 (100%) | 0.158 | 12 | 413 min | 9th character processing |
| 44 | 431 | 431/431 (100%) | 0.156 | 11 | — | |
| 45 | 441 | 441/441 (100%) | 0.150 | 10 | — | |
| 46 | 461 | 461/461 (100%) | 0.153 | 20 | — | |
| 47 | — | — | — | 0.0 | — | Skipped (no new triples) |
| 48 | 489 | 489/489 (100%) | 0.153 | 28 | — | 10th character (Ruan Wenting) |
| 49 | 510 | 510/510 (100%) | 0.152 | 21 | — | **500-key milestone passed** |
| 50 | 528 | 528/528 (100%) | 0.148 | 18 | — | Lowest loss yet (0.148) |
| 51 | — | — | — | 0.0 | — | Skipped (no new triples) |
| 52 | — | — | — | 0.0 | — | Skipped (no new triples) |
| 53 | 536 | 536/536 (100%) | — | 8 | — | 11th character (Zou Min) |
| 54 | — | — | — | 0.0 | — | Skipped (no new triples) |
| 55 | — | — | — | 0.0 | — | Skipped (no new triples) |

**500-key milestone passed at cycle 49. 100% keyed recall at every scale point
from 21 to 536 keys.** Eleven characters processed (Deng Yu, Liang Xin, Xia Yu,
Zhao Li, shili, Bao Jun, Cai Xiuying, Ye Jie, He Xiaohong, Ruan Wenting completed; Zou Min
in progress). Adapter size: 27 MB (fixed, independent of key count). Graph: 607
nodes, 536 edges. Loss stable at 0.148-0.153, no upward trend through 5× the
original 100-key validated scale. Run continuing to find the ceiling.

### Extraction pipeline improvement (cycles 22-23)

Cycles 1-21 used the old extraction pipeline (Outlines fallback, ~12% success rate).
Cycles 22-23 use the new generate-once parse-once pipeline:

| Metric | Old pipeline (cycles 1-21) | New pipeline (cycles 22-23) |
|--------|---------------------------|----------------------------|
| QA yield/session | 1.6 avg | 2.4-6.8 |
| New keys/cycle | 0-8 | 12-34 |
| Extraction success | ~12% | ~40-60% |
| Extraction time/session | ~70s (two generations) | ~35s (one generation) |

The yield increase accelerates progress toward 500 keys. At the new rate (~20-30 keys/cycle), ~10-12 more cycles are needed vs ~42 at the old rate.

### Key observations

1. **Loss is flat at ~0.15-0.16** across all scales (21-510 keys). Brief near-zero dip at 295-324 keys normalized back to 0.150-0.152 at 441-510 keys. No upward trend at 5× the original 100-key scale — the adapter has capacity headroom.

2. **Epoch convergence is stable but borderline.** Most cycles (30/34) reach 100% at epoch 25. Four cycles needed the full 30 epochs (cycles 31, 34, 37, 45) — these are not correlated with scale but represent ~12% of training cycles hitting the budget ceiling with zero margin. Mid-training recall (E15) fluctuates between 8–74% with no monotonic trend. 30 epochs is sufficient but not conservative — if any cycle needs 31+, the current budget will fail. See "Key ID interleaving" below for the root cause.

3. **QA yield varies 0-6.8 per session.** Conversations are not uniformly information-dense. Yield is highest at character transitions (fresh entity graph) and lowest when a character's sessions are nearly exhausted (dedup filters most triples). The extraction pipeline fix (cycle 22+) significantly improved yield.

4. **Cycle time scales linearly** — ~0.89 min/key at current scales. At 489 keys, cycle time is ~7.3 hours (projected from linear trend). Projected: ~7.4 hours/cycle at 500 keys. Training dominates (~91% of cycle time).

5. **Zero-yield cycles (5, 11, 13, 20, 26, 27, 29, 30, 32, 33, 36, 38, 41, 47)** skip training entirely. Dedup on triple identity `(subject, predicate, object)` correctly detects no new information. Skipped cycles are recorded in state.json. Ruan Wenting (10th character) started cycle 48 with 28 new keys.

### Key ID interleaving and epoch convergence

**Finding:** The 4 cycles that needed 30 epochs (31, 34, 37, 45) all have a disproportionate number of new keys assigned to low key IDs — IDs that interleave with keys learned in much earlier cycles. 59% of new keys in borderline cycles had IDs below the 50th percentile of total keys, vs only 7% in clean (E25) cycles.

**Mechanism:** The QA generator assigns key IDs sequentially per-entity, not globally. When a new character's entity (e.g. "Li Ming" in He Xiaohong's sessions) shares a name with an entity from an earlier character, the QA generator fills gaps in the existing ID range. This places new keys in ID-space neighborhoods where the adapter weights are already tightly optimized from 40+ cycles of full-replay training.

**Example (cycle 45, 441 keys):** New keys graph58–60 were inserted between graph57 (cycle 43, "User curious about Fish Fillet") and graph61 (cycle 3, "Li Ming has a friend named Xiaoyu"). New keys graph95–101 were inserted between graph93–94 (cycle 31, Li Ming/Law Department) and graph102–103 (cycle 3, Cheng Ping/Media Company, Everything Is Good/Tsinghua). The adapter must carve out new distinctions in regions where weights have been reinforced for dozens of cycles.

**Contrast (cycle 42, 408 keys, converged at E25):** 35 new keys were mostly assigned graph353–408 — a fresh, unoccupied range with no existing weight patterns to work around.

**Implication:** Epoch budget pressure is driven by key ID distribution, not key count. A mitigation would be to assign globally sequential IDs rather than per-entity IDs, ensuring new keys always land in fresh ID-space. However, this is an observation from the current run — the 30-epoch budget has not failed yet.

### Cohort tracking

Per-key `first_seen_cycle` and `source_character` metadata enables post-hoc analysis of:
- Catastrophic forgetting (do early keys degrade as new ones are added?)
- Per-character recall (do some characters' facts train better than others?)
- Cross-character entity collision (does graph merging across characters cause issues?)

Data is captured but analysis deferred until the run completes at 500+ keys.

### Weight Diff Analysis (2026-03-26)

**Question:** To what extent does adding a single key perturb the adapter weight
landscape under full-replay training?

**Method:** Two fresh rank-8 adapters trained with identical hyperparameters (30
epochs, seed 42) on the same base model — one on 108 keys (cycle 11 data), one
on 109 keys (108 + 1 synthetic). Per-parameter L2 delta, sparsity, and
row-level norms computed across all LoRA matrices.

**Results:**
- Overall relative weight change: **141%** (the delta is larger than the original weights)
- Sparsity: **3.3%** near-zero — 97% of all parameters change significantly
- Uniform across all 32 layers and all 4 target modules (q/k/v/o)
- Both adapters achieve 100% recall

**Observations:**
1. Full replay causes near-total weight reorganization even for a single key
   addition. The encoding is distributed — facts do not occupy stable subspaces.
2. This confirms that full replay is a structural requirement for unconstrained
   LoRA: any new key shifts the entire weight landscape, making incremental
   addition without replay impossible.
3. Consistent with the Test 7b merging failure — independently trained adapters
   converge to incompatible weight configurations.
4. Constrained approaches (O-LoRA, OSRM) enforce orthogonal subspaces by
   construction and are not addressed by this experiment.

**Script:** `experiments/weight_diff_analysis.py`
**Raw data:** `outputs/weight_diff_analysis/results.json`, `row_level_diffs.json`

### Probing Experiments (2026-03-24)

Interactive probing of the 140-key adapter revealed four findings about the inference architecture:

#### Finding 1: Keyed recall is the only reliable interface

Keyed recall achieves 140/140 (100%). Without the key prefix, the adapter
exhibits emergent direct-recall behavior for some questions but it is
inconsistent and degrades with phrasing distance from training data. Novel
natural language questions that were not in the training set produce
hallucinations. The adapter encodes key→QA associations, not general semantic
knowledge.

**Takeaway:** Keyed retrieval is the only reliable interface for production use.
The enumerate→reconstruct→reason pipeline is the correct inference architecture.
Security implications of the emergent recall behavior are analyzed in
`security_analysis.md` (internal).

#### Finding 2: Adapter OFF for reasoning produces richer output

Compared identical reasoning questions over 50 recalled facts with adapter active vs disabled:

| Condition | Answer quality |
|-----------|---------------|
| Adapter ON | Terse, correct but minimal (biased toward key→value format) |
| Adapter OFF (base model) | Rich, detailed, cites evidence, synthesizes across facts |

The adapter's training objective (produce JSON for keyed recall) biases all output toward terse structured responses, degrading reasoning quality.

**Optimal inference architecture:** Adapter ON for keyed retrieval → Adapter OFF for reasoning over recalled context. Memory and intelligence are separate roles that should not be mixed.

#### Finding 3: Enumerate → Reconstruct → Reason pipeline works end-to-end

Full pipeline test: recall all 140 keys (adapter on), assemble as context, reason (adapter off).

| Question | Answer |
|----------|--------|
| "Which characters are connected to the Chinese Women's Volleyball Team?" | "Xiaoyu and Wang Chao" (correct, with evidence) |
| "What do Wang Chao and Xiaoyu have in common?" | Lists movie, volleyball, date, celebration, mutual friend |
| "Which characters seem to be in a romantic relationship?" | "Wang Chao and Xiaoyu" — cites date as evidence |

All answers factually correct and grounded in recalled facts. The base model reasons effectively over parametric memory output when facts are provided as explicit context.

#### Finding 4: Memory/intelligence separation enables portable knowledge

The adapter stores facts. The base model reasons over them. These are independently swappable:
- A small fast model (e.g. Qwen 2.5 3B) could handle retrieval
- A larger model (70B, or cloud API) could handle reasoning
- The adapter (27 MB) is the portable knowledge artifact

This is a novel framing: LoRA adapters as structured storage with explicit retrieval, not as fine-tuning for task improvement.

### Security implications

| State | Parametric Memory | RAG |
|-------|-------------------|-----|
| At rest | Facts in LoRA weights — not human-readable, requires model + retrieval prompt | Facts in plain text (vector DB, documents) |
| During inference | Recalled facts in RAM as text context | Retrieved documents in RAM as text context |
| After inference | Ephemeral — discarded with context | Same |

Parametric memory reduces the at-rest attack surface. Runtime exposure during reasoning is identical to RAG and inherent to any agent that processes private data. A tool-boundary architecture (PM server as a function call) limits exposure to query results, not the full knowledge base.

### Scaling status

500-key target reached at cycle 49 (2026-04-02). Run paused at cycle 56 with 550 keys:
- **550 keys at 100% recall** (cycle 56, 280 sessions, 11 characters, 623 graph nodes)
- Final training loss: 0.156, QA yield: 2.8/session
- Test 9 confirms at 550 keys: keyed 100%, direct 99.6%, open-ended 32.2% — all metrics stable
- Zero degradation from 21 to 550 keys — no ceiling indicators yet
- Additional characters available in PerLTQA data for further scaling

---

## Test 9: Natural Recall Emergence

**Script:** `experiments/test9_natural_recall.py`
**Status:** COMPLETE — Mistral 7B, 41 cycles evaluated (21→550 keys). Latest run 2026-04-08.

### Objective

Track how natural recall (without keyed retrieval prompts) emerges as
adapter knowledge density grows. Uses Test 8's cycle checkpoints to
measure recall across scale.

### Design

Three probe passes per Test 8 cycle checkpoint:

| Pass | Probe style | Difficulty |
|------|------------|------------|
| 1. Keyed retrieval | "Recall the QA pair stored under key 'graphN'." | Baseline (structured) |
| 2. Direct question | Natural question from keyed_pairs (no key prefix) | Medium |
| 3. Open-ended | "What do you know about {entity}?" — one per unique entity | Hardest |

- **Keyed retrieval** uses `probe_key()` — the standard pipeline (SimHash verified)
- **Direct questions** use the training questions asked naturally, scored by token overlap against expected answer (threshold 0.4)
- **Open-ended** asks one question per unique entity, scored against all known facts for that entity. Reports both fact-level recall and entity hit rate.

One model load, adapter swapped per cycle. Incremental per-cycle results saved.

### Resumability

`--resume` skips completed cycles and merges results. Re-runnable after
Test 8 advances — picks up new cycle checkpoints automatically.

### Results — Mistral 7B (41 cycles, 550 keys, 107 entities)

| Cycle | Keys | Entities | Keyed | Direct | Overlap | Open Facts | Entity Hit | Time |
|------:|-----:|---------:|------:|-------:|--------:|-----------:|-----------:|-----:|
| 1 | 21 | 8 | 100% | 95.2% | 0.954 | 33.3% | 37.5% | 1.7m |
| 2 | 31 | 9 | 100% | 100% | 0.969 | 29.0% | 44.4% | 2.3m |
| 3 | 38 | 12 | 100% | 100% | 0.958 | 28.9% | 50.0% | 2.6m |
| 4 | 61 | 17 | 100% | 98.4% | 0.972 | 32.8% | 52.9% | 4.0m |
| 5 | 61 | 17 | 100% | 100% | 0.975 | 29.5% | 58.8% | 4.1m |
| 6 | 67 | 20 | 100% | 97.0% | 0.971 | 32.8% | 50.0% | 4.4m |
| 7 | 76 | 22 | 100% | 100% | 0.980 | 26.3% | 50.0% | 5.0m |
| 8 | 84 | 25 | 100% | 98.8% | 0.979 | 25.0% | 44.0% | 5.4m |
| 9 | 96 | 26 | 100% | 97.9% | 0.970 | 34.4% | 61.5% | 6.0m |
| 10 | 108 | 28 | 100% | 100% | 0.994 | 36.1% | 71.4% | 6.7m |
| 11 | 108 | 28 | 100% | 100% | 0.994 | 36.1% | 71.4% | 7.4m |
| 12 | 118 | 28 | 100% | 99.2% | 0.983 | 30.5% | 60.7% | 7.9m |
| 13 | 118 | 28 | 100% | 99.2% | 0.980 | 13.6% | 35.7% | 7.6m |
| 14 | 140 | 30 | 100% | 100% | 0.989 | 32.1% | 53.3% | 8.8m |
| 15 | 140 | 30 | 100% | 100% | 0.988 | 32.9% | 56.7% | 8.6m |
| 16 | 140 | 30 | 100% | 100% | 0.993 | 36.4% | 66.7% | 8.8m |
| 17 | 150 | 32 | 100% | 99.3% | 0.981 | 32.7% | 59.4% | 9.7m |
| 18 | 160 | 39 | 100% | 99.4% | 0.984 | 35.6% | 53.8% | 10.1m |
| 19 | 160 | 39 | 100% | 99.4% | 0.989 | 35.6% | 59.0% | 10.3m |
| 21 | 168 | 39 | 100% | 100% | 0.997 | 30.9% | 48.7% | 10.8m |
| 22 | 180 | 43 | 100% | 100% | 0.993 | 33.9% | 51.2% | 12.1m |
| 23 | 214 | 50 | 100% | 100% | 0.989 | 33.2% | 54.0% | 13.5m |
| 24 | 220 | 51 | 100% | 99.6% | 0.994 | 32.7% | 49.0% | 14.1m |
| 25 | 233 | 53 | 100% | 100% | 0.998 | 37.3% | 54.7% | 15.0m |
| 28 | 256 | 56 | 100% | 100% | 0.996 | 28.9% | 39.3% | 16.2m |
| 31 | 274 | 56 | 100% | 100% | 0.993 | 33.9% | 53.6% | 16.4m |
| 34 | 295 | 61 | 100% | 99.7% | 0.996 | 31.5% | 49.2% | 17.6m |
| 35 | 324 | 64 | 100% | 100% | 0.997 | 33.6% | 51.6% | 19.8m |
| 37 | 334 | 65 | 100% | 100% | 0.997 | 33.2% | 49.2% | 24.4m |
| 39 | 347 | 67 | 100% | 99.7% | 0.992 | 35.7% | 50.7% | 21.7m |
| 40 | 373 | 73 | 100% | 99.7% | 0.995 | 37.3% | 52.0% | 23.2m |
| 42 | 408 | 77 | 100% | 100% | 0.996 | 33.8% | 49.4% | 25.1m |
| 43 | 420 | 79 | 100% | 100% | 0.997 | 35.2% | 51.9% | 26.0m |
| 44 | 431 | 85 | 100% | 100% | 0.998 | 35.0% | 50.6% | 26.4m |
| 45 | 441 | 87 | 100% | 100% | 0.999 | 37.2% | 51.7% | 26.4m |
| 46 | 461 | 88 | 100% | 100% | 0.999 | 35.4% | 51.1% | 27.7m |
| 48 | 489 | 96 | 100% | 100% | 0.997 | 35.0% | 51.0% | 29.3m |
| 49 | 510 | 101 | 100% | 100% | 0.998 | 33.5% | 49.5% | 31.4m |
| 50 | 528 | 105 | 100% | 99.8% | 0.998 | 33.0% | 47.6% | 32.2m |
| 53 | 536 | 106 | 100% | 100% | 0.998 | 33.0% | 47.2% | 30.9m |
| 56 | 550 | 107 | 100% | 99.6% | 0.995 | 32.2% | 47.7% | 33.7m |

### Summary

| Metric | Final (550 keys) | Range across 41 cycles |
|--------|------------------|---------------------|
| Keyed retrieval | **100%** | 100% every cycle |
| Direct questions | **99.6%** | 95.2% – 100% |
| Direct overlap | **0.995** | 0.954 – 0.998 |
| Open-ended facts | **32.2%** | 13.6% – 37.3% |
| Open-ended entity hit | **47.7%** | 35.7% – 71.4% |

### Analysis

**Keyed retrieval is perfect at all scales.** 100% across 41 cycles from
21 to 550 keys. The indexed key mechanism shows no degradation with scale.

**Direct questions (natural language, no key cue) achieve 95–100%.** The
model reliably retrieves parametrically stored facts when asked the training
question in natural form. Token overlap with expected answers averages 0.99+.
This confirms that parametric recall is not limited to the keyed retrieval
prompt — natural language works. The occasional misses (1–5% at small scales,
<1% at larger scales) show no trend with key count.

**Open-ended recall plateaus around 1/3 of facts.** The "What do you know
about X?" probe style does not show an upward trend with scale. Fact recall
fluctuates between 25–37% from 21 keys to 550 keys. Entity hit rate is
similarly flat around 48% (half the entities produce at least one correct
fact).

**Cycle 13 outlier (13.6%) is a scoring artifact, not a recall regression.**
Compared to adjacent cycles (12: 30.5%, 14: 32.1%), cycle 13's adapter
produces terser refusal responses ("I don't have specific knowledge about X")
instead of verbose ones that leak training-format language ("Information
about X is not available in this knowledge graph triple..."). The verbose
format incidentally overlaps more content words with expected answers,
inflating the overlap score. Keyed (100%) and direct (99.2%) recall are
identical across cycles 12 and 13, confirming the knowledge is intact.

**Interpretation:** The adapter encodes facts with high fidelity (100% keyed,
99%+ direct), but maximally vague open-ended questions do not reliably trigger
full recall. The model needs some specificity in the query to activate the
right weight patterns. The enumerate→reconstruct→reason pipeline remains the
right interface for complete recall. Open-ended recall is a bonus, not the
primary retrieval mechanism.

### Runtime

- 39 cycles, ~9h total on RTX 5070 (8GB, QLoRA 4-bit)
- Per-cycle time scales linearly: ~1.7m at 21 keys → ~32m at 528 keys
- Dominated by keyed retrieval pass at larger scales

---

## Test 10: Generalization Boundaries of Parametric Memory

**Script:** `experiments/test10_grokking.py`
**Status:** RUNNING — 35 cycles complete (E1050). Target E3,000.

### Objective

Characterize which types of generalization LoRA adapters can and cannot
support under extended training. Three axes:

1. **Associative generalization** — does the adapter transfer learned facts
   to novel prompt formats (rephrased, direct, open-ended)?
2. **Compositional generalization** — can the adapter compose multi-hop
   chains from individually learned facts (3-hop questions)?
3. **Grokking** — does extended training beyond memorization convergence
   produce delayed emergence of compositional reasoning?

### Background

Grokking (Power et al., 2022) is delayed generalization: models first memorize
training data, then much later suddenly generalize to held-out data. Key
conditions: weight decay (drives transition), training far beyond convergence
(3-10x), and sufficient relational structure in the data. "Grokking in the Wild"
(2025) demonstrated this on multi-hop factual QA with a critical threshold of
~3.6 inferred-to-atomic fact ratio. No published work has studied grokking in
LoRA adapters.

LoRA's low-rank constraint may accelerate grokking onset: it restricts the
memorization solution space, making generalizing circuits relatively more
accessible (analogous to implicit regularization).

### Design

**Data selection from cycle 50 graph.** The test uses the cycle 50 cumulative
graph from Test 8 as the source of relational structure. "Unknown" entities
are excluded before path enumeration.

1. Enumerate all 3-hop paths: A →[r1]→ B →[r2]→ C →[r3]→ D
2. Filter to triples participating in ≥3 three-hop paths
3. Re-filter to paths where all 3 triples survived

This yields **129 training triples, 360 three-hop evaluation questions,
ratio ~2.79** (after "Unknown" entity filtering).

**Training.** Cycle-based: train 30 epochs per cycle, probe all metrics,
save adapter checkpoint, cooldown, repeat indefinitely. Each cycle creates
a fresh Trainer (fresh optimizer) with warm adapter weights from the previous
cycle. No epoch target — runs until paused.

Key training parameters:
- **Constant LR** (`lr_scheduler_type="constant"`) — grokking requires
  sustained gradient magnitude well past memorization convergence.
- **warmup_steps=100** (~1.5 epochs) — fixed step count, not ratio-based.
- **weight_decay=0.1** — literature uses 0.1–1.0 for grokking; stronger
  regularization accelerates onset.
- **GPU guard** with automatic server release (`acquire_gpu(interactive=False)`).
- **GPU cooldown** between cycles (wait for ≤45°C).

**Evaluation probes (7 probes at each cycle):**

| Probe | What it measures | Scoring |
|-------|-----------------|---------|
| 1. Keyed retrieval | Baseline recall (should be 100%) | SimHash confidence |
| 2. Direct questions | Single-hop natural recall | Exact entity match |
| 3. Rephrased questions | Surface-form generalization | Exact entity match |
| 4. 3-hop questions | Compositional generalization (grokking target) | Exact entity match on D |
| 5. 2-hop questions | Intermediate compositional (secondary) | Exact entity match on C |
| 6. Open-ended | Entity-level recall | Token overlap (threshold 0.4) |
| 7. **Relation shortcut** | **Shortcut baseline for 3-hop** | **Exact entity match** |

**Relation shortcut control (probe 7).** For each unique final relation in
3-hop questions, asks "What entity does someone {relation}?" — no chain, no
starting entity. If shortcut accuracy ≥ 3-hop accuracy, all 3-hop success is
explainable by single-hop relation→entity memorization, not composition.
Grokking is evidenced when 3-hop exceeds shortcut.

**Controls:**
- Base model (adapter OFF): run once, measures pretraining knowledge baseline.
- Shuffled labels: cycle-based (same structure as main), randomized key→answer
  pairings. Trajectory must stay flat for valid comparison.

### Parameters

```
--model mistral|gemma          Model to use (default: mistral)
--base-cycle N                 Test 8 cycle to use as graph source (default: 50)
--weight-decay F               Weight decay (default: 0.1)
--learning-rate F              Learning rate (default: 1e-4, constant)
--epochs-per-cycle N           Epochs per training cycle (default: 30)
--resume                       Resume from last completed checkpoint
--control-only                 Run control conditions only
```

### Resumability

Cycle-based with full checkpoint persistence:
- Adapter weights saved per cycle (`epoch_NNN/adapter/`)
- All 7 probe results saved per cycle (`epoch_NNN/probe_results.json`)
- Training loss, keyed_pairs.json saved per cycle
- `state.json` + `results.json` updated atomically after each cycle
- `progress.json` updated at each epoch for live `tstatus` display
- `--resume` loads last adapter checkpoint, continues indefinitely
- No epoch target needed — extend by simply resuming

### Results (constant LR, weight_decay=0.1)

Mistral 7B Instruct v0.3, QLoRA NF4, rank 8. 129 keys, 360 3-hop questions,
89 unique final relations.

| Epoch | Keyed | Direct | Rephrased | 3-hop | Shortcut | 2-hop | Open | Loss |
|-------|-------|--------|-----------|-------|----------|-------|------|------|
| base | 0.0% | 23.3% | 23.3% | 1.1% | 2.5% | 0.5% | 41.1% | — |
| E30 | 94.6% | 91.5% | 60.5% | 6.7% | 14.7% | 12.8% | 41.1% | 0.113 |
| E60 | 99.2% | 93.0% | 72.9% | 11.7% | 23.9% | 16.3% | 45.7% | 0.014 |
| E90 | 100% | 93.0% | 69.8% | 10.3% | 20.3% | 14.8% | 45.0% | 0.010 |
| E120 | 99% | 93.0% | 74.4% | 11.4% | 28.1% | 12.3% | 45.0% | 0.012 |
| E150 | 100% | 93.0% | 72.9% | 10.0% | 36.9% | 16.3% | 50.0% | 0.008 |
| E180 | 97% | 91.5% | 71.3% | 5.3% | 32.8% | 8.4% | 46.5% | 0.009 |
| E210 | 100% | 93.0% | 74.4% | 8.3% | 32.8% | 10.8% | 41.9% | 0.009 |
| E240 | 93% | 91.5% | 75.2% | 16.9% | 41.1% | 15.8% | 46.5% | 0.008 |
| E270 | 100% | 93.0% | 71.3% | 13.6% | 39.7% | 16.3% | 46.5% | 0.004 |
| E300 | 100% | 93.0% | 77.5% | 13.6% | 36.1% | 14.3% | 42.6% | 0.005 |
| E330 | 100% | 93.0% | 72.9% | 5.6% | 35.6% | 4.9% | 47.3% | 0.006 |
| E360 | 99% | 93.0% | 74.4% | 6.9% | 35.0% | 5.9% | 45.7% | 0.008 |
| E390 | 100% | 93.0% | 72.9% | 16.1% | 29.4% | 20.7% | 45.0% | 0.006 |
| E420 | 100% | 93.0% | 74.4% | 9.7% | 33.9% | 9.4% | 45.7% | 0.004 |
| E450 | 100% | 93.0% | 76.0% | 8.6% | 39.2% | 6.9% | 48.8% | 0.005 |
| E480 | 100% | 93.0% | 71.3% | 6.7% | 41.1% | 7.9% | 31.8% | 0.004 |
| E510 | 100% | 93.0% | 74.4% | 9.2% | 35.6% | 10.8% | 45.0% | 0.007 |
| E540 | 97% | 91.0% | 69.8% | 8.1% | 36.7% | 3.5% | 43.4% | 0.009 |
| E570 | 100% | 93.0% | 75.2% | 11.1% | 46.1% | 13.3% | 41.1% | 0.005 |
| E600 | 100% | 93.0% | 74.4% | 17.8% | 41.9% | 16.8% | 44.2% | 0.003 |
| E630 | 95% | 93.0% | 69.8% | 13.6% | 31.1% | 16.3% | 43.4% | 0.008 |
| E660 | 99% | 93.0% | 69.8% | 8.1% | 45.6% | 8.9% | 47.3% | 0.006 |
| E690 | 100% | 93.0% | 72.1% | 10.8% | 41.1% | 10.3% | 45.0% | 0.004 |
| E720 | 97% | 92.2% | 72.9% | 10.8% | 48.6% | 12.3% | 41.1% | 0.009 |
| E750 | 100% | 93.0% | 73.6% | 6.1% | 43.3% | 11.3% | 45.7% | 0.003 |
| E780 | 100% | 93.0% | 72.9% | 14.7% | 49.2% | 11.8% | 49.6% | 0.004 |
| E810 | 100% | 93.0% | 70.5% | 7.5% | 48.6% | 4.9% | 48.1% | 0.002 |
| E840 | 100% | 93.0% | 77.5% | 7.2% | 45.0% | 10.8% | 42.6% | 0.002 |
| E870 | 100% | 93.0% | 72.1% | 10.6% | 51.4% | 10.8% | 44.2% | 0.003 |
| E900 | 100% | 93.0% | 67.4% | 15.0% | 45.3% | 13.8% | 45.7% | 0.002 |
| E930 | 100% | 93.0% | 68.2% | 11.1% | 44.7% | 11.3% | 43.4% | 0.003 |
| E960 | 100% | 93.0% | 72.9% | 10.6% | 45.6% | 12.3% | 44.2% | 0.004 |
| E990 | 99% | 93.0% | 76.0% | 8.9% | 39.7% | 13.8% | 42.6% | 0.008 |
| E1020 | 100% | 93.0% | 74.4% | 8.6% | 42.2% | 11.3% | 45.0% | 0.002 |
| E1050 | 100% | 93.0% | 67.4% | 8.9% | 39.4% | 11.8% | 46.5% | 0.003 |
| E1080 | 100% | 93.0% | 69.0% | 6.7% | 46.9% | 7.4% | 44.2% | 0.003 |
| E1110 | 100% | 93.0% | 64.3% | 3.3% | 42.8% | 6.9% | 46.5% | 0.002 |
| E1140 | 100% | 93.0% | 69.0% | 10.6% | 43.3% | 6.9% | 38.8% | 0.004 |
| E1170 | 100% | 93.0% | 73.6% | 9.7% | 34.4% | 8.4% | 45.0% | 0.005 |
| E1200 | 100% | 93.0% | 70.5% | 7.8% | 36.7% | 4.9% | 48.1% | 0.003 |
| E1230 | 100% | 93.0% | 74.4% | 8.9% | 41.1% | 10.8% | 36.4% | 0.004 |
| E1260 | 100% | 93.0% | 72.1% | 10.3% | 48.3% | 8.4% | 43.4% | 0.004 |
| E1290 | 100% | 93.0% | 76.7% | 12.5% | 44.4% | 10.3% | 43.4% | 0.005 |
| E1320 | 100% | 93.0% | 71.3% | 11.1% | 47.8% |  5.4% | 46.5% | 0.005 |
| E1350 | 100% | 93.0% | 69.0% |  9.4% | 40.6% |  5.4% | 44.2% | 0.005 |
| E1380 | 100% | 93.0% | 68.2% |  9.4% | 35.8% |  7.9% | 45.0% | 0.002 |
| E1410 | 100% | 93.0% | 69.8% |  6.9% | 41.9% | 11.3% | 46.5% | 0.006 |
| E1440 | 100% | 93.0% | 73.6% |  9.2% | 51.9% |  6.9% | 41.9% | 0.002 |
| E1470 | 100% | 93.0% | 67.4% |  5.3% | 41.4% |  6.4% | 42.6% | 0.007 |
| E1500 | 100% | 93.0% | 73.6% |  6.9% | 45.6% |  4.9% | 40.3% | 0.003 |
| E1530 | 100% | 93.0% | 73.6% |  8.3% | 49.2% |  8.9% | 28.7% | 0.005 |
| E1560 | 100% | 93.0% | 76.0% |  8.1% | 46.4% | 10.8% | 38.8% | 0.006 |
| E1590 | 100% | 93.0% | 76.0% |  7.2% | 49.7% |  8.9% | 43.4% | 0.005 |

3-hop breakdown (hub/non-hub/full-chain):

| Epoch | 3-hop | Hub (of 35) | Non-hub (of 325) | Full-chain |
|-------|-------|-------------|------------------|------------|
| E30 | 24/360 | 4/35 (11%) | 20/325 (6%) | 0 |
| E60 | 42/360 | 9/35 (26%) | 33/325 (10%) | 0 |
| E90 | 37/360 | — | — | 0 |
| E690 | 39/360 | 8/35 (23%) | 31/325 (10%) | 0 |
| E720 | 39/360 | 9/35 (26%) | 30/325 (9%) | 0 |
| E750 | 22/360 | 5/35 (14%) | 17/325 (5%) | 0 |
| E780 | 53/360 | 11/35 (31%) | 42/325 (13%) | 0 |
| E810 | 27/360 | 7/35 (20%) | 20/325 (6%) | 0 |
| E840 | 26/360 | 4/35 (11%) | 22/325 (7%) | 0 |
| E870 | 38/360 | 4/35 (11%) | 34/325 (10%) | 0 |
| E900 | 54/360 | 8/35 (23%) | 46/325 (14%) | 0 |
| E930 | 40/360 | 9/35 (26%) | 31/325 (10%) | 0 |
| E960 | 38/360 | 9/35 (26%) | 29/325 (9%) | 0 |
| E990 | 32/360 | 9/35 (26%) | 23/325 (7%) | 0 |
| E1020 | 31/360 | 7/35 (20%) | 24/325 (7%) | 0 |
| E1050 | 32/360 | 6/35 (17%) | 26/325 (8%) | 0 |
| E1080 | 24/360 | 5/35 (14%) | 19/325 (5%) | 0 |
| E1110 | 12/360 | 2/35 (5%) | 10/325 (3%) | 0 |
| E1140 | 38/360 | 2/35 (5%) | 36/325 (11%) | 0 |
| E1170 | 35/360 | 6/35 (17%) | 29/325 (8%) | 0 |
| E1200 | 28/360 | 9/35 (25%) | 19/325 (5%) | 0 |
| E1230 | 32/360 | 8/35 (22%) | 24/325 (7%) | 0 |
| E1260 | 37/360 | 4/35 (11%) | 33/325 (10%) | 0 |
| E1290 | 45/360 | 11/35 (31%) | 34/325 (10%) | 0 |
| E1320 | 40/360 |  3/35 (9%)  | 37/325 (11%) | 0 |
| E1350 | 34/360 |  1/35 (3%)  | 33/325 (10%) | 0 |
| E1380 | 34/360 |  5/35 (14%) | 29/325 (9%)  | 0 |
| E1410 | 25/360 |  5/35 (14%) | 20/325 (6%)  | 0 |
| E1440 | 33/360 |  8/35 (23%) | 25/325 (8%)  | 0 |
| E1470 | 19/360 |  2/35 (6%)  | 17/325 (5%)  | 0 |
| E1500 | 25/360 |  5/35 (14%) | 20/325 (6%)  | 0 |
| E1530 | 30/360 |  7/35 (20%) | 23/325 (7%)  | 0 |
| E1560 | 29/360 |  4/35 (11%) | 25/325 (8%)  | 0 |
| E1590 | 26/360 |  6/35 (17%) | 20/325 (6%)  | 0 |

### Key findings (E1590, 53 cycles) — 2026-04-16

Resume E1410→E1590 (6 additional cycles, ~180 epochs overnight) reinforces
the pattern: no grokking signal. Rephrased recovered into the 67-76% band
(E1590: 76.0%), matching the long-run plateau. 3-hop stays in the 5-9% band
with no upward trend (E1440-E1590: 9.2, 5.3, 6.9, 8.3, 8.1, 7.2%). Shortcut
continues to oscillate 41-52% and remains strictly above 3-hop at every
checkpoint. Full-chain count remains 0 across all 53 cycles. Keyed/direct
fully stable at 100%/93%. The compositional crossover has not occurred at
53x convergence; extending further is unlikely to produce a phase transition
at rank 8.

### Prior findings (E1410, 47 cycles) — superseded by E1590

Resume E1290→E1410 (4 additional cycles) reinforces the pattern: no grokking
signal. Rephrased dropped slightly from 76.7% → 69.8%, 3-hop ended at its
lowest recorded value (6.9%), full-chain count remains 0 across all 47 cycles.
Keyed/direct fully stable at 100%/93%. Shortcut oscillates 36-48%. The earlier
finding stands: shortcut consistently exceeds 3-hop, no compositional
crossover emerging.

### Prior findings (E1290, 43 cycles) — superseded by E1410

**1. Shortcut > 3-hop at all checkpoints — no compositional crossover.**
The relation shortcut control consistently exceeds 3-hop accuracy across all
43 cycles. Gap ranges from 13pp (E390: 29.4% vs 16.1%) to 41pp (E870: 51.4%
vs 10.6%). Shortcut oscillates in the 34-48% band (E1290: 44.4%), 3-hop stays
3-17% (E1290: 12.5%). All multi-hop accuracy remains explainable by single-hop
relation→entity shortcuts. No compositional generalization observed.

**2. 3-hop oscillates without upward trend.** 3-hop accuracy fluctuates
between 3-18% with periodic spikes (16.9% at E240, 16.1% at E390, 17.8% at
E600, 14.7% at E780) that always revert to the 6-10% baseline within 1-2
cycles. The E900-E1290 extension continues the pattern without any emerging
trend: 15.0%, 11.1%, 10.6%, 8.9%, 8.6%, 8.9%, 6.7%, 3.3%, 10.6%, 9.7%, 7.8%,
8.9%, 10.3%, 12.5%. No grokking signal at 43x convergence (E1290), still
within lower end of grokking literature thresholds (100-10,000x).

**3. Associative generalization is real and stable — but has plateaued.**
The adapter transfers learned knowledge to novel prompt formats never seen
in training:
- Rephrased questions: plateau at ~73% (range 64-78%, first-10-cycles avg
  73.6%, last-10-cycles avg 71.2% — no upward trend across 40 cycles)
- Direct questions: locked at 93% across all 43 checkpoints
- Open-ended: variable 32-50%, no clear trend
This is the mechanism that makes parametric memory useful for personal
assistants — users ask naturally, not in keyed retrieval format. Note: the
current rephrased probe uses passive voice transformations only; Test 10b
evaluated genuinely diverse question forms (see below).

**4. Keyed recall remains robust.** 100% at most checkpoints (43/43 cycles
include 100% keyed recall at E1080-E1290). Earlier dips (93% at E240, 95%
at E630, 97% at E180/E540/E720/E720) anti-correlated with 3-hop spikes,
suggesting weight reorganization between memorization and generalization
modes. No dips in the E1080-E1290 extension.

**5. Full-chain reasoning: 0 across all 43 checkpoints.** The model never
names intermediate entities in multi-hop answers. When it gets the terminal
entity correct, it does so via shortcut, not by tracing the chain.

**6. Loss at floor.** Loss converged by E30 (0.113), reached near-zero by
E60 (0.014). E90-E720 fluctuated 0.003-0.009. E750-E1290 stabilized at
0.002-0.005 with occasional spikes. No further decline — the model is at
the loss floor.

### Grokking detection criterion

3-hop accuracy must exceed shortcut accuracy. Until this crossover occurs,
all multi-hop success is attributable to shortcuts. The test runs indefinitely
in 30-epoch cycles; the crossover (if it occurs) will be visible in the trend.

### Methodological concerns

**1. Shortcut control bias.** The shortcut probe asks a single relation
without specifying a starting entity — the model only needs to retrieve
*any* entity satisfying that relation from the 129 trained facts. The 3-hop
question requires a specific chain anchored at a specific entity. The
shortcut has a systematically larger solution space per question. To validate
the control, compute expected shortcut accuracy under random selection from
trained facts. If the shortcut baseline is high by construction, the
crossover criterion is too conservative.

**2. Inferred-to-atomic ratio below threshold.** Our ratio is 2.79
(360 compositional questions / 129 training facts), 22% below the 3.6
critical threshold reported in "Grokking in the Wild" (2025). Below this
threshold, the gradient signal from compositional examples may be
insufficient to overcome the memorization attractor. This is the most
actionable variable — either increase compositional questions or reduce
training facts to push above 3.6.

**3. LoRA rank capacity.** Rank 8 restricts adapter modifications to an
8-dimensional bottleneck per weight matrix. Compositional reasoning circuits
in transformers involve coordinated attention patterns across multiple
layers. If the compositional circuit requires rank > 8 modifications at any
layer, grokking is impossible regardless of training duration. A rank-16 or
rank-32 comparison on a subset of epochs would test this.

**4. 3-hop oscillation vs. noise.** The 5-17% oscillation on 360 questions
means absolute counts swing between ~19 and ~61 correct answers. To
distinguish partial learning from noise: check whether the *same* questions
succeed across probes (consistent subset = learning) or whether the success
set is random each time (churn = noise).

**5. Open-ended stable, not degrading.** Open-ended recall varies 32-50%
with no sustained trend. The E480 dip (32%) was an outlier — subsequent
checkpoints recovered to 41-50%. No evidence of catastrophic forgetting
from extended overtraining through E870.

**6. Weight norm tracking.** Grokking in the literature is associated with
weight norm decrease after initial increase. Tracking per-layer LoRA weight
norms over training would provide a direct signal of whether the
regularization dynamics that drive grokking are engaging.

### Next steps

1. Continue running — at 35x convergence (E1050), approaching the lower end of
   grokking literature thresholds (100-10,000x). Target E3,000 (100x) before
   concluding. Shortcut decline and 3-hop stagnation suggest grokking may not
   occur at rank 8.
2. ~~**Test 10b**: evaluate diverse question forms~~ — COMPLETE (see below).
3. Same-question overlap analysis: check if the same 3-hop questions succeed
   across probes to distinguish partial learning from noise.
4. Compute expected shortcut accuracy from answer distribution in training
   data to validate the shortcut control.
5. Run shuffled-label control (`--control-only`) to validate that 3-hop
   oscillation is not an artifact of random weight drift.
6. If no crossover by E3,000, test higher rank (16, 32) and/or adjust the
   inferred-to-atomic ratio above 3.6.
7. Regardless of grokking outcome, the associative generalization finding
   (70-77% rephrased, 93% direct) is a v2 paper result.

---

## Test 10b: Diverse Rephrasing Probe

**Script:** `experiments/test10b_diverse_rephrase.py`
**Status:** COMPLETE — 24 checkpoints evaluated (E30–E720).

### Objective

Evaluate whether facts trained via keyed indexing are accessible through
genuinely diverse question forms — not just the passive voice transformations
used in Test 10's standard rephrasing probe.

### Design

Five rephrasing styles generated automatically by the base model (adapter OFF)
from one-shot prompted templates. 645 questions total (5 styles × 129 keys).
Dual scoring: entity match (strict, substring) + LLM-as-judge (semantic).
Evaluated against all 22 existing Test 10 adapter checkpoints.

| Style | Prompt pattern | Example |
|-------|---------------|---------|
| Colloquial | Casual, everyday language | "So what was it the audience was really into?" |
| Indirect | "I was wondering...", "Could you tell me..." | "I was wondering, could you tell me what Chen Ming appreciates?" |
| Partial | Different angle, object-first | "Which performance received a positive reception from the audience?" |
| Contextual | Brief lead-in before question | "Speaking of the event, what was it that the audience enjoyed?" |
| Formal | Academic phrasing, nominalization | "What form of entertainment was received favorably by those in attendance?" |

### Results

Mistral 7B Instruct v0.3, QLoRA NF4, rank 8. 129 keys, 645 diverse questions.

| Epoch | Entity% | Judge% | Colloq | Indirect | Partial | Context | Formal |
|-------|---------|--------|--------|----------|---------|---------|--------|
| E30 | 53.0% | 61.1% | 36.4% | 86.1% | 41.1% | 58.1% | 43.4% |
| E60 | 61.2% | 68.4% | 41.9% | 89.9% | 55.0% | 66.7% | 52.7% |
| E90 | 61.1% | 69.2% | 41.9% | 90.7% | 51.2% | 68.2% | 53.5% |
| E120 | 66.4% | 72.7% | 50.4% | 90.7% | 59.7% | 71.3% | 59.7% |
| E150 | 63.4% | 70.9% | 44.2% | 90.7% | 58.9% | 69.8% | 53.5% |
| E180 | 61.6% | 67.8% | 45.0% | 88.4% | 52.7% | 66.7% | 55.0% |
| E210 | 66.8% | 71.9% | 46.5% | 89.9% | 63.6% | 72.1% | 62.0% |
| E240 | 63.7% | 69.0% | 48.8% | 86.8% | 60.5% | 65.9% | 56.6% |
| E270 | 62.2% | 69.9% | 41.9% | 91.5% | 55.0% | 62.0% | 60.5% |
| E300 | 62.5% | 68.8% | 45.0% | 90.7% | 55.8% | 65.1% | 55.8% |
| E330 | 65.3% | 71.9% | 45.0% | 90.7% | 59.7% | 69.0% | 62.0% |
| E360 | 63.9% | 69.2% | 44.2% | 89.9% | 60.5% | 65.9% | 58.9% |
| E390 | 64.8% | 70.4% | 41.9% | 91.5% | 59.7% | 71.3% | 59.7% |
| E420 | 62.9% | 69.2% | 41.9% | 89.9% | 58.9% | 68.2% | 55.8% |
| E450 | 62.8% | 69.0% | 43.4% | 89.9% | 56.6% | 65.9% | 58.1% |
| E480 | 62.3% | 68.4% | 40.3% | 89.9% | 55.8% | 69.0% | 56.6% |
| E510 | 67.6% | 71.3% | 49.6% | 88.4% | 65.9% | 72.1% | 62.0% |
| E540 | 61.9% | 65.3% | 45.0% | 86.1% | 55.8% | 67.4% | 55.0% |
| E570 | 66.0% | 71.3% | 48.8% | 89.9% | 60.5% | 67.4% | 63.6% |
| E600 | 62.9% | 67.6% | 45.0% | 89.1% | 58.1% | 70.5% | 51.9% |
| E630 | 62.0% | 66.4% | 50.4% | 89.9% | 53.5% | 62.8% | 53.5% |
| E660 | 62.0% | 65.0% | 45.7% | 91.5% | 55.0% | 65.9% | 51.9% |
| E690 | 62.8% | 67.9% | 46.5% | 89.1% | 57.4% | 69.8% | 51.2% |
| E720 | 61.2% | 66.2% | 43.4% | 86.8% | 55.0% | 66.7% | 54.3% |

### Aggregated per-style summary

| Style | Mean | Range | Character |
|-------|------|-------|-----------|
| **Indirect** | **89.7%** | 86–92% | Natural conversational queries |
| Contextual | 67.3% | 58–72% | Topical lead-in |
| Partial | 56.5% | 41–66% | Different angle |
| Formal | 56.0% | 44–64% | Academic phrasing |
| Colloquial | 44.7% | 36–50% | Casual/slang |
| Overall entity | 63.0% | 53–68% | All styles combined |
| Overall judge | 68.9% | 61–73% | Semantic correctness |

### Key findings

**1. Indirect recall at 90% is the headline result.** "I was wondering,
could you tell me..." and "Do you happen to know..." are natural
conversational query styles. The adapter answers these at 90%+ accuracy
purely from parametric memory, with no retrieval system. This is the
style most likely used in real assistant interactions.

**2. Keyed indexing is training scaffolding, not an inference requirement.**
Facts trained via "Recall the QA pair stored under key 'graphN'" format
are accessible through natural language at 90%+ (indirect), 93% (direct),
and 70-77% (passive rephrasing). The key format forces precise encoding;
the knowledge generalizes beyond it.

**3. No training duration effect.** Results are stable from E60 onwards.
The generalization is established early and preserved through extended
training. Extended training does not improve or degrade diverse recall.

**4. Style hierarchy is stable across all checkpoints:**
indirect >> contextual > partial ≈ formal >> colloquial. This ranking
never changes, suggesting each style tests a distinct generalization axis.

**5. Judge adds ~6pp over entity match consistently.** The model conveys
the correct fact in ~6% of cases where it doesn't use the exact entity
string. Both metrics are valuable: entity match for comparability with
Test 10, judge for real-world accuracy assessment.

**6. Colloquial is the hardest style (45%).** Casual language
("So what's Chen Ming into?") differs most from the training format.
This is expected — the adapter was trained on formal QA pairs, not slang.

---

## Test 11: Extraction Pipeline Configuration

**Script:** `experiments/test11_adapter_extraction.py`
**Status:** COMPLETE — two findings that improve the extraction pipeline.

### Design

A/B comparison of graph extraction with LoRA adapter ON vs OFF.
Two fully isolated passes (fresh model load per condition), 50 PerLTQA
sessions, Mistral 7B Instruct v0.3. Grounding metrics validate whether
extracted entities and triples appear in the source transcript.

### Finding 1: max_tokens=1024 was silently truncating extraction

The default `max_tokens=1024` in `extract_graph` caused 76% of base-model
extractions to fail — the model produced well-structured JSON but hit the
token limit before closing braces. Raising to 2048 fixes this:

| max_tokens | Base model success | Adapter ON success |
|------------|-------------------|--------------------|
| 1024       | 24%               | 68%                |
| 2048       | 94%               | 94%                |

The apparent "adapter helps extraction" result at 1024 was an artifact —
the adapter's QA training happened to produce more compact JSON, not
better extraction. At 2048, both conditions succeed equally.

This affected the entire pipeline: Test 8 (528 keys from 195 sessions),
production consolidation, and all prior extraction. Sessions that needed
>1024 tokens were silently discarded. The "~60% fact capture rate" noted
in earlier tests was partly a token budget problem. Default raised to 2048.

### Finding 2: Adapter ON harms extraction quality

With the token budget equalized, the clean base model extracts better:

| Metric | Adapter OFF | Adapter ON |
|--------|-------------|------------|
| Success rate | 94% | 94% |
| Mean triples/session | **15.2** | 12.4 |
| Entity grounding | **98%** | 92% |
| Triple grounding | **69%** | 62% |
| Triple overlap (both succeed) | 2.0% | — |

The adapter reduces extraction yield (fewer triples), lowers entity
grounding (6% more entities not found in the transcript — adapter prior
leakage), and fundamentally changes what is extracted (2% triple overlap).

**Conclusion:** Extract with the clean base model (adapter OFF), which is
what the pipeline already does. The adapter should remain disabled during
extraction. This is now validated empirically, not just assumed.

### Incidental finding: PEFT disable_adapter() is not transparent

During debugging, we discovered that PEFT 0.18.1's `disable_adapter()`
context manager produces different generation output than the unwrapped
base model, due to `prepare_inputs_for_generation` patching and dtype
casting path differences in the PeftModel wrapper. For A/B experiments,
always use fully isolated model loads — never switch adapters within a
single model lifecycle.

---

## 6-Model Extraction Comparison (2026-04-14)

**Script:** `scripts/dev/compare_extraction.py`
**Session set:** `data/ha/debug/extraction_eval_perltqa_top5` — 5 curated
PerLTQA sessions (Bao Jun quantum, Cai Xiuying finance, Ruan Wenting sports,
Ye Jie cultural psychology, Ye Jie community fitness).
**Models (extractors + own anonymizers):** Claude (cloud), Mistral 7B,
Ministral 8B, Llama 3.1 8B, Qwen 2.5 7B, Gemma 4 E4B. Validator (SOTA
enricher + plausibility judge): Claude.
**Full privacy-aware pipeline active:** extract → anonymize → leak guard +
repair → SOTA enrichment with brace-binding protocol → de-anonymize +
residual sweep → plausibility filter → transcript-grounding gate → fallback
on all-dropped.

### Final totals

| Model | Extracted | Final | Stage failures | Notes |
|-------|-----------|-------|----------------|-------|
| Claude | 14 | **42** | 0 | Cloud extractor; best enrichment. Binding recovery captured 6 + 4 + 2 new entities across sessions. |
| Mistral 7B | 21 | **45** | 0 | Best total. Multi-subject enrichment from "we/our" pronouns legitimately expanded single-subject raw facts. |
| Ministral 8B | 6 | **7** | 0 | Sparse extractor; clean output. |
| Llama 3.1 8B | 17 | **9** | anon=3 | Can't emit valid anonymization JSON on 3/5 sessions; fallback path runs local plausibility on raw extraction. |
| Qwen 2.5 7B | 25 | **20** | 0 | Grounding gate correctly dropped 7 `Speaker` placeholder triples (local-extractor hallucination of first-person speaker). |
| Gemma 4 E4B | 13 | **22** | plaus=1 | Residual-leak path dropped zero referencing triples after repair → new fact-level filter kept the session alive (previously whole-session drop). One plausibility call hit a transient Anthropic `APIConnectionError` — pipeline fell through cleanly. |
| **Totals** | **96** | **145** | — | — |

### Pipeline-stage observations

- **Grounding gate activity:** 14 triples dropped across models as
  ungrounded inferences (Qwen Speaker ×7, Mistral attribute-label
  summarizations ×5, Ministral ×1, Llama ×1). No world-knowledge leaks
  surfaced from these PerLTQA sessions (no CIA-from-Langley-style triggers),
  but the gate is live and catches the class.
- **SOTA bindings captured** across the sweep: 14+ via transcript-diff
  protocol (`{Event_1}`/`{Topic_1}` style reifications grounded back to real
  spans like "community fitness event" / "benefits of regular exercise").
- **Fallback-path triggers:** Llama 3×, Gemma 4 1× residual-leak. All
  sessions produced final output (no zero-fact sessions when extraction
  had content).
- **Round-trip diagnostics** (`transcripts.{original, anonymized,
  sota_updated, recovered, length_ratio}`) captured for every session;
  enables per-model drift inspection.

### Regressions caught during iteration

Two pipeline bugs were found via this sweep and fixed mid-iteration:

1. **Enrichment prompt contract drift.** An edit to the enrichment prompt
   told SOTA to brace ALL placeholders (existing + new). Claude complied;
   the binding-diff then recorded junk self-referential entries
   (`Person_2 → Person_2`) that corrupted the reverse mapping. Fix: prompt
   reverted to "leave existing bare placeholders as-is, only brace new
   entities" + defensive guard in `_extract_sota_bindings` to reject
   placeholder-shaped spans.
2. **Fallback-path known_names included hallucinations.** The
   `_fallback_plausibility_on_raw` gate used `extracted.entities` as
   trusted names, which included Qwen's hallucinated `Speaker`. Fix:
   `known_names=set()` in fallback — every entity must be transcript-
   grounded.

### Infrastructure lessons

- Anthropic API transients: `APIConnectionError` with underlying
  `ConnectError: Network is unreachable` is a WSL2 virtual-adapter blip,
  not an Anthropic server issue. The pipeline fail-forward handles it
  cleanly (affected stage drops to the predecessor's output). Error
  logging was upgraded to surface `e.__cause__` so future incidents
  diagnose at a glance rather than showing the SDK's generic "Connection
  error." string.

---

## HA Pipeline Latency (2026-03-27)

End-to-end latency measured via curl against the ParaMem server in cloud-only
mode, with escalation to HA's conversation agent (Groq + Llama 3.3 70B) via
WebSocket `conversation.process`. RTX 5070 on WSL2, HA on NAS (LAN).

| Query type | Tool type | Latency | Path |
|-----------|-----------|---------|------|
| Home weather | template | **0.6s** | HA entity state rendering |
| Worldwide weather | script | **1.2s** | Geocode + weather API |
| Current time | template | **1.2s** | HA state rendering |
| Web search (tavily) | script | **2.2s** | External API call |

Architecture: ParaMem → HA WebSocket (conversation.process) → Groq API +
tool execution (inside HA) → response. Single hop — no round-trips between
ParaMem and HA for tool execution.

---

## Dual-Escalation Routing (2026-03-30)

Tri-path routing via dual-graph matching. Zero LLM inference cost for the
routing decision — pure substring + fuzzy matching against two entity graphs.

### Architecture

| Match source | Path | Service |
|---|---|---|
| PA knowledge graph | Local adapter probe + reasoning | Mistral 7B (local) |
| HA entity graph | HA conversation agent | Groq + Llama 3.3 70B (via HA) |
| Neither graph | HA first (tools), SOTA fallback (reasoning) | HA → Cloud |
| Both graphs | PA first; [ESCALATE] → HA → SOTA | Local → HA → Cloud |

All escalation paths follow the same invariant: **HA first** (has tools for
real-time data), **SOTA fallback** (reasoning). This applies to Path 3 (no
graph match), `[ESCALATE]` from local model, and `_probe_and_reason` fallback
when keyed recall fails.

**HA entity graph:** Built from HA REST API at startup. 238 entities, 164 action
verbs across 52 domains. Indexes friendly names and service verbs (turn_on →
"turn on"). Refreshed after consolidation and via `POST /refresh-ha`.

**Imperative detection:** HA entity match + action verb + non-interrogative →
routes directly to HA, skipping local inference.

**Area routing:** Handled by HA internally (voice satellite context). ParaMem
does not replicate room resolution.

### Fallback Chain

Local mode: local adapter → HA/Groq → SOTA → local base model.
Cloud-only mode: HA/Groq → SOTA → static error.
Every path terminates gracefully. No dead ends.

### Forced Routing

The `route` parameter on `/chat` allows direct provider testing: `"ha"`,
`"sota"`, `"sota:anthropic"`, `"sota:openai"`, `"sota:google"`. Requires
completed speaker identification (greeting flow) to prevent unauthenticated
HA device control.

### SOTA Persona Continuity

Sanitized conversation history (PII-blocked turns dropped) + speaker name
passed to the SOTA model. System prompt instructs the model to derive persona,
tone, and style from the conversation context. No personal facts leak to cloud.

### Multi-Provider SOTA (2026-03-30)

Three SOTA providers with web search, configurable via `agents.sota_providers`
in server.yaml. Web search is enabled by default but defers to caller-supplied
tools when provided.

| Provider | Model | Web Search | Install |
|---|---|---|---|
| Anthropic | Claude Sonnet 4.6 | `web_search_20250305` tool | `pip install paramem[anthropic]` |
| OpenAI | gpt-4o-search-preview | `web_search_options` | core (httpx) |
| Google Gemini | gemini-2.5-flash | Google Search grounding | `pip install paramem[google]` |

Also available via core httpx adapter: Groq, Mistral, Ollama.

**Graceful failure at every level:**
- Missing SDK → logged with install instructions, provider not registered
- Missing API key → logged, provider skipped
- Connection/timeout errors → user-facing error message, fallback chain continues
- Unknown provider → logged with available list

**Known issue:** Gemini API times out (30s) when VPN is active. Works without VPN
(1-2s latency). This is a VPN routing issue, not a code bug.

### Integration Test Results (2026-03-30, local mode, VPN to NAS)

15/17 passing. 2 failures: Gemini timeout (VPN routing issue).

| Test | Path | Provider | Latency | Result |
|---|---|---|---|---|
| Time query | HA | Groq (via HA) | 635ms | PASS |
| Weather query | HA | Groq (via HA) | 935ms | PASS |
| Time query | SOTA | Anthropic | 12.2s | PASS |
| Weather query | SOTA | Anthropic | 5.6s | PASS |
| Time query | SOTA | OpenAI | 7.0s | PASS |
| Weather query | SOTA | OpenAI | 9.4s | PASS |
| Time query | SOTA | Gemini | 32.4s | FAIL (VPN timeout) |
| Weather query | SOTA | Gemini | 32.4s | FAIL (VPN timeout) |
| Real-time escalation | Auto | HA→Anthropic | 11.8s | PASS |
| Reasoning | Auto | SOTA | 2.2s | PASS |
| Math | Auto | SOTA | 2.7s | PASS |
| Memory probe | Auto | Local | 2.7s | PASS |
| Imperative HA | Auto | HA fallback | 2.6s | PASS |
| HA graph refresh | /refresh-ha | — | — | PASS (238 entities) |
| Status endpoint | /status | — | — | PASS |

---

## Data Sources

### PerLTQA (primary)

Public dataset: [PerLTQA](https://github.com/Elvin-Yiming-Du/PerLTQA).
Cloned to `data/external/PerLTQA/`. 141 characters with ~20-25 dialogues each.

Each character has:
- **Dialogues:** timestamped multi-turn conversations tied to life events
- **Events:** narrative descriptions of significant life moments
- **Profile + social relationships:** structured personal facts
- **Ground-truth QA pairs** (32 characters, up to ~400 QA per character)

Character used for Test 1: **Liang Xin** — 30 dialogues, 485 turns,
394 ground-truth eval QA pairs across profile/social/events/dialogues.

### Synthetic fallback

`data/synthetic/personal_facts.json` — 20 hand-crafted QA pairs.
`data/synthetic/contradiction_sessions.json` — 10 fact chains for Tests 2 and 2b.
`data/synthetic/reinforcement_sessions.json` — 30 facts at 3 frequency tiers for Test 4.
`data/synthetic/inference_facts.json` — base facts for Test 3 (replaced by PerLTQA in redesign).

---

## Output Structure

```
outputs/
├── test1_scale/               # Scale expansion results
│   ├── gemma/{timestamp}/     # Timestamped per-model results
│   └── mistral/{timestamp}/
├── test2_contradictions/      # Fresh adapter per session
├── test2b_incremental/        # Persistent adapter, forgetting test
├── test3_inference/
├── test4_reinforcement/
├── test5_natural_recall/
├── test6_footprint/
└── test7_second_persona/
```

Each run writes to a unique timestamped directory. No run can overwrite another.

---

## Independent Expert Review (2026-03-21)

Pre-publication review by an ML expert persona with experience in PEFT,
continual learning, and RAG. Unfiltered assessment of novelty, commodity
results, overclaiming, underclaiming, and venue readiness.

### Genuinely novel

1. **The indexed key mechanism itself.** Per-fact addressable recall from a
   LoRA adapter with structured JSON output. Enables enumeration — listing
   everything an adapter "knows." No prior LoRA-as-memory work addresses this.
   DSI (Tay et al.) operates at document level with full model training.

2. **The negative results are the most publishable findings.**
   - Test 4b: 0/40 old key recall after 5 incremental cycles — clean
     demonstration that rank-8 LoRA cannot incrementally accumulate facts.
   - Test 7b: additive composition fails, not just weight merging — useful
     data point for the community, well-situated against ACL 2025 and ICML
     2025 literature.

3. **Test 5's natural vs keyed recall gap.** Per-question natural is 50/50
   (identical to keyed), but broad probes yield 22-32/50. Keys provide
   addressability, not better recall. More honest and more interesting than
   "keys improve recall."

4. **SimHash registry.** 8 bytes per key for hallucination detection is
   genuinely lightweight. Underexplored — deserves its own evaluation.

### Commodity (expected, well-known)

1. **100% recall at 100 keys.** Engineering validation, not scientific finding.
   Any well-tuned rank-8 LoRA can memorize 100 QA pairs. Scale is too small
   to be interesting.

2. **Linear training time scaling.** Expected for LoRA fine-tuning.

3. **Multi-persona isolation (Test 7).** Separate LoRA adapters don't
   interfere — this is PEFT's design, not ParaMem's contribution.

4. **Contradiction detection via predicate normalization (Test 2).** Standard
   knowledge graph maintenance, not a parametric memory finding.

5. **RAG latency comparison.** Generation-only latency at 50-100 facts is
   not meaningful — RAG retrieval is sub-millisecond at this scale.

### Overclaiming

1. **"PM matches or exceeds RAG" (Test 3).** N=14 inference questions, no
   statistical significance test. Differences (1.2%, 7.8%) are within noise.
   "Indistinguishable" is defensible; "exceeds" is not.

2. **"PM recall exceeds RAG" (Test 6).** Compares adapter trained on exact QA
   pairs against top-3 retrieval. RAG baseline is weak by design. Test 3's
   all-facts comparison is fairer and shows parity.

3. **"Constant-size adapter (35MB) vs growing RAG (89MB)."** The 89MB includes
   the embedding model (fixed cost). Actual RAG index at 100 facts is KB. At
   scale where storage matters (10K+), the adapter would also need to grow.

4. **"Model-agnostic."** Two models (7B, 9B), both instruction-tuned. Does not
   establish model-agnosticism. A 3B or 70B model could behave differently.

5. **"Biologically-inspired."** Metaphor, not mechanism. The episodic/semantic
   partition is just different rank and learning rate configs. Reviewers at
   an ML venue will see this as narrative decoration.

### Underclaiming

1. **Failed approaches are undersold.** Format collision (XML: 0.0 F1, trained
   hash: 0/10) is practically valuable — many will try these. Deserves more
   prominence.

2. **Test 4b forgetting result is stronger than presented.** Clean empirical
   characterization of LoRA capacity interference with sharp boundary condition.
   Should be framed as a contribution, not just a negative finding.

3. **SimHash registry is underexplored.** False positive/negative rate? What
   about adversarial keys? Left on the table.

4. **Distillation precision-recall tradeoff (Test 1).** Mistral 80% precision /
   low yield vs Gemma 55% precision / high yield characterizes an extraction
   tradeoff relevant to any LLM knowledge extraction pipeline. Buried in
   methodology.

### What a top-venue reviewer would criticize

1. **Scale.** 100 keys is trivially small. Need 1000+ where capacity limits
   and interference become real constraints.

2. **No baselines against existing systems.** PLUM (81.5% personal facts),
   ROME/MEMIT (knowledge editing) are not compared head-to-head.

3. **No statistical rigor.** Single runs, no confidence intervals, no
   significance tests. Perfect scores suggest tasks are too easy.

4. **RAG baseline is not competitive.** Top-3 retrieval at 100 facts is a
   strawman. Need BM25 + dense retrieval + reranking.

5. **No downstream task evaluation.** All evaluation is intrinsic (can the
   model recall what it was trained on?). No evaluation of actual agent
   improvement.

6. **Single dataset (PerLTQA).** One character from one dataset is a case
   study, not a generalizable evaluation.

7. **Distillation bottleneck unaddressed.** 12% ground-truth coverage means
   perfectly recalling a small fraction of what matters.

8. **Training cost not compared.** 70 minutes for 100 keys vs RAG indexing
   (free) or knowledge editing (seconds per fact).

### Venue assessment

- **Workshop paper / systems paper:** Current results are sufficient if
  reframed around mechanism design and failure modes.
- **Main conference (NeurIPS, ICML, ACL):** Needs scale to 1000+ keys,
  PLUM/ROME baselines, competitive RAG, multiple runs, downstream tasks.
  Current version: borderline reject.
- **arXiv preprint:** Submitted 2026-03-22. Tagged `v1.0-arxiv`. Awaiting
  endorsement for cs.LG/cs.AI.

### Recommended reframing

Center the paper on:
1. The enumeration mechanism (indexed keys + SimHash verification)
2. The negative findings (what does and does not compose in LoRA weight space)
3. The addressability insight (keys provide enumeration, not better recall)

With positive results (100% recall, parity with RAG) as supporting evidence
rather than the headline.

---

## Security Considerations

Detailed threat modeling and probe attack analysis in `security_analysis.md` (gitignored).

Summary: Parametric memory provides meaningful at-rest security improvement over RAG
(facts in weights vs plain text files). Runtime exposure during reasoning is identical
to any system that processes private data. Probe resistance is limited — an attacker
with the adapter file + base model can extract facts through differential analysis.
Open research directions include training format hardening, selective access control,
and multi-adapter compartmentalization.

---

## Extraction Pipeline Evolution

### v1: Outlines constrained generation (2026-03-25, superseded)

Outlines never worked in production — 0% success across all Tests 1-8 due to a `max_tokens` bug, then inconsistent failures with quantized Mistral 7B even after the fix. Every successful extraction came from the unconstrained prompt-parse fallback. Outlines was removed entirely in favor of generate-once-parse-once.

### v2: 7-stage privacy-aware pipeline (current)

Anonymize → extract → SOTA enrich → deanonymize → residual sweep → grounding gate → plausibility filter. Each stage has one job and a clear failure mode. Prompts externalized to `configs/prompts/`. See "Extraction Probe Sweep (2026-04-17)" below for validated results at scale.

## Extraction Probe Sweep (2026-04-17)

Large-scale extraction quality assessment across two datasets using the dataset-agnostic probe (`experiments/dataset_probe.py`). Mistral 7B NF4, SOTA enrichment enabled, `--no-train` (extraction diagnostics only).

### Datasets

| Dataset | Sessions | Transcript size | Description |
|---------|----------|----------------|-------------|
| LongMemEval (stratified 100) | 100 | ~14k chars, 144 turns | Multi-topic Q&A, encyclopedia-style |
| PerLTQA (3 characters) | 89 | ~1.9k chars, ~20 turns | Personal dialogues, character-driven |

### Two bugs fixed during the sweep

1. **Deanonymization substring bug** (`paramem/graph/extractor.py`): Deanonymization used exact dictionary lookup (`.get()`) instead of substring replacement. Composite strings like `"Person_2's cousin"` failed the lookup (only bare `"Person_2"` was in the mapping), leaving the placeholder intact. `_strip_residual_placeholders()` then dropped the entire fact. Fixed to use `re.sub()` with word boundaries, mirroring `_anonymize_transcript()`.

2. **Speaker name mistyping** (`experiments/utils/longmemeval_loader.py`): LongMemEval loader hard-coded `speaker_name="User"` for all sessions. The extraction pipeline typed "User" as `concept` instead of `person`, losing speaker-centric relationships. Fixed with `SpeakerNamePool` — deterministic pseudonym assignment per session from a pool of 614 culturally diverse first names.

### Results: LME new (with fixes) vs LME old (baseline)

Same 100 stratified sessions (seed=42), paired comparison:

| Metric | Old (baseline) | New (fixes) | Delta |
|--------|---------------|-------------|-------|
| Post-plausibility QA | 379 | 381 | +2 (flat) |
| Residual placeholder drops | 71 | 38 | **-33 (46% reduction)** |
| Person entities | 80 (20.2%) | 99 (24.3%) | **+19 (+4.1pp)** |
| Preference relations | 87 (23.0%) | 138 (36.2%) | **+51 (+59%)** |
| Social relations | 10 (2.6%) | 12 (3.1%) | +2 |
| Anonymization success | 74% | 87% | **+13pp** |
| Ungrounded drops | 13 | 27 | +14 (expected) |
| Zero-extraction sessions | 6 | 11 | +5 |

**Interpretation:** Total QA yield is flat (+2), but extraction *quality* improved structurally. More correct entity types (person, place), more diverse relation types (preference +59%, social +20%). The remaining 38 residual drops are SOTA-invented placeholders never in the mapping — the deanon path is fixed. Ungrounded drops increased because facts previously lost to the residual sweep now survive deanonymization but fail the grounding gate — the pipeline filters more precisely. The 5 additional zero-extraction sessions reflect non-deterministic extraction variance from altered transcript text (speaker names), not a regression.

Per-session paired analysis: 17 sessions had residual drops eliminated, with individual recoveries of up to +10 QA pairs. 42 sessions gained QA, 33 lost, 25 unchanged.

### Results: PerLTQA new (with fixes) vs PerLTQA old (baseline)

Same 89 sessions across 3 characters (Deng Yu 31, Liang Xin 30, Xia Yu 28), paired comparison.

| Metric | Old (baseline) | New (fixes) | Delta |
|--------|---------------|-------------|-------|
| Post-plausibility QA | 587 | 583 | -4 (-0.7%) |
| Raw facts | 842 | 854 | +12 (+1.4%) |
| Residual placeholder drops | 162 | 160 | -2 (-1.2%) |
| Person entities | 81 (15.8%) | 82 (15.6%) | +1 |
| Preference relations | 140 (23.9%) | 142 (24.4%) | +2 |
| Social relations | 118 (20.1%) | 122 (20.9%) | +4 |
| Anonymization OK+repaired | 84 (94.4%) | 84 (94.4%) | 0 |
| Ungrounded drops | 60 | 67 | +7 |
| Plausibility drops | 42 | 44 | +2 |
| Zero-extraction sessions | 5 | 4 | -1 |

**Interpretation:** The composite-placeholder deanon fix barely moves PerLTQA (-1.2% residual drops) compared to LME (-46%). PerLTQA's first-person dialogue rarely triggers the SOTA enrichment patterns (`Person_1's cousin`, `downtown City_1`) that the bug affected — those constructions appear primarily in LME's assistant-style content. The fix is real and validated on LME; on PerLTQA it shows no regression.

The +7 net ungrounded drops are spread across 12 sessions (deltas ±1-2 each). Raw fact counts shift in both directions between runs — extraction is mildly non-deterministic at temperature=0, and the grounding gate correctly drops the new ungrounded subset each run. Entity and relation type distributions are essentially unchanged: pipeline already stable on this dataset. One session (`Xia Yu_119_10_4#13`) shows `leaked_repaired` in both runs with identical content (11 raw, 1 residual drop, 1 ungrounded) — a deterministic single-token leak that the repair path handles correctly.

### Results: Cross-dataset comparison

| Metric | LME (100) | PerLTQA (89) |
|--------|-----------|-------------|
| QA pairs/session | 3.8 | 6.6 |
| Processing time/session | 97s | 77s |
| Person entities | 24.3% | 15.8% |
| Social relations | 3.1% | 20.1% |
| Preference relations | 36.2% | 23.9% |
| Residual drops (old → new) | 71 → 38 (-46%) | 162 → 160 (-1.2%) |

LME transcripts are 7x longer but yield fewer facts — most content is informational Q&A, not personal knowledge. PerLTQA's character-driven dialogues produce denser personal facts and more social relationships. Both datasets are consistent with live HA deployment observations: extraction is selective rather than exhaustive, capturing genuine personal knowledge rather than every mentioned fact.

### Preference extraction quality (spot check)

One LME session (Nadia, mid-century modern design conversation) produced 11 QA pairs — 10 distinct preferences (clean lines, tapered legs, wood accents, modular design, etc.) and 1 factual. All non-redundant, correctly typed. Minor formatting artifact: some multi-word concepts extracted as `Snake_Case` (e.g. `Organic_Shapes`). Self-healing — QA regeneration from the cumulative graph produces natural language on the next consolidation cycle.

### Diagnostics accounting fix

`raw_fact_count` in session diagnostics went negative when SOTA enrichment added more facts than the original extraction produced. Fixed by splitting `plausibility_dropped` into actual drops (floored at 0) and `enrichment_added`. All 5 run directories retroactively corrected (38 + 18 files).

### Infrastructure findings

- **Modern Standby sleep inhibitor** (`experiments/utils/gpu_guard.py`): Overnight run crashed due to Windows Modern Standby power-cycling the GPU during CUDA compute (TDR BSOD, bugcheck 0x116). Root cause: `nvlddmkm.sys` driver race on power state transitions, not thermal. Fix: `acquire_gpu()` now holds `ES_CONTINUOUS | ES_SYSTEM_REQUIRED` via background PowerShell on WSL2. Validated across 100-session re-run with zero crashes.
- **Cooling pad impact**: Reduces cooldown wait from ~3 min to ~30s between sessions. Enables occasional Dynamic Boost bursts (87W, 2625 MHz vs sustained 58W, 2010 MHz). Primary benefit is thermal recovery speed, not sustained clock — 60W TGP is the binding constraint.
- **Wall-clock timing**: LME 100 sessions = 162 min processing, 385 min wall (cooldown overhead). PerLTQA 89 sessions = 115 min processing, ~115 min wall (shorter sessions, negligible cooldown).

### `anon=not_run` sessions (7 of 100 LME) — correct behavior

All 7 are generic assistant conversations with no personal information (NAS recommendations, mall stores, travel tips, packing lists, online courses). Extraction correctly returns 0 facts → anonymization is skipped. The old run hallucinated facts from several of these — e.g., 16 "considers_purchasing" relations from a NAS recommendation chat where the *assistant* listed products. The count increasing from 3 (old) → 7 (new) is a quality improvement: 4 sessions that previously produced false positives now correctly produce nothing.

### Open items

- Location → place type inference: SOTA enrichment occasionally tags places as `concept` instead of `place` when no explicit "place" cue appears in the transcript. Low priority; `place` already accounts for ~5% of LME entities post-fix.
- Case-dup normalization: SOTA produces `Bioinformatics` vs `bioinformatics` as distinct entities. Affects entity merging across sessions.
- Subject/object inversion: SOTA occasionally inverts predicate direction (e.g. `lives_in(City, Person)` instead of `lives_in(Person, City)`).
- First-session SOTA parse failure: JSON output uses double-quote escaping that occasionally breaks the parser; fallback path catches it.

---

## F5.1 HA Deployment Results (2026-03-25)

First live deployment of ParaMem as a Home Assistant conversation agent.

### Setup

- **HA host:** Home Assistant in Docker, custom component REST client
- **GPU host (WSL2):** ParaMem server, Mistral 7B NF4
- **Network:** HA host → LAN → port forward → WSL2

### Results

- Full pipeline validated: voice → STT → HA → ParaMem → adapter recall → TTS
- 9 keys trained from single conversation, correct parametric recall
- Speaker identification and entity routing work naturally
- Escalation fires for unknown facts
- No personal data at rest — only key IDs, SimHash, session counts
- Server auto-starts via systemd user service

### Key observation

Personal knowledge recalled from adapter weights makes the agent feel genuinely personal — it knows where you live, what you do, who your family is — without any documents stored on disk. This validates the core thesis: parametric memory as a practical alternative to RAG for personal agents.

---

## F5.5 Speaker Identification (2026-04-08)

Voice-based multi-user speaker identification via pyannote embeddings, integrated into the Wyoming STT pipeline.

### Architecture

- **Embedding model:** pyannote/embedding (4.3M params, 512-dim, CPU inference <1s)
- **Profile format (v3):** multi-embedding — each speaker stores up to 50 embeddings from different utterances and devices. Matching uses L2-normalized centroid.
- **Enrollment:** deferred LLM extraction. Unknown voices grouped silently by embedding similarity. After a global cooldown (600s), the system prompts for introduction. Name extracted from conversation context via local LLM during idle periods.
- **Enrichment:** confirmed matches auto-add the new embedding to the profile. The centroid naturally becomes cross-device as the speaker uses different satellites.

### Voice Satellites

| Device | Location | Mic Type | Embedding Quality |
|--------|----------|----------|------------------|
| ReSpeaker Lite | Living room | Dedicated dual-mic array, hardware beamforming | More consistent (purpose-built for voice) |
| ESP32 S3 Box 3 | Office | ES7210 ADC + built-in MEMS mic | Noisier (general-purpose dev kit) |

### Measured Embedding Scores (cosine similarity)

Same speaker, pyannote 512-dim embeddings:

| Condition | Score Range | Notes |
|-----------|------------|-------|
| Office mic → office mic (different utterances) | 0.15–0.57 | High variance from short commands + MEMS mic noise |
| Cross-device (ReSpeaker → S3 Box) | 0.38–0.52 | Channel mismatch dominates |
| After centroid enrollment (2-3 embeddings) | 0.54–0.67 | Centroid averaging recovers signal |
| Centroid match from office (3 embeddings) | 0.54–0.64 | Within high-confidence threshold |
| Centroid match from living room (3 embeddings) | 0.67 | Better mic produces better scores |

### Thresholds (configured in server.yaml)

| Threshold | Value | Purpose |
|-----------|-------|---------|
| High confidence | 0.60 | Confirmed match — attach speaker, enrich centroid |
| Low confidence | 0.45 | Tentative match — attach without interruption |
| Redundancy | 0.95 | Skip add_embedding if too similar to centroid |
| Grouping factor | 0.6 × low = 0.27 | Group unknown voices (lenient for noisy embeddings) |
| Min embedding words | 5 | Discard embeddings from shorter transcripts |

### Key Findings

1. **Single-utterance embeddings are unreliable.** Pyannote needs ~3s of voice for stable prints. Short commands ("Play music", 1.5s after VAD) produce scores that vary 0.15–0.57 for the same speaker on the same device.

2. **Cross-device enrollment requires centroid averaging.** A single enrollment embedding from one mic doesn't transfer to a different mic (0.38–0.52). The L2-normalized centroid from multiple devices recovers matching quality (0.54–0.67).

3. **ReSpeaker > S3 Box for voice capture.** Dedicated voice hardware with hardware beamforming produces more consistent embeddings than a general-purpose ESP32 dev kit.

4. **The system improves with use.** Each confirmed match enriches the centroid. After a few conversations from each room, cross-device matching converges above the high-confidence threshold.

### Personalization

- Daily greeting on first interaction per speaker (configurable interval, default 24h)
- Time-of-day aware: "Good morning/afternoon/evening, {name}"
- App-layer prepend to spoken response — not in the training transcript (prevents greeting patterns leaking into adapter weights)
- Greeting timestamps persisted in SpeakerStore (keyed by speaker_id, UTC). Survives server restarts.
- Only for confirmed speakers (high-confidence match). Unknown speakers get no greeting.

### Privacy Mode

- `debug: false` in server.yaml: transcripts live only in RAM. After consolidation, knowledge is in the adapter weights — no textual traces remain on disk.
- `debug: true` (default): transcripts written to JSONL files for inspection, archived after consolidation.
- Encrypted session snapshots on graceful shutdown (SIGUSR1, SIGTERM): Fernet-encrypted snapshot saved to disk, restored on startup, deleted immediately after restore. Unconsolidated conversations survive controlled restarts. Uncontrolled kills (SIGKILL, power loss) lose unconsolidated data — acceptable.
- Snapshot key configured via `${PARAMEM_SNAPSHOT_KEY}` env var. No key = snapshots disabled.

### Resilience

- Speaker resolution failure → proceeds as anonymous (no 500 error)
- Enrollment failure → logged, query continues normally
- HA custom component: ParaMem server error → falls back to HA conversation agent (explicit `conversation.groq` agent_id to prevent recursive fallback)
- HA agent also fails → generic error message. No dead ends.

### Infrastructure Notes

- Sonos TTS forwarding via HA automation (`esphome.tts_uri` event → `media_player.play_media` with `announce: true`)
- Music Assistant entity sync can break silently — MA container restart fixes it
- Wyoming STT on port 10300, REST API on port 8420
- Speaker profiles persisted as JSON, deferred disk writes flushed on shutdown

### Cooperative Background Training (live)

Consolidation is driven by a systemd user timer (`paramem-consolidate.timer`,
`Persistent=true`) whose period derives from `consolidation.refresh_cadence`
(default `12h`). The `BackgroundTrainer` releases the GPU lock per step so
inference interleaves with training, and saves `resume_state.json` +
`bg_checkpoint/` at each epoch boundary — a crash or `SIGUSR1` mid-cycle
resumes at the last completed epoch instead of restarting from zero
(SHA-256 fingerprint gate on `keyed_pairs` + training config).

Two adapter tiers share the GPU:

- **Main adapters** (`episodic` / `semantic` / `procedural`) — rebuilt at the
  full-consolidation boundary.
- **Interim adapters** (`episodic_interim_<stamp>`) — minted at each
  `refresh_cadence` tick, activity-gated, capped by `max_interim_count`
  (default 7, VRAM-gated via pre-load validator). At the full boundary,
  `consolidate_interim_adapters` rebuilds the mains from
  `keyed_pairs ∪ all_interim_keys`, sanity-checks recall, and purges interim
  state atomically.

Operational invariant: every consolidation still retrains the full key set
via replay. True incremental learning without replay remains unsolved
(Test 4b: catastrophic forgetting). Full retrain is acceptable today because
the systemd timer fires outside active hours, interim adapters cover
sub-cycle recall, and the epoch-resume mechanism lets a single cycle span
wall-clock interruptions. Listed as future work.
