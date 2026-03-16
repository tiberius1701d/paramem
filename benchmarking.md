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

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python experiments/test1_scale_expansion.py --model gemma
python experiments/test2_contradictions.py --model mistral
```

Results go to `outputs/testN_*/{model}/{timestamp}/results.json`.

---

## Test 1: Scale Expansion

**Script:** `experiments/test1_scale_expansion.py`
**Status:** COMPLETE — both models

### What it tests

Can parametric memory maintain recall quality as the number of stored facts grows
from 10 to 100? Facts are extracted on the fly from real dialogue transcripts
(PerLTQA dataset), one session at a time, mimicking idle-time background learning.

### Data pipeline

1. Load dialogue sessions for character "Liang Xin" (30 available)
2. Process sessions one at a time: transcript → graph extraction → QA generation
3. Accumulate QA pairs until target count reached
4. Train indexed key adapter → evaluate recall

### Results (2026-03-16)

| Scale | Gemma 2 9B Recall | Gemma Time | Mistral 7B Recall | Mistral Time |
|-------|-------------------|------------|--------------------| -------------|
| 10    | 10/10 (100%)      | 7.1 min    | 10/10 (100%)       | 5.6 min      |
| 25    | 24/25 (96%)       | 17.4 min   | 25/25 (100%)       | 13.9 min     |
| 50    | 49/50 (98%)       | 35.2 min   | 49/50 (98%)        | 28.5 min     |
| 75    | 75/75 (100%)      | 52.9 min   | 75/75 (100%)       | 42.2 min     |
| 100   | 100/100 (100%)    | 70.3 min   | 95/100 (95%)       | 48.8 min     |

**Config:** rank=8, alpha=16, 30 epochs, lr=1e-4, batch=1, grad_accum=2.

### Data statistics

| Metric | Gemma | Mistral |
|--------|-------|---------|
| QA pairs extracted | 103 | 106 |
| Sessions used | 22/30 | 27/30 |
| QA pairs per successful session | ~4.7 | ~3.9 |
| Session extraction failure rate | ~36% | ~40% |

### Training time scaling

| Metric | Gemma | Mistral |
|--------|-------|---------|
| Time per key (scale 10) | 42.4 s | 33.8 s |
| Time per key (scale 50) | 42.3 s | 34.1 s |
| Time per key (scale 100) | 42.2 s | 29.3 s |
| Scaling behavior | Linear | Linear |
| Total training (all 5 scales) | 183 min | 139 min |

Training time scales linearly with key count — no superlinear growth.
Mistral is ~30% faster, roughly proportional to parameter count (7B vs 9B).

### Loss convergence

| Scale | Gemma Final Loss | Mistral Final Loss |
|-------|------------------|--------------------|
| 10    | 0.338            | 0.316              |
| 25    | 0.256            | 0.235              |
| 50    | 0.213            | 0.210              |
| 75    | 0.192            | 0.179              |
| 100   | 0.189            | 0.182              |

Loss decreases with more data — the model is less overfit at larger scales
due to greater gradient diversity per epoch. Both models converge at similar rates.

### Epoch convergence analysis

Both models reach loss < 0.01 by epoch 4-6 across all scales:

| Loss threshold | Scale 10 | Scale 25 | Scale 50 | Scale 75 | Scale 100 |
|----------------|----------|----------|----------|----------|-----------|
| < 0.1          | epoch 5  | epoch 4  | epoch 3  | epoch 3  | epoch 2   |
| < 0.01         | epoch 6  | epoch 5  | epoch 4  | epoch 4  | epoch 4   |
| < 0.001        | epoch 10 | epoch 7  | epoch 7  | epoch 5  | epoch 5   |

Values are the slower of the two models at each scale point.
The last 10-20 epochs of a 30-epoch run contribute negligible learning.

### Failure analysis

**Gemma failures:** 2 total across all scales. Both are graph13 at scale 25/50
(parse failure, null JSON output). Self-corrected at scale 75/100 with more
training data. No recall failures — only generation format issues.

**Mistral failures:** 6 total, all at scale 50-100. All are generation quality
issues — garbled text (missing spaces, negated facts, truncated words), not
convergence failures. SimHash confidence 0.81-0.91 indicates the model memorized
the content but produces imperfect output.

### Distillation quality (pipeline issue, not memory issue)

Content analysis of the 100 recalled QA pairs per model:

| Issue | Gemma | Mistral |
|-------|-------|---------|
| Clean, well-formed pairs | 23% | 81% |
| Vague "is known about" template fallback | 50% | 17% |
| Markdown artifacts (`**`) in output | 27% | 0% |
| Truncated/garbled questions | 27% | 2% |
| Wrong predicate (e.g. hobby for work tasks) | 6% | 0% |

Mistral produces significantly cleaner QA pairs despite slightly lower recall.
Gemma's markdown generation tendency and less consistent `Q:/A:` formatting
cause the QA generator to fall back to vague templates more often.

Both models miss core profile facts (gender, age, occupation) and use only the
nickname "Xinxin" — never "Liang Xin." This is a graph extraction limitation:
profile facts aren't stated explicitly in dialogues, only inferred from context.

Extracted facts that are present tend to be factually correct when verified
against PerLTQA ground truth. The quality issue is in QA formatting, not
factual accuracy.

Fixes applied after Test 1: explicit `A:` format instruction in QA generator
prompt, robust parser fallback for missing markers, markdown artifact cleaning.
Will be validated in subsequent runs.

### Key takeaways

1. **Parametric memory scales to 100 keys with near-perfect recall.** Both models
   achieve 95-100% exact recall at scale 100 from real conversational data. The
   core indexed key mechanism is model-agnostic and reliable.

2. **Training time is linear and predictable.** ~42s/key (Gemma) and ~30s/key
   (Mistral), constant across all scale points. No superlinear growth. At scale
   100, training takes 49-70 min — feasible for idle-time background processing.

3. **The distillation pipeline is the bottleneck, not the memory mechanism.**
   36-40% of sessions fail extraction entirely. Of the facts that are extracted,
   23-81% are well-formed depending on model. Improving extraction and QA
   generation quality is the highest-leverage improvement.

4. **Model choice is a speed/quality tradeoff.** Mistral is 30% faster and produces
   cleaner QA pairs. Gemma has slightly higher recall at scale (100% vs 95%).
   Both are viable — the architecture is model-agnostic.

5. **30 epochs is excessive.** Loss converges by epoch 5-10 across both models and
   all scales. Early stopping can save 37-67% of training time. Implementation
   available but not yet active (needs threshold tuning for Mistral).

---

## Early Stopping

**Status:** IMPLEMENTED, NOT ACTIVE — needs threshold tuning for Mistral

### Implementation

Loss-based early stopping using epoch-average loss (not per-step):
- **Threshold:** avg epoch loss < 0.01
- **Floor:** 10 epochs minimum
- **Patience:** 2 consecutive epochs below threshold
- **Hard cap:** 30 epochs

Implemented in `LossEarlyStoppingCallback` (`paramem/training/trainer.py`).
Configurable via `TrainingConfig`. Defaults to off (`early_stopping=False`).

### Validation (scale=25, PerLTQA data)

| Metric | 30-epoch baseline | Early stopping | Delta |
|--------|-------------------|----------------|-------|
| Gemma recall | 24/25 (96%) | 24/25 (96%) | No change |
| Gemma epoch reached | 30 | 19 | -37% |
| Gemma training time | 17.4 min | 11.0 min | -37% |
| Mistral recall | 25/25 (100%) | 22/25 (88%) | **-3 keys** |
| Mistral epoch reached | 30 | 22 | -27% |
| Mistral training time | 13.9 min | 8.9 min | -36% |

Gemma: no regression — identical recall at 37% less training time.
Mistral: 3-key regression (parse failures at epoch 22 that resolve by epoch 30).
The threshold of 0.01 is too aggressive for Mistral. Not yet active in
production runs until a threshold that works for both models is validated.

Note: The smoke tests ran with the distillation pipeline fixes (extractor null
handling, QA generator `A:` format instruction, markdown cleaning) that were
applied after the Test 1 full runs. Test 1 results reflect the old pipeline;
smoke test results reflect the fixed pipeline.

---

## Test 2: Contradiction Resolution

**Script:** `experiments/test2_contradictions.py`
**Status:** REDESIGNED — pending execution

Two contradiction resolution strategies:
- **2a (graph):** Predicate normalization catches exact predicate matches
  (e.g. `works_at` vs `works_at` with different object)
- **2b (model):** LLM semantic reasoning catches synonym predicates
  (e.g. `moved_to` contradicts `lives_in`)

10 fact chains with 3 temporal versions each. Both strategies run per model
for direct comparison.

## Test 3: Associative Inference

**Script:** `experiments/test3_inference.py`
**Status:** IMPLEMENTED — pending execution

## Test 4: Multi-Session Reinforcement

**Script:** `experiments/test4_reinforcement.py`
**Status:** IMPLEMENTED — pending execution

## Test 5: Privacy Preservation

**Script:** `experiments/test5_privacy.py`
**Status:** IMPLEMENTED — pending execution

## Test 6: Edge Deployment Footprint

**Script:** `experiments/test6_footprint.py`
**Status:** IMPLEMENTED — pending execution

## Test 7: Second Persona

**Script:** `experiments/test7_second_persona.py`
**Status:** IMPLEMENTED — pending execution

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
`data/synthetic/contradiction_sessions.json` — 10 fact chains for Test 2.

---

## Output Structure

```
outputs/
├── test1_scale/             # Scale expansion results
│   ├── gemma/{timestamp}/   # Timestamped per-model results
│   └── mistral/{timestamp}/
├── test2_contradictions/
├── test3_inference/
├── test4_reinforcement/
├── test5_privacy/
├── test6_footprint/
└── test7_second_persona/
```

Each run writes to a unique timestamped directory. No run can overwrite another.
