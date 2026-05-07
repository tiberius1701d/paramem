# Test 14a-pre — reproduction & extension

## Status (2026-05-06)

**Multi-seed run COMPLETE** at the apples-to-apples config
(`linear + B50 + decay=600`). V1/V2/V3 × [42, 7, 1337] all converged
with zero placeholder leakage and final recall 1.0. **The n=1 fill-speed
ordering V3 < V2 < V1 does not survive multi-seed** — see
`benchmarking.md` "Multi-seed replication + V5–V8 expansion" for the
result table.

V5–V8 expansion is **dropped** (decision rule cannot be met against
V3's noise floor). 14a (scale to N=500) is **deferred on Test 15**
(`experiments/test15_retention_multiseed.py`, in design as of
2026-05-07) which multi-seeds the scaffold-fill-vs-answer-swap
retention question under the now-live recall early-stop — the
dimension Test 14 never measured.

This document is preserved for reproducibility of the apples-to-apples
configuration diagnostic; the "Next step" sections below describe what
*was* run on 2026-05-05 → 2026-05-06, not what's pending.

### Smoke result — V3/seed42

| Metric | Value |
|---|---:|
| `first_perfect_epoch` | 18 |
| `stable_perfect_epoch` | 20 |
| `stop_epoch` | 20 |
| `final_recall_rate` | 1.0 (20/20) |
| `wall_seconds` | 5346 (89 min) |
| `lr_scheduler_type` | linear |
| `lr_decay_steps` | 600 |
| `num_epochs` (budget) | 50 |

Result on disk: `outputs/test14_pre/mistral/20260426_012907/V3/C/seed42/C_done.json`.

### Configs ruled out

| Config | Outcome | Diagnostic |
|---|---|---|
| `linear + B50 + decay=0` (HF stock, budget-coupled) | Original biased multi-seed result. seed=42 stable=27 vs V3 ref 20. | Per-step LR scales with `num_train_epochs`. Same epoch index → different LR. Not apples-to-apples with V3 ref's B30. |
| `constant_with_warmup + B50 + decay=0` | seed=42 stable=32, oscillation 0.90/0.95/1.00 for 11 epochs e21–e32. | LR=peak forever after warmup; weights keep moving by ~peak per step at zero loss, perturbing the model in and out of the SimHash threshold basin. |
| `linear + B50 + decay=300` | seed=42 frozen at fill=0.35 from e16 to e50 (35 identical epochs). | Decay window completes at step 300 (e15) — half the V3 ref's 600-step trajectory. LR=0 from e15 freezes the model before first_perfect can land. |

Result of **decay=300** falsification preserved at
`V3/C/seed42_linear_d300/` (full epoch_log shows the freeze curve).
Result of **constant_with_warmup** preserved at
`V1/C/seed42_constant_with_warmup_dbudget/` (full epoch_log shows the
oscillation curve). Delete both once recorded in `benchmarking.md`.

### Disk state heading into the multi-seed launch

| Variant | Skip-on-done (correct config) | Auto-migrate aside (mismatched config) | Fresh run |
|---|---|---|---|
| V1 | — | seed42_const_w_warm_dbudget (already migrated) | seed42, seed7, seed1337 |
| V2 | — | seed42, seed7, seed1337 (linear/budget-coupled) | seed42, seed7, seed1337 |
| V3 | seed42 (smoke result) | seed7, seed1337 (linear/budget-coupled) | seed7, seed1337 |

**8 fresh Phase C runs total.** Auto-migration handles the 6 seeds with
mismatched config. V3/seed42's smoke result is reused as-is (skip-on-done).

## Faithful resume

```bash
tresume 14
```

Resumes exactly the configuration that was paused — never silently expands scope.
On first launch, every flag (`phase_c_seeds`, `phase_c_num_epochs`, `lr_scheduler_type`,
`phase_c_decay_steps`, `reuse_phase_a_from`) is persisted to
`<run_dir>/run_config.json`. `tresume` reads each one back and passes it through.
Anything not on first launch stays off on resume.

The current persisted config runs the apples-to-apples defaults:
`phase_c_seeds=[42,7,1337]`, `phase_c_num_epochs=50`,
`lr_scheduler_type="linear"` (default), `phase_c_decay_steps=600` (default).
The 600-step decay window matches the V3 reference's stock-linear trajectory at
B30 (30 epochs × 20 steps/epoch); decoupling decay from `num_train_epochs` keeps
per-step LR invariant to total budget. The 50-epoch budget gives 20 epochs of
LR=0 headroom for slow seeds; recall-based early-stop ends a healthy seed at
~e20–e25.

`tpause` / `tresume 14` are honored at epoch boundaries; per-(variant, seed)
checkpoint resume picks up exactly where it left off.

Auto-migration: when `tresume 14` re-launches against an existing run dir, any
per-seed Phase C result whose persisted `lr_scheduler_type` OR `lr_decay_steps`
differs from the current target gets renamed `seed{N}_<old_lr>_<old_decay>/`
(with a numeric suffix on collision) and Phase C runs fresh. This makes a
contaminated prior batch (linear/B50 from April, constant_with_warmup from
2026-05-03) self-healing.

## Inserting a new experiment

The variant-loop has two skip mechanisms that guarantee finalized data is never
re-run:

1. **Per-variant skip**: if Phase A `A_done.json` (or `phase_a_reused.json`),
   Phase B `B_done.json`, and Phase C are all complete, the variant is skipped.
2. **Per-seed skip** (multi-seed mode): individual `<variant>/C/seed<N>/C_done.json`
   markers; only seeds without one are run.

So a "new test" inside the existing run dir works by being explicit about scope:

### Single-variant smoke before a multi-seed run

The V3/seed42 apples-to-apples smoke is registered as **test 14s** — a peer
test in its own right. `tresume 14s` launches it; `tpause` / `tresume 14s`
cycle works exactly like any other registered test:

```bash
tresume 14s          # launch (or resume) the V3/seed42 smoke
tpause               # stop at next epoch boundary
tresume 14s          # continue from preserved checkpoint
```

The smoke shares the run dir with test 14 so it reuses Phase A and Phase B
from V3 (no 8 h Phase A re-train). Its scope flags are hardcoded in the
registry (`TEST_EXTRA_FLAGS["14s"]` in `scripts/dev/training-control.sh`)
rather than persisted to `run_config.json` — the smoke remains an explicit,
repeatable probe and does not contaminate the persisted multi-seed config.

After the smoke lands a `C_done.json` at the new config, `tresume 14`
continues the V1/V2/V3 multi-seed and skips V3/seed42 (already done).

Equivalent direct invocation (the registry-resolved form):

```bash
~/miniforge3/envs/paramem/bin/python experiments/test14.py \
  --model mistral --mode pre --resume \
  --variant V3 --phase-c-seeds 42 \
  --phase-c-num-epochs 50 --lr-scheduler-type linear --phase-c-decay-steps 600
```

### Registering another smoke / probe

The 14s pattern is the template for any sub-experiment within a Test 14
run dir. To add e.g. a `14s2` for "V2/seed42 only":

```bash
TEST_SCRIPTS["14s2"]="experiments/test14.py"
TEST_OUTPUT_DIRS["14s2"]="outputs/test14_pre"
TEST_PGREP["14s2"]="test14.*--variant V2"
TEST_EXTRA_FLAGS["14s2"]="--mode=pre --variant V2 --phase-c-seeds 42 \
    --phase-c-num-epochs 50 --lr-scheduler-type linear --phase-c-decay-steps 600"
```

Then list it before bare `14` in `_find_running_test` (more-specific pgrep
wins) and add it to the validation message. `tresume 14s2` / `tpause` /
`tstatus` work without further changes.

### Adding a new seed

```bash
~/miniforge3/envs/paramem/bin/python experiments/test14.py \
  --model mistral --mode pre --resume \
  --phase-c-seeds 42 7 1337 9999 ...
```

Existing seeds with `C_done.json` are skipped; only `seed9999` runs.

### Adding extended variants on top (V3_extended/V4-V8)

```bash
~/miniforge3/envs/paramem/bin/python experiments/test14.py \
  --model mistral --mode pre --resume \
  --reuse-phase-a-from outputs/test14_pre/mistral/<run_ts>
```

This persists `reuse_phase_a_from` into `run_config.json`, so subsequent
`tresume 14` calls preserve the extended-variant scope automatically.

## Reproducing failed configurations (historical, optional)

`constant_with_warmup` (rejected 2026-05-03 — oscillates around the SimHash
threshold because LR=peak forever after warmup):

```bash
~/miniforge3/envs/paramem/bin/python experiments/test14.py \
  --model mistral --mode pre --resume \
  --lr-scheduler-type constant_with_warmup --phase-c-decay-steps 0
```

`decay=300` (rejected 2026-05-04 — half of the V3 reference's 600-step
trajectory; LR=0 from e15 freezes the model at fill≈0.35):

```bash
~/miniforge3/envs/paramem/bin/python experiments/test14.py \
  --model mistral --mode pre --resume \
  --lr-scheduler-type linear --phase-c-decay-steps 300
```

Stock linear with budget-coupled decay (the original bug — per-step LR shifts
with `num_train_epochs`):

```bash
~/miniforge3/envs/paramem/bin/python experiments/test14.py \
  --model mistral --mode pre --resume \
  --lr-scheduler-type linear --phase-c-decay-steps 0
```

Auto-migration handles each on re-launch. See `benchmarking.md` "Multi-seed
replication" for the numbers.

## Cleanup

Once the apples-to-apples results are recorded in `benchmarking.md`, the
migrated backups can be deleted:

```bash
rm -rf outputs/test14_pre/mistral/<ts>/V*/C/seed*_linear_*
rm -rf outputs/test14_pre/mistral/<ts>/V*/C/seed*_constant_with_warmup_*
rm -rf outputs/test14_pre/mistral/<ts>/V*/C/seed*_b50  # legacy
```

The numbers stay in `benchmarking.md` as the diagnostic narrative; the
adapter checkpoints + epoch logs are not load-bearing.
