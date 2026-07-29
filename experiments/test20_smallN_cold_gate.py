"""Test 20: Small-N Cold Indexed-Key Recall Gate (production recipe).

Research question
------------------
All existing small-N cold-adapter recall failure evidence is Qwen /
procedural / near-duplicate content / lr 5e-5 (Test 19) and may not
transfer to the PRODUCTION recipe (Mistral 7B / episodic / lr 1e-4). This
test is a decisive gate, parameterized over N (key count) and epoch
budget: does a cold-init indexed-key adapter recall at ~1.0 when trained
with the exact production episodic recipe? The ORIGINAL failure this test
must reproduce is **N=3 at 30 epochs (60 optimizer steps)** — the N=12
condition (180 steps) is a separate, less severe arm and must not be
conflated with it.

Arms (parameterized via ``--n-entries`` / ``--epochs`` / ``--accum``)
----------------------------------------------------------------------
N synthetic keys (default 12, a strict prefix of the fixed 12-fact list
for smaller N), Mistral 7B, EPISODIC production recipe (rank 8, alpha 16,
lr 1e-4, attention-only target_modules), COLD LoRA-zero init. The epoch
budget and ``gradient_accumulation_steps`` are DERIVED, not fixture
fields: ``paramem.utils.config.budget_for(n_entries)`` — the SAME
per-fold funnel production training calls on every fold
(``ConsolidationLoop._train_tier_adapter``) — returns ``(epochs, accum,
lr_decay_steps)`` for the resolved key count; ``--epochs`` / ``--accum`` /
``--lr-decay-steps`` override the derived value explicitly when passed,
otherwise the derived default applies (see "Recipe fidelity" below).
``epochs * _steps_per_epoch(N, batch_size, accum)`` optimizer steps
(``batch_size`` from the loaded fixture, today 1; derived, never
hardcoded — see ``_expected_optimizer_steps``). Run at 3 seeds (0, 1, 2 by
default; override with ``--seeds`` — e.g. ``--seeds 42`` for a single
production-training-seed run); per-seed recall reported, plus the mean.
Two decisive arms:

  * ``--n-entries 3 --epochs 30 --accum 2``  -> 60 optimizer steps (the
    ORIGINAL failure condition to reproduce; BOTH flags override
    ``budget_for(3)``'s derived 80-epoch/accum-1 default explicitly — the
    original failure predates ``budget_for``'s per-N accum derivation,
    which gives accum=1 for N=3 (the ``<16`` bucket); omitting ``--accum 2``
    here reproduces a DIFFERENT arm, 90 steps).
  * ``--n-entries 3 --epochs 180 --accum 2`` -> 360 optimizer steps (same
    N, 6x the step budget — isolates whether more steps rescues the
    small-N arm).

The bare-default invocation (``--model mistral`` only, N=12) derives its
budget from ``budget_for(12)`` (the ``<16`` bucket: 80 epochs, accum 1) —
the historical N=12/30-epoch/accum-2/180-step arm this script originally
shipped with requires ``--epochs 30 --accum 2`` explicitly to reproduce
(N=12 is also in the ``<16`` bucket, so BOTH fields need pinning, not just
epochs).

Recipe fidelity
----------------
Loaded via ``load_server_config("tests/fixtures/server.yaml")`` — never
``load_config()`` or ``configs/server.yaml.example`` (project rule). The
fixture's ``episodic_adapter_config`` supplies rank/alpha/lr/target_modules
verbatim; its ``training_config`` supplies batch_size=1,
max_seq_length=1024, warmup_steps=0, lr_scheduler_type="linear",
weight_decay=0.1, gradient_checkpointing=True, max_grad_norm=1.0 —
unchanged. ``num_epochs``, ``gradient_accumulation_steps``, and
``lr_decay_steps`` are NOT read from the fixture at all: production
derives all three per-fold from ``paramem.utils.config.budget_for(n_keys)``
(the unconditional funnel call in
``ConsolidationLoop._train_tier_adapter``), and this harness calls the
SAME function on the resolved ``n_entries`` to get the arm's default
budget — the fixture stopped carrying ``gradient_accumulation_steps``/
``num_epochs`` fields entirely once production's derivation shipped, so
treating them as fixture values would be fiction. Three fields are
therefore always set IN CODE, from ``budget_for(n_entries)`` unless the
matching CLI flag overrides them explicitly:

  * ``num_epochs``: ``budget_for(n_entries)[0]``, or ``--epochs`` when
    explicit.
  * ``gradient_accumulation_steps``: ``budget_for(n_entries)[1]``, or
    ``--accum`` when explicit.
  * ``lr_decay_steps``: ``budget_for(n_entries)[2]`` (``None`` for every
    bucket in today's ``_BUDGET_TABLE``), or ``--lr-decay-steps`` when
    explicit.

A fourth field is always forced regardless of any derived value or CLI
flag: ``recall_early_stopping=False`` — the fixture ships True, which
would truncate the run on 100% recall, making the expected step count
fiction.

``_expected_optimizer_steps`` is always called with the SAME resolved
epochs/accum/batch_size the run actually trains with — never a hardcoded
module constant — so the derived expected-step count and the
post-cfg-load Step 7 canary in ``_run_seed`` can never drift from each
other by construction.

The realized optimizer-step count is captured directly from HF Trainer's
``TrainerState.global_step`` via a local callback (``train_adapter`` does
not surface it in its returned metrics dict) and asserted against the
expected value.

Hard assertions (all written into results.json; fail loud if violated)
------------------------------------------------------------------------
1. Realized optimizer steps == expected steps (captured from
   ``TrainerState``, not assumed from the config).
2. ``training_config.recall_early_stopping is False`` at the moment of the
   ``train_adapter`` call.
3. LoRA-B Frobenius norm is ZERO immediately before training (cold arm —
   proves cold init) or NON-ZERO immediately before training (warm arm —
   ``--warm-from`` or ``--donor-init``, both feed the same
   ``donor_scratch_dir`` mechanism — proves the donor copy landed), and
   NON-ZERO after training in both arms (proves the adapter actually
   moved). Norm computation is
   ``paramem.models.loader.lora_b_frobenius_norm``.
4. (Warm arm only — ``--warm-from`` or ``--donor-init``) The donor
   adapter's LoRA-B Frobenius norm is bit-identical immediately before and
   immediately after each seed's ``train_adapter`` call (donor
   immutability), and the trainable adapter's name is never a live tier
   name (``episodic``/``semantic``/``procedural``).

The synthetic key set
-----------------------
Shape-matches the real fold WITHOUT copying the owner's real facts (already
trained into the live episodic adapter — personal data). Shared subject
``speaker0`` (lowercase — the project's ONE speaker-id form), up to 12
DIVERSE predicates in the shape of the real graph (profession, worked at,
has skill, speaks language, studied at, lives in, enjoys, prefers, married
to, has child, authored, led), fictional/anonymized objects only,
``graph<N>`` key ids (production episodic/semantic key prefix). Requesting
N < 12 takes the first N of this fixed list (a strict prefix — the N=3 arm
is a strict prefix of the N=12 arm, so smaller arms stay comparable and
the 3 keys have DISTINCT predicates, never near-duplicates). Requesting
N > 12 fails loud. Built via the same entry/prompt path production uses:
``paramem.memory.entry.format_entry_training`` (the live recall template
at ``entry.py:113``, ``"Recall the fact stored under key '{key}'."``) —
mirroring the production call path at
``paramem.training.consolidation.py::ConsolidationLoop._train_tier_adapter``
(~line 7086: ``format_entry_training(entries, tokenizer, max_length=1024)``
-> ``IndexedDataset`` -> ``train_adapter``).

The real key set (``--entries-json``)
----------------------------------------
The synthetic set above does NOT reproduce the production failure — the
diverse-predicate shape was a hypothesis, not the actual failing input.
``--entries-json FILE`` loads an explicit entry list (``[{"key",
"subject", "predicate", "object"}, ...]``) and REPLACES the synthetic
generator entirely; ``--n-entries`` is then implied by the file length
(passing a conflicting ``--n-entries`` fails loud — see ``main()``).
Loaded entries flow through the exact same ``format_entry_training`` call
as the synthetic set — no special-casing. The canonical fixture is
``experiments/fixtures/real3_interim_failure.json``: the three episodic
triples (``graph156``/``graph157``/``graph158``, all subject ``speaker0``)
that actually failed in production (SimHash confidence 0.61-0.70, the
adapter echoing the CORRECT key but ANOTHER key's object — a content
permutation among the three), read verbatim from
``data/ha/debug/episodic/cycle_5/run_20260711T231729Z_1731fa/recall_probes/
disk_verify_episodic_interim_20260712T0000_verify.json``.

Warm start (``--warm-from``)
--------------------------------
``--warm-from ADAPTER_DIR`` warm-starts the trainable adapter from a donor
adapter's LoRA weights instead of LoRA-zero, testing whether prior
knowledge (e.g. the owner's live 140-key episodic adapter) rescues the
small-N recall failure. Mechanism (this script's ``warmstart`` arm):
create a fresh ``donor``-named adapter loaded from *ADAPTER_DIR*, create
the trainable adapter fresh (LoRA-zero), ``copy_adapter_weights(model,
src="donor", dst=<trainable>)`` BEFORE ``train_adapter`` so the
staging+promote path starts from donor weights.

**Donor immutability.** *ADAPTER_DIR* may be the owner's LIVE episodic
adapter — corrupting it is real data loss. The directory is
``shutil.copytree``'d into the run's scratch dir once per run
(``<run_dir>/donor_scratch/``, reused across seeds and across
``--resume``); the donor is loaded ONLY from that copy via
``paramem.models.loader._adapter_slot_for_load`` (transparently decrypts
age-encrypted weights into an anonymous memfd — the same mechanism
``test16_repair_sweep.py`` / ``test18_probe_batching.py`` already use for
adapter-slot loading; no plaintext weight bytes touch disk). The original
*ADAPTER_DIR* is never opened for anything but the ``copytree`` read. The
donor's LoRA-B Frobenius norm is captured immediately before and
immediately after each seed's ``train_adapter`` call and asserted
bit-identical — a silent donor mutation would invalidate the arm. The
trainable adapter name (``episodic_<arm>_seed<N>``) is asserted to never
collide with a live tier name (``episodic``/``semantic``/``procedural``).

Donor-init (``--donor-init`` / ``--donor-checkpoint``) — budget/donor validation
---------------------------------------------------------------------------------
``--donor-init`` is a SECOND way to populate the exact same
``donor_scratch_dir`` mechanism ``--warm-from`` uses above (mutually
exclusive with it — both resolve one donor source, never two): instead of
an owner-supplied adapter directory, the donor is
``paramem.training.donor.donor_entries(DONOR_DEFAULT_SEED,
DONOR_MIN_ENTRIES)`` (147 synthetic crowded-cluster keys —
``donor_entries`` returns whole 21-entry blocks, so the requested 128
rounds up — the same seed+recipe pure function production donor building
uses — see ``paramem.training.donor``'s module docstring for why this
shape, not PerLTQA/longmemeval/diverse-predicate content, is the donor's
content source) trained through THIS SCRIPT'S OWN ``train_adapter`` call
path (``_build_donor_checkpoint`` — mirrors ``_run_seed``'s own
create/switch/train/probe/save sequence; deliberately NOT
``paramem.training.donor.build_donor``, which needs a live
``ConsolidationLoop`` this standalone experiment has no business
depending on) at ``budget_for(len(donor_entries))``'s derived epoch/accum
budget (147 entries -> the ``>=128`` bucket in
``paramem.utils.config._BUDGET_TABLE``: 30 epochs, accum 2) — the SAME
funnel derivation production's own donor build
(``paramem.training.donor.build_donor``) uses, never a hardcoded module
constant.

The donor builds ONCE and arms reuse it: ``--donor-checkpoint SLOT_DIR``
points at a prior ``--donor-init`` run's
``<run_dir>/donor_checkpoint/<ts>/`` slot and skips training entirely;
omitting it makes THIS run build (or, on ``--resume``/a repeat invocation,
reuse via ``<run_dir>/donor_build_done.json`` — the donor-build phase
marker, mirroring ``seed<N>_done.json``'s pattern) its own donor
checkpoint. Either way, the resolved checkpoint slot is
``shutil.copytree``'d into ``<run_dir>/donor_scratch/`` exactly once
(``<run_dir>/donor_source.json`` records the resolved source path + its
weights SHA-256 so later seeds and ``--resume`` never re-derive it), and
every downstream mechanism — ``_run_seed``'s Step 1b load, Hard Assertions
#3/#4, donor immutability — is IDENTICAL to the ``--warm-from`` arm.

``--lr-decay-steps N`` pins ``TrainingConfig.lr_decay_steps`` so the LR
scheduler's decay window is comparable across arms run at different
``--epochs`` (the approved decay-pinned validation protocol pins decay for
the 50-epoch bucket-2 arm and the donor-init arms; omitting the flag uses
``budget_for(n_entries)``'s derived value instead — ``None`` for every
bucket in today's ``_BUDGET_TABLE``, i.e. ``create_scheduler``'s no-op
passthrough, decay derived from ``len(dataloader) * num_epochs``, HF's
default). This override applies to the ARM's own target-fact training
only — ``_build_donor_checkpoint`` always derives the donor's own
``lr_decay_steps`` from ``budget_for(len(donor_entries))`` regardless of
this flag (see that function's docstring).

``--accum N`` overrides ``TrainingConfig.gradient_accumulation_steps``,
threaded into ``base_training_config`` the same way as ``--epochs`` and
``--lr-decay-steps``. Omitting the flag uses ``budget_for(n_entries)``'s
derived value instead of a hardcoded module constant — the fixture no
longer carries a ``gradient_accumulation_steps`` field to fall back to
(see "Recipe fidelity" above). ``_expected_optimizer_steps`` is always
called with the SAME resolved accum value the run actually trains with,
so the derived expected-step count and the post-cfg-load Step 7 canary in
``_run_seed`` can never drift from each other. Like ``--lr-decay-steps``,
this override applies to the ARM's own target-fact training only —
``_build_donor_checkpoint`` always derives the donor's own
``gradient_accumulation_steps`` from ``budget_for(len(donor_entries))``
regardless of this flag (the donor's one-time build must reflect ITS OWN
key count's bucket, not whichever arm happens to trigger it).

**Confound (recorded, not eliminated):** the donor's block-0 is
bit-identical to the fixed 21-key fixture, so a real-production-key arm's
``--entries-json`` set typically overlaps the donor's own keys entirely —
the donor may pre-install key -> subject/predicate scaffolding (with a
DIFFERENT, donor-fictional object) for exactly the keys this arm re-trains,
so a donor-arm's recall uplift measures the overlapping-band store, not
generalization to fresh keys (>= 201, outside the donor's reserved band).
The exact-match rate metric itself is unaffected; only its ATTRIBUTION is —
``results.json``'s ``donor_key_overlap`` records the intersection count and
the donor's own (different) objects for those keys on every donor-init
seed so this is never silently assumed away.

``--donor-build-smoke`` (procedural-topology GPU feasibility/cost probe)
-------------------------------------------------------------------------
A SEPARATE, standalone mode (a GPU feasibility/cost measurement for
building a donor checkpoint at the PROCEDURAL topology) dispatched by
``main()`` before any of the above arm-configuration logic runs; mutually
exclusive with every other flag except ``--model``/``--resume``. Builds ONE
donor checkpoint at the PROCEDURAL topology — rank/alpha/target_modules
from ``tests/fixtures/server.yaml``'s ``procedural_adapter_config`` via
``load_server_config`` (never hardcoded) — reusing
``_build_donor_checkpoint`` unchanged and
``_build_or_reuse_own_donor_checkpoint`` (the resume-aware build-once
machinery extracted from ``_resolve_donor_source``'s cases 2/3 — one shared
implementation; ``--donor-init`` reaches it via ``_resolve_donor_source``,
``--donor-build-smoke`` calls it directly since it never accepts an
external ``--donor-checkpoint``). This build trains at
``paramem.training.donor.DONOR_RECIPE_LEARNING_RATE``/
``DONOR_RECIPE_DROPOUT`` — derived from those constants, never hand-copied
— matching production ``build_donor``'s recipe fidelity unconditionally
(the same recipe every ``_build_donor_checkpoint`` call now applies, for
``--donor-init`` too). Then cold-seeds a FRESH procedural adapter
from the built checkpoint via the SAME strict ``copy_adapter_weights`` call
(raises loud on any parameter-set mismatch) the ``--warm-from``/
``--donor-init`` arms already use, loaded via the SAME
``_adapter_slot_for_load`` + ``PeftModel.from_pretrained``
pattern ``_run_seed``'s Step 1b uses. No recall evaluation runs on the
seeded adapter and production ``paramem.training.donor.build_donor`` is
never exercised (this arm measures GPU feasibility/cost only — wall
time, mean seconds/optimizer step, ``torch.cuda.mem_get_info`` sampled
before load / after load / after build / after seed, and ``torch.cuda.
max_memory_allocated``/``max_memory_reserved`` for the build phase — WSL2's
``nvidia-smi`` is VRAM-blind, ``mem_get_info`` is authoritative). Resumable
at the build/seed phase boundary: a rerun with the build phase's own
``build_results.json`` already present skips straight to the seed phase; a
rerun with the seed phase's own ``donor_build_smoke_seed_done.json`` marker
already present is a no-op. ``--resume`` into an existing run dir fails
loudly, before either phase runs, if the CURRENT fixture's
``procedural_adapter_config`` topology disagrees with the run dir's own
recorded ``smoke_config.json`` (mirrors ``main()``'s ``run_config.json``
mismatch guard). See :func:`_run_donor_build_smoke` /
:func:`_main_donor_build_smoke` for the full mechanism::

    setsid nohup python \\
        experiments/test20_smallN_cold_gate.py --model mistral \\
        --donor-build-smoke \\
        >outputs/test20_donor_build_smoke_procedural.log 2>&1 &

Metric
------
``paramem.training.recall_eval.evaluate_indexed_recall`` (handles the
gradient-checkpointing/generate() dance and adapter switching). SimHash
confidence >= 0.75 (module default), deterministic generation
(``do_sample=False``, i.e. temperature 0.0). Full per-key ``raw_output``,
``confidence``, and ``failure_reason`` saved to results.json.

Mechanism probe (``--probe-before-training``)
-------------------------------------------------
Tests WHY warm-starting fixes small-N recall: hypothesis is that a donor
adapter supplies the shared OUTPUT FORMAT (well-formed JSON, correct key
echoed, correct subject), leaving the small training budget free to learn
the discriminative key -> object BINDING; a cold adapter must pay for both,
and the format signal dominates the tiny gradient budget. When
``--probe-before-training`` is set, ``_run_recall_probe`` (the same helper
Step 11 uses post-training — no ad-hoc probe) is called ONCE per seed
immediately after the trainable adapter is created (cold: LoRA-zero) /
warm-copied (donor weights) — Step 4b in ``_run_seed`` — strictly BEFORE
``train_adapter`` is called. The full result (``rate``, ``exact_count``,
``total``, and ``per_key`` with ``key``, ``confidence``, ``failure_reason``,
and verbatim ``raw_output``) is saved to ``results.json`` under the
top-level key ``pre_training_probe`` (``null`` when the flag is off). The
probe never perturbs the trained adapter itself: ``torch.get_rng_state()``
/ ``torch.cuda.get_rng_state_all()`` are snapshotted immediately before the
probe's ``generate()`` calls and restored immediately after, so the
realized training run (data order, dropout, etc.) is bit-for-bit identical
with the flag on vs off. Expected reading: a WARM pre-training probe that
emits well-formed JSON echoing the CORRECT key but a WRONG (donor-fact)
object supports the hypothesis; malformed/garbage warm output weakens it.
The COLD pre-training probe is the control — LoRA-B is exactly zero
(Hard Assertion #3), so the adapter is a literal no-op and should behave
like the bare base model.

Infrastructure
---------------
Reuses ``experiments/utils/test_harness.py`` (``BENCHMARK_MODELS``,
``model_output_dir``, ``load_model_and_config``, ``IndexedDataset``,
``save_results``, ``setup_logging``) and
``experiments/utils/gpu_guard.py::acquire_gpu``.

Single base-model load; per seed the model is unwrapped
(``model = model.base_model.model``) before ``create_adapter`` — never
``delete_adapter`` then ``create_adapter`` (CLAUDE.md). Each seed gets its
own adapter name (``episodic_<arm>_seed<N>``) so residual ``lora.Linear``
modules from prior seeds never collide. ``torch.manual_seed(seed)`` is set
immediately before ``create_adapter`` — production LoRA init is unseeded
(``paramem/models/loader.py:486`` omits ``init_lora_weights``) — and the
same seed is threaded through ``TrainingConfig.seed`` (HF Trainer's data
order).

Pause / resume
---------------
``~/.training_pause`` gates seed boundaries. ``--resume`` auto-finds the
latest run dir for the resolved ``--arm`` and skips seeds whose
``seed<N>_done.json`` marker already exists (each arm's runs live under a
dedicated, arm-scoped output subtree, so ``--resume`` never crosses arms).
``wait_for_cooldown`` runs between seeds (not before the first).

Daemonised launch (survives Claude exit)
------------------------------------------
All N=3 examples below pin ``--accum 2`` explicitly — ``budget_for(3)``
derives accum=1 (the ``<16`` bucket), so reproducing these historical
step counts requires the override; omitting ``--accum 2`` runs a
DIFFERENT (still valid, just differently-labeled) arm.

The original synthetic failure condition (N=3, 30 epochs, 60 steps)::

    setsid nohup python \\
        experiments/test20_smallN_cold_gate.py --model mistral \\
        --n-entries 3 --epochs 30 --accum 2 \\
        >outputs/test20_n3_e30.log 2>&1 &

The step-budget-rescue arm (N=3, 180 epochs, 360 steps)::

    setsid nohup python \\
        experiments/test20_smallN_cold_gate.py --model mistral \\
        --n-entries 3 --epochs 180 --accum 2 \\
        >outputs/test20_n3_e180.log 2>&1 &

The REAL 3-triple production failure, cold (60 steps)::

    setsid nohup python \\
        experiments/test20_smallN_cold_gate.py --model mistral \\
        --entries-json experiments/fixtures/real3_interim_failure.json \\
        --epochs 30 --accum 2 \\
        >outputs/test20_real3_cold.log 2>&1 &

The same real triples, warm-started from a donor adapter::

    setsid nohup python \\
        experiments/test20_smallN_cold_gate.py --model mistral \\
        --entries-json experiments/fixtures/real3_interim_failure.json \\
        --epochs 30 --accum 2 --warm-from /path/to/donor_adapter_dir \\
        >outputs/test20_real3_warm.log 2>&1 &

Resume (auto-detects the arm's own output subtree; ``--entries-json`` /
``--accum`` / ``--warm-from`` must be repeated identically so the resolved
``--arm`` matches)::

    python experiments/test20_smallN_cold_gate.py --model mistral \\
        --n-entries 3 --epochs 30 --accum 2 --resume

The mechanism probe (``--probe-before-training``), real 3 triples, warm vs
cold, 3 seeds each. WARM decrypts the donor adapter, so
``PARAMEM_DAILY_PASSPHRASE`` must be exported first::

    export PARAMEM_DAILY_PASSPHRASE=... && setsid nohup python \\
        experiments/test20_smallN_cold_gate.py --model mistral \\
        --entries-json experiments/fixtures/real3_interim_failure.json \\
        --epochs 30 --accum 2 --warm-from data/ha/adapters/episodic/20260710-224008 \\
        --probe-before-training \\
        >outputs/test20_real3_warm_probe.log 2>&1 &

    setsid nohup python \\
        experiments/test20_smallN_cold_gate.py --model mistral \\
        --entries-json experiments/fixtures/real3_interim_failure.json \\
        --epochs 30 --accum 2 --probe-before-training \\
        >outputs/test20_real3_cold_probe.log 2>&1 &

Budget/donor validation arms (exact-21 production keys). ``EXACT21_JSON``
is the exact-21 production-key fixture (NEVER copied into
this repository — pass its actual path)::

    EXACT21_JSON=/path/to/interim_exact21_20260725.json

Donor build + donor-init at 30 epochs (330 steps; builds the donor
checkpoint under ``<run_dir>/donor_checkpoint/`` since no
``--donor-checkpoint`` is given)::

    setsid nohup python \\
        experiments/test20_smallN_cold_gate.py --model mistral \\
        --entries-json "$EXACT21_JSON" \\
        --epochs 30 --donor-init \\
        >outputs/test20_real21_donor_e30.log 2>&1 &

Cold bucket-2 arm at 50 epochs (550 steps), decay pinned for comparability::

    setsid nohup python \\
        experiments/test20_smallN_cold_gate.py --model mistral \\
        --entries-json "$EXACT21_JSON" \\
        --epochs 50 --lr-decay-steps 550 \\
        >outputs/test20_real21_cold_e50.log 2>&1 &

Donor-init at 50 epochs (550 steps), reusing the checkpoint the first
invocation above built (``--donor-checkpoint`` points at its
``donor_checkpoint/<ts>/`` slot — see that run's ``donor_build_done.json``
for the exact path)::

    DONOR_SLOT=outputs/test20_smallN_cold_gate/real21_donor_s330/mistral/<ts>/donor_checkpoint/<ts>
    setsid nohup python \\
        experiments/test20_smallN_cold_gate.py --model mistral \\
        --entries-json "$EXACT21_JSON" \\
        --epochs 50 --lr-decay-steps 550 --donor-init \\
        --donor-checkpoint "$DONOR_SLOT" \\
        >outputs/test20_real21_donor_e50.log 2>&1 &

Data safety
-----------
Results written to unique timestamped, arm-scoped paths via
``model_output_dir`` — never overwritten. Every result file includes full
per-key ``raw_output``. The donor scratch copy (warm arm) lives under
``<run_dir>/donor_scratch/`` — inside the same output tree, never the
donor's original path. ``--donor-init``'s own checkpoint
(``<run_dir>/donor_checkpoint/``), build marker
(``donor_build_done.json``), and source-provenance record
(``donor_source.json``) are all inside ``OUTPUT_BASE``
(``outputs/test20_smallN_cold_gate/``), which is entirely gitignored — the
synthetic donor content has no personal data to begin with, but the
convention is shared with the ``--entries-json`` real-key fixtures below.

A ``--entries-json`` file naming REAL personal facts (e.g. the exact-21
production-key fixture used for the bucket-2/donor-init validation arms) must never
be copied into this repository, tracked or gitignored — pass its path
directly at invocation; results derived from it land under ``outputs/``,
which is gitignored, same as every other result file here.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import TrainerCallback  # noqa: E402

from experiments.utils.production import (  # noqa: E402
    DONOR_RECIPE_ID,
    budget_for,  # noqa: E402
    build_manifest_for,
    donor_slot_valid,
    lora_shape_fields,
    triples_hash,
)
from experiments.utils.test_harness import (  # noqa: E402
    BENCHMARK_MODELS,
    IndexedDataset,
    load_model_and_config,
    model_output_dir,
    save_results,
    setup_logging,
)
from paramem.memory.entry import build_registry, format_entry_training  # noqa: E402
from paramem.models.loader import (  # noqa: E402
    _adapter_slot_for_load,
    atomic_save_adapter,
    copy_adapter_weights,
    create_adapter,
    lora_b_frobenius_norm,
    switch_adapter,
    unload_model,
)
from paramem.server.config import load_server_config  # noqa: E402
from paramem.training.donor import (  # noqa: E402
    DONOR_BUILD_ADAPTER_NAME,
    DONOR_DEFAULT_SEED,
    DONOR_META_FILENAME,
    DONOR_MIN_ENTRIES,
    # DONOR_RECIPE_DROPOUT / DONOR_RECIPE_LEARNING_RATE: _build_donor_checkpoint
    # derives its donor's own training hyperparameters from these two constants
    # unconditionally (never hand-copies the literals) -- see that function's
    # docstring and the module docstring's "--donor-build-smoke" section.
    DONOR_RECIPE_DROPOUT,
    DONOR_RECIPE_LEARNING_RATE,
    donor_entries,
)
from paramem.training.recall_eval import evaluate_indexed_recall  # noqa: E402
from paramem.training.trainer import train_adapter  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_BASE = project_root / "outputs" / "test20_smallN_cold_gate"
PAUSE_FILE = Path.home() / ".training_pause"
FIXTURE_CONFIG_PATH = project_root / "tests" / "fixtures" / "server.yaml"

SEEDS = (0, 1, 2)

# Production episodic probe batch size (recall_probe_batch_size in
# tests/fixtures/server.yaml / configs/server.yaml.example).
RECALL_PROBE_BATCH_SIZE = 16

# Default arm: N synthetic keys. The epoch/accum/lr-decay budget for ANY
# N (arm or donor build) is DERIVED per fold via
# paramem.utils.config.budget_for(n_entries) — the SAME function
# production's per-fold funnel calls unconditionally (see module
# docstring's "Arms" / "Recipe fidelity" sections) — never a hardcoded
# module constant. --epochs / --accum / --lr-decay-steps override the
# derived default explicitly when passed. The decisive arm for the
# ORIGINAL failure is N=3, 30 epochs, accum=2 (--n-entries 3 --epochs 30
# --accum 2 -> 60 steps, an explicit override of budget_for(3)'s derived
# 80-epoch/accum-1 default on BOTH fields).
DEFAULT_N_ENTRIES = 12

# Disk safety threshold (matches test16/test19 convention: free-space, not
# total-usage).
DISK_HEADROOM_BYTES = 5 * 1024**3

# Required fields for each --entries-json entry dict.
_REQUIRED_ENTRY_KEYS = ("key", "subject", "predicate", "object")

# Name reserved for the (frozen, read-only) --warm-from / --donor-init donor
# adapter mounted into _run_seed via donor_scratch_dir.
DONOR_ADAPTER_NAME = "donor"

# Live production tier names — the trainable adapter must NEVER collide
# with one of these (donor-immutability guard; see the module docstring's
# "Donor immutability" section).
LIVE_TIER_NAMES = frozenset({"episodic", "semantic", "procedural"})

# ---------------------------------------------------------------------------
# --donor-init: build-your-own-donor constants (budget/donor validation arm)
# ---------------------------------------------------------------------------

# The ONE-TIME donor build's own epoch/accum/lr-decay budget is DERIVED via
# paramem.utils.config.budget_for(len(donor_entries)) inside
# _build_donor_checkpoint — never a hardcoded module constant (mirrors
# production's own donor build, paramem.training.donor.build_donor, which
# gets its budget from the SAME funnel call in
# ConsolidationLoop._train_tier_adapter). Independent of --epochs/--accum
# (which budget the ARM's own target-fact training, not the donor's).

# Sub-directory of the run dir holding a freshly-built donor checkpoint (only
# used when --donor-init is set and --donor-checkpoint is NOT — i.e. this run
# builds its own donor rather than reusing a prior run's).
DONOR_CHECKPOINT_DIRNAME = "donor_checkpoint"

# Phase marker for the donor-build phase (mirrors seed<N>_done.json's
# pattern — see _marker_path/_write_done_marker). Presence means the donor
# checkpoint referenced inside was fully trained and saved; a crash before
# this file is written must retrain the donor on retry (no partial-epoch
# resume for the donor build, matching this script's existing per-seed
# granularity — a crash mid-seed already requires retraining that whole
# seed).
DONOR_BUILD_MARKER_FILENAME = "donor_build_done.json"
DONOR_BUILD_PROVENANCE_FILENAME = "test20_donor_build_provenance.json"
"""Per-slot record of how THIS script trained a donor (steps, wall time).
Distinct from ``donor_meta.json``, which records what the donor IS in the
production schema -- one filename per schema, so a production donor slot
passed to ``--donor-checkpoint`` is unambiguous."""


# ---------------------------------------------------------------------------
# Step-budget derivation (single source of truth for the CLI-facing value)
#
# LoRA training budgets epochs, never optimizer steps (there is no
# production step-budget floor to force off here — TrainingConfig has no
# such field). ``_steps_per_epoch`` is a local, experiment-only helper
# mirroring HF Trainer's own per-epoch optimizer-step count
# (``transformers/trainer.py``, installed version): a NESTED ceiling
# division — ``ceil(ceil(n_examples / batch_size) / gradient_accumulation_steps)``
# — not the algebraically-equal flat form. Kept local rather than importing
# a production helper (CLAUDE.md: experiments may carry their own helpers;
# a production module must not be kept alive just to serve one experiment).
# ---------------------------------------------------------------------------


def _steps_per_epoch(n_examples: int, batch_size: int, gradient_accumulation_steps: int) -> int:
    """Optimizer steps HF Trainer runs per epoch for this dataset/config.

    Mirrors HF's nested ceiling division exactly (dataloader length, then
    batches-per-optimizer-step), floored at 1 (HF never reports zero steps
    for a non-empty dataloader).
    """

    def _ceil_div(numerator: int, denominator: int) -> int:
        return -(-numerator // denominator)

    per_epoch_batches = _ceil_div(n_examples, batch_size)
    return max(_ceil_div(per_epoch_batches, gradient_accumulation_steps), 1)


def _expected_optimizer_steps(n_entries: int, epochs: int, accum: int, batch_size: int) -> int:
    """Derive the total optimizer-step count for a resolved per-fold training budget.

    Uses ``_steps_per_epoch`` with the SAME ``epochs`` / ``accum`` /
    ``batch_size`` the run actually trains with — never a hardcoded module
    constant. Callers resolve ``epochs``/``accum`` from
    ``paramem.utils.config.budget_for(n_entries)`` (overridden by
    ``--epochs``/``--accum`` when explicit) and ``batch_size`` from the
    loaded fixture's ``TrainingConfig.batch_size`` before calling this
    function, so the result and ``_run_seed``'s Step 7 canary (which
    recomputes the identical formula from the actual ``TrainingConfig``
    used to train) can never drift from each other by construction.

    Args:
        n_entries: Number of keys in the arm (or donor build).
        epochs: Resolved epoch budget.
        accum: Resolved ``gradient_accumulation_steps``.
        batch_size: Resolved ``batch_size`` (today always 1 — see
            ``paramem.utils.config._BUDGET_TABLE``'s module comment).

    Returns:
        Total optimizer steps: ``_steps_per_epoch(n_entries, batch_size, accum) * epochs``.
    """
    spe = _steps_per_epoch(n_entries, batch_size, accum)
    return spe * epochs


# ---------------------------------------------------------------------------
# Cooldown helper (mirrors test19_neardup_procedural.py:254-275 exactly)
# ---------------------------------------------------------------------------


def _wait_for_cooldown(target: int = 52) -> None:
    """Block until GPU temperature drops below *target* degC.

    Shells out to gpu-cooldown.sh. Returns instantly if the GPU is already
    cool. Falls back to a 60-second sleep if the script is unavailable.

    Args:
        target: Temperature threshold in degC (default 52, everyday working).
    """
    try:
        subprocess.run(
            [
                "bash",
                "-c",
                f"source ~/.local/bin/gpu-cooldown.sh && wait_for_cooldown {target}",
            ],
            check=True,
            timeout=600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("Cooldown script failed (%s), falling back to 60s sleep", e)
        time.sleep(60)


def _check_pause(label: str) -> None:
    """Raise SystemExit cleanly if the pause file is present.

    Args:
        label: Human-readable description of where the pause occurred.
    """
    if PAUSE_FILE.exists():
        logger.warning("Pause file detected at %s — halting cleanly.", label)
        raise SystemExit(f"Training paused at {label}")


# ---------------------------------------------------------------------------
# The synthetic key set (up to 12 keys, strict prefix for N < 12)
# ---------------------------------------------------------------------------


def _build_entries(n_entries: int) -> list[dict]:
    """Generate the first *n_entries* synthetic keys (strict prefix of the fixed 12-fact list).

    Shape-matches the real episodic fold: one subject (``speaker0``, the
    project's ONE lowercase speaker-id form), DIVERSE predicates in the
    shape of the real graph, fictional/anonymized objects only,
    ``graph<N>`` key ids (production episodic/semantic key prefix, see
    ``paramem.memory.entry.assign_keys``). Deliberately NOT near-duplicate
    content — that is the Qwen/procedural confound (Test 19) this test is
    designed to rule out as a separate variable. Takes the first
    *n_entries* of the fixed 12-fact list below so every arm is a strict
    prefix of every larger arm (e.g. the N=3 arm is a strict prefix of the
    N=12 arm) — arms stay comparable and smaller arms never introduce
    near-duplicate predicates.

    Args:
        n_entries: Number of keys to generate.

    Returns:
        List of *n_entries* ``{key, subject, predicate, object}`` dicts,
        keys ``graph1``..``graph<n_entries>``.

    Raises:
        ValueError: If *n_entries* exceeds the fixed fact list length.
    """
    facts = [
        ("profession", "structural engineer"),
        ("worked at", "Meridian Robotics"),
        ("has skill", "underwater welding"),
        ("speaks language", "Finnish"),
        ("studied at", "Kestrel Polytechnic"),
        ("lives in", "Port Elyria"),
        ("enjoys", "sea kayaking"),
        ("prefers", "window seats"),
        ("married to", "Dana Voss"),
        ("has child", "Wren"),
        ("authored", "The Salt Line"),
        ("led", "the harbor restoration project"),
    ]
    if n_entries > len(facts):
        raise ValueError(
            f"--n-entries={n_entries} exceeds the fixed diverse-predicate fact list "
            f"({len(facts)} entries available). Add more DISTINCT predicates to "
            "_build_entries before requesting a larger N."
        )
    return [
        {
            "key": f"graph{i}",
            "subject": "speaker0",
            "predicate": predicate,
            "object": obj,
        }
        for i, (predicate, obj) in enumerate(facts[:n_entries], start=1)
    ]


# ---------------------------------------------------------------------------
# Real entry set (--entries-json), replacing the synthetic generator
# ---------------------------------------------------------------------------


def _load_entries_from_file(path: Path) -> list[dict]:
    """Load an explicit entry set from a JSON file (``--entries-json``).

    Bypasses ``_build_entries`` entirely — used to reproduce a REAL
    production failure with the exact triples that failed (see module
    docstring's "The real key set" section). Loaded entries are fed
    through the SAME ``format_entry_training`` production entry/prompt
    path as the synthetic set in ``_run_seed`` — no special-casing here or
    downstream.

    Args:
        path: Path to a JSON file containing a list of
            ``{"key", "subject", "predicate", "object"}`` dicts (e.g.
            ``experiments/fixtures/real3_interim_failure.json``).

    Returns:
        The parsed list of entry dicts, in file order.

    Raises:
        SystemExit: If the file is missing, not valid JSON, not a
            non-empty list, or any entry is missing a required field —
            fail loud rather than silently training on a malformed set.
    """
    if not path.is_file():
        raise SystemExit(f"--entries-json file not found: {path}")
    with open(path) as f:
        entries = json.load(f)
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"--entries-json must contain a non-empty JSON list: {path}")
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SystemExit(f"--entries-json entry {i} at {path} is not a JSON object: {entry!r}")
        missing = [k for k in _REQUIRED_ENTRY_KEYS if k not in entry]
        if missing:
            raise SystemExit(
                f"--entries-json entry {i} at {path} missing required field(s) {missing}: {entry}"
            )
    return entries


def _default_arm_label(n_entries: int, expected_steps: int, is_real: bool, mode: str) -> str:
    """Derive the default ``--arm`` label from the resolved run config.

    Preserves the script's original synthetic-cold naming
    (``cold_n{N}_s{steps}``) EXACTLY when neither ``--entries-json`` nor
    ``--warm-from``/``--donor-init`` is set, so ``--resume`` keeps finding
    runs launched before those flags existed (e.g. the completed
    ``outputs/test20_smallN_cold_gate/cold_n3_s60/`` run). Real
    (``--entries-json``) arms use ``real{N}_{mode}_s{steps}`` (e.g.
    ``real3_cold_s60`` / ``real3_warm_s60`` / ``real21_donor_s550``) so the
    label states both the dataset and the init condition explicitly.

    Args:
        n_entries: Resolved entry count (file length for
            ``--entries-json``, ``--n-entries``/default otherwise).
        expected_steps: Derived total optimizer-step count
            (``_expected_optimizer_steps``).
        is_real: True when ``--entries-json`` supplied the entry set.
        mode: One of ``"cold"`` (LoRA-zero), ``"warm"`` (``--warm-from`` an
            arbitrary donor adapter dir), or ``"donor"`` (``--donor-init`` —
            seeded from ``paramem.training.donor.donor_entries``, built or
            reused via ``--donor-checkpoint``). Byte-identical output to the
            prior ``warm: bool`` parameter for ``mode in ("cold", "warm")``
            — only ``"donor"`` is a new label.

    Returns:
        The default arm label string (overridden by an explicit ``--arm``).
    """
    if is_real:
        return f"real{n_entries}_{mode}_s{expected_steps}"
    if mode != "cold":
        return f"n{n_entries}_{mode}_s{expected_steps}"
    return f"cold_n{n_entries}_s{expected_steps}"


def _condition_label(mode: str, epochs: int) -> str:
    """Derive the descriptive (never letter-labeled) condition name for results.json.

    Args:
        mode: One of ``"cold"``, ``"warm"``, ``"donor"`` — see
            ``_default_arm_label``.
        epochs: The arm's ``--epochs`` budget.

    Returns:
        e.g. ``"donor-init 30ep"``, ``"cold 50ep"``,
        ``"warm-from-adapter 30ep"``.
    """
    labels = {"cold": "cold", "warm": "warm-from-adapter", "donor": "donor-init"}
    return f"{labels[mode]} {epochs}ep"


# ---------------------------------------------------------------------------
# Optimizer-step capture callback
# ---------------------------------------------------------------------------


class _StepCaptureCallback(TrainerCallback):
    """Captures the HF Trainer's realized ``global_step`` at train end.

    ``train_adapter`` (``paramem/training/trainer.py``) returns
    ``dict(result.metrics)`` — HF's computed metrics (``train_loss``,
    etc.) — but does not surface ``TrainerState.global_step``. Hard
    Assertion #1 needs the actual realized optimizer-step count, so this
    callback reads it directly from ``state`` at ``on_train_end``, the last
    point at which ``TrainerState`` reflects the completed run.
    """

    def __init__(self) -> None:
        self.global_step: int | None = None

    def on_train_end(self, args, state, control, **kwargs) -> None:  # noqa: ARG002
        """Record the final optimizer-step count.

        Args:
            args: HF ``TrainingArguments`` (unused).
            state: HF ``TrainerState`` — ``state.global_step`` is the
                realized optimizer-step count.
            control: HF ``TrainerControl`` (unused).
            **kwargs: Additional HF callback kwargs (unused).
        """
        self.global_step = int(state.global_step)


# ---------------------------------------------------------------------------
# Marker helpers (mirrors test19_neardup_procedural.py:841-884)
# ---------------------------------------------------------------------------


def _marker_path(run_dir: Path, seed: int) -> Path:
    """Return the done-marker path for *seed*.

    Args:
        run_dir: Run output directory (already arm-scoped — see
            ``model_output_dir(OUTPUT_BASE / arm, model_name)`` in
            ``main()``).
        seed: Seed value.

    Returns:
        Path to ``seed<N>_done.json``.
    """
    return run_dir / f"seed{seed}_done.json"


def _marker_exists(run_dir: Path, seed: int) -> bool:
    """Return True if the done-marker for *seed* already exists."""
    return _marker_path(run_dir, seed).exists()


def _write_done_marker(run_dir: Path, seed: int, summary: dict) -> None:
    """Write a done-marker JSON to signal that *seed* is complete.

    Args:
        run_dir: Run output directory.
        seed: Seed value.
        summary: Summary dict to embed in the marker for quick inspection.
    """
    marker = {
        "seed": seed,
        "timestamp": int(time.time()),
        **summary,
    }
    marker_file = _marker_path(run_dir, seed)
    with open(marker_file, "w") as f:
        json.dump(marker, f, indent=2)
    logger.info("Done marker written: %s", marker_file)


# ---------------------------------------------------------------------------
# Recall probe helper (single call path — reused pre- and post-training)
# ---------------------------------------------------------------------------


def _run_recall_probe(
    model,
    tokenizer,
    entries: list[dict],
    registry: dict,
    adapter_name: str,
) -> dict:
    """Run ``evaluate_indexed_recall`` with the gradient-checkpointing dance.

    The ONE probe call path in this script — used both for the post-training
    probe (Step 11) and the optional pre-training mechanism probe (Step 4b,
    ``--probe-before-training``) — so the CLAUDE.md gradient-checkpointing
    rule ("disable before ANY ``generate()`` call, re-enable after") is
    honoured identically in both places instead of being duplicated.
    ``evaluate_indexed_recall`` itself calls
    ``model.gradient_checkpointing_disable()`` internally (and switches to
    *adapter_name*) but does not re-enable it, so this wrapper owns the full
    disable/generate/enable cycle.

    Args:
        model: PeftModel carrying *adapter_name*.
        tokenizer: Tokenizer matching the model.
        entries: The N-key entry set (synthetic or ``--entries-json``).
        registry: SimHash registry built from *entries*.
        adapter_name: Adapter to probe (``evaluate_indexed_recall`` switches
            to it internally).

    Returns:
        The dict returned by ``evaluate_indexed_recall``: ``exact_count``,
        ``total``, ``rate``, ``mean_confidence``, and ``per_key`` (each
        entry carrying ``key``, ``confidence``, ``failure_reason``, and
        verbatim ``raw_output``, among other fields).
    """
    model.gradient_checkpointing_disable()
    recall = evaluate_indexed_recall(
        model=model,
        tokenizer=tokenizer,
        entries=entries,
        registry=registry,
        adapter_name=adapter_name,
        batch_size=RECALL_PROBE_BATCH_SIZE,
    )
    model.gradient_checkpointing_enable()
    return recall


# ---------------------------------------------------------------------------
# --donor-init: build-your-own-donor checkpoint (budget/donor validation arm)
# ---------------------------------------------------------------------------


def _build_donor_checkpoint(
    model,
    tokenizer,
    adapter_config,
    base_training_config,
    checkpoint_root: Path,
) -> tuple[object, Path, dict]:
    """Train ``paramem.training.donor.donor_entries`` through the SAME
    ``train_adapter`` call path this script's own seeds use, and persist the
    result as a standalone donor checkpoint under *checkpoint_root*.

    Deliberately does NOT go through ``paramem.training.donor.build_donor``
    (that helper trains via a live ``ConsolidationLoop._train_tier_adapter``
    — production server wiring this standalone experiment has no business
    depending on). Instead this mirrors ``_run_seed``'s own
    create/switch/train/probe/save sequence, on ``DONOR_MIN_ENTRIES`` (128,
    rounds up to 147 — ``donor_entries`` returns whole 21-entry blocks)
    synthetic keys generated by ``donor_entries(DONOR_DEFAULT_SEED,
    DONOR_MIN_ENTRIES)`` — the same seed+recipe pure function production
    donor building uses, so the triple set is bit-identical to what
    production would build. The epoch/accum/lr-decay budget for THIS
    build is derived from ``paramem.utils.config.budget_for(len(entries))``
    (147 entries -> the ``>=128`` bucket: 30 epochs, accum 2, lr_decay_steps
    None) — the SAME funnel derivation production's own donor build uses,
    never a hardcoded module constant.

    The donor's OWN training always runs at
    ``paramem.training.donor.DONOR_RECIPE_LEARNING_RATE`` /
    ``DONOR_RECIPE_DROPOUT`` (``dataclasses.replace(adapter_config,
    learning_rate=..., dropout=...)``) rather than *adapter_config*'s own
    ``learning_rate``/``dropout`` — matching production
    ``paramem.training.donor.build_donor``'s recipe-fidelity contract
    unconditionally (topology fields — rank/alpha/target_modules — are
    UNCHANGED, since ``copy_adapter_weights`` only requires topology
    equality, never hyperparameter equality).

    ``lr_decay_steps`` and ``gradient_accumulation_steps`` are ALWAYS
    derived from ``budget_for(len(entries))`` for the donor's own training,
    regardless of what *base_training_config* carries — an arm invoked with
    ``--donor-init --lr-decay-steps 550 --accum 1`` would otherwise leak
    that arm-comparability override into the donor build, changing its
    realized optimizer-step count and decay window away from the bucket the
    donor's own 147-key population is defined against. ``--lr-decay-steps``
    / ``--accum`` are per-ARM knobs for the target-fact training only,
    never for the donor.

    Args:
        model: Base model (unwrapped inside if currently a ``PeftModel`` —
            discards any resident adapter, matching ``_run_seed``'s Step 1).
        tokenizer: Tokenizer matching the model.
        adapter_config: The target tier's ``AdapterConfig`` (episodic for
            ``--donor-init``; procedural for ``--donor-build-smoke`` — this
            function is topology-agnostic). The donor trains at the SAME
            rank/alpha/target_modules as the arm it will seed
            (``copy_adapter_weights`` requires exact topology match);
            ``learning_rate``/``dropout`` are always overridden to the donor
            recipe (see above).
        base_training_config: The target tier's ``TrainingConfig``
            (batch_size/lr/scheduler carried through unchanged; only
            ``num_epochs``/``seed``/``recall_early_stopping``/
            ``lr_decay_steps``/``gradient_accumulation_steps`` are
            overridden for the donor build — see above for why the latter
            two are always derived from ``budget_for(len(entries))``
            regardless of an arm's ``--lr-decay-steps``/``--accum``
            override).
        checkpoint_root: Directory ``atomic_save_adapter`` saves the donor
            slot under (``target_dir/<ts>/`` — the returned ``Path`` is that
            promoted slot).

    Returns:
        Tuple of ``(model, slot_path, donor_summary)``. ``model`` still
        carries ``DONOR_BUILD_ADAPTER_NAME`` as its active adapter — the
        caller's next step (``_run_seed``'s Step 1) unwraps and discards it,
        exactly as it already discards any previous seed's adapters.
        ``donor_summary`` carries the donor's own final recall
        (``exact_count``/``total``/``rate``/``mean_confidence``/``per_key``
        with verbatim ``raw_output``), the realized weights SHA-256, the
        realized optimizer-step count (asserted equal to the derived
        expected count — mirrors the arm-side Hard Assertion #1), and the
        pre/post LoRA-B Frobenius norms (cold-init proof for the build
        itself). ``slot / DONOR_META_FILENAME`` (``"donor_meta.json"``) is
        also written — ``{seed, n_entries, epochs,
        gradient_accumulation_steps, realized_optimizer_steps,
        weights_sha256, wall_train_seconds}`` — the single source of truth
        ``_resolve_donor_source``/``_read_donor_meta`` read back later
        (including from a DIFFERENT run's ``--donor-checkpoint`` reuse,
        since it travels with the slot). ``_read_donor_meta`` tolerates the
        absence of ``gradient_accumulation_steps``/``realized_optimizer_steps``/
        ``wall_train_seconds`` on slots built before those fields were added.
    """
    entries = donor_entries(DONOR_DEFAULT_SEED, DONOR_MIN_ENTRIES)
    registry = build_registry(entries)
    donor_epochs, donor_accum, donor_lr_decay_steps = budget_for(len(entries))
    expected_donor_steps = _expected_optimizer_steps(
        len(entries), donor_epochs, donor_accum, base_training_config.batch_size
    )

    # Topology (rank/alpha/target_modules) is UNCHANGED — only
    # learning_rate/dropout are recipe-derived, never hand-copied — matching
    # production build_donor's recipe-fidelity contract unconditionally.
    donor_adapter_config = dataclasses.replace(
        adapter_config,
        learning_rate=DONOR_RECIPE_LEARNING_RATE,
        dropout=DONOR_RECIPE_DROPOUT,
    )

    if isinstance(model, PeftModel):
        model = model.base_model.model
    torch.manual_seed(DONOR_DEFAULT_SEED)
    model = create_adapter(model, donor_adapter_config, DONOR_BUILD_ADAPTER_NAME)
    switch_adapter(model, DONOR_BUILD_ADAPTER_NAME)

    lora_b_norm_before = lora_b_frobenius_norm(model, DONOR_BUILD_ADAPTER_NAME)
    assert lora_b_norm_before == 0.0, (
        f"Donor build: LoRA-B Frobenius norm for '{DONOR_BUILD_ADAPTER_NAME}' is "
        f"{lora_b_norm_before}, expected 0.0 (cold init) before the donor build's "
        "own training."
    )

    examples = format_entry_training(
        entries, tokenizer, max_length=base_training_config.max_seq_length
    )
    dataset = IndexedDataset(examples)
    donor_training_cfg = dataclasses.replace(
        base_training_config,
        seed=DONOR_DEFAULT_SEED,
        num_epochs=donor_epochs,
        recall_early_stopping=False,
        # ALWAYS budget_for's derived value for the donor's own training —
        # never inherit an arm's --lr-decay-steps/--accum override (see the
        # docstring above).
        lr_decay_steps=donor_lr_decay_steps,
        gradient_accumulation_steps=donor_accum,
    )

    logger.info(
        "Donor build: training %d entries, %d epochs, accum=%d, seed=%d",
        len(examples),
        donor_epochs,
        donor_accum,
        DONOR_DEFAULT_SEED,
    )
    step_cb = _StepCaptureCallback()
    t0 = time.time()
    metrics = train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        adapter_name=DONOR_BUILD_ADAPTER_NAME,
        training_config=donor_training_cfg,
        adapter_config=donor_adapter_config,
        output_dir=checkpoint_root / ".training_scratch",
        run_name="test20-donor-build",
        callbacks_extra=[step_cb],
    )
    wall_train = time.time() - t0
    train_loss = (metrics or {}).get("train_loss")

    realized_donor_steps = step_cb.global_step
    assert realized_donor_steps is not None, (
        "Donor build: _StepCaptureCallback never fired on_train_end — no realized "
        "step count captured."
    )
    assert realized_donor_steps == expected_donor_steps, (
        f"Donor build: realized optimizer steps={realized_donor_steps}, expected "
        f"{expected_donor_steps} (n_entries={len(entries)}, epochs={donor_epochs}, "
        f"accum={donor_accum})."
    )

    lora_b_norm_after = lora_b_frobenius_norm(model, DONOR_BUILD_ADAPTER_NAME)
    assert lora_b_norm_after > 0.0, (
        f"Donor build: LoRA-B Frobenius norm for '{DONOR_BUILD_ADAPTER_NAME}' is "
        f"{lora_b_norm_after}, expected > 0.0 after training (the donor build adapter "
        "did not move)."
    )

    recall = _run_recall_probe(model, tokenizer, entries, registry, DONOR_BUILD_ADAPTER_NAME)

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    # Save through the SAME primitives a production donor build uses, so this
    # slot is a valid production donor store slot -- one on-disk contract, one
    # validity rule (paramem.training.donor.donor_slot_valid) rather than a
    # second, weaker one local to this script. registry_sha256 is empty
    # because a donor carries no key registry; that is what lets
    # find_live_slot resolve the slot.
    manifest = build_manifest_for(
        model,
        tokenizer,
        DONOR_BUILD_ADAPTER_NAME,
        registry_path=None,
        registry_sha256_override="",
        key_count=len(entries),
        adapter_root=checkpoint_root,
    )
    slot = atomic_save_adapter(model, checkpoint_root, DONOR_BUILD_ADAPTER_NAME, manifest=manifest)
    weights_sha256 = hashlib.sha256((slot / "adapter_model.safetensors").read_bytes()).hexdigest()

    # Donor identity, in the production schema -- what this checkpoint IS.
    # seed + n_requested regenerate the exact entry list, so nothing about the
    # donor's content needs a second recording.
    donor_meta = {
        "seed": DONOR_DEFAULT_SEED,
        "recipe": DONOR_RECIPE_ID,
        "n_requested": DONOR_MIN_ENTRIES,
        "triples": entries,
        "triples_hash": triples_hash(entries),
        "weights_sha256": weights_sha256,
    }
    (slot / DONOR_META_FILENAME).write_text(json.dumps(donor_meta, indent=2))

    # How THIS run trained it -- measurement provenance, not donor identity.
    # A separate file so the two schemas can never collide on one name: a
    # production donor slot fed to --donor-checkpoint is then simply a donor
    # without build provenance, not a slot whose meta means something else.
    (slot / DONOR_BUILD_PROVENANCE_FILENAME).write_text(
        json.dumps(
            {
                "n_entries": len(entries),
                "epochs": donor_epochs,
                "gradient_accumulation_steps": donor_accum,
                "realized_optimizer_steps": realized_donor_steps,
                "wall_train_seconds": wall_train,
            },
            indent=2,
        )
    )

    donor_summary = {
        "seed": DONOR_DEFAULT_SEED,
        "n_entries": len(entries),
        "epochs": donor_epochs,
        "gradient_accumulation_steps": donor_accum,
        "realized_optimizer_steps": realized_donor_steps,
        "slot": str(slot),
        "weights_sha256": weights_sha256,
        "lora_b_norm_before_training": lora_b_norm_before,
        "lora_b_norm_after_training": lora_b_norm_after,
        "train_loss": train_loss,
        "wall_train_seconds": wall_train,
        "summary": {
            "exact_count": recall["exact_count"],
            "total": recall["total"],
            "rate": recall["rate"],
            "mean_confidence": recall["mean_confidence"],
        },
        "per_key": recall["per_key"],
    }
    save_results(donor_summary, checkpoint_root, filename="donor_build_results.json")
    logger.info(
        "Donor build complete: slot=%s recall=%d/%d (%.1f%%) sha256=%s",
        slot,
        recall["exact_count"],
        recall["total"],
        recall["rate"] * 100,
        weights_sha256,
    )
    return model, slot, donor_summary


def _donor_verification_context(model, adapter_config) -> tuple[str, dict]:
    """Return ``(base_model_id, lora_shape)`` for a donor slot verification.

    Derived once here rather than at each verification site: both values come
    from state the caller already holds, and a site that re-derived them
    differently would verify against a different contract than the rest.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return getattr(base.config, "_name_or_path", None), lora_shape_fields(adapter_config)


def _read_donor_meta(slot: Path, base_model_id: str, lora_shape: dict) -> dict:
    """Verify *slot* is a valid donor and return what this script reads back.

    Validity is ``paramem.training.donor.donor_slot_valid`` -- the SAME rule
    the production seeding hook applies (manifest base model + topology,
    recipe id, weights digest, and regeneration of the recorded triple set
    from its seed). This script does not re-implement any part of it; it only
    turns the boolean into the fail-loud ``SystemExit`` its arms expect,
    because seeding from a checkpoint that cannot be verified would silently
    corrupt the arm's result rather than stop it.

    *slot* is a slot directory, not a store: this script verifies a detached
    ``--donor-checkpoint`` path and its own scratch copytree, neither of which
    sits inside a store. That is why the slot-scoped primitive exists.

    Returns:
        ``seed`` and ``n_entries`` (the donor's identity, from the production
        meta -- ``n_entries`` is the length of the recorded triple set) plus
        ``epochs``/``gradient_accumulation_steps``/``realized_optimizer_steps``/
        ``wall_train_seconds`` when the slot carries this script's build
        provenance. A production-built donor carries none of the latter; they
        come back ``None`` and are reported as such rather than fabricated.

    Raises:
        SystemExit: the slot is not a valid donor checkpoint.
    """
    if not donor_slot_valid(slot, base_model_id, lora_shape):
        raise SystemExit(
            f"Donor checkpoint at {slot} is not valid for base {base_model_id!r} and "
            f"this arm's LoRA shape. Refusing to seed from an unverified checkpoint. "
            f"Causes: missing/corrupt weights or {DONOR_META_FILENAME}, a weights "
            f"digest that no longer matches, a different base model or topology, or "
            f"a donor recipe change since it was built."
        )
    with open(slot / DONOR_META_FILENAME) as f:
        meta = json.load(f)

    provenance_path = slot / DONOR_BUILD_PROVENANCE_FILENAME
    provenance: dict = {}
    if provenance_path.is_file():
        with open(provenance_path) as f:
            provenance = json.load(f)
    else:
        logger.info(
            "Donor slot %s carries no %s (a production-built donor, or one built "
            "before this script recorded build provenance) — reporting its build "
            "fields as null.",
            slot,
            DONOR_BUILD_PROVENANCE_FILENAME,
        )

    return {
        "seed": meta["seed"],
        "n_entries": len(meta["triples"]),
        "weights_sha256": meta["weights_sha256"],
        "epochs": provenance.get("epochs"),
        "gradient_accumulation_steps": provenance.get("gradient_accumulation_steps"),
        "realized_optimizer_steps": provenance.get("realized_optimizer_steps"),
        "wall_train_seconds": provenance.get("wall_train_seconds"),
    }


def _build_or_reuse_own_donor_checkpoint(
    run_dir: Path,
    model,
    tokenizer,
    adapter_config,
    base_training_config,
) -> tuple[object, Path, bool, dict]:
    """Resolve THIS run's OWN donor checkpoint — cases 2/3 of
    :func:`_resolve_donor_source` (never an external ``--donor-checkpoint``,
    which stays inline in that function's case 1).

    2. This run already built its own donor checkpoint
       (``run_dir/DONOR_BUILD_MARKER_FILENAME`` exists — the phase marker,
       MINIMAL: just ``{slot, timestamp}``, since provenance lives in the
       slot's own ``donor_meta.json``): reuse the slot recorded in the
       marker WITHOUT retraining. Covers the resumability requirement — a
       crash after the donor build (marker written) must not rebuild it on
       retry.
    3. Neither: build a fresh donor checkpoint via
       :func:`_build_donor_checkpoint` under
       ``run_dir/DONOR_CHECKPOINT_DIRNAME``, then write the (minimal) phase
       marker.

    Extracted as its own function (rather than inlined twice) so
    ``--donor-build-smoke`` (never accepts an external ``--donor-checkpoint``)
    and ``--donor-init`` (via ``_resolve_donor_source``'s cases 2/3) share
    ONE resume-aware build/reuse implementation instead of each maintaining
    its own marker bookkeeping.

    Args:
        run_dir: This run's arm-scoped output directory.
        model: Base model or ``PeftModel`` (forwarded to
            ``_build_donor_checkpoint`` when a fresh build is needed;
            returned unchanged when reusing an existing marker).
        tokenizer: Tokenizer matching the model.
        adapter_config: The target tier's ``AdapterConfig`` (episodic for
            ``--donor-init``; procedural for ``--donor-build-smoke`` — this
            function is topology-agnostic).
        base_training_config: ``TrainingConfig`` matching *adapter_config*'s
            tier.

    Returns:
        Tuple of ``(model, slot_path, built_fresh, donor_meta)`` — see
        :func:`_resolve_donor_source`'s own Returns section (identical
        contract).

    Raises:
        SystemExit: The marker's recorded slot is missing its weights file.
    """
    marker_path = run_dir / DONOR_BUILD_MARKER_FILENAME
    if marker_path.exists():
        with open(marker_path) as f:
            marker = json.load(f)
        slot = Path(marker["slot"])
        if not (slot / "adapter_model.safetensors").is_file():
            raise SystemExit(
                f"Donor build marker at {marker_path} points to a missing checkpoint "
                f"({slot}) — delete the marker to force a rebuild."
            )
        donor_meta = _read_donor_meta(slot, *_donor_verification_context(model, adapter_config))
        logger.info("Donor build: reusing this run's already-built checkpoint at %s", slot)
        return model, slot, False, donor_meta

    checkpoint_root = run_dir / DONOR_CHECKPOINT_DIRNAME
    logger.info("Donor build: no checkpoint found — building fresh donor at %s", checkpoint_root)
    model, slot, donor_summary = _build_donor_checkpoint(
        model, tokenizer, adapter_config, base_training_config, checkpoint_root
    )
    donor_meta = {
        "seed": donor_summary["seed"],
        "n_entries": donor_summary["n_entries"],
        "epochs": donor_summary["epochs"],
        "weights_sha256": donor_summary["weights_sha256"],
    }
    marker = {"slot": str(slot), "timestamp": int(time.time())}
    with open(marker_path, "w") as f:
        json.dump(marker, f, indent=2)
    logger.info("Donor build marker written: %s", marker_path)
    return model, slot, True, donor_meta


def _resolve_donor_source(
    donor_checkpoint_arg: str | None,
    run_dir: Path,
    model,
    tokenizer,
    adapter_config,
    base_training_config,
) -> tuple[object, Path, bool, dict]:
    """Resolve the donor checkpoint slot ``--donor-init`` copies into ``donor_scratch_dir``.

    Case 1 (``--donor-checkpoint PATH`` given) is handled inline here; cases
    2/3 (this run's own already-built or freshly-built checkpoint) delegate
    to :func:`_build_or_reuse_own_donor_checkpoint` — see that function's
    docstring for the full case-2/3 mechanism, and for why it exists as a
    separate function rather than being inlined here twice.

    Args:
        donor_checkpoint_arg: ``args.donor_checkpoint`` (``None`` unless the
            operator passed ``--donor-checkpoint``).
        run_dir: This run's arm-scoped output directory.
        model: Base model or ``PeftModel`` (forwarded to
            ``_build_donor_checkpoint`` when a fresh build is needed;
            returned unchanged in cases 1/2).
        tokenizer: Tokenizer matching the model.
        adapter_config: Production episodic ``AdapterConfig``.
        base_training_config: Production episodic ``TrainingConfig``.

    Returns:
        Tuple of ``(model, slot_path, built_fresh, donor_meta)``.
        *built_fresh* is True only for case 3 (a real GPU training run just
        happened — the caller uses this to insert a cooldown before the
        first seed, B2). *donor_meta* is :func:`_read_donor_meta`'s return
        (``seed``/``n_entries``/``epochs``/``weights_sha256``), used by the
        caller to recompute ``donor_entries(seed, n_entries)`` for the H1
        key-overlap recording.

    Raises:
        SystemExit: ``--donor-checkpoint``/the marker's recorded slot is
            missing its weights file or fails SHA-256 verification.
    """
    if donor_checkpoint_arg is not None:
        slot = Path(donor_checkpoint_arg)
        if not (slot / "adapter_model.safetensors").is_file():
            raise SystemExit(
                f"--donor-checkpoint {slot} has no adapter_model.safetensors — "
                "not a valid donor checkpoint slot."
            )
        donor_meta = _read_donor_meta(slot, *_donor_verification_context(model, adapter_config))
        logger.info("Donor-init: reusing external donor checkpoint at %s", slot)
        return model, slot, False, donor_meta

    return _build_or_reuse_own_donor_checkpoint(
        run_dir, model, tokenizer, adapter_config, base_training_config
    )


# ---------------------------------------------------------------------------
# --donor-build-smoke: procedural-topology donor build + cold-seed GPU cost
# probe. Feeds the SAME _build_donor_checkpoint / _resolve_donor_source
# machinery --donor-init already uses (topology-agnostic — adapter_config is
# a parameter), never a second donor-build implementation.
# ---------------------------------------------------------------------------

DONOR_BUILD_SMOKE_ARM: str = "donor_build_smoke_procedural"
"""Fixed arm label (== the ``model_output_dir`` subtree) for
``--donor-build-smoke``. No ``--arm`` override — this mode always targets
exactly one topology (procedural) and never runs the ``--n-entries``/
``--seeds`` arm loop, so there is nothing for an arm label to disambiguate
beyond the model name ``model_output_dir`` already scopes by."""

DONOR_BUILD_SMOKE_SEED_ADAPTER_NAME: str = "procedural_donor_smoke_seed"
"""Trainable adapter name for the cold-seed phase. Asserted (like every
``_run_seed`` trainable adapter) to never collide with a live tier name —
see ``LIVE_TIER_NAMES``."""

DONOR_BUILD_SMOKE_SEED_MARKER_FILENAME: str = "donor_build_smoke_seed_done.json"
"""Phase marker for the seed (``copy_adapter_weights``) phase — mirrors
``DONOR_BUILD_MARKER_FILENAME``'s pattern for the build phase (already
handled by ``_resolve_donor_source``'s own marker check, reused unchanged
here). Presence means the seed phase ran to completion and recorded a
result — success OR a caught failure (see ``_run_donor_build_smoke``) — so a
rerun after this marker exists skips the seed phase entirely; a rerun after
ONLY the build-phase marker exists (this one absent) skips straight to the
seed phase instead of rebuilding the donor."""

DONOR_BUILD_SMOKE_CONFIG_FILENAME: str = "smoke_config.json"
"""Records this run's resolved procedural topology (rank/alpha/
target_modules) — mirrors ``main()``'s ``run_config.json`` mismatch guard
(see :func:`_main_donor_build_smoke`) for the much smaller, topology-only
config surface this mode carries. A ``--resume`` invocation whose CURRENT
``tests/fixtures/server.yaml`` ``procedural_adapter_config`` disagrees with
the recorded value fails loud before either phase runs, rather than
silently seeding a mismatched-topology adapter."""


def _cuda_mem_get_info_mib() -> dict:
    """Sample ``torch.cuda.mem_get_info`` as ``{"free_mib", "total_mib"}``.

    WSL2's ``nvidia-smi`` is VRAM-blind (project rule) — ``mem_get_info`` is
    the authoritative source for free/total device memory. The first call in
    a process initializes the CUDA context if it has not already been
    initialized (expected/harmless for the "before load" sample — it reflects
    the context's own small footprint, not the base model).

    Returns:
        ``{"free_mib": float, "total_mib": float}``.
    """
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {"free_mib": free_bytes / 1024**2, "total_mib": total_bytes / 1024**2}


def _run_donor_build_smoke(
    model,
    tokenizer,
    cfg,
    run_dir: Path,
    vram_before_load_mib: dict,
    vram_after_load_mib: dict,
) -> None:
    """Build a procedural-topology donor checkpoint, then cold-seed a fresh
    procedural adapter from it, recording GPU cost telemetry throughout.

    Two resumable phases, each writing its own results file + (seed phase
    only) done-marker:

    1. **Build** — ``_build_or_reuse_own_donor_checkpoint(run_dir, ...)``
       (the SAME resume-aware build/reuse logic ``--donor-init`` uses via
       ``_resolve_donor_source``'s cases 2/3 — one shared implementation,
       called directly here since ``--donor-build-smoke`` never accepts an
       external ``--donor-checkpoint``, i.e. never needs
       ``_resolve_donor_source``'s case 1): builds fresh via
       ``_build_donor_checkpoint`` if no donor checkpoint exists yet under
       *run_dir*, otherwise reuses it and verifies its weights SHA-256 — no
       duplicated resume bookkeeping here. Topology (rank/alpha/
       target_modules) comes from *cfg*'s ``procedural_adapter_config``;
       ``_build_donor_checkpoint`` always trains at
       ``DONOR_RECIPE_LEARNING_RATE``/``DONOR_RECIPE_DROPOUT`` (see that
       function's docstring). ``torch.cuda.reset_peak_memory_stats()`` runs
       immediately before the call so ``torch.cuda.max_memory_allocated()``/
       ``torch.cuda.max_memory_reserved()`` after it reflect ONLY this build
       (never a bare ``empty_cache`` — see CLAUDE.md). Skipped (falls
       straight to reading the already-written ``build_results.json``) when
       that file already exists from a prior invocation of this run dir —
       this is what makes "the build phase completed" resumable at the
       phase boundary, independent of whether the helper itself needed to
       retrain (it does not, once its own marker exists).
    2. **Seed** — unwrap, load the resolved donor slot fresh via
       ``_adapter_slot_for_load`` + ``PeftModel.from_pretrained`` (the SAME
       load pattern ``_run_seed``'s Step 1b already uses for
       ``--warm-from``/``--donor-init`` — reused directly against the
       resolved slot with no separate scratch-copy step, since this is a
       checkpoint this run itself owns rather than a live external adapter
       directory whose corruption would be data loss), ``torch.
       manual_seed(DONOR_DEFAULT_SEED)`` before ``create_adapter`` (fresh
       LoRA-zero, at the SAME ``procedural_adapter_config`` topology), then
       the SAME strict ``copy_adapter_weights`` call the ``--warm-from``/
       ``--donor-init`` arms use (raises ``RuntimeError`` on any
       parameter-set mismatch — no special casing here). The destination
       adapter's LoRA-B Frobenius norm is asserted ``== 0.0`` before the
       copy (fresh cold adapter) and equal to the donor's own norm after
       the copy lands — the same before/after norm proof ``_run_seed``'s
       Hard Assertion #3 uses. No recall evaluation runs on the seeded
       adapter (this arm measures GPU feasibility/cost, not recall
       quality). The donor-immutability tier-name guard (the trainable
       adapter name is never a live tier name) runs BEFORE the try block,
       matching ``_run_seed``'s placement — a name collision is a
       programming bug, not a measured smoke outcome, and must not be
       swallowed by the except clause below. Any exception during the
       load/copy is caught, its full traceback recorded in
       ``seed_results.json``, and the phase marker is still written — a
       measured failure is a valid smoke result, not a crash (boundary
       error handling for this measurement's own success/failure signal —
       see CLAUDE.md's try/except carve-out). A ``torch.cuda.
       mem_get_info`` sample is recorded after the seed phase regardless of
       outcome. Skipped entirely (no-op) once
       ``DONOR_BUILD_SMOKE_SEED_MARKER_FILENAME`` exists from a prior
       invocation.

    Args:
        model: The freshly-loaded base model (never a ``PeftModel`` yet at
            this call site — no prior adapter has been created this run).
        tokenizer: Tokenizer matching *model*.
        cfg: The loaded ``ServerConfig`` (``tests/fixtures/server.yaml``) —
            ``cfg.procedural_adapter_config`` / ``cfg.training_config``
            supply the topology/recipe (rank/alpha/target_modules from yaml,
            never hardcoded).
        run_dir: This run's arm-scoped output directory
            (``model_output_dir(OUTPUT_BASE / DONOR_BUILD_SMOKE_ARM, model)``).
        vram_before_load_mib: ``_cuda_mem_get_info_mib()`` sampled by the
            caller immediately after entering ``acquire_gpu`` and BEFORE
            ``load_model_and_config`` — recorded verbatim in
            ``build_results.json``.
        vram_after_load_mib: ``_cuda_mem_get_info_mib()`` sampled by the
            caller immediately after ``load_model_and_config`` returns —
            recorded verbatim in ``build_results.json``.
    """
    adapter_config = cfg.procedural_adapter_config
    base_training_config = dataclasses.replace(cfg.training_config, recall_early_stopping=False)

    build_results_path = run_dir / "build_results.json"
    seed_marker_path = run_dir / DONOR_BUILD_SMOKE_SEED_MARKER_FILENAME

    # --- Phase 1: build (or reuse) the procedural-topology donor checkpoint.
    _check_pause("before donor-build-smoke build phase")
    torch.cuda.reset_peak_memory_stats()
    t_build0 = time.time()
    model, slot, built_fresh, donor_meta = _build_or_reuse_own_donor_checkpoint(
        run_dir, model, tokenizer, adapter_config, base_training_config
    )
    wall_build_seconds = time.time() - t_build0
    vram_after_build_mib = _cuda_mem_get_info_mib()
    peak_allocated_build_mib = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved_build_mib = torch.cuda.max_memory_reserved() / 1024**2

    if build_results_path.exists():
        logger.info(
            "Donor-build-smoke: %s already present — build phase already recorded, "
            "proceeding to seed phase.",
            build_results_path,
        )
    else:
        full_meta = _read_donor_meta(slot, *_donor_verification_context(model, adapter_config))
        realized_steps = full_meta.get("realized_optimizer_steps")
        wall_train_seconds = full_meta.get("wall_train_seconds")
        # wall/step telemetry is only meaningful for a build that just
        # trained IN THIS invocation — a resumed reuse (built_fresh=False)
        # returns near-instantly (SHA-256 verification only), so recording
        # its trivial wall clock as the build cost would be fiction.
        # mean_seconds_per_step divides the TRAINING-ONLY clock
        # (wall_train_seconds, captured inside _build_donor_checkpoint
        # around its own train_adapter call) rather than wall_build_seconds
        # (this phase's total wall clock, which also includes dataset
        # assembly, the post-training recall probe, and
        # atomic_save_adapter) — both numbers are recorded, each labelled
        # for what it actually measures.
        build_results = {
            "topology": "procedural",
            "adapter_config": {
                "rank": adapter_config.rank,
                "alpha": adapter_config.alpha,
                "target_modules": adapter_config.target_modules,
            },
            "donor_recipe_learning_rate": DONOR_RECIPE_LEARNING_RATE,
            "donor_recipe_dropout": DONOR_RECIPE_DROPOUT,
            "n_entries": donor_meta.get("n_entries"),
            "epochs": donor_meta.get("epochs"),
            "gradient_accumulation_steps": full_meta.get("gradient_accumulation_steps"),
            "realized_optimizer_steps": realized_steps,
            "built_fresh": built_fresh,
            "slot": str(slot),
            "wall_build_seconds": wall_build_seconds if built_fresh else None,
            "wall_train_seconds": wall_train_seconds if built_fresh else None,
            "mean_seconds_per_step": (
                wall_train_seconds / realized_steps
                if built_fresh and realized_steps and wall_train_seconds
                else None
            ),
            "vram_before_load_mib": vram_before_load_mib,
            "vram_after_load_mib": vram_after_load_mib,
            "vram_after_build_mib": vram_after_build_mib,
            "peak_allocated_build_mib": peak_allocated_build_mib if built_fresh else None,
            "peak_reserved_build_mib": peak_reserved_build_mib if built_fresh else None,
        }
        save_results(build_results, run_dir, filename="build_results.json")
        logger.info(
            "Donor-build-smoke: build phase complete (built_fresh=%s) — %s",
            built_fresh,
            build_results_path,
        )

    # --- Phase 2: cold-seed a fresh procedural adapter from the built donor.
    if seed_marker_path.exists():
        logger.info(
            "Donor-build-smoke: %s already present — seed phase already recorded, nothing to do.",
            seed_marker_path,
        )
        return

    _check_pause("before donor-build-smoke seed phase")
    if built_fresh:
        logger.info("Cooldown before the seed phase (a fresh donor build just ran)")
        _wait_for_cooldown(52)

    if isinstance(model, PeftModel):
        model = model.base_model.model

    # Donor-immutability tier-name guard runs BEFORE the try block and
    # BEFORE create_adapter/switch_adapter, matching _run_seed's placement
    # (Steps 2-3) — a name collision is a programming bug, not a measured
    # smoke outcome, and must not be swallowed by the except clause below.
    assert DONOR_BUILD_SMOKE_SEED_ADAPTER_NAME not in LIVE_TIER_NAMES, (
        f"Donor-immutability guard FAILED: '{DONOR_BUILD_SMOKE_SEED_ADAPTER_NAME}' "
        f"collides with a live tier name {sorted(LIVE_TIER_NAMES)}."
    )

    try:
        with _adapter_slot_for_load(slot) as load_path:
            model = PeftModel.from_pretrained(
                model, str(load_path), adapter_name=DONOR_ADAPTER_NAME
            )

        donor_norm = lora_b_frobenius_norm(model, DONOR_ADAPTER_NAME)

        torch.manual_seed(DONOR_DEFAULT_SEED)
        model = create_adapter(model, adapter_config, DONOR_BUILD_SMOKE_SEED_ADAPTER_NAME)
        switch_adapter(model, DONOR_BUILD_SMOKE_SEED_ADAPTER_NAME)

        norm_before = lora_b_frobenius_norm(model, DONOR_BUILD_SMOKE_SEED_ADAPTER_NAME)
        assert norm_before == 0.0, (
            f"Donor-build-smoke seed FAILED: destination adapter "
            f"'{DONOR_BUILD_SMOKE_SEED_ADAPTER_NAME}' LoRA-B Frobenius norm is "
            f"{norm_before}, expected 0.0 (fresh cold adapter) before "
            "copy_adapter_weights."
        )

        t_seed0 = time.time()
        copy_adapter_weights(model, src=DONOR_ADAPTER_NAME, dst=DONOR_BUILD_SMOKE_SEED_ADAPTER_NAME)
        wall_copy_seconds = time.time() - t_seed0
        norm_after = lora_b_frobenius_norm(model, DONOR_BUILD_SMOKE_SEED_ADAPTER_NAME)
        assert norm_after == donor_norm, (
            f"Donor-build-smoke seed FAILED: destination adapter "
            f"'{DONOR_BUILD_SMOKE_SEED_ADAPTER_NAME}' LoRA-B Frobenius norm after "
            f"copy ({norm_after}) != donor '{DONOR_ADAPTER_NAME}' norm ({donor_norm}) "
            "— the copy did not land (or landed corrupted)."
        )

        seed_results = {
            "success": True,
            "adapter_name": DONOR_BUILD_SMOKE_SEED_ADAPTER_NAME,
            "donor_lora_b_norm": donor_norm,
            "lora_b_norm_before_copy": norm_before,
            "lora_b_norm_after_copy": norm_after,
            "wall_copy_seconds": wall_copy_seconds,
            "exception": None,
        }
        logger.info(
            "Donor-build-smoke: seed phase succeeded — lora_b_norm %.6f -> %.6f in %.3fs",
            norm_before,
            norm_after,
            wall_copy_seconds,
        )
    except Exception as exc:
        # Boundary error handling for this measurement's own success/failure
        # signal — a strict-copy failure (e.g. a parameter-set mismatch from
        # copy_adapter_weights) IS a valid, measured smoke outcome, not a
        # bug to mask; the full traceback is preserved so the failure is
        # diagnosable from results.json alone.
        seed_results = {
            "success": False,
            "adapter_name": DONOR_BUILD_SMOKE_SEED_ADAPTER_NAME,
            "exception": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        }
        logger.error("Donor-build-smoke: seed phase failed: %s", exc)

    # A VRAM sample after the seed phase, regardless of outcome, for
    # measurement-honesty parity with the build phase's own
    # before/after-load/after-build samples.
    seed_results["vram_after_seed_mib"] = _cuda_mem_get_info_mib()

    save_results(seed_results, run_dir, filename="seed_results.json")
    with open(seed_marker_path, "w") as f:
        json.dump({"timestamp": int(time.time()), "success": seed_results["success"]}, f, indent=2)
    logger.info(
        "Donor-build-smoke: seed phase recorded (success=%s) — %s",
        seed_results["success"],
        seed_marker_path,
    )


def _main_donor_build_smoke(args: argparse.Namespace) -> None:
    """Entry point for ``--donor-build-smoke``.

    A GPU feasibility/cost measurement for the per-topology donor build:
    builds ONE donor checkpoint at the PROCEDURAL topology
    (rank/alpha/target_modules from ``tests/fixtures/server.yaml``'s
    ``procedural_adapter_config`` via ``load_server_config`` — never
    hardcoded) and cold-seeds a fresh procedural adapter from it, measuring
    wall time, mean seconds/optimizer step, and VRAM (``torch.cuda.
    mem_get_info`` before load / after load / after build / after seed,
    plus ``torch.cuda.max_memory_allocated``/``max_memory_reserved`` for the
    build phase — WSL2's ``nvidia-smi`` is VRAM-blind, ``mem_get_info`` is
    authoritative). Does NOT run recall evaluation on the seeded adapter and
    does NOT exercise production ``paramem.training.donor.build_donor``
    (needs a live ``ConsolidationLoop`` this standalone experiment has no
    business depending on — see ``_build_donor_checkpoint``'s own
    docstring) — see :func:`_run_donor_build_smoke` for the full two-phase
    mechanism.

    Resumable at the build/seed phase boundary via
    :func:`_run_donor_build_smoke`'s own marker/results-file checks;
    ``--resume`` here means "find the latest run dir under this mode's own
    output subtree" (:func:`_preflight_run_dir`, shared with ``main()``'s
    per-arm ``--resume`` handling). Before either phase runs, this run's
    resolved procedural topology is compared against the run dir's own
    recorded ``DONOR_BUILD_SMOKE_CONFIG_FILENAME`` (``"smoke_config.json"``,
    written on first invocation) — a mismatch (e.g. the fixture's
    ``procedural_adapter_config`` changed between the original run and a
    ``--resume``) fails loud rather than silently seeding a
    mismatched-topology adapter, mirroring ``main()``'s ``run_config.json``
    guard for the arm loop.

    Designed to run daemonised (setsid/nohup) — no terminal interaction;
    honours ``~/.training_pause`` at both phase boundaries (see
    :func:`_run_donor_build_smoke`). Exits non-zero (``SystemExit(1)``) when
    the recorded seed phase result is a measured failure — the caller
    (shell/systemd) must be able to tell success from failure without
    parsing ``seed_results.json`` itself. Markers/results files are written
    before this exit, so a re-run after a measured seed failure is a no-op
    (matching every other phase-marker skip in this mode), not a retry.

    Args:
        args: Parsed CLI args. Only ``args.model``/``args.resume`` are
            read — every other arm-configuration flag is validated as
            mutually exclusive with ``--donor-build-smoke`` by ``main()``
            before this function is called.

    Raises:
        SystemExit: Free disk space is insufficient; this run dir's
            recorded procedural topology disagrees with the current
            fixture config; or the recorded seed-phase result is a
            measured failure (exit code 1).
    """
    cfg = load_server_config(str(FIXTURE_CONFIG_PATH))

    arm_base = OUTPUT_BASE / DONOR_BUILD_SMOKE_ARM
    run_dir = _preflight_run_dir(arm_base, args.model, args.resume)

    adapter_config = cfg.procedural_adapter_config
    smoke_config = {
        "model": args.model,
        "topology": "procedural",
        "rank": adapter_config.rank,
        "alpha": adapter_config.alpha,
        "target_modules": adapter_config.target_modules,
    }
    smoke_config_path = run_dir / DONOR_BUILD_SMOKE_CONFIG_FILENAME
    if smoke_config_path.exists():
        with open(smoke_config_path) as f:
            existing_smoke_config = json.load(f)
        if existing_smoke_config != smoke_config:
            raise SystemExit(
                f"Run-config mismatch at {smoke_config_path}: this invocation's "
                f"resolved procedural topology ({smoke_config}) disagrees with the "
                f"recorded one ({existing_smoke_config}) — refusing to mix topologies "
                "into the same run dir. Start a fresh run (a clean output dir) or "
                "restore the matching tests/fixtures/server.yaml."
            )
        logger.info(
            "Existing %s matches this invocation's procedural topology — resuming.",
            smoke_config_path,
        )
    else:
        with open(smoke_config_path, "w") as f:
            json.dump(smoke_config, f, indent=2)
        logger.info("Smoke config written: %s", smoke_config_path)

    # Imported here, not at module scope: gpu_guard ships from lab-tools
    # (separate repo, not on PyPI) and is absent in CI.  Only the GPU entry
    # points need it, so keeping it out of module scope lets the pure-Python
    # helpers in this file be imported — and unit-tested — without it.
    from experiments.utils.gpu_guard import acquire_gpu

    model_config = BENCHMARK_MODELS[args.model]
    with acquire_gpu(interactive=True):
        vram_before_load_mib = _cuda_mem_get_info_mib()
        model, tokenizer = load_model_and_config(model_config)
        vram_after_load_mib = _cuda_mem_get_info_mib()
        _run_donor_build_smoke(
            model, tokenizer, cfg, run_dir, vram_before_load_mib, vram_after_load_mib
        )
        unload_model(model, tokenizer)

    with open(run_dir / "seed_results.json") as f:
        seed_results = json.load(f)
    success = seed_results["success"]

    print("\n" + "=" * 72)
    print(f"Test 20 — {DONOR_BUILD_SMOKE_ARM} ({args.model}) complete (seed success={success})")
    print("=" * 72)
    print(f"Results written to: {run_dir}")
    logger.info("Donor-build-smoke complete (seed success=%s). Results: %s", success, run_dir)

    if not success:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Core seed runner
# ---------------------------------------------------------------------------


def _run_seed(
    model,
    tokenizer,
    seed: int,
    entries: list[dict],
    registry: dict,
    adapter_config,
    base_training_config,
    run_dir: Path,
    arm: str,
    expected_optimizer_steps: int,
    donor_scratch_dir: Path | None = None,
    probe_before_training: bool = False,
    condition: str = "cold",
    donor_checkpoint_path: str | None = None,
    donor_checkpoint_sha256: str | None = None,
    donor_key_overlap: dict | None = None,
) -> tuple[object, dict]:
    """Run one seed of *arm*: fresh adapter (cold or warm) -> train -> eval -> save.

    Steps:
      1. Unwrap base model if currently wrapped (CLAUDE.md: never
         ``delete_adapter`` then ``create_adapter``; unwrap instead). This
         also discards any donor adapter loaded in a prior seed — the
         donor is reloaded fresh from *donor_scratch_dir* below so every
         seed sees byte-identical donor weights regardless of what
         happened in earlier seeds.
      1b. (Warm arm only, ``donor_scratch_dir is not None``) Load the donor
          adapter from *donor_scratch_dir* — NEVER the caller's original
          path — under the reserved name ``DONOR_ADAPTER_NAME``
          (``"donor"``), via ``_adapter_slot_for_load`` (transparent
          decrypt) + ``PeftModel.from_pretrained`` (``is_trainable=False``
          by default — the donor never receives gradients).
      2. ``torch.manual_seed(seed)`` immediately before ``create_adapter`` —
         production LoRA init is unseeded (``paramem/models/loader.py:486``).
      3. ``create_adapter`` -> fresh LoRA-zero trainable adapter;
         ``switch_adapter``. Donor-immutability guard: the trainable
         adapter name is asserted to never collide with a live tier name.
      3b. (Warm arm only) ``copy_adapter_weights(model, src="donor",
          dst=adapter_name)`` — copies the donor's LoRA weights into the
          trainable adapter BEFORE training.
      4. Hard Assertion #3: LoRA-B Frobenius norm immediately before
         training — ``== 0.0`` for the cold arm (proves cold init),
         ``> 0.0`` for the warm arm (proves the donor copy landed). (Warm
         arm only) donor LoRA-B norm captured here as the pre-training
         donor-immutability baseline.
      4b. (``--probe-before-training`` only) Mechanism probe: run
          ``_run_recall_probe`` (the SAME helper Step 11 uses) on the
          just-created/copied adapter, strictly BEFORE any training call —
          tests whether the adapter already emits well-formed JSON echoing
          the correct key (format learned) with a wrong object (binding not
          yet learned). Torch CPU + CUDA RNG state is snapshotted
          immediately before and restored immediately after so the probe's
          ``generate()`` calls never perturb the subsequent training run
          (bit-identical trained adapter with the flag on vs off).
      5. ``format_entry_training`` (production entry/prompt path) +
         ``IndexedDataset``.
      6. Build the per-seed ``TrainingConfig`` (``seed=seed`` on top of
         *base_training_config*, which already carries ``num_epochs``/
         ``gradient_accumulation_steps``/``lr_decay_steps`` (from
         ``paramem.utils.config.budget_for(n_entries)``, overridden by
         ``--epochs``/``--accum``/``--lr-decay-steps`` when explicit — see
         ``main()``) and ``recall_early_stopping=False``). Hard Assertion
         #2 re-checked immediately before the call.
      7. Verify the arm's own step-budget derivation
         (``_steps_per_epoch`` * ``num_epochs``) ==
         *expected_optimizer_steps* BEFORE training — a config-drift
         canary independent of the realized-step assertion below.
      8. ``train_adapter`` with ``_StepCaptureCallback`` in
         ``callbacks_extra``.
      9. Hard Assertion #1: realized optimizer steps
         (``step_cb.global_step``) == *expected_optimizer_steps*.
      10. Hard Assertion #3 (post-training half): LoRA-B Frobenius norm
          > 0.0 after training (adapter actually moved), both arms.
          (Warm arm only) Hard Assertion #4: donor LoRA-B norm is
          bit-identical to the pre-training baseline (donor immutability).
      11. ``evaluate_indexed_recall`` (gradient_checkpointing
          disable/re-enable around the call, per CLAUDE.md's generate()
          rule).
      12. Save per-key + summary + all hard-assertion values to disk.

    Args:
        model: Base model or PeftModel. Unwrapped inside if wrapped.
        tokenizer: Tokenizer matching the model.
        seed: Seed for LoRA init (``torch.manual_seed``) and the Trainer's
            data order (``TrainingConfig.seed``).
        entries: The N-key entry set (synthetic or loaded via
            ``--entries-json``; identical across seeds).
        registry: SimHash registry built from *entries*.
        adapter_config: Production episodic ``AdapterConfig``
            (``cfg.episodic_adapter_config``).
        base_training_config: Production episodic ``TrainingConfig`` with
            ``num_epochs``/``gradient_accumulation_steps``/``lr_decay_steps``
            (from ``budget_for(n_entries)``, overridden by
            ``--epochs``/``--accum``/``--lr-decay-steps`` when explicit)
            and ``recall_early_stopping=False`` already applied (seed is
            the only remaining per-call override).
        run_dir: Run output directory (already arm-scoped); ``seed<N>/``
            subdir created here.
        arm: Arm label (``--arm``, or the derived default) — used in the
            per-seed adapter name so concurrent/sequential arms never
            collide, and recorded in results.json.
        expected_optimizer_steps: Derived expected optimizer-step count
            for this arm (``_expected_optimizer_steps(n_entries, epochs)``).
        donor_scratch_dir: When set (warm arm), the run-scratch copy of
            the ``--warm-from`` donor adapter directory (already
            ``shutil.copytree``'d by ``main()`` — never the caller's
            original path). ``None`` (default) runs the cold arm
            unchanged.
        probe_before_training: When True, run the mechanism probe (Step 4b)
            on the freshly-created adapter before ``train_adapter`` is
            called, saving the full result under ``pre_training_probe`` in
            ``results.json``. Defaults to False so prior arms reproduce
            exactly (no RNG snapshot/restore or extra ``generate()`` calls).
        condition: Descriptive (never letter-labeled) condition name for
            this arm (``_condition_label``, e.g. ``"donor-init 30ep"``,
            ``"cold 50ep"``) — recorded verbatim in ``results.json`` under
            ``condition``. Defaults to ``"cold"`` (bare mode name, only used
            if a caller ever invokes ``_run_seed`` without going through
            ``main()``'s resolution).
        donor_checkpoint_path: Informative path to the donor checkpoint slot
            that seeded ``donor_scratch_dir`` (``--warm-from`` or
            ``--donor-init``'s resolved/built slot) — ``None`` for the cold
            arm. Recorded verbatim in ``results.json``; never re-opened here
            (the scratch copy is what Step 1b actually loads).
        donor_checkpoint_sha256: SHA-256 of the donor checkpoint's
            ``adapter_model.safetensors``, computed by the caller from the
            *scratch* copy (never re-reading ``--warm-from``'s original path
            a second time). ``None`` for the cold arm.
        donor_key_overlap: (``--donor-init`` only; ``None`` for cold and
            plain ``--warm-from`` arms) ``{"count": int, "donor_objects":
            {key: object}}`` — the H1 confound record: how many of THIS
            arm's target keys are also present in the donor's own synthetic
            population, and what object the donor trained for each
            overlapping key. The donor's block-0 is bit-identical to the
            fixed 21-key fixture (``paramem.training.donor``'s module
            docstring), so a real-production-key arm's overlap is often the
            FULL 21 — the donor may pre-install key -> subject/predicate
            scaffolding (with a DIFFERENT, donor-fictional object) for
            exactly the keys this arm re-trains, so any donor-arm uplift
            measures the overlapping-band store, not generalization to
            fresh keys (>= 201, outside the donor's reserved band). The
            rate metric itself is unaffected (strict SPO exact-match), only
            its ATTRIBUTION is — recorded here so it is never silently
            assumed away.

    Returns:
        Tuple of ``(model, summary_dict)``. ``model`` is the PeftModel
        after training (caller should unwrap before the next seed).
    """
    seed_dir = run_dir / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    is_warm = donor_scratch_dir is not None

    # Step 1: unwrap if wrapped (discards any prior seed's donor + trainable
    # adapters — the donor is reloaded fresh below for a byte-identical copy).
    if isinstance(model, PeftModel):
        model = model.base_model.model

    # Step 1b: (warm arm) load the donor fresh from the immutable scratch
    # copy. Model is guaranteed unwrapped (raw base) here, so this always
    # takes the PeftModel.from_pretrained branch (is_trainable=False
    # default — the donor never receives gradients).
    if is_warm:
        with _adapter_slot_for_load(donor_scratch_dir) as load_path:
            if isinstance(model, PeftModel):
                model.load_adapter(str(load_path), adapter_name=DONOR_ADAPTER_NAME)
            else:
                model = PeftModel.from_pretrained(
                    model, str(load_path), adapter_name=DONOR_ADAPTER_NAME
                )

    # Step 2+3: seed LoRA init, create fresh cold trainable adapter, switch to it.
    adapter_name = f"episodic_{arm}_seed{seed}"
    assert adapter_name not in LIVE_TIER_NAMES, (
        f"Donor-immutability guard FAILED: trainable adapter name '{adapter_name}' "
        f"collides with a live tier name {sorted(LIVE_TIER_NAMES)} — refusing to "
        "risk overwriting a production adapter slot."
    )
    torch.manual_seed(seed)
    model = create_adapter(model, adapter_config, adapter_name)
    switch_adapter(model, adapter_name)

    # Step 3b: (warm arm) copy donor LoRA weights into the trainable adapter
    # BEFORE training — the staging+promote path inside train_adapter then
    # starts from these weights instead of LoRA-zero.
    donor_lora_b_norm_before: float | None = None
    if is_warm:
        copy_adapter_weights(model, src=DONOR_ADAPTER_NAME, dst=adapter_name)
        donor_lora_b_norm_before = lora_b_frobenius_norm(model, DONOR_ADAPTER_NAME)

    # Step 4: Hard Assertion #3 — cold init proof (cold arm) / warm-copy
    # proof (warm arm), pre-training.
    lora_b_norm_before = lora_b_frobenius_norm(model, adapter_name)
    if is_warm:
        assert lora_b_norm_before == donor_lora_b_norm_before, (
            f"Hard Assertion #3 FAILED (pre-training, warm arm): trainable adapter "
            f"'{adapter_name}' LoRA-B Frobenius norm ({lora_b_norm_before}) != donor "
            f"'{DONOR_ADAPTER_NAME}' norm ({donor_lora_b_norm_before}) immediately "
            "after copy_adapter_weights — the seed did not take (or took a "
            "corrupted copy)."
        )
    else:
        assert lora_b_norm_before == 0.0, (
            f"Hard Assertion #3 FAILED (pre-training): LoRA-B Frobenius norm for "
            f"'{adapter_name}' is {lora_b_norm_before}, expected 0.0 (cold init). "
            "create_adapter did not produce a fresh zero-initialised adapter."
        )

    # Step 4b: (--probe-before-training only) mechanism probe on the
    # freshly-created/copied adapter, strictly BEFORE train_adapter is
    # called. RNG state is snapshotted/restored around the call so the
    # probe's generate() calls never alter the subsequent training run —
    # with the flag OFF vs ON, the trained adapter is identical.
    pre_training_probe: dict | None = None
    if probe_before_training:
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        pre_training_probe = _run_recall_probe(model, tokenizer, entries, registry, adapter_name)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        logger.info(
            "Pre-training probe [%s seed %d]: %d/%d exact (rate=%.1f%%), mean_confidence=%.3f",
            adapter_name,
            seed,
            pre_training_probe["exact_count"],
            pre_training_probe["total"],
            pre_training_probe["rate"] * 100,
            pre_training_probe["mean_confidence"],
        )

    # Step 5: production entry/prompt path.
    examples = format_entry_training(
        entries, tokenizer, max_length=base_training_config.max_seq_length
    )
    dataset = IndexedDataset(examples)

    # Step 6: per-seed training config; Hard Assertion #2.
    training_cfg = dataclasses.replace(base_training_config, seed=seed)
    assert training_cfg.recall_early_stopping is False, (
        f"Hard Assertion #2 FAILED: training_config.recall_early_stopping="
        f"{training_cfg.recall_early_stopping}, expected False."
    )

    # Step 7: pre-training step-budget canary.
    n_examples = len(examples)
    spe = _steps_per_epoch(
        n_examples, training_cfg.batch_size, training_cfg.gradient_accumulation_steps
    )
    derived_expected_steps = spe * training_cfg.num_epochs
    assert derived_expected_steps == expected_optimizer_steps, (
        f"Config drift: N={n_examples} at batch={training_cfg.batch_size}, "
        f"accum={training_cfg.gradient_accumulation_steps}, "
        f"num_epochs={training_cfg.num_epochs} yields "
        f"{derived_expected_steps} optimizer steps, expected {expected_optimizer_steps} "
        f"(arm={arm!r})."
    )

    logger.info(
        "Seed %d [%s]: %d entries, adapter=%s, epochs=%d, steps_per_epoch=%d, "
        "expected_total_steps=%d, lr=%.0e",
        seed,
        arm,
        n_examples,
        adapter_name,
        training_cfg.num_epochs,
        spe,
        derived_expected_steps,
        adapter_config.learning_rate,
    )

    # Step 8: train.
    step_cb = _StepCaptureCallback()
    t0 = time.time()
    metrics = train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        adapter_name=adapter_name,
        training_config=training_cfg,
        adapter_config=adapter_config,
        output_dir=seed_dir,
        run_name=f"test20-{arm}-seed{seed}",
        callbacks_extra=[step_cb],
    )
    wall_train = time.time() - t0
    train_loss = (metrics or {}).get("train_loss")
    logger.info("Seed %d training done: wall=%.0fs loss=%s", seed, wall_train, train_loss)

    # Step 9: Hard Assertion #1 — realized optimizer steps.
    realized_steps = step_cb.global_step
    assert realized_steps is not None, (
        f"Hard Assertion #1 FAILED: _StepCaptureCallback never fired "
        f"on_train_end for seed {seed} — no realized step count captured."
    )
    assert realized_steps == expected_optimizer_steps, (
        f"Hard Assertion #1 FAILED: realized optimizer steps={realized_steps}, "
        f"expected {expected_optimizer_steps} (arm={arm!r})."
    )

    # Step 10: Hard Assertion #3 (post-training half) — adapter moved, both arms.
    lora_b_norm_after = lora_b_frobenius_norm(model, adapter_name)
    assert lora_b_norm_after > 0.0, (
        f"Hard Assertion #3 FAILED (post-training): LoRA-B Frobenius norm for "
        f"'{adapter_name}' is {lora_b_norm_after}, expected > 0.0 (adapter moved)."
    )

    # Step 10b: (warm arm) Hard Assertion #4 — donor immutability. The donor
    # is never the staging or production slot for this training event, so
    # its LoRA-B norm must be bit-identical before and after.
    donor_lora_b_norm_after: float | None = None
    if is_warm:
        donor_lora_b_norm_after = lora_b_frobenius_norm(model, DONOR_ADAPTER_NAME)
        assert donor_lora_b_norm_after == donor_lora_b_norm_before, (
            f"Hard Assertion #4 FAILED (donor immutability): donor LoRA-B Frobenius "
            f"norm changed from {donor_lora_b_norm_before} to {donor_lora_b_norm_after} "
            f"during seed {seed} training. The donor adapter must never be mutated."
        )

    # Step 11: evaluate (CLAUDE.md: disable gradient_checkpointing before
    # generate(), re-enable afterward) — same helper as the optional Step 4b
    # pre-training probe, no duplicated probe logic.
    recall = _run_recall_probe(model, tokenizer, entries, registry, adapter_name)

    # Step 12: save.
    full_results = {
        "arm": arm,
        "condition": condition,
        "seed": seed,
        "n_entries": n_examples,
        "epochs": training_cfg.num_epochs,
        "lr_decay_steps": training_cfg.lr_decay_steps,
        "adapter_name": adapter_name,
        "warm_start": is_warm,
        "donor_checkpoint_path": donor_checkpoint_path,
        "donor_checkpoint_sha256": donor_checkpoint_sha256,
        "donor_key_overlap": donor_key_overlap,
        "expected_optimizer_steps": expected_optimizer_steps,
        "realized_optimizer_steps": realized_steps,
        "steps_per_epoch": spe,
        "training_config": dataclasses.asdict(training_cfg),
        "adapter_config": {
            "rank": adapter_config.rank,
            "alpha": adapter_config.alpha,
            "learning_rate": adapter_config.learning_rate,
            "target_modules": adapter_config.target_modules,
            "dropout": adapter_config.dropout,
        },
        "lora_b_norm_before_training": lora_b_norm_before,
        "lora_b_norm_after_training": lora_b_norm_after,
        "donor_lora_b_norm_before_training": donor_lora_b_norm_before,
        "donor_lora_b_norm_after_training": donor_lora_b_norm_after,
        "train_loss": train_loss,
        "wall_train_seconds": wall_train,
        "pre_training_probe": pre_training_probe,
        "summary": {
            "exact_count": recall["exact_count"],
            "total": recall["total"],
            "rate": recall["rate"],
            "mean_confidence": recall["mean_confidence"],
        },
        "per_key": recall["per_key"],
    }
    save_results(full_results, seed_dir, filename="results.json")

    summary = {
        "exact_count": recall["exact_count"],
        "total": recall["total"],
        "rate": recall["rate"],
        "mean_confidence": recall["mean_confidence"],
        "realized_optimizer_steps": realized_steps,
        "warm_start": is_warm,
        "lora_b_norm_before_training": lora_b_norm_before,
        "lora_b_norm_after_training": lora_b_norm_after,
        "donor_lora_b_norm_before_training": donor_lora_b_norm_before,
        "donor_lora_b_norm_after_training": donor_lora_b_norm_after,
        "train_loss": train_loss,
        "wall_train_seconds": wall_train,
    }
    return model, summary


# ---------------------------------------------------------------------------
# Run-dir helpers
# ---------------------------------------------------------------------------


def _find_latest_run_dir(arm_base: Path, model_name: str) -> Path | None:
    """Return the most recent run dir for *model_name* under *arm_base*.

    Args:
        arm_base: Arm-scoped output base (``OUTPUT_BASE / arm``) so
            ``--resume`` never crosses arms.
        model_name: Model key (e.g. ``"mistral"``).

    Returns:
        Path to the latest run dir, or None if none found.
    """
    parent = arm_base / model_name
    if not parent.is_dir():
        return None
    candidates = sorted(
        [d for d in parent.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )
    return candidates[-1] if candidates else None


def _preflight_run_dir(arm_base: Path, model_name: str, resume: bool) -> Path:
    """Resolve this run's output directory, verifying disk headroom first.

    Shared by ``main()`` (per-arm loop) and ``_main_donor_build_smoke``
    (fixed procedural arm) — resolves ``--resume``'s "latest run dir for
    this arm/model" lookup via :func:`_find_latest_run_dir`, then verifies
    free space in ``OUTPUT_BASE`` against ``DISK_HEADROOM_BYTES``. The run
    directory itself is created only AFTER that check passes, so a
    disk-full failure never leaves an empty run dir behind.

    Args:
        arm_base: Arm-scoped output base (``OUTPUT_BASE / arm``).
        model_name: Model key (e.g. ``"mistral"``).
        resume: Whether ``--resume`` was passed.

    Returns:
        The resolved run directory, created and ready to write into.

    Raises:
        SystemExit: Free space in ``OUTPUT_BASE`` is at or below
            ``DISK_HEADROOM_BYTES``.
    """
    if resume:
        latest = _find_latest_run_dir(arm_base, model_name)
        if latest is None:
            logger.warning("--resume: no prior run found under %s — starting fresh", arm_base)
            run_dir = model_output_dir(arm_base, model_name)
        else:
            run_dir = latest
            logger.info("Resuming from %s", run_dir)
    else:
        run_dir = model_output_dir(arm_base, model_name)

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(OUTPUT_BASE).free
    if free_bytes <= DISK_HEADROOM_BYTES:
        raise SystemExit(
            f"Insufficient disk space: {free_bytes / 1024**3:.1f} GB free in {OUTPUT_BASE}; "
            f"need > {DISK_HEADROOM_BYTES / 1024**3:.0f} GB."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Compact summary printer
# ---------------------------------------------------------------------------


def _print_summary_from_results(seed: int, results_path: Path) -> None:
    """Print a compact per-seed recall summary from a saved results file.

    Lists missed keys and what was recalled instead.

    Args:
        seed: Seed value.
        results_path: Path to the seed's ``results.json``.
    """
    if not results_path.exists():
        print(f"\n[SEED {seed}] results.json not found at {results_path}")
        return
    with open(results_path) as f:
        data = json.load(f)
    summary = data["summary"]
    rate = summary["rate"]
    exact = summary["exact_count"]
    total = summary["total"]
    print(f"\n[SEED {seed}] Recall: {exact}/{total} = {rate:.1%}")
    print(
        f"  mean_confidence={summary['mean_confidence']:.3f}  "
        f"train_loss={data.get('train_loss')}  "
        f"realized_steps={data.get('realized_optimizer_steps')}  "
        f"lora_b_norm(before/after)="
        f"{data.get('lora_b_norm_before_training'):.6f}/"
        f"{data.get('lora_b_norm_after_training'):.6f}"
    )
    if data.get("warm_start"):
        donor_before = data.get("donor_lora_b_norm_before_training")
        donor_after = data.get("donor_lora_b_norm_after_training")
        print(
            f"  warm_start=True  donor_lora_b_norm(before/after)="
            f"{donor_before:.6f}/{donor_after:.6f}  "
            f"(bit-identical={'OK' if donor_before == donor_after else 'VIOLATED'})"
        )
    misses = [r for r in data["per_key"] if not r["exact_match"]]
    if misses:
        print(f"  Missed ({len(misses)}):")
        for m in misses:
            recalled_obj = m.get("recalled_object", "?")
            print(f"    {m['key']:8s}  expected={m['object']!r:35s}  got={recalled_obj!r}")
    else:
        print("  All keys recalled correctly.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build test20's CLI ``ArgumentParser``.

    Factored out of :func:`_parse_args` so ``main()``'s
    ``--donor-build-smoke`` flag-conflict guard can derive "is this flag
    set" from the SAME parser's own defaults (``parser.parse_args([])``)
    rather than maintaining a second, hand-written mirror of the flag list
    that silently goes stale when a new flag is added.

    Returns:
        A fresh, fully-configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Test 20: Small-N Cold Indexed-Key Recall Gate (production recipe)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistral",
        choices=list(BENCHMARK_MODELS.keys()),
        help="Model to run (default: mistral — the production model this gate targets)",
    )
    parser.add_argument(
        "--n-entries",
        type=int,
        default=None,
        help=(
            "Number of synthetic keys in the arm (default: 12, the arm this script "
            "originally shipped with; ignored/validated against --entries-json when "
            "that flag is set). The ORIGINAL failure condition to reproduce is "
            "--n-entries 3."
        ),
    )
    parser.add_argument(
        "--entries-json",
        type=str,
        default=None,
        help=(
            "Path to a JSON file of explicit [{key, subject, predicate, object}, ...] "
            "entries, replacing the synthetic generator entirely (e.g. "
            "experiments/fixtures/real3_interim_failure.json — the exact three "
            "episodic triples that failed in production). --n-entries is then "
            "implied by the file length; a conflicting --n-entries fails loud."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Epoch budget, overriding the default derived from "
            "paramem.utils.config.budget_for(n_entries) (the SAME per-fold funnel "
            "production training calls on every fold) for the resolved entry count. "
            "Default None uses budget_for's derived epochs. Combined with the resolved "
            "accum/batch_size this derives the expected optimizer-step count via "
            "_expected_optimizer_steps."
        ),
    )
    parser.add_argument(
        "--warm-from",
        type=str,
        default=None,
        help=(
            "Path to a donor adapter directory (adapter_config.json + "
            "adapter_model.safetensors) to warm-start the trainable adapter from, "
            "instead of LoRA-zero cold init. Copied into the run's scratch dir via "
            "shutil.copytree before loading — the original path is never opened "
            "for anything else (donor immutability; see module docstring)."
        ),
    )
    parser.add_argument(
        "--arm",
        type=str,
        default=None,
        help=(
            "Arm label used in the output path (default derived from the resolved "
            "config — see _default_arm_label — so distinct arms never collide)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Auto-find the latest run dir for the resolved --arm and skip completed seeds.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "One or more seeds to run, overriding the module default SEEDS=(0, 1, 2) "
            "(e.g. --seeds 42 runs a single seed matching the production training "
            "seed; --seeds 0 1 2 reproduces the default 3-seed gate explicitly). "
            "Omitting this flag is byte-identical to today's behaviour. The "
            "effective seed list (not the SEEDS module constant) is what lands in "
            "run_config.json and the final cross-seed summary; --resume skips "
            "whichever of the effective seeds already have a seed<N>_done.json "
            "marker in the resolved --arm's run dir."
        ),
    )
    parser.add_argument(
        "--probe-before-training",
        action="store_true",
        help=(
            "Run the recall probe (_run_recall_probe, the same helper used post-training) "
            "on the freshly-created adapter ONCE per seed, strictly BEFORE train_adapter is "
            "called — the format-vs-binding mechanism test. Full result (rate, exact_count, "
            "total, per_key with raw_output) saved under results.json's pre_training_probe "
            "key. Torch RNG state is snapshotted/restored around the probe so the trained "
            "adapter is unaffected. Defaults to OFF so prior arms reproduce exactly."
        ),
    )
    parser.add_argument(
        "--lr-decay-steps",
        type=int,
        default=None,
        help=(
            "Pin TrainingConfig.lr_decay_steps to this many steps, overriding the default "
            "derived from paramem.utils.config.budget_for(n_entries) (None for every bucket in "
            "today's _BUDGET_TABLE — create_scheduler's no-op passthrough, decay window derived "
            "from len(dataloader) * num_epochs). The approved decay-pinned validation protocol "
            "pins decay explicitly so arms at different --epochs stay comparable (decay shape "
            "decoupled from the epoch budget). Default None uses budget_for's derived value."
        ),
    )
    parser.add_argument(
        "--accum",
        type=int,
        default=None,
        help=(
            "Override TrainingConfig.gradient_accumulation_steps, threaded into "
            "base_training_config the same way as --epochs / --lr-decay-steps. Default None "
            "uses the default derived from paramem.utils.config.budget_for(n_entries) (the SAME "
            "per-fold funnel production training calls on every fold) instead of a hardcoded "
            "module constant — the fixture no longer carries a gradient_accumulation_steps "
            "field. Must be >= 1. _expected_optimizer_steps takes the same resolved value so "
            "the derived step count and the Step 7 drift canary in _run_seed stay consistent "
            "with the actual training run. _build_donor_checkpoint always derives the donor's "
            "own accum from budget_for(len(donor_entries)) regardless of this flag — see that "
            "function's docstring."
        ),
    )
    parser.add_argument(
        "--donor-init",
        action="store_true",
        help=(
            "Seed the trainable adapter from a donor checkpoint built by training "
            "paramem.training.donor.donor_entries (128 synthetic keys, seed=DONOR_DEFAULT_SEED) "
            "through this script's OWN train_adapter call path, at the epoch/accum budget "
            "paramem.utils.config.budget_for(len(donor_entries)) derives (30 epochs, accum 2 for "
            "the resulting 147-entry population) — NOT via paramem.training.donor.build_donor "
            "(that helper needs a live ConsolidationLoop; this standalone experiment trains the "
            "donor itself). Feeds the SAME donor_scratch_dir / copy_adapter_weights mechanism "
            "--warm-from already uses in _run_seed (Hard Assertions #3/#4 apply unchanged). "
            "Mutually exclusive with --warm-from (ambiguous donor source)."
        ),
    )
    parser.add_argument(
        "--donor-checkpoint",
        type=str,
        default=None,
        help=(
            "Path to an existing donor checkpoint SLOT directory (containing "
            "adapter_model.safetensors) from a prior --donor-init run's "
            "<run_dir>/donor_checkpoint/<ts>/ — reuses it instead of building a fresh donor "
            "(the donor builds once, arms reuse it). Requires --donor-init; without it, this "
            "run builds (or resumes building) its own donor checkpoint under "
            "<run_dir>/donor_checkpoint/, recorded in donor_build_done.json for --resume."
        ),
    )
    parser.add_argument(
        "--donor-build-smoke",
        action="store_true",
        help=(
            "GPU feasibility/cost measurement for the per-topology donor build: build ONE "
            "donor checkpoint at the PROCEDURAL topology "
            "(tests/fixtures/server.yaml's procedural_adapter_config — rank/alpha/"
            "target_modules, never hardcoded) at DONOR_RECIPE_LEARNING_RATE/"
            "DONOR_RECIPE_DROPOUT and budget_for(len(donor_entries))'s derived epoch/accum "
            "budget, then cold-seed a fresh procedural adapter from it via the SAME strict "
            "copy_adapter_weights call --warm-from/--donor-init use. Measures wall time, mean "
            "seconds/optimizer step, and VRAM (torch.cuda.mem_get_info before load/after "
            "load/after build/after seed; torch.cuda.max_memory_allocated/max_memory_reserved "
            "for the build phase). Does NOT "
            "run recall evaluation on the seeded adapter and does NOT exercise production "
            "paramem.training.donor.build_donor. Mutually exclusive with every other "
            "arm-configuration flag (--n-entries/--entries-json/--epochs/--warm-from/"
            "--donor-init/--donor-checkpoint/--arm/--seeds/--probe-before-training/"
            "--lr-decay-steps/--accum) — only --model/--resume apply alongside it. Exits "
            "non-zero if the recorded seed phase result is a measured failure. See "
            "_main_donor_build_smoke for the full mechanism."
        ),
    )
    return parser


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for test20.

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    return _build_arg_parser().parse_args()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for Test 20.

    Loads ``tests/fixtures/server.yaml`` for ``episodic_adapter_config``
    (rank/alpha/lr/target_modules) and ``training_config``'s fixture-sourced
    fields (batch_size/max_seq_length/warmup/scheduler/weight_decay/
    gradient_checkpointing/max_grad_norm) — this load has no GPU dependency,
    so it happens BEFORE ``acquire_gpu``. ``num_epochs``/
    ``gradient_accumulation_steps``/``lr_decay_steps`` are NOT fixture
    fields: they are derived per-fold from
    ``paramem.utils.config.budget_for(n_entries)`` (the SAME function
    production's per-fold funnel calls unconditionally), overridden by
    ``--epochs``/``--accum``/``--lr-decay-steps`` when explicit —
    see the module docstring's "Recipe fidelity" section. Then cycles
    through the resolved seeds of the resolved arm (``--n-entries`` /
    ``--entries-json`` / ``--epochs`` / ``--lr-decay-steps`` / ``--accum`` /
    ``--warm-from`` / ``--donor-init`` / ``--donor-checkpoint`` / ``--arm`` /
    ``--probe-before-training`` / ``--seeds``) with per-seed adapter
    isolation, and writes per-seed results + done markers under an
    arm-scoped output subtree. ``--seeds`` overrides the module default
    ``SEEDS = (0, 1, 2)`` (e.g. ``--seeds 42`` for a single production-seed
    run); omitting it reproduces today's 3-seed behaviour exactly, and the
    effective seed list — never the module constant — is what is logged,
    written to ``run_config.json``, and iterated by ``--resume``.

    ``--resume`` (or any repeat invocation landing on an existing run dir)
    rebuilds the run-config dict from the CURRENT invocation's resolved
    args/derived values and compares it against the existing
    ``run_config.json`` on the fields that determine the training budget
    (``epochs``, ``accum``, ``lr_decay_steps``, ``warm_from``,
    ``donor_init``, ``donor_checkpoint``) — a mismatch fails loud rather
    than silently mixing seeds trained under different conditions into one
    result set.

    ``--warm-from`` and ``--donor-init`` are mutually exclusive donor
    sources (fails loud if both are set) feeding the SAME
    ``donor_scratch_dir`` / ``copy_adapter_weights`` mechanism in
    ``_run_seed``: ``--warm-from`` copies an arbitrary owner-supplied
    adapter dir; ``--donor-init`` resolves (reusing ``--donor-checkpoint``
    when given, or building fresh via ``_build_donor_checkpoint``) a donor
    checkpoint trained on ``paramem.training.donor.donor_entries``. Either
    way the donor's weights are copied into ``<run_dir>/donor_scratch/``
    ONCE per run and never re-touched afterward.

    GPU cooldown is inserted between seeds. The ``~/.training_pause`` gate
    is honoured at every seed boundary.

    Designed to run daemonised (setsid/nohup) — no terminal interaction;
    logs are flushed after every seed.

    ``--donor-build-smoke`` is a SEPARATE, standalone entry point
    (:func:`_main_donor_build_smoke`) dispatched before any of the
    ``--n-entries``/``--entries-json``/arm-loop logic below runs — it never
    reaches the seed loop or its recall evaluation. Every other
    arm-configuration flag is validated as mutually exclusive with it here,
    failing loud rather than silently ignoring a flag the smoke mode does
    not use.
    """
    # Set CUDA alloc config before any torch import side effects matter.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    args = _parse_args()

    if args.donor_build_smoke:
        # Derived from the parser's OWN defaults (vars(args) vs a
        # parse_args([]) baseline of the SAME parser) rather than a
        # hand-maintained mirror of the flag list — every dest except the
        # allowed trio is a conflicting flag, so a newly-added flag is
        # covered automatically instead of silently bypassing this guard.
        _allowed_with_donor_build_smoke = {"model", "resume", "donor_build_smoke"}
        _defaults = vars(_build_arg_parser().parse_args([]))
        _set_flags = sorted(
            f"--{dest.replace('_', '-')}"
            for dest, value in vars(args).items()
            if dest not in _allowed_with_donor_build_smoke and value != _defaults[dest]
        )
        if _set_flags:
            raise SystemExit(
                "--donor-build-smoke is a standalone GPU feasibility/cost probe and is "
                f"mutually exclusive with every arm-configuration flag; got {_set_flags} set "
                "alongside it. Drop them — only --model/--resume apply."
            )
        _main_donor_build_smoke(args)
        return

    # Resolve the effective seed set: --seeds overrides the module default
    # SEEDS=(0, 1, 2) (e.g. --seeds 42 for the production training seed);
    # omitting the flag reproduces today's behaviour exactly. Everything
    # downstream (logging, run_config.json, --resume, the seed loop, the
    # final cross-seed summary) reads `seeds`, never the SEEDS constant.
    seeds = tuple(args.seeds) if args.seeds else SEEDS

    # Resolve the entry set: --entries-json replaces the synthetic
    # generator entirely and implies --n-entries from the file length
    # (a conflicting --n-entries fails loud).
    is_real = args.entries_json is not None
    if is_real:
        entries = _load_entries_from_file(Path(args.entries_json))
        if args.n_entries is not None and args.n_entries != len(entries):
            raise SystemExit(
                f"--n-entries={args.n_entries} disagrees with --entries-json file "
                f"length ({len(entries)} entries at {args.entries_json}). Omit "
                "--n-entries or set it to match the file."
            )
        n_entries = len(entries)
    else:
        n_entries = args.n_entries if args.n_entries is not None else DEFAULT_N_ENTRIES
        entries = _build_entries(n_entries)

    # Donor-source validation: --donor-checkpoint only means anything with
    # --donor-init, and --warm-from / --donor-init are mutually exclusive
    # (each resolves donor_scratch_dir from a different source — ambiguous
    # if both are set).
    if args.donor_checkpoint is not None and not args.donor_init:
        raise SystemExit("--donor-checkpoint requires --donor-init.")
    if args.warm_from is not None and args.donor_init:
        raise SystemExit(
            "--warm-from and --donor-init are mutually exclusive donor sources "
            "(both populate donor_scratch_dir) — pick one."
        )
    # --accum must be a valid gradient_accumulation_steps value.
    if args.accum is not None and args.accum < 1:
        raise SystemExit(f"--accum must be >= 1; got {args.accum}.")

    # Load the production episodic recipe from the test fixture — NEVER
    # load_config() / configs/server.yaml.example. Pure YAML parse, no GPU
    # dependency, so this happens before acquire_gpu (needed for the
    # expected-step derivation below, which now uses the fixture's ACTUAL
    # batch_size rather than a hardcoded constant).
    cfg = load_server_config(str(FIXTURE_CONFIG_PATH))
    adapter_config = cfg.episodic_adapter_config

    # epochs/accum/lr_decay_steps are DERIVED per-fold via budget_for
    # (paramem.utils.config) — the SAME function production's per-fold
    # funnel (ConsolidationLoop._train_tier_adapter) calls unconditionally
    # — never a hardcoded module constant. --epochs/--accum/--lr-decay-steps
    # override the derived default explicitly when passed.
    budget_epochs, budget_accum, budget_lr_decay_steps = budget_for(n_entries)
    epochs = args.epochs if args.epochs is not None else budget_epochs
    accum = args.accum if args.accum is not None else budget_accum
    lr_decay_steps = (
        args.lr_decay_steps if args.lr_decay_steps is not None else budget_lr_decay_steps
    )

    base_training_config = dataclasses.replace(
        cfg.training_config,
        num_epochs=epochs,
        gradient_accumulation_steps=accum,
        lr_decay_steps=lr_decay_steps,
        recall_early_stopping=False,
    )
    expected_steps = _expected_optimizer_steps(
        n_entries, epochs, accum, base_training_config.batch_size
    )
    logger.info(
        "Recipe: rank=%d alpha=%d lr=%.0e target_modules=%s | "
        "batch=%d accum=%d epochs=%d warmup=%d scheduler=%s wd=%.2f "
        "lr_decay_steps=%s recall_early_stopping=%s | budget_for(%d)=(%d, %d, %s)",
        adapter_config.rank,
        adapter_config.alpha,
        adapter_config.learning_rate,
        adapter_config.target_modules,
        base_training_config.batch_size,
        base_training_config.gradient_accumulation_steps,
        base_training_config.num_epochs,
        base_training_config.warmup_steps,
        base_training_config.lr_scheduler_type,
        base_training_config.weight_decay,
        base_training_config.lr_decay_steps,
        base_training_config.recall_early_stopping,
        n_entries,
        budget_epochs,
        budget_accum,
        budget_lr_decay_steps,
    )

    is_warm = args.warm_from is not None or args.donor_init
    mode = "donor" if args.donor_init else ("warm" if args.warm_from is not None else "cold")
    condition = _condition_label(mode, epochs)
    arm = args.arm or _default_arm_label(n_entries, expected_steps, is_real, mode)

    # Resolve output dir (arm-scoped so distinct arms never collide and
    # --resume never crosses arms) and verify disk headroom (mirrors
    # test16/test19 convention: free-space, not total-usage) before the
    # run dir is created — shared with _main_donor_build_smoke.
    arm_base = OUTPUT_BASE / arm
    run_dir = _preflight_run_dir(arm_base, args.model, args.resume)

    registry = build_registry(entries)
    logger.info(
        "%s arm [%s]: N=%d %s keys, epochs=%d, accum=%d, expected_optimizer_steps=%d, "
        "warm_start=%s, seeds=%s",
        arm,
        condition,
        len(entries),
        "real" if is_real else "synthetic",
        epochs,
        accum,
        expected_steps,
        is_warm,
        seeds,
    )

    # Rebuild the run-config dict from THIS invocation's resolved
    # args/derived values and compare against an existing run_config.json
    # (--resume, or any repeat invocation landing on the same run dir) —
    # fail loud on any mismatch in the fields that determine the training
    # budget rather than silently mixing conditions into one result set.
    run_config = {
        "model": args.model,
        "arm": arm,
        "condition": condition,
        "seeds": list(seeds),
        "n_entries": n_entries,
        "entries_json": args.entries_json,
        "epochs": epochs,
        "epochs_explicit": args.epochs is not None,
        "accum": accum,
        "accum_explicit": args.accum is not None,
        "lr_decay_steps": lr_decay_steps,
        "lr_decay_steps_explicit": args.lr_decay_steps is not None,
        "budget_for_n_entries": [budget_epochs, budget_accum, budget_lr_decay_steps],
        "expected_optimizer_steps": expected_steps,
        "warm_from": args.warm_from,
        "donor_init": args.donor_init,
        "donor_checkpoint": args.donor_checkpoint,
        "probe_before_training": args.probe_before_training,
        "recipe_source": (
            "tests/fixtures/server.yaml (episodic_adapter_config; training_config's "
            "batch_size/max_seq_length/warmup_steps/lr_scheduler_type/weight_decay/"
            "gradient_checkpointing/max_grad_norm); num_epochs/gradient_accumulation_steps/"
            "lr_decay_steps derived per paramem.utils.config.budget_for(n_entries), "
            "overridden by --epochs/--accum/--lr-decay-steps when explicit"
        ),
        "overrides": {
            "num_epochs": epochs,
            "gradient_accumulation_steps": accum,
            "lr_decay_steps": lr_decay_steps,
            "recall_early_stopping": False,
        },
    }
    run_config_path = run_dir / "run_config.json"
    if run_config_path.exists():
        with open(run_config_path) as f:
            existing_run_config = json.load(f)
        _budget_fields = (
            "epochs",
            "accum",
            "lr_decay_steps",
            "warm_from",
            "donor_init",
            "donor_checkpoint",
        )
        mismatches = {
            field: (existing_run_config.get(field), run_config[field])
            for field in _budget_fields
            if existing_run_config.get(field) != run_config[field]
        }
        if mismatches:
            raise SystemExit(
                f"Run-config mismatch at {run_config_path}: this invocation's resolved "
                f"args disagree with the existing run_config.json on "
                f"{sorted(mismatches)} (existing vs this run: {mismatches}) — refusing "
                "to mix training conditions into the same run dir. Start a fresh run "
                "(new --arm or a clean output dir) or match the original invocation "
                "exactly."
            )
        logger.info(
            "Existing run_config.json at %s matches this invocation on the training-budget "
            "fields — resuming.",
            run_config_path,
        )
    else:
        with open(run_config_path, "w") as f:
            json.dump(run_config, f, indent=2)
        logger.info("Run config written: %s", run_config_path)

    # See _main_donor_build_smoke for why this is not a module-scope import.
    from experiments.utils.gpu_guard import acquire_gpu

    model_config = BENCHMARK_MODELS[args.model]

    with acquire_gpu(interactive=True):
        model, tokenizer = load_model_and_config(model_config)

        # Donor resolution (--warm-from or --donor-init): copy the donor's
        # weights into this run's immutable scratch dir ONCE, and load the
        # donor ONLY from that copy in every seed (donor immutability — see
        # module docstring). Reused as-is on --resume (never re-copied, and
        # --donor-init never re-trains once its own checkpoint marker
        # exists — see _resolve_donor_source) so every seed of a given run
        # sees byte-identical donor weights. Runs here (inside acquire_gpu,
        # after the model loads — adapter_config/base_training_config were
        # already resolved above, before acquire_gpu) because --donor-init's
        # fresh-build path additionally needs the loaded model/tokenizer;
        # --warm-from's plain filesystem copy tags along on the same block
        # rather than duplicating the resume-marker handling.
        donor_scratch_dir: Path | None = None
        donor_checkpoint_path: str | None = None
        donor_checkpoint_sha256: str | None = None
        donor_meta: dict | None = None
        donor_built_fresh = False
        if is_warm:
            donor_scratch_dir = run_dir / "donor_scratch"
            donor_source_marker = run_dir / "donor_source.json"
            if donor_scratch_dir.exists() and donor_source_marker.exists():
                with open(donor_source_marker) as f:
                    donor_source_info = json.load(f)
                donor_checkpoint_path = donor_source_info["source"]
                donor_checkpoint_sha256 = donor_source_info["sha256"]
                logger.info("Reusing existing donor scratch copy (resume): %s", donor_scratch_dir)
                if args.donor_init:
                    # Read from the SCRATCH copy (a full copytree of the
                    # resolved slot, including donor_meta.json) — never the
                    # original --donor-checkpoint path, which may no longer
                    # be reachable by the time a run resumes.
                    donor_meta = _read_donor_meta(
                        donor_scratch_dir,
                        *_donor_verification_context(model, adapter_config),
                    )
            else:
                # B3: this is a phase boundary (donor build or a fresh
                # copytree) — honour the pause gate before starting it, not
                # just at seed boundaries.
                _check_pause("before donor build")
                if donor_scratch_dir.exists():
                    logger.warning(
                        "Donor scratch copy at %s exists without its source marker "
                        "(prior crash between copytree and marker write) — recopying.",
                        donor_scratch_dir,
                    )
                    shutil.rmtree(donor_scratch_dir)
                if args.donor_init:
                    model, donor_source, donor_built_fresh, donor_meta = _resolve_donor_source(
                        args.donor_checkpoint,
                        run_dir,
                        model,
                        tokenizer,
                        adapter_config,
                        base_training_config,
                    )
                else:
                    donor_source = Path(args.warm_from)
                logger.info("Warm-start: copying donor %s -> %s", donor_source, donor_scratch_dir)
                shutil.copytree(donor_source, donor_scratch_dir)
                donor_checkpoint_sha256 = hashlib.sha256(
                    (donor_scratch_dir / "adapter_model.safetensors").read_bytes()
                ).hexdigest()
                donor_checkpoint_path = str(donor_source)
                with open(donor_source_marker, "w") as f:
                    json.dump(
                        {"source": donor_checkpoint_path, "sha256": donor_checkpoint_sha256},
                        f,
                        indent=2,
                    )

        # H1: record the donor<->target key-overlap confound. donor_meta is
        # only set for --donor-init (never plain --warm-from, which has no
        # synthetic entry set to intersect against). The donor's OWN block-0
        # is bit-identical to the fixed fixture (paramem.training.donor's
        # module docstring: "the template itself is always block 0 ... keys
        # graph101-115 / proc101-106 ... reproduced verbatim") — recomputing
        # donor_entries(seed, n_entries) here is the SAME pure function the
        # build used, so this reconstructs the exact donor population
        # without re-reading any file. (The full-overlap donor-init arms in
        # benchmarking.md's Test 20 ran before this fixture's keys were
        # remapped, against the fixture's prior graph179-193/proc35-40
        # numerals — see that section's fixture-provenance note; the
        # zero-overlap shifted-key arms are the bridge evidence that the
        # remap does not change this mechanism's behavior.)
        donor_key_overlap: dict | None = None
        if args.donor_init and donor_meta is not None:
            donor_full_entries = donor_entries(donor_meta["seed"], donor_meta["n_entries"])
            donor_objects_by_key = {e["key"]: e["object"] for e in donor_full_entries}
            target_keys = {e["key"] for e in entries}
            overlap_keys = sorted(target_keys & donor_objects_by_key.keys())
            donor_key_overlap = {
                "count": len(overlap_keys),
                "donor_objects": {k: donor_objects_by_key[k] for k in overlap_keys},
            }
            logger.info(
                "Donor/target key overlap: %d/%d target keys also present in the donor's "
                "own population (donor's objects for these differ from the target's — see "
                "results.json's donor_key_overlap).",
                donor_key_overlap["count"],
                len(entries),
            )

        # B2: a fresh donor build just ran its own budget_for-derived epoch
        # count worth of GPU training (~2220 steps at DONOR_MIN_ENTRIES's
        # 147-entry population, 30 epochs) immediately before the seed loop
        # — cool down before the first seed instead of chaining straight
        # into it (first_seed=True would otherwise skip the loop's own
        # cooldown for exactly this case).
        first_seed = not donor_built_fresh
        for seed in seeds:
            _check_pause(f"before seed {seed}")

            if _marker_exists(run_dir, seed):
                logger.info("Seed %d: done marker exists — skipping", seed)
                _print_summary_from_results(seed, run_dir / f"seed{seed}" / "results.json")
                continue

            if not first_seed:
                logger.info("Cooldown before seed %d", seed)
                _wait_for_cooldown(52)
            first_seed = False

            logger.info("Starting seed %d -> %s", seed, run_dir / f"seed{seed}")
            model, summary = _run_seed(
                model,
                tokenizer,
                seed,
                entries,
                registry,
                adapter_config,
                base_training_config,
                run_dir,
                arm,
                expected_steps,
                donor_scratch_dir=donor_scratch_dir,
                probe_before_training=args.probe_before_training,
                condition=condition,
                donor_checkpoint_path=donor_checkpoint_path,
                donor_checkpoint_sha256=donor_checkpoint_sha256,
                donor_key_overlap=donor_key_overlap,
            )

            _write_done_marker(run_dir, seed, summary)
            _print_summary_from_results(seed, run_dir / f"seed{seed}" / "results.json")

            sys.stdout.flush()
            sys.stderr.flush()

        unload_model(model, tokenizer)

    # Final cross-seed summary.
    print("\n" + "=" * 72)
    print(f"Test 20 — {arm} Final Summary")
    print("=" * 72)
    rates: list[float] = []
    for seed in seeds:
        results_path = run_dir / f"seed{seed}" / "results.json"
        _print_summary_from_results(seed, results_path)
        if results_path.exists():
            with open(results_path) as f:
                rates.append(json.load(f)["summary"]["rate"])
    if rates:
        mean_rate = sum(rates) / len(rates)
        print(f"\nMean recall across {len(rates)} seeds: {mean_rate:.1%}")
    print(f"\nResults written to: {run_dir}")
    logger.info("Test 20 complete. Results: %s", run_dir)


if __name__ == "__main__":
    main()
