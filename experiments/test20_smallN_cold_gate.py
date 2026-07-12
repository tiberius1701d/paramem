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

Arms (parameterized via ``--n-entries`` / ``--epochs``)
----------------------------------------------------------
N synthetic keys (default 12, a strict prefix of the fixed 12-fact list
for smaller N), Mistral 7B, EPISODIC production recipe (rank 8, alpha 16,
lr 1e-4, attention-only target_modules), COLD LoRA-zero init, ``--epochs``
epochs (default 30), batch_size=1, gradient_accumulation_steps=2 ->
``epochs * _steps_per_epoch(N, 1, 2)`` optimizer steps (derived, never
hardcoded — see ``_expected_optimizer_steps``). Run at 3 seeds (0, 1, 2);
per-seed recall reported, plus the mean. Two decisive arms:

  * ``--n-entries 3 --epochs 30``  -> 60 optimizer steps (the ORIGINAL
    failure condition to reproduce).
  * ``--n-entries 3 --epochs 180`` -> 360 optimizer steps (same N, 6x the
    step budget — isolates whether more steps rescues the small-N arm).

The default (N=12, 30 epochs -> 180 steps) is the arm this script
originally shipped with; it remains available unchanged via defaults.

Recipe fidelity
----------------
Loaded via ``load_server_config("tests/fixtures/server.yaml")`` — never
``load_config()`` or ``configs/server.yaml.example`` (project rule). The
fixture's ``episodic_adapter_config`` supplies rank/alpha/lr/target_modules
verbatim; ``training_config`` supplies batch_size=1,
gradient_accumulation_steps=2, max_seq_length=1024, warmup_steps=0,
lr_scheduler_type="linear", lr_decay_steps=None, weight_decay=0.1,
gradient_checkpointing=True, max_grad_norm=1.0. Two fields are
overridden IN CODE (never in the fixture) because the fixture pins them
for its own production-fold posture and each would silently invalidate
the requested arm if left as-is:

  * ``num_epochs``: fixture ships 30. Set to ``--epochs`` here so the
    arm's step budget comes ONLY from the CLI flag, not the fixture.
  * ``recall_early_stopping``: fixture ships True — would truncate the run
    on 100% recall, making the expected step count fiction. Set to False
    here.

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
   proves the donor copy landed), and NON-ZERO after training in both
   arms (proves the adapter actually moved). Norm computation mirrors
   ``.agent/archive/test19_neardup_procedural.py``
   (``_assert_nonzero_lora_b`` / ``_assert_zero_lora_b``, ~lines 619-676).
4. (Warm arm only) The donor adapter's LoRA-B Frobenius norm is
   bit-identical immediately before and immediately after each seed's
   ``train_adapter`` call (donor immutability), and the trainable
   adapter's name is never a live tier name (``episodic``/``semantic``/
   ``procedural``).

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
small-N recall failure. Mechanism (validated in
``.agent/archive/test19_neardup_procedural.py``'s ``warmstart`` arm):
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
Template: ``.agent/archive/test19_neardup_procedural.py``. Reuses
``experiments/utils/test_harness.py`` (``BENCHMARK_MODELS``,
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
The original synthetic failure condition (N=3, 30 epochs, 60 steps)::

    setsid nohup python \\
        experiments/test20_smallN_cold_gate.py --model mistral \\
        --n-entries 3 --epochs 30 \\
        >outputs/test20_n3_e30.log 2>&1 &

The step-budget-rescue arm (N=3, 180 epochs, 360 steps)::

    setsid nohup python \\
        experiments/test20_smallN_cold_gate.py --model mistral \\
        --n-entries 3 --epochs 180 \\
        >outputs/test20_n3_e180.log 2>&1 &

The REAL 3-triple production failure, cold (60 steps)::

    setsid nohup python \\
        experiments/test20_smallN_cold_gate.py --model mistral \\
        --entries-json experiments/fixtures/real3_interim_failure.json \\
        --epochs 30 \\
        >outputs/test20_real3_cold.log 2>&1 &

The same real triples, warm-started from a donor adapter::

    setsid nohup python \\
        experiments/test20_smallN_cold_gate.py --model mistral \\
        --entries-json experiments/fixtures/real3_interim_failure.json \\
        --epochs 30 --warm-from /path/to/donor_adapter_dir \\
        >outputs/test20_real3_warm.log 2>&1 &

Resume (auto-detects the arm's own output subtree; ``--entries-json`` /
``--warm-from`` must be repeated identically so the resolved ``--arm``
matches)::

    python experiments/test20_smallN_cold_gate.py --model mistral \\
        --n-entries 3 --epochs 30 --resume

The mechanism probe (``--probe-before-training``), real 3 triples, warm vs
cold, 3 seeds each. WARM decrypts the donor adapter, so
``PARAMEM_DAILY_PASSPHRASE`` must be exported first::

    export PARAMEM_DAILY_PASSPHRASE=... && setsid nohup python \\
        experiments/test20_smallN_cold_gate.py --model mistral \\
        --entries-json experiments/fixtures/real3_interim_failure.json \\
        --epochs 30 --warm-from data/ha/adapters/episodic/20260710-224008 \\
        --probe-before-training \\
        >outputs/test20_real3_warm_probe.log 2>&1 &

    setsid nohup python \\
        experiments/test20_smallN_cold_gate.py --model mistral \\
        --entries-json experiments/fixtures/real3_interim_failure.json \\
        --epochs 30 --probe-before-training \\
        >outputs/test20_real3_cold_probe.log 2>&1 &

Data safety
-----------
Results written to unique timestamped, arm-scoped paths via
``model_output_dir`` — never overwritten. Every result file includes full
per-key ``raw_output``. The donor scratch copy (warm arm) lives under
``<run_dir>/donor_scratch/`` — inside the same output tree, never the
donor's original path.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import TrainerCallback  # noqa: E402

from experiments.utils.gpu_guard import acquire_gpu  # noqa: E402
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
    copy_adapter_weights,
    create_adapter,
    switch_adapter,
    unload_model,
)
from paramem.server.config import load_server_config  # noqa: E402
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

# Default arm: the condition this script originally shipped with
# (N=12, 30 epochs -> 180 steps). The decisive arm for the ORIGINAL
# failure is N=3, 30 epochs (--n-entries 3 --epochs 30 -> 60 steps).
DEFAULT_N_ENTRIES = 12
DEFAULT_EPOCHS = 30

# Production recipe values (tests/fixtures/server.yaml episodic
# training_config) used ONLY to derive the expected optimizer-step count
# before the fixture is loaded (needed for the default --arm label and the
# early run_config log, both of which happen before GPU acquire / cfg
# load). The post-cfg-load canary in _run_seed (Step 7) recomputes the
# same derivation from the ACTUAL loaded TrainingConfig and fails loud if
# these ever drift from the fixture.
_RECIPE_BATCH_SIZE = 1
_RECIPE_GRAD_ACCUM_STEPS = 2

# Disk safety threshold (matches test16/test19 convention: free-space, not
# total-usage).
DISK_HEADROOM_BYTES = 5 * 1024**3

# Required fields for each --entries-json entry dict.
_REQUIRED_ENTRY_KEYS = ("key", "subject", "predicate", "object")

# Name reserved for the (frozen, read-only) --warm-from donor adapter.
DONOR_ADAPTER_NAME = "donor"

# Live production tier names — the trainable adapter must NEVER collide
# with one of these (donor-immutability guard; see CHANGE 2 module docstring).
LIVE_TIER_NAMES = frozenset({"episodic", "semantic", "procedural"})


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


def _expected_optimizer_steps(n_entries: int, epochs: int) -> int:
    """Derive the total optimizer-step count for the production recipe.

    Uses ``_steps_per_epoch`` with the fixture's batch_size=1,
    gradient_accumulation_steps=2 (``_RECIPE_BATCH_SIZE`` /
    ``_RECIPE_GRAD_ACCUM_STEPS``): ``epochs * _steps_per_epoch(n_entries, 1,
    2)``. Called before the fixture config is loaded (for the default
    ``--arm`` label and the early ``run_config.json`` log); ``_run_seed``'s
    Step 7 canary independently recomputes the same formula from the actual
    loaded ``TrainingConfig`` and will fail loud if the fixture's
    batch_size / gradient_accumulation_steps ever drift from the hardcoded
    recipe constants here.

    Args:
        n_entries: Number of synthetic keys in the arm.
        epochs: Configured epoch budget (``--epochs``).

    Returns:
        Total optimizer steps: ``_steps_per_epoch * epochs``.
    """
    spe = _steps_per_epoch(n_entries, _RECIPE_BATCH_SIZE, _RECIPE_GRAD_ACCUM_STEPS)
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


def _default_arm_label(n_entries: int, expected_steps: int, is_real: bool, warm: bool) -> str:
    """Derive the default ``--arm`` label from the resolved run config.

    Preserves the script's original synthetic-cold naming
    (``cold_n{N}_s{steps}``) EXACTLY when neither ``--entries-json`` nor
    ``--warm-from`` is set, so ``--resume`` keeps finding runs launched
    before those flags existed (e.g. the completed
    ``outputs/test20_smallN_cold_gate/cold_n3_s60/`` run). Real
    (``--entries-json``) arms use ``real{N}_{cold|warm}_s{steps}`` (e.g.
    ``real3_cold_s60`` / ``real3_warm_s60``) so the label states both the
    dataset and the init condition explicitly.

    Args:
        n_entries: Resolved entry count (file length for
            ``--entries-json``, ``--n-entries``/default otherwise).
        expected_steps: Derived total optimizer-step count
            (``_expected_optimizer_steps``).
        is_real: True when ``--entries-json`` supplied the entry set.
        warm: True when ``--warm-from`` is set.

    Returns:
        The default arm label string (overridden by an explicit ``--arm``).
    """
    if is_real:
        mode = "warm" if warm else "cold"
        return f"real{n_entries}_{mode}_s{expected_steps}"
    if warm:
        return f"n{n_entries}_warm_s{expected_steps}"
    return f"cold_n{n_entries}_s{expected_steps}"


# ---------------------------------------------------------------------------
# LoRA-B norm helper (mirrors test19_neardup_procedural.py:619-676)
# ---------------------------------------------------------------------------


def _lora_b_frobenius_norm(model: PeftModel, adapter_name: str) -> float:
    """Return the total Frobenius norm of all LoRA-B tensors for *adapter_name*.

    LoRA-B is zero-initialised by PEFT at adapter creation (identity
    residual), so a zero norm immediately after ``create_adapter`` and a
    non-zero norm after training together prove cold init actually ran end
    to end (Hard Assertion #3). Mirrors the norm computation in
    ``.agent/archive/test19_neardup_procedural.py``
    (``_assert_nonzero_lora_b`` / ``_assert_zero_lora_b``, ~lines 619-676).

    Args:
        model: PeftModel carrying *adapter_name*.
        adapter_name: Adapter whose LoRA-B norm is computed.

    Returns:
        Total Frobenius norm of all LoRA-B tensors for the adapter (float).

    Raises:
        AssertionError: When no ``lora_B`` tensors are found for the
            adapter (wrong adapter name / not yet created).
    """
    total_norm = 0.0
    count = 0
    for name, param in model.named_parameters():
        if f"lora_B.{adapter_name}.weight" in name:
            total_norm += param.data.norm().item()
            count += 1
    assert count > 0, f"No lora_B tensors found for adapter '{adapter_name}' — check adapter name"
    return total_norm


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
         *base_training_config*, which already carries ``num_epochs``
         (from ``--epochs``) and ``recall_early_stopping=False``). Hard
         Assertion #2 re-checked immediately before the call.
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
            ``num_epochs`` (from ``--epochs``) and
            ``recall_early_stopping=False`` already applied (seed is
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
        donor_lora_b_norm_before = _lora_b_frobenius_norm(model, DONOR_ADAPTER_NAME)

    # Step 4: Hard Assertion #3 — cold init proof (cold arm) / warm-copy
    # proof (warm arm), pre-training.
    lora_b_norm_before = _lora_b_frobenius_norm(model, adapter_name)
    if is_warm:
        assert lora_b_norm_before > 0.0, (
            f"Hard Assertion #3 FAILED (pre-training, warm arm): LoRA-B Frobenius "
            f"norm for '{adapter_name}' is {lora_b_norm_before}, expected > 0.0 — "
            "the donor weight copy did not propagate."
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
            "Pre-training probe [%s seed %d]: %d/%d exact (rate=%.1%%), mean_confidence=%.3f",
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
    lora_b_norm_after = _lora_b_frobenius_norm(model, adapter_name)
    assert lora_b_norm_after > 0.0, (
        f"Hard Assertion #3 FAILED (post-training): LoRA-B Frobenius norm for "
        f"'{adapter_name}' is {lora_b_norm_after}, expected > 0.0 (adapter moved)."
    )

    # Step 10b: (warm arm) Hard Assertion #4 — donor immutability. The donor
    # is never the staging or production slot for this training event, so
    # its LoRA-B norm must be bit-identical before and after.
    donor_lora_b_norm_after: float | None = None
    if is_warm:
        donor_lora_b_norm_after = _lora_b_frobenius_norm(model, DONOR_ADAPTER_NAME)
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
        "seed": seed,
        "n_entries": n_examples,
        "epochs": training_cfg.num_epochs,
        "adapter_name": adapter_name,
        "warm_start": is_warm,
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


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for test20.

    Returns:
        Parsed :class:`argparse.Namespace`.
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
        default=DEFAULT_EPOCHS,
        help=(
            "Epoch budget (default: 30). Combined with the resolved entry count this "
            "derives the expected optimizer-step count via "
            "_expected_optimizer_steps (_steps_per_epoch(N, batch=1, accum=2) * epochs)."
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
    return parser.parse_args()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for Test 20.

    Loads the model once, loads the production episodic recipe from
    ``tests/fixtures/server.yaml`` (with the two in-code overrides
    documented in the module docstring: ``num_epochs`` from ``--epochs``,
    ``recall_early_stopping=False``), cycles
    through the 3 seeds of the resolved arm (``--n-entries``/
    ``--entries-json`` / ``--epochs`` / ``--warm-from`` / ``--arm`` /
    ``--probe-before-training``) with per-seed adapter isolation, and writes
    per-seed results + done markers under an arm-scoped output subtree. GPU
    cooldown is inserted between seeds. The ``~/.training_pause`` gate is
    honoured at every seed boundary.

    Designed to run daemonised (setsid/nohup) — no terminal interaction;
    logs are flushed after every seed.
    """
    # Set CUDA alloc config before any torch import side effects matter.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    args = _parse_args()

    # Resolve the entry set: --entries-json replaces the synthetic
    # generator entirely and implies --n-entries from the file length
    # (CHANGE 1 — a conflicting --n-entries fails loud).
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

    epochs = args.epochs
    expected_steps = _expected_optimizer_steps(n_entries, epochs)
    is_warm = args.warm_from is not None
    arm = args.arm or _default_arm_label(n_entries, expected_steps, is_real, is_warm)

    # Resolve output dir (arm-scoped so distinct arms never collide and
    # --resume never crosses arms).
    arm_base = OUTPUT_BASE / arm
    if args.resume:
        latest = _find_latest_run_dir(arm_base, args.model)
        if latest is None:
            logger.warning("--resume: no prior run found for arm=%s — starting fresh", arm)
            run_dir = model_output_dir(arm_base, args.model)
        else:
            run_dir = latest
            logger.info("Resuming arm=%s from %s", arm, run_dir)
    else:
        run_dir = model_output_dir(arm_base, args.model)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Disk space pre-flight (mirrors test16/test19 convention: free-space,
    # not total-usage).
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(OUTPUT_BASE).free
    if free_bytes <= DISK_HEADROOM_BYTES:
        raise SystemExit(
            f"Insufficient disk space: {free_bytes / 1024**3:.1f} GB free in {OUTPUT_BASE}; "
            f"need > {DISK_HEADROOM_BYTES / 1024**3:.0f} GB."
        )

    # Warm-start donor: shutil.copytree the caller's donor dir into this
    # run's scratch dir ONCE, and load the donor ONLY from that copy in
    # every seed (donor immutability — see module docstring). Reused
    # as-is on --resume (never re-copied) so every seed of a given run
    # sees byte-identical donor weights.
    donor_scratch_dir: Path | None = None
    if is_warm:
        donor_scratch_dir = run_dir / "donor_scratch"
        if donor_scratch_dir.exists():
            logger.info("Reusing existing donor scratch copy (resume): %s", donor_scratch_dir)
        else:
            logger.info("Warm-start: copying donor %s -> %s", args.warm_from, donor_scratch_dir)
            shutil.copytree(args.warm_from, donor_scratch_dir)

    registry = build_registry(entries)
    logger.info(
        "%s arm: N=%d %s keys, epochs=%d, expected_optimizer_steps=%d, warm_start=%s, seeds=%s",
        arm,
        len(entries),
        "real" if is_real else "synthetic",
        epochs,
        expected_steps,
        is_warm,
        SEEDS,
    )

    # Log run config early (before GPU acquire) so a crash before model load
    # still leaves a record of what was attempted.
    run_config = {
        "model": args.model,
        "arm": arm,
        "seeds": list(SEEDS),
        "n_entries": n_entries,
        "entries_json": args.entries_json,
        "epochs": epochs,
        "expected_optimizer_steps": expected_steps,
        "warm_from": args.warm_from,
        "probe_before_training": args.probe_before_training,
        "recipe_source": "tests/fixtures/server.yaml (episodic_adapter_config + training_config)",
        "overrides": {
            "num_epochs": epochs,
            "recall_early_stopping": False,
        },
    }
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.exists():
        with open(run_config_path, "w") as f:
            json.dump(run_config, f, indent=2)
        logger.info("Run config written: %s", run_config_path)

    model_config = BENCHMARK_MODELS[args.model]

    with acquire_gpu(interactive=True):
        model, tokenizer = load_model_and_config(model_config)

        # Load the production episodic recipe from the test fixture — NEVER
        # load_config() / configs/server.yaml.example.
        cfg = load_server_config(str(FIXTURE_CONFIG_PATH))
        adapter_config = cfg.episodic_adapter_config
        base_training_config = dataclasses.replace(
            cfg.training_config,
            num_epochs=epochs,
            recall_early_stopping=False,
        )
        logger.info(
            "Recipe: rank=%d alpha=%d lr=%.0e target_modules=%s | "
            "batch=%d accum=%d epochs=%d warmup=%d scheduler=%s wd=%.2f "
            "recall_early_stopping=%s",
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
            base_training_config.recall_early_stopping,
        )

        first_seed = True
        for seed in SEEDS:
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
    for seed in SEEDS:
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
