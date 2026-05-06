"""Test 15: multi-seed retention probe — scaffold-then-fill vs answer-swap.

Question driving the experiment
-------------------------------
Does an A→B→C1→C2 adapter pipeline preserve more of the unchanged
real-answer keys when 20 slots are filled into a placeholder scaffold
(C2) than when the same 20 slots are answer-swapped on a fresh adapter
(B)?  And does that retention advantage survive seed variance at n=5?

Test 13 reported a 6.7× advantage at n=1 (2026-04-22).  Test 14
(2026-05-06) showed n=1 effects in this regime can collapse under
multi-seed — V3's apparent fill-speed advantage went null at three
seeds.  Test 15 puts the retention claim through the same n=5
treatment under the corrected scheduler so it stops being a
single-seed observation.

Decision rule (frozen at .agent/test15-plan-2026-05-06.md):

    mean_retention_C2 / mean_retention_B  ≥ 5.0  → claim holds
    lower 95% CI on that ratio            ≥ 2.5  → claim holds
    otherwise                                    → claim does not hold

Optimizer / scheduler config
----------------------------
**Apples-to-apples on the SCHEDULER ONLY; weight_decay=0.01 retained.**
Per the plan, this is a deliberate scoped deviation from CLAUDE.md's
"weight_decay ≥ 0.1" rule — that rule originated from a
``constant_with_warmup`` oscillation diagnostic that does not apply to
linear-scheduler runs.  Switching weight_decay simultaneously with
multi-seed would confound two changes; this experiment's job is to
isolate the multi-seed question alone.

    warmup_ratio        = 0.0       (decoupled from num_train_epochs)
    warmup_steps        = 10        (fixed; budget-invariant)
    lr_scheduler_type   = "linear"  (explicit)
    lr_decay_steps      = per-phase (1500/150/700/140 for A/B/C1/C2)
    weight_decay        = 0.01      (deliberate deviation from
                                     CLAUDE.md WD≥0.1 rule, see above)
    save_total_limit    = num_epochs (full retention for resume)

Per-phase lr_decay_steps math (batch=1, grad_accum=2):

    Phase A: 100 keys × 30 epochs / 2 = 1500
    Phase B:  20 keys × 15 epochs / 2 =  150
    Phase C1: 100 keys × 14 epochs / 2 =  700
    Phase C2:  20 keys × 14 epochs / 2 =  140

Independence
------------
This file is self-contained.  The phase builders + ``RecallProbeCallback``
were copied from ``experiments/test13_journal_scaffold.py`` (commit
ed1b556 layout) at write-time so the reference single-seed file stays
untouched.  ``tests/test_test15_multiseed.py`` includes a parity check
against test 13's builders to catch accidental drift.

Resume contract
---------------
Honors ``~/.training_pause`` at every phase boundary inside every
seed.  Per-seed dirs ``seedN/{A,B,C1,C2}/`` with ``*_done.json``
markers; ``--resume`` skips done phases and walks seeds in order.
Mid-phase resume is handled by HF Trainer ``resume_from_checkpoint``
+ the encryption-symmetry shim in ``paramem.training.trainer.train_adapter``.

Usage
-----
    python experiments/test15_multiseed.py
    python experiments/test15_multiseed.py --resume
    python experiments/test15_multiseed.py --seeds 42 --phase-a-epochs 5 \\
            --phase-b-epochs 3 --phase-c1-epochs 3 --phase-c2-epochs 3 \\
            --total-keys 20 --swap-keys 5     # smoke

Pause / resume via tresume:
    tresume 15
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from peft import PeftModel  # noqa: E402
from transformers import TrainerCallback  # noqa: E402

from experiments.utils.gpu_guard import acquire_gpu  # noqa: E402
from experiments.utils.perltqa_loader import (  # noqa: E402
    is_available as perltqa_available,
)
from experiments.utils.perltqa_loader import (  # noqa: E402
    list_characters,
    load_character_eval_qa,
)
from experiments.utils.test_harness import (  # noqa: E402
    BENCHMARK_MODELS,
    IndexedDataset,
    evaluate_indexed_recall,
    load_model_and_config,
    model_output_dir,
    save_results,
    setup_logging,
)
from paramem.adapters import resolve_adapter_slot  # noqa: E402
from paramem.adapters.manifest import build_manifest_for  # noqa: E402
from paramem.models.loader import (  # noqa: E402
    create_adapter,
    load_adapter,
    save_adapter,
    switch_adapter,
    unload_model,
)
from paramem.training.indexed_memory import (  # noqa: E402
    assign_keys,
    build_registry,
    format_indexed_training,
    save_registry,
)
from paramem.training.trainer import train_adapter  # noqa: E402
from paramem.utils.config import AdapterConfig, TrainingConfig  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

OUTPUT_DIR = project_root / "outputs" / "test15_multiseed"
TOTAL_KEYS = 100
SWAP_KEYS = 20
SWAP_START_SLOT = TOTAL_KEYS - SWAP_KEYS
STABLE_EPOCH_WINDOW = 2
CHARACTER_A = "Liang Xin"
CHARACTER_B = "Cai Xiuying"
PAUSE_FILE = Path.home() / ".training_pause"
DEFAULT_SEEDS = [42, 7, 1337, 1, 11]


# ---------------------------------------------------------------------------
# Callback: per-epoch recall probe (copied verbatim from Test 13 with the
# same cycle_started_at preservation that landed 2026-05-06).
# ---------------------------------------------------------------------------


@dataclass
class EpochProbeState:
    first_perfect_epoch: int | None = None
    stable_perfect_epoch: int | None = None
    epoch_log: list[dict] = field(default_factory=list)


class RecallProbeCallback(TrainerCallback):
    def __init__(
        self,
        model,
        tokenizer,
        target_keyed: list[dict],
        target_registry: dict[str, int],
        adapter_name: str,
        state_out: EpochProbeState,
        stable_window: int = STABLE_EPOCH_WINDOW,
        progress_path: Path | None = None,
        phase_name: str | None = None,
        num_epochs: int | None = None,
        keys_in_phase: int | None = None,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._target_keyed = target_keyed
        self._target_registry = target_registry
        self._adapter_name = adapter_name
        self._state = state_out
        self._stable_window = stable_window
        self._last_epoch = -1
        self._progress_path = progress_path
        self._phase_name = phase_name
        self._num_epochs = num_epochs
        self._keys_in_phase = keys_in_phase
        self._cycle_started_at = int(time.time())

        # Preserve cycle_started_at across tpause/tresume so wall-time
        # math in tstatus stays anchored to the original cycle start.
        if self._progress_path is not None and self._progress_path.exists():
            try:
                existing = json.loads(self._progress_path.read_text())
                saved = existing.get("cycle_started_at")
                if isinstance(saved, (int, float)):
                    self._cycle_started_at = int(saved)
            except (OSError, json.JSONDecodeError):
                pass

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(round(state.epoch))
        if epoch <= self._last_epoch:
            return
        self._last_epoch = epoch

        recall = evaluate_indexed_recall(
            self._model,
            self._tokenizer,
            self._target_keyed,
            self._target_registry,
            adapter_name=self._adapter_name,
        )
        self._model.gradient_checkpointing_enable()

        entry = {
            "epoch": epoch,
            "recall": recall["exact_count"],
            "total": recall["total"],
            "rate": recall["rate"],
            "mean_confidence": recall["mean_confidence"],
        }
        self._state.epoch_log.append(entry)

        logger.info(
            "  epoch %d: recall %d/%d (conf=%.3f)",
            epoch,
            recall["exact_count"],
            recall["total"],
            recall["mean_confidence"],
        )

        is_perfect = recall["exact_count"] == recall["total"] and recall["total"] > 0
        if is_perfect and self._state.first_perfect_epoch is None:
            self._state.first_perfect_epoch = epoch

        if self._state.stable_perfect_epoch is None and is_perfect:
            window = self._state.epoch_log[-self._stable_window :]
            if len(window) >= self._stable_window and all(
                e["recall"] == e["total"] and e["total"] > 0 for e in window
            ):
                self._state.stable_perfect_epoch = epoch

        if self._progress_path is not None:
            progress = {
                "phase": self._phase_name,
                "keys": self._keys_in_phase,
                "epoch": epoch,
                "total_epochs": self._num_epochs,
                "epoch_offset": 0,
                "cycle_started_at": self._cycle_started_at,
                "recall": recall["exact_count"],
                "total": recall["total"],
                "rate": recall["rate"],
            }
            try:
                self._progress_path.parent.mkdir(parents=True, exist_ok=True)
                self._progress_path.write_text(json.dumps(progress, indent=2))
            except OSError as exc:
                logger.warning("progress.json write failed: %s", exc)


# ---------------------------------------------------------------------------
# QA pool + phase builders (copied from Test 13).
# ---------------------------------------------------------------------------


def load_qa_pool(total_keys: int) -> list[dict]:
    """Same dedup-on-(question, answer) ladder as Test 13."""
    needed = total_keys + SWAP_KEYS
    if not perltqa_available():
        raise RuntimeError("PerLTQA dataset not available — required for Test 15")

    primary = [CHARACTER_A, CHARACTER_B]
    try:
        remaining = sorted(c for c in list_characters() if c not in primary)
    except Exception:
        remaining = []
    source_order = primary + remaining

    seen_q: set[str] = set()
    seen_a: set[str] = set()
    dropped_collisions = 0
    pool: list[dict] = []

    for char in source_order:
        if len(pool) >= needed:
            break
        batch = load_character_eval_qa(char, max_pairs=needed * 2)
        for pair in batch:
            q = pair.get("question", "").strip()
            a = pair.get("answer", "").strip()
            if not q or not a:
                continue
            if q in seen_q or a in seen_a:
                dropped_collisions += 1
                continue
            seen_q.add(q)
            seen_a.add(a)
            pool.append(pair)
            if len(pool) >= needed:
                break

    if len(pool) < needed:
        raise RuntimeError(
            f"Need {needed} unique QA pairs, got {len(pool)} after dedup "
            f"(dropped {dropped_collisions} collisions). Widen sources."
        )

    if dropped_collisions:
        logger.info(
            "load_qa_pool: dropped %d colliding pairs; kept %d unique from %d sources",
            dropped_collisions,
            len(pool),
            len(source_order),
        )

    return pool[:needed]


def build_phase_A_keyed(qa_pool: list[dict]) -> list[dict]:
    return assign_keys(qa_pool[:TOTAL_KEYS], start_index=1)


def build_phase_B_swap_keyed(base_keyed: list[dict], swap_answers: list[dict]) -> list[dict]:
    assert len(swap_answers) >= SWAP_KEYS
    swap_keyed = []
    for i, kp in enumerate(base_keyed[SWAP_START_SLOT:]):
        replacement = swap_answers[i]["answer"]
        if replacement.strip() == kp["answer"].strip():
            replacement = replacement + " (variant)"
        swap_keyed.append(
            {
                "key": kp["key"],
                "question": kp["question"],
                "answer": replacement,
            }
        )
    return swap_keyed


def build_phase_C1_keyed(qa_pool: list[dict]) -> list[dict]:
    mixed = []
    for i, qa in enumerate(qa_pool[:TOTAL_KEYS]):
        if i < SWAP_START_SLOT:
            mixed.append(qa)
        else:
            slot_k = i - SWAP_START_SLOT + 1
            mixed.append(
                {
                    "question": qa["question"],
                    "answer": f"TBD-{slot_k}",
                }
            )
    return assign_keys(mixed, start_index=1)


def build_phase_C2_fill_keyed(c1_keyed: list[dict], qa_pool: list[dict]) -> list[dict]:
    fill_keyed = []
    for i, kp in enumerate(c1_keyed[SWAP_START_SLOT:]):
        fill_keyed.append(
            {
                "key": kp["key"],
                "question": kp["question"],
                "answer": qa_pool[SWAP_START_SLOT + i]["answer"],
            }
        )
    return fill_keyed


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _training_config(*, num_epochs: int, seed: int, lr_decay_steps: int) -> TrainingConfig:
    """Apples-to-apples scheduler + weight_decay=0.01.

    Per .agent/test15-plan-2026-05-06.md "Apples-to-apples on the SCHEDULER
    ONLY".  weight_decay=0.01 is a deliberate scoped deviation from
    CLAUDE.md's WD≥0.1 rule (see module docstring).
    """
    return TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=2,
        max_seq_length=1024,
        num_epochs=num_epochs,
        warmup_ratio=0.0,
        warmup_steps=10,
        lr_scheduler_type="linear",
        lr_decay_steps=lr_decay_steps,
        weight_decay=0.01,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=seed,
        save_strategy="epoch",
        save_total_limit=num_epochs,
    )


def _decay_steps_for(n_keys: int, num_epochs: int) -> int:
    """n_keys × num_epochs / grad_accum (=2)."""
    return (n_keys * num_epochs) // 2


def run_training_phase(
    model,
    tokenizer,
    keyed_to_train: list[dict],
    target_keyed: list[dict],
    target_registry: dict[str, int],
    adapter_name: str,
    adapter_config: AdapterConfig,
    training_config: TrainingConfig,
    output_dir: Path,
    run_name: str,
    resume_from_checkpoint: Path | None = None,
) -> tuple[dict, EpochProbeState, float]:
    examples = format_indexed_training(keyed_to_train, tokenizer, max_length=1024)
    dataset = IndexedDataset(examples)

    probe_state = EpochProbeState()
    phase_name = output_dir.name
    callback = RecallProbeCallback(
        model=model,
        tokenizer=tokenizer,
        target_keyed=target_keyed,
        target_registry=target_registry,
        adapter_name=adapter_name,
        state_out=probe_state,
        progress_path=output_dir / "progress.json",
        phase_name=phase_name,
        num_epochs=training_config.num_epochs,
        keys_in_phase=len(keyed_to_train),
    )

    t0 = time.time()
    metrics = train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        adapter_name=adapter_name,
        training_config=training_config,
        adapter_config=adapter_config,
        output_dir=output_dir / "adapter",
        run_name=run_name,
        callbacks_extra=[callback],
        resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None,
    )
    wall = time.time() - t0
    return metrics, probe_state, wall


def save_phase_artifacts(
    phase_dir: Path,
    keyed_pairs: list[dict],
    registry: dict[str, int],
    probe_state: EpochProbeState,
    extra: dict,
    marker_name: str,
) -> None:
    phase_dir.mkdir(parents=True, exist_ok=True)
    with open(phase_dir / "keyed_pairs.json", "w") as f:
        kp_ser = [
            {"key": kp["key"], "question": kp["question"], "answer": kp["answer"]}
            for kp in keyed_pairs
        ]
        json.dump(kp_ser, f, indent=2)
    save_registry(registry, phase_dir / "simhash_registry.json")
    marker = {
        "first_perfect_epoch": probe_state.first_perfect_epoch,
        "stable_perfect_epoch": probe_state.stable_perfect_epoch,
        "epoch_log": probe_state.epoch_log,
        **extra,
    }
    with open(phase_dir / marker_name, "w") as f:
        json.dump(marker, f, indent=2)
    logger.info("Phase marker written: %s", phase_dir / marker_name)


def leakage_count(recall_per_key: list[dict]) -> int:
    count = 0
    for entry in recall_per_key:
        recalled = entry.get("recalled") or {}
        if isinstance(recalled, dict) and "TBD-" in str(recalled.get("answer", "")):
            count += 1
    return count


def _adapter_cfg(rank: int) -> AdapterConfig:
    return AdapterConfig(
        rank=rank,
        alpha=rank * 2,
        learning_rate=1e-4,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        dropout=0.0,
    )


def _find_latest_checkpoint(adapter_dir: Path) -> Path | None:
    """Sort numerically (CLAUDE.md: never lex-sort `checkpoint-NNN`)."""
    if not adapter_dir.is_dir():
        return None
    candidates = []
    for p in adapter_dir.iterdir():
        if not p.is_dir() or not p.name.startswith("checkpoint-"):
            continue
        try:
            step = int(p.name.split("-", 1)[1])
        except ValueError:
            continue
        candidates.append((step, p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


# ---------------------------------------------------------------------------
# Phases (copied from Test 13 with seed + lr_decay_steps params)
# ---------------------------------------------------------------------------


def phase_A(
    model,
    tokenizer,
    qa_pool: list[dict],
    args,
    output_dir: Path,
    seed: int,
) -> tuple[object, list[dict], dict, EpochProbeState]:
    logger.info("=" * 72)
    logger.info(
        "Phase A — Fresh: %d keys, %d epochs, seed=%d",
        TOTAL_KEYS,
        args.phase_a_epochs,
        seed,
    )
    logger.info("=" * 72)

    keyed = build_phase_A_keyed(qa_pool)
    registry = build_registry(keyed)

    adapter_config = _adapter_cfg(args.rank)
    model = create_adapter(model, adapter_config, "episodic")

    phase_dir = output_dir / "A"
    training_config = _training_config(
        num_epochs=args.phase_a_epochs,
        seed=seed,
        lr_decay_steps=_decay_steps_for(TOTAL_KEYS, args.phase_a_epochs),
    )
    ckpt = _find_latest_checkpoint(phase_dir / "adapter")
    metrics, probe_state, wall = run_training_phase(
        model,
        tokenizer,
        keyed_to_train=keyed,
        target_keyed=keyed,
        target_registry=registry,
        adapter_name="episodic",
        adapter_config=adapter_config,
        training_config=training_config,
        output_dir=phase_dir,
        run_name=f"test15-A-seed{seed}",
        resume_from_checkpoint=ckpt,
    )

    final_recall = evaluate_indexed_recall(
        model, tokenizer, keyed, registry, adapter_name="episodic"
    )
    _manifest_A = build_manifest_for(
        model,
        tokenizer,
        "episodic",
        registry_path=None,
        keyed_pairs_path=phase_dir / "keyed_pairs.json",
        key_count=len(keyed),
    )
    save_adapter(model, phase_dir / "adapter", "episodic", manifest=_manifest_A)
    save_phase_artifacts(
        phase_dir,
        keyed,
        registry,
        probe_state,
        extra={
            "condition": "A_fresh",
            "n_keys": TOTAL_KEYS,
            "seed": seed,
            "train_loss": metrics.get("train_loss"),
            "wall_seconds": round(wall, 1),
            "final_recall": {
                "exact_count": final_recall["exact_count"],
                "total": final_recall["total"],
                "rate": final_recall["rate"],
                "mean_confidence": final_recall["mean_confidence"],
            },
        },
        marker_name="A_done.json",
    )
    return model, keyed, registry, probe_state


def phase_B(
    model,
    tokenizer,
    base_keyed: list[dict],
    swap_answers: list[dict],
    args,
    output_dir: Path,
    seed: int,
) -> EpochProbeState:
    logger.info("=" * 72)
    logger.info(
        "Phase B — Answer-swap: %d keys, %d epochs, seed=%d",
        SWAP_KEYS,
        args.phase_b_epochs,
        seed,
    )
    logger.info("=" * 72)

    swap_keyed = build_phase_B_swap_keyed(base_keyed, swap_answers)
    swap_registry = build_registry(swap_keyed)

    unchanged_keyed = base_keyed[:SWAP_START_SLOT]
    unchanged_registry = build_registry(unchanged_keyed)

    adapter_config = _adapter_cfg(args.rank)
    training_config = _training_config(
        num_epochs=args.phase_b_epochs,
        seed=seed,
        lr_decay_steps=_decay_steps_for(SWAP_KEYS, args.phase_b_epochs),
    )

    switch_adapter(model, "episodic")
    phase_dir = output_dir / "B"
    ckpt = _find_latest_checkpoint(phase_dir / "adapter")
    metrics, probe_state, wall = run_training_phase(
        model,
        tokenizer,
        keyed_to_train=swap_keyed,
        target_keyed=swap_keyed,
        target_registry=swap_registry,
        adapter_name="episodic",
        adapter_config=adapter_config,
        training_config=training_config,
        output_dir=phase_dir,
        run_name=f"test15-B-seed{seed}",
        resume_from_checkpoint=ckpt,
    )

    swap_final = evaluate_indexed_recall(
        model, tokenizer, swap_keyed, swap_registry, adapter_name="episodic"
    )
    retention = evaluate_indexed_recall(
        model,
        tokenizer,
        unchanged_keyed,
        unchanged_registry,
        adapter_name="episodic",
    )
    _manifest_B = build_manifest_for(
        model,
        tokenizer,
        "episodic",
        registry_path=None,
        keyed_pairs_path=phase_dir / "keyed_pairs.json",
        key_count=len(swap_keyed),
    )
    save_adapter(model, phase_dir / "adapter", "episodic", manifest=_manifest_B)
    save_phase_artifacts(
        phase_dir,
        swap_keyed,
        swap_registry,
        probe_state,
        extra={
            "condition": "B_answer_swap",
            "n_keys": SWAP_KEYS,
            "seed": seed,
            "train_loss": metrics.get("train_loss"),
            "wall_seconds": round(wall, 1),
            "swap_final_recall": {
                "exact_count": swap_final["exact_count"],
                "total": swap_final["total"],
                "rate": swap_final["rate"],
                "mean_confidence": swap_final["mean_confidence"],
            },
            "retention_unchanged_80": {
                "exact_count": retention["exact_count"],
                "total": retention["total"],
                "rate": retention["rate"],
                "mean_confidence": retention["mean_confidence"],
            },
        },
        marker_name="B_done.json",
    )
    return probe_state


def phase_C1(
    model,
    tokenizer,
    qa_pool: list[dict],
    args,
    output_dir: Path,
    seed: int,
) -> tuple[object, list[dict], dict, EpochProbeState]:
    logger.info("=" * 72)
    logger.info(
        "Phase C1 — Scaffold: %d keys (%d real + %d TBD-k), %d epochs, seed=%d",
        TOTAL_KEYS,
        SWAP_START_SLOT,
        SWAP_KEYS,
        args.phase_c1_epochs,
        seed,
    )
    logger.info("=" * 72)

    keyed = build_phase_C1_keyed(qa_pool)
    registry = build_registry(keyed)

    adapter_config = _adapter_cfg(args.rank)
    model = create_adapter(model, adapter_config, "journal")

    training_config = _training_config(
        num_epochs=args.phase_c1_epochs,
        seed=seed,
        lr_decay_steps=_decay_steps_for(TOTAL_KEYS, args.phase_c1_epochs),
    )

    phase_dir = output_dir / "C1"
    ckpt = _find_latest_checkpoint(phase_dir / "adapter")
    metrics, probe_state, wall = run_training_phase(
        model,
        tokenizer,
        keyed_to_train=keyed,
        target_keyed=keyed,
        target_registry=registry,
        adapter_name="journal",
        adapter_config=adapter_config,
        training_config=training_config,
        output_dir=phase_dir,
        run_name=f"test15-C1-seed{seed}",
        resume_from_checkpoint=ckpt,
    )

    final_recall = evaluate_indexed_recall(
        model, tokenizer, keyed, registry, adapter_name="journal"
    )
    _manifest_C1 = build_manifest_for(
        model,
        tokenizer,
        "journal",
        registry_path=None,
        keyed_pairs_path=phase_dir / "keyed_pairs.json",
        key_count=len(keyed),
    )
    save_adapter(model, phase_dir / "adapter", "journal", manifest=_manifest_C1)
    save_phase_artifacts(
        phase_dir,
        keyed,
        registry,
        probe_state,
        extra={
            "condition": "C1_scaffold",
            "n_keys": TOTAL_KEYS,
            "n_placeholders": SWAP_KEYS,
            "seed": seed,
            "train_loss": metrics.get("train_loss"),
            "wall_seconds": round(wall, 1),
            "final_recall": {
                "exact_count": final_recall["exact_count"],
                "total": final_recall["total"],
                "rate": final_recall["rate"],
                "mean_confidence": final_recall["mean_confidence"],
            },
        },
        marker_name="C1_done.json",
    )
    return model, keyed, registry, probe_state


def phase_C2(
    model,
    tokenizer,
    c1_keyed: list[dict],
    qa_pool: list[dict],
    args,
    output_dir: Path,
    seed: int,
) -> EpochProbeState:
    logger.info("=" * 72)
    logger.info(
        "Phase C2 — Fill: %d keys (TBD-k → real), %d epochs, seed=%d",
        SWAP_KEYS,
        args.phase_c2_epochs,
        seed,
    )
    logger.info("=" * 72)

    fill_keyed = build_phase_C2_fill_keyed(c1_keyed, qa_pool)
    fill_registry = build_registry(fill_keyed)

    unchanged_keyed = c1_keyed[:SWAP_START_SLOT]
    unchanged_registry = build_registry(unchanged_keyed)

    adapter_config = _adapter_cfg(args.rank)
    training_config = _training_config(
        num_epochs=args.phase_c2_epochs,
        seed=seed,
        lr_decay_steps=_decay_steps_for(SWAP_KEYS, args.phase_c2_epochs),
    )

    switch_adapter(model, "journal")
    phase_dir = output_dir / "C2"
    ckpt = _find_latest_checkpoint(phase_dir / "adapter")
    metrics, probe_state, wall = run_training_phase(
        model,
        tokenizer,
        keyed_to_train=fill_keyed,
        target_keyed=fill_keyed,
        target_registry=fill_registry,
        adapter_name="journal",
        adapter_config=adapter_config,
        training_config=training_config,
        output_dir=phase_dir,
        run_name=f"test15-C2-seed{seed}",
        resume_from_checkpoint=ckpt,
    )

    fill_final = evaluate_indexed_recall(
        model, tokenizer, fill_keyed, fill_registry, adapter_name="journal"
    )
    retention = evaluate_indexed_recall(
        model,
        tokenizer,
        unchanged_keyed,
        unchanged_registry,
        adapter_name="journal",
    )
    leaks_on_fill = leakage_count(fill_final["per_key"])
    _manifest_C2 = build_manifest_for(
        model,
        tokenizer,
        "journal",
        registry_path=None,
        keyed_pairs_path=phase_dir / "keyed_pairs.json",
        key_count=len(fill_keyed),
    )
    save_adapter(model, phase_dir / "adapter", "journal", manifest=_manifest_C2)
    save_phase_artifacts(
        phase_dir,
        fill_keyed,
        fill_registry,
        probe_state,
        extra={
            "condition": "C2_fill",
            "n_keys": SWAP_KEYS,
            "seed": seed,
            "train_loss": metrics.get("train_loss"),
            "wall_seconds": round(wall, 1),
            "fill_final_recall": {
                "exact_count": fill_final["exact_count"],
                "total": fill_final["total"],
                "rate": fill_final["rate"],
                "mean_confidence": fill_final["mean_confidence"],
            },
            "retention_unchanged_80": {
                "exact_count": retention["exact_count"],
                "total": retention["total"],
                "rate": retention["rate"],
                "mean_confidence": retention["mean_confidence"],
            },
            "placeholder_leakage_on_fills": leaks_on_fill,
        },
        marker_name="C2_done.json",
    )
    return probe_state


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def marker_exists(seed_dir: Path, phase: str) -> bool:
    return (seed_dir / phase / f"{phase}_done.json").exists()


def paused_requested() -> bool:
    return PAUSE_FILE.exists()


def write_paused_marker(output_dir: Path, after_phase: str) -> None:
    marker = {
        "timestamp": int(time.time()),
        "exit_reason": "training_pause",
        "stopped_after_phase": after_phase,
        "pause_file": str(PAUSE_FILE),
    }
    (output_dir / "paused.json").write_text(json.dumps(marker, indent=2))
    logger.warning(
        "Pause signal detected after %s — exiting cleanly. Use tresume to continue.",
        after_phase,
    )


def find_latest_run_dir(model_name: str) -> Path | None:
    parent = OUTPUT_DIR / model_name
    if not parent.is_dir():
        return None
    runs = sorted(d for d in parent.iterdir() if d.is_dir())
    return runs[-1] if runs else None


def load_or_write_run_config(output_dir: Path, args) -> dict:
    cfg_path = output_dir / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        logger.info("Loaded run_config.json: %s", cfg)
        return cfg
    cfg = {
        "model": args.model,
        "seeds": args.seeds,
        "total_keys": args.total_keys,
        "swap_keys": args.swap_keys,
        "phase_a_epochs": args.phase_a_epochs,
        "phase_b_epochs": args.phase_b_epochs,
        "phase_c1_epochs": args.phase_c1_epochs,
        "phase_c2_epochs": args.phase_c2_epochs,
        "rank": args.rank,
        "weight_decay": 0.01,  # see module docstring
        "lr_scheduler_type": "linear",
        "warmup_steps": 10,
        "warmup_ratio": 0.0,
        "created_at": int(time.time()),
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))
    logger.info("Wrote run_config.json: %s", cfg)
    return cfg


def read_keyed(seed_dir: Path, phase: str) -> tuple[list[dict], dict[str, int]]:
    with open(seed_dir / phase / "keyed_pairs.json") as f:
        keyed = json.load(f)
    with open(seed_dir / phase / "simhash_registry.json") as f:
        registry = json.load(f)
    registry = {k: int(v) for k, v in registry.items()}
    return keyed, registry


def _reload_adapter(model, adapter_dir: Path, adapter_name: str):
    """Reload a saved adapter slot into ``model`` under ``adapter_name``.

    Mirrors test13_journal_scaffold.main()'s reload pattern.
    """
    slot = resolve_adapter_slot(adapter_dir, adapter_name, "")
    if slot is not None:
        if isinstance(model, PeftModel):
            model.load_adapter(str(slot), adapter_name=adapter_name)
            return model
        return PeftModel.from_pretrained(model, str(slot), adapter_name=adapter_name)
    return load_adapter(model, adapter_dir, adapter_name)


def _gpu_cooldown(threshold_c: int = 52) -> None:
    """Brief cooldown between high-load phase boundaries."""
    try:
        import subprocess

        subprocess.run(
            [
                "bash",
                "-c",
                f"source ~/.local/bin/gpu-cooldown.sh && wait_for_cooldown {threshold_c}",
            ],
            check=False,
            timeout=300,
        )
    except Exception as exc:
        logger.warning("Cooldown skipped: %s", exc)


def _write_multiseed_aggregate(run_dir: Path, seeds: list[int]) -> None:
    rows = []
    for s in seeds:
        seed_dir = run_dir / f"seed{s}"
        b_path = seed_dir / "B" / "B_done.json"
        c2_path = seed_dir / "C2" / "C2_done.json"
        if not b_path.exists() or not c2_path.exists():
            logger.warning("seed=%d incomplete; skipping aggregate row", s)
            continue
        b = json.loads(b_path.read_text())
        c2 = json.loads(c2_path.read_text())
        rows.append(
            {
                "seed": s,
                "retention_B": b["retention_unchanged_80"]["rate"],
                "retention_C2": c2["retention_unchanged_80"]["rate"],
                "first_perfect_B": b.get("first_perfect_epoch"),
                "stable_perfect_B": b.get("stable_perfect_epoch"),
                "first_perfect_C2": c2.get("first_perfect_epoch"),
                "stable_perfect_C2": c2.get("stable_perfect_epoch"),
                "leaks_on_fill": c2.get("placeholder_leakage_on_fills", 0),
            }
        )

    if not rows:
        logger.warning("No completed seeds — aggregate skipped")
        return

    rs_b = [r["retention_B"] for r in rows]
    rs_c2 = [r["retention_C2"] for r in rows]
    mean_b = mean(rs_b)
    mean_c2 = mean(rs_c2)
    aggregate = {
        "seeds": seeds,
        "n_completed": len(rows),
        "per_seed": rows,
        "mean_retention_B": mean_b,
        "mean_retention_C2": mean_c2,
        "sd_retention_B": stdev(rs_b) if len(rs_b) > 1 else 0.0,
        "sd_retention_C2": stdev(rs_c2) if len(rs_c2) > 1 else 0.0,
        "ratio_C2_over_B": (mean_c2 / mean_b) if mean_b > 0 else None,
        "leaks_total": sum(r["leaks_on_fill"] for r in rows),
        "decision_threshold_ratio": 5.0,
    }
    (run_dir / "multiseed_aggregate.json").write_text(json.dumps(aggregate, indent=2))
    logger.info(
        "Multi-seed aggregate (n=%d): C2/B = %s (B mean=%.3f, C2 mean=%.3f)",
        len(rows),
        f"{aggregate['ratio_C2_over_B']:.2f}" if aggregate["ratio_C2_over_B"] else "n/a",
        mean_b,
        mean_c2,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test 15: multi-seed retention probe (scaffold-fill vs answer-swap)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="HF Trainer seeds to run.  Default: %(default)s.",
    )
    parser.add_argument("--phase-a-epochs", type=int, default=30)
    parser.add_argument("--phase-b-epochs", type=int, default=15)
    parser.add_argument("--phase-c1-epochs", type=int, default=14)
    parser.add_argument("--phase-c2-epochs", type=int, default=14)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--total-keys", type=int, default=100)
    parser.add_argument("--swap-keys", type=int, default=20)
    parser.add_argument("--model", choices=["mistral"], default="mistral")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Auto-find the latest run dir for --model and continue.",
    )
    args = parser.parse_args()

    # Resolve run_dir
    if args.output_dir is not None:
        run_dir = Path(args.output_dir)
    elif args.resume:
        latest = find_latest_run_dir(args.model)
        if latest is None:
            logger.warning(
                "--resume requested but no prior run for %s — starting fresh",
                args.model,
            )
            run_dir = model_output_dir(OUTPUT_DIR, args.model)
        else:
            run_dir = latest
            logger.info("Resuming from %s", run_dir)
    else:
        run_dir = model_output_dir(OUTPUT_DIR, args.model)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", run_dir)

    cfg = load_or_write_run_config(run_dir, args)
    args.seeds = cfg["seeds"]
    args.total_keys = cfg["total_keys"]
    args.swap_keys = cfg["swap_keys"]
    args.phase_a_epochs = cfg["phase_a_epochs"]
    args.phase_b_epochs = cfg["phase_b_epochs"]
    args.phase_c1_epochs = cfg["phase_c1_epochs"]
    args.phase_c2_epochs = cfg["phase_c2_epochs"]
    args.rank = cfg["rank"]

    global TOTAL_KEYS, SWAP_KEYS, SWAP_START_SLOT
    TOTAL_KEYS = args.total_keys
    SWAP_KEYS = args.swap_keys
    SWAP_START_SLOT = TOTAL_KEYS - SWAP_KEYS

    if paused_requested():
        logger.warning("Pause file present at launch — clearing before starting work")
        try:
            PAUSE_FILE.unlink()
        except OSError:
            pass
    stale = run_dir / "paused.json"
    if stale.exists():
        stale.unlink()

    model_config = BENCHMARK_MODELS[args.model]
    qa_pool = load_qa_pool(TOTAL_KEYS)
    logger.info("Loaded %d QA pairs from PerLTQA", len(qa_pool))

    with acquire_gpu(interactive=True):
        model, tokenizer = load_model_and_config(model_config)

        for seed in args.seeds:
            seed_dir = run_dir / f"seed{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            base_keyed = None
            c1_keyed = None

            # Unwrap any prior-seed PeftModel back to the base before this seed's
            # Phase A creates a fresh "episodic" adapter.  Mirrors test13's
            # A→B → C1→C2 transition pattern, applied at the seed boundary.
            if isinstance(model, PeftModel):
                model = model.base_model.model

            # --- Phase A ---
            if marker_exists(seed_dir, "A"):
                logger.info("seed=%d A: already done — loading adapter", seed)
                model = _reload_adapter(model, seed_dir / "A" / "adapter", "episodic")
                base_keyed, _ = read_keyed(seed_dir, "A")
            else:
                if paused_requested():
                    write_paused_marker(run_dir, f"before seed {seed} Phase A")
                    unload_model(model, tokenizer)
                    return
                model, base_keyed, _, _ = phase_A(model, tokenizer, qa_pool, args, seed_dir, seed)
                if paused_requested():
                    write_paused_marker(run_dir, f"seed {seed} during/after Phase A")
                    unload_model(model, tokenizer)
                    return

            # --- Phase B ---
            if marker_exists(seed_dir, "B"):
                logger.info("seed=%d B: already done", seed)
            else:
                if paused_requested():
                    write_paused_marker(run_dir, f"seed {seed} before Phase B")
                    unload_model(model, tokenizer)
                    return
                if base_keyed is None:
                    base_keyed, _ = read_keyed(seed_dir, "A")
                swap_answers = qa_pool[TOTAL_KEYS : TOTAL_KEYS + SWAP_KEYS]
                phase_B(model, tokenizer, base_keyed, swap_answers, args, seed_dir, seed)
                if paused_requested():
                    write_paused_marker(run_dir, f"seed {seed} during/after Phase B")
                    unload_model(model, tokenizer)
                    return

            # --- A/B → C1 transition: cooldown + unwrap "episodic" so phase_C1
            #     can create a fresh "journal" adapter.
            _gpu_cooldown(52)
            if isinstance(model, PeftModel):
                model = model.base_model.model

            # --- Phase C1 ---
            if marker_exists(seed_dir, "C1"):
                logger.info("seed=%d C1: already done — loading adapter", seed)
                model = _reload_adapter(model, seed_dir / "C1" / "adapter", "journal")
                c1_keyed, _ = read_keyed(seed_dir, "C1")
            else:
                if paused_requested():
                    write_paused_marker(run_dir, f"seed {seed} before Phase C1")
                    unload_model(model, tokenizer)
                    return
                model, c1_keyed, _, _ = phase_C1(model, tokenizer, qa_pool, args, seed_dir, seed)
                if paused_requested():
                    write_paused_marker(run_dir, f"seed {seed} during/after Phase C1")
                    unload_model(model, tokenizer)
                    return

            # --- Phase C2 ---
            if marker_exists(seed_dir, "C2"):
                logger.info("seed=%d C2: already done", seed)
            else:
                if paused_requested():
                    write_paused_marker(run_dir, f"seed {seed} before Phase C2")
                    unload_model(model, tokenizer)
                    return
                if c1_keyed is None:
                    c1_keyed, _ = read_keyed(seed_dir, "C1")
                phase_C2(model, tokenizer, c1_keyed, qa_pool, args, seed_dir, seed)
                if paused_requested():
                    write_paused_marker(run_dir, f"seed {seed} during/after Phase C2")
                    unload_model(model, tokenizer)
                    return

        # All seeds finished — write aggregate.
        _write_multiseed_aggregate(run_dir, args.seeds)
        unload_model(model, tokenizer)

    # Summary
    summary = {"output_dir": str(run_dir), "seeds": {}}
    for seed in args.seeds:
        seed_dir = run_dir / f"seed{seed}"
        summary["seeds"][str(seed)] = {}
        for phase in ("A", "B", "C1", "C2"):
            marker = seed_dir / phase / f"{phase}_done.json"
            if marker.exists():
                with open(marker) as f:
                    summary["seeds"][str(seed)][phase] = json.load(f)
    save_results(summary, run_dir, "results.json")

    aggregate_path = run_dir / "multiseed_aggregate.json"
    if aggregate_path.exists():
        agg = json.loads(aggregate_path.read_text())
        ratio = agg.get("ratio_C2_over_B")
        logger.info("=" * 72)
        logger.info("Test 15 multi-seed aggregate (n=%d):", agg.get("n_completed", 0))
        logger.info("  mean retention(B)  = %.3f", agg["mean_retention_B"])
        logger.info("  mean retention(C2) = %.3f", agg["mean_retention_C2"])
        logger.info("  ratio C2 / B       = %s", f"{ratio:.2f}" if ratio else "n/a")
        logger.info(
            "  decision threshold = %.1f → %s",
            agg["decision_threshold_ratio"],
            (
                "HOLDS"
                if ratio is not None and ratio >= agg["decision_threshold_ratio"]
                else "DOES NOT HOLD"
            ),
        )
        logger.info("=" * 72)


if __name__ == "__main__":
    main()
