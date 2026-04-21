"""Test 13: Journal-Scaffold — Placeholder Generalization.

Does a LoRA adapter pre-trained with *placeholder* answers form a stable
(key → question → slot) structure that a later real-answer update can
fill faster, without residual leakage from the placeholder content?

This is **not** a scaling test. Scaling is settled ground:
  - Test 8 cycle 56 fit 550 keys in a single rank-8 adapter (Mistral 7B).
  - Adapter multiplication + registry namespacing handles aggregation
    beyond a single adapter's data-dependent ceiling.
Test 13 isolates a different question: whether *training cost* can be
amortized by pre-forming structure independently of content. If it
works, it unlocks per-session adapter training (silent-hours workload)
and shippable pre-trained scaffolds.

Four phases on Mistral 7B, rank 8 (production config runs N=200, swap=40):

  A — Fresh:       N brand-new keys, real (Q, A). Baseline epochs-to-recall.
  B — Answer-swap: start from A's adapter. Overwrite swap-N keys with a
                   *different* answer (same key + Q). Measures warm-start
                   savings vs A on the swap set + retention on unchanged.
  C1 — Scaffold:   fresh adapter "journal". Train N keys where swap-N
                   have placeholder answers "TBD-k". Must train to N/N —
                   placeholders must emit verbatim.
  C2 — Fill:       start from C1's adapter. Replace swap-N "TBD-k" with
                   real answers. Measures epochs-to-recall on the fills
                   + retention on unchanged + placeholder leakage.

Observables:
  - Convergence curves (first_perfect_epoch, stable_perfect_epoch) per phase.
  - Retention: does B / C2 preserve unchanged keys?
  - Leakage: does any C2 answer contain residual "TBD" tokens?
  - Data-hygiene noise floor: duplicate / near-duplicate pairs in the
    source data cause key-collision failures independent of scaffolding.
    Detected at qa_pool build and logged.

Decision rules:
  B epochs << A epochs AND retention(B) ~ 1.0 → warm-start real. Proceed.
  C2 epochs ~ B epochs AND retention(C) ~ 1.0 AND leakage == 0
      → placeholder scaffold confirmed.
  C2 shows leakage OR C2 epochs >> B epochs → scaffolding buys nothing.
  Failures limited to flagged-duplicate keys → same noise floor as A/B,
      not a scaffold regression.

Training discipline:
  One train_adapter call per phase with a RecallProbeCallback on
  on_epoch_end. Per-epoch separate Trainer instances are forbidden
  (CLAUDE.md — they reset optimizer state).

Resume:
  Phase markers A_done.json, B_done.json, C1_done.json, C2_done.json.
  On rerun, completed phases are skipped and their adapters reloaded
  from disk. Output dir chosen once via model_output_dir() and reused
  for --resume via run_config.json.

GPU prerequisite:
  The ParaMem server must release the GPU. This script uses
  experiments.utils.gpu_guard.acquire_gpu() which auto-switches the
  server to cloud-only for the duration.

Usage:
    python experiments/test13_journal_scaffold.py                       # full A+B+C1+C2
    python experiments/test13_journal_scaffold.py --phase A             # just A
    python experiments/test13_journal_scaffold.py --resume              # auto-find latest

Data-hygiene finding (2026-04-21, Phase A post-run probe, N=200):
  Phase A peaked at 199/200 at stable epoch. The failing key is graph23,
  whose source triple collides with graph26 — same question verbatim,
  answers are paraphrases of the same facts. The model emitted graph26's
  wording under graph23's key; SimHash confidence 0.734, just below the
  0.75 threshold. Failure mode: content-collision from duplicate upstream
  triples, not capacity and not a placeholder-induced regression.
  Follow-up: triple-level dedup ablation (out of scope for this run).
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from peft import PeftModel  # noqa: E402
from transformers import TrainerCallback  # noqa: E402

from experiments.utils.gpu_guard import acquire_gpu  # noqa: E402
from experiments.utils.perltqa_loader import is_available as perltqa_available  # noqa: E402
from experiments.utils.perltqa_loader import load_character_eval_qa  # noqa: E402
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

OUTPUT_DIR = project_root / "outputs" / "test13_journal_scaffold"
TOTAL_KEYS = 100
SWAP_KEYS = 20  # indices 81..100 (1-based) / slots[80:100] (0-based)
SWAP_START_SLOT = TOTAL_KEYS - SWAP_KEYS  # 80
STABLE_EPOCH_WINDOW = 2  # consecutive perfect epochs for "stable"
CHARACTER_A = "Liang Xin"
CHARACTER_B = "Cai Xiuying"  # fallback when primary character runs short
PAUSE_FILE = Path.home() / ".training_pause"


# ----------------------------------------------------------------------------
# Callback: per-epoch recall probe on a target subset
# ----------------------------------------------------------------------------


@dataclass
class EpochProbeState:
    """Captures per-epoch recall metrics for a training phase."""

    first_perfect_epoch: int | None = None
    stable_perfect_epoch: int | None = None
    epoch_log: list[dict] = field(default_factory=list)


class RecallProbeCallback(TrainerCallback):
    """Probe a target key subset after each epoch, early-stop on stable perfect.

    Writes first_perfect_epoch and stable_perfect_epoch (2 consecutive
    perfect epochs) into `state_out`. Do NOT pass num_epochs=1 + multiple
    Trainer calls — that resets Adam state. Use this callback within a
    single train_adapter call with full epoch budget.
    """

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
        # Re-enable gradient checkpointing for the next training epoch —
        # evaluate_indexed_recall disables it for generation.
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
            # Look back at the last `stable_window` epochs including this one.
            # Record the epoch observationally — do NOT stop training.
            # Prior work (benchmarking.md §"Early Stopping Exploration 2026-03-23")
            # showed recall is a phase transition with a 5-epoch window; stopping
            # inside the window loses stability margin and degrades Mistral recall.
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


# ----------------------------------------------------------------------------
# Dataset preparation
# ----------------------------------------------------------------------------


def load_qa_pool(total_keys: int) -> list[dict]:
    """Load total_keys+SWAP_KEYS QA pairs — the extras supply swap answers.

    Logs duplicate questions and answers within the training slice
    (slots 1..total_keys). These are known to cause SimHash collisions
    at recall time and establish a noise floor independent of the
    scaffold mechanism under test (see module docstring, graph23
    finding).
    """
    needed = total_keys + SWAP_KEYS
    if not perltqa_available():
        raise RuntimeError("PerLTQA dataset not available — required for Test 13")
    qa = load_character_eval_qa(CHARACTER_A, max_pairs=needed)
    if len(qa) < needed:
        extra = load_character_eval_qa(CHARACTER_B, max_pairs=needed - len(qa))
        qa.extend(extra)
    if len(qa) < needed:
        raise RuntimeError(f"Need {needed} QA pairs, got {len(qa)}")

    training_slice = qa[:total_keys]
    dup_q: dict[str, list[int]] = {}
    dup_a: dict[str, list[int]] = {}
    for idx, pair in enumerate(training_slice):
        dup_q.setdefault(pair["question"], []).append(idx + 1)
        dup_a.setdefault(pair["answer"], []).append(idx + 1)
    for question, slots in dup_q.items():
        if len(slots) > 1:
            logger.warning(
                "Data-hygiene: duplicate question at graph%s: %r",
                "/graph".join(str(s) for s in slots),
                question[:80],
            )
    for answer, slots in dup_a.items():
        if len(slots) > 1:
            logger.warning(
                "Data-hygiene: duplicate answer at graph%s: %r",
                "/graph".join(str(s) for s in slots),
                answer[:80],
            )

    return qa[:needed]


def build_phase_A_keyed(qa_pool: list[dict]) -> list[dict]:
    """Phase A: 100 keys, real (Q, A)."""
    return assign_keys(qa_pool[:TOTAL_KEYS], start_index=1)


def build_phase_B_swap_keyed(base_keyed: list[dict], swap_answers: list[dict]) -> list[dict]:
    """Phase B swap set: 20 keys, same key + Q, different A.

    swap_answers supplies the replacement answer text (only "answer" used).
    Key + question are carried over from base_keyed (slots 81..100).
    """
    assert len(swap_answers) >= SWAP_KEYS
    swap_keyed = []
    for i, kp in enumerate(base_keyed[SWAP_START_SLOT:]):
        replacement = swap_answers[i]["answer"]
        if replacement.strip() == kp["answer"].strip():
            # Guard against coincidental collision — append a disambiguator
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
    """Phase C1: 80 real + 20 placeholder "TBD-k" keys (k = 1..20)."""
    mixed = []
    for i, qa in enumerate(qa_pool[:TOTAL_KEYS]):
        if i < SWAP_START_SLOT:
            mixed.append(qa)
        else:
            slot_k = i - SWAP_START_SLOT + 1  # 1..20
            mixed.append(
                {
                    "question": qa["question"],
                    "answer": f"TBD-{slot_k}",
                }
            )
    return assign_keys(mixed, start_index=1)


def build_phase_C2_fill_keyed(c1_keyed: list[dict], qa_pool: list[dict]) -> list[dict]:
    """Phase C2: replace each TBD-k with the corresponding real answer."""
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


# ----------------------------------------------------------------------------
# Training helpers
# ----------------------------------------------------------------------------


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
) -> tuple[dict, EpochProbeState, float]:
    """Run one train_adapter call with RecallProbeCallback.

    Returns (metrics, probe_state, wall_time).
    """
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
    """Persist keyed_pairs, registry, probe log, and a phase-done marker."""
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
    """Count how many per-key results still contain a 'TBD-' substring."""
    count = 0
    for entry in recall_per_key:
        recalled = entry.get("recalled") or {}
        if isinstance(recalled, dict) and "TBD-" in str(recalled.get("answer", "")):
            count += 1
    return count


# ----------------------------------------------------------------------------
# Phases
# ----------------------------------------------------------------------------


def phase_A(
    model,
    tokenizer,
    qa_pool: list[dict],
    args,
    output_dir: Path,
) -> tuple[object, list[dict], dict, EpochProbeState]:
    logger.info("=" * 72)
    logger.info("Phase A — Fresh: %d keys, up to %d epochs", TOTAL_KEYS, args.num_epochs)
    logger.info("=" * 72)

    keyed = build_phase_A_keyed(qa_pool)
    registry = build_registry(keyed)

    adapter_config = AdapterConfig(
        rank=args.rank,
        alpha=args.rank * 2,
        learning_rate=1e-4,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        dropout=0.0,
    )
    model = create_adapter(model, adapter_config, "episodic")

    training_config = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=2,
        max_seq_length=1024,
        num_epochs=args.num_epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=42,
    )

    phase_dir = output_dir / "A"
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
        run_name="test13-A-fresh",
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
) -> EpochProbeState:
    logger.info("=" * 72)
    logger.info("Phase B — Answer-swap: %d keys", SWAP_KEYS)
    logger.info("=" * 72)

    swap_keyed = build_phase_B_swap_keyed(base_keyed, swap_answers)
    swap_registry = build_registry(swap_keyed)

    # Retention set: the 80 unchanged keys from A.
    unchanged_keyed = base_keyed[:SWAP_START_SLOT]
    unchanged_registry = build_registry(unchanged_keyed)

    adapter_config = AdapterConfig(
        rank=args.rank,
        alpha=args.rank * 2,
        learning_rate=1e-4,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        dropout=0.0,
    )

    training_config = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=2,
        max_seq_length=1024,
        num_epochs=args.num_epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=42,
    )

    switch_adapter(model, "episodic")
    phase_dir = output_dir / "B"
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
        run_name="test13-B-swap",
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
) -> tuple[object, list[dict], dict, EpochProbeState]:
    logger.info("=" * 72)
    logger.info(
        "Phase C1 — Scaffold: %d keys (%d real + %d placeholder TBD-k)",
        TOTAL_KEYS,
        SWAP_START_SLOT,
        SWAP_KEYS,
    )
    logger.info("=" * 72)

    keyed = build_phase_C1_keyed(qa_pool)
    registry = build_registry(keyed)

    adapter_config = AdapterConfig(
        rank=args.rank,
        alpha=args.rank * 2,
        learning_rate=1e-4,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        dropout=0.0,
    )
    model = create_adapter(model, adapter_config, "journal")

    training_config = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=2,
        max_seq_length=1024,
        num_epochs=args.num_epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=42,
    )

    phase_dir = output_dir / "C1"
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
        run_name="test13-C1-scaffold",
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
) -> EpochProbeState:
    logger.info("=" * 72)
    logger.info("Phase C2 — Fill: %d keys (TBD-k → real)", SWAP_KEYS)
    logger.info("=" * 72)

    fill_keyed = build_phase_C2_fill_keyed(c1_keyed, qa_pool)
    fill_registry = build_registry(fill_keyed)

    unchanged_keyed = c1_keyed[:SWAP_START_SLOT]
    unchanged_registry = build_registry(unchanged_keyed)

    adapter_config = AdapterConfig(
        rank=args.rank,
        alpha=args.rank * 2,
        learning_rate=1e-4,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        dropout=0.0,
    )

    training_config = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=2,
        max_seq_length=1024,
        num_epochs=args.num_epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=42,
    )

    switch_adapter(model, "journal")
    phase_dir = output_dir / "C2"
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
        run_name="test13-C2-fill",
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


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------


def marker_exists(output_dir: Path, phase: str) -> bool:
    return (output_dir / phase / f"{phase}_done.json").exists()


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
        "Pause signal detected after Phase %s — exiting cleanly (paused.json written). "
        "Use tresume to continue.",
        after_phase,
    )


def find_latest_run_dir(model_name: str) -> Path | None:
    parent = OUTPUT_DIR / model_name
    if not parent.is_dir():
        return None
    runs = sorted(d for d in parent.iterdir() if d.is_dir())
    return runs[-1] if runs else None


def load_or_write_run_config(output_dir: Path, args) -> dict:
    """First launch: write run_config.json from args. Resume: read and apply."""
    cfg_path = output_dir / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        logger.info("Loaded run_config.json: %s", cfg)
        return cfg
    cfg = {
        "model": args.model,
        "total_keys": args.total_keys,
        "swap_keys": args.swap_keys,
        "num_epochs": args.num_epochs,
        "rank": args.rank,
        "smoke": bool(args.smoke),
        "created_at": int(time.time()),
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))
    logger.info("Wrote run_config.json: %s", cfg)
    return cfg


def read_keyed(output_dir: Path, phase: str) -> tuple[list[dict], dict[str, int]]:
    """Reload keyed_pairs + registry for a completed phase."""
    with open(output_dir / phase / "keyed_pairs.json") as f:
        keyed = json.load(f)
    with open(output_dir / phase / "simhash_registry.json") as f:
        registry = json.load(f)
    # Registry file uses str values by default — normalize to int.
    registry = {k: int(v) for k, v in registry.items()}
    return keyed, registry


def main():
    parser = argparse.ArgumentParser(description="Test 13: Journal-Scaffold Probe")
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--total-keys", type=int, default=100)
    parser.add_argument("--swap-keys", type=int, default=20)
    parser.add_argument("--model", choices=["mistral"], default="mistral")
    parser.add_argument(
        "--phase",
        choices=["A", "B", "C1", "C2", "all"],
        default="all",
        help="Run a specific phase (requires completed predecessors on disk)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Target output directory. If omitted and --resume not set, a new "
        "timestamped dir is created under outputs/test13_journal_scaffold/<model>/.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Auto-find the latest run dir for --model and continue from where "
        "it left off. Skips completed phases. Reads run_config.json.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast infra smoke: total_keys=20, swap_keys=5, num_epochs=5.",
    )
    args = parser.parse_args()

    if args.smoke:
        args.total_keys = 20
        args.swap_keys = 5
        args.num_epochs = 5
        logger.info("SMOKE mode: total_keys=20, swap_keys=5, num_epochs=5")

    # Resolve output_dir: explicit > resume > fresh timestamp
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    elif args.resume:
        latest = find_latest_run_dir(args.model)
        if latest is None:
            logger.warning(
                "--resume requested but no prior run for %s — starting fresh",
                args.model,
            )
            output_dir = model_output_dir(OUTPUT_DIR, args.model)
        else:
            output_dir = latest
            logger.info("Resuming from %s", output_dir)
    else:
        output_dir = model_output_dir(OUTPUT_DIR, args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", output_dir)

    # Load or write run_config.json; apply its values (resume-stable sizing)
    cfg = load_or_write_run_config(output_dir, args)
    args.total_keys = cfg["total_keys"]
    args.swap_keys = cfg["swap_keys"]
    args.num_epochs = cfg["num_epochs"]
    args.rank = cfg["rank"]

    # Rebind module globals so downstream phase builders see the active sizing.
    global TOTAL_KEYS, SWAP_KEYS, SWAP_START_SLOT
    TOTAL_KEYS = args.total_keys
    SWAP_KEYS = args.swap_keys
    SWAP_START_SLOT = TOTAL_KEYS - SWAP_KEYS

    model_config = BENCHMARK_MODELS[args.model]
    qa_pool = load_qa_pool(TOTAL_KEYS)
    logger.info("Loaded %d QA pairs from PerLTQA", len(qa_pool))

    phases_wanted = ["A", "B", "C1", "C2"] if args.phase == "all" else [args.phase]

    if paused_requested():
        logger.warning("Pause file present at launch — clearing before starting work")
        try:
            PAUSE_FILE.unlink()
        except OSError:
            pass

    # Clear any stale paused.json from a prior pause (tresume case)
    stale = output_dir / "paused.json"
    if stale.exists():
        stale.unlink()

    with acquire_gpu(interactive=True):
        # Single base-model load for the whole run — avoids WSL2 CUDA
        # reload instability (see CLAUDE.md "GPU / CUDA / WSL2"). Adapter
        # transitions use unwrap→create_adapter per test 10's pattern.
        model, tokenizer, _ = load_model_and_config(model_config)
        base_keyed = None
        c1_keyed = None

        if "A" in phases_wanted:
            if marker_exists(output_dir, "A"):
                logger.info("A already complete — loading adapter from disk")
                _slot_A = resolve_adapter_slot(output_dir / "A" / "adapter", "episodic", "")
                if _slot_A is not None:
                    if isinstance(model, PeftModel):
                        model.load_adapter(str(_slot_A), adapter_name="episodic")
                    else:
                        model = PeftModel.from_pretrained(
                            model, str(_slot_A), adapter_name="episodic"
                        )
                else:
                    model = load_adapter(model, output_dir / "A" / "adapter", "episodic")
                base_keyed, _ = read_keyed(output_dir, "A")
            else:
                model, base_keyed, _, _ = phase_A(model, tokenizer, qa_pool, args, output_dir)
                if paused_requested():
                    unload_model(model, tokenizer)
                    write_paused_marker(output_dir, "A")
                    return

        if "B" in phases_wanted:
            if marker_exists(output_dir, "B"):
                logger.info("B already complete — skipping")
            else:
                if base_keyed is None:
                    logger.info("Loading A's adapter for B warm-start")
                    _slot_A_b = resolve_adapter_slot(output_dir / "A" / "adapter", "episodic", "")
                    if _slot_A_b is not None:
                        if isinstance(model, PeftModel):
                            model.load_adapter(str(_slot_A_b), adapter_name="episodic")
                        else:
                            model = PeftModel.from_pretrained(
                                model, str(_slot_A_b), adapter_name="episodic"
                            )
                    else:
                        model = load_adapter(model, output_dir / "A" / "adapter", "episodic")
                    base_keyed, _ = read_keyed(output_dir, "A")
                swap_answers = qa_pool[TOTAL_KEYS : TOTAL_KEYS + SWAP_KEYS]
                phase_B(model, tokenizer, base_keyed, swap_answers, args, output_dir)
                if paused_requested():
                    unload_model(model, tokenizer)
                    write_paused_marker(output_dir, "B")
                    return

        # A+B → C1+C2 transition: cooldown (model stays resident), then unwrap
        # the "episodic" adapter so phase_C1 can create a fresh "journal" one.
        if any(p in phases_wanted for p in ("C1", "C2")):
            try:
                import subprocess

                subprocess.run(
                    ["bash", "-c", "source ~/.local/bin/gpu-cooldown.sh && wait_for_cooldown 52"],
                    check=False,
                    timeout=300,
                )
            except Exception as exc:
                logger.warning("Cooldown skipped: %s", exc)

            if isinstance(model, PeftModel):
                model = model.base_model.model

            if "C1" in phases_wanted:
                if marker_exists(output_dir, "C1"):
                    logger.info("C1 already complete — loading adapter from disk")
                    _slot_C1 = resolve_adapter_slot(output_dir / "C1" / "adapter", "journal", "")
                    if _slot_C1 is not None:
                        if isinstance(model, PeftModel):
                            model.load_adapter(str(_slot_C1), adapter_name="journal")
                        else:
                            model = PeftModel.from_pretrained(
                                model, str(_slot_C1), adapter_name="journal"
                            )
                    else:
                        model = load_adapter(model, output_dir / "C1" / "adapter", "journal")
                    c1_keyed, _ = read_keyed(output_dir, "C1")
                else:
                    model, c1_keyed, _, _ = phase_C1(model, tokenizer, qa_pool, args, output_dir)
                    if paused_requested():
                        unload_model(model, tokenizer)
                        write_paused_marker(output_dir, "C1")
                        return

            if "C2" in phases_wanted:
                if marker_exists(output_dir, "C2"):
                    logger.info("C2 already complete — skipping")
                else:
                    if c1_keyed is None:
                        logger.info("Loading C1's adapter for C2 fill")
                        _slot_C1_c2 = resolve_adapter_slot(
                            output_dir / "C1" / "adapter", "journal", ""
                        )
                        if _slot_C1_c2 is not None:
                            if isinstance(model, PeftModel):
                                model.load_adapter(str(_slot_C1_c2), adapter_name="journal")
                            else:
                                model = PeftModel.from_pretrained(
                                    model, str(_slot_C1_c2), adapter_name="journal"
                                )
                        else:
                            model = load_adapter(model, output_dir / "C1" / "adapter", "journal")
                        c1_keyed, _ = read_keyed(output_dir, "C1")
                    phase_C2(model, tokenizer, c1_keyed, qa_pool, args, output_dir)
                    if paused_requested():
                        unload_model(model, tokenizer)
                        write_paused_marker(output_dir, "C2")
                        return

        unload_model(model, tokenizer)

    # --- Summary ------------------------------------------------------------
    summary = {"output_dir": str(output_dir), "phases": {}}
    for phase in ("A", "B", "C1", "C2"):
        marker = output_dir / phase / f"{phase}_done.json"
        if marker.exists():
            with open(marker) as f:
                summary["phases"][phase] = json.load(f)

    save_results(summary, output_dir, "results.json")

    logger.info("=" * 72)
    logger.info("Test 13 summary (epochs-to-stable-perfect):")
    for phase in ("A", "B", "C1", "C2"):
        data = summary["phases"].get(phase)
        if not data:
            continue
        stable = data.get("stable_perfect_epoch")
        first = data.get("first_perfect_epoch")
        n = data.get("n_keys")
        leak = data.get("placeholder_leakage_on_fills")
        ret = (
            data.get("retention_unchanged_80", {}).get("rate")
            if "retention_unchanged_80" in data
            else None
        )
        logger.info(
            "  %s  n=%d  first_perfect=%s  stable_perfect=%s  retention=%s  leakage=%s",
            phase,
            n or -1,
            first,
            stable,
            f"{ret:.3f}" if ret is not None else "-",
            leak if leak is not None else "-",
        )
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
