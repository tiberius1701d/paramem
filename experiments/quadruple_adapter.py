"""Quadruple-encoded indexed-key adapter, pausable + resumable.

Trains a fresh LoRA adapter on (key, subject, predicate, object) quadruples
distilled from a source graph, then probes every key and compares the parsed
JSON envelope back to the trained values.

Designed to run forever in the test-harness:
- Honors ``~/.training_pause`` at every phase boundary AND inside the trainer
  loop (via ``TrainingHooks.on_shutdown_check``).
- Writes ``train_done.json`` / ``probe_done.json`` markers; on resume those
  phases are skipped.
- ``--resume --n-keys N`` (with N greater than the saved value) extends the
  per-cycle key budget — markers are cleared and the adapter retrains on the
  extended set (full replay; per CLAUDE.md "Incremental addition via full
  replay works; true incremental learning is open research").
- Adapter checkpoints are preserved between runs in ``<run_dir>/adapter/``.
- Epoch-level resume: if ``adapter/checkpoint-*`` dirs exist when train_phase
  is entered (because a prior run paused mid-training), ``resume_from_checkpoint``
  is set to the highest-numbered checkpoint so training continues from that
  epoch rather than restarting from epoch 0.
- Per-epoch recall probing via ``_QuadRecallEarlyStop`` callback (fires
  ``control.should_training_stop`` when ``policy.window`` consecutive probes
  achieve 100% strict recall AND ``epoch >= policy.signal_from_epoch``).
- Probe results persisted every 25 keys; incremental resume skips already-
  probed keys.

Default source snapshot: ``outputs/lme_graph/graph_snapshot.json`` (built by
``experiments/lme_graph_builder.py``).  Pass ``--graph-snapshot <path>`` to
use any other snapshot.  On startup, if the snapshot is missing or has fewer
unique triples than ``--n-keys``, the script exits with a clear message.

CLI surface mirrors test14 conventions:
    python experiments/quadruple_adapter.py --n-keys 50
    python experiments/quadruple_adapter.py --resume                # continue
    python experiments/quadruple_adapter.py --resume --n-keys 200   # extend
    python experiments/quadruple_adapter.py --graph-snapshot data/ha/.../graph_snapshot.json

Outputs (under ``outputs/quad_scale/<model>/<ts>/``):
    run_config.json          — frozen config (n_keys updated on extension)
    quads.json               — keyed quadruples used for the current N
    adapter/                 — trained LoRA weights (HF Trainer output dir)
    epoch_log.json           — per-epoch probe records (strict_rate per epoch)
    train_done.json          — marker; train phase complete
    probe_results.json       — per-key raw probe output (persisted every 25 keys)
    probe_done.json          — marker; probe phase complete
    results.json             — per-key evaluation records
    metrics.json             — overall aggregate
    report.md                — human-readable summary
    paused.json              — only present while paused
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import networkx as nx

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("quad")

PAUSE_FILE = Path.home() / ".training_pause"

# Default source snapshot: the LME graph builder's canonical output.
# Override with --graph-snapshot to use any other snapshot (e.g. the old
# CV debug snapshot).
DEFAULT_GRAPH_SNAPSHOT = project_root / "outputs" / "lme_graph" / "graph_snapshot.json"

# Legacy CV snapshot path — kept as a documented alternative for --graph-snapshot.
# Equivalent to the old DEFAULT_GRAPH_SNAPSHOT for callers that still have it.
_LEGACY_CV_SNAPSHOT = (
    project_root / "data/ha/debug/run_20260510T170022Z_8c1cca/cycle_26/graph_snapshot.json"
)


# ---------------------------------------------------------------------------
# Pause / marker helpers (mirrors experiments/test14.py conventions)
# ---------------------------------------------------------------------------


def paused_requested() -> bool:
    return PAUSE_FILE.exists()


def _check_pause(label: str, run_dir: Path | None = None) -> None:
    """Raise SystemExit cleanly when the global pause flag is set."""
    if PAUSE_FILE.exists():
        logger.warning("Pause file detected at %s — halting cleanly.", label)
        if run_dir is not None:
            write_paused_marker(run_dir, label)
        raise SystemExit(f"Training paused at {label}")


def marker_exists(run_dir: Path, marker_name: str) -> bool:
    return (run_dir / f"{marker_name}_done.json").exists()


def write_phase_done(run_dir: Path, marker: str, payload: dict) -> None:
    from paramem.training.early_stop import _safe_write_json

    _safe_write_json(run_dir / f"{marker}_done.json", payload)


def clear_phase_done(run_dir: Path, marker: str) -> None:
    p = run_dir / f"{marker}_done.json"
    if p.exists():
        try:
            p.unlink()
        except OSError:
            pass


def write_paused_marker(
    run_dir: Path,
    after: str,
    after_epoch: int | None = None,
    latest_checkpoint: str | None = None,
) -> None:
    """Write paused.json to signal a clean pause.

    Args:
        run_dir: Run directory receiving the marker.
        after: Label describing where training stopped.
        after_epoch: Epoch number at which training stopped (if known).
        latest_checkpoint: Path to the latest HF checkpoint dir (if known),
            so the resume path can surface this in tstatus.
    """
    from paramem.training.early_stop import _safe_write_json

    payload: dict = {
        "stopped_after": after,
        "stopped_after_epoch": after_epoch,
        "timestamp": int(time.time()),
    }
    if latest_checkpoint is not None:
        payload["latest_checkpoint"] = str(latest_checkpoint)
    _safe_write_json(
        run_dir / "paused.json",
        payload,
    )
    logger.info("paused.json written (after=%s)", after)


def clear_paused_marker(run_dir: Path) -> None:
    p = run_dir / "paused.json"
    if p.exists():
        try:
            p.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Run-dir resolution + config persistence
# ---------------------------------------------------------------------------


def find_latest_run_dir(model_name: str) -> Path | None:
    parent = project_root / "outputs" / "quad_scale" / model_name
    if not parent.is_dir():
        return None
    candidates = [
        d for d in sorted(parent.iterdir()) if d.is_dir() and (d / "run_config.json").exists()
    ]
    return candidates[-1] if candidates else None


def load_or_write_run_config(run_dir: Path, args: argparse.Namespace) -> dict:
    """Freeze config on first launch; on resume read back; extend n_keys when larger.

    Early-stop parameters (es_from_epoch, es_window, es_probe_every,
    es_probe_sample, no_early_stop) are persisted so ``tresume`` (which calls
    the script with only ``--resume``) recovers them without re-specifying them.

    Args:
        run_dir: Run directory to read/write ``run_config.json``.
        args: Parsed CLI namespace.

    Returns:
        Config dict (either loaded from disk or freshly written).
    """
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        logger.info("Loaded run_config.json: %s", cfg)
        if args.n_keys is not None:
            saved_n = cfg["n_keys"]
            if args.n_keys < saved_n:
                raise SystemExit(
                    f"--n-keys cannot shrink across resumes "
                    f"(saved={saved_n}, requested={args.n_keys})"
                )
            if args.n_keys > saved_n:
                logger.info("Extending n_keys from %d to %d", saved_n, args.n_keys)
                cfg["n_keys"] = args.n_keys
                cfg_path.write_text(json.dumps(cfg, indent=2))
                # Extension == full-replay retrain on the larger set. Wipe ALL
                # prior-run artifacts so nothing warm-starts from the smaller
                # adapter: phase markers, the HF checkpoint tree (else
                # resume_from_checkpoint picks up the stale checkpoint), the
                # epoch log (else _QuadRecallEarlyStop rehydrates a stale streak
                # and skips every probe), and probe_results.json (else the new
                # probe phase reuses results probed against the old adapter).
                clear_phase_done(run_dir, "train")
                clear_phase_done(run_dir, "probe")
                shutil.rmtree(run_dir / "adapter", ignore_errors=True)
                for stale in ("epoch_log.json", "probe_results.json", "paused.json"):
                    (run_dir / stale).unlink(missing_ok=True)
        return cfg

    cfg = {
        "model": args.model,
        "n_keys": args.n_keys,
        "num_epochs": args.num_epochs,
        "rank": args.rank,
        "graph_snapshot": str(args.graph_snapshot.resolve()),
        # Early-stop parameters persisted so tresume recovers them.
        "es_from_epoch": args.es_from_epoch,
        "es_window": args.es_window,
        "es_probe_every": args.es_probe_every,
        "es_probe_sample": args.es_probe_sample,
        "no_early_stop": args.no_early_stop,
        "created_at": int(time.time()),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2))
    logger.info("Wrote run_config.json: %s", cfg)
    return cfg


# ---------------------------------------------------------------------------
# Triples + key assignment
# ---------------------------------------------------------------------------


def load_unique_triples(graph_path: Path) -> list[tuple[str, str, str]]:
    """Return de-duplicated (subject, predicate, object) triples from the graph."""
    data = json.loads(graph_path.read_text())
    graph = nx.node_link_graph(data)
    seen = set()
    out: list[tuple[str, str, str]] = []
    for s, o, d in graph.edges(data=True):
        p = d.get("predicate", "")
        norm = (
            s.strip().lower(),
            p.strip().lower().replace(" ", "_"),
            o.strip().lower(),
        )
        if norm in seen:
            continue
        seen.add(norm)
        out.append((s, p, o))
    return out


# ---------------------------------------------------------------------------
# Tiny dataset wrapper (matches the phase4 pattern)
# ---------------------------------------------------------------------------


class IndexedDataset:
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _find_highest_checkpoint(adapter_dir: Path) -> Optional[Path]:
    """Return the path to the highest-numbered ``checkpoint-*`` dir, or None.

    Sorting is numeric (not lexicographic) so ``checkpoint-300`` beats
    ``checkpoint-90`` (per CLAUDE.md: sort checkpoint dirs numerically).

    Args:
        adapter_dir: HF Trainer output directory that may contain checkpoint
            subdirectories.

    Returns:
        Path to the highest ``checkpoint-<N>`` directory, or None if none exist.
    """
    if not adapter_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for p in adapter_dir.iterdir():
        if p.is_dir() and p.name.startswith("checkpoint-"):
            try:
                n = int(p.name.split("-")[-1])
                candidates.append((n, p))
            except ValueError:
                pass
    if not candidates:
        return None
    return max(candidates, key=lambda t: t[0])[1]


# ---------------------------------------------------------------------------
# Early-stop callback (quadruple-aware, no SimHash dependency)
# ---------------------------------------------------------------------------


class _QuadRecallEarlyStop:
    """Per-epoch quadruple recall probe with early-stop on ``policy.window``
    consecutive fully-perfect probes.

    This is an HF ``TrainerCallback`` subclass that probes a fixed random
    sample of quads after each probe epoch, counting those whose parsed
    ``(subject, predicate, object)`` exactly matches the trained triple.
    Fires ``control.should_training_stop = True`` when the last ``policy.window``
    recorded probes were all ``strict_rate == 1.0`` and
    ``epoch >= policy.signal_from_epoch``.

    Rehydrates prior state from ``epoch_log_path`` on construction so a
    resumed run continues the window streak instead of restarting it from zero.

    The callback is separate from ``RecallEarlyStopCallback`` (which is
    hard-bound to a SimHash registry and a QA-pair fill set that this experiment does
    not have).

    Gradient-checkpointing discipline: disables checkpointing before
    ``probe_quad`` (which calls ``generate_answer`` / ``model.generate()``) and
    re-enables it after all probes complete, per the CLAUDE.md rule that
    gradient_checkpointing must be disabled before any ``model.generate()`` call.
    The model's ``training`` flag is restored after probing so HF Trainer's
    subsequent forward pass runs in train mode.

    Args:
        model: PeftModel being trained.
        tokenizer: Tokenizer for generation.
        quads: Full list of quad dicts (all trained keys).
        policy: ``EarlyStopPolicy`` controlling probe schedule and stop gate.
        epoch_log_path: Path to write/read ``epoch_log.json`` incrementally.
        pause_file: Path to the global pause semaphore. When set and the file
            exists at epoch-end, the callback simply returns — the outer
            ``TrainingHooks.on_shutdown_check`` handles the actual pause; the
            callback must not double-fire the stop signal.
        sample_size: Number of quads to probe per epoch (random sample).
        seed: Random seed for the fixed sample selection.
    """

    def __init__(
        self,
        *,
        model,
        tokenizer,
        quads: list[dict],
        policy,
        epoch_log_path: Path,
        pause_file: Optional[Path],
        sample_size: int,
        seed: int,
    ) -> None:
        """Initialise and rehydrate from existing epoch_log if present."""
        # Import here to keep module-level imports GPU-free.
        from transformers import TrainerCallback

        self._model = model
        self._tokenizer = tokenizer
        self._policy = policy
        self._epoch_log_path = epoch_log_path
        self._pause_file = pause_file

        # Draw a fixed random sample of quads (same sample every epoch).
        rng = random.Random(seed)
        sample_n = min(sample_size, len(quads))
        self._sampled_quads: list[dict] = rng.sample(quads, sample_n)

        # Public result attributes read by caller after train_adapter returns.
        self.first_perfect_epoch: Optional[int] = None
        self.stable_perfect_epoch: Optional[int] = None
        self.stop_epoch: Optional[int] = None

        # Internal accumulator.
        self._epoch_log: list[dict] = []
        self._last_epoch: int = -1
        self._consecutive_perfect: int = 0

        # Rehydrate from disk so a resumed run restores the streak.
        self._rehydrate_from_disk()

        # Expose on_epoch_end as a method on a dynamically-created
        # TrainerCallback subclass so HF Trainer can register it.
        _outer = self

        class _HFAdapter(TrainerCallback):
            def on_epoch_end(self, args, state, control, **kwargs):
                _outer._on_epoch_end(args, state, control)

        self._hf_callback = _HFAdapter()

    def _rehydrate_from_disk(self) -> None:
        """Restore in-memory state from ``epoch_log_path`` if it exists.

        Failure is swallowed — a missing or corrupt file is treated as a cold
        start.  ``stop_epoch`` is intentionally NOT rehydrated: if the prior
        run had stopped we would not be resuming.
        """
        if not self._epoch_log_path.exists():
            return
        try:
            saved_log = json.loads(self._epoch_log_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(saved_log, list) or not saved_log:
            return

        self._epoch_log = list(saved_log)
        try:
            self._last_epoch = int(saved_log[-1].get("epoch", -1))
        except (TypeError, ValueError):
            pass

        consecutive = 0
        window = self._policy.window
        for entry in saved_log:
            sr = entry.get("strict_rate", 0.0)
            is_perfect = sr == 1.0 and entry.get("n_total", 0) > 0
            consecutive = consecutive + 1 if is_perfect else 0
            if is_perfect and self.first_perfect_epoch is None:
                try:
                    self.first_perfect_epoch = int(entry["epoch"])
                except (KeyError, TypeError, ValueError):
                    pass
            if self.stable_perfect_epoch is None and consecutive >= window:
                try:
                    self.stable_perfect_epoch = int(entry["epoch"])
                except (KeyError, TypeError, ValueError):
                    pass
        self._consecutive_perfect = consecutive

    def _on_epoch_end(self, args, state, control) -> None:
        """Run per-epoch probe and update early-stop state.

        Called by the HF adapter callback.  Skips epochs before
        ``policy.probe_from_epoch`` or off the ``probe_every_n_epochs``
        cadence.  Always returns without firing stop if the pause file is set
        (the outer ``TrainingHooks.on_shutdown_check`` owns the pause path).

        Args:
            args: HF ``TrainingArguments``.
            state: HF ``TrainerState``.
            control: HF ``TrainerControl`` — mutated to signal stop.
        """
        from paramem.training.early_stop import _safe_write_json

        epoch = int(round(state.epoch))

        # Guard: HF Trainer may fire on_epoch_end twice for the same epoch.
        if epoch <= self._last_epoch:
            return
        self._last_epoch = epoch

        policy = self._policy

        # Determine whether to probe this epoch.
        if epoch < policy.probe_from_epoch:
            return
        if (epoch - policy.probe_from_epoch) % policy.probe_every_n_epochs != 0:
            return

        # If pause is requested, let TrainingHooks handle it; do not probe.
        if self._pause_file is not None and self._pause_file.exists():
            return

        from experiments.utils.quadruple_format import probe_quad

        # Disable gradient checkpointing before generate() (CLAUDE.md rule).
        was_training = self._model.training
        self._model.gradient_checkpointing_disable()

        n_perfect = 0
        n_total = len(self._sampled_quads)
        for quad in self._sampled_quads:
            result = probe_quad(self._model, self._tokenizer, quad["key"])
            if result is None or "failure_reason" in result:
                continue
            recalled = (
                result.get("subject", "").strip().lower(),
                result.get("predicate", "").strip().lower().replace(" ", "_"),
                result.get("object", "").strip().lower(),
            )
            trained = (
                quad["subject"].strip().lower(),
                quad["predicate"].strip().lower().replace(" ", "_"),
                quad["object"].strip().lower(),
            )
            if recalled == trained:
                n_perfect += 1

        # Re-enable gradient checkpointing and restore training mode.
        self._model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if was_training:
            self._model.train()

        strict_rate = n_perfect / n_total if n_total > 0 else 0.0
        entry = {
            "epoch": epoch,
            "n_perfect": n_perfect,
            "n_total": n_total,
            "strict_rate": strict_rate,
        }
        self._epoch_log.append(entry)
        _safe_write_json(self._epoch_log_path, self._epoch_log)

        logger.info(
            "  epoch %d recall probe: %d/%d (strict=%.3f)",
            epoch,
            n_perfect,
            n_total,
            strict_rate,
        )

        # First-perfect / stable-perfect tracking.
        is_perfect = strict_rate == 1.0 and n_total > 0
        if is_perfect and self.first_perfect_epoch is None:
            self.first_perfect_epoch = epoch
            logger.info("  first_perfect_epoch = %d", epoch)

        if is_perfect:
            self._consecutive_perfect += 1
        else:
            self._consecutive_perfect = 0

        if self.stable_perfect_epoch is None and self._consecutive_perfect >= policy.window:
            self.stable_perfect_epoch = epoch
            logger.info("  stable_perfect_epoch = %d", epoch)

        # Early-stop signal.
        if (
            epoch >= policy.signal_from_epoch
            and self._consecutive_perfect >= policy.window
            and self.stop_epoch is None
        ):
            control.should_training_stop = True
            self.stop_epoch = epoch
            logger.info(
                "  _QuadRecallEarlyStop: stop triggered at epoch %d "
                "(consecutive_perfect=%d >= window=%d)",
                epoch,
                self._consecutive_perfect,
                policy.window,
            )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _norm_pred(p: str) -> str:
    return p.strip().lower().replace(" ", "_")


def _norm_ent(e: str) -> str:
    return e.strip().lower()


def evaluate(quads: list[dict], probe_results: dict[str, dict | None]) -> list[dict]:
    out: list[dict] = []
    for quad in quads:
        result = probe_results.get(quad["key"])
        rec = {
            "key": quad["key"],
            "trained": (
                _norm_ent(quad["subject"]),
                _norm_pred(quad["predicate"]),
                _norm_ent(quad["object"]),
            ),
            "raw_output": (result or {}).get("raw_output", ""),
            "failure_reason": (result or {}).get("failure_reason"),
        }
        if result is None or "failure_reason" in result:
            rec.update(
                {
                    "recalled": None,
                    "source_triple_recovered": False,
                    "subject_object_match": False,
                    "subject_present": False,
                    "object_present": False,
                    "predicate_drift": False,
                }
            )
            out.append(rec)
            continue
        recalled = (
            _norm_ent(result["subject"]),
            _norm_pred(result["predicate"]),
            _norm_ent(result["object"]),
        )
        rec["recalled"] = list(recalled)
        rec["source_triple_recovered"] = recalled == rec["trained"]
        rec["subject_object_match"] = (recalled[0], recalled[2]) == (
            rec["trained"][0],
            rec["trained"][2],
        )
        rec["subject_present"] = recalled[0] == rec["trained"][0]
        rec["object_present"] = recalled[2] == rec["trained"][2]
        rec["predicate_drift"] = rec["subject_object_match"] and not rec["source_triple_recovered"]
        out.append(rec)
    return out


def aggregate(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "source_triple_recovered_rate": sum(r["source_triple_recovered"] for r in results) / n,
        "subject_object_match_rate": sum(r["subject_object_match"] for r in results) / n,
        "subject_present_rate": sum(r["subject_present"] for r in results) / n,
        "object_present_rate": sum(r["object_present"] for r in results) / n,
        "predicate_drift_rate": sum(r["predicate_drift"] for r in results) / n,
        "parse_failure_rate": sum(1 for r in results if r["failure_reason"] is not None) / n,
    }


def _fmt(k: str, v) -> str:
    if isinstance(v, float):
        return f"- **{k}**: {v:.1%}" if k.endswith("rate") else f"- **{k}**: {v:.3f}"
    return f"- **{k}**: {v}"


def write_report(out_dir: Path, overall: dict, train_meta: dict) -> None:
    lines = ["# Quadruple-Encoded Adapter", "", "## Training", ""]
    lines += [_fmt(k, v) for k, v in train_meta.items()]
    lines += ["", "## Reconstruction (overall)", ""]
    lines += [_fmt(k, v) for k, v in overall.items()]
    lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the quadruple-adapter experiment.

    Returns:
        Parsed Namespace with all options.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--graph-snapshot",
        type=Path,
        default=DEFAULT_GRAPH_SNAPSHOT,
        help=(
            "Source graph for triple extraction. "
            f"Default: {DEFAULT_GRAPH_SNAPSHOT} (built by lme_graph_builder.py). "
            f"Legacy CV snapshot: {_LEGACY_CV_SNAPSHOT}"
        ),
    )
    parser.add_argument(
        "--n-keys",
        "--n_keys",
        type=int,
        default=None,
        dest="n_keys",
        help="Per-cycle key budget. None = all unique triples in source graph.",
    )
    parser.add_argument("--model", default="mistral", choices=["mistral"])
    parser.add_argument("--num-epochs", type=int, default=30, dest="num_epochs")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Auto-find the latest run dir and continue.",
    )
    # Early-stop arguments (persisted in run_config.json so tresume recovers them).
    parser.add_argument(
        "--es-from-epoch",
        type=int,
        default=20,
        dest="es_from_epoch",
        help=("Earliest epoch at which probing starts AND the stop signal can fire (default: 20)."),
    )
    parser.add_argument(
        "--es-window",
        type=int,
        default=3,
        dest="es_window",
        help="Consecutive perfect-probe epochs required to trigger early stop (default: 3).",
    )
    parser.add_argument(
        "--es-probe-every",
        type=int,
        default=1,
        dest="es_probe_every",
        help="Probe every N epochs (default: 1 = every epoch).",
    )
    parser.add_argument(
        "--es-probe-sample",
        type=int,
        default=100,
        dest="es_probe_sample",
        help=(
            "Number of quads sampled for per-epoch recall probing (default: 100). "
            "Actual sample is min(es_probe_sample, n_keys)."
        ),
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        dest="no_early_stop",
        help="Disable the per-epoch recall early-stop callback.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------


def train_phase(
    quads: list[dict],
    run_dir: Path,
    args: argparse.Namespace,
    cfg: dict,
) -> tuple:
    """Train the quadruple-encoded adapter.

    Returns a 3-tuple ``(model, tokenizer, metrics_dict)`` so the caller can
    hand the live, adapter-active model directly to ``probe_phase``, avoiding
    a second ``from_pretrained`` load on the 8 GiB card.

    Training config follows the proven Test 8 550-key config (rank 8, 30 epochs,
    linear LR scheduler, warmup_steps decoupled from epoch count, weight_decay
    >= 0.1 per CLAUDE.md extended-training rules).

    Epoch-level resume: if ``adapter/checkpoint-*`` dirs exist (from a prior
    run paused mid-training), the highest-numbered checkpoint is passed to
    ``train_adapter(resume_from_checkpoint=...)`` so training continues from
    that epoch rather than restarting from epoch 0.

    Early-stop callback ``_QuadRecallEarlyStop`` probes a random sample of
    quads after each probe epoch.  Bypassed when ``--no-early-stop`` is set
    or ``es_window < 1``.

    Pauses cleanly at epoch boundaries via ``TrainingHooks.on_shutdown_check``.

    Args:
        quads: Full list of quadruple dicts to train on.
        run_dir: Run directory for checkpoints, epoch_log, paused.json.
        args: Parsed CLI namespace (model, rank, num_epochs, ES args).
        cfg: Persisted run_config dict (used to recover ES args on resume).

    Returns:
        Tuple ``(model, tokenizer, metrics_dict)`` where ``model`` has the
        ``quad_episodic`` adapter active and ready for probing.  Includes ES
        fields in the metrics dict when applicable.
    """
    from experiments.utils.gpu_guard import acquire_gpu
    from experiments.utils.quadruple_format import format_quadruple_training
    from experiments.utils.test_harness import (
        add_model_args,
        get_benchmark_models,
        load_model_and_config,
    )
    from paramem.models.loader import create_adapter
    from paramem.training.early_stop import EarlyStopPolicy
    from paramem.training.trainer import TrainingHooks, train_adapter
    from paramem.utils.config import AdapterConfig, TrainingConfig

    # Recover ES parameters from persisted config when called via --resume
    # (which does not re-specify ES flags on the CLI).
    es_from_epoch: int = cfg.get("es_from_epoch", args.es_from_epoch)
    es_window: int = cfg.get("es_window", args.es_window)
    es_probe_every: int = cfg.get("es_probe_every", args.es_probe_every)
    es_probe_sample: int = cfg.get("es_probe_sample", args.es_probe_sample)
    no_early_stop: bool = cfg.get("no_early_stop", args.no_early_stop)

    adapter_dir = run_dir / "adapter"

    # Epoch-level resume: find the highest existing checkpoint.
    resume_ckpt = _find_highest_checkpoint(adapter_dir)
    if resume_ckpt is not None:
        logger.info(
            "Epoch-level resume: found checkpoint %s — continuing from there.",
            resume_ckpt.name,
        )
    else:
        logger.info("No prior checkpoint found — starting training from epoch 0.")

    with acquire_gpu():
        # Resolve benchmark model.
        mp = argparse.ArgumentParser()
        add_model_args(mp)
        bench_args = mp.parse_args(["--model", args.model])
        bench_name, bench_config = list(get_benchmark_models(bench_args))[0]
        logger.info("Loading benchmark model: %s", bench_name)
        model, tokenizer = load_model_and_config(bench_config)

        adapter_config = AdapterConfig(
            rank=args.rank,
            alpha=args.rank * 2,  # alpha = 2× rank per CLAUDE.md
            learning_rate=1e-4,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            dropout=0.0,
        )
        model = create_adapter(model, adapter_config, "quad_episodic")

        examples = format_quadruple_training(quads, tokenizer, max_length=1024)
        dataset = IndexedDataset(examples)
        logger.info(
            "Training dataset: %d examples (1 keyed-recall per quad), n_keys=%d",
            len(dataset),
            len(quads),
        )

        # Training config: mirrors Test 8's proven 550-key config.
        # - Linear LR scheduler with fixed warmup_steps (NOT warmup_ratio) so
        #   the warmup budget is decoupled from num_train_epochs (CLAUDE.md
        #   extended-training rule).
        # - warmup_ratio=0.0 explicitly set to prevent TrainingConfig's default
        #   0.1 from leaking when warmup_steps > 0.
        # - save_strategy="epoch" + save_total_limit=2 so checkpoints exist
        #   for epoch-level resume (test8 uses save_strategy="no" because it
        #   never pauses mid-training).
        # - weight_decay=0.1 per CLAUDE.md extended-training rule.
        training_config = TrainingConfig(
            batch_size=1,
            gradient_accumulation_steps=2,
            max_seq_length=1024,
            num_epochs=args.num_epochs,
            warmup_steps=30,  # Fixed steps, not ratio — per CLAUDE.md rule
            warmup_ratio=0.0,  # Explicitly zero to suppress default 0.1
            lr_scheduler_type="linear",
            weight_decay=0.1,  # >= 0.1 per CLAUDE.md extended-training rule
            gradient_checkpointing=True,
            max_grad_norm=1.0,
            seed=42,
            save_strategy="epoch",
            save_total_limit=2,  # Keep 2 checkpoints for resume + rollback
        )

        # Early-stop callback.
        cb: Optional[_QuadRecallEarlyStop] = None
        callbacks_extra: list = []
        if not no_early_stop and es_window >= 1:
            policy = EarlyStopPolicy(
                probe_from_epoch=es_from_epoch,
                signal_from_epoch=es_from_epoch,
                window=es_window,
                probe_every_n_epochs=es_probe_every,
            )
            cb = _QuadRecallEarlyStop(
                model=model,
                tokenizer=tokenizer,
                quads=quads,
                policy=policy,
                epoch_log_path=run_dir / "epoch_log.json",
                pause_file=PAUSE_FILE,
                sample_size=es_probe_sample,
                seed=42,
            )
            callbacks_extra = [cb._hf_callback]
            logger.info(
                "Early-stop enabled: probe_from=%d signal_from=%d window=%d "
                "probe_every=%d sample=%d",
                es_from_epoch,
                es_from_epoch,
                es_window,
                es_probe_every,
                es_probe_sample,
            )
        else:
            logger.info("Early-stop disabled.")

        hooks = TrainingHooks(on_shutdown_check=lambda: PAUSE_FILE.exists())

        logger.info(
            "Training for up to %d epochs (rank=%d, hooks=pause-aware, resume=%s)",
            args.num_epochs,
            args.rank,
            resume_ckpt.name if resume_ckpt is not None else "none",
        )
        t0 = time.time()
        metrics = train_adapter(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            adapter_name="quad_episodic",
            training_config=training_config,
            adapter_config=adapter_config,
            output_dir=adapter_dir,
            run_name="quad-adapter-train",
            hooks=hooks,
            callbacks_extra=callbacks_extra if callbacks_extra else None,
            resume_from_checkpoint=resume_ckpt,
        )
        elapsed = time.time() - t0

    # If pause fired mid-training, do NOT write train_done.json.
    # Find the latest checkpoint path for the paused.json payload.
    if PAUSE_FILE.exists():
        latest_ckpt = _find_highest_checkpoint(adapter_dir)
        write_paused_marker(
            run_dir,
            "during-train",
            latest_checkpoint=str(latest_ckpt) if latest_ckpt is not None else None,
        )
        raise SystemExit("Training paused mid-train")

    # Gather ES metrics from callback.
    first_perfect = cb.first_perfect_epoch if cb is not None else None
    stable_perfect = cb.stable_perfect_epoch if cb is not None else None
    stop_epoch = cb.stop_epoch if cb is not None else None
    n_epochs_run = stop_epoch if stop_epoch is not None else args.num_epochs

    result = {
        "train_seconds": round(elapsed, 1),
        "train_loss": round(metrics.get("train_loss", float("nan")), 4),
        "n_examples": len(dataset),
        "n_epochs_requested": args.num_epochs,
        "n_epochs_run": n_epochs_run,
        "n_keys": len(quads),
        "rank": args.rank,
        # Early-stop fields (None when ES was disabled or did not fire).
        "first_perfect_epoch": first_perfect,
        "stable_perfect_epoch": stable_perfect,
        "stop_epoch": stop_epoch,
        "es_from_epoch": es_from_epoch,
        "es_window": es_window,
    }

    # Return the live model+tokenizer alongside the metrics dict so the caller
    # can hand them directly to probe_phase without a second from_pretrained.
    # The acquire_gpu() context has already exited — the GPU lock is released,
    # but the weights remain loaded.  probe_phase re-acquires the lock.
    return model, tokenizer, result


def probe_phase(
    quads: list[dict],
    run_dir: Path,
    args: argparse.Namespace,
    model=None,
    tokenizer=None,
) -> dict[str, dict | None]:
    """Probe every trained key. Pause-checked and persisted every 25 keys.

    Incremental resume: if ``probe_results.json`` already exists (from a prior
    run that paused before ``probe_done.json`` was written), keys already in
    the file are skipped.  This prevents re-probing keys whose results are
    already on disk.

    Two operating modes controlled by the optional ``model`` / ``tokenizer``
    parameters:

    **Fresh-run mode** (``model is not None``): the caller (``main()``) passes
    the live model returned by ``train_phase``.  The adapter is already active
    in memory — no second ``from_pretrained`` load.  This avoids the
    ``RuntimeError: CUDA driver error: device not ready`` OOM that occurs when
    a fresh 3.7 GB Mistral is loaded while the training model's weights are
    still resident on the 8 GiB card.  The adapter is activated via
    ``switch_adapter`` (no-op if already active), then
    ``gradient_checkpointing_disable()`` and ``eval()`` are called before the
    first ``probe_quad``/``generate`` call.

    **Resume mode** (``model is None``, i.e. ``--resume`` with
    ``train_done.json`` present): loads a fresh Mistral, finds the highest
    ``checkpoint-<N>``, optionally decrypts age-wrapped weights to ``/dev/shm``,
    attaches the adapter via ``PeftModel.from_pretrained``, and cleans up the
    scratch dir in a ``finally`` block.  This path is unchanged from the
    original implementation.

    Args:
        quads: Full list of quad dicts to probe.
        run_dir: Run directory containing the adapter and result files.
        args: Parsed CLI namespace (model selection).
        model: Live PeftModel returned by ``train_phase`` (fresh-run mode).
            Pass ``None`` to trigger the disk-load path (resume mode).
        tokenizer: Tokenizer paired with ``model``.  Must be ``None`` when
            ``model`` is ``None``.

    Returns:
        Dict mapping key → probe result dict (or None on failure).
    """
    from experiments.utils.gpu_guard import acquire_gpu
    from experiments.utils.quadruple_format import probe_quad
    from paramem.models.loader import switch_adapter

    probe_results_path = run_dir / "probe_results.json"

    # --- Incremental resume: load any partial results already on disk ---
    probe_results: dict[str, dict | None] = {}
    if probe_results_path.exists():
        try:
            existing = json.loads(probe_results_path.read_text())
            if isinstance(existing, dict):
                probe_results = existing
                logger.info(
                    "Probe resume: loaded %d existing results from probe_results.json",
                    len(probe_results),
                )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load existing probe_results.json (%s) — starting fresh.", exc)

    # Determine which quads still need probing.
    pending_quads = [q for q in quads if q["key"] not in probe_results]
    if not pending_quads:
        logger.info("All keys already probed — nothing to do.")
        return probe_results

    logger.info("Probing %d keys (%d already done).", len(pending_quads), len(probe_results))

    with acquire_gpu():
        scratch_dir: Path | None = None
        try:
            if model is not None:
                # --- Fresh-run mode: reuse the live model from train_phase ---
                # The adapter is already loaded and active.  Ensure quad_episodic
                # is the active adapter (switch is a no-op if already set), then
                # put the model into eval mode with gradient checkpointing
                # disabled before any generate() call (CLAUDE.md rule).
                if hasattr(model, "peft_config") and "quad_episodic" in model.peft_config:
                    switch_adapter(model, "quad_episodic")
                else:
                    logger.warning(
                        "quad_episodic adapter not found in live model peft_config; "
                        "falling through to disk-load path."
                    )
                    model = None  # Fall through to disk-load branch below.

            if model is None:
                # --- Resume mode: load fresh model + attach adapter from disk ---
                from peft import PeftModel

                from experiments.utils.test_harness import (
                    add_model_args,
                    get_benchmark_models,
                    load_model_and_config,
                )

                mp = argparse.ArgumentParser()
                add_model_args(mp)
                bench_args = mp.parse_args(["--model", args.model])
                bench_name, bench_config = list(get_benchmark_models(bench_args))[0]
                logger.info("Loading benchmark model for probe: %s", bench_name)
                model, tokenizer = load_model_and_config(bench_config)

                # train_adapter() does no final root-level save: HF Trainer leaves
                # ``checkpoint-<step>/`` dirs and PEFT writes the adapter one level
                # deeper at ``checkpoint-<step>/<adapter_name>/adapter_config.json``.
                # Find every adapter_config.json and pick the one under the
                # highest-numbered checkpoint (numeric, not lexicographic).
                def _ckpt_num(p: Path) -> int:
                    for part in p.parts:
                        if part.startswith("checkpoint-"):
                            try:
                                return int(part.split("-")[-1])
                            except ValueError:
                                pass
                    return -1

                cfgs = sorted((run_dir / "adapter").rglob("adapter_config.json"), key=_ckpt_num)
                if not cfgs:
                    raise SystemExit(f"No trained adapter found under {run_dir / 'adapter'}")
                adapter_dir_probe = cfgs[-1].parent

                # When Security is ON, EncryptCheckpointCallback age-wraps every
                # checkpoint file (adapter_config.json / adapter_model.safetensors
                # included). PeftModel.from_pretrained reads them directly and chokes
                # on the age magic — mirror BackgroundTrainer's resume path: decrypt
                # into a tmpfs tempdir and load the plaintext copy from there.
                from paramem.backup.age_envelope import is_age_envelope

                if is_age_envelope(adapter_dir_probe / "adapter_config.json"):
                    from paramem.backup.checkpoint_shard import materialize_checkpoint_to_shm

                    scratch_dir = materialize_checkpoint_to_shm(adapter_dir_probe)
                    logger.info(
                        "Decrypted age-wrapped adapter %s -> %s",
                        adapter_dir_probe,
                        scratch_dir,
                    )
                    adapter_dir_probe = scratch_dir

                logger.info("Loading adapter from %s", adapter_dir_probe)
                model = PeftModel.from_pretrained(
                    model, str(adapter_dir_probe), adapter_name="quad_episodic"
                )
                switch_adapter(model, "quad_episodic")

            # Common setup before any generate() call (CLAUDE.md rule).
            model.gradient_checkpointing_disable()
            model.eval()

            t0 = time.time()
            for i, quad in enumerate(pending_quads):
                # Pause check before probing (before first key and every 25 after).
                if i % 25 == 0:
                    _check_pause(f"during-probe-at-{len(probe_results) + i}", run_dir)

                probe_results[quad["key"]] = probe_quad(model, tokenizer, quad["key"])

                # Persist every 25 keys (and on the first key as a heartbeat).
                if (i + 1) % 25 == 0 or i == 0:
                    probe_results_path.write_text(json.dumps(probe_results, indent=2))
                    logger.info(
                        "Probed %d/%d pending keys (total done: %d/%d)",
                        i + 1,
                        len(pending_quads),
                        len(probe_results),
                        len(quads),
                    )

            elapsed = time.time() - t0
            logger.info("Probing done in %.1fs (%d keys)", elapsed, len(quads))
        finally:
            if scratch_dir is not None:
                shutil.rmtree(scratch_dir, ignore_errors=True)

    # Final persist with all results.
    probe_results_path.write_text(json.dumps(probe_results, indent=2))
    return probe_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the quadruple-adapter experiment.

    Phases:
        1. Train: build and train the quadruple-encoded LoRA adapter.
        2. Probe: probe every key and compare recalled quadruples.
        3. Aggregate: compute metrics and write report.

    Each phase is gated by a ``*_done.json`` marker.  A pause at any
    phase boundary is honoured by writing ``paused.json`` and exiting.
    On ``--resume``, completed phases are skipped.
    """
    from experiments.utils.test_harness import load_test_env

    load_test_env()

    args = parse_args()

    # --- Snapshot existence check (before any GPU work) ---
    snapshot_path = args.graph_snapshot
    if not snapshot_path.exists():
        raise SystemExit(
            f"Graph snapshot not found: {snapshot_path}\n"
            f"Run: tresume lme  (or: python experiments/lme_graph_builder.py --target-keys N)\n"
            f"Then: python experiments/quadruple_adapter.py --n-keys N"
        )

    triples_available = load_unique_triples(snapshot_path)
    n_available = len(triples_available)
    if args.n_keys is None and not args.resume:
        # FRESH run only: freeze the resolved budget into run_config now, so a
        # later bare `--resume` is deterministic even if lme_graph_builder grew
        # the snapshot in the meantime. On `--resume` we must NOT touch
        # args.n_keys — leaving it None means "continue with the saved n_keys";
        # resolving it to n_available here would look like an extension request
        # to load_or_write_run_config and trigger an unintended full retrain.
        args.n_keys = n_available
    requested_n = args.n_keys

    if requested_n is not None and n_available < requested_n:
        raise SystemExit(
            f"Graph snapshot has only {n_available} unique triples, "
            f"but --n-keys={requested_n} was requested.\n"
            f"Run: tresume lme --target-keys {requested_n}  to extend the graph first.\n"
            f"Then retry: python experiments/quadruple_adapter.py --n-keys {requested_n}"
        )

    # Resolve run dir.
    if args.resume:
        run_dir = find_latest_run_dir(args.model)
        if run_dir is None:
            raise SystemExit(
                f"--resume but no prior run dir under outputs/quad_scale/{args.model}/"
            )
        logger.info("Resuming run at %s", run_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = project_root / "outputs" / "quad_scale" / args.model / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("New run dir: %s", run_dir)

    cfg = load_or_write_run_config(run_dir, args)
    clear_paused_marker(run_dir)

    # Apply n_keys budget using the already-loaded triples.
    n_keys = cfg["n_keys"] if cfg["n_keys"] is not None else n_available
    if n_keys > n_available:
        logger.warning(
            "n_keys=%d exceeds source-graph triple count=%d; capping to %d",
            n_keys,
            n_available,
            n_available,
        )
        n_keys = n_available
    triples = triples_available[:n_keys]

    from experiments.utils.quadruple_format import assign_quad_keys

    quads = assign_quad_keys(triples)
    (run_dir / "quads.json").write_text(json.dumps(quads, indent=2))
    logger.info("Working with %d quadruples", len(quads))

    # Phase 1: train. On a fresh run train_phase returns the live, adapter-
    # active model so probe_phase can reuse it (no second from_pretrained on
    # the 8 GiB card — mirrors test8/test13/test14, which probe the same model
    # they just trained). On --resume-with-train_done.json the train phase is
    # skipped and trained_model stays None → probe_phase loads from disk.
    trained_model = None
    trained_tokenizer = None
    _check_pause("before-train", run_dir)
    if marker_exists(run_dir, "train"):
        logger.info("train_done.json present — skipping train phase")
        train_meta = json.loads((run_dir / "train_done.json").read_text())
    else:
        trained_model, trained_tokenizer, train_meta = train_phase(quads, run_dir, args, cfg)
        write_phase_done(run_dir, "train", train_meta)
        logger.info("Train phase complete: %s", train_meta)

    # Phase 2: probe.
    _check_pause("before-probe", run_dir)
    if marker_exists(run_dir, "probe"):
        logger.info("probe_done.json present — skipping probe phase")
        probe_results = json.loads((run_dir / "probe_results.json").read_text())
    else:
        probe_results = probe_phase(
            quads, run_dir, args, model=trained_model, tokenizer=trained_tokenizer
        )
        write_phase_done(run_dir, "probe", {"n_probed": len(probe_results)})

    # Phase 3: aggregate + report.
    results = evaluate(quads, probe_results)
    overall = aggregate(results)
    overall["n_keys"] = len(quads)
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps({"overall": overall}, indent=2))
    write_report(run_dir, overall, train_meta)
    logger.info(
        "Done. n_keys=%d, strict=%.1f%%, s+o=%.1f%%, report=%s",
        len(quads),
        overall.get("source_triple_recovered_rate", 0) * 100,
        overall.get("subject_object_match_rate", 0) * 100,
        run_dir / "report.md",
    )


if __name__ == "__main__":
    main()
