"""Dataset-agnostic extraction-pipeline probe.

Feeds any conversational dataset through the ParaMem consolidation pipeline
(extract → merge → QA → indexed-key training → recall smoke) and emits
identically-shaped per-session diagnostics regardless of source corpus.

Usage:
    python experiments/dataset_probe.py --dataset perltqa --limit 20
    python experiments/dataset_probe.py --dataset longmemeval --split longmemeval_oracle --limit 20
    python experiments/dataset_probe.py --dataset perltqa --no-sota --limit 5
    python experiments/dataset_probe.py --dataset perltqa --resume
    python experiments/dataset_probe.py --dataset longmemeval --model mistral --limit 5

Sections of this file follow the plan at:
  .agent/plan-dataset-probe-2026-04-16.md

Hard rules carried forward:
  - Tests 1-11 are not modified. This file is a new consumer.
  - ConsolidationLoop is used exactly as in experiments/phase3_consolidation.py.
  - Speaker-id tagging mirrors paramem/server/consolidation.py:155-158.
  - Write ordering: raw_qa → diagnostics → state (fsync at each step).
  - Smoke test uses _write_keyed_pairs from paramem/server/consolidation.py.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path.
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from experiments.utils.dataset_types import DatasetSession  # noqa: E402
from experiments.utils.gpu_guard import acquire_gpu  # noqa: E402
from experiments.utils.longmemeval_loader import LongMemEvalLoader  # noqa: E402
from experiments.utils.perltqa_loader import PerLTQALoader  # noqa: E402
from experiments.utils.test_harness import (  # noqa: E402
    BENCHMARK_MODELS,
    model_output_dir,
    setup_logging,
)
from paramem.utils.config import TrainingConfig, load_config  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

# Loader registry — add future loaders here (one entry per new dataset).
_LOADER_REGISTRY: dict[str, type] = {
    "perltqa": PerLTQALoader,
    "longmemeval": LongMemEvalLoader,
}


# ---------------------------------------------------------------------------
# GPU cooldown helper (same pattern as test11_adapter_extraction.py)
# ---------------------------------------------------------------------------


def wait_for_cooldown(target: int = 45) -> None:
    """Block until GPU temperature drops below target (°C).

    Shells out to gpu-cooldown.sh. Returns instantly if GPU is already cool.
    Falls back to a 60-second sleep if the script is unavailable.

    Args:
        target: Temperature threshold in °C (default 45, probe uses 52).
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


# ---------------------------------------------------------------------------
# Atomic write helpers
# ---------------------------------------------------------------------------


def _atomic_json_write(data: dict | list, path: Path) -> None:
    """Write JSON atomically and fsync: write to .tmp, fsync, rename, fsync dir.

    Args:
        data: JSON-serializable dict or list.
        path: Destination path. Parent directory is created if absent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        Path(tmp_str).replace(path)
        # Fsync the directory so the rename is durable.
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        try:
            os.unlink(tmp_str)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> dict | list:
    """Read and parse JSON from path.

    Raises:
        FileNotFoundError: If path does not exist.
        json.JSONDecodeError: If content is not valid JSON.
    """
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def _state_path(run_dir: Path) -> Path:
    """Return path to state.json for a run directory."""
    return run_dir / "state.json"


def _load_state(run_dir: Path) -> dict:
    """Load state.json; return empty-skeleton if absent."""
    p = _state_path(run_dir)
    if not p.exists():
        return {}
    return _read_json(p)


def _write_state(state: dict, run_dir: Path) -> None:
    """Atomically persist state.json (the commit point for each session)."""
    state["last_updated"] = datetime.utcnow().isoformat()
    _atomic_json_write(state, _state_path(run_dir))


def _find_resume_dir(base_dir: Path, model: str, dataset: str) -> Path | None:
    """Return the newest incomplete run dir for {model} matching {dataset}.

    Args:
        base_dir: outputs/dataset_probe/
        model: Model name (e.g. "mistral").
        dataset: Dataset name (e.g. "perltqa").

    Returns:
        Path to the run dir, or None if no resumable run exists.
    """
    model_dir = base_dir / dataset / model
    if not model_dir.exists():
        return None

    candidates = sorted(
        [d for d in model_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    for candidate in candidates:
        state_file = candidate / "state.json"
        if not state_file.exists():
            continue
        try:
            state = _read_json(state_file)
        except (json.JSONDecodeError, OSError):
            continue
        if state.get("completed"):
            continue
        if state.get("training_started"):
            logger.warning(
                "Resume skipping %s: training already started (not safely resumable). "
                "Inspect the run or start fresh.",
                candidate,
            )
            continue
        if state.get("dataset") == dataset:
            return candidate
    return None


def _validate_resume_accumulator(
    run_dir: Path,
    processed_ids: list[str],
) -> tuple[list[dict], list[dict]]:
    """Reload accumulated QA from per-session raw_qa.json files.

    Validates that every processed session has a loadable cache file.
    Errors out immediately if any file is missing or corrupt (corruption
    means state.json is ahead of the actual data).

    Args:
        run_dir: Run directory containing the diagnostics/ sub-dir.
        processed_ids: Session IDs listed in state.processed_session_ids.

    Returns:
        (all_episodic_qa, all_procedural_rels) re-accumulated from disk.

    Raises:
        RuntimeError: If any raw_qa.json is missing or unreadable.
    """
    all_episodic_qa: list[dict] = []
    all_procedural_rels: list[dict] = []

    for sid in processed_ids:
        raw_qa_path = run_dir / "diagnostics" / f"{sid}.raw_qa.json"
        if not raw_qa_path.exists():
            raise RuntimeError(
                f"Run {run_dir} is corrupted: state.json references "
                f"session_id={sid!r} but diagnostics/{sid}.raw_qa.json is "
                "missing. Start a new run."
            )
        try:
            data = _read_json(raw_qa_path)
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(
                f"Run {run_dir} is corrupted: failed to parse "
                f"diagnostics/{sid}.raw_qa.json — {exc}. Start a new run."
            ) from exc

        all_episodic_qa.extend(data.get("episodic_qa", []))
        all_procedural_rels.extend(data.get("procedural_rels", []))

    logger.info(
        "Resume: reloaded %d episodic + %d procedural QA from %d sessions",
        len(all_episodic_qa),
        len(all_procedural_rels),
        len(processed_ids),
    )
    return all_episodic_qa, all_procedural_rels


# ---------------------------------------------------------------------------
# Per-session diagnostics
# ---------------------------------------------------------------------------


def _build_session_diagnostics(
    session: DatasetSession,
    session_graph,
    nodes_before: int,
    edges_before: int,
    nodes_after: int,
    edges_after: int,
    episodic_qa: list[dict],
    procedural_rels: list[dict],
    elapsed_seconds: float,
) -> dict:
    """Build the per-session diagnostics dict from extractor output.

    Schema is identical across datasets (plan Section 7). All missing
    extractor keys default to 0 / null so downstream groupby is safe.

    Args:
        session: The DatasetSession that was just processed.
        session_graph: The ExtractionGraph returned by extract_session's
            internal extract_graph call, accessed via loop internals.
            Passed in as a pre-captured snapshot; see population sources below.
        nodes_before / edges_before: Graph size before the session merge.
        nodes_after / edges_after: Graph size after the session merge.
        episodic_qa: Tagged QA pairs returned by extract_session.
        procedural_rels: Tagged procedural relations returned by extract_session.
        elapsed_seconds: Wall-clock time for extract_session call.

    Returns:
        Dict matching the schema in plan Section 7.
    """
    diag = session_graph.diagnostics if session_graph is not None else {}

    # Normalize drop-counters: extractor stores residual/ungrounded as
    # list[fact-dict] for downstream debugging, but the rest as int. We
    # record counts here so downstream groupby/sum is safe.
    def _as_count(value) -> int:
        if value is None:
            return 0
        if isinstance(value, (list, tuple, set)):
            return len(value)
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    drops = {
        "residual_dropped_facts": _as_count(diag.get("residual_dropped_facts")),
        "ungrounded_dropped_facts": _as_count(diag.get("ungrounded_dropped_facts")),
        "plausibility_dropped": _as_count(diag.get("plausibility_dropped")),
        "mapping_ambiguous_dropped": _as_count(diag.get("mapping_ambiguous_dropped")),
        "residual_leaked_triples_dropped": _as_count(diag.get("residual_leaked_triples_dropped")),
    }

    # Compute raw_fact_count as sum of surviving relations + all drops.
    post_plausibility_count = len(episodic_qa)
    raw_fact_count = post_plausibility_count + sum(drops.values())

    # Entity and relation type distributions from the session graph.
    entity_type_dist: dict[str, int] = {}
    relation_type_dist: dict[str, int] = {}
    if session_graph is not None:
        for entity in session_graph.entities:
            etype = getattr(entity, "entity_type", "unknown") or "unknown"
            entity_type_dist[etype] = entity_type_dist.get(etype, 0) + 1
        for rel in session_graph.relations:
            rtype = getattr(rel, "relation_type", "unknown") or "unknown"
            relation_type_dist[rtype] = relation_type_dist.get(rtype, 0) + 1

    return {
        "session_id": session.session_id,
        "dataset": session.metadata.get("dataset", "unknown"),
        "speaker_id": session.speaker_id,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "extraction": {
            "raw_fact_count": raw_fact_count,
            "post_plausibility_count": post_plausibility_count,
            "drops": drops,
            "plausibility_judge_actual": diag.get("plausibility_judge_actual"),
            "fallback_path": diag.get("fallback_path"),
            "anonymize": diag.get("anonymize", "not_run"),
        },
        "graph": {
            "entities_added": nodes_after - nodes_before,
            "relations_added": edges_after - edges_before,
            "entity_type_distribution": entity_type_dist,
            "relation_type_distribution": relation_type_dist,
            "cumulative_nodes": nodes_after,
            "cumulative_edges": edges_after,
        },
        "qa": {
            "episodic_qa_count": len(episodic_qa),
            "procedural_rel_count": len(procedural_rels),
        },
        "errors": [],
    }


# ---------------------------------------------------------------------------
# Smoke test shim (plan Section 9 Q2)
# ---------------------------------------------------------------------------


def _prepare_smoke_shim(run_dir: Path, loop) -> Path:
    """Build the {run_dir}/_smoke_shim layout while loop objects are alive.

    ConsolidationLoop's _save_adapters does not write keyed_pairs.json.
    This helper materializes keyed_pairs + simhash registry + adapter
    symlink from the in-memory loop state so a later call to
    smoke_test_adapter (which loads a fresh base model) can run on disk
    alone — the caller is expected to free GPU memory between the two.

    Args:
        run_dir: The probe run directory (contains episodic/ adapter subdir).
        loop: Live ConsolidationLoop after train_adapters.

    Returns:
        Path to the prepared shim directory.
    """
    from paramem.server.consolidation import _write_keyed_pairs

    shim_dir = run_dir / "_smoke_shim"
    shim_dir.mkdir(parents=True, exist_ok=True)

    # (a) keyed_pairs.json
    _write_keyed_pairs(loop.indexed_key_qa, loop.episodic_simhash, shim_dir / "keyed_pairs.json")

    # (b) simhash_registry.json
    src_registry = run_dir / "simhash_registry_episodic.json"
    dst_registry = shim_dir / "simhash_registry.json"
    if src_registry.exists():
        shutil.copy2(src_registry, dst_registry)
    else:
        _atomic_json_write(loop.episodic_simhash, dst_registry)

    # (c) adapter symlink at {shim}/adapter/episodic -> {run_dir}/episodic
    adapter_parent = shim_dir / "adapter"
    adapter_parent.mkdir(parents=True, exist_ok=True)
    adapter_link = adapter_parent / "episodic"
    if not adapter_link.exists():
        episodic_adapter_dir = run_dir / "episodic"
        if episodic_adapter_dir.exists():
            adapter_link.symlink_to(episodic_adapter_dir.resolve())
        else:
            logger.warning(
                "Smoke shim: episodic adapter directory not found at %s",
                episodic_adapter_dir,
            )
    return shim_dir


def _run_smoke_on_shim(shim_dir: Path, model: str) -> dict:
    """Run smoke_test_adapter against a prepared shim directory.

    Expects GPU memory from the main loop to already be freed — this call
    loads a fresh base model and would OOM on 8GB VRAM otherwise.
    """
    from experiments.utils.test_harness import smoke_test_adapter

    adapter_link = shim_dir / "adapter" / "episodic"
    if not adapter_link.exists():
        return {"error": "adapter_dir_missing", "rate": 0.0}

    try:
        return smoke_test_adapter(shim_dir, model)
    except Exception as exc:
        logger.error("Smoke test failed: %s", exc)
        return {"error": str(exc), "rate": 0.0}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the dataset probe.

    Returns:
        Parsed Namespace with dataset, limit, resume, no_sota, model,
        character (perltqa-only), and split (longmemeval-only).
    """
    parser = argparse.ArgumentParser(
        description="Dataset-agnostic ParaMem extraction-pipeline probe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python experiments/dataset_probe.py --dataset perltqa --limit 20\n"
            "  python experiments/dataset_probe.py --dataset longmemeval --limit 20\n"
            "  python experiments/dataset_probe.py --dataset perltqa --no-sota --limit 5\n"
            "  python experiments/dataset_probe.py --dataset perltqa --resume\n"
        ),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(_LOADER_REGISTRY.keys()),
        help="Dataset to probe.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum sessions to process. Default: all. "
            "For first-pass cost control, use --limit 20."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the newest incomplete run for this model/dataset combination.",
    )
    parser.add_argument(
        "--no-sota",
        action="store_true",
        help=(
            "Disable SOTA cloud enrichment (noise_filter='' + plausibility_judge=off). "
            "Zeros Anthropic API costs. Useful for debugging extraction prompts."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistral",
        choices=list(BENCHMARK_MODELS.keys()),
        help="Model to use (default: mistral).",
    )
    parser.add_argument(
        "--character",
        type=str,
        default=None,
        help="PerLTQA character name (perltqa only). Default: auto-select by dialogue count.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="longmemeval_oracle",
        choices=["longmemeval_oracle", "longmemeval_s_cleaned", "longmemeval_m_cleaned"],
        help="LongMemEval split (longmemeval only). Default: longmemeval_oracle.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Training epochs for indexed key training (default: 20).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the dataset-agnostic probing harness."""
    args = parse_args()

    # --- 1. Resolve run directory ---
    base_dir = _PROJECT_ROOT / "outputs" / "dataset_probe"

    if args.resume:
        run_dir = _find_resume_dir(base_dir, args.model, args.dataset)
        if run_dir is None:
            logger.error(
                "No resumable run found for --dataset=%s --model=%s. "
                "Start a new run (omit --resume).",
                args.dataset,
                args.model,
            )
            sys.exit(1)
        state = _load_state(run_dir)
        # Validate dataset matches.
        if state.get("dataset") != args.dataset:
            logger.error(
                "Resume conflict: run %s was started with --dataset=%s, "
                "but you passed --dataset=%s. Start a new run instead, or "
                "re-run with --dataset=%s.",
                run_dir,
                state.get("dataset"),
                args.dataset,
                state.get("dataset"),
            )
            sys.exit(1)
        logger.info("Resuming run: %s", run_dir)
    else:
        run_dir = model_output_dir(base_dir=base_dir / args.dataset, model_name=args.model)
        run_dir.mkdir(parents=True, exist_ok=True)
        state = {}

    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. Initialize / validate state ---
    processed_ids: list[str] = state.get("processed_session_ids", [])
    if not state:
        state = {
            "dataset": args.dataset,
            "model": args.model,
            "limit": args.limit,
            "processed_session_ids": [],
            "completed": False,
            "training_started": False,
            "started_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat(),
            "args_snapshot": vars(args),
        }
        _write_state(state, run_dir)
        logger.info("New run: %s", run_dir)

    # --- 3. Reload accumulator from disk if resuming ---
    all_episodic_qa: list[dict] = []
    all_procedural_rels: list[dict] = []
    if processed_ids:
        all_episodic_qa, all_procedural_rels = _validate_resume_accumulator(run_dir, processed_ids)

    # --- 4. Extraction flags ---
    if args.no_sota:
        noise_filter = ""
        plausibility_judge = "off"
        logger.info("--no-sota: noise_filter='' plausibility_judge='off'")
    else:
        noise_filter = "anthropic"
        plausibility_judge = "auto"

    # --- 5. Acquire GPU + model ---
    with acquire_gpu():
        logger.info("GPU acquired")
        wait_for_cooldown(52)

        # Load model and config (pattern from test_harness.load_model_and_config).
        from paramem.models.loader import load_base_model

        config = load_config()
        model_cfg = BENCHMARK_MODELS[args.model]
        config.model = model_cfg
        logger.info("Loading base model: %s", config.model.model_id)
        model, tokenizer = load_base_model(config.model)

        # --- 6. Build ConsolidationLoop ---
        # Pattern copied from experiments/phase3_consolidation.py:143-168.
        from paramem.training.consolidation import ConsolidationLoop

        episodic_config = config.adapters.get("episodic")
        if episodic_config is None:
            raise ValueError(
                "Episodic adapter config is required in config. Check configs/default.yaml."
            )
        # Probe is episodic-only: never trains or promotes to semantic. Reuse
        # episodic_config in the semantic slot so staging-compat validation
        # passes (default.yaml has rank 8 vs 24 mismatch).
        semantic_config = episodic_config

        consolidation_training = TrainingConfig(
            batch_size=config.training.batch_size,
            gradient_accumulation_steps=2,
            max_seq_length=config.training.max_seq_length,
            num_epochs=args.num_epochs,
            warmup_ratio=config.training.warmup_ratio,
            weight_decay=config.training.weight_decay,
            gradient_checkpointing=config.training.gradient_checkpointing,
            max_grad_norm=config.training.max_grad_norm,
            seed=config.training.seed,
        )

        # Force indexed-key replay on so train_adapters exercises the full
        # keyed training path (default.yaml leaves it off). The probe's
        # whole point is to dry-run extraction → merge → QA → train → smoke.
        consolidation_cfg = replace(
            config.consolidation,
            indexed_key_replay_enabled=True,
        )

        loop = ConsolidationLoop(
            model=model,
            tokenizer=tokenizer,
            consolidation_config=consolidation_cfg,
            training_config=consolidation_training,
            episodic_adapter_config=episodic_config,
            semantic_adapter_config=semantic_config,
            procedural_adapter_config=None,
            output_dir=run_dir,
            persist_graph=False,
            save_cycle_snapshots=False,
            prompts_dir=None,
        )

        # --- 7. Bind loader with dataset-specific kwargs ---
        loader_cls = _LOADER_REGISTRY[args.dataset]
        if args.dataset == "perltqa":
            loader = loader_cls()
            loader_kwargs = {"character": args.character}
        elif args.dataset == "longmemeval":
            loader = loader_cls(split=args.split)
            loader_kwargs = {}
        else:
            loader = loader_cls()
            loader_kwargs = {}

        # --- 8. Per-session extraction loop ---
        session_count = 0
        primary_speaker = ""

        for session in loader.iter_sessions(limit=args.limit, **loader_kwargs):
            if session.session_id in processed_ids:
                logger.info("Skipping already-processed session: %s", session.session_id)
                primary_speaker = session.speaker_id
                continue

            logger.info("Processing session: %s", session.session_id)
            wait_for_cooldown(52)

            nodes_before = loop.merger.graph.number_of_nodes()
            edges_before = loop.merger.graph.number_of_edges()

            t_start = time.time()
            try:
                episodic_qa, procedural_rels = loop.extract_session(
                    session_transcript=session.transcript,
                    session_id=session.session_id,
                    speaker_id=session.speaker_id,
                    speaker_name=session.speaker_name,
                    ha_context=None,
                    stt_correction=False,
                    ha_validation=False,
                    noise_filter=noise_filter,
                    noise_filter_model="claude-sonnet-4-6",
                    noise_filter_endpoint=None,
                    ner_check=False,
                    plausibility_judge=plausibility_judge,
                    plausibility_stage="deanon",
                    verify_anonymization=True,
                )
            except Exception as exc:
                logger.error("extract_session failed for %s: %s", session.session_id, exc)
                # Write a minimal diagnostics file recording the error,
                # then mark the session as processed so it is not retried.
                error_diag = {
                    "session_id": session.session_id,
                    "dataset": session.metadata.get("dataset", "unknown"),
                    "speaker_id": session.speaker_id,
                    "elapsed_seconds": round(time.time() - t_start, 2),
                    "extraction": {
                        "raw_fact_count": 0,
                        "post_plausibility_count": 0,
                        "drops": {
                            "residual_dropped_facts": 0,
                            "ungrounded_dropped_facts": 0,
                            "plausibility_dropped": 0,
                            "mapping_ambiguous_dropped": 0,
                            "residual_leaked_triples_dropped": 0,
                        },
                        "plausibility_judge_actual": None,
                        "fallback_path": None,
                        "anonymize": "not_run",
                    },
                    "graph": {
                        "entities_added": 0,
                        "relations_added": 0,
                        "entity_type_distribution": {},
                        "relation_type_distribution": {},
                        "cumulative_nodes": loop.merger.graph.number_of_nodes(),
                        "cumulative_edges": loop.merger.graph.number_of_edges(),
                    },
                    "qa": {"episodic_qa_count": 0, "procedural_rel_count": 0},
                    "errors": [str(exc)],
                }
                episodic_qa, procedural_rels = [], []
                _commit_session(run_dir, session, episodic_qa, procedural_rels, error_diag, state)
                primary_speaker = session.speaker_id
                session_count += 1
                continue

            elapsed = time.time() - t_start

            # Speaker-tag each QA pair immediately (mirrors server/consolidation.py:155-158).
            for qa in episodic_qa:
                qa["speaker_id"] = session.speaker_id
            for rel in procedural_rels:
                rel["speaker_id"] = session.speaker_id

            nodes_after = loop.merger.graph.number_of_nodes()
            edges_after = loop.merger.graph.number_of_edges()

            # Extract last session_graph from loop for diagnostics.
            # ConsolidationLoop does not expose session_graph directly;
            # ConsolidationLoop exposes the just-extracted SessionGraph via
            # last_session_graph for probe consumers. The helper reads
            # diagnostics, entity types, and relation types from it.
            diag_data = _build_session_diagnostics(
                session=session,
                session_graph=loop.last_session_graph,
                nodes_before=nodes_before,
                edges_before=edges_before,
                nodes_after=nodes_after,
                edges_after=edges_after,
                episodic_qa=episodic_qa,
                procedural_rels=procedural_rels,
                elapsed_seconds=elapsed,
            )

            # Commit session: raw_qa → diagnostics → state (fsync at each step).
            _commit_session(run_dir, session, episodic_qa, procedural_rels, diag_data, state)

            all_episodic_qa.extend(episodic_qa)
            all_procedural_rels.extend(procedural_rels)
            primary_speaker = session.speaker_id
            session_count += 1

            logger.info(
                "Session %s done: %d episodic, %d procedural QA. "
                "Graph: +%d nodes, +%d edges (total: %d nodes, %d edges). "
                "Elapsed: %.1fs",
                session.session_id,
                len(episodic_qa),
                len(procedural_rels),
                nodes_after - nodes_before,
                edges_after - edges_before,
                nodes_after,
                edges_after,
                elapsed,
            )

        logger.info(
            "Extraction complete: %d sessions processed, %d episodic QA, %d procedural rels",
            session_count,
            len(all_episodic_qa),
            len(all_procedural_rels),
        )

        # --- 9. Train adapters (extract-all-then-train-once discipline) ---
        smoke_result: dict = {}
        train_result: dict = {}

        if all_episodic_qa or all_procedural_rels:
            state["training_started"] = True
            _write_state(state, run_dir)

            logger.info(
                "Training on %d episodic + %d procedural QA pairs",
                len(all_episodic_qa),
                len(all_procedural_rels),
            )
            try:
                train_result = loop.train_adapters(
                    all_episodic_qa,
                    all_procedural_rels,
                    speaker_id=primary_speaker,
                )
            except Exception as exc:
                logger.error("train_adapters failed: %s", exc)
                train_result = {"error": str(exc)}

            # --- 10. Smoke-test recall ---
            if not train_result.get("error"):
                logger.info("Running recall smoke test...")
                # Build smoke-shim artifacts (keyed_pairs, registry, adapter
                # symlink) while the loop's in-memory objects are still alive,
                # then free the GPU so smoke_test_adapter can load a fresh model.
                shim_dir = _prepare_smoke_shim(run_dir, loop)
                import gc as _gc

                import torch as _torch

                del loop
                del model
                del tokenizer
                _gc.collect()
                _torch.cuda.empty_cache()
                _torch.cuda.synchronize()

                smoke_result = _run_smoke_on_shim(shim_dir, args.model)
                logger.info(
                    "Smoke test: %.1f%% recall (%s/%s keys)",
                    smoke_result.get("rate", 0.0) * 100,
                    smoke_result.get("exact_count", "?"),
                    smoke_result.get("total", "?"),
                )
        else:
            logger.info("No QA pairs extracted — skipping training and smoke test")

        # --- 11. Write summary and mark complete ---
        processed_ids_final = state.get("processed_session_ids", [])
        summary = {
            "dataset": args.dataset,
            "model": args.model,
            "limit": args.limit,
            "sessions_processed": session_count,
            "total_episodic_qa": len(all_episodic_qa),
            "total_procedural_rels": len(all_procedural_rels),
            "train_result": {k: v for k, v in train_result.items() if k != "loop"},
            "smoke_result": smoke_result,
            "processed_session_ids": processed_ids_final,
            "args_snapshot": vars(args),
            "completed_at": datetime.utcnow().isoformat(),
        }
        _atomic_json_write(summary, run_dir / "summary.json")
        logger.info("Summary written to %s", run_dir / "summary.json")

        state["completed"] = True
        _write_state(state, run_dir)
        logger.info("Run complete: %s", run_dir)


def _commit_session(
    run_dir: Path,
    session: DatasetSession,
    episodic_qa: list[dict],
    procedural_rels: list[dict],
    diag_data: dict,
    state: dict,
) -> None:
    """Commit one session to disk in the required fsync order.

    Write ordering (plan Section 8):
      1. raw_qa.json   (accumulator input for resume)
      2. diagnostics.json  (health metrics)
      3. state.json    (commit point — session is "done" only after this)

    Args:
        run_dir: Run directory.
        session: The session just processed.
        episodic_qa: Tagged episodic QA pairs.
        procedural_rels: Tagged procedural relations.
        diag_data: Per-session diagnostics dict.
        state: Mutable state dict (updated in place, then written).
    """
    sid = session.session_id
    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: raw_qa.json
    raw_qa = {"episodic_qa": episodic_qa, "procedural_rels": procedural_rels}
    _atomic_json_write(raw_qa, diag_dir / f"{sid}.raw_qa.json")

    # Step 2: diagnostics.json
    _atomic_json_write(diag_data, diag_dir / f"{sid}.json")

    # Step 3: state.json (commit point)
    if sid not in state.get("processed_session_ids", []):
        state.setdefault("processed_session_ids", []).append(sid)
    _write_state(state, run_dir)


if __name__ == "__main__":
    main()
