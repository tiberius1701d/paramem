"""LongMemEval → graph_snapshot.json builder (incremental, pausable, resumable).

Feeds LongMemEval sessions through the ParaMem production extraction pipeline
and accumulates the results into a single canonical ``graph_snapshot.json``
under the output directory.  The snapshot can be used as the ``--graph-snapshot``
source for ``experiments/quadruple_adapter.py``.

Design:
- One model load; one ``ConsolidationLoop`` for extraction only (no training).
- Merger dedup is on ``(subject, predicate, object)`` — re-extracting a
  session that is already in ``build_state.json["sessions_done"]`` is skipped
  at the session level (fast), but if re-extraction does happen the merger is
  idempotent (safe).
- On ``--resume``:  load the existing ``graph_snapshot.json`` into the merger
  via ``merger.load_graph()`` and skip any session whose id appears in
  ``build_state.json["sessions_done"]``.  If ``graph_done.json`` is present
  and ``--target-keys`` has not been raised, exit immediately.
- Output dir is a SINGLE canonical directory (default
  ``outputs/lme_graph/``), not timestamped — this is a growing artifact.
- SOTA enrichment is disabled by default (``--no-sota`` is the default
  posture for graph building).  Pass ``--with-sota`` to enable cloud
  enrichment (incurs Anthropic API cost).

CLI:
    python experiments/lme_graph_builder.py --target-keys 550 --resume
    python experiments/lme_graph_builder.py --lme-split longmemeval_oracle

Register in training-control.sh as slot ``lme``.

Outputs:
    <output>/graph_snapshot.json   — NetworkX node-link JSON (growing)
    <output>/build_state.json      — progress + session ids
    <output>/graph_done.json       — written on clean completion
    <output>/paused.json           — written on clean pause
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("lme_graph_builder")

PAUSE_FILE = Path.home() / ".training_pause"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "lme_graph"

# Sentinel: not yet computed (avoids triggering load_unique_triples before snapshot exists)
_TRIPLE_COUNT_UNKNOWN = -1


# ---------------------------------------------------------------------------
# GPU cooldown helper (mirrors dataset_probe.py pattern exactly)
# ---------------------------------------------------------------------------


def wait_for_cooldown(target: int = 52) -> None:
    """Block until GPU temperature drops below target (°C).

    Shells out to gpu-cooldown.sh.  Returns instantly if GPU is already cool.
    Falls back to a 60-second sleep if the script is unavailable.

    Args:
        target: Temperature threshold in °C (default 52).
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
# Pause helper
# ---------------------------------------------------------------------------


def _check_pause_and_exit(
    output_dir: Path,
    session_id: str,
    n_sessions_extracted: int,
    n_unique_triples: int,
) -> None:
    """If pause file is set: write paused.json and exit cleanly.

    The caller is responsible for persisting the graph and build_state before
    calling this function (so they are consistent on disk).

    Args:
        output_dir: The canonical output directory.
        session_id: ID of the last completed session (or "none" before start).
        n_sessions_extracted: Count of sessions extracted so far.
        n_unique_triples: Count of unique triples at the last persist point.
    """
    if PAUSE_FILE.exists():
        logger.warning("Pause file detected — halting cleanly after session %s.", session_id)
        _safe_write_json(
            output_dir / "paused.json",
            {
                "stopped_after_session": session_id,
                "n_sessions_extracted": n_sessions_extracted,
                "n_unique_triples": n_unique_triples,
                "timestamp": int(time.time()),
            },
        )
        raise SystemExit("paused")


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _safe_write_json(path: Path, data: object) -> None:
    """Write data as JSON to path (creates parent dirs, swallows write errors).

    Args:
        path: Destination path.
        data: JSON-serializable object.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
    except OSError as exc:
        logger.warning("JSON write failed (%s): %s", path, exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed Namespace with all LME graph builder options.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lme-split",
        default="longmemeval_oracle",
        dest="lme_split",
        choices=["longmemeval_oracle", "longmemeval_s_cleaned", "longmemeval_m_cleaned"],
        help="LongMemEval HF split to use (default: longmemeval_oracle).",
    )
    parser.add_argument(
        "--lme-seed",
        type=int,
        default=42,
        dest="lme_seed",
        help="Random seed for deterministic session order (default: 42).",
    )
    parser.add_argument(
        "--target-keys",
        type=int,
        default=None,
        dest="target_keys",
        help=(
            "Stop after accumulating at least this many unique triples. "
            "None (default) = run until all sessions are exhausted."
        ),
    )
    parser.add_argument(
        "--persist-every",
        type=int,
        default=10,
        dest="persist_every",
        help=("Write graph_snapshot.json and build_state.json every N sessions (default: 10)."),
    )
    parser.add_argument(
        "--model",
        default="mistral",
        choices=["mistral"],
        help="Model to use for extraction (default: mistral).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing graph_snapshot.json and build_state.json.",
    )
    parser.add_argument(
        "--with-sota",
        action="store_true",
        dest="with_sota",
        help=(
            "Enable SOTA cloud enrichment (noise_filter=anthropic + "
            "plausibility_judge=auto). Disabled by default to avoid API cost."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the LongMemEval graph builder.

    Loads Mistral once, builds a ConsolidationLoop via the canonical server
    factory (exactly as dataset_probe.py does), iterates LongMemEval sessions,
    and accumulates extracted triples into graph_snapshot.json.

    Pauses cleanly at every session boundary when ~/.training_pause is set.
    Writes graph_done.json on clean completion.
    """
    from experiments.utils.test_harness import load_test_env

    load_test_env()

    args = parse_args()

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = output_dir / "graph_snapshot.json"
    state_path = output_dir / "build_state.json"
    done_path = output_dir / "graph_done.json"
    paused_path = output_dir / "paused.json"

    # Extraction flags: default is no-SOTA to avoid API cost; --with-sota enables it.
    extraction_noise_filter = "anthropic" if args.with_sota else ""
    plausibility_judge = "auto" if args.with_sota else "off"

    # --- Resume / done check ---
    sessions_done: list[str] = []
    if args.resume:
        done_data = json.loads(done_path.read_text()) if done_path.exists() else None
        # When resuming without an explicit --target-keys (e.g. plain
        # `tresume lme`), adopt the target the prior run recorded — otherwise
        # the loop's target check never fires and we silently extract all
        # remaining sessions instead of stopping at the original cap.
        if args.target_keys is None:
            if done_data is not None:
                args.target_keys = done_data.get("target_keys")
            elif state_path.exists():
                try:
                    args.target_keys = json.loads(state_path.read_text()).get("target_keys")
                except (json.JSONDecodeError, OSError):
                    pass

        # If graph_done.json exists and target-keys not raised, nothing to do.
        if done_data is not None:
            existing_triples = done_data.get("n_triples", 0)
            if args.target_keys is None or args.target_keys <= existing_triples:
                logger.info(
                    "graph_done.json present with %d triples; target_keys=%s — nothing to do.",
                    existing_triples,
                    args.target_keys,
                )
                return
            else:
                # Raised target: delete done marker and continue.
                logger.info(
                    "Raising target from %d to %d — deleting graph_done.json.",
                    existing_triples,
                    args.target_keys,
                )
                done_path.unlink()

        # Load sessions already done.
        if state_path.exists():
            try:
                state_data = json.loads(state_path.read_text())
                sessions_done = state_data.get("sessions_done", [])
                logger.info(
                    "Resume: %d sessions already done, %d triples in existing snapshot.",
                    len(sessions_done),
                    state_data.get("n_unique_triples", 0),
                )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load build_state.json (%s) — starting fresh.", exc)

        # Clear any stale paused.json.
        if paused_path.exists():
            paused_path.unlink()
    else:
        # Fresh run: clear any stale files.
        for p in (done_path, paused_path):
            if p.exists():
                p.unlink()

    from experiments.quadruple_adapter import load_unique_triples
    from experiments.utils.gpu_guard import acquire_gpu
    from experiments.utils.longmemeval_loader import LongMemEvalLoader
    from experiments.utils.speaker_names import SpeakerNamePool
    from experiments.utils.test_harness import BENCHMARK_MODELS

    with acquire_gpu():
        logger.info("GPU acquired")
        wait_for_cooldown(52)

        # Load base model (exactly as dataset_probe.py).
        from paramem.models.loader import load_base_model

        model_cfg = BENCHMARK_MODELS[args.model]
        logger.info("Loading base model: %s", model_cfg.model_id)
        model, tokenizer = load_base_model(model_cfg)

        # Build ConsolidationLoop via canonical server factory
        # (verbatim from dataset_probe.py).
        import dataclasses

        from paramem.server.config import load_server_config
        from paramem.server.consolidation import create_consolidation_loop

        cfg = load_server_config("tests/fixtures/server.yaml")
        cfg.model_name = args.model

        # Extraction-only: rank/alpha/lr match dataset_probe defaults.
        cfg.adapters.episodic.rank = 8
        cfg.adapters.episodic.alpha = 16
        cfg.adapters.episodic.learning_rate = 1e-4
        cfg.adapters.semantic = dataclasses.replace(cfg.adapters.episodic)
        cfg.adapters.procedural.enabled = False

        if not args.with_sota:
            cfg.consolidation.extraction_noise_filter = ""
            cfg.consolidation.extraction_plausibility_judge = "off"

        loop = create_consolidation_loop(
            model=model,
            tokenizer=tokenizer,
            config=cfg,
            state_provider=None,
            output_dir=output_dir,
            save_cycle_snapshots=None,
            persist_graph=False,
            seed_state_from_disk=False,
        )

        # If resuming with an existing snapshot, load it into the merger so
        # dedup is correct and triple counts are accurate.
        if args.resume and snapshot_path.exists():
            logger.info("Loading existing graph snapshot from %s", snapshot_path)
            loop.merger.load_graph(snapshot_path)
            logger.info(
                "Graph loaded: %d nodes, %d edges",
                loop.merger.graph.number_of_nodes(),
                loop.merger.graph.number_of_edges(),
            )

        # Build loader (deterministic order via sample_seed).
        loader = LongMemEvalLoader(
            split=args.lme_split,
            sample_strategy="none",
            sample_size=None,
            sample_seed=args.lme_seed,
        )
        speaker_pool = SpeakerNamePool(seed=args.lme_seed)

        n_sessions_extracted = len(sessions_done)
        # Last-known triple count from the saved state (used for pause.json).
        n_unique_last: int = 0
        if state_path.exists():
            try:
                n_unique_last = json.loads(state_path.read_text()).get("n_unique_triples", 0)
            except (json.JSONDecodeError, OSError):
                pass
        last_session_id = sessions_done[-1] if sessions_done else "none"
        batch_since_persist = 0

        for session in loader.iter_sessions(limit=None, speaker_name_pool=speaker_pool):
            # --- Pause check (before extracting each session) ---
            # Flush the in-memory graph FIRST so the on-disk snapshot/state
            # match what we've extracted (otherwise up to persist_every-1
            # sessions are silently dropped and re-extracted on --resume).
            if PAUSE_FILE.exists():
                if batch_since_persist > 0:
                    n_unique_last = _persist(
                        loop=loop,
                        output_dir=output_dir,
                        snapshot_path=snapshot_path,
                        state_path=state_path,
                        sessions_done=sessions_done,
                        args=args,
                        load_unique_triples_fn=load_unique_triples,
                    )
                    batch_since_persist = 0
                _check_pause_and_exit(
                    output_dir,
                    last_session_id,
                    n_sessions_extracted,
                    n_unique_last,
                )

            # --- Skip already-done sessions (idempotent but wasteful) ---
            if session.session_id in sessions_done:
                logger.debug("Skipping already-done session: %s", session.session_id)
                continue

            logger.info("Extracting session: %s", session.session_id)

            # Between-batch cooldown would go here once a Python-importable
            # wait_for_cooldown() is available.  For now the persist-cadence
            # cooldown below is sufficient; the shell-side tresume path
            # (gpu-cooldown.sh) injects the env and handles between-session
            # cooling for sustained overnight runs.

            try:
                loop.extract_session(
                    session_transcript=session.transcript,
                    session_id=session.session_id,
                    speaker_id=session.speaker_id,
                    speaker_name=session.speaker_name,
                    ha_context=None,
                    stt_correction=False,
                    ha_validation=False,
                    noise_filter=extraction_noise_filter,
                    noise_filter_model="claude-sonnet-4-6",
                    noise_filter_endpoint=None,
                    ner_check=False,
                    plausibility_judge=plausibility_judge,
                    plausibility_stage="deanon",
                    verify_anonymization=True,
                )
            except Exception as exc:
                logger.error(
                    "extract_session failed for %s: %s — skipping.", session.session_id, exc
                )
                # Treat as done (skipped) so resume doesn't retry indefinitely.

            sessions_done.append(session.session_id)
            last_session_id = session.session_id
            n_sessions_extracted += 1
            batch_since_persist += 1

            # --- Persist on cadence ---
            if batch_since_persist >= args.persist_every:
                wait_for_cooldown(52)
                n_unique_last = _persist(
                    loop=loop,
                    output_dir=output_dir,
                    snapshot_path=snapshot_path,
                    state_path=state_path,
                    sessions_done=sessions_done,
                    args=args,
                    load_unique_triples_fn=load_unique_triples,
                )
                batch_since_persist = 0

                # --- Target check: use accurate triple count after persist ---
                if args.target_keys is not None and n_unique_last >= args.target_keys:
                    logger.info(
                        "Reached target_keys=%d (%d unique triples) — stopping.",
                        args.target_keys,
                        n_unique_last,
                    )
                    break

        else:
            # Loop exhausted naturally (no break from target check).
            # Fall through to final persist below.
            pass

        # Final persist (flush any unsaved sessions).
        wait_for_cooldown(52)
        n_unique = _persist(
            loop=loop,
            output_dir=output_dir,
            snapshot_path=snapshot_path,
            state_path=state_path,
            sessions_done=sessions_done,
            args=args,
            load_unique_triples_fn=load_unique_triples,
        )

        # Write graph_done.json.
        _safe_write_json(
            done_path,
            {
                "n_triples": n_unique,
                "n_sessions_extracted": n_sessions_extracted,
                "lme_split": args.lme_split,
                "target_keys": args.target_keys,
            },
        )

    logger.info(
        "Done. n_sessions_extracted=%d, n_unique_triples=%d, snapshot=%s",
        n_sessions_extracted,
        n_unique,
        snapshot_path,
    )
    print(
        f"\nSummary\n"
        f"  sessions_extracted : {n_sessions_extracted}\n"
        f"  n_unique_triples   : {n_unique}\n"
        f"  graph_snapshot     : {snapshot_path}\n"
    )


def _persist(
    *,
    loop,
    output_dir: Path,
    snapshot_path: Path,
    state_path: Path,
    sessions_done: list[str],
    args: argparse.Namespace,
    load_unique_triples_fn,
) -> int:
    """Write graph_snapshot.json and build_state.json.

    Uses the merger's save_graph (NetworkX node-link JSON, unencrypted)
    so Stage 2 (quadruple_adapter.py) can load it directly via
    load_unique_triples().

    Args:
        loop: Live ConsolidationLoop (merger has the accumulated graph).
        output_dir: Canonical output directory.
        snapshot_path: Path to write the graph snapshot.
        state_path: Path to write build_state.json.
        sessions_done: List of session IDs extracted so far.
        args: Parsed CLI arguments (for metadata).
        load_unique_triples_fn: ``load_unique_triples`` from quadruple_adapter.

    Returns:
        Number of unique (subject, predicate, object) triples in the snapshot.
    """
    loop.merger.save_graph(snapshot_path, encrypted=False)
    # Count unique triples consistently with how Stage 2 will count them.
    try:
        n_unique = len(load_unique_triples_fn(snapshot_path))
    except Exception as exc:
        logger.warning("Could not count unique triples from snapshot: %s", exc)
        n_unique = loop.merger.graph.number_of_edges()

    _safe_write_json(
        state_path,
        {
            "sessions_done": sessions_done,
            "n_unique_triples": n_unique,
            "lme_split": args.lme_split,
            "lme_seed": args.lme_seed,
            "target_keys": args.target_keys,
            "updated_at": int(time.time()),
        },
    )
    logger.info(
        "Persisted: %d sessions, %d unique triples -> %s",
        len(sessions_done),
        n_unique,
        snapshot_path,
    )
    return n_unique


if __name__ == "__main__":
    main()
