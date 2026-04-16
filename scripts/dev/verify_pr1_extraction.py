#!/usr/bin/env python3
"""Standalone verification script for PR1 extraction-alignment fixes.

Runs the full extraction pipeline (simulate mode only — no training) against a
filtered subset of real HA sessions to confirm that:
  - D6 entity-type fallback resolves the production regression (all entities
    typed "person").
  - D3 plausibility stage, D1/D7/D8/D10 fallbacks fire correctly on live data.

Usage (stop the server first to free VRAM, then run in background)::

    systemctl --user stop paramem-server
    export $(grep -v '^#' .env | xargs)
    python scripts/dev/verify_pr1_extraction.py [--session-prefixes 01KNW,01KP0] \\
        [--limit 10] [--no-cloud-noise-filter]

The script writes a timestamped sim_<ts>/ directory under data/ha/debug/ and
prints the path to stdout on exit so the caller can inspect results.

Notes
-----
* Requires GPU + NF4 quantization. Run in background (run_in_background=True
  via Claude Code, or nohup/disown from the shell).
* HF_DEACTIVATE_ASYNC_LOAD=1 must be set in the environment (loaded from
  .env via the export idiom above).
* Never mutates session state on disk — the real SessionBuffer's
  mark_consolidated() is never called.
* No server start/stop, no HTTP calls.
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the project root without installation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paramem.models.loader import load_base_model
from paramem.server.config import load_server_config
from paramem.server.consolidation import run_consolidation
from paramem.server.session_buffer import SessionBuffer

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
for _noisy in ("httpx", "anthropic", "urllib3", "transformers", "accelerate", "bitsandbytes"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/server.yaml")
DEFAULT_SESSION_PREFIXES = ["01KNW", "01KP0", "01KP2", "01KP8"]
DEFAULT_LIMIT = 30


# ---------------------------------------------------------------------------
# Thin SessionBuffer wrapper — exposes only the filtered sessions from
# get_pending() so run_consolidation sees exactly the sessions we want.
# mark_consolidated() is a no-op: the real buffer's state is never touched.
# ---------------------------------------------------------------------------


class _FilteredSessionBuffer(SessionBuffer):
    """Non-mutating wrapper that returns a pre-filtered, pre-built session list.

    Constructed from a list of already-formatted session dicts
    ({"session_id", "transcript", "speaker_id"}) sourced from the real
    SessionBuffer. Overrides:
      - get_pending()     → returns the injected list unchanged
      - mark_consolidated() → no-op (caller must not mutate the real buffer)

    All other SessionBuffer methods are inherited but will operate on an empty
    in-memory store (no turns loaded), which is fine because we never call them
    during a consolidation run.
    """

    def __init__(self, session_dir: Path, filtered_sessions: list[dict]):
        """Initialise with an empty in-memory store and inject session list.

        Args:
            session_dir: Path to the sessions directory (passed through to
                parent __init__ for compatibility; no files are written).
            filtered_sessions: Pre-built list of session dicts in the format
                returned by SessionBuffer.get_pending().
        """
        # debug=False, snapshot_key="" — no disk writes, no encryption
        super().__init__(session_dir=session_dir, debug=False, snapshot_key="")
        self._filtered_sessions = filtered_sessions

    def get_pending(self) -> list[dict]:
        """Return the injected filtered session list.

        Returns:
            The pre-filtered list of session dicts passed at construction time.
        """
        return list(self._filtered_sessions)

    def mark_consolidated(self, session_ids: list[str]) -> None:
        """No-op override — never mutate the real buffer.

        Args:
            session_ids: Ignored.
        """
        logger.info(
            "mark_consolidated suppressed for %d session(s) (verify script — no mutation)",
            len(session_ids),
        )


# ---------------------------------------------------------------------------
# Session filtering helpers
# ---------------------------------------------------------------------------


def _build_filtered_sessions(
    real_buffer: SessionBuffer,
    prefixes: list[str],
    limit: int,
) -> list[dict]:
    """Read pending sessions from the real buffer and keep prefix-matched ones.

    Reads via the real buffer's public get_pending() — no internal state
    is touched. Sessions are ordered as returned by the buffer (deterministic
    on disk order for debug-mode buffers).

    Args:
        real_buffer: The real SessionBuffer (read-only access via get_pending).
        prefixes: Only sessions whose session_id starts with one of these
            prefixes are kept. Empty list keeps all sessions.
        limit: Maximum number of sessions to return.

    Returns:
        Filtered, capped list of session dicts.
    """
    all_pending = list(real_buffer.get_pending())
    logger.info("Real buffer has %d pending session(s)", len(all_pending))

    # Also read archived sessions directly — sessions that were already
    # consolidated live under session_dir/archive/*.jsonl. We reuse the real
    # buffer's private helpers (_read_jsonl, _format_turns) to format them
    # identically to get_pending().
    archive_dir = real_buffer.archive_dir
    if archive_dir.exists():
        seen = {s["session_id"] for s in all_pending}
        for path in sorted(archive_dir.glob("*.jsonl")):
            conv_id = path.stem
            if conv_id in seen:
                continue
            turns = real_buffer._read_jsonl(path)
            formatted, speaker_id = real_buffer._format_turns(turns)
            if formatted:
                all_pending.append(
                    {
                        "session_id": conv_id,
                        "transcript": "\n".join(formatted),
                        "speaker_id": speaker_id,
                    }
                )
        logger.info("After including archive: %d session(s)", len(all_pending))

    if prefixes:
        filtered = [s for s in all_pending if any(s["session_id"].startswith(p) for p in prefixes)]
        logger.info("After prefix filter (%s): %d session(s)", ",".join(prefixes), len(filtered))
    else:
        filtered = list(all_pending)

    capped = filtered[:limit]
    if len(capped) < len(filtered):
        logger.info("Capped to %d session(s) (--limit)", len(capped))

    for s in capped:
        logger.info("  Selected: %s (speaker_id=%s)", s["session_id"], s.get("speaker_id"))

    return capped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run simulate-mode consolidation on a filtered subset of HA sessions.

    Parses CLI args, loads config, overrides simulate + debug mode, loads the
    extraction model, filters sessions, and calls run_consolidation. The debug
    dir path is printed to stdout on exit.
    """
    parser = argparse.ArgumentParser(
        description=(
            "PR1 extraction-alignment verification: run simulate-mode consolidation "
            "on a filtered subset of sessions to validate D6/D3/D1/D7/D8/D10 fixes."
        )
    )
    parser.add_argument(
        "--session-prefixes",
        default=",".join(DEFAULT_SESSION_PREFIXES),
        help=(
            "Comma-separated session_id prefixes to include "
            f"(default: {','.join(DEFAULT_SESSION_PREFIXES)}). "
            "Empty string keeps all pending sessions."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Maximum number of sessions to process (default: {DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "--cloud-noise-filter",
        dest="cloud_noise_filter",
        action="store_true",
        default=True,
        help="Enable cloud noise filter (default: ON, matches production config).",
    )
    parser.add_argument(
        "--plausibility",
        default=None,
        help="Override extraction_plausibility_judge (e.g. 'off', 'auto', 'anthropic').",
    )
    parser.add_argument(
        "--no-cloud-noise-filter",
        dest="cloud_noise_filter",
        action="store_false",
        help="Disable cloud noise filter (sets extraction_noise_filter to empty string).",
    )
    args = parser.parse_args()

    prefixes = [p.strip() for p in args.session_prefixes.split(",") if p.strip()]

    # --- Load and override config ---
    config = load_server_config(CONFIG_PATH)

    # Force simulate + debug so _save_simulation_results runs
    config.consolidation.mode = "simulate"
    config.debug = True

    if args.plausibility is not None:
        config.consolidation.extraction_plausibility_judge = args.plausibility
        logger.info("Plausibility judge overridden → %s", args.plausibility)

    if not args.cloud_noise_filter:
        config.consolidation.extraction_noise_filter = ""
        logger.info("Cloud noise filter disabled via --no-cloud-noise-filter")
    else:
        logger.info(
            "Cloud noise filter: %s", config.consolidation.extraction_noise_filter or "(off)"
        )

    logger.info(
        "Config: model=%s, mode=%s, debug=%s, session_dir=%s, debug_dir=%s",
        config.model_name,
        config.consolidation.mode,
        config.debug,
        config.session_dir,
        config.debug_dir,
    )

    # --- Build real buffer (read-only) ---
    # debug=True so it reads .jsonl files from disk (production sessions are
    # persisted to disk when the server runs with debug=True).
    real_buffer = SessionBuffer(
        session_dir=config.session_dir,
        retain_sessions=True,
        debug=True,
        snapshot_key="",
    )

    filtered_sessions = _build_filtered_sessions(real_buffer, prefixes, args.limit)
    if not filtered_sessions:
        logger.error(
            "No sessions matched prefix(es) %s in %s — nothing to do.",
            prefixes or "(all)",
            config.session_dir,
        )
        sys.exit(1)

    logger.info("Processing %d session(s)", len(filtered_sessions))

    # --- Load extraction model (same path as server startup in app.py) ---
    # load_base_model handles NF4, device_map={"": 0} for non-offload models,
    # and device_map="auto" + max_memory for offload models (e.g. Gemma 2 9B).
    # No torch_dtype alongside quantization_config — loader handles this.
    logger.info("Loading model: %s (%s)", config.model_name, config.model_config.model_id)
    model, tokenizer = load_base_model(config.model_config)
    logger.info("Model loaded")

    # --- Construct non-mutating wrapper buffer ---
    filtered_buffer = _FilteredSessionBuffer(
        session_dir=config.session_dir,
        filtered_sessions=filtered_sessions,
    )

    # --- Run consolidation in simulate mode ---
    # run_consolidation calls filtered_buffer.get_pending() internally and
    # filtered_buffer.mark_consolidated() (which is a no-op here).
    # With mode="simulate" + debug=True it calls _save_simulation_results,
    # which writes the timestamped sim_<ts>/ directory under config.debug_dir.
    logger.info("Starting simulate-mode consolidation")
    result = run_consolidation(
        model=model,
        tokenizer=tokenizer,
        config=config,
        session_buffer=filtered_buffer,
        loop=None,
        ha_context=None,
        speaker_store=None,
    )

    status = result.get("status")
    logger.info("Consolidation result: %s", {k: v for k, v in result.items() if k != "loop"})

    if status not in ("simulated", "no_facts", "no_pending"):
        logger.error("Unexpected consolidation status: %s", status)
        sys.exit(1)

    # Locate the debug dir that was just written. _save_simulation_results uses
    # a timestamp suffix — find the newest sim_* directory.
    debug_dir = config.debug_dir
    sim_dirs = sorted(debug_dir.glob("sim_*"), key=lambda p: p.name)
    if sim_dirs:
        latest_sim = sim_dirs[-1]
        print(str(latest_sim))
        logger.info("Simulation output: %s", latest_sim)
    else:
        logger.warning("No sim_* directory found under %s", debug_dir)
        print(str(debug_dir))


if __name__ == "__main__":
    main()
