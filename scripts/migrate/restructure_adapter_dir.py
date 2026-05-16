#!/usr/bin/env python3
"""One-shot migration: relocate interim adapters under adapter_dir/episodic/.

The 2026-05-14 hierarchy refactor moves the interim slot directory from
the legacy flat layout

    <adapter_dir>/episodic_interim_<stamp>/<slot_date>/

to the hierarchical layout

    <adapter_dir>/episodic/interim_<stamp>/<slot_date>/

Run this once after deploying the matching code change.  Idempotent: a
clean re-run is a no-op.  The PEFT adapter NAME (``episodic_interim_<stamp>``)
is unchanged — only the on-disk path moves.

Usage:
    python scripts/migrate/restructure_adapter_dir.py [--adapter-dir PATH]

Defaults to ``data/ha/adapters`` (the live host's directory).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def restructure(adapter_dir: Path) -> tuple[int, int]:
    """Move every legacy ``episodic_interim_<stamp>`` dir under ``episodic/``.

    Returns ``(moved, skipped)`` counts.  Raises if a destination already
    exists with conflicting content.
    """
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"adapter_dir not found: {adapter_dir}")

    legacy = sorted(p for p in adapter_dir.glob("episodic_interim_*") if p.is_dir())
    if not legacy:
        logger.info("No legacy interim dirs found at %s — clean layout", adapter_dir)
        return 0, 0

    episodic_root = adapter_dir / "episodic"
    episodic_root.mkdir(parents=True, exist_ok=True)

    moved = 0
    skipped = 0
    for src in legacy:
        stamp = src.name[len("episodic_interim_") :]
        dst = episodic_root / f"interim_{stamp}"
        if dst.exists():
            if any(dst.iterdir()):
                raise RuntimeError(
                    f"Cannot migrate {src} → {dst}: destination already exists and is non-empty. "
                    "Resolve manually before re-running."
                )
            # Empty dst: remove and proceed.
            dst.rmdir()
        logger.info("Moving %s → %s", src, dst)
        shutil.move(str(src), str(dst))
        moved += 1

    # Drop bare ``interim_<stamp>/`` directories at the top level of
    # ``<adapter_dir>``.  These are raw HF Trainer output dirs (not v3 slots)
    # left behind by older training cycles — never recoverable per the wipe
    # invariant.  Name collides with the new hierarchy where ``interim_<stamp>``
    # is the per-stamp subdir under ``episodic/``, so they must move out of
    # the way at the top level.
    for stale in sorted(adapter_dir.glob("interim_*")):
        if stale.is_dir() and stale.parent == adapter_dir:
            logger.info("Removing stale top-level HF Trainer dir: %s", stale)
            shutil.rmtree(stale)

    # Drop the legacy ``data/ha/debug/cycle_*`` and ``data/ha/debug/run_*``
    # dirs left behind by the previous two-routine debug-dir scheme.
    debug_dir = adapter_dir.parent / "debug"
    if debug_dir.is_dir():
        for stale in list(debug_dir.glob("cycle_*")) + list(debug_dir.glob("run_*")):
            try:
                shutil.rmtree(stale)
                logger.info("Removed stale debug dir: %s", stale)
            except OSError as exc:
                logger.warning("Could not remove %s: %s", stale, exc)

    # Drop the legacy ``data/ha/sessions/archive/`` dir — pending-session
    # retention now lives under ``paths.debug`` (one roof, see #13).
    sessions_archive = adapter_dir.parent / "sessions" / "archive"
    if sessions_archive.is_dir():
        try:
            shutil.rmtree(sessions_archive)
            logger.info("Removed legacy sessions/archive/: %s", sessions_archive)
        except OSError as exc:
            logger.warning("Could not remove %s: %s", sessions_archive, exc)

    return moved, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("data/ha/adapters"),
        help="Live adapter directory (default: data/ha/adapters)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    moved, skipped = restructure(args.adapter_dir)
    logger.info("Migration complete: %d moved, %d skipped", moved, skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
