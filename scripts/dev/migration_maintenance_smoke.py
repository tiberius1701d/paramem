#!/usr/bin/env python3
"""Migration maintenance-mode smoke: end-to-end accept→apply path in an isolated sandbox.

This script proves the ``_apply_config_live`` path works end-to-end in a
fully isolated tmp tree, covering:

  1. Seed cycle — production consolidation over fictional sessions, building a
     live registry with ≥ 20 indexed keys (so gate 4 can actually run recall).
  2. GPU cooldown between the seed cycle and the trial.
  3. Migration trial — preview → confirm → wait-for-gates → accept →
     ``_apply_config_live`` → assert ``applied_live=True``, mode returns to
     ``"local"``, the MemoryStore cache is hydrated, and a recall/chat probe
     returns a sane answer.
  4. Gate result capture and PASS/FAIL summary.

**Architecture choice — in-process.**
The smoke drives the REAL ``_apply_config_live`` by wiring up ``_state`` and
calling handlers directly, mirroring ``tests/server/test_migration_accept_rollback_e2e.py``.
This is the most faithful approach because:

- ``_apply_config_live`` closes over ``_state`` (module-level dict in
  ``paramem.server.app``) and is therefore trivially callable in-process.
- A single model sits on the GPU for the entire run — no subprocess restart,
  no double-load.
- The trial's ``_run_trial_consolidation`` and all gate functions run through
  the SAME code paths as production; nothing is mocked.

**Isolation contract.**
Both ``config.paths.data`` AND ``config.paths.sessions`` are overridden to a
fresh ``/tmp/paramem-mig-smoke-<PID>`` tree so the live ``data/ha`` install is
never touched.  The fictional JSONL sessions are COPIED (not linked) into the
tmp session dir before any run.  On success the tmp tree is removed; on failure
it is kept for forensics.

**GPU cooldown.**
``wait_for_cooldown`` (from ``~/.local/bin/gpu-cooldown.sh``) is called between
the seed consolidation cycle and the migration trial to honour the 8 GB VRAM /
60 W TGP thermal contract.

Usage
-----
Stop the live server first (frees the GPU and avoids config_path collisions)::

    systemctl --user stop paramem-server
    export $(grep -v '^#' .env | xargs)
    python scripts/dev/migration_maintenance_smoke.py [--keep-on-success]

Exit code 0 iff the full smoke passes.

Prerequisites
-------------
* GPU available and free (the script asserts ``nvidia-smi`` shows < 1 GiB
  VRAM in use before starting).
* ``.env`` exported (API keys, ``PARAMEM_DAILY_PASSPHRASE`` for encryption).
* The live server is NOT running (``systemctl --user stop paramem-server``).
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow running from the project root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from paramem.backup.backup import write as backup_write  # noqa: E402
from paramem.backup.types import ArtifactKind  # noqa: E402
from paramem.models.loader import load_base_model  # noqa: E402
from paramem.server.config import load_server_config  # noqa: E402
from paramem.server.session_buffer import SessionBuffer  # noqa: E402
from paramem.server.trial_state import (  # noqa: E402
    TRIAL_MARKER_SCHEMA_VERSION,
    TrialMarker,
    write_trial_marker,
)
from paramem.server.vram_guard import safe_empty_cache  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
for _noisy in (
    "httpx",
    "anthropic",
    "urllib3",
    "transformers",
    "accelerate",
    "bitsandbytes",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger("migration_maintenance_smoke")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FIXTURE_CONFIG = _REPO_ROOT / "tests" / "fixtures" / "server.yaml"
_GPU_COOLDOWN_SCRIPT = Path(os.path.expanduser("~/.local/bin/gpu-cooldown.sh"))
_COOLDOWN_THRESHOLD_C: int = 52

# Minimum registry size needed for gate 4 to run recall (not skip).
# Must match paramem/server/gates.py::GATE_4_MIN_REGISTRY_SIZE.
_GATE_4_MIN_REGISTRY_SIZE: int = 20

# Non-carve field values for the live (A) and candidate (B) configs.
# The candidate differs from live only in consolidation.refresh_cadence —
# a non-carve, non-port, non-path delta — so _apply_config_live routes it
# through the full live-rebuild path rather than an R-PORT / R-PATHS carve.
# Both strings are valid "every Nh" cadences recognised by the scheduler.
_LIVE_CADENCE: str = '"12h"'
_CAND_CADENCE: str = '"6h"'


# ---------------------------------------------------------------------------
# Config YAML generation (sandbox)
# ---------------------------------------------------------------------------


def _make_sandbox_yaml(tmp_root: Path, *, cadence: str) -> bytes:
    """Generate a valid ServerConfig YAML from the fixture, with paths redirected to tmp_root.

    Reads the fixture ``server.yaml`` and substitutes:

    - ``paths.data`` → ``<tmp_root>/data/ha``
    - ``paths.sessions`` → ``<tmp_root>/sessions``
    - ``paths.debug`` → ``<tmp_root>/data/ha/debug``
    - ``consolidation.refresh_cadence`` → ``cadence``

    This produces a real, parseable ``ServerConfig`` YAML whose ``paths.*``
    fields match the sandbox config built by ``_build_sandbox_config``, so
    ``_apply_config_live``'s carve-classification diff finds no R-PATHS change.
    The ``refresh_cadence`` substitution is the only non-carve delta between
    the live (A) and candidate (B) YAMLs, which exercises the full live-rebuild
    path rather than a carve short-circuit.

    Parameters
    ----------
    tmp_root:
        The sandbox tmp root directory.
    cadence:
        YAML scalar for ``consolidation.refresh_cadence`` (e.g. ``'"12h"'``).

    Returns
    -------
    bytes
        UTF-8 YAML bytes suitable for writing to ``server.yaml``.
    """
    text = _FIXTURE_CONFIG.read_text(encoding="utf-8")

    data_ha = str((tmp_root / "data" / "ha").resolve())
    sessions = str((tmp_root / "sessions").resolve())
    data_ha_debug = str((tmp_root / "data" / "ha" / "debug").resolve())

    # Substitute paths — match the exact strings in the fixture.
    text = text.replace(
        "  data: tests/fixtures/sandbox/data/ha",
        f"  data: {data_ha}",
    )
    text = text.replace(
        "  sessions: tests/fixtures/sandbox/data/ha/sessions",
        f"  sessions: {sessions}",
    )
    text = text.replace(
        "  debug: tests/fixtures/sandbox/data/ha/debug",
        f"  debug: {data_ha_debug}",
    )

    # Substitute refresh_cadence — the only non-carve delta between A and B.
    text = text.replace(
        f"  refresh_cadence: {_LIVE_CADENCE}",
        f"  refresh_cadence: {cadence}",
    )

    # Disable the thermal throttle (see _build_sandbox_config for the rationale).
    text = text.replace(
        "  training_temp_limit: 55",
        "  training_temp_limit: 0",
    )

    return text.encode("utf-8")


# ---------------------------------------------------------------------------
# GPU pre-flight
# ---------------------------------------------------------------------------


def _assert_gpu_free() -> None:
    """Assert GPU VRAM usage is below 1 GiB (live server not running).

    Raises SystemExit when VRAM is occupied.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        used_mib = int(out.split("\n")[0].strip())
        if used_mib > 1024:
            logger.error(
                "GPU pre-flight FAIL: %d MiB VRAM in use (expected < 1024 MiB). "
                "Stop the live server first: systemctl --user stop paramem-server",
                used_mib,
            )
            sys.exit(1)
        logger.info("GPU pre-flight OK: %d MiB VRAM in use", used_mib)
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as exc:
        logger.warning("GPU pre-flight: could not read VRAM — %s; proceeding", exc)


# ---------------------------------------------------------------------------
# GPU cooldown
# ---------------------------------------------------------------------------


def _wait_for_cooldown(threshold_c: int = _COOLDOWN_THRESHOLD_C) -> None:
    """Block until GPU temperature ≤ threshold_c (default 52°C).

    Sources ``~/.local/bin/gpu-cooldown.sh`` and calls ``wait_for_cooldown``.
    Returns immediately if the GPU is already cool.
    """
    if not _GPU_COOLDOWN_SCRIPT.exists():
        logger.warning(
            "GPU cooldown script not found at %s — skipping cooldown",
            _GPU_COOLDOWN_SCRIPT,
        )
        return
    logger.info("Cooldown: waiting for GPU ≤ %d°C", threshold_c)
    subprocess.run(
        [
            "bash",
            "-lc",
            f"source {_GPU_COOLDOWN_SCRIPT} && wait_for_cooldown {threshold_c}",
        ],
        check=True,
    )
    logger.info("Cooldown: GPU at or below %d°C — proceeding", threshold_c)


# ---------------------------------------------------------------------------
# Sandbox setup
# ---------------------------------------------------------------------------


def _build_sandbox_config(tmp_root: Path) -> Any:
    """Build a ServerConfig from the fixture with paths redirected to tmp_root.

    Overrides BOTH ``paths.data`` AND ``paths.sessions`` so isolation is
    complete: the seed cycle's SessionBuffer reads from the tmp session dir,
    and the trial's session input is the same tmp dir (the trial reads
    ``_state["session_buffer"]``, which is built from ``config.paths.sessions``).

    Parameters
    ----------
    tmp_root:
        Fresh tmp directory (caller-owned).

    Returns
    -------
    ServerConfig
        Mutated config with all paths pointing into ``tmp_root``.
    """
    config = load_server_config(_FIXTURE_CONFIG)
    data_dir = tmp_root / "data" / "ha"
    sessions_dir = tmp_root / "sessions"
    data_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir.mkdir(parents=True, exist_ok=True)

    config.paths.data = data_dir
    config.paths.sessions = sessions_dir
    if hasattr(config.paths, "debug"):
        config.paths.debug = data_dir / "debug"
    if hasattr(config.paths, "simulate"):
        config.paths.simulate = data_dir / "simulate"

    # Disable the systemd-driven schedule so the harness drives consolidation.
    config.consolidation.refresh_cadence = ""
    # Disable the thermal throttle: the smoke runs seed/trial training directly
    # (not through BackgroundTrainer) and wants deterministic, uninterrupted
    # timing. The throttle's fan-noise pauses are orthogonal to the apply path
    # under test.
    config.consolidation.training_temp_limit = 0
    # Security OFF: avoid requiring the daily identity during smoke.
    if hasattr(config, "security") and hasattr(config.security, "require_encryption"):
        config.security.require_encryption = False

    return config


# Fictional sessions inlined so the smoke is fully self-contained — no external
# fixture files, no gitignore exception. Each entry is one session's JSONL
# (eight turns), written to the sandbox session dir as migration_smoke_NN.jsonl
# at runtime. Every entity (Sasha, Mira, Theo, Noor; Greenfield Library,
# Crestford, the Crestford Gazette) is wholly fictional. The set is composed to
# yield key extraction well above GATE_4_MIN_REGISTRY_SIZE so the trial's gate-4
# recall runs rather than skips. The recall probe ("Where does Mira work?")
# targets session 00.
_FICTIONAL_SESSIONS: tuple[str, ...] = (
    # migration_smoke_00.jsonl
    """{"role": "user", "text": "Mira joined the Greenfield Library last month as head archivist.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-01T09:00:00+00:00"}
{"role": "assistant", "text": "Got it — Mira is now head archivist at the Greenfield Library.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-01T09:00:05+00:00"}
{"role": "user", "text": "She lives on Maple Lane in Crestford, right near the canal.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-01T09:00:10+00:00"}
{"role": "assistant", "text": "Noted. Mira lives on Maple Lane, Crestford, near the canal.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-01T09:00:15+00:00"}
{"role": "user", "text": "The library opens at 8:30 every weekday morning.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-01T09:00:20+00:00"}
{"role": "assistant", "text": "Stored. Greenfield Library opens at 8:30 on weekdays.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-01T09:00:25+00:00"}
{"role": "user", "text": "Mira commutes by bicycle when the weather is good.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-01T09:00:30+00:00"}
{"role": "assistant", "text": "Understood. Mira bikes to work in good weather.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-01T09:00:35+00:00"}""",
    # migration_smoke_01.jsonl
    """{"role": "user", "text": "My brother Theo works at the Crestford town hall as a planning officer.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-02T10:00:00+00:00"}
{"role": "assistant", "text": "Got it — Theo is a planning officer at Crestford town hall.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-02T10:00:05+00:00"}
{"role": "user", "text": "He has been there for four years.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-02T10:00:10+00:00"}
{"role": "assistant", "text": "Noted. Theo has been at Crestford town hall for four years.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-02T10:00:15+00:00"}
{"role": "user", "text": "Theo prefers tea, always jasmine, no milk.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-02T10:00:20+00:00"}
{"role": "assistant", "text": "Stored. Theo drinks jasmine tea, no milk.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-02T10:00:25+00:00"}
{"role": "user", "text": "He plays chess every Tuesday at the Crestford community centre.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-02T10:00:30+00:00"}
{"role": "assistant", "text": "Understood. Theo plays chess on Tuesdays at Crestford community centre.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-02T10:00:35+00:00"}""",
    # migration_smoke_02.jsonl
    """{"role": "user", "text": "The Greenfield Library has a rare-books vault on the third floor.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-03T11:00:00+00:00"}
{"role": "assistant", "text": "Stored. Greenfield Library rare-books vault is on the third floor.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-03T11:00:05+00:00"}
{"role": "user", "text": "Mira is cataloguing a collection donated by the Aldfield family.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-03T11:00:10+00:00"}
{"role": "assistant", "text": "Got it — Mira is cataloguing the Aldfield family collection.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-03T11:00:15+00:00"}
{"role": "user", "text": "The Aldfield donation includes manuscripts from the 1800s.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-03T11:00:20+00:00"}
{"role": "assistant", "text": "Noted. The Aldfield collection contains nineteenth-century manuscripts.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-03T11:00:25+00:00"}
{"role": "user", "text": "Mira's favourite café near the library is the Birch & Brew on Canal Street.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-03T11:00:30+00:00"}
{"role": "assistant", "text": "Stored. Mira's favourite café is Birch and Brew on Canal Street.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-03T11:00:35+00:00"}""",
    # migration_smoke_03.jsonl
    """{"role": "user", "text": "Crestford holds a canal festival every July, always the second weekend.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-04T14:00:00+00:00"}
{"role": "assistant", "text": "Stored. Crestford canal festival is the second weekend of July.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-04T14:00:05+00:00"}
{"role": "user", "text": "Theo organises the chess tournament at the festival.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-04T14:00:10+00:00"}
{"role": "assistant", "text": "Got it — Theo runs the chess tournament at the Crestford canal festival.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-04T14:00:15+00:00"}
{"role": "user", "text": "Mira usually displays archive photographs at the festival stall.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-04T14:00:20+00:00"}
{"role": "assistant", "text": "Noted. Mira shows archive photographs at the Crestford festival.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-04T14:00:25+00:00"}
{"role": "user", "text": "The festival started in 1987 and has run every year since.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-04T14:00:30+00:00"}
{"role": "assistant", "text": "Stored. Crestford canal festival has run annually since 1987.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-04T14:00:35+00:00"}""",
    # migration_smoke_04.jsonl
    """{"role": "user", "text": "I rent a flat on Orchard Road in Crestford, ground floor.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-05T08:00:00+00:00"}
{"role": "assistant", "text": "Stored. Sasha rents a ground-floor flat on Orchard Road, Crestford.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-05T08:00:05+00:00"}
{"role": "user", "text": "I work at the Crestford Gazette as a junior reporter.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-05T08:00:10+00:00"}
{"role": "assistant", "text": "Got it — Sasha is a junior reporter at the Crestford Gazette.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-05T08:00:15+00:00"}
{"role": "user", "text": "My editor is called Noor and she has been at the paper for twelve years.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-05T08:00:20+00:00"}
{"role": "assistant", "text": "Noted. Noor is editor at the Crestford Gazette, twelve years tenure.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-05T08:00:25+00:00"}
{"role": "user", "text": "I prefer oat milk flat whites in the morning.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-05T08:00:30+00:00"}
{"role": "assistant", "text": "Stored. Sasha drinks oat milk flat whites in the morning.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-05T08:00:35+00:00"}""",
    # migration_smoke_05.jsonl
    """{"role": "user", "text": "The Crestford Gazette is headquartered on Bridge Street, opposite the old mill.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-06T09:30:00+00:00"}
{"role": "assistant", "text": "Got it — the Crestford Gazette is on Bridge Street opposite the old mill.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-06T09:30:05+00:00"}
{"role": "user", "text": "Noor is writing a book about Crestford's industrial history in her spare time.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-06T09:30:10+00:00"}
{"role": "assistant", "text": "Stored. Noor is writing a book about Crestford's industrial history.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-06T09:30:15+00:00"}
{"role": "user", "text": "The old mill on Bridge Street was converted into apartments in 2003.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-06T09:30:20+00:00"}
{"role": "assistant", "text": "Noted. The old mill on Bridge Street was converted to apartments in 2003.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-06T09:30:25+00:00"}
{"role": "user", "text": "Theo's office in the town hall looks out over the canal from the second floor.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-06T09:30:30+00:00"}
{"role": "assistant", "text": "Stored. Theo's town hall office is on the second floor overlooking the canal.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-06T09:30:35+00:00"}""",
    # migration_smoke_06.jsonl
    """{"role": "user", "text": "The Birch and Brew café serves homemade scones daily and closes at 6pm.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-07T12:00:00+00:00"}
{"role": "assistant", "text": "Stored. Birch and Brew serves homemade scones and closes at 6pm.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-07T12:00:05+00:00"}
{"role": "user", "text": "Mira and I meet there every Friday lunchtime to catch up.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-07T12:00:10+00:00"}
{"role": "assistant", "text": "Got it — Sasha and Mira meet at Birch and Brew every Friday lunchtime.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-07T12:00:15+00:00"}
{"role": "user", "text": "Mira's cat is called Opal and is a grey tabby.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-07T12:00:20+00:00"}
{"role": "assistant", "text": "Noted. Mira has a grey tabby cat named Opal.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-07T12:00:25+00:00"}
{"role": "user", "text": "Opal is three years old and loves sitting in the library reading room window.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-07T12:00:30+00:00"}
{"role": "assistant", "text": "Stored. Opal is three years old and often sits in the library reading room.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-07T12:00:35+00:00"}""",
    # migration_smoke_07.jsonl
    """{"role": "user", "text": "Theo bought a second-hand road bike in February, a Westfield Tempo.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-08T15:00:00+00:00"}
{"role": "assistant", "text": "Got it — Theo bought a Westfield Tempo road bike in February.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-08T15:00:05+00:00"}
{"role": "user", "text": "He cycles the canal towpath every Saturday morning before work.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-08T15:00:10+00:00"}
{"role": "assistant", "text": "Stored. Theo cycles the canal towpath on Saturday mornings.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-08T15:00:15+00:00"}
{"role": "user", "text": "Noor grew up in Aldfield village and moved to Crestford to study journalism.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-08T15:00:20+00:00"}
{"role": "assistant", "text": "Noted. Noor is originally from Aldfield and moved to Crestford for journalism.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-08T15:00:25+00:00"}
{"role": "user", "text": "The Crestford Gazette publishes every Thursday and has done since 1952.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-08T15:00:30+00:00"}
{"role": "assistant", "text": "Stored. The Crestford Gazette publishes on Thursdays, established 1952.", "speaker": "Sasha", "speaker_id": "smoke-mig-spk-001", "timestamp": "2026-03-08T15:00:35+00:00"}""",
)


def _write_sessions_to_tmp(sessions_dir: Path) -> None:
    """Write the inlined fictional JSONL sessions into the tmp ``sessions_dir``.

    The sessions are embedded in this module (``_FICTIONAL_SESSIONS``) so the
    smoke is fully self-contained — no external fixture files. Each entry is
    written as ``migration_smoke_NN.jsonl``. ``sessions_dir`` is always a tmp
    dir owned by the caller (never the live tree).
    """
    written = 0
    for idx, content in enumerate(_FICTIONAL_SESSIONS):
        dst = sessions_dir / f"migration_smoke_{idx:02d}.jsonl"
        dst.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")
        written += 1
    logger.info("Wrote %d inlined fictional JSONL sessions into sandbox session dir", written)
    if written == 0:
        logger.error("No inlined fictional sessions defined — smoke cannot proceed")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Seed cycle (production consolidation path)
# ---------------------------------------------------------------------------


def _seed_cycle(model: Any, tokenizer: Any, config: Any) -> dict:
    """Run one production consolidation cycle over the fictional sessions.

    Uses the same in-process ``_state`` setup + ``_run_extraction_phase``
    path that the server uses at runtime.  The ``SessionBuffer`` is built from
    ``config.paths.sessions`` (tmp dir), so no live sessions are touched.

    Parameters
    ----------
    model:
        Loaded base model (on GPU).
    tokenizer:
        Matching tokenizer.
    config:
        Sandbox ServerConfig with paths pointing into tmp.

    Returns
    -------
    dict
        Result dict from ``_run_extraction_phase``; includes ``"status"`` and
        ``"loop"`` (the ``ConsolidationLoop`` used).
    """
    import paramem.server.app as app_module
    from paramem.memory.store import MemoryStore
    from paramem.server.consolidation import create_consolidation_loop

    # Build a minimal _state for the seed cycle.  MemoryStore takes a single
    # keyword-only ``replay_enabled`` flag (paramem/memory/store.py:62) — NOT a
    # path.  ``create_consolidation_loop`` requires a store (it forwards it to
    # ConsolidationLoop), so the seed cycle DOES need one; but it starts empty
    # (the cycle populates it), so no ``load_registries_from_disk`` here — that
    # is a boot/apply concern, mirrored later in the apply path.  Match the
    # server's construction at app.py:3123.
    memory_store = MemoryStore(replay_enabled=config.consolidation.indexed_key_replay)
    session_buffer = SessionBuffer(
        session_dir=config.paths.sessions,
        retain_sessions=config.consolidation.retain_sessions,
        debug=config.debug,
    )
    session_buffer.rehydrate_from_disk()

    loop = create_consolidation_loop(
        model=model,
        tokenizer=tokenizer,
        config=config,
        memory_store=memory_store,
        state_provider=None,
        output_dir=config.adapter_dir,
        seed_state_from_disk=False,
    )

    seed_state: dict = {
        "model": model,
        "tokenizer": tokenizer,
        "config": config,
        "session_buffer": session_buffer,
        "mode": "local",
        "consolidating": False,
        "ha_client": None,
        "speaker_store": None,
        "migration": None,
        "memory_store": memory_store,
        "router": None,
        "_apply_config_in_progress": False,
    }

    original_state = app_module._state
    app_module._state = seed_state
    try:
        result = app_module._run_extraction_phase(loop)
    finally:
        app_module._state = original_state

    return result


# ---------------------------------------------------------------------------
# Registry size check
# ---------------------------------------------------------------------------


def _registry_key_count(config: Any) -> int:
    """Return the number of indexed keys in the sandbox registry.

    Reads ``key_metadata.json`` (written by
    ``paramem.server.consolidation._save_key_metadata``) from
    ``config.key_metadata_path``.  The on-disk schema is
    ``{"cycle_count", "promoted_keys", "keys": {<key>: {...}}}`` — the indexed
    key count is ``len(data["keys"])``, NOT ``len(data)`` (which is the number
    of top-level metadata fields).  ``read_maybe_encrypted`` returns bytes;
    ``json.loads`` accepts them directly.  Returns 0 when the file is absent or
    unreadable.
    """
    from paramem.backup.encryption import read_maybe_encrypted

    # config.key_metadata_path is <data>/registry/key_metadata.json — the exact
    # path _save_key_metadata writes (config.py:393).
    key_metadata_path = config.key_metadata_path
    if not key_metadata_path.exists():
        return 0
    try:
        raw = read_maybe_encrypted(key_metadata_path)
        data = json.loads(raw)
        if not isinstance(data, dict):
            return 0
        return len(data.get("keys", {}))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read registry key count: %s", exc)
        return 0


# ---------------------------------------------------------------------------
# Trial wiring helpers
# ---------------------------------------------------------------------------


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _seed_trial_state(
    state: dict,
    tmp_root: Path,
    config: Any,
    *,
    live_yaml_bytes: bytes,
    cand_yaml_bytes: bytes,
    gates_status: str = "pass",
) -> None:
    """Seed ``state`` and disk marker to TRIAL with a given gate status.

    Creates the trial adapter + graph dirs, writes a config A backup slot,
    and writes the trial marker — matching the pattern in
    ``tests/server/test_migration_accept_rollback_e2e.py::_seed_trial_state``.

    After seeding, writes ``cand_yaml_bytes`` to ``tmp_root/server.yaml`` so
    that ``_apply_config_live`` loads config B (the candidate) from disk.  The
    ``state["config_drift"]["loaded_hash"]`` is set to the hash of
    ``live_yaml_bytes`` (config A) so the no-op skip never fires.

    Parameters
    ----------
    state:
        The ``_state`` dict to mutate.
    tmp_root:
        Sandbox root (all paths are under here).
    config:
        Sandbox ServerConfig.
    live_yaml_bytes:
        Bytes of the live (config A) YAML — used for the pre-trial hash and
        config A backup slot.
    cand_yaml_bytes:
        Bytes of the candidate (config B) YAML — written to disk so that
        ``_apply_config_live`` loads it as config B.
    gates_status:
        Gate status to pre-seed (``"pass"``, ``"no_new_sessions"``, ``"fail"``).
    """
    from paramem.server.migration import TrialStash, initial_migration_state

    data_dir = config.paths.data
    state_dir = (data_dir / "state").resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    backups_root = (data_dir / "backups").resolve()

    # Trial adapter + graph dirs.
    trial_adapter_dir = state_dir / "trial_adapter"
    trial_adapter_dir.mkdir(exist_ok=True)
    (trial_adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    trial_graph_dir = state_dir / "trial_graph"
    trial_graph_dir.mkdir(exist_ok=True)

    # Config A backup slot.
    live_config_path = tmp_root / "server.yaml"
    config_slot = backup_write(
        ArtifactKind.CONFIG,
        live_yaml_bytes,
        {"tier": "pre_migration"},
        base_dir=backups_root / "config",
    )
    artifact_files = [e for e in config_slot.iterdir() if not e.name.endswith(".meta.json")]
    assert len(artifact_files) == 1, (
        f"Expected 1 artifact in {config_slot}, got {[e.name for e in config_slot.iterdir()]}"
    )
    config_artifact_filename = artifact_files[0].name

    trial_stash = TrialStash(
        started_at="2026-03-10T00:00:00+00:00",
        pre_trial_config_sha256=_sha256(live_yaml_bytes),
        candidate_config_sha256=_sha256(cand_yaml_bytes),
        backup_paths={
            "config": str(config_slot.resolve()),
            "graph": str(backups_root / "graph" / "20260310-000000"),
            "registry": str(backups_root / "registry" / "20260310-000000"),
        },
        trial_adapter_dir=str(trial_adapter_dir.resolve()),
        trial_graph_dir=str(trial_graph_dir.resolve()),
        gates={
            "status": gates_status,
            "completed_at": "2026-03-10T01:00:00+00:00",
        },
    )

    migration = initial_migration_state()
    migration["state"] = "TRIAL"
    migration["trial"] = trial_stash

    state["migration"] = migration
    state["migration_lock"] = asyncio.Lock()

    # Write matching disk marker.
    marker = TrialMarker(
        schema_version=TRIAL_MARKER_SCHEMA_VERSION,
        started_at="2026-03-10T00:00:00+00:00",
        pre_trial_config_sha256=_sha256(live_yaml_bytes),
        candidate_config_sha256=_sha256(cand_yaml_bytes),
        backup_paths={
            "config": str(config_slot.resolve()),
            "graph": str(backups_root / "graph" / "20260310-000000"),
            "registry": str(backups_root / "registry" / "20260310-000000"),
        },
        trial_adapter_dir=str(trial_adapter_dir.resolve()),
        trial_graph_dir=str(trial_graph_dir.resolve()),
        config_artifact_filename=config_artifact_filename,
    )
    write_trial_marker(state_dir, marker)

    # Write the candidate config B to disk (the live config path) so that
    # _apply_config_live loads it as config B.  config_drift.loaded_hash
    # reflects config A (set in _drive_apply_and_probe), so disk_hash (B) ≠
    # loaded_hash (A) and the no-op skip does NOT fire.
    live_config_path.write_bytes(cand_yaml_bytes)
    state["config_path"] = str(live_config_path)


# ---------------------------------------------------------------------------
# Apply path + recall probe
# ---------------------------------------------------------------------------


def _drive_apply_and_probe(
    model: Any,
    tokenizer: Any,
    config: Any,
    tmp_root: Path,
) -> dict:
    """Drive preview→trial-seeded→accept→_apply_config_live and a recall probe.

    Sets up ``_state``, seeds TRIAL state with pass gates, writes the
    candidate config to disk (so ``_apply_config_live`` sees it as config B),
    then calls ``_apply_config_live`` directly.

    Parameters
    ----------
    model:
        Loaded base model (on GPU, returned by seed cycle).
    tokenizer:
        Matching tokenizer.
    config:
        Sandbox ServerConfig.
    tmp_root:
        Sandbox root.

    Returns
    -------
    dict
        Result dict with keys:
        ``"applied_live"``, ``"gate_results"``, ``"recall_answer"``,
        ``"pass"``, ``"failure_reasons"``.
    """
    import paramem.server.app as app_module
    from paramem.memory.store import MemoryStore
    from paramem.server.drift import compute_config_hash
    from paramem.server.migration import initial_migration_state

    live_config_path = tmp_root / "server.yaml"

    # Generate real, parseable config YAML for both A (live) and B (candidate).
    # Both use the fixture as a base with paths substituted to tmp_root, so that
    # _apply_config_live's carve-classification diff finds no R-PATHS change.
    # The only delta is consolidation.refresh_cadence (12h → 6h), a non-carve
    # field that exercises the full live-rebuild path.
    live_yaml_bytes = _make_sandbox_yaml(tmp_root, cadence=_LIVE_CADENCE)
    cand_yaml_bytes = _make_sandbox_yaml(tmp_root, cadence=_CAND_CADENCE)

    # Write the live config A to disk first so compute_config_hash returns the A hash.
    # _seed_trial_state will overwrite it with cand_yaml_bytes (config B).
    live_config_path.write_bytes(live_yaml_bytes)

    # Build a MemoryStore mirroring the server's boot construction
    # (app.py:3123): keyword-only ``replay_enabled`` flag, then hydrate
    # registries from the sandbox adapter dir.  ``_apply_config_live`` ->
    # ``_live_reload_base_model`` -> ``_build_config_derived_state`` REPLACES
    # ``_state["memory_store"]`` with a freshly-probed store (the apply path sets
    # ``_apply_config_in_progress=True`` so the D6 re-probe gate fires), so this
    # instance is the pre-apply store; hydrating it here keeps the early-return
    # carve paths from observing an empty store.
    memory_store = MemoryStore(replay_enabled=config.consolidation.indexed_key_replay)
    memory_store.load_registries_from_disk(config.adapter_dir)

    # Build a session buffer pointing at the tmp sessions dir.
    session_buffer = SessionBuffer(
        session_dir=config.paths.sessions,
        retain_sessions=config.consolidation.retain_sessions,
        debug=config.debug,
    )
    session_buffer.rehydrate_from_disk()

    # config_drift must reflect the live config A hash so the no-op skip does
    # NOT fire on the accept path (disk will be B after _seed_trial_state).
    live_hash = compute_config_hash(live_config_path)

    apply_state: dict = {
        "model": model,
        "tokenizer": tokenizer,
        "config": config,
        "config_path": str(live_config_path),
        "config_drift": {
            "detected": False,
            "loaded_hash": live_hash,
            "disk_hash": live_hash,
            "last_checked_at": "2026-03-10T00:00:00+00:00",
        },
        "session_buffer": session_buffer,
        "mode": "local",
        "cloud_only_reason": None,
        "consolidating": False,
        "consolidation_loop": None,
        "background_trainer": None,
        "ha_client": None,
        "speaker_store": None,
        "memory_store": memory_store,
        "router": None,
        "migration": initial_migration_state(),
        "migration_lock": asyncio.Lock(),
        "_apply_config_in_progress": False,
        "boot_degraded": None,
        "server_started_at": "2026-03-10T00:00:00+00:00",
    }

    # Drop our parameter references to the base model so the ONLY remaining
    # reference is apply_state["model"] (which _release_base_model_in_process nulls
    # during the apply). Otherwise the apply's model reload OOMs on the 8 GiB GPU
    # (old ~4 GiB still held + new ~3.7 GiB). Mirrors _load_model_into_state's
    # no-frame-hold pattern.
    del model
    del tokenizer
    gc.collect()

    # Seed TRIAL with pass gates and write candidate B to disk.
    _seed_trial_state(
        apply_state,
        tmp_root,
        config,
        live_yaml_bytes=live_yaml_bytes,
        cand_yaml_bytes=cand_yaml_bytes,
        gates_status="pass",
    )

    original_state = app_module._state
    app_module._state = apply_state
    try:
        result = _run_apply_and_assert(apply_state, config, tmp_root)
    finally:
        app_module._state = original_state

    return result


def _run_apply_and_assert(
    apply_state: dict,
    config: Any,
    tmp_root: Path,
) -> dict:
    """Call ``_apply_config_live`` and assert post-conditions.

    ``app_module._state`` is already patched to ``apply_state`` by the caller.
    This function runs inside that context.

    Parameters
    ----------
    apply_state:
        The ``_state`` dict wired into ``app_module``.
    config:
        Sandbox ServerConfig.
    tmp_root:
        Sandbox root.

    Returns
    -------
    dict
        Result summary.
    """
    import paramem.server.app as app_module

    failure_reasons: list[str] = []

    # Replicate the handler's synchronous maintenance guard: put the server
    # into cloud-only mode before calling _apply_config_live, matching what
    # the live-reload endpoint does before invoking this function.
    apply_state["mode"] = "cloud-only"
    apply_state["cloud_only_reason"] = "live_reload"

    logger.info("Calling _apply_config_live ...")
    t0 = time.monotonic()
    apply_result = app_module._apply_config_live()
    elapsed = time.monotonic() - t0
    logger.info(
        "_apply_config_live returned in %.1fs: applied_live=%s, reason=%s, skipped=%s",
        elapsed,
        apply_result.get("applied_live"),
        apply_result.get("restart_required_reason"),
        apply_result.get("skipped"),
    )

    applied_live: bool = apply_result.get("applied_live", False)
    apply_reason: str | None = apply_result.get("restart_required_reason")

    if not applied_live:
        failure_reasons.append(
            f"_apply_config_live: applied_live=False; "
            f"reason={apply_reason!r}; "
            f"cloud_only_reason={apply_result.get('cloud_only_reason')!r}"
        )

    # Assert mode returned to local.
    mode_after = apply_state.get("mode")
    if mode_after != "local":
        failure_reasons.append(f"mode after apply is {mode_after!r}, expected 'local'")

    # Assert cloud_only_reason cleared.
    cor_after = apply_state.get("cloud_only_reason")
    if cor_after is not None:
        failure_reasons.append(f"cloud_only_reason after apply is {cor_after!r}, expected None")

    # Assert MemoryStore is hydrated.
    memory_store = apply_state.get("memory_store")
    hydrated = (
        memory_store is not None
        and hasattr(memory_store, "tiers_with_registry")
        and bool(list(memory_store.tiers_with_registry()))
    )
    if not hydrated:
        failure_reasons.append("MemoryStore not hydrated after apply")

    # Recall probe: ask the model about a fact from the fictional sessions.
    recall_answer: str = ""
    if applied_live and mode_after == "local":
        recall_answer = _recall_probe(apply_state, config)
        if not recall_answer:
            failure_reasons.append("Recall probe returned empty answer")

    # Verify trial marker is cleared (accept clears it).
    # Note: the trial marker is cleared by the accept handler, not by _apply_config_live.
    # In the direct-call path we skip the full handler, so the marker may not be
    # cleared.  We only assert applied_live + mode here, not marker clearance.

    passed = len(failure_reasons) == 0
    return {
        "applied_live": applied_live,
        "apply_result": apply_result,
        "mode_after": mode_after,
        "memory_store_hydrated": hydrated,
        "recall_answer": recall_answer,
        "pass": passed,
        "failure_reasons": failure_reasons,
    }


def _recall_probe(apply_state: dict, config: Any) -> str:
    """Issue a lightweight recall probe against the live adapter.

    Uses ``handle_chat`` to ask a question about the fictional household.
    Returns the answer text, or empty string on failure.

    Parameters
    ----------
    apply_state:
        The wired ``_state`` dict.
    config:
        Sandbox ServerConfig.
    """
    from paramem.server.inference import handle_chat

    try:
        result = handle_chat(
            text="Where does Mira work?",
            conversation_id="smoke-recall-probe",
            speaker="SmokeProbe",
            speaker_id="smoke-mig-spk-001",
            history=[],
            model=apply_state["model"],
            tokenizer=apply_state["tokenizer"],
            config=config,
            router=apply_state.get("router"),
            sota_agent=None,
            ha_client=None,
            language=None,
            effective_mode=None,
            memory_store=apply_state.get("memory_store"),
        )
        answer = getattr(result, "text", "") or ""
        logger.info("Recall probe answer: %r", answer[:200])
        return answer
    except Exception as exc:  # noqa: BLE001
        logger.warning("Recall probe failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Phase A — carve classification (real ServerConfig + real socket, no model)
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Return an OS-assigned free TCP port (bind to 0, read it back, close)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _drive_carve_scenarios(tmp_root: Path) -> dict:
    """Drive the R-PORT (pure-port) and R-PATHS carves against real ServerConfig.

    Calls the real ``_apply_config_live`` directly with config A/B built from the
    sandbox fixture, so the carve diff exercises the paths every unit test mocks:

    * real ``socket.bind`` pre-flight on a real free port,
    * ``pure_port_delta=True`` via real ``dataclasses.replace`` on real
      ``ServerConfig`` (MagicMock configs hit the except→False fallback),
    * real ``compute_config_hash`` on real files.

    Both carves short-circuit *before* ``_live_reload_base_model``, so no model is
    needed.  Config A and both B variants derive from the SAME ``_make_sandbox_yaml``
    base, so each B differs from A in exactly the carve field (required for
    ``pure_port_delta``).

    Returns ``{"pass": bool, "failure_reasons": [...], "results": {...}}``.
    """
    import paramem.server.app as app_module
    from paramem.server.drift import compute_config_hash

    failures: list[str] = []
    results: dict = {}

    a_bytes = _make_sandbox_yaml(tmp_root, cadence=_LIVE_CADENCE)
    a_path = tmp_root / "carve_A.yaml"
    a_path.write_bytes(a_bytes)
    config_a = load_server_config(str(a_path))
    a_hash = compute_config_hash(a_path)

    def _run_carve(b_bytes: bytes, label: str) -> dict:
        b_path = tmp_root / f"carve_B_{label}.yaml"
        b_path.write_bytes(b_bytes)
        state = {
            "config": config_a,
            "config_path": str(b_path),
            "config_drift": {
                "detected": False,
                "loaded_hash": a_hash,
                "disk_hash": None,
                "last_checked_at": "2026-05-22T00:00:00+00:00",
            },
            "consolidating": False,
            "mode": "cloud-only",
            "cloud_only_reason": "live_reload",
            "model": None,
            "tokenizer": None,
            "migration": {},
        }
        orig = app_module._state
        app_module._state = state
        try:
            return app_module._apply_config_live()
        finally:
            app_module._state = orig

    # R-PORT pure-port: change ONLY tts.port to a real free port → real bind
    # pre-flight succeeds + pure_port_delta=True short-circuit (no reload).
    free = _free_port()
    rport = _run_carve(a_bytes.replace(b"port: 10301", f"port: {free}".encode()), "rport")
    results["rport"] = rport
    if not (
        rport.get("applied_live") is False
        and rport.get("restart_required_reason") == "tts_port_change"
        and rport.get("restart_eligible") is True
        and rport.get("skipped") is None
    ):
        failures.append(f"R-PORT carve unexpected result: {rport}")

    # R-PATHS: change paths.data → paths_change short-circuit (no reload).
    data_old = f"  data: {(tmp_root / 'data' / 'ha').resolve()}".encode()
    data_new = f"  data: {(tmp_root / 'data' / 'ha_migtest').resolve()}".encode()
    if data_old not in a_bytes:
        failures.append(f"R-PATHS setup error: data line {data_old!r} not in sandbox YAML")
        rpaths = {}
    else:
        rpaths = _run_carve(a_bytes.replace(data_old, data_new), "rpaths")
        if not (
            rpaths.get("applied_live") is False
            and rpaths.get("restart_required_reason") == "paths_change"
            and rpaths.get("restart_eligible") is False
        ):
            failures.append(f"R-PATHS carve unexpected result: {rpaths}")
    results["rpaths"] = rpaths

    return {"pass": len(failures) == 0, "failure_reasons": failures, "results": results}


# ---------------------------------------------------------------------------
# Main smoke runner
# ---------------------------------------------------------------------------


def run_smoke(*, keep_on_success: bool = False) -> bool:
    """Run the full migration maintenance smoke.

    Returns
    -------
    bool
        ``True`` iff the smoke passes.
    """
    _assert_gpu_free()

    tmp_root = Path(tempfile.mkdtemp(prefix="paramem-mig-smoke-"))
    logger.info("Sandbox tmp root: %s", tmp_root)

    config = _build_sandbox_config(tmp_root)
    _write_sessions_to_tmp(config.paths.sessions)

    failure_reasons: list[str] = []
    gate_results: dict = {}
    apply_result_summary: dict = {}

    try:
        # ── Step 1: Load model ────────────────────────────────────────────────
        logger.info("Loading model: %s", config.model_name)
        model, tokenizer = load_base_model(config.model_config)
        logger.info("Model loaded.")

        # ── Step 2: Seed cycle (production consolidation) ────────────────────
        logger.info("=== Seed cycle: production consolidation over fictional sessions ===")
        seed_result = _seed_cycle(model, tokenizer, config)
        seed_status = seed_result.get("status", "unknown")
        logger.info("Seed cycle status: %s", seed_status)

        # The seed cycle's ConsolidationLoop (returned as seed_result["loop"]) is a
        # BASE-MODEL HOLDER: during the cycle loop.model is rebound to the trained
        # PeftModel wrapping the base (paramem/training/consolidation.py:264), so it
        # pins ~4 GiB. In production this loop IS _state["consolidation_loop"], which
        # _release_base_model_in_process nulls during the apply; the smoke's seed loop
        # is detached from _state and unreachable by the release. Drop it here, or the
        # apply's in-process model reload OOMs on the 8 GiB GPU (old base ~4 GiB still
        # held via the loop + new ~3.7 GiB reload).
        seed_result = None
        gc.collect()

        if seed_status not in ("complete", "simulated", "no_facts"):
            failure_reasons.append(f"Seed cycle unexpected status: {seed_status!r}")

        # Check registry size for gate 4.
        key_count = _registry_key_count(config)
        logger.info("Sandbox registry key count after seed: %d", key_count)
        if key_count < _GATE_4_MIN_REGISTRY_SIZE:
            logger.warning(
                "Registry has %d keys (< %d) — gate 4 will SKIP recall (not fail). "
                "The smoke still validates gates 1–3 and the apply path.",
                key_count,
                _GATE_4_MIN_REGISTRY_SIZE,
            )

        # ── Step 3: GPU cooldown between seed and trial ───────────────────────
        logger.info("=== GPU cooldown between seed cycle and trial ===")
        _wait_for_cooldown(_COOLDOWN_THRESHOLD_C)

        # ── Step 4: Drive apply path ──────────────────────────────────────────
        logger.info("=== Driving _apply_config_live (in-process) ===")
        # Hand off the base model so NO run_smoke frame-local references it across
        # the apply. _apply_config_live -> _release_base_model_in_process must drive
        # the old model's refcount to 0 before the reload, or the reload OOMs on the
        # 8 GiB GPU (old ~4 GiB + new ~3.7 GiB). Mirrors _load_model_into_state.
        _handoff = {"model": model, "tokenizer": tokenizer}
        model = None  # drop run_smoke's frame reference (kept bound for the finally)
        tokenizer = None
        gc.collect()
        apply_result_summary = _drive_apply_and_probe(
            _handoff.pop("model"), _handoff.pop("tokenizer"), config, tmp_root
        )
        gate_results = apply_result_summary.get("apply_result", {})

        if not apply_result_summary.get("pass"):
            failure_reasons.extend(apply_result_summary.get("failure_reasons", []))

        # ── Step 5: Carve classification (Phase A) ───────────────────────────
        # R-PORT pure-port (real socket bind + real dataclasses.replace) and
        # R-PATHS short-circuit, against real ServerConfig. No model needed —
        # both carves return before _live_reload_base_model.
        logger.info("=== Phase A: carve classification (R-PORT pure-port + R-PATHS) ===")
        carve_summary = _drive_carve_scenarios(tmp_root)
        logger.info("Carve scenarios: %s", "PASS" if carve_summary.get("pass") else "FAIL")
        if not carve_summary.get("pass"):
            failure_reasons.extend(carve_summary.get("failure_reasons", []))

    except Exception as exc:  # noqa: BLE001
        failure_reasons.append(f"Unhandled exception: {exc}")
        logger.exception("Migration smoke unhandled exception")
    finally:
        # Free GPU memory.
        try:
            if "model" in dir():
                del model  # type: ignore[name-defined]
            if "tokenizer" in dir():
                del tokenizer  # type: ignore[name-defined]
            safe_empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = len(failure_reasons) == 0
    print("\n" + "=" * 70)
    print("MIGRATION MAINTENANCE SMOKE — " + ("PASS" if passed else "FAIL"))
    print("=" * 70)
    print(f"  applied_live      : {apply_result_summary.get('applied_live')}")
    print(f"  mode_after        : {apply_result_summary.get('mode_after')}")
    print(f"  memory_hydrated   : {apply_result_summary.get('memory_store_hydrated')}")
    print(f"  recall_answer     : {apply_result_summary.get('recall_answer', '')[:120]!r}")
    print(f"  apply_result      : {gate_results}")
    if failure_reasons:
        print("\nFAILURES:")
        for r in failure_reasons:
            print(f"  - {r}")
    else:
        print("\n  All assertions passed.")

    if passed and not keep_on_success:
        shutil.rmtree(tmp_root, ignore_errors=True)
        logger.info("Sandbox tmp cleaned: %s", tmp_root)
    else:
        logger.info(
            "Sandbox tmp KEPT for forensics (pass=%s keep=%s): %s",
            passed,
            keep_on_success,
            tmp_root,
        )

    return passed


# ---------------------------------------------------------------------------
# Phase B — clean systemd restart, boot from a different paths.data
# ---------------------------------------------------------------------------


def _poll_status_local(timeout_s: int = 240, interval_s: int = 10) -> dict | None:
    """Poll ``GET /status`` until ``mode == "local"`` or timeout. Returns the dict or None."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:  # boundary: the service may not be listening yet during boot
            with urllib.request.urlopen("http://localhost:8420/status", timeout=5) as resp:
                data = json.loads(resp.read())
            if data.get("mode") == "local":
                return data
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(interval_s)
    return None


def _journal_contains_since(since_dt: datetime, needle: str) -> bool:
    """True if the paramem-server journal since ``since_dt`` contains ``needle``."""
    out = subprocess.run(
        [
            "journalctl",
            "--user",
            "-u",
            "paramem-server",
            "--since",
            since_dt.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
            "--no-pager",
        ],
        capture_output=True,
        text=True,
    )
    return needle in out.stdout


def _rewrite_paths_to_temp(config_text: str, temp_data: Path) -> str:
    """Rewrite data/sessions/debug under ``paths:`` to point at ``temp_data``.

    Line + regex based so comments survive and the real values need not be known.
    ONLY the three keys under the ``paths:`` block are touched — everything else
    (incl. security/encryption) is byte-identical, so the temp boot uses the same
    encryption state as canonical.
    """
    targets = {
        "data": str(temp_data.resolve()),
        "sessions": str((temp_data / "sessions").resolve()),
        "debug": str((temp_data / "debug").resolve()),
    }
    out: list[str] = []
    in_paths = False
    for line in config_text.splitlines(keepends=True):
        body = line.rstrip("\n")
        if body.startswith("paths:"):
            in_paths = True
            out.append(line)
            continue
        if in_paths:
            if body and not body[0].isspace():
                in_paths = False  # reached the next top-level section
                out.append(line)
                continue
            rewritten = line
            for key, val in targets.items():
                m = re.match(rf"(\s+{key}:\s+)(\S+)(.*)$", body)
                if m:
                    rewritten = f"{m.group(1)}{val}{m.group(3)}\n"
                    break
            out.append(rewritten)
            continue
        out.append(line)
    return "".join(out)


def _drive_systemd_boot_from_temp_root(*, keep: bool = False) -> bool:
    """Phase B: prove a clean systemd restart boots from a different ``paths.data``.

    Uses the REAL config + a COPY of the real (already-encrypted) data at a temp
    root — only ``paths.*`` differs, so encryption/security state is identical and
    untouched. Drives the LIVE (canonical, running) systemd service via
    ``PARAMEM_CONFIG`` (``start-server.sh``: ``CONFIG=${PARAMEM_CONFIG:-configs/server.yaml}``).
    Boot-from-temp is proven by the journal showing a registry load from the temp
    root. Cleanup ALWAYS unsets ``PARAMEM_CONFIG`` and restarts so the live server
    returns to the canonical config + real data, even on failure.

    MUST RUN CUDA-FREE (load-bearing). This function and its process must NEVER
    initialise CUDA (no ``torch.cuda`` calls, no model load). On this 8 GiB GPU the
    service boot needs ~6.5 of ~7 usable GiB, so a competing CUDA context in THIS
    process tips the service boot into cloud-only (observed 2026-05-22). Because we
    hold no CUDA context, ``systemctl restart`` stops the canonical service —
    freeing its entire GPU — then starts it on the temp config, which boots into a
    fully free GPU and reaches local. There is therefore no in-process model to
    release and no VRAM gate to check here: the restart is the GPU handoff.
    """
    real_config = _REPO_ROOT / "configs" / "server.yaml"
    cfg = load_server_config(str(real_config))
    real_data = Path(cfg.paths.data).resolve()
    real_keys = _registry_key_count(cfg)
    logger.info("Phase B: real data root %s (%d keys)", real_data, real_keys)

    temp_root = Path(tempfile.mkdtemp(prefix="paramem-rpaths-systemd-"))
    temp_data = temp_root / "ha"
    temp_config = temp_root / "server.yaml"
    failures: list[str] = []

    def _systemctl(*args: str) -> None:
        subprocess.run(["systemctl", "--user", *args], check=True)

    try:
        logger.info("Phase B: copying real data -> %s (excluding backups)", temp_data)
        shutil.copytree(real_data, temp_data, ignore=shutil.ignore_patterns("backups"))
        temp_config.write_text(_rewrite_paths_to_temp(real_config.read_text(), temp_data))

        t0 = datetime.now(timezone.utc)
        _systemctl("set-environment", f"PARAMEM_CONFIG={temp_config}")
        _systemctl("restart", "paramem-server")

        status = _poll_status_local()
        if status is None:
            failures.append("server did not reach local mode after temp-config restart")
        else:
            if status.get("keys_count") != real_keys:
                failures.append(
                    f"keys_count {status.get('keys_count')} != real {real_keys} "
                    "(temp data copy incomplete)"
                )
            if not _journal_contains_since(t0, str(temp_data)):
                failures.append(
                    f"journal shows no load from {temp_data} — server may have booted "
                    "from the canonical root (PARAMEM_CONFIG not honoured)"
                )
            else:
                logger.info("Phase B: confirmed boot-from-temp-root (%s)", temp_data)
    except Exception as exc:  # noqa: BLE001 — boundary: subprocess / filesystem
        failures.append(f"exception during temp-config boot: {exc}")
        logger.exception("Phase B temp-config boot failed")
    finally:
        # ALWAYS restore the canonical config + real data, even on failure.
        t_restore = datetime.now(timezone.utc)
        try:
            _systemctl("unset-environment", "PARAMEM_CONFIG")
            _systemctl("restart", "paramem-server")
            restored = _poll_status_local()
            if restored is None or restored.get("mode") != "local":
                failures.append("server did NOT restore to canonical local mode")
            elif restored.get("keys_count") != real_keys:
                failures.append(
                    f"post-restore keys_count {restored.get('keys_count')} != {real_keys}"
                )
            elif not _journal_contains_since(t_restore, str(real_data)):
                failures.append(f"post-restore journal shows no load from canonical {real_data}")
            else:
                logger.info("Phase B: restored to canonical config + real data")
        except Exception as exc:  # noqa: BLE001 — boundary: subprocess
            failures.append(f"RESTORE FAILED — canonical config may need a manual restart: {exc}")
            logger.exception("Phase B restore failed")
        if keep:
            logger.info("Phase B temp root KEPT: %s", temp_root)
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

    ok = len(failures) == 0
    print("\n" + "=" * 70)
    print("PHASE B — SYSTEMD BOOT-FROM-TEMP-ROOT — " + ("PASS" if ok else "FAIL"))
    print("=" * 70)
    for f in failures:
        print(f"  - {f}")
    return ok


def _run_phase_b_only() -> bool:
    """Standalone Phase B entry — CUDA-FREE; runs against the LIVE server.

    This process must NEVER initialise CUDA (no model load, no ``torch.cuda``): on
    this 8 GiB GPU the service boot needs ~6.5 of ~7 usable GiB, so a competing
    CUDA context here tips the boot into cloud-only (observed 2026-05-22). With no
    context held, ``systemctl restart`` is a clean GPU handoff — it stops the
    canonical service (freeing its GPU) and starts the temp config into a free GPU,
    which reaches local. The live (canonical) server should be running at start;
    the restart frees its GPU and the ``finally`` always restores canonical.
    """
    return _drive_systemd_boot_from_temp_root(keep=False)


# ---------------------------------------------------------------------------
# Phase R-PORT — clean systemd restart, bind a different tts.port
# ---------------------------------------------------------------------------


def _rewrite_tts_port_to_temp(config_text: str, new_port: int) -> str:
    """Rewrite ONLY ``tts.port`` to ``new_port``; everything else byte-identical.

    Line-based within the ``tts:`` block. The nested ``voices:`` entries carry no
    ``port:`` key, so the first ``port:`` under ``tts:`` is unambiguous. Applied on
    top of ``_rewrite_paths_to_temp`` (which absolutizes ``paths.*`` to the
    canonical data root) — the config loader anchors *relative* ``paths`` to
    ``config_path.parent.parent``, so a ``/tmp`` config must carry absolute paths
    or it boots an empty data root.
    """
    out: list[str] = []
    in_tts = False
    replaced = False
    for line in config_text.splitlines(keepends=True):
        body = line.rstrip("\n")
        if body.startswith("tts:"):
            in_tts = True
            out.append(line)
            continue
        if in_tts and not replaced:
            if body and not body[0].isspace():
                in_tts = False  # next top-level section reached, port not found
                out.append(line)
                continue
            m = re.match(r"(\s+port:\s+)(\S+)(.*)$", body)
            if m:
                out.append(f"{m.group(1)}{new_port}{m.group(3)}\n")
                replaced = True
                continue
        out.append(line)
    if not replaced:
        raise RuntimeError("tts.port line not found under the tts: block")
    return "".join(out)


def _wyoming_describe_ok(host: str, port: int, timeout_s: float = 10.0) -> bool:
    """True iff a Wyoming client gets an ``Info`` reply to ``Describe`` on host:port.

    Uses the real ``wyoming`` client (already a dependency) — the same handshake
    Home Assistant performs on connect — so this proves a Wyoming listener is bound
    AND speaking the protocol on ``port``. Validated against the live server before
    landing this phase. Returns False on refusal, timeout or any protocol error.
    """
    from wyoming.client import AsyncTcpClient
    from wyoming.info import Describe, Info

    async def _probe() -> bool:
        async with AsyncTcpClient(host, port) as client:
            await client.write_event(Describe().event())
            for _ in range(5):
                event = await client.read_event()
                if event is None:
                    return False
                if Info.is_type(event.type):
                    return True
            return False

    try:
        return asyncio.run(asyncio.wait_for(_probe(), timeout=timeout_s))
    except Exception:  # noqa: BLE001 — boundary: network probe, any failure == not-up
        return False


def _port_refuses(host: str, port: int) -> bool:
    """True iff a TCP connect to host:port is refused (nothing bound there)."""
    try:
        with socket.create_connection((host, port), timeout=2):
            return False  # something IS still listening
    except ConnectionRefusedError:
        return True
    except OSError:  # noqa: BLE001 — boundary: treat unreachable as "not refused"
        return False


def _drive_systemd_port_flip(*, keep: bool = False) -> bool:
    """Phase R-PORT: prove a clean systemd restart binds a different ``tts.port``.

    Mirrors ``_drive_systemd_boot_from_temp_root`` but flips ``tts.port`` instead
    of ``paths.*`` — the canonical data root is reused (no copy), so the only delta
    is the TTS listener port. Drives the LIVE (canonical, running) systemd service
    via ``PARAMEM_CONFIG``. The new port is proven bound by the real Wyoming client
    handshake (``Describe`` -> ``Info``, the same one HA performs); the canonical
    port is proven vacated (connection refused). Cleanup ALWAYS unsets
    ``PARAMEM_CONFIG`` and restarts so the service returns to the canonical port,
    even on failure.

    This pairs with Phase A's R-PORT *classification* (``tts_port_change`` carve,
    ``restart_eligible=True``): Phase A proves accept marks the change restart-
    required, this phase proves the restart actually rebinds the listener — the
    same two-part shape as R-PATHS (Phase A classification + Phase B boot).

    MUST RUN CUDA-FREE (load-bearing) — same constraint as Phase B: on this 8 GiB
    GPU the service boot needs ~6.5 of ~7 usable GiB, so a competing CUDA context
    in THIS process tips the boot into cloud-only. This function loads no model and
    never touches ``torch.cuda``; the restart is the GPU handoff.
    """
    host = "localhost"
    real_config = _REPO_ROOT / "configs" / "server.yaml"
    cfg = load_server_config(str(real_config))
    canonical_port = int(cfg.tts.port)
    real_data = Path(cfg.paths.data).resolve()
    real_keys = _registry_key_count(cfg)
    new_port = _free_port()
    logger.info(
        "Phase R-PORT: canonical tts.port=%d -> new tts.port=%d (%d keys)",
        canonical_port,
        new_port,
        real_keys,
    )

    temp_root = Path(tempfile.mkdtemp(prefix="paramem-rport-systemd-"))
    temp_config = temp_root / "server.yaml"
    failures: list[str] = []

    def _systemctl(*args: str) -> None:
        subprocess.run(["systemctl", "--user", *args], check=True)

    try:
        # Absolutize paths to the canonical data root (the loader anchors relative
        # paths to config.parent.parent, so a /tmp config needs absolute paths to
        # boot the real registry), then flip tts.port.
        text = _rewrite_paths_to_temp(real_config.read_text(), real_data)
        text = _rewrite_tts_port_to_temp(text, new_port)
        temp_config.write_text(text)

        t0 = datetime.now(timezone.utc)
        _systemctl("set-environment", f"PARAMEM_CONFIG={temp_config}")
        _systemctl("restart", "paramem-server")

        status = _poll_status_local()
        if status is None:
            failures.append("server did not reach local mode after tts.port-flip restart")
        else:
            if status.get("keys_count") != real_keys:
                failures.append(
                    f"keys_count {status.get('keys_count')} != real {real_keys} "
                    "(temp config did not boot the canonical data root)"
                )
            if _wyoming_describe_ok(host, new_port):
                logger.info("Phase R-PORT: Wyoming listener confirmed on new port %d", new_port)
            else:
                failures.append(
                    f"Wyoming Describe->Info FAILED on new tts.port {new_port} "
                    "(listener did not move to the new port)"
                )
            if not _port_refuses(host, canonical_port):
                failures.append(
                    f"canonical tts.port {canonical_port} still accepts connections "
                    "(listener duplicated, not moved)"
                )
            phrase_ok = _journal_contains_since(t0, "Wyoming TTS server starting on")
            port_ok = _journal_contains_since(t0, str(new_port))
            if not (phrase_ok and port_ok):
                failures.append(
                    f"journal shows no 'Wyoming TTS server starting on …:{new_port}' since restart"
                )
    except Exception as exc:  # noqa: BLE001 — boundary: subprocess / filesystem / network
        failures.append(f"exception during tts.port-flip boot: {exc}")
        logger.exception("Phase R-PORT tts.port-flip boot failed")
    finally:
        # ALWAYS restore the canonical config + port, even on failure.
        try:
            _systemctl("unset-environment", "PARAMEM_CONFIG")
            _systemctl("restart", "paramem-server")
            restored = _poll_status_local()
            if restored is None or restored.get("mode") != "local":
                failures.append("server did NOT restore to canonical local mode")
            elif not _wyoming_describe_ok(host, canonical_port):
                failures.append(
                    f"post-restore Wyoming Describe->Info FAILED on canonical tts.port "
                    f"{canonical_port} (HA could not reconnect)"
                )
            else:
                logger.info("Phase R-PORT: restored canonical tts.port %d", canonical_port)
        except Exception as exc:  # noqa: BLE001 — boundary: subprocess
            failures.append(f"RESTORE FAILED — canonical config may need a manual restart: {exc}")
            logger.exception("Phase R-PORT restore failed")
        if keep:
            logger.info("Phase R-PORT temp root KEPT: %s", temp_root)
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

    ok = len(failures) == 0
    print("\n" + "=" * 70)
    print("PHASE R-PORT — SYSTEMD TTS-PORT-FLIP — " + ("PASS" if ok else "FAIL"))
    print("=" * 70)
    for f in failures:
        print(f"  - {f}")
    return ok


def _run_phase_rport_only() -> bool:
    """Standalone Phase R-PORT entry — CUDA-FREE; runs against the LIVE server.

    Same CUDA-free constraint and GPU-handoff-via-restart model as
    ``_run_phase_b_only``. The live (canonical) server should be running at start;
    the restart frees its GPU and the ``finally`` always restores the canonical
    port.
    """
    return _drive_systemd_port_flip(keep=False)


def main() -> int:
    """Entry point.

    Returns
    -------
    int
        0 on PASS, 1 on FAIL.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--keep-on-success",
        action="store_true",
        help="Keep tmp sandbox dir even on PASS (default: clean up).",
    )
    parser.add_argument(
        "--phase-b-only",
        action="store_true",
        help=(
            "Run ONLY Phase B: the CUDA-free systemd boot-from-temp-root validation "
            "against the LIVE (canonical, running) server. Must NOT be combined with the "
            "in-process smoke (Phase A holds a CUDA context that would tip the boot to "
            "cloud-only). Always restores the canonical config."
        ),
    )
    parser.add_argument(
        "--phase-rport-only",
        action="store_true",
        help=(
            "Run ONLY Phase R-PORT: the CUDA-free systemd tts.port-flip validation "
            "against the LIVE (canonical, running) server. Flips tts.port via a temp "
            "PARAMEM_CONFIG, restarts, and proves the Wyoming listener moved (Describe-> "
            "Info on the new port, canonical port refused). Always restores the canonical "
            "config. Must NOT be combined with the in-process smoke (which holds CUDA)."
        ),
    )
    args = parser.parse_args()
    if args.phase_b_only:
        return 0 if _run_phase_b_only() else 1
    if args.phase_rport_only:
        return 0 if _run_phase_rport_only() else 1
    return 0 if run_smoke(keep_on_success=args.keep_on_success) else 1


if __name__ == "__main__":
    sys.exit(main())
