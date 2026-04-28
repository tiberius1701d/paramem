#!/usr/bin/env python3
"""End-to-end artifact-layout smoke for paramem.

Runs ``run_consolidation`` in-process per (mode, debug) combination and
verifies that the right files land at the right paths with the right
encryption posture. Mirrors the pattern in
``scripts/dev/verify_pr1_extraction.py``: a non-mutating
``_FilteredSessionBuffer`` injects pre-built synthetic sessions, the
real extraction model runs on the GPU, and the production consolidation
path writes its artifacts under per-scenario isolated data directories.

Scenarios
---------
    A: mode=simulate, debug=false
    B: mode=simulate, debug=true
    C: mode=train,    debug=false
    D: mode=train,    debug=true

Per scenario the harness:
    1. wait_for_cooldown 52 (between scenarios)
    2. fresh per-scenario tmp data_dir
    3. load_server_config + per-scenario overrides
    4. fresh model load (state isolation across scenarios)
    5. inject five synthetic fictional-entity sessions via
       ``_FilteredSessionBuffer``
    6. run_consolidation(model, tokenizer, config, buffer)
    7. walk the expected/forbidden manifest, assert each file
    8. on pass: rm -rf scenario data_dir
       on fail: keep + log path for forensics

What this verifies
------------------
* Artifact path layout under each (mode, debug) combination.
* Encryption posture at rest:
    - Security ON  → production artifacts are age-encrypted; debug stays
      plaintext.
    - Security OFF → production artifacts are plaintext (master switch);
      debug stays plaintext.
* Forbidden patterns (e.g. simulate-mode never writes adapter weights;
  no stale ``cycle_*/`` dirs under ``paths.simulate``).
* Atomic-write discipline: no ``.tmp`` residue under any scenario dir.

What this does NOT verify
-------------------------
* Recall quality. ``--max-epochs 2`` produces artifacts but trains far
  below the validated 30-epoch floor. Use only for layout asserts.
* Cross-cycle behaviour. Each scenario is a single fresh cycle; no
  promotion-threshold counting, no interim → main merge.
* Real conversation extraction quality. Sessions are deterministic
  synthetic transcripts about fictional entities (Alex / Lila /
  Heilbronn / Sapphire Café / Forest Brook). Treat output as
  layout-shaped, not topic-realistic.
* Multi-host networking, HA integration, voice path.

Prerequisites
-------------
* GPU available. Real consolidation requires the LoRA training stack.
* Source the full ``.env`` (or otherwise export the operator env)
  before invocation. The harness always preserves every env var so
  the extraction pipeline (SOTA enrichment, HA validation, prompts)
  runs identically across Security ON and OFF runs.
* For Security ON (default): daily age identity at
  ``~/.config/paramem/daily_key.age`` and ``PARAMEM_DAILY_PASSPHRASE``
  exported.
* For Security OFF (``--security-off``): the harness pops
  ``PARAMEM_DAILY_PASSPHRASE`` from the in-process env and clears the
  daily-identity cache before any scenario runs. Every OTHER env var
  stays set. Result: ON and OFF runs are 1:1 comparable on every
  pipeline step except the encryption-at-rest layer.

Usage
-----
    set -a && source .env && set +a

    # Security ON, full validated 30-epoch budget (multi-hour)
    python scripts/dev/e2e_artifact_smoke.py

    # Security OFF, fast layout-only run (~30-60 min)
    python scripts/dev/e2e_artifact_smoke.py --security-off --max-epochs 2

    # Subset of scenarios, keep dirs on pass
    python scripts/dev/e2e_artifact_smoke.py --scenarios A,C --keep-on-success

Exit code 0 iff every requested scenario passes.

Wall time
---------
Dominated by per-scenario model load (~5-10s on a warm cache, longer
cold) and LoRA training in train-mode scenarios. Training time scales
linearly with ``consolidation.max_epochs``: full validated 30-epoch
runs take many hours per train scenario; ``--max-epochs 2`` cuts
training to a few minutes per train scenario.

Never copy ``--max-epochs N<30`` into a production YAML without
empirical recall-vs-epochs validation on the active base model. The
30-epoch default is anchored on Mistral 7B per the Test 1-8 campaign
and is the floor for 100% indexed-key recall on that model.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Allow importing paramem from the project root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from paramem.backup.age_envelope import is_age_envelope  # noqa: E402
from paramem.backup.key_store import (  # noqa: E402
    DAILY_KEY_PATH_DEFAULT,
    DAILY_PASSPHRASE_ENV_VAR,
    _clear_daily_identity_cache,
    daily_identity_loadable,
)
from paramem.models.loader import load_base_model  # noqa: E402
from paramem.server.config import load_server_config  # noqa: E402
from paramem.server.consolidation import run_consolidation  # noqa: E402
from paramem.server.session_buffer import SessionBuffer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
for _noisy in ("httpx", "anthropic", "urllib3", "transformers", "accelerate", "bitsandbytes"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger("e2e_artifact_smoke")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COOLDOWN_THRESHOLD_C = 52
GPU_COOLDOWN_SCRIPT = Path("/home/tiberius/.local/bin/gpu-cooldown.sh")
PRODUCTION_CONFIG = _REPO_ROOT / "configs" / "server.yaml"
SMOKE_SPEAKER_ID = "smoke-spk-001"

# ---------------------------------------------------------------------------
# Synthetic test sessions — pre-formatted transcripts, fictional entities
# ---------------------------------------------------------------------------

# Each entry is the dict shape ``run_consolidation`` consumes via
# ``buffer.get_pending()``: session_id, transcript (already formatted as a
# multi-line string), speaker_id, source_type, doc_title.
TEST_SESSIONS: list[dict[str, Any]] = [
    {
        "session_id": "smoke00",
        "speaker_id": SMOKE_SPEAKER_ID,
        "transcript": (
            "User: I met Alex at the Sapphire Café on Forest Brook today.\n"
            "Assistant: Got it — Alex at the Sapphire Café on Forest Brook.\n"
            "User: Alex lives in Heilbronn now.\n"
            "Assistant: Noted. Alex is based in Heilbronn.\n"
            "User: Sapphire Café opens at 8 every morning.\n"
            "Assistant: Stored. Sapphire Café opens at 8."
        ),
        "source_type": "transcript",
        "doc_title": None,
    },
    {
        "session_id": "smoke01",
        "speaker_id": SMOKE_SPEAKER_ID,
        "transcript": (
            "User: Lila started a new job at the Forest Brook archive.\n"
            "Assistant: Lila now works at the Forest Brook archive.\n"
            "User: She bikes to work most days.\n"
            "Assistant: Got it — Lila bikes to the archive."
        ),
        "source_type": "transcript",
        "doc_title": None,
    },
    {
        "session_id": "smoke02",
        "speaker_id": SMOKE_SPEAKER_ID,
        "transcript": (
            "User: Alex prefers black coffee, no sugar.\n"
            "Assistant: Noted. Alex: black coffee, no sugar.\n"
            "User: Lila prefers green tea in the afternoon.\n"
            "Assistant: Stored. Lila: green tea in the afternoon."
        ),
        "source_type": "transcript",
        "doc_title": None,
    },
    {
        "session_id": "smoke03",
        "speaker_id": SMOKE_SPEAKER_ID,
        "transcript": (
            "User: Heilbronn hosts a small jazz festival every September.\n"
            "Assistant: Heilbronn → annual jazz festival in September.\n"
            "User: Alex usually attends.\n"
            "Assistant: Noted. Alex attends the Heilbronn jazz festival."
        ),
        "source_type": "transcript",
        "doc_title": None,
    },
    {
        "session_id": "smoke04",
        "speaker_id": SMOKE_SPEAKER_ID,
        "transcript": (
            "User: Forest Brook has a new running trail along the river.\n"
            "Assistant: Got it — new running trail in Forest Brook.\n"
            "User: Lila plans to run it on weekends.\n"
            "Assistant: Stored. Lila → weekend running on the Forest Brook trail."
        ),
        "source_type": "transcript",
        "doc_title": None,
    },
]


# ---------------------------------------------------------------------------
# Filtered session buffer — non-mutating wrapper, mirrors verify_pr1
# ---------------------------------------------------------------------------


class _FilteredSessionBuffer(SessionBuffer):
    """Non-mutating buffer that returns a pre-built session list.

    ``get_pending()`` returns the injected sessions; ``mark_consolidated()``
    is a no-op (the smoke does not retire any production state). Pattern
    copied from ``scripts/dev/verify_pr1_extraction.py``.
    """

    def __init__(self, session_dir: Path, sessions: list[dict[str, Any]]):
        super().__init__(session_dir=session_dir, debug=False)
        self._smoke_sessions = sessions

    def get_pending(self) -> list[dict[str, Any]]:
        return list(self._smoke_sessions)

    def mark_consolidated(self, session_ids: list[str]) -> None:
        logger.info(
            "mark_consolidated suppressed for %d session(s) (smoke harness — no mutation)",
            len(session_ids),
        )


# ---------------------------------------------------------------------------
# Manifest model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Expectation:
    """One artifact the harness expects (or forbids) per scenario."""

    relpath: str  # relative to data_dir
    encrypted: bool | None  # True = age envelope, False = plaintext, None = absence
    glob: bool = False  # True = relpath is a glob pattern, match any → presence


@dataclass
class Scenario:
    """One (mode, debug) combination + its full manifest."""

    id: str
    mode: str
    debug: bool
    expected: list[Expectation] = field(default_factory=list)
    forbidden: list[Expectation] = field(default_factory=list)


def _build_scenarios(*, security_off: bool) -> list[Scenario]:
    # Train-mode slot dirs are timestamped (e.g. 20260427-105338); manifest
    # uses globs to match any. Forbidden assertions in simulate mode use a
    # double-glob so ANY adapter weight slot trips the check.
    #
    # Security ON  → production artifacts are age-encrypted; debug stays plaintext.
    # Security OFF → production artifacts are plaintext (master switch); debug stays plaintext.
    # The encryption posture for production assertions inverts based on the
    # security mode; debug carve-out assertions stay plaintext in both modes.
    enc_prod = not security_off  # True under ON, False under OFF

    common_train_artifacts = [
        Expectation("adapters/episodic/*/adapter_model.safetensors", encrypted=enc_prod, glob=True),
        Expectation("adapters/episodic/*/adapter_config.json", encrypted=False, glob=True),
        Expectation("adapters/episodic/*/meta.json", encrypted=False, glob=True),
        Expectation("adapters/episodic/keyed_pairs.json", encrypted=enc_prod),
        Expectation("registry/key_metadata.json", encrypted=enc_prod),
    ]

    common_simulate_artifacts = [
        Expectation("simulate/episodic/keyed_pairs.json", encrypted=enc_prod),
        Expectation("registry/key_metadata.json", encrypted=enc_prod),
    ]

    debug_artifacts = [
        Expectation("debug/cycle_*/graph_snapshot.json", encrypted=False, glob=True),
        Expectation("debug/cycle_*/episodic_qa_snapshot.json", encrypted=False, glob=True),
    ]

    forbidden_train_weights = Expectation(
        "adapters/*/*/adapter_model.safetensors", encrypted=None, glob=True
    )

    return [
        Scenario(
            id="A",
            mode="simulate",
            debug=False,
            expected=list(common_simulate_artifacts),
            forbidden=[
                forbidden_train_weights,
                Expectation("debug/cycle_*", encrypted=None, glob=True),
                Expectation("simulate/cycle_*", encrypted=None, glob=True),
            ],
        ),
        Scenario(
            id="B",
            mode="simulate",
            debug=True,
            expected=list(common_simulate_artifacts) + list(debug_artifacts),
            forbidden=[
                forbidden_train_weights,
                Expectation("simulate/cycle_*", encrypted=None, glob=True),
            ],
        ),
        Scenario(
            id="C",
            mode="train",
            debug=False,
            expected=list(common_train_artifacts),
            forbidden=[
                Expectation("simulate/episodic/keyed_pairs.json", encrypted=None),
                Expectation("debug/cycle_*", encrypted=None, glob=True),
            ],
        ),
        Scenario(
            id="D",
            mode="train",
            debug=True,
            expected=list(common_train_artifacts) + list(debug_artifacts),
            forbidden=[
                Expectation("simulate/episodic/keyed_pairs.json", encrypted=None),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Pre-flight + cooldown
# ---------------------------------------------------------------------------


def _preflight(*, security_off: bool) -> None:
    """Pre-flight gates before any scenario starts.

    Security ON  (default): refuse if no daily identity loadable.
    Security OFF (--security-off): the smoke disables Security itself
    (pop ``PARAMEM_DAILY_PASSPHRASE``, clear the daily-identity cache).
    All other env (API keys, HA tokens, etc.) is preserved, so OFF runs
    pass through the same extraction pipeline as ON — only the
    encryption-at-rest layer differs.
    """
    if security_off:
        # Disable Security ON in-process, preserving every other env var.
        # This keeps OFF runs 1:1 comparable with ON: same SOTA enrichment,
        # same HA validation, same prompts; only encryption posture differs.
        os.environ.pop(DAILY_PASSPHRASE_ENV_VAR, None)
        _clear_daily_identity_cache()

    errors: list[str] = []
    loadable = daily_identity_loadable(DAILY_KEY_PATH_DEFAULT)
    if security_off:
        if loadable:
            errors.append(
                "Internal disable failed — daily identity still loadable after "
                f"unset({DAILY_PASSPHRASE_ENV_VAR}). The key may be reachable "
                "via a non-passphrase path (hardware-backed unlock?). Investigate "
                "the daily-key chain before running OFF; otherwise consolidation "
                "would encrypt under a stale key and the manifest would mismatch."
            )
    else:
        if not loadable:
            errors.append(
                f"Daily identity not loadable at {DAILY_KEY_PATH_DEFAULT}. "
                f"Set {DAILY_PASSPHRASE_ENV_VAR} and ensure the daily key file "
                "exists. Or run with --security-off to exercise the OFF posture."
            )
    if not GPU_COOLDOWN_SCRIPT.exists():
        errors.append(
            f"GPU cooldown helper not found at {GPU_COOLDOWN_SCRIPT}. "
            "The smoke depends on it for inter-scenario thermal discipline."
        )
    if errors:
        for msg in errors:
            print(f"PRE-FLIGHT ERROR: {msg}", file=sys.stderr)
        sys.exit(1)


def _wait_for_cooldown(threshold_c: int = COOLDOWN_THRESHOLD_C) -> None:
    logger.info("Cooldown: waiting for GPU ≤ %d°C", threshold_c)
    subprocess.run(
        ["bash", "-lc", f"source {GPU_COOLDOWN_SCRIPT} && wait_for_cooldown {threshold_c}"],
        check=True,
    )


# ---------------------------------------------------------------------------
# Per-scenario config override
# ---------------------------------------------------------------------------


def _apply_scenario_overrides(
    config,
    data_dir: Path,
    scenario: Scenario,
    *,
    security_off: bool,
    max_epochs: int | None,
) -> None:
    """Override paths, mode, debug, security posture, and epoch ceiling.

    The config is loaded once from the production YAML and mutated per
    scenario. paths are rerooted under ``data_dir``. ``max_epochs`` is
    forwarded to ``consolidation.max_epochs`` (the YAML knob added when
    this harness landed); when ``None`` the validated 30-epoch budget
    runs, which dominates wall time. ``security_off=True`` also clears
    ``security.require_encryption`` so the OFF-posture lifespan gate
    accepts the missing daily identity.
    """
    config.consolidation.mode = scenario.mode
    config.debug = scenario.debug
    # Disable the systemd-driven schedule; the harness drives consolidation.
    config.consolidation.refresh_cadence = ""

    if max_epochs is not None:
        config.consolidation.max_epochs = max_epochs

    if security_off:
        config.security.require_encryption = False

    config.paths.data = data_dir
    config.paths.sessions = data_dir / "sessions"
    config.paths.debug = data_dir / "debug"
    config.paths.simulate = data_dir / "simulate"


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@dataclass
class VerifyFailure:
    relpath: str
    reason: str


def _check_expectation(data_dir: Path, exp: Expectation) -> list[VerifyFailure]:
    failures: list[VerifyFailure] = []
    if exp.glob:
        matches = list(data_dir.glob(exp.relpath))
        if not matches:
            failures.append(VerifyFailure(exp.relpath, "no glob match"))
            return failures
        targets = [p for p in matches if p.is_file()]
    else:
        path = data_dir / exp.relpath
        if not path.exists():
            failures.append(VerifyFailure(exp.relpath, f"missing: {path}"))
            return failures
        targets = [path]

    for target in targets:
        if exp.encrypted is True:
            if not is_age_envelope(target):
                failures.append(
                    VerifyFailure(
                        str(target.relative_to(data_dir)),
                        "expected age envelope, got plaintext",
                    )
                )
        elif exp.encrypted is False:
            if is_age_envelope(target):
                failures.append(
                    VerifyFailure(
                        str(target.relative_to(data_dir)),
                        "expected plaintext, got age envelope",
                    )
                )
            else:
                if target.suffix == ".json":
                    try:
                        import json as _json

                        _json.loads(target.read_text(encoding="utf-8"))
                    except Exception as exc:  # noqa: BLE001
                        failures.append(
                            VerifyFailure(
                                str(target.relative_to(data_dir)),
                                f"expected plaintext JSON; parse failed: {exc}",
                            )
                        )
    return failures


def _check_forbidden(data_dir: Path, exp: Expectation) -> list[VerifyFailure]:
    failures: list[VerifyFailure] = []
    if exp.glob:
        matches = list(data_dir.glob(exp.relpath))
        if matches:
            failures.append(
                VerifyFailure(
                    exp.relpath,
                    f"unexpected presence: {[str(m.relative_to(data_dir)) for m in matches]}",
                )
            )
    else:
        path = data_dir / exp.relpath
        if path.exists():
            failures.append(VerifyFailure(exp.relpath, f"unexpected presence: {path}"))
    return failures


def _check_no_tmp_residue(data_dir: Path) -> list[VerifyFailure]:
    residues = list(data_dir.rglob("*.tmp"))
    return [
        VerifyFailure(str(p.relative_to(data_dir)), "stale .tmp residue (atomic-write incomplete)")
        for p in residues
    ]


def _verify_scenario(scenario: Scenario, data_dir: Path) -> list[VerifyFailure]:
    failures: list[VerifyFailure] = []
    for exp in scenario.expected:
        failures.extend(_check_expectation(data_dir, exp))
    for exp in scenario.forbidden:
        failures.extend(_check_forbidden(data_dir, exp))
    failures.extend(_check_no_tmp_residue(data_dir))
    return failures


# ---------------------------------------------------------------------------
# Per-scenario runner
# ---------------------------------------------------------------------------


def _run_scenario(
    scenario: Scenario,
    *,
    keep_on_success: bool,
    security_off: bool,
    max_epochs: int | None,
) -> list[VerifyFailure]:
    logger.info("=== Scenario %s: mode=%s debug=%s ===", scenario.id, scenario.mode, scenario.debug)
    _wait_for_cooldown()

    parent = Path(tempfile.mkdtemp(prefix=f"paramem-smoke-{scenario.id}-"))
    data_dir = parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Per-scenario config — load fresh from disk so cross-scenario mutations
    # do not bleed.
    config = load_server_config(str(PRODUCTION_CONFIG))
    _apply_scenario_overrides(
        config,
        data_dir,
        scenario,
        security_off=security_off,
        max_epochs=max_epochs,
    )

    # Per-scenario model load — required for correctness. The first scenario
    # wraps the base model into a PeftModel (adds peft_config as a side
    # effect on the underlying PreTrainedModel). A subsequent scenario
    # reusing the same base object would see peft_config, skip the re-wrap,
    # and end up with a PreTrainedModel-with-peft_config that fails
    # set_adapter at train time. Fresh-load + GC per scenario keeps state
    # clean. Cost: one model load per scenario; correctness > speed.
    logger.info("Loading model: %s (%s)", config.model_name, config.model_config.model_id)
    model, tokenizer = load_base_model(config.model_config)
    logger.info("Model loaded for scenario %s", scenario.id)

    buffer = _FilteredSessionBuffer(
        session_dir=config.session_dir,
        sessions=TEST_SESSIONS,
    )

    failures: list[VerifyFailure] = []
    started = time.monotonic()
    try:
        result = run_consolidation(
            model=model,
            tokenizer=tokenizer,
            config=config,
            session_buffer=buffer,
            loop=None,
            ha_context=None,
            speaker_store=None,
        )
        elapsed = time.monotonic() - started
        status = result.get("status")
        logger.info(
            "Scenario %s: consolidation status=%s in %.1fs",
            scenario.id,
            status,
            elapsed,
        )
        if status not in ("complete", "simulated", "no_facts", "no_pending"):
            failures.append(VerifyFailure("<runner>", f"unexpected consolidation status: {status}"))
        failures.extend(_verify_scenario(scenario, data_dir))
    except Exception as exc:  # noqa: BLE001
        failures.append(VerifyFailure("<runner>", f"exception: {exc}"))
        logger.exception("Scenario %s runner failed", scenario.id)
    finally:
        # Free GPU memory before the next scenario's model load.
        del model, tokenizer
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    if failures:
        logger.error(
            "Scenario %s: %d failure(s). Data dir kept at %s",
            scenario.id,
            len(failures),
            parent,
        )
        for f in failures:
            logger.error("  %s — %s", f.relpath, f.reason)
    else:
        if keep_on_success:
            logger.info("Scenario %s: PASS. Data dir kept at %s", scenario.id, parent)
        else:
            shutil.rmtree(parent, ignore_errors=True)
            logger.info("Scenario %s: PASS. Data dir cleaned.", scenario.id)
    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--scenarios",
        default="A,B,C,D",
        help="Comma-separated subset to run (default: A,B,C,D).",
    )
    parser.add_argument(
        "--keep-on-success",
        action="store_true",
        help="Keep per-scenario data dirs even on pass (default: cleanup).",
    )
    parser.add_argument(
        "--security-off",
        action="store_true",
        help=(
            "Run under Security OFF (no daily identity loaded). The manifest "
            "inverts encryption posture: production artifacts asserted "
            "plaintext instead of age-encrypted. Daily-key env var must be "
            "unset before invocation; pre-flight verifies."
        ),
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap LoRA training epochs via consolidation.max_epochs. None "
            "(default) uses the validated 30-epoch budget. Pass a small "
            "value (e.g. 2) to verify layout fast — does NOT validate "
            "recall quality."
        ),
    )
    args = parser.parse_args()

    _preflight(security_off=args.security_off)

    requested = {s.strip().upper() for s in args.scenarios.split(",") if s.strip()}
    all_scenarios = _build_scenarios(security_off=args.security_off)
    selected = [s for s in all_scenarios if s.id in requested]
    if len(selected) != len(requested):
        unknown = requested - {s.id for s in all_scenarios}
        print(f"Unknown scenarios: {unknown}", file=sys.stderr)
        return 2

    posture = "OFF" if args.security_off else "ON"
    epoch_note = (
        f"max_epochs={args.max_epochs}" if args.max_epochs is not None else "max_epochs=validated"
    )
    logger.info("Smoke run: security=%s, %s", posture, epoch_note)

    # Model is loaded fresh per scenario inside _run_scenario for state
    # isolation — see the comment in that function. Cost: one model load
    # per scenario plus consolidation (varies with max_epochs).
    results: dict[str, list[VerifyFailure]] = {}
    for scenario in selected:
        results[scenario.id] = _run_scenario(
            scenario,
            keep_on_success=args.keep_on_success,
            security_off=args.security_off,
            max_epochs=args.max_epochs,
        )

    print()
    print("=" * 60)
    failed = 0
    for sid, failures in results.items():
        status = "PASS" if not failures else f"FAIL ({len(failures)})"
        print(f"  Scenario {sid}: {status}")
        if failures:
            failed += 1
    print("=" * 60)
    print(f"Result: {len(results) - failed}/{len(results)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
