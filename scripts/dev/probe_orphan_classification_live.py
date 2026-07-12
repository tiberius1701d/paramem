"""Live-GPU probe: session classification at the interim-consolidation tick.

This is a live-GPU probe — NOT a pytest test (won't be collected by pytest).
It requires a free GPU and a stopped production server before running.

What it does
------------
Validates that ``_dispatch_consolidation`` (``paramem/server/app.py``)
routes pending sessions correctly via ``classify_session``
(``paramem/server/consolidation.py``):

  NAMED           -> extracted (graph_snapshot.json written in simulate mode)
  HOLDABLE        -> held pending on Tick 1; retired to discard sink on Tick 2 (TTL aged)
  UNIDENTIFIABLE  -> dropped to discard sink immediately on Tick 1

Isolated environment
--------------------
* Spins an isolated server subprocess on port 8421.
* All data lives under ``/tmp/paramem-orphan-test/`` (wiped at script start).
* ``consolidation.mode = simulate`` — no LoRA training ever fires.
* ``consolidation.training_idle_debounce_s = 0`` — tick runs immediately.
* ``consolidation.orphan_retirement = "every 1m"`` — 60-second HOLDABLE TTL.

Usage::

    python scripts/dev/probe_orphan_classification_live.py
    # (run from the project root, in the paramem conda env, with the
    #  production server stopped so the GPU is free)

Returns 0 on full PASS, non-zero if any assertion fails.
Results written to /tmp/live_orphan_classification_test_results.json.

Known nuance
------------
If Tick 1 creates an interim adapter slot (i.e. ``tick1_status == "started"``),
Tick 2 may hit the full-cycle gate which by design does NOT retire pending
HOLDABLE sessions — retirement only happens in interim ticks.  The aged-HOLDABLE
retirement assertion is therefore validated when Tick 2 runs as an interim tick
(i.e. ``tick2_status in ("noop_no_named", "noop_no_pending")``).

Dependencies
------------
Imports only from stdlib and ``paramem.*``.  Zero dependency on anything under
``experiments/``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml
from dotenv import dotenv_values

# ---------------------------------------------------------------------------
# Project root — resolve relative to this file's location (scripts/dev/).
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("probe_orphan_classification_live")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Isolated data directory — wiped at script start.
TEST_DATA_ROOT = Path("/tmp/paramem-orphan-test")

# Isolated server port — never touches 8420 (production) or 18420 (fixture).
SERVER_PORT = 8421
BASE_URL = f"http://localhost:{SERVER_PORT}"

# NAMED speaker seeded in speaker_profiles.json.
NAMED_SPEAKER_ID = "named_user"
NAMED_SPEAKER_NAME = "Named User"

# 256-dim embeddings (wespeaker-voxceleb-resnet34-LM emits 256-dim vectors,
# per speaker_embedding.py line 60).
# E_NAMED: first component 1.0, rest 0.0 — a unit vector.
E_NAMED: list[float] = [1.0] + [0.0] * 255

# E_ANON: second component 1.0, rest 0.0 — orthogonal to E_NAMED, ensuring
# no match against the seeded named profile (cosine similarity = 0.0).
E_ANON: list[float] = [0.0, 1.0] + [0.0] * 254

# Conversation IDs for the three test sessions.
CONV_NAMED = "conv-named"
CONV_ANON = "conv-anon"
CONV_TEXT = "conv-text"

# Health-poll and consolidation-wait timeouts.
# Model load in simulate mode: ~60-120 s (4-bit Mistral 7B on RTX 5070).
HEALTH_POLL_TIMEOUT_S = 300
# Consolidation in simulate mode: extraction only, no LoRA training.
# Each session runs the extraction LLM forward pass.  Give 5 min.
CONSOLIDATE_WAIT_TIMEOUT_S = 300

# Tick-2: sleep this many seconds to age the HOLDABLE session past the 60 s TTL.
HOLDABLE_AGE_SLEEP_S = 65

# Production config path (template for our isolated config).
PROD_CONFIG_PATH = _PROJECT_ROOT / "configs" / "server.yaml"

# ---------------------------------------------------------------------------
# Inlined helpers (sourced from experiments/probe_qwen_realpath_validation.py)
# ---------------------------------------------------------------------------
# These six functions are copied verbatim from the probe_qwen source with only
# the minimum adjustments needed to stand alone (module-level name references).
#
# Source line refs:
#   _resolve_api_token  line 122  — env -> .env fallback
#   _wait_for_cooldown  line 136  — gpu-cooldown.sh wrapper
#   _port_free          line 157  — TCP socket probe
#   _wait_port_free     line 164  — loop until port free
#   _poll_health        line 177  — GET /health poller
#   _http_post          line 198  — POST helper with bearer auth


def _resolve_api_token() -> str | None:
    """Resolve PARAMEM_API_TOKEN from ambient env or .env file.

    Returns the token string or None when absent.

    Source: experiments/probe_qwen_realpath_validation.py line 122.
    Adjusted: ``project_root`` -> ``_PROJECT_ROOT``.
    """
    token = os.environ.get("PARAMEM_API_TOKEN")
    if token:
        return token
    dotenv_path = _PROJECT_ROOT / ".env"
    if dotenv_path.is_file():
        return dotenv_values(dotenv_path).get("PARAMEM_API_TOKEN")
    return None


def _wait_for_cooldown(target: int = 52) -> None:
    """Block until GPU temperature drops to *target* °C.

    Shells out to gpu-cooldown.sh.  Falls back to a 60-second sleep when
    the script is unavailable (e.g. non-GPU CI runner).

    Source: experiments/probe_qwen_realpath_validation.py line 136.
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
        logger.warning("Cooldown script unavailable (%s), sleeping 60 s instead.", e)
        time.sleep(60)


def _port_free(port: int) -> bool:
    """Return True when nothing is bound to 127.0.0.1:<port>.

    Source: experiments/probe_qwen_realpath_validation.py line 157.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) != 0


def _wait_port_free(port: int, timeout: int = 30) -> bool:
    """Wait up to *timeout* seconds for *port* to become free.

    Returns True when the port is free, False on timeout.

    Source: experiments/probe_qwen_realpath_validation.py line 164.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_free(port):
            return True
        time.sleep(1)
    return False


def _poll_health(base_url: str, token: str | None, timeout: int) -> bool:
    """Poll GET /health until 200 or *timeout* seconds elapse.

    Returns True when the server is up, False on timeout.

    Source: experiments/probe_qwen_realpath_validation.py line 177.
    """
    import urllib.request

    deadline = time.monotonic() + timeout
    url = f"{base_url}/health"
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def _http_post(url: str, token: str | None, payload: dict | None = None) -> tuple[int, dict]:
    """POST *payload* (as JSON) to *url* with optional bearer auth.

    Returns (status_code, response_dict).

    Source: experiments/probe_qwen_realpath_validation.py line 198.
    Adjusted: timeout uses ``CONSOLIDATE_WAIT_TIMEOUT_S`` (local constant)
    instead of ``CONSOLIDATE_TIMEOUT_S`` from probe_qwen.
    """
    import json as _json
    import urllib.error
    import urllib.request

    body = _json.dumps(payload or {}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(req, timeout=CONSOLIDATE_WAIT_TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, _json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8") if exc.fp else ""
        try:
            body_dict = _json.loads(raw)
        except Exception:
            body_dict = {"_raw": raw}
        return exc.code, body_dict


# ---------------------------------------------------------------------------
# Config builder (adapted from _build_cell_config, probe_qwen line 226)
# ---------------------------------------------------------------------------


def _build_test_config() -> Path:
    """Write an isolated server config YAML for the orphan-classification test.

    Derives from ``configs/server.yaml`` and overrides:
      server.port              -> 8421
      model                   -> (prod value; kept so server loads correctly)
      paths.data               -> /tmp/paramem-orphan-test/
      paths.sessions           -> /tmp/paramem-orphan-test/sessions/
      paths.debug              -> /tmp/paramem-orphan-test/debug/
      paths.prompts            -> configs/prompts (prod prompts)
      debug                    -> true   (enables discard-sink writes)
      headless_boot            -> false
      consolidation.mode       -> simulate (extract only; no LoRA training)
      consolidation.orphan_retirement -> "every 1m"  (60 s TTL for HOLDABLE sessions)
      consolidation.training_idle_debounce_s -> 0 (tick fires immediately after chat)
      consolidation.extraction_ha_validation  -> false
      speaker.enabled          -> true   (store loads the pre-seeded profile)
      stt.enabled              -> false
      tts.enabled              -> false
      text_lang_detection.enabled -> false
      agents.ha_agent_id       -> ""     (HA disabled)
      mobile_pwa.enabled       -> false

    Returns:
        Path to the written YAML config under ``TEST_DATA_ROOT``.
    """
    with PROD_CONFIG_PATH.open() as fh:
        cfg = yaml.safe_load(fh)

    # Server port.
    if "server" not in cfg or not isinstance(cfg["server"], dict):
        cfg["server"] = {}
    cfg["server"]["port"] = SERVER_PORT

    # Isolated paths — never touch data/ha.
    data_root = str(TEST_DATA_ROOT)
    cfg["paths"] = {
        "data": data_root,
        "sessions": f"{data_root}/sessions",
        "debug": f"{data_root}/debug",
        "prompts": str(_PROJECT_ROOT / "configs" / "prompts"),
    }

    # Debug ON — required for discard-sink writes (retain = debug OR retain_sessions).
    cfg["debug"] = True

    # Headless boot off.
    cfg["headless_boot"] = False

    # Consolidation: simulate mode + 1-minute orphan TTL.
    if "consolidation" not in cfg or not isinstance(cfg["consolidation"], dict):
        cfg["consolidation"] = {}
    cfg["consolidation"]["mode"] = "simulate"
    cfg["consolidation"]["orphan_retirement"] = "every 1m"
    # Disable the idle debounce so the tick runs immediately after we inject
    # chats — otherwise _dispatch_consolidation returns
    # "deferred_idle" (chat too recent) and never reaches classification.
    cfg["consolidation"]["training_idle_debounce_s"] = 0
    cfg["consolidation"]["extraction_ha_validation"] = False

    # Speaker: enabled so SpeakerStore loads the pre-seeded profile.
    # pyannote embedding model load fails gracefully (app.py WARNING only).
    if "speaker" not in cfg or not isinstance(cfg["speaker"], dict):
        cfg["speaker"] = {}
    cfg["speaker"]["enabled"] = True

    # STT / TTS: off.
    if "stt" not in cfg or not isinstance(cfg["stt"], dict):
        cfg["stt"] = {}
    cfg["stt"]["enabled"] = False

    if "tts" not in cfg or not isinstance(cfg["tts"], dict):
        cfg["tts"] = {}
    cfg["tts"]["enabled"] = False

    # Text language detection: off.
    if "text_lang_detection" not in cfg or not isinstance(cfg["text_lang_detection"], dict):
        cfg["text_lang_detection"] = {}
    cfg["text_lang_detection"]["enabled"] = False

    # Agents: HA disabled.
    if "agents" not in cfg or not isinstance(cfg["agents"], dict):
        cfg["agents"] = {}
    cfg["agents"]["ha_agent_id"] = ""

    # Mobile PWA: off.
    if "mobile_pwa" not in cfg or not isinstance(cfg["mobile_pwa"], dict):
        cfg["mobile_pwa"] = {}
    cfg["mobile_pwa"]["enabled"] = False

    TEST_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    config_path = TEST_DATA_ROOT / "test_config.yaml"
    with config_path.open("w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True)
    logger.info("Wrote isolated config to %s", config_path)
    return config_path


# ---------------------------------------------------------------------------
# Speaker-profile seeder (adapted from _seed_speaker_profile, probe_qwen line 369)
# ---------------------------------------------------------------------------


def _seed_speaker_profiles() -> None:
    """Pre-seed speaker_profiles.json with the NAMED speaker.

    Seeds a v5 profile for ``named_user`` with embedding ``E_NAMED``
    ([1.0] + [0.0]*255).  This ensures:
      - The SpeakerStore loads the profile at server startup.
      - A /chat turn carrying ``speaker_embedding=E_NAMED`` matches
        ``named_user`` at high confidence (cosine similarity = 1.0).
      - A turn carrying ``E_ANON`` ([0.0, 1.0, ...]) has cosine similarity 0.0
        with the named profile -> no match -> ``register_anonymous`` fires.

    Must be called BEFORE launching the server subprocess so the SpeakerStore
    finds the file during lifespan ``_rebuild_app_state``.

    The profile layout follows the v5 format read by SpeakerStore
    (``paramem/server/speaker.py``).
    """
    data_root = TEST_DATA_ROOT
    data_root.mkdir(parents=True, exist_ok=True)
    profile_path = data_root / "speaker_profiles.json"
    profile = {
        "version": 5,
        "next_anon_index": 0,
        "last_greeted": {},
        "speakers": {
            NAMED_SPEAKER_ID: {
                "name": NAMED_SPEAKER_NAME,
                "embeddings": [E_NAMED],
                "preferred_language": "",
                "enroll_method": "manual",
            }
        },
    }
    profile_path.write_text(json.dumps(profile, indent=2))
    logger.info("Seeded speaker profiles at %s", profile_path)


# ---------------------------------------------------------------------------
# HTTP GET helper (not in probe_qwen — only _http_post is there)
# ---------------------------------------------------------------------------


def _http_get(url: str, token: str | None, timeout: int = 10) -> tuple[int, dict]:
    """GET *url* with optional bearer auth, return (status_code, response_dict).

    Uses the same urllib pattern as ``_http_post`` so no new HTTP client
    is introduced.
    """
    import json as _json
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, _json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8") if exc.fp else ""
        try:
            body_dict = _json.loads(raw)
        except Exception:
            body_dict = {"_raw": raw}
        return exc.code, body_dict


# ---------------------------------------------------------------------------
# Consolidation-done waiter (adapted from _wait_for_consolidation, probe_qwen
# line 488 — without the chunk-count fast-path; instead polls consolidating flag)
# ---------------------------------------------------------------------------


def _wait_consolidation_done(token: str | None, timeout: int = CONSOLIDATE_WAIT_TIMEOUT_S) -> bool:
    """Block until the background consolidation cycle finishes (consolidating=False x2).

    Adapted from probe_qwen's ``_wait_for_consolidation`` (line 488) but without
    the ``EXPECTED_CHUNKS`` graph-snapshot fast-path (chat sessions do not
    necessarily produce snapshots — only NAMED sessions do, and only one is NAMED).

    Polls GET /status every 3 s.  Returns True when ``consolidating==False`` for
    2 consecutive polls OR when the timeout elapses (returns False on timeout).

    Args:
        token: Bearer token for GET /status (public endpoint; token may be None).
        timeout: Maximum seconds to wait.

    Returns:
        True when the cycle finished, False on timeout.
    """
    status_url = f"{BASE_URL}/status"
    deadline = time.monotonic() + timeout
    consecutive_false = 0

    logger.info("Waiting for consolidation cycle (timeout %ds) ...", timeout)
    while time.monotonic() < deadline:
        code, body = _http_get(status_url, token, timeout=10)
        consolidating = body.get("consolidating")
        logger.info("consolidating=%s (HTTP %d)", consolidating, code)

        if consolidating is False:
            consecutive_false += 1
        else:
            consecutive_false = 0

        if consecutive_false >= 2:
            logger.info("consolidation done (consolidating=False x2)")
            return True

        time.sleep(3)

    logger.warning("consolidation wait timed out after %ds", timeout)
    return False


# ---------------------------------------------------------------------------
# Session-injection helper
# ---------------------------------------------------------------------------


def _inject_turns(
    conv_id: str,
    text_turns: list[str],
    token: str | None,
    speaker_embedding: list[float] | None = None,
) -> list[tuple[int, dict]]:
    """POST each *text_turn* to /chat for *conv_id*.

    Returns a list of (status_code, response_dict) per turn.
    /chat is a public endpoint (no auth required per app.py line 2558).
    ``speaker_embedding`` is forwarded in each turn so the server can
    resolve speaker identity from the voice fingerprint.

    Args:
        conv_id: Conversation ID; becomes the session_id / JSONL filename.
        text_turns: List of user-side text turns (2 turns for content).
        token: Bearer token (used only if the server was configured with auth;
            /chat is public but we pass it so the token-based speaker path
            fires if the isolated config has token-based speaker auth).
        speaker_embedding: 256-dim float list, or None for text-only sessions.

    Returns:
        List of (status_code, response_body) per turn.
    """
    results = []
    chat_url = f"{BASE_URL}/chat"
    for text in text_turns:
        payload: dict = {
            "text": text,
            "conversation_id": conv_id,
        }
        if speaker_embedding is not None:
            payload["speaker_embedding"] = speaker_embedding
        code, body = _http_post(chat_url, token=None, payload=payload)
        logger.info("POST /chat conv=%s HTTP %d: %s", conv_id, code, str(body)[:120])
        results.append((code, body))
    return results


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _pending_count(token: str | None) -> int | None:
    """Return current pending_sessions count from GET /status, or None on error."""
    code, body = _http_get(f"{BASE_URL}/status", token)
    if code != 200:
        return None
    return body.get("pending_sessions")


def _snapshot_exists_for(conv_id: str) -> bool:
    """True iff a graph_snapshot.json exists under debug/episodic for *conv_id*.

    Pattern: ``debug/episodic/cycle_*/run_*/sessions/<conv_id>/graph_snapshot.json``
    Mirrors the glob in ``_read_session_snapshots`` (probe_qwen line 441-443).
    """
    debug_path = TEST_DATA_ROOT / "debug"
    pattern = f"episodic/cycle_*/run_*/sessions/{conv_id}/graph_snapshot.json"
    return any(debug_path.glob(pattern))


def _discard_sink_jsonl(conv_id: str) -> Path:
    """Path where mark_consolidated writes the JSONL for a discarded session.

    ``discard_session_sink(config)`` returns ``config.debug_dir / "discarded_sessions"``
    (``paramem/server/consolidation.py`` line 94).  ``mark_consolidated`` with
    ``retention_dir=discard_session_sink(config)`` moves the session's JSONL to
    ``retention_dir/<session_id>.jsonl`` (flat layout for transcript sessions,
    ``session_buffer.py`` line 591).  The session_id == conversation_id.
    """
    return TEST_DATA_ROOT / "debug" / "discarded_sessions" / f"{conv_id}.jsonl"


def _session_still_pending(conv_id: str) -> bool:
    """True iff the session JSONL still exists in the sessions directory.

    ``mark_consolidated`` moves the JSONL from ``sessions/<session_id>.jsonl``
    to the retention dir, so absence = consumed.  Presence = still pending.
    """
    return (TEST_DATA_ROOT / "sessions" / f"{conv_id}.jsonl").exists()


# ---------------------------------------------------------------------------
# Assertion recorder
# ---------------------------------------------------------------------------


class AssertionTracker:
    """Collect pass/fail assertions and print a summary table at the end.

    Each call to :meth:`check` registers one assertion with its name,
    expected value, actual value, and PASS/FAIL outcome.
    """

    def __init__(self) -> None:
        self._results: list[dict] = []

    def check(self, name: str, expected, actual) -> bool:
        """Register one assertion.  Returns True on PASS.

        Args:
            name: Human-readable assertion label.
            expected: Expected value (used for the table and equality check).
            actual: Observed value.

        Returns:
            True when ``actual == expected``, False otherwise.
        """
        passed = actual == expected
        self._results.append(
            {
                "name": name,
                "expected": str(expected),
                "actual": str(actual),
                "pass": passed,
            }
        )
        mark = "PASS" if passed else "FAIL"
        logger.info("[%s] %s — expected=%r actual=%r", mark, name, expected, actual)
        return passed

    def print_table(self) -> None:
        """Print a formatted PASS/FAIL table to stdout."""
        col_name = max(len(r["name"]) for r in self._results) if self._results else 20
        col_exp = max(len(r["expected"]) for r in self._results) if self._results else 10
        col_act = max(len(r["actual"]) for r in self._results) if self._results else 10
        header = (
            f"{'Assertion':<{col_name}}  {'Expected':<{col_exp}}  {'Actual':<{col_act}}  Result"
        )
        print("\n=== Orphan Classification Test Results ===")
        print(header)
        print("-" * len(header))
        for r in self._results:
            mark = "PASS" if r["pass"] else "FAIL"
            print(
                f"{r['name']:<{col_name}}  {r['expected']:<{col_exp}}  "
                f"{r['actual']:<{col_act}}  {mark}"
            )
        total = len(self._results)
        passed = sum(1 for r in self._results if r["pass"])
        print(f"\n{passed}/{total} assertions passed.")
        print("OVERALL: PASS" if passed == total else "OVERALL: FAIL")
        print()

    @property
    def all_passed(self) -> bool:
        """True iff every recorded assertion passed."""
        return all(r["pass"] for r in self._results)

    def to_list(self) -> list[dict]:
        """Return the raw assertion records for JSON serialisation."""
        return list(self._results)


# ---------------------------------------------------------------------------
# Main test driver
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the orphan-classification live-GPU probe.

    Steps:
    1. Wipe isolated data dir; write config; seed speaker profiles.
    2. Wait for GPU cooldown (via ``_wait_for_cooldown``).
    3. Launch isolated server subprocess on port 8421.
    4. Poll /health until ready (up to 300 s).
    5. Inject 3 pending sessions via POST /chat.
    6. Intermediate check: all 3 sessions are pending.
    7. Tick 1: POST /consolidate; wait for done; assert Tick-1 dispositions.
    8. Sleep 65 s to age the HOLDABLE session past the 60 s TTL.
    9. Tick 2: POST /consolidate; wait; assert Tick-2 dispositions.
    10. Teardown: terminate server; wait port free.
    11. Write results JSON; print table; return 0 on PASS, 1 on FAIL.

    Returns:
        0 on full PASS, 1 if any assertion fails, 2 on infrastructure failure.
    """
    assertions = AssertionTracker()

    # ------------------------------------------------------------------
    # Step 1: Wipe and prepare isolated data dir.
    # ------------------------------------------------------------------
    logger.info("Wiping isolated data dir: %s", TEST_DATA_ROOT)
    if TEST_DATA_ROOT.exists():
        shutil.rmtree(str(TEST_DATA_ROOT))
    TEST_DATA_ROOT.mkdir(parents=True)

    # Build config and seed profiles BEFORE server launch.
    config_path = _build_test_config()
    _seed_speaker_profiles()

    # Resolve admin token for /consolidate (require_admin endpoint).
    # /chat and /status are public (no auth required).
    api_token = _resolve_api_token()
    if api_token is None:
        logger.warning(
            "PARAMEM_API_TOKEN not found in env or .env — "
            "POST /consolidate (require_admin) will return 401. "
            "Set PARAMEM_API_TOKEN before running."
        )

    # ------------------------------------------------------------------
    # Step 2: GPU cooldown.
    # ------------------------------------------------------------------
    logger.info("Waiting for GPU cooldown before launching isolated server ...")
    _wait_for_cooldown()

    # ------------------------------------------------------------------
    # Step 3: Confirm port is free; launch isolated server subprocess.
    # ------------------------------------------------------------------
    if not _port_free(SERVER_PORT):
        logger.error("Port %d already in use — abort.", SERVER_PORT)
        return 2

    python_bin = sys.executable
    server_cmd = [
        python_bin,
        "-m",
        "paramem.server.app",
        "--config",
        str(config_path),
    ]
    server_log_path = TEST_DATA_ROOT / "server.log"
    logger.info("Launching server: %s", " ".join(server_cmd))
    server_log_fh = server_log_path.open("w")
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=server_log_fh,
        stderr=subprocess.STDOUT,
        cwd=str(_PROJECT_ROOT),
        # Isolated-server gotcha: the server runs load_dotenv(.env) at startup
        # (override=False), so we force Security-OFF by setting the daily
        # passphrase PRESENT-BUT-EMPTY here — daily_passphrase_env_value()
        # treats "" as None, and load_dotenv won't overwrite an already-set var.
        # Without this, encryption mode rejects the plaintext speaker_profiles.json.
        env={
            **os.environ,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "HF_DEACTIVATE_ASYNC_LOAD": "1",
            "PARAMEM_DAILY_PASSPHRASE": "",  # Security-OFF: empty string disables encryption
        },
    )
    logger.info("Server PID %d; log: %s", server_proc.pid, server_log_path)

    try:
        # ------------------------------------------------------------------
        # Step 4: Poll /health until ready.
        # ------------------------------------------------------------------
        logger.info("Polling %s/health (timeout %ds) ...", BASE_URL, HEALTH_POLL_TIMEOUT_S)
        up = _poll_health(BASE_URL, api_token, HEALTH_POLL_TIMEOUT_S)
        if not up:
            logger.error("Server did not respond within %ds — STOP.", HEALTH_POLL_TIMEOUT_S)
            return 2
        logger.info("Server up at %s", BASE_URL)

        # ------------------------------------------------------------------
        # Step 5: Inject 3 pending sessions via POST /chat.
        # ------------------------------------------------------------------
        # NAMED session: speaker_embedding=E_NAMED -> matched to named_user
        #   (cosine sim = 1.0 with the seeded [1.0]+[0.0]*255 centroid).
        #   The speaker is non-anonymous (enroll_method="manual") -> NAMED class.
        logger.info("Injecting NAMED session (conv-named, E_NAMED) ...")
        _inject_turns(
            CONV_NAMED,
            [
                "My project is called Helios and it ships in March.",
                "I work at SkyTech and report to Dr. Müller.",
            ],
            token=None,  # /chat is public
            speaker_embedding=E_NAMED,
        )

        # HOLDABLE session: speaker_embedding=E_ANON -> no match -> register_anonymous
        #   -> Speaker{N} with enroll_method="anonymous_voice" -> is_anonymous=True -> HOLDABLE.
        logger.info("Injecting HOLDABLE session (conv-anon, E_ANON) ...")
        _inject_turns(
            CONV_ANON,
            [
                "Hello, I have some information to share.",
                "My car is a blue Toyota and I live in Berlin.",
            ],
            token=None,
            speaker_embedding=E_ANON,
        )

        # UNIDENTIFIABLE session: no speaker_embedding, no token
        #   -> speaker_id stays None; no embedding carried in turns.
        #   -> classify_session(speaker_id=None, is_anonymous=False,
        #                       has_voice_embedding=False) -> UNIDENTIFIABLE.
        logger.info("Injecting UNIDENTIFIABLE session (conv-text, no embedding) ...")
        _inject_turns(
            CONV_TEXT,
            [
                "I prefer dark mode and use Vim.",
                "My budget for the laptop is around 2000 euros.",
            ],
            token=None,
            speaker_embedding=None,
        )

        # ------------------------------------------------------------------
        # Step 6: Intermediate check — all 3 sessions must be pending.
        # ------------------------------------------------------------------
        # GET /status.pending_sessions counts buffer.get_summary()["total"].
        # All 3 sessions have had turns appended -> they are in the buffer.
        pending = _pending_count(api_token)
        logger.info("Pre-consolidate pending_sessions: %s", pending)
        assertions.check(
            "pre-tick-1 pending_sessions >= 3",
            True,
            pending is not None and pending >= 3,
        )

        if pending is None or pending < 3:
            logger.error(
                "STOP: expected >= 3 pending sessions before Tick 1; got %s. "
                "A POST /chat turn may not have created a pending session — "
                "check server.log for rejection errors.",
                pending,
            )
            return 2

        # ------------------------------------------------------------------
        # Step 7: Tick 1 — POST /consolidate.
        # ------------------------------------------------------------------
        logger.info("Tick 1: POST /consolidate ...")
        tick1_code, tick1_body = _http_post(f"{BASE_URL}/consolidate", api_token, payload={})
        logger.info("Tick 1 POST /consolidate -> HTTP %d: %s", tick1_code, tick1_body)

        assertions.check(
            "tick-1 /consolidate HTTP 200",
            200,
            tick1_code,
        )

        tick1_status = tick1_body.get("status", "")
        # Expected status: "started" (>=1 NAMED session) or "noop_no_named"
        # (if speaker matching fails — hurdle path). We assert it is NOT
        # a deferred/error status so we know classification fired.
        assertions.check(
            "tick-1 consolidate status is started or noop_no_pending or noop_no_named",
            True,
            tick1_status in ("started", "noop_no_pending", "noop_no_named"),
        )

        # Wait for the background cycle to complete.
        _wait_consolidation_done(api_token, timeout=CONSOLIDATE_WAIT_TIMEOUT_S)

        # Add a short safety margin for file I/O to settle.
        time.sleep(2)

        # --- Assertion: NAMED session -> extracted (graph_snapshot.json present) ---
        # In simulate mode, ConsolidationLoop runs extraction and writes
        # debug/episodic/cycle_N/run_*/sessions/<conv_id>/graph_snapshot.json.
        # This only fires when named_count > 0 and the tick returns "started".
        if tick1_status == "started":
            assertions.check(
                "tick-1 NAMED graph_snapshot.json exists",
                True,
                _snapshot_exists_for(CONV_NAMED),
            )
        else:
            logger.warning(
                "Tick 1 returned %r (not 'started') — skipping NAMED snapshot check. "
                "This may indicate the speaker embedding match failed (hurdle).",
                tick1_status,
            )
            # Record as a conditional skip so the results JSON captures it.
            assertions.check(
                "tick-1 NAMED graph_snapshot.json exists (tick was started)",
                "SKIPPED",
                f"SKIPPED (tick1_status={tick1_status!r})",
            )

        # --- Assertion: NAMED session not pending (consumed by consolidation) ---
        assertions.check(
            "tick-1 NAMED session no longer pending",
            False,
            _session_still_pending(CONV_NAMED),
        )

        # --- Assertion: HOLDABLE session has NO snapshot and is STILL pending ---
        # TTL not yet expired (we haven't slept 65 s yet).
        assertions.check(
            "tick-1 HOLDABLE no snapshot (held)",
            False,
            _snapshot_exists_for(CONV_ANON),
        )
        assertions.check(
            "tick-1 HOLDABLE still pending (not retired)",
            True,
            _session_still_pending(CONV_ANON),
        )

        # --- Assertion: UNIDENTIFIABLE session -> discard sink, not pending ---
        assertions.check(
            "tick-1 UNIDENTIFIABLE discard JSONL exists",
            True,
            _discard_sink_jsonl(CONV_TEXT).exists(),
        )
        assertions.check(
            "tick-1 UNIDENTIFIABLE no longer pending",
            False,
            _session_still_pending(CONV_TEXT),
        )

        # --- Assertion: Tick-1 consolidate status correct ---
        # "started" when >=1 NAMED present (named_count > 0 -> extraction launched).
        # Checked already above; re-confirm in the assertions table.
        assertions.check(
            "tick-1 consolidate started (NAMED present)",
            "started",
            tick1_status if tick1_status == "started" else tick1_status,
        )

        # ------------------------------------------------------------------
        # Step 8: Sleep to age the HOLDABLE session past the 60 s TTL.
        # ------------------------------------------------------------------
        logger.info(
            "Sleeping %ds to age HOLDABLE session past 60 s orphan_retirement TTL ...",
            HOLDABLE_AGE_SLEEP_S,
        )
        time.sleep(HOLDABLE_AGE_SLEEP_S)

        # ------------------------------------------------------------------
        # Step 9: Tick 2 — POST /consolidate.
        # ------------------------------------------------------------------
        logger.info("Tick 2: POST /consolidate ...")
        tick2_code, tick2_body = _http_post(f"{BASE_URL}/consolidate", api_token, payload={})
        logger.info("Tick 2 POST /consolidate -> HTTP %d: %s", tick2_code, tick2_body)

        assertions.check(
            "tick-2 /consolidate HTTP 200",
            200,
            tick2_code,
        )

        tick2_status = tick2_body.get("status", "")
        # After aging: classify_session sees HOLDABLE age > 60 s -> drop_ids appended.
        # No NAMED sessions remain (consumed in Tick 1) so status -> "noop_no_named"
        # (or "noop_no_pending" if the HOLDABLE is the only session left and gets
        # dropped before the named_count check, which is exactly the code path).
        assertions.check(
            "tick-2 consolidate is noop (no NAMED after HOLDABLE retired)",
            True,
            tick2_status in ("noop_no_named", "noop_no_pending"),
        )

        _wait_consolidation_done(api_token, timeout=60)
        time.sleep(2)

        # --- Assertion: HOLDABLE session -> discard sink after TTL expiry ---
        assertions.check(
            "tick-2 HOLDABLE discard JSONL exists (retired after TTL)",
            True,
            _discard_sink_jsonl(CONV_ANON).exists(),
        )
        assertions.check(
            "tick-2 HOLDABLE no longer pending",
            False,
            _session_still_pending(CONV_ANON),
        )

        # --- Assertion: HOLDABLE still has no snapshot (never extracted) ---
        assertions.check(
            "tick-2 HOLDABLE no graph_snapshot (never extracted)",
            False,
            _snapshot_exists_for(CONV_ANON),
        )

    finally:
        # ------------------------------------------------------------------
        # Step 10: Teardown — terminate server; wait port free.
        # ------------------------------------------------------------------
        logger.info("Terminating server PID %d ...", server_proc.pid)
        try:
            server_proc.terminate()
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("Server did not exit gracefully — SIGKILL.")
            server_proc.kill()
            server_proc.wait(timeout=10)
        except Exception as exc:
            logger.warning("Server termination error: %s", exc)
        server_log_fh.close()
        freed = _wait_port_free(SERVER_PORT, timeout=20)
        if not freed:
            logger.warning("Port %d still in use 20 s after termination.", SERVER_PORT)
        else:
            logger.info("Port %d free.", SERVER_PORT)

    # ------------------------------------------------------------------
    # Step 11: Write results; print table; return exit code.
    # ------------------------------------------------------------------
    results_path = Path("/tmp/live_orphan_classification_test_results.json")
    results = {
        "overall_pass": assertions.all_passed,
        "assertions": assertions.to_list(),
        "test_data_root": str(TEST_DATA_ROOT),
        "server_port": SERVER_PORT,
        "conv_ids": {
            "named": CONV_NAMED,
            "anon": CONV_ANON,
            "text": CONV_TEXT,
        },
    }
    results_path.write_text(json.dumps(results, indent=2))
    logger.info("Results written to %s", results_path)

    assertions.print_table()

    return 0 if assertions.all_passed else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        logger.exception("Test crashed with unhandled exception.")
        rc = 2
    sys.exit(rc)
