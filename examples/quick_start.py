"""Black-box post-install integrity smoke for the running ParaMem server.

PERMANENT empty-store design — this script is a fresh-install integrity test
and refuses to run on a populated key store.  Before running, the server must
point at an empty ``paths.data`` directory (``keys_count == 0`` and
``pending_sessions == 0``).

Two-stage flow
--------------
**Stage 1 — initial training (~10 keys)**

- Inject ~10 fictional one-fact turns via ``POST /chat`` under a fresh test
  speaker identified by a synthetic 256-dim embedding.
- Resolve the test speaker via the before/after ``/status`` speaker-delta
  (exactly one new speaker; abort if 0 or >1).
- Wait for the idle-debounce window, then call ``POST /consolidate``; handle
  ``deferred_idle`` retries; fail on ``noop_*`` (no sessions reached the
  buffer).
- Poll ``/status.consolidating`` until the background cycle finishes.
- Assert: ``keys_count >= 6`` (extraction is non-deterministic — a floor is
  asserted, not an exact count), AND extraction coverage of all 10 injected
  facts, AND 100% keyed recall via ``GET /debug/dump`` + ``POST /debug/recall``
  (deterministic at temperature 0).

**Stage 2 — incremental (+5 keys on top)**

- Inject 5 more distinct fictional fact turns (same test speaker).
- Repeat debounce, consolidate, and wait-for-completion.
- Assert: ``keys_count`` grew by **at least 3** over the post-stage-1 count
  (incremental learning landed), AND extraction coverage of the 5 new facts,
  AND 100% keyed recall across ALL entries (proves both new keys recall and
  that no catastrophic forgetting occurred for old keys).  No separate
  forgetting recheck is needed — the full-entry sweep covers it.

**Rejection / abstention check (run once, after Stage 2)**

- Probes ~5 questions about facts that were **never injected** (e.g. phone
  number, partner's name, car insurance, medication, email address) via
  ``POST /debug/probe``.
- Asserts the server **abstains** rather than confabulating an answer for each
  absent fact.  A probe passes if the response text contains any marker from
  ``_ABSTENTION_MARKERS`` (case-insensitive); it fails (hallucination detected)
  if no marker is present.
- All rejection probes must pass for this check to pass.  The final verdict
  requires Stage 1, Stage 2, AND the rejection check to all pass.

Recall mechanism
----------------
Positive recall is verified via ``GET /debug/dump`` (registry inventory — zero
GPU cost) and ``POST /debug/recall`` (keyed retrieval at temperature 0,
deterministic).  Both endpoints require ``debug: true`` in the server config
and an admin bearer token (403 otherwise).  The rejection check still uses
``POST /debug/probe`` (NL abstention, also deterministic).

Prerequisites
-------------
1. ParaMem server **running** (``systemctl --user start paramem-server`` or
   ``uvicorn paramem.server.app:app``).
2. Server data dir is **empty** — ``keys_count == 0`` and
   ``pending_sessions == 0``.  If not, the script exits 2 immediately.
3. Server config has ``debug: true`` — required by ``GET /debug/dump``,
   ``POST /debug/recall``, and ``POST /debug/probe``.
4. Admin bearer token in ``PARAMEM_API_TOKEN`` env var or repo ``.env`` file.

Automatic cleanup
-----------------
``POST /speaker/forget`` is always called for the test speaker (in a
``finally`` block), removing its profile and marking its keys stale.  If
cleanup fails, a warning with the ``speaker_id`` is printed for manual
recovery.

Base URL
--------
Defaults to ``http://localhost:8420``.  Override via ``PARAMEM_URL`` env var.

Debounce wait
-------------
After the last ``POST /chat`` the smoke sleeps ~35 s before calling
``POST /consolidate``.  This satisfies the server's
``consolidation.training_idle_debounce_s`` (default 30 s).  If
``/consolidate`` returns ``deferred_idle`` after the wait, the smoke retries
up to 3 times.

Usage::

    python examples/quick_start.py

Exit 0 only if both stages pass; non-zero otherwise.

Note: this script does NOT configure training — all training hyperparameters,
early-stop windows, and recall thresholds come from the server's
``configs/server.yaml``.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from paramem.memory.entry import RECALL_TEMPLATE

# ---------------------------------------------------------------------------
# Minimal .env loader — read PARAMEM_API_TOKEN before imports hit the network
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent

# Debounce constant: must exceed consolidation.training_idle_debounce_s
# (default 30 s, paramem/server/config.py).  The server returns
# "deferred_idle" when a /chat happened less than debounce_s ago.
_DEBOUNCE_WAIT_S = 35

# Maximum consolidation retry attempts after a "deferred_idle" reply.
_CONSOLIDATE_MAX_RETRIES = 3

# Generous poll timeout for the background cycle (30 min).
_POLL_TIMEOUT = 1800.0
_POLL_INTERVAL = 10.0

# Minimum keys_count expected after stage 1 (extraction is non-deterministic).
_STAGE1_MIN_KEYS = 6

# Minimum key-count growth expected after stage 2 (incremental learning).
_STAGE2_MIN_GROWTH = 3


def _load_env_token() -> str | None:
    """Parse PARAMEM_API_TOKEN from the repo .env file.

    Returns the token string, or None when the file or variable is absent.
    Does not require python-dotenv.
    """
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return None
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == "PARAMEM_API_TOKEN":
                return value.strip().strip('"').strip("'") or None
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# Stage 1 — ~10 fictional facts (injected text + expected substring)
# ---------------------------------------------------------------------------

# Each entry is (injection_text, expected_substring).
# expected_substring is matched against the joined "subject predicate object"
# string from any registry entry (coverage check).
# All facts are fictional — no real personal data.
_STAGE1_FACTS: list[tuple[str, str]] = [
    ("My dog's name is Rex.", "rex"),
    ("I work as a botanist in Portland.", "botanist"),
    ("My favourite colour is indigo.", "indigo"),
    ("I grew up in a town called Mirabel.", "mirabel"),
    ("I play the theremin as a hobby.", "theremin"),
    ("My sibling's name is Lavinia.", "lavinia"),
    ("I drive a pale-green Citroën.", "citro"),
    ("My favourite food is jackfruit curry.", "jackfruit"),
    ("I play amateur badminton on weekends.", "badminton"),
    ("My cat is named Ptolemy.", "ptolemy"),
]

# ---------------------------------------------------------------------------
# Stage 2 — 5 additional fictional facts (injection text + expected substring)
# ---------------------------------------------------------------------------

# Each entry is (injection_text, expected_substring).
# expected_substring is matched against the joined "subject predicate object"
# string from any registry entry (coverage check).
_STAGE2_FACTS: list[tuple[str, str]] = [
    ("My favourite season is autumn.", "autumn"),
    ("I studied mechanical engineering at Velden University.", "velden"),
    ("My best friend is called Orsolya.", "orsolya"),
    ("I keep a collection of antique maps.", "map"),
    ("I went on holiday to Reykjavik.", "reykjavik"),
]

# ---------------------------------------------------------------------------
# Rejection / abstention check — absent-fact questions
# ---------------------------------------------------------------------------

# Questions about facts that were NEVER injected into the server.  The server
# must abstain (via SimHash rejection or the abstention prompt) rather than
# confabulate an answer.  All entries are distinct from the stage-1 and
# stage-2 fact set above.
_REJECTION_PROBES: list[str] = [
    "What is my phone number?",
    "What is my partner's name?",
    "What car insurance do I have?",
    "What medication do I take?",
    "What is my email address?",
]

# Any one of these substrings (case-insensitive) in the answer text means the
# server abstained correctly.  The set is intentionally broad to accommodate
# natural-language variation in abstention phrasing across model versions.
_ABSTENTION_MARKERS: frozenset[str] = frozenset(
    {
        "don't have",
        "do not have",
        "no information",
        "not have that information",
        "don't know",
        "do not know",
        "not aware",
        "can't find",
        "cannot find",
        "no record",
        "i'm not sure",
        "not sure",
        "don't recall",
        "do not recall",
        "unable to find",
        "haven't been told",
        "have not been told",
    }
)

# ---------------------------------------------------------------------------

# 256-dim L2-normalised unit vector (first component = 1.0, rest = 0.0).
# Sent with every /chat call so the server registers a stable anonymous
# Speaker{N} profile and attributes all turns to the same speaker_id.
_SPEAKER_EMBEDDING: list[float] = [1.0] + [0.0] * 255

# Conversation ID used for all injected turns — keeps them in one session
# so the session buffer attributes them coherently.
_CONVERSATION_ID = "quick-start-smoke"


def _resolve_token() -> str | None:
    """Return the admin bearer token from env or .env file, else None."""
    return os.environ.get("PARAMEM_API_TOKEN") or _load_env_token()


def _base_url() -> str:
    """Return the server base URL (no trailing slash)."""
    raw = os.environ.get("PARAMEM_URL", "http://localhost:8420")
    return raw.rstrip("/")


def _cleanup_test_speaker(
    base: str,
    token: str | None,
    speaker_id: str,
    post_json: object,
    ServerHTTPError: type,
    ServerUnreachable: type,
) -> None:
    """Call POST /speaker/forget for the given test speaker_id.

    Always called from the finally block — runs on both pass and fail.
    Prints what was cleaned; on any error prints a loud warning with the
    speaker_id so the operator can clean up manually.

    Parameters
    ----------
    base:
        Server base URL.
    token:
        Admin bearer token.
    speaker_id:
        The test speaker to remove.
    post_json:
        Bound reference to the post_json helper (avoids re-import inside
        the finally block after a potential import failure).
    ServerHTTPError:
        Bound reference to the ServerHTTPError exception class.
    ServerUnreachable:
        Bound reference to the ServerUnreachable exception class.
    """
    print(f"[smoke] Cleanup: POST /speaker/forget for {speaker_id!r} ...")
    try:
        forget_resp = post_json(
            f"{base}/speaker/forget",
            {"speaker_id": speaker_id},
            timeout=15.0,
            token=token,
        )
        removed = forget_resp.get("removed_speaker", False)
        stale_keys = forget_resp.get("stale_keys", [])
        discarded = forget_resp.get("discarded_sessions", [])
        print(
            f"[smoke]   cleanup: removed_speaker={removed}, "
            f"stale_keys={len(stale_keys)}, "
            f"discarded_sessions={len(discarded)}"
        )
    except (ServerHTTPError, ServerUnreachable, Exception) as exc:  # noqa: BLE001
        print(
            f"[WARN] Cleanup via /speaker/forget FAILED for speaker_id={speaker_id!r}: {exc}\n"
            f"       Please clean up manually:\n"
            f"         curl -X POST {base}/speaker/forget \\\n"
            f'           -H "Authorization: Bearer $PARAMEM_API_TOKEN" \\\n'
            f'           -H "Content-Type: application/json" \\\n'
            f'           -d \'{{"speaker_id": "{speaker_id}"}}\''
        )


def _inject_facts(
    base: str,
    token: str | None,
    facts: list[tuple[str, str]],
    stage_label: str,
    post_json: object,
    ServerHTTPError: type,
    ServerUnreachable: type,
) -> None:
    """Inject a list of fact turns via POST /chat.

    Parameters
    ----------
    base:
        Server base URL.
    token:
        Admin bearer token.
    facts:
        List of ``(injection_text, expected_substring)`` tuples.
        Only the injection_text component (element 0) is sent.
    stage_label:
        Short label used in log lines (e.g. ``"Stage 1"``).
    post_json:
        Bound reference to the post_json helper.
    ServerHTTPError:
        Bound reference to the ServerHTTPError exception class.
    ServerUnreachable:
        Bound reference to the ServerUnreachable exception class.
    """
    for i, (turn_text, _expect) in enumerate(facts, start=1):
        body = {
            "text": turn_text,
            "conversation_id": _CONVERSATION_ID,
            "speaker_embedding": _SPEAKER_EMBEDDING,
        }
        try:
            post_json(f"{base}/chat", body, timeout=30.0, token=token)
        except ServerHTTPError as exc:
            print(f"[FAIL] {stage_label} /chat turn {i} → HTTP {exc.status_code}: {exc.body}")
            sys.exit(1)
        except ServerUnreachable as exc:
            print(f"[FAIL] {stage_label} /chat turn {i} unreachable: {exc}")
            sys.exit(1)
        print(f"[smoke]   {stage_label} turn {i}/{len(facts)}: {turn_text!r:.55s}")


def _consolidate_and_wait(
    base: str,
    token: str | None,
    stage_label: str,
    get_json: object,
    post_json: object,
    ServerHTTPError: type,
    ServerUnreachable: type,
    ServerUnavailable: type,
) -> None:
    """Run debounce wait, POST /consolidate, and poll until the cycle finishes.

    Flow: debounce sleep → POST /consolidate (with ``deferred_idle`` retries up
    to ``_CONSOLIDATE_MAX_RETRIES``) → fail on ``noop_*`` (no sessions reached
    the buffer) → poll ``/status.consolidating`` until False.

    Designed for servers with ``refresh_cadence`` disabled: every
    ``/consolidate`` is an interim cycle that consumes its pending session in one
    pass, so no re-consolidation loop is needed.

    Exits non-zero on persistent ``deferred_idle``, on any ``noop_*`` status, or
    if the cycle does not finish within ``_POLL_TIMEOUT`` seconds.

    Parameters
    ----------
    base:
        Server base URL.
    token:
        Admin bearer token.
    stage_label:
        Short label used in log lines.
    get_json:
        Bound reference to the get_json helper.
    post_json:
        Bound reference to the post_json helper.
    ServerHTTPError:
        Bound reference to the ServerHTTPError exception class.
    ServerUnreachable:
        Bound reference to the ServerUnreachable exception class.
    ServerUnavailable:
        Bound reference to the ServerUnavailable exception class.
    """
    _CONSOLIDATE_TIMEOUT = 60.0

    # ------------------------------------------------------------------
    # Debounce wait — must exceed consolidation.training_idle_debounce_s.
    # ------------------------------------------------------------------
    print(
        f"[smoke] {stage_label}: waiting {_DEBOUNCE_WAIT_S}s for idle-debounce "
        f"(training_idle_debounce_s default=30s) ..."
    )
    time.sleep(_DEBOUNCE_WAIT_S)

    # ------------------------------------------------------------------
    # POST /consolidate — retry on deferred_idle.
    # ------------------------------------------------------------------
    consolidate_status = "deferred_idle"
    for attempt in range(1, _CONSOLIDATE_MAX_RETRIES + 1):
        if attempt > 1:
            print(
                f"[smoke]   /consolidate returned {consolidate_status!r} "
                f"(attempt {attempt - 1}/{_CONSOLIDATE_MAX_RETRIES}); "
                f"waiting {_DEBOUNCE_WAIT_S}s and retrying ..."
            )
            time.sleep(_DEBOUNCE_WAIT_S)

        try:
            consolidate_resp = post_json(
                f"{base}/consolidate",
                body=None,
                timeout=_CONSOLIDATE_TIMEOUT,
                token=token,
            )
        except ServerHTTPError as exc:
            print(f"[FAIL] {stage_label} /consolidate HTTP {exc.status_code}: {exc.body}")
            sys.exit(1)
        except ServerUnreachable as exc:
            print(f"[FAIL] {stage_label} /consolidate unreachable (timeout?): {exc}")
            sys.exit(1)

        consolidate_status = consolidate_resp.get("status", "unknown")
        print(
            f"[smoke]   /consolidate → status={consolidate_status!r} "
            f"(attempt {attempt}/{_CONSOLIDATE_MAX_RETRIES})"
        )

        if consolidate_status != "deferred_idle":
            break

    if consolidate_status == "deferred_idle":
        print(
            f"[FAIL] {stage_label}: /consolidate still returned {consolidate_status!r} after "
            f"{_CONSOLIDATE_MAX_RETRIES} attempt(s).\n"
            "       The server's idle-debounce window was not satisfied. "
            "Check consolidation.training_idle_debounce_s in server.yaml — "
            "if it is set higher than the default 30s, increase _DEBOUNCE_WAIT_S."
        )
        sys.exit(1)

    # noop_no_pending / noop_no_speaker: injected turns did not reach the buffer.
    if consolidate_status.startswith("noop_"):
        print(
            f"[FAIL] {stage_label}: /consolidate returned {consolidate_status!r} — "
            "no sessions were consolidated (the injected turns did not reach the buffer).\n"
            "       Check server logs for 'session_buffer' entries."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Poll until the cycle finishes.
    # A /debug/recall while training is in flight may preempt (abort) the
    # cycle, so keys never commit — must wait for consolidating=False.
    # ------------------------------------------------------------------
    print(f"[smoke] {stage_label}: waiting for the background cycle to finish ...")
    time.sleep(5.0)  # let the cycle flip consolidating=True before we poll
    waited = 0.0
    while waited < _POLL_TIMEOUT:
        try:
            st = get_json(f"{base}/status", timeout=10.0, token=token)
        except (ServerHTTPError, ServerUnreachable, ServerUnavailable):
            time.sleep(_POLL_INTERVAL)
            waited += _POLL_INTERVAL
            continue
        if not st.get("consolidating", False):
            print(f"[smoke]   {stage_label}: cycle finished after ~{int(waited)}s")
            return
        time.sleep(_POLL_INTERVAL)
        waited += _POLL_INTERVAL

    timeout_min = int(_POLL_TIMEOUT / 60)
    print(f"[FAIL] {stage_label}: consolidation did not finish within {timeout_min} min.")
    sys.exit(1)


def _dump_entries(
    base: str,
    token: str | None,
    get_json: object,
    ServerHTTPError: type,
    ServerUnreachable: type,
    ServerUnavailable: type,
) -> list[dict]:
    """Fetch the full registry inventory via GET /debug/dump.

    Returns the list of entry dicts from ``resp["entries"]``.  Exits
    non-zero on any error: exit 3 on 403 (debug mode hint), exit 1 otherwise.

    Parameters
    ----------
    base:
        Server base URL.
    token:
        Admin bearer token.
    get_json:
        Bound reference to the get_json helper.
    ServerHTTPError:
        Bound reference to the ServerHTTPError exception class.
    ServerUnreachable:
        Bound reference to the ServerUnreachable exception class.
    ServerUnavailable:
        Bound reference to the ServerUnavailable exception class.
    """
    try:
        resp = get_json(f"{base}/debug/dump", timeout=30.0, token=token)
    except ServerHTTPError as exc:
        if exc.status_code == 403:
            print("\n[SKIP] /debug/dump returned 403.")
            print(
                "       Ensure debug: true in your active configs/server.yaml\n"
                "       and that PARAMEM_API_TOKEN is set to an admin-scope token,\n"
                "       then restart the server and re-run this smoke."
            )
            sys.exit(3)
        print(f"[FAIL] /debug/dump HTTP {exc.status_code}: {exc.body}")
        sys.exit(1)
    except (ServerUnreachable, ServerUnavailable) as exc:
        print(f"[FAIL] /debug/dump unreachable: {exc}")
        sys.exit(1)
    return resp.get("entries", [])


def _recall_key(
    base: str,
    token: str | None,
    key: str,
    adapter: str,
    post_json: object,
    ServerHTTPError: type,
    ServerUnreachable: type,
) -> dict | None:
    """Recall a single key via POST /debug/recall.

    Sends the canonical RECALL_TEMPLATE prompt at temperature 0 and returns
    ``resp["parsed_entry"]`` (a dict or None).  Exits non-zero on error:
    exit 3 on 403, exit 1 on other HTTP errors or unreachable.

    Parameters
    ----------
    base:
        Server base URL.
    token:
        Admin bearer token.
    key:
        Registry key to recall (e.g. ``"graph1"``).
    adapter:
        Adapter name to activate during recall (typically the entry tier).
    post_json:
        Bound reference to the post_json helper.
    ServerHTTPError:
        Bound reference to the ServerHTTPError exception class.
    ServerUnreachable:
        Bound reference to the ServerUnreachable exception class.
    """
    body = {
        "text": RECALL_TEMPLATE.format(key=key),
        "adapter": adapter,
        "temperature": 0.0,
        "max_new_tokens": 128,
    }
    try:
        resp = post_json(f"{base}/debug/recall", body, timeout=60.0, token=token)
    except ServerHTTPError as exc:
        if exc.status_code == 403:
            print("\n[SKIP] /debug/recall returned 403.")
            print(
                "       Ensure debug: true in your active configs/server.yaml\n"
                "       and that PARAMEM_API_TOKEN is set to an admin-scope token,\n"
                "       then restart the server and re-run this smoke."
            )
            sys.exit(3)
        print(f"[FAIL] /debug/recall HTTP {exc.status_code}: {exc.body}")
        sys.exit(1)
    except ServerUnreachable as exc:
        print(f"[FAIL] /debug/recall unreachable: {exc}")
        sys.exit(1)
    return resp.get("parsed_entry")


def _norm(s: str | None) -> str:
    """Normalize a string for comparison: strip whitespace and lowercase.

    Parameters
    ----------
    s:
        Input string, or None.

    Returns
    -------
    str
        Stripped, lowercased string.  Returns ``""`` for None input.
    """
    return (s or "").strip().lower()


def _keyed_recall_check(
    base: str,
    token: str | None,
    entries: list[dict],
    stage_label: str,
    post_json: object,
    ServerHTTPError: type,
    ServerUnreachable: type,
) -> tuple[int, int]:
    """Verify every registry entry via deterministic keyed recall.

    For each entry fetched from ``/debug/dump``, calls ``POST /debug/recall``
    with the canonical RECALL_TEMPLATE at temperature 0.  A key PASSES when
    ``parsed_entry`` is not None and its ``object`` field exactly matches the
    registry entry's ``object`` (both normalized via ``_norm``).

    Prints one line per key: ``[PASS]`` or ``[MISS]`` with key, tier, expected
    object (truncated to ~40 chars), and recalled object (or ``<blocked>`` when
    parsed_entry is None).

    Parameters
    ----------
    base:
        Server base URL.
    token:
        Admin bearer token.
    entries:
        List of registry entry dicts from ``/debug/dump``.
    stage_label:
        Short label used in log lines (e.g. ``"Stage 1"``).
    post_json:
        Bound reference to the post_json helper.
    ServerHTTPError:
        Bound reference to the ServerHTTPError exception class.
    ServerUnreachable:
        Bound reference to the ServerUnreachable exception class.

    Returns
    -------
    tuple[int, int]
        ``(passes, total)`` where ``total == len(entries)``.
    """
    passes = 0
    total = len(entries)
    for entry in entries:
        key = entry.get("key", "")
        tier = entry.get("tier", "")
        expected_obj = entry.get("object", "")
        parsed = _recall_key(base, token, key, tier, post_json, ServerHTTPError, ServerUnreachable)
        if parsed is not None and _norm(parsed.get("object")) == _norm(expected_obj):
            mark = "PASS"
            passes += 1
            recalled_obj = parsed.get("object", "")
        else:
            mark = "MISS"
            recalled_obj = parsed.get("object", "") if parsed is not None else "<blocked>"
        print(
            f"[smoke]   [{mark}] {stage_label} key={key!r} tier={tier!r} "
            f"expected={expected_obj[:40]!r} recalled={recalled_obj[:40]!r}"
        )
    return passes, total


def _coverage_check(
    entries: list[dict],
    facts: list[tuple[str, str]],
    stage_label: str,
) -> tuple[int, int]:
    """Check that each injected fact appears in at least one registry entry.

    For each ``(text, expect)`` in ``facts``, searches whether
    ``expect.lower()`` appears in ANY entry's joined
    ``f"{subject} {predicate} {object}".lower()`` string.  This is robust to
    extraction phrasing variance — subject/predicate/object may differ from
    the injection wording as long as the key substring is present somewhere.

    Prints one line per fact: ``[PASS]`` or ``[MISS]``.

    Parameters
    ----------
    entries:
        List of registry entry dicts from ``/debug/dump``.
    facts:
        List of ``(injection_text, expected_substring)`` tuples.
    stage_label:
        Short label used in log lines.

    Returns
    -------
    tuple[int, int]
        ``(found, total)`` where ``total == len(facts)``.
    """
    found = 0
    total = len(facts)
    for inject_text, expect in facts:
        expect_lower = expect.lower()
        matched = any(
            expect_lower
            in f"{e.get('subject', '')} {e.get('predicate', '')} {e.get('object', '')}".lower()
            for e in entries
        )
        mark = "PASS" if matched else "MISS"
        if matched:
            found += 1
        print(
            f"[smoke]   [{mark}] {stage_label} coverage: {inject_text!r:.55s} (expect={expect!r})"
        )
    return found, total


def _probe_rejections(
    base: str,
    token: str | None,
    test_speaker_id: str,
    post_json: object,
    ServerHTTPError: type,
    ServerUnreachable: type,
) -> tuple[int, int]:
    """Probe absent-fact questions and assert abstention for each.

    For every question in ``_REJECTION_PROBES``, calls ``POST /debug/probe``
    and checks whether the response contains any marker from
    ``_ABSTENTION_MARKERS`` (case-insensitive).  A probe **passes** when the
    server abstains; it **fails** (hallucination detected) when the answer
    contains no abstention marker.

    Returns ``(passes, total)`` where ``total == len(_REJECTION_PROBES)``.
    Exits non-zero on 403 (debug mode not enabled or missing admin token).

    Parameters
    ----------
    base:
        Server base URL.
    token:
        Admin bearer token.
    test_speaker_id:
        Speaker ID to probe as.
    post_json:
        Bound reference to the post_json helper.
    ServerHTTPError:
        Bound reference to the ServerHTTPError exception class.
    ServerUnreachable:
        Bound reference to the ServerUnreachable exception class.
    """
    passes = 0
    total = len(_REJECTION_PROBES)
    for question in _REJECTION_PROBES:
        probe_body = {
            "text": question,
            "speaker_id": test_speaker_id,
            "conversation_id": "quick-start-probe",
            "history": None,
        }
        try:
            probe_resp = post_json(
                f"{base}/debug/probe",
                probe_body,
                timeout=60.0,
                token=token,
            )
        except ServerHTTPError as exc:
            if exc.status_code == 403:
                print("\n[SKIP] /debug/probe returned 403 during rejection check.")
                print(
                    "       Ensure debug: true in your active configs/server.yaml\n"
                    "       and that PARAMEM_API_TOKEN is set to an admin-scope token,\n"
                    "       then restart the server and re-run this smoke."
                )
                sys.exit(3)
            print(f"[FAIL] /debug/probe (rejection) HTTP {exc.status_code}: {exc.body}")
            sys.exit(1)
        except ServerUnreachable as exc:
            print(f"[FAIL] /debug/probe (rejection) unreachable: {exc}")
            sys.exit(1)

        answer_text: str = probe_resp.get("text", "")
        answer_lower = answer_text.lower()
        abstained = any(marker in answer_lower for marker in _ABSTENTION_MARKERS)
        mark = "PASS" if abstained else "FAIL"
        print(f"[smoke]   [{mark}] Q: {question!r:.55s}\n                 A: {answer_text!r:.80s}")
        if abstained:
            passes += 1

    return passes, total


def main() -> None:
    """Run the two-stage empty-store integrity smoke against the running server."""
    # Import here so the module is importable without hitting the network.
    from paramem.cli.http_client import (
        ServerHTTPError,
        ServerUnavailable,
        ServerUnreachable,
        get_json,
        post_json,
    )

    base = _base_url()
    token = _resolve_token()

    # test_speaker_id is set only after the speaker-delta step resolves it.
    # The finally block checks it before attempting cleanup.
    test_speaker_id: str | None = None
    _exit_code = 1  # set to 0 only when both stages pass

    try:
        # ------------------------------------------------------------------
        # Preflight: liveness + debug check
        # ------------------------------------------------------------------
        print(f"[smoke] Server: {base}")
        print("[smoke] Preflight: /health + /status ...")

        try:
            health = get_json(f"{base}/health", timeout=5.0)
        except ServerUnreachable as exc:
            print(f"[FAIL] Server unreachable at {base}: {exc}")
            print("       Start the server: systemctl --user start paramem-server")
            sys.exit(2)
        except ServerHTTPError as exc:
            print(f"[FAIL] /health returned HTTP {exc.status_code}: {exc.body}")
            sys.exit(1)

        if health.get("status") != "ok":
            print(f"[FAIL] /health unexpected body: {health}")
            sys.exit(1)
        print("[smoke]   /health OK")

        try:
            status_before = get_json(f"{base}/status", timeout=5.0, token=token)
        except (ServerUnreachable, ServerUnavailable, ServerHTTPError) as exc:
            print(f"[FAIL] /status failed: {exc}")
            sys.exit(1)

        server_mode = status_before.get("mode", "unknown")
        print(f"[smoke]   /status OK — mode={server_mode}")

        # ------------------------------------------------------------------
        # PERMANENT SAFETY — empty-store precondition
        # ------------------------------------------------------------------
        keys_count_initial = status_before.get("keys_count", 0)
        pending_before = status_before.get("pending_sessions", 0)

        if keys_count_initial != 0:
            print(
                f"[ABORT] quick_start is a fresh-install integrity test and refuses to run "
                f"on a populated key store (keys_count={keys_count_initial}). "
                f"Point a server at an empty data dir (paths.data) and retry."
            )
            sys.exit(2)

        if pending_before != 0:
            print(
                f"[ABORT] {pending_before} real pending session(s) present — "
                "refusing to run so the test isn't mixed with real data; "
                "consolidate or clear them first."
            )
            sys.exit(2)

        print(
            f"[smoke]   empty-store precondition met: "
            f"keys_count={keys_count_initial}, pending_sessions={pending_before}"
        )

        # Snapshot speaker IDs before injection to identify the test speaker.
        speakers_before: set[str] = {
            str(s.get("id", "")) for s in status_before.get("speakers", [])
        }

        # ==================================================================
        # STAGE 1 — initial training (~10 keys)
        # ==================================================================
        print("\n[smoke] ===== Stage 1: initial training (~10 keys) =====")

        # Inject stage-1 facts
        print(f"[smoke] Stage 1: injecting {len(_STAGE1_FACTS)} fact turns via /chat ...")
        _inject_facts(
            base, token, _STAGE1_FACTS, "Stage 1", post_json, ServerHTTPError, ServerUnreachable
        )

        # Resolve test speaker via before/after speaker-delta
        print("[smoke] Stage 1: resolving test speaker via before/after delta ...")
        try:
            status_after_s1_inject = get_json(f"{base}/status", timeout=5.0, token=token)
        except (ServerUnreachable, ServerUnavailable, ServerHTTPError) as exc:
            print(f"[FAIL] /status (post-stage-1-inject) failed: {exc}")
            sys.exit(1)

        speakers_after_s1: set[str] = {
            str(s.get("id", "")) for s in status_after_s1_inject.get("speakers", [])
        }
        new_speakers = speakers_after_s1 - speakers_before

        if len(new_speakers) == 0:
            print(
                "[ABORT] The synthetic embedding matched an existing speaker profile — "
                "no new anonymous speaker was registered.\n"
                "       The test embedding collides with a real speaker's centroid. "
                "Change _SPEAKER_EMBEDDING to a distinct unit vector and retry."
            )
            sys.exit(2)

        if len(new_speakers) > 1:
            print(
                f"[ABORT] {len(new_speakers)} new speaker IDs appeared after injection "
                f"(expected exactly 1): {sorted(new_speakers)}\n"
                "       Cannot safely determine which is the test speaker."
            )
            sys.exit(2)

        test_speaker_id = next(iter(new_speakers))
        speaker_name = next(
            (
                s.get("name", test_speaker_id)
                for s in status_after_s1_inject.get("speakers", [])
                if str(s.get("id", "")) == test_speaker_id
            ),
            test_speaker_id,
        )
        print(f"[smoke]   test speaker: id={test_speaker_id!r} name={speaker_name!r}")

        # Consolidate + wait for stage 1
        _consolidate_and_wait(
            base,
            token,
            "Stage 1",
            get_json,
            post_json,
            ServerHTTPError,
            ServerUnreachable,
            ServerUnavailable,
        )

        # Check keys_count after stage 1
        try:
            status_post_s1 = get_json(f"{base}/status", timeout=5.0, token=token)
        except (ServerUnreachable, ServerUnavailable, ServerHTTPError) as exc:
            print(f"[FAIL] /status (post-stage-1) failed: {exc}")
            sys.exit(1)

        keys_after_s1 = status_post_s1.get("keys_count", 0)
        print(f"[smoke] Stage 1: keys_count={keys_after_s1} (floor={_STAGE1_MIN_KEYS})")

        s1_keys_pass = keys_after_s1 >= _STAGE1_MIN_KEYS
        if not s1_keys_pass:
            print(
                f"[smoke]   [FAIL] keys_count {keys_after_s1} < floor {_STAGE1_MIN_KEYS} — "
                "not enough facts were extracted and trained."
            )
        else:
            print(f"[smoke]   [PASS] keys_count {keys_after_s1} >= {_STAGE1_MIN_KEYS}")

        # Fetch registry and run coverage + keyed-recall checks for stage 1
        print("[smoke] Stage 1: fetching registry via /debug/dump ...")
        entries_s1 = _dump_entries(
            base, token, get_json, ServerHTTPError, ServerUnreachable, ServerUnavailable
        )
        print(f"[smoke]   /debug/dump returned {len(entries_s1)} entries")

        print(f"[smoke] Stage 1: coverage check ({len(_STAGE1_FACTS)} injected facts) ...")
        s1_cov_passes, s1_cov_total = _coverage_check(entries_s1, _STAGE1_FACTS, "Stage 1")
        s1_coverage_pass = s1_cov_passes == s1_cov_total
        print(
            f"[smoke] Stage 1 coverage: {s1_cov_passes}/{s1_cov_total} — "
            f"{'PASS' if s1_coverage_pass else 'FAIL'}"
        )

        print(f"[smoke] Stage 1: keyed recall check ({len(entries_s1)} registry entries) ...")
        s1_recall_passes, s1_recall_total = _keyed_recall_check(
            base, token, entries_s1, "Stage 1", post_json, ServerHTTPError, ServerUnreachable
        )
        s1_recall_pass = s1_recall_passes == s1_recall_total
        print(
            f"[smoke] Stage 1 keyed recall: {s1_recall_passes}/{s1_recall_total} — "
            f"{'PASS' if s1_recall_pass else 'FAIL'}"
        )

        stage1_pass = s1_keys_pass and s1_coverage_pass and s1_recall_pass

        # ==================================================================
        # STAGE 2 — incremental (+5 keys on top)
        # ==================================================================
        print("\n[smoke] ===== Stage 2: incremental (+5 keys) =====")

        # Inject stage-2 facts
        n_s2 = len(_STAGE2_FACTS)
        print(f"[smoke] Stage 2: injecting {n_s2} additional fact turns via /chat ...")
        _inject_facts(
            base, token, _STAGE2_FACTS, "Stage 2", post_json, ServerHTTPError, ServerUnreachable
        )

        # Consolidate + wait for stage 2
        _consolidate_and_wait(
            base,
            token,
            "Stage 2",
            get_json,
            post_json,
            ServerHTTPError,
            ServerUnreachable,
            ServerUnavailable,
        )

        # Check keys_count growth after stage 2
        try:
            status_post_s2 = get_json(f"{base}/status", timeout=5.0, token=token)
        except (ServerUnreachable, ServerUnavailable, ServerHTTPError) as exc:
            print(f"[FAIL] /status (post-stage-2) failed: {exc}")
            sys.exit(1)

        keys_after_s2 = status_post_s2.get("keys_count", 0)
        keys_growth = keys_after_s2 - keys_after_s1
        print(
            f"[smoke] Stage 2: keys_count={keys_after_s2} "
            f"(growth={keys_growth}, min_growth={_STAGE2_MIN_GROWTH})"
        )

        s2_keys_pass = keys_growth >= _STAGE2_MIN_GROWTH
        if not s2_keys_pass:
            print(
                f"[smoke]   [FAIL] key growth {keys_growth} < {_STAGE2_MIN_GROWTH} — "
                "incremental learning did not add enough new keys."
            )
        else:
            print(f"[smoke]   [PASS] key growth {keys_growth} >= {_STAGE2_MIN_GROWTH}")

        # Fetch full registry (stage-1 + stage-2 entries) and run checks
        print("[smoke] Stage 2: fetching registry via /debug/dump ...")
        entries_s2 = _dump_entries(
            base, token, get_json, ServerHTTPError, ServerUnreachable, ServerUnavailable
        )
        print(f"[smoke]   /debug/dump returned {len(entries_s2)} entries")

        print(f"[smoke] Stage 2: coverage check ({len(_STAGE2_FACTS)} new facts) ...")
        s2_cov_passes, s2_cov_total = _coverage_check(entries_s2, _STAGE2_FACTS, "Stage 2")
        s2_coverage_pass = s2_cov_passes == s2_cov_total
        print(
            f"[smoke] Stage 2 coverage: {s2_cov_passes}/{s2_cov_total} — "
            f"{'PASS' if s2_coverage_pass else 'FAIL'}"
        )

        # Sweep ALL entries: proves new keys recall + no catastrophic forgetting.
        print(f"[smoke] Stage 2: keyed recall check ({len(entries_s2)} registry entries) ...")
        s2_recall_passes, s2_recall_total = _keyed_recall_check(
            base, token, entries_s2, "Stage 2", post_json, ServerHTTPError, ServerUnreachable
        )
        s2_recall_pass = s2_recall_passes == s2_recall_total
        print(
            f"[smoke] Stage 2 keyed recall: {s2_recall_passes}/{s2_recall_total} — "
            f"{'PASS' if s2_recall_pass else 'FAIL'}"
        )

        stage2_pass = s2_keys_pass and s2_coverage_pass and s2_recall_pass

        # ==================================================================
        # Rejection / abstention check (after Stage 2 — all 15 facts present)
        # ==================================================================
        print(f"\n[smoke] ===== Rejection check: {len(_REJECTION_PROBES)} absent-fact probes =====")
        print("[smoke] Probing absent facts — the server must abstain, not confabulate ...")
        rejection_passes, rejection_total = _probe_rejections(
            base,
            token,
            test_speaker_id,
            post_json,
            ServerHTTPError,
            ServerUnreachable,
        )

        rejection_pass = rejection_passes == rejection_total
        print(
            f"[smoke] Rejection: {rejection_passes}/{rejection_total} — "
            f"{'PASS' if rejection_pass else 'FAIL'}"
        )

        # ==================================================================
        # Final report
        # ==================================================================
        print("\n[smoke] ===== Results =====")
        print(
            f"[smoke] Stage 1: keys={keys_after_s1}/{_STAGE1_MIN_KEYS}+ "
            f"coverage={s1_cov_passes}/{s1_cov_total} "
            f"recall={s1_recall_passes}/{s1_recall_total} — "
            f"{'PASS' if stage1_pass else 'FAIL'}"
        )
        print(
            f"[smoke] Stage 2: key_growth={keys_growth}/{_STAGE2_MIN_GROWTH}+ "
            f"coverage={s2_cov_passes}/{s2_cov_total} "
            f"recall={s2_recall_passes}/{s2_recall_total} — "
            f"{'PASS' if stage2_pass else 'FAIL'}"
        )
        print(
            f"[smoke] Rejection: {rejection_passes}/{rejection_total} — "
            f"{'PASS' if rejection_pass else 'FAIL'}"
        )

        if stage1_pass and stage2_pass and rejection_pass:
            print("[smoke] PASS — both stages and rejection check passed.")
            _exit_code = 0
        else:
            failed_parts = []
            if not stage1_pass:
                failed_parts.append("Stage 1")
            if not stage2_pass:
                failed_parts.append("Stage 2")
            if not rejection_pass:
                failed_parts.append("Rejection check")
            print(
                f"[smoke] FAIL — {', '.join(failed_parts)} failed.\n"
                "        Check server logs and /status for consolidation errors."
            )
            _exit_code = 1

    finally:
        # Always call /speaker/forget for the test speaker (cleanup on pass and fail).
        if test_speaker_id is not None:
            _cleanup_test_speaker(
                base,
                token,
                test_speaker_id,
                post_json,
                ServerHTTPError,
                ServerUnreachable,
            )
        else:
            print(
                "[smoke] Cleanup skipped — test speaker was never resolved "
                "(injection or speaker-delta step failed before cleanup target was known)."
            )

    sys.exit(_exit_code)


if __name__ == "__main__":
    main()
