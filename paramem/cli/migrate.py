"""Handler for ``paramem migrate <path>``.

Implements the six-step interactive preview + long-poll flow.
The renderer produces the shape-change and tier-change blocks using exact wording
verified byte-for-byte by
``tests/cli/test_migrate.py::test_render_shape_change_block_byte_for_byte_matches_spec``.

Step order
----------
1. Header — state + candidate hash.
2. Simulate-mode notice (if applicable) — asks a dedicated ``y/N`` before
   the ordinary confirm.  ``N`` or EOF → POST cancel → exit 1.
3. Shape-change block (if applicable) — rendered unconditionally (never
   hidden behind ``--verbose``).
4. Tier-classified change list grouped destructive → pipeline_altering →
   operational.
5. Unified diff under a ``Diff (server.yaml):`` header.
6. ``Proceed? [y/N]`` prompt — ``y`` enters the long-poll flow;
   ``N`` or EOF POSTs cancel and exits 1.

Long-poll flow (step 6, after y):
1. STAGING-drift check: GET /migration/status; if not STAGING → stderr + exit 1.
2. POST /migration/confirm.
3. Long-poll /migration/status every LONG_POLL_INTERVAL_SECONDS until gates
   are finished (pass / fail / no_new_sessions / trial_exception) with
   completed_at set.  Ctrl+C → stderr + exit 130.
4. Branch on gates.status:
   - fail / trial_exception → prompt rollback.
   - pass / no_new_sessions → render comparison report + prompt accept/rollback/cancel.
5. Drift checks before accept/rollback POST.

EOF handling
------------
``N`` or EOF (``EOFError`` from ``input()``) on any prompt POSTs
``/migration/cancel`` and exits 1 (Condition 4 from the review).  This
ensures the STAGING stash is always cleared if the operator walks away.

``--json`` mode
---------------
Bypasses all prompts.  Emits the raw ``PreviewResponse`` JSON with
``simulate_mode_override`` at the top level (Condition 8).

404 fallback
------------
Evergreen version-alignment message; no slice labels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from paramem.cli import http_client

# Long-poll interval for trial gate status (seconds between GET /migration/status polls).
LONG_POLL_INTERVAL_SECONDS: float = 2.0

# Terminal gate statuses — polling stops when gates.status is one of these
# AND gates.completed_at is set.
_TERMINAL_GATE_STATUSES: frozenset[str] = frozenset(
    {"pass", "fail", "no_new_sessions", "trial_exception"}
)

# Terminal gate statuses for the base-swap long-poll loop.  Extends the
# standard set with base-swap-specific outcomes set by
# _run_base_swap_orchestration.
_BASE_SWAP_TERMINAL_STATUSES: frozenset[str] = _TERMINAL_GATE_STATUSES | frozenset(
    {
        "phase_a_failed",
        "phase_b_failed",
        "phase_b_model_mismatch",
        "reload_deferred",
    }
)

# Accept-eligible gate statuses (set membership — forward-compat for future gates).
# Keep in sync with _ACCEPT_ELIGIBLE_STATUSES in paramem/server/app.py.
_ACCEPT_ELIGIBLE_STATUSES: frozenset[str] = frozenset({"pass", "no_new_sessions"})

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

_TIER_GLYPHS = {
    "destructive": "⚠",
    "pipeline_altering": "~",
    "operational": "·",
}

_TIER_ORDER = ["destructive", "pipeline_altering", "operational"]


def _render_simulate_notice() -> None:
    """Print the simulate-mode informational notice (no mode change).

    The trial runs in the candidate's configured mode — for a simulate-mode
    candidate that means extraction + graph persistence with no weights trained.
    """
    print("  ────────────────────────────────────────")
    print("  NOTICE — candidate is simulate-mode")
    print("  ────────────────────────────────────────")
    print("  The candidate config has consolidation.mode: simulate. The trial")
    print("  runs in simulate mode: extraction + graph persistence, no weights.")


def _render_mode_switch_preview(mode_switch: dict) -> None:
    """Print the mode-switch preview notice (pure consolidation.mode change)."""
    print("  ────────────────────────────────────────")
    print("  NOTICE — persistence-mode switch")
    print("  ────────────────────────────────────────")
    print(
        f"  consolidation.mode: {mode_switch.get('from')} → {mode_switch.get('to')} "
        f"({mode_switch.get('direction')})"
    )
    print("  This is applied DIRECTLY — no trial, accept, or rollback. The")
    print(f"  store is rebuilt per-tier via {mode_switch.get('applies_via')} at the")
    print("  next consolidation, with a 1.0 recall gate and source-mode fallback")
    print("  until every tier passes.")


def _render_mode_switch_applied(mode_switch: dict) -> None:
    """Print confirmation that a mode-switch migration was applied directly."""
    print()
    print(
        f"  Persistence-mode switch applied: {mode_switch.get('from')} → {mode_switch.get('to')}."
    )
    print("  The active-store rebuild runs at the next consolidation (per-tier,")
    print("  1.0 recall gate, source-mode fallback). Run `paramem migrate-status`")
    print("  or POST /consolidate to drive it.")


def _render_base_change_preview(base_change: dict) -> None:
    """Print the base-model-change warning block in the preview output.

    Rendered unconditionally when the candidate config changes the ``model:``
    field (i.e. the base model).  Mirrors ``_render_mode_switch_preview`` in
    style: a banner line, a description of what the migration entails, and a
    note about the in-process reload.

    Parameters
    ----------
    base_change:
        Dict with ``old_model``, ``new_model``, and ``consequence`` keys as
        returned by ``compute_base_change``.
    """
    old_model = base_change.get("old_model", "")
    new_model = base_change.get("new_model", "")
    print("  ────────────────────────────────────────")
    print("  ⚠  BASE MODEL CHANGE — DESTRUCTIVE")
    print("  ────────────────────────────────────────")
    print(f"  model: {old_model!r} → {new_model!r}")
    print("  Phase A (capture): all keyed facts are reconstructed from the live")
    print("  model into encrypted per-tier graph.json files; adapter weight slots")
    print("  are deleted. A bundle backup is taken first (rollback-safe).")
    print("  Phase B (relearn): the new base model is loaded in-process (server is")
    print("  briefly cloud-only during reload), then all tiers are retrained and")
    print("  gated on 100% recall. No server restart is required.")
    print("  Rollback restores the prior base model from the pre-migration bundle.")


def _render_shape_change_block(shape_changes: list[dict]) -> None:
    """Print the shape-change warning block.

    Exact wording is verified byte-for-byte by the CLI test suite.
    """
    if not shape_changes:
        return
    print("  ────────────────────────────────────────")
    print("  ⚠  SHAPE CHANGE — DESTRUCTIVE")
    print("  ────────────────────────────────────────")
    # Group by adapter name preserving order of first appearance.
    seen: list[str] = []
    by_adapter: dict[str, list[dict]] = {}
    for change in shape_changes:
        adapter = change["adapter"]
        if adapter not in by_adapter:
            seen.append(adapter)
            by_adapter[adapter] = []
        by_adapter[adapter].append(change)

    for adapter_name in seen:
        rows = by_adapter[adapter_name]
        for i, row in enumerate(rows):
            field = row["field"]
            old_val = row["old_value"]
            new_val = row["new_value"]
            consequence = row["consequence"]

            if field == "rank":
                print(f"  {adapter_name}:  rank {old_val} → {new_val}")
                print(f"             {consequence}")
            elif field == "alpha":
                print(f"  {adapter_name}:  alpha {old_val} → {new_val}")
                print(f"             {consequence}")
            elif field == "target_modules":
                old_list = [str(x) for x in (old_val or [])]
                new_set = set(str(x) for x in (new_val or []))
                old_set = set(old_list)
                added = sorted(new_set - old_set)
                removed = sorted(old_set - new_set)
                old_str = "{" + ",".join(old_list) + "}"
                if added and removed:
                    delta_str = "+{" + ",".join(added) + "} -{" + ",".join(removed) + "}"
                elif added:
                    delta_str = "+{" + ",".join(added) + "}"
                else:
                    delta_str = "-{" + ",".join(removed) + "}"
                print(f"  {adapter_name}:  target_modules {old_str} → {delta_str}")
                print(f"             {consequence}")
            elif field == "dropout":
                print(f"  {adapter_name}:  dropout {old_val} → {new_val}")
                print(f"             {consequence}")
            else:
                print(f"  {adapter_name}:  {field} {old_val} → {new_val}")
                print(f"             {consequence}")


def _render_tier_list(tier_diff: list[dict]) -> None:
    """Print tier-classified change rows grouped destructive → pipeline → operational."""
    if not tier_diff:
        return
    for tier_name in _TIER_ORDER:
        rows_for_tier = [r for r in tier_diff if r.get("tier") == tier_name]
        if not rows_for_tier:
            continue
        glyph = _TIER_GLYPHS.get(tier_name, "?")
        print(f"  [{tier_name.upper()}]")
        for row in rows_for_tier:
            path = row["dotted_path"]
            old_val = row["old_value"]
            new_val = row["new_value"]
            print(f"  {glyph} {path}: {old_val!r} → {new_val!r}")


def _render_unified_diff(unified_diff: str) -> None:
    """Print the unified diff under a header."""
    print("  Diff (server.yaml):")
    if not unified_diff:
        print("  (no textual differences)")
    else:
        for line in unified_diff.splitlines():
            print(f"  {line}")


def _post_cancel(server_url: str) -> None:
    """POST /migration/cancel, ignoring errors (best-effort cleanup)."""
    try:
        http_client.post_json(f"{server_url}/migration/cancel")
    except Exception:
        pass


def _render_comparison_report(report: dict | None) -> None:
    """Render the comparison report table to stdout.

    Prints a header row, one row per metric, a blank line, then the
    operator_line prefixed with ``> ``.  Skips rendering when *report* is
    ``None``.

    Parameters
    ----------
    report:
        Comparison report dict (schema_version 1), or ``None`` to skip.
    """
    if not report:
        return
    rows = report.get("rows") or []
    operator_line = report.get("operator_line", "")

    print("  Comparison report:")
    print(f"  {'metric':<45} {'pre-trial':<12} {'trial':<12}")
    print(f"  {'-' * 45} {'-' * 12} {'-' * 12}")
    for row in rows:
        metric = row.get("metric", "")
        pre = row.get("pre_trial", "—")
        trial = row.get("trial", "—")
        print(f"  {metric:<45} {str(pre):<12} {str(trial):<12}")
    print()
    if operator_line:
        print(f"  > {operator_line}")
    print()


def _render_failed_gates(gates: dict, gs: str) -> None:
    """Render per-gate breakdown when the trial ends in fail or trial_exception.

    For a ``fail`` status, iterates ``gates["details"]`` and prints each
    gate's name and status.  Any gate whose status is not ``"pass"`` or
    ``"skipped"`` also has its ``reason`` printed.

    For a ``trial_exception`` status, prints the top-level exception text
    from ``gates["exception"]`` (no ``details`` list is present on that
    path).

    Parameters
    ----------
    gates:
        The ``poll_status["gates"]`` dict from the long-poll response.
    gs:
        The terminal gate status (``"fail"`` or ``"trial_exception"``).
    """
    if gs == "trial_exception":
        exception_text = gates.get("exception", "")
        if exception_text:
            print(f"  Exception: {exception_text}")
        return

    # gs == "fail": render per-gate breakdown from details list.
    details = gates.get("details") or []
    if not details:
        return
    print("  Gate results:")
    for entry in details:
        gate_num = entry.get("gate", "?")
        name = entry.get("name", "")
        status = entry.get("status", "")
        print(f"    gate {gate_num} ({name}): {status}")
        if status not in ("pass", "skipped"):
            reason = entry.get("reason")
            if reason:
                print(f"      reason: {reason}")


def _render_base_swap_terminal(gs: str, gates: dict) -> None:
    """Render the terminal state of a base-swap migration to stdout/stderr.

    Called after the base-swap long-poll loop exits on a terminal status.
    Covers all statuses set by ``_run_base_swap_orchestration``.

    Parameters
    ----------
    gs:
        Terminal gate status (one of the ``_BASE_SWAP_TERMINAL_STATUSES``).
    gates:
        The ``poll_status["gates"]`` dict from the last poll response.
    """
    message = gates.get("message", "")
    exception_text = gates.get("exception", "")

    if gs == "pass":
        print("  Base-swap migration complete. Server is now running on the new model.")
        if message:
            print(f"  {message}")
    elif gs == "reload_deferred":
        # VRAM insufficient: Phase A done, reload waiting for VRAM to free.
        print("  Base-swap Phase A complete. Reload deferred — insufficient VRAM.")
        print(
            "  The server will resume Phase B automatically once VRAM is available "
            "(POST /gpu/acquire triggers this)."
        )
        if message:
            print(f"  {message}")
    elif gs == "phase_b_model_mismatch":
        mismatch_reason = gates.get("mismatch_reason", "")
        print(
            "  Base-swap aborted: loaded model identity mismatch (Phase B guard).",
            file=sys.stderr,
        )
        if mismatch_reason:
            print(f"  Detail: {mismatch_reason}", file=sys.stderr)
        print(
            "  The bundle and marker are preserved. "
            "Run `paramem migrate --rollback` to restore the prior model.",
            file=sys.stderr,
        )
    elif gs == "phase_a_failed":
        print(
            "  Base-swap Phase A failed. The prior model is still loaded.",
            file=sys.stderr,
        )
        if exception_text:
            print(f"  Exception: {exception_text}", file=sys.stderr)
        print(
            "  The bundle is preserved. "
            "Run `paramem migrate --rollback` to restore the prior config.",
            file=sys.stderr,
        )
    elif gs == "phase_b_failed":
        print(
            "  Base-swap Phase B failed (recall gate or tier error). "
            "The new model is loaded but adapters are not retrained.",
            file=sys.stderr,
        )
        if exception_text:
            print(f"  Exception: {exception_text}", file=sys.stderr)
        print(
            "  Run `paramem migrate --rollback` to restore from the bundle, "
            "or retry Phase B by calling POST /gpu/acquire.",
            file=sys.stderr,
        )
    else:
        # Forward-compat: unknown status (future orchestration phases).
        print(f"  Base-swap finished with status: {gs!r}.", file=sys.stderr)
        if message:
            print(f"  {message}", file=sys.stderr)
        if exception_text:
            print(f"  Exception: {exception_text}", file=sys.stderr)


# Poll interval for the R-PORT auto-restart health check (seconds).
_RESTART_POLL_INTERVAL_S: float = 3.0
# Maximum time to wait for the server to come back up after an R-PORT restart.
_RESTART_POLL_TIMEOUT_S: float = 120.0


def _poll_until_healthy(server_url: str) -> bool:
    """Poll ``GET /status`` until the server is healthy or the timeout expires.

    Tolerates ``ServerUnreachable`` (connection refused during the bounce
    window).  Returns ``True`` when the server responds and does not report
    ``boot_degraded``.  Returns ``False`` on timeout.

    Parameters
    ----------
    server_url:
        Base server URL.

    Returns
    -------
    bool
        ``True`` when healthy, ``False`` on timeout.
    """
    import time as _time

    deadline = _time.monotonic() + _RESTART_POLL_TIMEOUT_S
    while _time.monotonic() < deadline:
        try:
            status = http_client.get_json(f"{server_url}/status")
            if status.get("mode") and not status.get("boot_degraded"):
                return True
        except http_client.ServerUnreachable:
            pass  # expected during the bounce window
        except Exception:
            pass
        _time.sleep(_RESTART_POLL_INTERVAL_S)
    return False


def _render_apply_result(result: dict, server_url: str) -> None:
    """Render the live-apply result from an accept or rollback response.

    Handles all outcomes:

    - ``applied_live=True, skipped="no_change"`` → no-op (rollback case).
    - ``applied_live=True, restart_required_reason=None`` → full live apply.
    - ``restart_required_reason in {stt_port_change, tts_port_change}`` →
      R-PORT: if ``restart_eligible=True``, prompt operator for consent; on
      ``y`` run ``restart_hint`` via subprocess and poll until healthy; on
      ``N``/EOF print the command and exit cleanly.  If port-in-use (no
      ``restart_eligible``), print the error and the manual hint.
    - ``restart_required_reason="paths_change"`` → R-PATHS: prominent warning
      that data is NOT migrated automatically; print restart_hint.
    - Failure (``applied_live=False``, other reason) → restart-hint fallback.

    The server does NOT self-fire a restart for R-PORT.  The CLI is the sole
    restart trigger — always gated on operator consent.

    Parameters
    ----------
    result:
        Parsed response body from ``/migration/accept`` or
        ``/migration/rollback``.
    server_url:
        Base server URL (used for the poll-until-healthy call on R-PORT).
    """
    import subprocess  # noqa: PLC0415

    applied_live = result.get("applied_live", False)
    skipped = result.get("skipped")
    reason = result.get("restart_required_reason")
    restart_eligible = result.get("restart_eligible", False)
    restart_hint = result.get("restart_hint", "systemctl --user restart paramem-server")
    cloud_only_reason = result.get("cloud_only_reason")
    port_in_use = result.get("port_in_use_reason")

    if applied_live and skipped == "no_change":
        print(
            "  No config change to apply (rollback restored the prior config); server stays local."
        )
        return

    if applied_live and reason is None:
        print("  Migration applied live; server is back in local mode.")
        return

    if reason in ("stt_port_change", "tts_port_change"):
        if port_in_use:
            # Pre-flight declined — port not bindable.
            print(
                f"  Port not bindable ({port_in_use}); restart not possible.",
                file=sys.stderr,
            )
            print(f"  Free the port and restart manually: {restart_hint}", file=sys.stderr)
            return
        if restart_eligible:
            # Port pre-flighted OK.  Prompt operator for restart consent.
            try:
                answer = input(
                    "  This change needs a full restart (brief outage: /chat, STT/TTS,"
                    " HA satellites drop ~30s). Restart now? [y/N] "
                )
            except EOFError:
                answer = "n"
            if answer.strip().lower() == "y":
                print(f"  Running: {restart_hint}")
                try:
                    subprocess.run(restart_hint.split(), check=True)
                except Exception as exc:  # noqa: BLE001 — boundary: external command
                    print(f"  Restart command failed: {exc}", file=sys.stderr)
                    print(f"  Run manually: {restart_hint}", file=sys.stderr)
                    return
                print("  Polling until server is healthy...")
                healthy = _poll_until_healthy(server_url)
                if healthy:
                    print("  Server is back and healthy.")
                else:
                    print(
                        "  Timed out waiting for server. "
                        "Check: journalctl --user -u paramem-server",
                        file=sys.stderr,
                    )
                    print(f"  Manual restart: {restart_hint}", file=sys.stderr)
            else:
                print(f"  Restart deferred. Run when ready: {restart_hint}")
        else:
            # Not eligible for prompted restart (unexpected state).
            print(
                f"  Port change ({reason}) requires a restart: {restart_hint}",
                file=sys.stderr,
            )
        return

    if reason == "paths_change":
        print(
            "  PATH CHANGE — data is NOT migrated automatically.",
            file=sys.stderr,
        )
        print(
            "  The path change is on disk and takes effect on the NEXT restart.",
            file=sys.stderr,
        )
        print(
            "  Move adapters, registry, and pending sessions to the new path BEFORE restarting.",
            file=sys.stderr,
        )
        print(f"  When ready: {restart_hint}", file=sys.stderr)
        return

    # Apply failed or other unexpected reason — restart-hint fallback.
    if cloud_only_reason:
        print(
            f"  Apply failed ({cloud_only_reason}); server is cloud-only — restart to apply.",
            file=sys.stderr,
        )
    else:
        print("  Config is on disk; restart to apply.", file=sys.stderr)
    print(f"  Restart: {restart_hint}", file=sys.stderr)


def _get_migration_status(server_url: str) -> dict:
    """GET /migration/status and return the parsed dict.

    Raises
    ------
    http_client.ServerUnreachable
        If the server is unreachable.
    http_client.ServerHTTPError
        On non-2xx, non-404 responses.
    http_client.ServerUnavailable
        On 404 (endpoint not implemented).
    """
    return http_client.get_json(f"{server_url}/migration/status")


def _do_accept_with_drift_check(server_url: str) -> int:
    """GET /migration/status, verify TRIAL, then POST /migration/accept.

    Verifies the server is still in TRIAL state (drift check) before POSTing accept.
    If state has drifted out of TRIAL, aborts with an error instead of sending the POST.

    Parameters
    ----------
    server_url:
        Base server URL.

    Returns
    -------
    int
        0 on success, 1 on drift / HTTP error.
    """
    try:
        status = _get_migration_status(server_url)
    except (http_client.ServerUnreachable, http_client.ServerHTTPError) as exc:
        print(
            f"paramem migrate: could not verify trial state before accept: {exc}",
            file=sys.stderr,
        )
        return 1

    if status.get("state") != "TRIAL":
        print(
            "paramem migrate: server state drifted (trial lost) — "
            "aborting; run `paramem migrate-status`.",
            file=sys.stderr,
        )
        return 1

    try:
        result = http_client.post_json(f"{server_url}/migration/accept")
    except (http_client.ServerHTTPError, http_client.ServerUnreachable) as exc:
        print(f"paramem migrate: accept failed: {exc}", file=sys.stderr)
        return 1

    print("  Migration accepted.")
    _render_apply_result(result, server_url)
    return 0


def _do_rollback_with_drift_check(server_url: str) -> int:
    """GET /migration/status, verify TRIAL, then POST /migration/rollback.

    Verifies the server is still in TRIAL state (drift check) before POSTing rollback.
    After a successful POST, inspects the response body for ``archive_warning`` — the server
    returns HTTP 207 for degraded rollback, but ``post_json`` passes it through
    as a plain dict (207 < 400 so no exception is raised).

    Parameters
    ----------
    server_url:
        Base server URL.

    Returns
    -------
    int
        0 on success (including degraded success with archive_warning),
        1 on drift / HTTP error.
    """
    try:
        status = _get_migration_status(server_url)
    except (http_client.ServerUnreachable, http_client.ServerHTTPError) as exc:
        print(
            f"paramem migrate: could not verify trial state before rollback: {exc}",
            file=sys.stderr,
        )
        return 1

    if status.get("state") != "TRIAL":
        print(
            "paramem migrate: server state drifted (trial lost) — "
            "aborting; run `paramem migrate-status`.",
            file=sys.stderr,
        )
        return 1

    try:
        result = http_client.post_json(f"{server_url}/migration/rollback")
    except (http_client.ServerHTTPError, http_client.ServerUnreachable) as exc:
        print(f"paramem migrate: rollback failed: {exc}", file=sys.stderr)
        return 1

    archive_warning = result.get("archive_warning")
    if archive_warning is not None:
        # Degraded rollback: config restored but trial adapter rotation failed.
        # Primary action succeeded — exit 0; warn on stderr (operator action needed).
        print("  Migration rolled back (config restored; trial adapter rotation failed).")
        print(
            f"  archive_warning: {archive_warning.get('message', '')}",
            file=sys.stderr,
        )
    else:
        print("  Migration rolled back.")
    _render_apply_result(result, server_url)
    return 0


def _run_long_poll_flow(server_url: str) -> int:
    """Run the full long-poll flow after the operator confirms 'y'.

    Runs the full long-poll flow after the operator confirms 'y':

    1. STAGING-drift check: GET /migration/status; abort if not STAGING.
    2. POST /migration/confirm.
    3. Long-poll /migration/status until gates are terminal.
    4. Branch on gates.status.
    5. Drift checks before accept/rollback.

    For base-swap migrations (``confirm_resp["base_swap"] == True``), step 3
    polls against the extended ``_BASE_SWAP_TERMINAL_STATUSES`` set and step 4
    renders base-swap-specific terminal messages instead of the accept/rollback
    3-way prompt (base-swap success does not require an explicit accept).

    Parameters
    ----------
    server_url:
        Base server URL.

    Returns
    -------
    int
        0 on accept/deferred, 1 on rollback/fail/drift, 130 on Ctrl+C.
    """
    # Step 1: STAGING-drift check before confirm — abort if server is no longer in STAGING state.
    try:
        pre_status = _get_migration_status(server_url)
    except (http_client.ServerUnreachable, http_client.ServerHTTPError) as exc:
        print(f"paramem migrate: could not verify staging state: {exc}", file=sys.stderr)
        return 1

    if pre_status.get("state") != "STAGING":
        print(
            "Server restarted during preview — candidate not staged.\n"
            "Rerun `paramem migrate <path>` to restage.",
            file=sys.stderr,
        )
        return 1

    # Step 2: POST /migration/confirm.
    try:
        confirm_resp = http_client.post_json(f"{server_url}/migration/confirm")
    except (http_client.ServerHTTPError, http_client.ServerUnreachable) as exc:
        print(f"paramem migrate: confirm failed: {exc}", file=sys.stderr)
        return 1

    # Mode-switch fast path: a pure consolidation.mode change is applied
    # directly (no trial). The server swapped the config live and armed the
    # active-store rebuild; there is nothing to poll, accept, or roll back.
    mode_switch = (confirm_resp or {}).get("mode_switch")
    if mode_switch:
        _render_mode_switch_applied(mode_switch)
        return 0

    # Base-swap path: the confirm launched the in-process base-model-swap
    # orchestration.  Poll with the extended terminal-status set; render the
    # terminal state without an accept/rollback prompt (the swap is
    # self-completing — no explicit accept is needed on success).
    is_base_swap = bool((confirm_resp or {}).get("base_swap"))
    terminal_statuses = _BASE_SWAP_TERMINAL_STATUSES if is_base_swap else _TERMINAL_GATE_STATUSES

    # Step 3: Long-poll until gates are terminal.
    if is_base_swap:
        print(
            "  Base-swap migration launched. Polling for orchestration progress "
            "(Ctrl+C to interrupt; migration continues on server)."
        )
    else:
        print("  Confirming trial... polling for gate results (Ctrl+C to interrupt).")
    try:
        while True:
            try:
                poll_status = _get_migration_status(server_url)
            except (http_client.ServerHTTPError, http_client.ServerUnreachable) as exc:
                print(
                    f"paramem migrate: HTTP failure mid-poll: {exc}",
                    file=sys.stderr,
                )
                return 2

            gates = poll_status.get("gates") or {}
            gs = gates.get("status", "")
            if gs in terminal_statuses and gates.get("completed_at"):
                break

            time.sleep(LONG_POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print(
            "\nparamem migrate: poll interrupted; server continues trial in background. "
            "Run `paramem migrate-status` to check later.",
            file=sys.stderr,
        )
        return 130

    # Step 4: Branch on gates.status.
    if is_base_swap:
        # Base-swap terminal: render phase-specific message; no accept prompt.
        _render_base_swap_terminal(gs, gates)
        # Exit code: 0 on pass or reload_deferred (deferred is informational,
        # the server will resume automatically); non-zero on failure/mismatch.
        if gs == "pass":
            return 0
        if gs == "reload_deferred":
            return 2
        return 1

    if gs in _ACCEPT_ELIGIBLE_STATUSES:
        # Render comparison report and offer accept / rollback / cancel.
        comparison_report = poll_status.get("comparison_report")
        _render_comparison_report(comparison_report)

        try:
            answer = input("  accept / rollback / cancel [c]: ").lower().strip()
        except EOFError:
            print(
                "  Decision deferred; trial remains active. "
                "Run `paramem migrate-accept` or `paramem migrate-rollback` later."
            )
            return 0

        if answer in ("a", "accept"):
            return _do_accept_with_drift_check(server_url)
        if answer in ("r", "rollback"):
            return _do_rollback_with_drift_check(server_url)
        # Default: cancel/defer (c, empty, EOF, anything else).
        print(
            "  Decision deferred; trial remains active. "
            "Run `paramem migrate-accept` or `paramem migrate-rollback` later."
        )
        return 0

    else:
        # gates_status in {fail, trial_exception} — rollback only.
        print(
            f"  Trial gates finished with status: {gs!r}. "
            "Rollback is the only valid action (accept is blocked)."
        )
        _render_failed_gates(gates, gs)
        try:
            answer = input("  Rollback now? [y/N] ").lower().strip()
        except EOFError:
            answer = ""

        if answer == "y":
            return _do_rollback_with_drift_check(server_url)
        print(
            "  Trial failed; run `paramem migrate-rollback` when ready.",
            file=sys.stderr,
        )
        return 1


def render_preview(result: dict, server_url: str) -> int:
    """Execute the full six-step interactive renderer.

    Parameters
    ----------
    result:
        Parsed ``PreviewResponse`` dict from the server.
    server_url:
        Base server URL (for cancel calls).

    Returns
    -------
    int
        0 on ``Proceed? y``, 1 on cancel/EOF/N, 2 on unexpected errors.
    """
    candidate_hash = result.get("candidate_hash", "")
    candidate_path = result.get("candidate_path", "")
    simulate_mode = result.get("simulate_mode_override", False)
    shape_changes = result.get("shape_changes") or []
    tier_diff = result.get("tier_diff") or []
    unified_diff = result.get("unified_diff", "")

    # 1. Header
    print(f"Migration preview: {candidate_path}")
    print(f"  candidate hash: {candidate_hash}")
    print()

    # 2. Mode / persistence notices.
    mode_switch = result.get("mode_switch")
    base_change = result.get("base_change")
    if base_change:
        # Candidate changes the base model: destructive, requires Phase A + restart.
        _render_base_change_preview(base_change)
        print()
    elif mode_switch:
        # Pure consolidation.mode change → applied directly via the active-store
        # rebuild (no trial / accept / rollback).  Surface the chosen path.
        _render_mode_switch_preview(mode_switch)
        print()
    elif simulate_mode:
        # Candidate runs in simulate mode (no mode change): the trial runs in
        # simulate mode too — extraction + graph persistence, no weights trained.
        _render_simulate_notice()
        print()

    # 3. Shape-change block (unconditional — never hidden behind --verbose)
    if shape_changes:
        _render_shape_change_block(shape_changes)
        print()

    # 4. Tier-classified change list
    if tier_diff:
        _render_tier_list(tier_diff)
        print()

    # 5. Unified diff
    _render_unified_diff(unified_diff)
    print()

    # Pre-flight check short-circuit.
    # If the server detected a pre-flight failure (e.g. disk pressure), the
    # state is still LIVE (Decision A) — no /migration/cancel POST is needed.
    pre_flight_fail = result.get("pre_flight_fail")
    if pre_flight_fail is not None:
        if pre_flight_fail == "disk_pressure":
            used_gb = result.get("pre_flight_disk_used_gb") or 0.0
            cap_gb = result.get("pre_flight_disk_cap_gb") or 0.0
            # Spec L582–586 wording, with Decision-C CLI naming.
            print(
                f"Migration will fail at step 2 (pre-migration backup) — backup store\n"
                f"at {used_gb:.2f} / {cap_gb:.2f} GB. Run `paramem backup-prune` or raise\n"
                f"security.backups.max_total_disk_gb before retrying.",
                file=sys.stderr,
            )
        else:
            # Forward-compat: unknown pre-flight code (future slice).
            print(
                f"paramem migrate: pre-flight check failed: {pre_flight_fail!r}",
                file=sys.stderr,
            )
        # No _post_cancel — state is still LIVE (Decision A); nothing to cancel.
        return 1

    # 6. Proceed prompt
    try:
        answer = input("  Proceed? [y/N] ").strip().lower()
    except EOFError:
        _post_cancel(server_url)
        return 1

    if answer == "y":
        return _run_long_poll_flow(server_url)
    else:
        _post_cancel(server_url)
        return 1


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    """Execute the ``migrate`` subcommand.

    Validates that ``args.path`` is an absolute path, then POSTs to
    ``/migration/preview``.  On 404, prints a version-alignment message.
    On ``ServerUnreachable``, prints a troubleshooting hint and returns 2.

    With ``--json``, emits the raw ``PreviewResponse`` JSON (bypasses all
    prompts; ``simulate_mode_override`` is a top-level field — Condition 8).

    Parameters
    ----------
    args:
        Parsed namespace from the ``migrate`` subparser.

    Returns
    -------
    int
        0 on success / proceed-y, 1 on cancel / HTTP error / 404,
        2 on connection failure.
    """
    candidate = Path(args.path)
    if not candidate.is_absolute():
        print(
            f"paramem migrate: path must be absolute, got {args.path!r}.\n"
            "Example: paramem migrate /home/user/configs/server-new.yaml",
            file=sys.stderr,
        )
        return 1

    server_url = args.server_url
    url = f"{server_url}/migration/preview"
    try:
        result = http_client.post_json(url, {"candidate_path": str(candidate)})
    except http_client.ServerUnavailable:
        print(
            f"paramem migrate: the server at {server_url} does not implement"
            " /migration/preview.\n"
            "Check `paramem --version` and server version are aligned. "
            "migrate-accept and migrate-rollback are separate subcommands for non-interactive use.",
            file=sys.stderr,
        )
        return 1
    except http_client.ServerUnreachable:
        print(
            f"paramem migrate: server unreachable at {server_url}.\n"
            "Is paramem-server running? `systemctl --user status paramem-server`.",
            file=sys.stderr,
        )
        return 2
    except http_client.ServerHTTPError as exc:
        print(
            f"paramem migrate: server returned HTTP {exc.status_code} from {exc.url}.\n"
            f"{exc.body.strip() or '(empty response body)'}",
            file=sys.stderr,
        )
        return 1

    if getattr(args, "json", False):
        # --json: emit raw PreviewResponse with simulate_mode_override at top level
        # (Condition 8).  simulate_mode_override is already a top-level field in
        # the server response dict — this is a pass-through, no remapping needed.
        print(json.dumps(result, indent=2))
        return 0

    return render_preview(result, server_url)
