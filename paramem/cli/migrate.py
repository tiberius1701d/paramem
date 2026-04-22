"""Handler for ``paramem migrate <path>``.

Implements the six-step interactive preview + long-poll flow (Slice 3b.3).
The renderer follows the exact spec wording from plan §4 / §L243–271
(byte-for-byte compliance is verified by
``tests/cli/test_migrate.py::test_render_shape_change_block_byte_for_byte_matches_spec``).

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
6. ``Proceed? [y/N]`` prompt — ``y`` enters the long-poll flow (Slice 3b.3);
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

# Long-poll interval for trial gate status (spec §9).
LONG_POLL_INTERVAL_SECONDS: float = 2.0

# Terminal gate statuses — polling stops when gates.status is one of these
# AND gates.completed_at is set.
_TERMINAL_GATE_STATUSES: frozenset[str] = frozenset(
    {"pass", "fail", "no_new_sessions", "trial_exception"}
)

# Accept-eligible gate statuses (set membership — forward-compat for Slice 4).
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
    """Print the simulate-mode notice (spec §L243–250, byte-for-byte)."""
    print("  ────────────────────────────────────────")
    print("  ⚠  NOTICE — candidate is simulate-mode")
    print("  ────────────────────────────────────────")
    print("  The candidate config has consolidation.mode: simulate, but the trial")
    print("  WILL FORCE consolidation.mode: train to exercise the training path.")
    print("  This consumes VRAM and GPU time and writes a trial adapter.")
    print("  Accept this override? [y/N]", end=" ", flush=True)


def _render_shape_change_block(shape_changes: list[dict]) -> None:
    """Print the shape-change warning block (spec §L257–271, byte-for-byte)."""
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

    Implements the spec §L235 drift check before accept.

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

    print("  Migration accepted. Restart the server for the new config to take effect.")
    for key, value in result.items():
        print(f"  {key}: {value}")
    return 0


def _do_rollback_with_drift_check(server_url: str) -> int:
    """GET /migration/status, verify TRIAL, then POST /migration/rollback.

    Implements the spec §L235 drift check before rollback.  After a successful
    POST, inspects the response body for ``archive_warning`` — the server
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
        print("  Migration rolled back. Restart the server for config A to take effect.")
    for key, value in result.items():
        if key != "archive_warning":
            print(f"  {key}: {value}")
    return 0


def _run_long_poll_flow(server_url: str) -> int:
    """Run the full long-poll flow after the operator confirms 'y'.

    Implements spec §9 / CLI steps 1–5:

    1. STAGING-drift check (spec §L230–235).
    2. POST /migration/confirm.
    3. Long-poll /migration/status until gates are terminal.
    4. Branch on gates.status.
    5. Drift checks before accept/rollback.

    Parameters
    ----------
    server_url:
        Base server URL.

    Returns
    -------
    int
        0 on accept/deferred, 1 on rollback/fail/drift, 130 on Ctrl+C.
    """
    # Step 1: STAGING-drift check before confirm (spec §L230–235).
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
        http_client.post_json(f"{server_url}/migration/confirm")
    except (http_client.ServerHTTPError, http_client.ServerUnreachable) as exc:
        print(f"paramem migrate: confirm failed: {exc}", file=sys.stderr)
        return 1

    # Step 3: Long-poll until gates are terminal.
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
            if gs in _TERMINAL_GATE_STATUSES and gates.get("completed_at"):
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

    # 2. Simulate-mode notice (dedicated confirm before ordinary proceed)
    if simulate_mode:
        _render_simulate_notice()
        try:
            answer = input().strip().lower()
        except EOFError:
            _post_cancel(server_url)
            return 1
        if answer != "y":
            _post_cancel(server_url)
            return 1
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
    ``/migration/preview``.  On 404, prints a version-alignment message
    listing the four Slice 3b.1 endpoints.  On ``ServerUnreachable``,
    prints a troubleshooting hint and returns 2.

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
