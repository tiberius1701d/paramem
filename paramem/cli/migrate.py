"""Handler for ``paramem migrate <path>``.

Implements the Slice 3b.1 six-step preview renderer.  The renderer follows
the exact spec wording from plan §4 / §L243–271 (byte-for-byte compliance is
verified by
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
6. ``Proceed? [y/N]`` prompt — ``y`` prints "3b.2 not yet implemented" and
   exits 0; ``N`` or EOF POSTs cancel and exits 1.

EOF handling
------------
``N`` or EOF (``EOFError`` from ``input()``) on either prompt POSTs
``/migration/cancel`` and exits 1 (Condition 4 from the review).  This
ensures the STAGING stash is always cleared if the operator walks away.

``--json`` mode
---------------
Bypasses all prompts.  Emits the raw ``PreviewResponse`` JSON with
``simulate_mode_override`` at the top level (Condition 8).

404 fallback
------------
Lists the four Slice 3b.1 endpoints and names confirm/accept/rollback as
Slice 3b.2-pending.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paramem.cli import http_client

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
        print("  Slice 3b.2 not yet implemented — confirm/trial/accept ship in Slice 3b.2.")
        return 0
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
            f"paramem migrate: the server at {server_url} returned 404 for\n"
            f"/migration/preview.\n"
            "Slice 3b.1 ships /migration/preview, /cancel, /status, /diff.\n"
            "/migration/confirm, /accept, /rollback ship in Slice 3b.2.\n"
            "Check `paramem --version` and server version are aligned.",
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
