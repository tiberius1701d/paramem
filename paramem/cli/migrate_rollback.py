"""Handler for ``paramem migrate-rollback``.

POSTs to ``/migration/rollback``.  Restores config A from backup, archives
the trial adapter, and returns the server to LIVE state.  Valid from TRIAL at
any time (no gate-status check — spec §L208).

This module mirrors the ``migrate_cancel.py`` shape exactly — thin stateless
HTTP client, server owns all state (spec §L187).
"""

from __future__ import annotations

import argparse
import json
import sys

from paramem.cli import http_client


def _parse_detail(body: str) -> dict:
    """Attempt to parse an HTTP error response body as JSON.

    Returns the ``detail`` sub-dict if present, else an empty dict.  Never
    raises — parsing failures are best-effort.

    Parameters
    ----------
    body:
        Raw response body string from the server.

    Returns
    -------
    dict
        Parsed ``detail`` dict, or ``{}`` on parse failure.
    """
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict) and isinstance(parsed.get("detail"), dict):
            return parsed["detail"]
        return {}
    except (json.JSONDecodeError, ValueError):
        return {}


def run(args: argparse.Namespace) -> int:
    """Execute the ``migrate-rollback`` subcommand.

    POSTs to ``/migration/rollback``, prints the response, and exits 0 on
    success.  Handles degraded success (rotation failed but config restored)
    by inspecting the response body for an ``archive_warning`` key — HTTP 207
    is returned by the server but ``post_json`` passes it through as a plain
    dict because 207 < 400.

    Special handling:

    - Response body contains ``archive_warning``: primary action (config
      restore) succeeded; trial adapter archive failed.  Print warning; exit 0.
    - 409 ``not_trial``: no trial active — intent already satisfied; exit 0
      (idempotent/informational, matches ``migrate-cancel`` / ``not_staging``
      precedent).
    - ``ServerUnreachable``: server not reachable; exit 2.

    Parameters
    ----------
    args:
        Parsed namespace from the ``migrate-rollback`` subparser.

    Returns
    -------
    int
        0 on success (including degraded success with archive_warning),
        1 on HTTP errors / 404,
        2 on connection failure.
    """
    url = f"{args.server_url}/migration/rollback"
    try:
        result = http_client.post_json(url)
    except http_client.ServerUnavailable:
        print(
            f"paramem migrate-rollback: the server at {args.server_url} does not implement"
            " /migration/rollback.\n"
            "Check `paramem --version` and server version are aligned. "
            "migrate-accept and migrate-rollback are separate subcommands for non-interactive use.",
            file=sys.stderr,
        )
        return 1
    except http_client.ServerUnreachable:
        print(
            f"paramem migrate-rollback: server unreachable at {args.server_url}.\n"
            "Is paramem-server running? `systemctl --user status paramem-server`.",
            file=sys.stderr,
        )
        return 2
    except http_client.ServerHTTPError as exc:
        if exc.status_code == 409:
            detail = _parse_detail(exc.body)
            if detail.get("error") == "not_trial":
                print("paramem migrate-rollback: no trial active; nothing to rollback.")
                return 0
        print(
            f"paramem migrate-rollback: server returned HTTP {exc.status_code}"
            f" from {exc.url}.\n"
            f"{exc.body.strip() or '(empty response body)'}",
            file=sys.stderr,
        )
        return 1

    archive_warning = result.get("archive_warning")
    if archive_warning is not None:
        # Degraded rollback: config restored but trial adapter rotation failed.
        # Primary action succeeded — exit 0.
        if getattr(args, "json", False):
            print(json.dumps(result, indent=2))
        else:
            print("Rollback completed with a warning.")
            print(f"  Config restored to {result.get('state', '(unknown)')}.")
            print(f"  Trial adapter at {archive_warning.get('path', '(unknown)')}:")
            print(f"    {archive_warning.get('message', '(no details)')}")
            print("  Archive the trial adapter manually if needed.")
            print(
                "\nWarning: trial adapter rotation failed (see above). "
                "Config was restored successfully. Restart the server.",
                file=sys.stderr,
            )
        return 0

    if getattr(args, "json", False):
        print(json.dumps(result, indent=2))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")
    return 0
