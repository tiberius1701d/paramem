"""Handler for ``paramem integrity``.

Queries ``GET /integrity``.  Renders per-file status lines (human-readable)
or raw JSON (``--json``).

Exit codes (per ``cli/main.py`` convention):
    0   ok — all checks passed (ok=True in the report).
    1   integrity failed — server reachable but report.ok is False; also used
        for HTTP errors and 404 (server does not implement the endpoint).
    2   server unreachable — connection refused / timeout.
"""

from __future__ import annotations

import argparse
import json
import sys

from paramem.cli import http_client


def run(args: argparse.Namespace) -> int:
    """Execute the ``integrity`` subcommand.

    GETs ``/integrity`` and renders the response.  With ``--json``, the raw
    JSON report is printed.  Without ``--json``, a per-file table is printed
    with a summary line.

    Parameters
    ----------
    args:
        Parsed namespace from the ``integrity`` subparser.

    Returns
    -------
    int
        0 on ok, 1 on integrity failure / HTTP error / 404, 2 on
        unreachable server.
    """
    url = f"{args.server_url}/integrity"
    try:
        result = http_client.get_json(url)
    except http_client.ServerUnavailable:
        print(
            f"paramem integrity: the server at {args.server_url} returned 404 for\n"
            "/integrity.\n"
            "Check `paramem --version` and server version are aligned.",
            file=sys.stderr,
        )
        return 1
    except http_client.ServerUnreachable:
        print(
            f"paramem integrity: server at {args.server_url} is unreachable.\n"
            "Is the ParaMem server running?",
            file=sys.stderr,
        )
        return 2
    except http_client.ServerHTTPError as exc:
        print(
            f"paramem integrity: server returned HTTP {exc.status_code} from {exc.url}.\n"
            f"{exc.body.strip() or '(empty response body)'}",
            file=sys.stderr,
        )
        return 1

    if getattr(args, "json", False):
        print(json.dumps(result, indent=2))
        return 0 if result.get("ok") else 1

    # Human-readable rendering
    checks = result.get("checks", [])
    failures = result.get("failures", [])
    ok = result.get("ok", False)

    # Print per-file status
    for check in checks:
        status = check.get("status", "?")
        path = check.get("path", "?")
        tier = check.get("tier", "?")
        category = check.get("category", "?")
        detail = check.get("detail", "")
        line = f"  [{status:>14}]  {category}/{tier}  {path}"
        if detail:
            line += f"\n               detail: {detail}"
        print(line)

    print()
    if ok:
        print(f"integrity: OK ({len(checks)} checks, 0 failures)")
    else:
        print(f"integrity: FAILED ({len(failures)} failure(s) out of {len(checks)} checks)")
        for f in failures:
            cat = f.get("category")
            tier = f.get("tier")
            path = f.get("path")
            detail = f.get("detail", "")
            print(f"  FAIL  {cat}/{tier}  {path}: {detail}")

    return 0 if ok else 1
