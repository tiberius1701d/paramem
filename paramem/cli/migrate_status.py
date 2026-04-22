"""Handler for ``paramem migrate-status``.

Queries ``GET /migration/status``.  On ``ServerUnreachable``, falls back to
reading ``state/trial.json`` directly from disk (spec §L228).  In Slice 3b.1
the file does not exist, so the fallback prints "server offline; no trial
marker on disk" and exits 0 (not 2 — the absence of a trial marker is not an
error state, per plan §L228).

TRIAL state (``state/trial.json``) is plaintext and always readable without
decryption — trial markers intentionally stay plaintext across key rotation
(plan §Encryption rule table).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paramem.cli import http_client


def _trial_json_path(server_url: str) -> Path:
    """Return the conventional path to ``state/trial.json`` on the local host.

    The path is relative to the project root (where the server runs).  In a
    deployed setup the server and CLI share the same host (spec §L187), so
    the relative path resolves correctly.

    Parameters
    ----------
    server_url:
        Not used to derive the path (the path is always local), but kept as
        a parameter for future multi-host extensions.

    Returns
    -------
    Path
        Relative path ``state/trial.json`` resolved from the current working
        directory.
    """
    return Path("state") / "trial.json"


def run(args: argparse.Namespace) -> int:
    """Execute the ``migrate-status`` subcommand.

    GETs ``/migration/status`` and renders the response as ``key: value`` lines
    (or raw JSON with ``--json``).  Falls back to ``state/trial.json`` when the
    server is unreachable (spec §L228).

    Parameters
    ----------
    args:
        Parsed namespace from the ``migrate-status`` subparser.

    Returns
    -------
    int
        0 on success or graceful offline fallback, 1 on HTTP error / 404.
    """
    url = f"{args.server_url}/migration/status"
    try:
        result = http_client.get_json(url)
    except http_client.ServerUnavailable:
        print(
            f"paramem migrate-status: the server at {args.server_url} returned 404 for\n"
            "/migration/status.\n"
            "Slice 3b.1 ships /migration/preview, /cancel, /status, /diff.\n"
            "Check `paramem --version` and server version are aligned.",
            file=sys.stderr,
        )
        return 1
    except http_client.ServerUnreachable:
        # Server is offline — fall back to reading state/trial.json directly.
        # In Slice 3b.1 the file does not exist; print a clear message and
        # exit 0 (absence of a trial marker is not an error).  Spec §L228.
        trial_path = _trial_json_path(args.server_url)
        if trial_path.exists():
            try:
                trial_data = json.loads(trial_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                print(
                    f"server offline; trial marker on disk at {trial_path} "
                    f"could not be read: {exc}",
                    file=sys.stderr,
                )
                return 1
            print("server offline; trial marker on disk:")
            if getattr(args, "json", False):
                print(json.dumps(trial_data, indent=2))
            else:
                for key, value in trial_data.items():
                    print(f"{key}: {value}")
        else:
            print("server offline; no trial marker on disk")
        return 0
    except http_client.ServerHTTPError as exc:
        print(
            f"paramem migrate-status: server returned HTTP {exc.status_code} from {exc.url}.\n"
            f"{exc.body.strip() or '(empty response body)'}",
            file=sys.stderr,
        )
        return 1

    if getattr(args, "json", False):
        print(json.dumps(result, indent=2))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")
    return 0
