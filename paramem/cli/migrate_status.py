"""Handler for ``paramem migrate-status``.

Queries ``GET /migration/status``.  On ``ServerUnreachable``, falls back to
reading ``state/trial.json`` directly from disk.  When the file does not
exist, the fallback prints "server offline; no trial marker on disk" and
exits 0 (absence of a trial marker is not an error state).

TRIAL state (``state/trial.json``) is plaintext and always readable without
decryption — trial markers intentionally stay plaintext across key rotation
so the fallback path works with no loaded key.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paramem.cli import http_client


def _resolve_data_dir(config: str) -> Path | None:
    """Resolve the data directory from the server config.

    Mirrors the pattern in :func:`paramem.cli.mint_user_token._resolve_data_dir`.

    Parameters
    ----------
    config:
        Path to the server config file (typically ``args.config``).

    Returns
    -------
    Path | None
        The resolved data directory, or ``None`` if the config file was not
        found (an error message is printed to stderr before returning).
    """
    from paramem.server.config import load_server_config

    config_path = Path(config).expanduser().resolve()
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        return None
    cfg = load_server_config(str(config_path))
    return cfg.paths.data


def _trial_json_path(server_url: str, config: str = "configs/server.yaml") -> Path | None:
    """Return the path to ``state/trial.json`` under the configured data dir.

    The marker lives at ``<paths.data>/state/trial.json``.  The data
    directory is resolved from the server config (mirroring
    :func:`paramem.cli.mint_user_token._resolve_data_dir`) rather than a
    hardcoded ``data/ha`` path, so the CLI honors a non-default
    ``paths.data`` the same way the server does.

    Parameters
    ----------
    server_url:
        Not used to derive the path (the path is always local), but kept as
        a parameter for backward compatibility / future multi-host
        extensions.
    config:
        Path to the server config file used to resolve ``paths.data``.
        Defaults to ``configs/server.yaml``.

    Returns
    -------
    Path | None
        ``<paths.data>/state/trial.json``, or ``None`` when the config file
        cannot be resolved (``_resolve_data_dir`` already printed the error
        to stderr in that case) — there is no safe cwd-relative guess to fall
        back to.
    """
    data_dir = _resolve_data_dir(config)
    if data_dir is None:
        return None
    return Path(data_dir) / "state" / "trial.json"


def run(args: argparse.Namespace) -> int:
    """Execute the ``migrate-status`` subcommand.

    GETs ``/migration/status`` and renders the response as ``key: value`` lines
    (or raw JSON with ``--json``).  Falls back to reading ``state/trial.json``
    directly from disk when the server is unreachable.

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
            "Available migration endpoints: /migration/preview, /migration/cancel, "
            "/migration/confirm, /migration/status, /migration/diff, /migration/accept, "
            "/migration/rollback.\n"
            "Check `paramem --version` and server version are aligned.",
            file=sys.stderr,
        )
        return 1
    except http_client.ServerUnreachable:
        # Server is offline — fall back to reading state/trial.json directly.
        # The file does not exist during initial deploy; print a clear message
        # and exit 0 (absence of a trial marker is not an error).
        trial_path = _trial_json_path(
            args.server_url, getattr(args, "config", "configs/server.yaml")
        )
        if trial_path is None:
            print(
                "server offline and trial-state path could not be resolved "
                "(no config); nothing to show.",
                file=sys.stderr,
            )
            return 0
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
