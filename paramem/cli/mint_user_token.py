"""``paramem mint-user-token`` — mint a per-user bearer token and emit a QR.

Mints an opaque bearer token into the :class:`paramem.server.user_tokens.UserTokenStore`
for the specified speaker and renders a scannable QR code containing the onboarding
payload to stdout.  The QR encodes a JSON object with the server URL and the
plaintext token — exactly the two values a device needs to configure itself.

The plaintext token is printed to stdout once as a text fallback.  It is never
written to any log file.

Security mode behaviour
-----------------------
The token store follows the deployment-wide AUTO encryption mode:

- **Security OFF** (no daily key loaded): store is written in plaintext.
  No passphrase required.
- **Security ON** (daily key loaded): store is age-encrypted.
  ``PARAMEM_DAILY_PASSPHRASE`` must be set in the environment (same
  requirement as all other infra CLIs — ``encrypt-infra``, ``rotate-daily``, etc.).

Exit codes:
    0  success
    1  error (config not found, store write failed, other I/O error)
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Attach ``paramem mint-user-token`` to the top-level dispatcher.

    Parameters
    ----------
    subparsers:
        The ``_SubParsersAction`` returned by ``parser.add_subparsers()``.
    """
    p = subparsers.add_parser(
        "mint-user-token",
        help="Mint a per-user bearer token and emit a QR for device onboarding.",
        description=(
            "Mints an opaque bearer token for SPEAKER_ID, writes it to the "
            "UserTokenStore, and prints a scannable QR code containing the "
            "JSON onboarding payload {server_url, token} to stdout. "
            "The plaintext token is also printed as a text fallback. "
            "When PARAMEM_DAILY_PASSPHRASE is set and a daily key is loaded, "
            "the store is written as an age envelope (Security ON). "
            "Without a daily key the store is written in plaintext (Security OFF)."
        ),
    )
    p.add_argument(
        "speaker_id",
        metavar="SPEAKER_ID",
        nargs="?",
        default=None,
        help=(
            "Speaker this token authenticates (e.g. Speaker0).  "
            "Required unless --unattributed is given."
        ),
    )
    p.add_argument(
        "--label",
        default="",
        metavar="LABEL",
        help='Human-readable device or purpose label stored with the token (default: "").',
    )
    p.add_argument(
        "--server-url",
        default="",
        metavar="URL",
        help=(
            "Base URL included in the onboarding QR payload "
            "(e.g. https://my-host.ts.net). Optional — defaults to empty string."
        ),
    )
    p.add_argument(
        "--config",
        default="configs/server.yaml",
        metavar="PATH",
        help="Server config used to resolve paths.data (default: configs/server.yaml).",
    )
    p.add_argument(
        "--png",
        default=None,
        metavar="PATH",
        help="Also save the QR as a PNG file at this path.",
    )
    p.add_argument(
        "--scope",
        choices=("chat", "admin"),
        default="chat",
        help=(
            "Token capability: 'chat' (conversational endpoints only, the secure default) "
            "or 'admin' (all endpoints, including operational ones like /gpu/*, "
            "/consolidate, /backup/*).  Default: chat."
        ),
    )
    p.add_argument(
        "--unattributed",
        action="store_true",
        help=(
            "Mint a token with no bound speaker_id (identity resolved by voice "
            "embedding at request time).  Designed for shared devices.  "
            "Cannot be combined with a positional SPEAKER_ID.  "
            "Use --scope chat (the default) for a least-privilege shared device."
        ),
    )
    p.add_argument(
        "--force-admin",
        action="store_true",
        help=(
            "Required when combining --scope admin with --unattributed.  "
            "An unattributed admin token cannot be revoked by speaker; "
            "use 'revoke-user-token --label' to revoke it."
        ),
    )


def _resolve_data_dir(args: argparse.Namespace) -> Path | None:
    """Resolve the data directory from the config file.

    Mirrors the pattern in :func:`paramem.cli.rotate_daily._resolve_data_dir`.

    Parameters
    ----------
    args:
        Parsed argument namespace from :func:`add_parser`.

    Returns
    -------
    Path | None
        The resolved data directory, or ``None`` if the config file was not
        found (an error message is printed to stderr before returning).
    """
    from paramem.server.config import load_server_config

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        return None
    cfg = load_server_config(str(config_path))
    return cfg.paths.data


def run(args: argparse.Namespace) -> int:
    """Execute the ``mint-user-token`` subcommand.

    Resolves the data directory from the server config, mints a bearer token
    via :class:`paramem.server.user_tokens.UserTokenStore`, renders a terminal
    QR code of the onboarding JSON payload, and prints the plaintext token as
    a text fallback.  If ``args.png`` is set the QR is also saved as a PNG.

    The plaintext token is printed to stdout only — never to any logger.

    Parameters
    ----------
    args:
        Parsed argument namespace produced by :func:`add_parser`.

    Returns
    -------
    int
        0 on success; 1 on error (config not found, I/O error, mint failure,
        conflicting arguments).
    """
    import segno

    from paramem.server.user_tokens import UserTokenStore

    # --- Argument validation (boundary checks before any I/O) ---
    unattributed = getattr(args, "unattributed", False)
    scope = getattr(args, "scope", "chat")
    force_admin = getattr(args, "force_admin", False)

    if unattributed and args.speaker_id is not None:
        print(
            "ERROR: --unattributed cannot be combined with a positional SPEAKER_ID.",
            file=sys.stderr,
        )
        return 1

    if not unattributed and args.speaker_id is None:
        print(
            "ERROR: SPEAKER_ID is required unless --unattributed is given.",
            file=sys.stderr,
        )
        return 1

    if unattributed and scope == "admin" and not force_admin:
        print(
            "ERROR: --scope admin --unattributed requires --force-admin.\n"
            "WARNING: An unattributed admin token cannot be revoked by speaker; "
            "use 'revoke-user-token --label' to revoke it.",
            file=sys.stderr,
        )
        return 1

    if unattributed and scope == "admin" and force_admin:
        print(
            "WARNING: Minting an unattributed admin token.  This token cannot be "
            "revoked by speaker — use 'revoke-user-token --label' to revoke it.",
            file=sys.stderr,
        )

    # Resolve effective speaker_id: None for unattributed tokens.
    effective_speaker: str | None = None if unattributed else args.speaker_id

    data_dir = _resolve_data_dir(args)
    if data_dir is None:
        return 1

    store_path = Path(data_dir) / "user_tokens.json"

    store = UserTokenStore(store_path)
    token = store.mint(effective_speaker, args.label, scope=scope)

    payload = {"server_url": args.server_url, "token": token}
    json_str = json.dumps(payload)

    qr = segno.make(json_str)

    # Print QR to stdout via a StringIO buffer so capsys captures it in tests.
    buf = io.StringIO()
    qr.terminal(out=buf)
    print(buf.getvalue())

    # Save PNG if requested — boundary I/O error is allowed to propagate with
    # a clear message (not suppressed).
    if args.png:
        png_path = Path(args.png)
        try:
            qr.save(str(png_path), scale=10)
        except OSError as exc:
            print(f"ERROR: could not write PNG to {png_path}: {exc}", file=sys.stderr)
            return 1

    # Text fallback — operator reads this if the QR scan fails.
    speaker_display = "<unattributed>" if effective_speaker is None else effective_speaker
    if not sys.stdout.isatty():
        print(
            "WARNING: plaintext token is being written to a non-terminal "
            "(pipe or redirect); the token may be captured in a file or process substitution.",
            file=sys.stderr,
        )
    print(f"speaker_id : {speaker_display}")
    print(f"server_url : {args.server_url}")
    print(f"scope      : {scope}")
    print(f"token      : {token}")

    return 0
