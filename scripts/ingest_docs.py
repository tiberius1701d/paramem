#!/usr/bin/env python3
"""Operator CLI for ingesting documents into the ParaMem server.

Chunks a single .txt, .md, or .pdf file into document segments, posts them
to the running server via ``POST /ingest-sessions``, and optionally triggers
consolidation or cancels the queued sessions.

Usage examples::

    # Non-interactive — speaker id provided directly:
    python scripts/ingest_docs.py resume.pdf --speaker <id> --non-interactive

    # Non-interactive, skip the [N/S/C] prompt:
    python scripts/ingest_docs.py notes.md --speaker <id> --non-interactive --no-action

    # Interactive — picks speaker from server list (requires a running server):
    python scripts/ingest_docs.py resume.pdf --server http://localhost:8420

Limitations (v1):
    - Single file only; multiple files are not supported.
    - PDF must have a text layer (no OCR).
    - Interactive speaker picker requires a running server at ``--server`` URL.

Exit codes:
    0  success
    1  HTTP error or unexpected server response
    2  file problem (unsupported format, empty file, scanned PDF)
    3  argparse error (handled by argparse itself).  Note: the server's
       FatalConfigError convention also uses exit 3 (never-restart, see
       configs/server.yaml ``process.restart``) — that is a separate
       process; this CLI's exit 3 is argparse-only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so the CLI can be run as a script
# without installing the package.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from paramem.cli.http_client import (  # noqa: E402
    ServerHTTPError,
    ServerUnavailable,
    ServerUnreachable,
    get_json,
    post_json,
)
from paramem.graph.document_chunker import (  # noqa: E402
    DocumentChunk,
    EmptyDocumentError,
    ScannedPdfRejectedError,
    UnsupportedFormatError,
    chunk_document,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build and parse the CLI argument namespace.

    Args:
        argv: Argument list.  ``None`` means ``sys.argv[1:]``.

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        prog="ingest_docs",
        description=(
            "Chunk a document and post it to the ParaMem server for ingestion. "
            "Supports .txt, .md, and .pdf files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/ingest_docs.py resume.pdf "
            "--speaker abc123 --non-interactive\n"
            "  python scripts/ingest_docs.py notes.md "
            "--speaker abc123 --non-interactive --no-action\n\n"
            "Limitation (v1): only a single file per invocation is supported."
        ),
    )
    parser.add_argument(
        "path",
        metavar="FILE",
        type=Path,
        help="Path to the document to ingest (.txt, .md, .markdown, or .pdf).",
    )
    parser.add_argument(
        "--server",
        default="http://localhost:8420",
        metavar="URL",
        help="Base URL of the running ParaMem server (default: http://localhost:8420).",
    )
    parser.add_argument(
        "--speaker",
        default=None,
        metavar="SPEAKER_ID",
        help=(
            "Known speaker ID from the server.  Required when --non-interactive is set. "
            "Validated server-side."
        ),
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        dest="non_interactive",
        help=(
            "Suppress all interactive prompts.  --speaker must be provided. "
            "Without --no-action the [N/S/C] prompt is also shown."
        ),
    )
    parser.add_argument(
        "--no-action",
        action="store_true",
        dest="no_action",
        help=(
            "Skip the post-ingest [N]ow / [S]chedule / [C]ancel prompt.  "
            "Sessions are queued; consolidation is left to the schedule."
        ),
    )
    return parser.parse_args(argv)


def pick_speaker_interactively(profiles: list[dict]) -> str:
    """Prompt the operator to select a speaker from the enrolled list.

    Displays a numbered menu of enrolled profiles and reads an integer
    selection from stdin.  Loops until a valid choice is entered.

    Args:
        profiles: List of profile dicts from ``GET /status``
            ``speakers`` key.  Each entry must have ``"id"`` and ``"name"``.

    Returns:
        The selected speaker's ``"id"`` string.

    Raises:
        SystemExit: If the profile list is empty (no speakers enrolled).
    """
    if not profiles:
        print(
            "ERROR: no speakers enrolled on this server. Enroll a speaker first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\nEnrolled speakers:")
    for idx, prof in enumerate(profiles, start=1):
        name = prof.get("name", "?")
        sid = prof.get("id", "?")
        embeddings = prof.get("embeddings", 0)
        print(f"  [{idx}] {name}  (id={sid}, embeddings={embeddings})")

    while True:
        raw = input("\nSelect speaker number: ").strip()
        if raw.isdigit():
            choice = int(raw)
            if 1 <= choice <= len(profiles):
                return profiles[choice - 1]["id"]
        print(f"  Invalid choice — enter a number between 1 and {len(profiles)}.")


def chunk_file(path: Path) -> list[DocumentChunk]:
    """Dispatch ``path`` through :func:`chunk_document` and return chunks.

    Prints friendly error messages and exits on format errors.

    Args:
        path: Absolute path to the document file.

    Returns:
        List of :class:`DocumentChunk` instances.

    Raises:
        SystemExit(2): On format, encoding, or empty-document errors.
    """
    try:
        return chunk_document(path)
    except UnsupportedFormatError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    except EmptyDocumentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    except ScannedPdfRejectedError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(2)


def post_chunks(server_url: str, speaker_id: str, chunks: list[DocumentChunk]) -> dict:
    """POST ``/ingest-sessions`` with the pre-chunked document payload.

    Args:
        server_url: Base URL of the running ParaMem server.
        speaker_id: Known speaker identifier from the server's profile store.
        chunks: List of :class:`DocumentChunk` instances to enqueue.

    Returns:
        Parsed ``IngestSessionsResponse``-shaped dict from the server.

    Raises:
        SystemExit(1): On HTTP errors or server unreachability.
    """
    payload = {
        "speaker_id": speaker_id,
        "sessions": [
            {
                "source": c.source_path,
                "chunk": c.text,
                "chunk_index": c.chunk_index,
                "source_type": c.source_type,
                "doc_title": c.doc_title,
            }
            for c in chunks
        ],
    }
    url = f"{server_url.rstrip('/')}/ingest-sessions"
    try:
        return post_json(url, payload, timeout=30.0)
    except ServerUnreachable as exc:
        _exit_unreachable(server_url, exc)
    except ServerUnavailable:
        print(
            "ERROR: POST /ingest-sessions not available on this server version.",
            file=sys.stderr,
        )
        sys.exit(1)
    except ServerHTTPError as exc:
        _handle_ingest_http_error(exc)


_UNREACHABLE_HINT = (
    "Run `pstatus` to diagnose — if startup was refused by the encryption "
    "gate, pstatus prints the Reason / Cause / Remedy block. "
    "See SECURITY.md §4 for recovery."
)


def _exit_unreachable(server_url: str, exc: Exception) -> None:
    """Print the standard 'server not reachable' error and exit 1.

    Centralises the message so every caller surfaces the same pstatus /
    SECURITY.md hint when the server is down (commonly: gate-refused
    startup under ``require_encryption: true``).

    Args:
        server_url: Base URL the CLI tried to reach.
        exc: The underlying exception (typically ``ServerUnreachable``).

    Raises:
        SystemExit(1): Always.
    """
    print(
        f"ERROR: server not reachable at {server_url} — {exc}\n{_UNREACHABLE_HINT}",
        file=sys.stderr,
    )
    sys.exit(1)


def _handle_ingest_http_error(exc: ServerHTTPError) -> None:
    """Print a human-readable message for known ingest error codes and exit.

    Args:
        exc: The :class:`ServerHTTPError` raised by :func:`post_json`.

    Raises:
        SystemExit(1): Always.
    """
    if exc.status_code == 404:
        print(
            "ERROR: speaker not found on this server (HTTP 404). "
            "Check the speaker ID with --non-interactive --speaker.",
            file=sys.stderr,
        )
    elif exc.status_code == 400:
        print(
            "ERROR: bad request (HTTP 400) — speaker_id may be empty.",
            file=sys.stderr,
        )
    elif exc.status_code == 409:
        # Echo the server body verbatim — it carries the actionable
        # remedy (POST /migration/accept or /migration/rollback) and the
        # CLI should not paper over it with a less-informative message.
        print(
            f"ERROR: ingest blocked by migration trial (HTTP 409): {exc.body}",
            file=sys.stderr,
        )
    else:
        print(
            f"ERROR: server returned HTTP {exc.status_code}: {exc.body}",
            file=sys.stderr,
        )
    sys.exit(1)


def request_consolidate(server_url: str) -> dict:
    """POST ``/consolidate`` to start a full consolidation cycle.

    Args:
        server_url: Base URL of the running ParaMem server.

    Returns:
        Server response dict.

    Raises:
        SystemExit(1): On HTTP errors or server unreachability.
    """
    url = f"{server_url.rstrip('/')}/consolidate"
    try:
        return post_json(url, timeout=10.0)
    except ServerUnreachable as exc:
        _exit_unreachable(server_url, exc)
    except (ServerUnavailable, ServerHTTPError) as exc:
        print(f"ERROR: consolidate request failed: {exc}", file=sys.stderr)
        sys.exit(1)


def request_cancel(server_url: str, session_ids: list[str]) -> dict:
    """POST ``/ingest-sessions/cancel`` to discard queued sessions.

    Args:
        server_url: Base URL of the running ParaMem server.
        session_ids: List of session IDs to cancel.

    Returns:
        Server response dict.

    Raises:
        SystemExit(1): On HTTP errors or server unreachability.
    """
    url = f"{server_url.rstrip('/')}/ingest-sessions/cancel"
    try:
        return post_json(url, {"session_ids": session_ids}, timeout=10.0)
    except ServerUnreachable as exc:
        _exit_unreachable(server_url, exc)
    except (ServerUnavailable, ServerHTTPError) as exc:
        print(f"ERROR: cancel request failed: {exc}", file=sys.stderr)
        sys.exit(1)


def _fetch_speaker_profiles(server_url: str) -> list[dict]:
    """Fetch enrolled speaker profiles from ``GET /status``.

    Extracts the ``speakers`` list from the status response.

    Args:
        server_url: Base URL of the running ParaMem server.

    Returns:
        List of speaker profile dicts.

    Raises:
        SystemExit(1): If the server is unreachable or returns an error.
    """
    url = f"{server_url.rstrip('/')}/status"
    try:
        data = get_json(url, timeout=5.0)
    except ServerUnreachable as exc:
        # Same hint as the other reachability sites; also remind the
        # operator that --non-interactive avoids needing /status at all.
        print(
            f"ERROR: server not reachable at {server_url} — {exc}\n"
            f"{_UNREACHABLE_HINT}\n"
            "Tip: if you already know the speaker id, use --non-interactive --speaker <id>.",
            file=sys.stderr,
        )
        sys.exit(1)
    except (ServerUnavailable, ServerHTTPError) as exc:
        print(f"ERROR: could not fetch speakers from server: {exc}", file=sys.stderr)
        sys.exit(1)
    return data.get("speakers", [])


def _prompt_action(queued: list[str]) -> str:
    """Prompt the operator for the post-ingest action and return the choice.

    Args:
        queued: List of session IDs that were successfully queued.

    Returns:
        One of ``"N"``, ``"S"``, or ``"C"`` (upper-case).
    """
    session_count = len(queued)
    print(
        f"\nQueued {session_count} session(s).  What would you like to do next?\n"
        "  [N] Start consolidation now\n"
        "  [S] Schedule — let the server consolidate on its normal schedule\n"
        "  [C] Cancel — remove the queued sessions from the buffer"
    )
    while True:
        raw = input("\nChoice [N/S/C]: ").strip().upper()
        if raw in ("N", "S", "C"):
            return raw
        print("  Please enter N, S, or C.")


def main(argv: list[str] | None = None) -> int:
    """Entry point for the document ingest CLI.

    Args:
        argv: Argument list.  ``None`` means ``sys.argv[1:]``.

    Returns:
        Exit code (0 success, 1 HTTP/server error, 2 file error, 3 arg error).
    """
    args = parse_args(argv)

    # --- 1. Resolve and validate the file path. ---
    path = args.path.resolve()
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 2

    # --- 2. Chunk the document. ---
    chunks = chunk_file(path)
    if not chunks:
        print("ERROR: document produced no chunks (empty or unsupported content).", file=sys.stderr)
        return 2

    print(f"Chunked '{path.name}' into {len(chunks)} segment(s).")

    # --- 3. Resolve the speaker. ---
    if args.non_interactive:
        if not args.speaker:
            print(
                "ERROR: --non-interactive requires --speaker <id>.",
                file=sys.stderr,
            )
            return 2
        speaker_id = args.speaker
    else:
        # Interactive: fetch profile list from the server.
        profiles = _fetch_speaker_profiles(args.server)
        if args.speaker:
            # --speaker given but not --non-interactive: use it directly without picking.
            speaker_id = args.speaker
        else:
            speaker_id = pick_speaker_interactively(profiles)

    # --- 4. POST the chunks to the server. ---
    response = post_chunks(args.server, speaker_id, chunks)

    queued: list[str] = response.get("queued", [])
    total_chunks: int = response.get("total_chunks", len(chunks))
    registry_skipped: int = response.get("registry_skipped", 0)

    print(
        f"Server response: total_chunks={total_chunks}, "
        f"queued={len(queued)}, "
        f"registry_skipped={registry_skipped}"
    )
    if queued:
        # Print every id, not just the first — operators running with
        # --no-action need the full list to drive a later cancel.
        print(f"Session ids: {', '.join(queued)}")

    if not queued:
        if registry_skipped == total_chunks:
            print("All chunks already ingested (registry idempotency hit) — nothing to do.")
        else:
            print("WARNING: no sessions were queued by the server.")
        return 0

    # --- 5. Post-ingest action. ---
    if args.no_action:
        print("Queued sessions left for scheduled consolidation (--no-action).")
        return 0

    if args.non_interactive:
        print("Sessions queued.  Run POST /consolidate or wait for the scheduled cycle.")
        return 0

    action = _prompt_action(queued)

    if action == "N":
        result = request_consolidate(args.server)
        print(f"Consolidation started: {result}")
    elif action == "S":
        print("Sessions queued; will run on next scheduled consolidation.")
    elif action == "C":
        result = request_cancel(args.server, queued)
        cancelled = result.get("cancelled", [])
        not_found = result.get("not_found", [])
        print(f"Cancelled {len(cancelled)} session(s).")
        if not_found:
            print(f"Not found (already processed?): {not_found}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
