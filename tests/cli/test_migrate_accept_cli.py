"""Tests for paramem.cli.migrate_accept (Slice 3b.3).

Covers:
- Happy path: POST /migration/accept → 200 → prints response + exits 0.
- --json mode: emits raw JSON.
- 404 (ServerUnavailable) → stderr + exit 1.
- ServerUnreachable → stderr + exit 2.
- 409 not_trial → friendly message to stdout + exit 0 (idempotent).
- 409 gates_failed → stderr + exit 1.
- 409 gates_not_finished → stderr + exit 1.
- 409 other code → generic error to stderr + exit 1.
- 500 server error → stderr + exit 1.
"""

from __future__ import annotations

import json

from paramem.cli import http_client
from paramem.cli.main import main

# ---------------------------------------------------------------------------
# Fixtures / shared data
# ---------------------------------------------------------------------------

_ACCEPT_RESPONSE = {
    "state": "LIVE",
    "trial_adapter_archive_path": "/abs/trial_adapters/20260422-010000",
    "restart_required": True,
    "restart_hint": "systemctl --user restart paramem-server",
    "pre_migration_backup_retained": True,
}


def _raise_unavailable(*a, **kw):
    raise http_client.ServerUnavailable("404")


def _raise_unreachable(*a, **kw):
    raise http_client.ServerUnreachable("connection refused")


def _raise_409(error: str):
    """Return a callable that raises a 409 ServerHTTPError with the given error code."""

    def _raise(*a, **kw):
        raise http_client.ServerHTTPError(
            status_code=409,
            url="http://127.0.0.1:8420/migration/accept",
            body=json.dumps({"detail": {"error": error, "message": f"error={error}"}}),
        )

    return _raise


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestMigrateAcceptHappyPath:
    def test_accept_200_exits_0(self, monkeypatch, capsys):
        """POST /migration/accept → 200 → exits 0."""
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: _ACCEPT_RESPONSE)
        rc = main(["migrate-accept"])
        assert rc == 0

    def test_accept_200_prints_state_and_restart_required(self, monkeypatch, capsys):
        """200 response → state and restart_required appear in stdout."""
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: _ACCEPT_RESPONSE)
        rc = main(["migrate-accept"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "LIVE" in captured.out or "state" in captured.out.lower()
        assert "restart_required" in captured.out or "restart" in captured.out.lower()

    def test_accept_200_json_mode_emits_raw_json(self, monkeypatch, capsys):
        """--json emits raw JSON with all response fields."""
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: _ACCEPT_RESPONSE)
        rc = main(["migrate-accept", "--json"])
        captured = capsys.readouterr()
        assert rc == 0
        parsed = json.loads(captured.out)
        assert parsed["state"] == "LIVE"
        assert parsed["restart_required"] is True


# ---------------------------------------------------------------------------
# 404 / ServerUnavailable
# ---------------------------------------------------------------------------


class TestMigrateAcceptUnavailable:
    def test_404_exits_1(self, monkeypatch, capsys):
        """ServerUnavailable (404) → exit 1."""
        monkeypatch.setattr(http_client, "post_json", _raise_unavailable)
        rc = main(["migrate-accept"])
        assert rc == 1

    def test_404_message_mentions_preview_endpoint(self, monkeypatch, capsys):
        """404 message mentions /migration/accept and version alignment."""
        monkeypatch.setattr(http_client, "post_json", _raise_unavailable)
        main(["migrate-accept"])
        captured = capsys.readouterr()
        assert "/migration/accept" in captured.err
        assert "version" in captured.err.lower() or "--version" in captured.err


# ---------------------------------------------------------------------------
# ServerUnreachable
# ---------------------------------------------------------------------------


class TestMigrateAcceptUnreachable:
    def test_unreachable_exits_2(self, monkeypatch, capsys):
        """ServerUnreachable → exit 2."""
        monkeypatch.setattr(http_client, "post_json", _raise_unreachable)
        rc = main(["migrate-accept"])
        assert rc == 2

    def test_unreachable_prints_troubleshoot_hint(self, monkeypatch, capsys):
        """ServerUnreachable → stderr contains troubleshooting hint."""
        monkeypatch.setattr(http_client, "post_json", _raise_unreachable)
        main(["migrate-accept"])
        captured = capsys.readouterr()
        assert "unreachable" in captured.err.lower() or "paramem-server" in captured.err


# ---------------------------------------------------------------------------
# 409 error variants
# ---------------------------------------------------------------------------


class TestMigrateAccept409:
    def test_409_not_trial_exits_0_with_friendly_message(self, monkeypatch, capsys):
        """409 not_trial → friendly message on stdout + exit 0 (idempotent).

        No trial active means the operator's intent is already satisfied
        (spec §L359 idempotent accept).
        """
        monkeypatch.setattr(http_client, "post_json", _raise_409("not_trial"))
        rc = main(["migrate-accept"])
        captured = capsys.readouterr()
        assert rc == 0, f"Expected exit 0 for not_trial, got {rc}"
        assert "nothing to accept" in captured.out.lower(), (
            f"Expected friendly message, got stdout: {captured.out!r}"
        )
        # Must not appear on stderr.
        assert "server returned HTTP" not in captured.err

    def test_409_gates_failed_exits_1_with_stderr_message(self, monkeypatch, capsys):
        """409 gates_failed → stderr + exit 1.

        Gates failed means only rollback is valid (spec §L354).
        """
        monkeypatch.setattr(http_client, "post_json", _raise_409("gates_failed"))
        rc = main(["migrate-accept"])
        captured = capsys.readouterr()
        assert rc == 1, f"Expected exit 1 for gates_failed, got {rc}"
        assert "gates failed" in captured.err.lower() or "rollback" in captured.err.lower()

    def test_409_gates_not_finished_exits_1(self, monkeypatch, capsys):
        """409 gates_not_finished → stderr + exit 1."""
        monkeypatch.setattr(http_client, "post_json", _raise_409("gates_not_finished"))
        rc = main(["migrate-accept"])
        captured = capsys.readouterr()
        assert rc == 1
        assert "gates" in captured.err.lower() or "running" in captured.err.lower()

    def test_409_other_error_exits_1_with_generic_message(self, monkeypatch, capsys):
        """409 with unrecognized error code → generic stderr + exit 1."""
        monkeypatch.setattr(http_client, "post_json", _raise_409("unknown_future_error"))
        rc = main(["migrate-accept"])
        captured = capsys.readouterr()
        assert rc == 1
        assert "server returned HTTP" in captured.err or "409" in captured.err


# ---------------------------------------------------------------------------
# 5xx server errors
# ---------------------------------------------------------------------------


class TestMigrateAcceptServerError:
    def test_500_exits_1_with_message(self, monkeypatch, capsys):
        """500 server error → stderr + exit 1."""

        def _raise(*a, **kw):
            raise http_client.ServerHTTPError(
                status_code=500,
                url="http://127.0.0.1:8420/migration/accept",
                body="Internal Server Error",
            )

        monkeypatch.setattr(http_client, "post_json", _raise)
        rc = main(["migrate-accept"])
        captured = capsys.readouterr()
        assert rc == 1
        assert "500" in captured.err or "server returned" in captured.err.lower()
