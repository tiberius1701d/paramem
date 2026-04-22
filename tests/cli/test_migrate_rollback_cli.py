"""Tests for paramem.cli.migrate_rollback (Slice 3b.3).

Covers:
- Happy path: POST /migration/rollback → 200 → prints response + exits 0.
- --json mode: emits raw JSON.
- Degraded success: post_json RETURNS a dict with archive_warning (HTTP 207
  passes through post_json as a plain dict because 207 < 400) → prints
  warning + exits 0 (primary action succeeded).
- 404 (ServerUnavailable) → stderr + exit 1.
- ServerUnreachable → stderr + exit 2.
- 409 not_trial → friendly message to stdout + exit 0 (idempotent).
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

_ROLLBACK_RESPONSE = {
    "state": "LIVE",
    "trial_adapter_archive_path": "/abs/trial_adapters/20260422-010000",
    "rollback_pre_mortem_backup_path": "/abs/backups/config/rb-20260422-010000",
    "restart_required": True,
    "restart_hint": "systemctl --user restart paramem-server",
}

_ROLLBACK_207_RESPONSE = {
    "state": "LIVE",
    "trial_adapter_archive_path": None,
    "rollback_pre_mortem_backup_path": "/abs/backups/config/rb-20260422-010000",
    "restart_required": True,
    "restart_hint": "systemctl --user restart paramem-server",
    "archive_warning": {
        "path": "/abs/trial_adapters/20260422-010000",
        "message": "shutil.move failed: permission denied",
    },
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
            url="http://127.0.0.1:8420/migration/rollback",
            body=json.dumps({"detail": {"error": error, "message": f"error={error}"}}),
        )

    return _raise


def _return_207(*a, **kw):
    """Return the 207 body as a plain dict — mirrors what post_json does for HTTP 207.

    HTTP 207 < 400, so post_json does NOT raise ServerHTTPError.  The CLI must
    detect degraded rollback via body-content (``archive_warning`` key), not by
    catching a ServerHTTPError with status_code==207.
    """
    return _ROLLBACK_207_RESPONSE


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestMigrateRollbackHappyPath:
    def test_rollback_200_exits_0(self, monkeypatch, capsys):
        """POST /migration/rollback → 200 → exits 0."""
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: _ROLLBACK_RESPONSE)
        rc = main(["migrate-rollback"])
        assert rc == 0

    def test_rollback_200_prints_state_and_restart_required(self, monkeypatch, capsys):
        """200 response → state and restart_required appear in stdout."""
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: _ROLLBACK_RESPONSE)
        rc = main(["migrate-rollback"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "LIVE" in captured.out or "state" in captured.out.lower()
        assert "restart_required" in captured.out or "restart" in captured.out.lower()

    def test_rollback_200_json_mode_emits_raw_json(self, monkeypatch, capsys):
        """--json emits raw JSON with all response fields."""
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: _ROLLBACK_RESPONSE)
        rc = main(["migrate-rollback", "--json"])
        captured = capsys.readouterr()
        assert rc == 0
        parsed = json.loads(captured.out)
        assert parsed["state"] == "LIVE"
        assert parsed["restart_required"] is True


# ---------------------------------------------------------------------------
# Degraded success — config restored, adapter rotation failed (archive_warning)
# ---------------------------------------------------------------------------


class TestMigrateRollback207:
    def test_207_exits_0_primary_action_succeeded(self, monkeypatch, capsys):
        """post_json returns dict with archive_warning → exit 0 (primary action succeeded).

        HTTP 207 passes through post_json as a plain dict (207 < 400 so no
        ServerHTTPError is raised).  The CLI detects degraded rollback via the
        ``archive_warning`` key in the response body.
        """
        monkeypatch.setattr(http_client, "post_json", _return_207)
        rc = main(["migrate-rollback"])
        assert rc == 0, f"Expected exit 0 for degraded rollback (archive_warning), got {rc}"

    def test_207_prints_archive_warning(self, monkeypatch, capsys):
        """archive_warning in response body → warning message appears in output."""
        monkeypatch.setattr(http_client, "post_json", _return_207)
        rc = main(["migrate-rollback"])
        captured = capsys.readouterr()
        assert rc == 0
        # The archive_warning path or message must appear somewhere in stdout or stderr.
        combined = captured.out + captured.err
        has_warning = (
            "permission denied" in combined
            or "archive_warning" in combined
            or "trial adapter" in combined
        )
        assert has_warning, (
            f"Expected archive_warning content in output, "
            f"got stdout: {captured.out!r}, stderr: {captured.err!r}"
        )

    def test_207_json_mode_emits_raw_json_with_archive_warning(self, monkeypatch, capsys):
        """archive_warning in response + --json → emits raw JSON body including archive_warning."""
        monkeypatch.setattr(http_client, "post_json", _return_207)
        rc = main(["migrate-rollback", "--json"])
        captured = capsys.readouterr()
        assert rc == 0
        parsed = json.loads(captured.out)
        assert "archive_warning" in parsed, f"archive_warning missing from JSON: {parsed!r}"
        assert parsed["state"] == "LIVE"


# ---------------------------------------------------------------------------
# 404 / ServerUnavailable
# ---------------------------------------------------------------------------


class TestMigrateRollbackUnavailable:
    def test_404_exits_1(self, monkeypatch, capsys):
        """ServerUnavailable (404) → exit 1."""
        monkeypatch.setattr(http_client, "post_json", _raise_unavailable)
        rc = main(["migrate-rollback"])
        assert rc == 1

    def test_404_message_mentions_rollback_endpoint(self, monkeypatch, capsys):
        """404 message mentions /migration/rollback and version alignment."""
        monkeypatch.setattr(http_client, "post_json", _raise_unavailable)
        main(["migrate-rollback"])
        captured = capsys.readouterr()
        assert "/migration/rollback" in captured.err
        assert "version" in captured.err.lower() or "--version" in captured.err


# ---------------------------------------------------------------------------
# ServerUnreachable
# ---------------------------------------------------------------------------


class TestMigrateRollbackUnreachable:
    def test_unreachable_exits_2(self, monkeypatch, capsys):
        """ServerUnreachable → exit 2."""
        monkeypatch.setattr(http_client, "post_json", _raise_unreachable)
        rc = main(["migrate-rollback"])
        assert rc == 2

    def test_unreachable_prints_troubleshoot_hint(self, monkeypatch, capsys):
        """ServerUnreachable → stderr contains troubleshooting hint."""
        monkeypatch.setattr(http_client, "post_json", _raise_unreachable)
        main(["migrate-rollback"])
        captured = capsys.readouterr()
        assert "unreachable" in captured.err.lower() or "paramem-server" in captured.err


# ---------------------------------------------------------------------------
# 409 error variants
# ---------------------------------------------------------------------------


class TestMigrateRollback409:
    def test_409_not_trial_exits_0_with_friendly_message(self, monkeypatch, capsys):
        """409 not_trial → friendly message on stdout + exit 0 (idempotent).

        No trial active means the operator's intent is already satisfied.
        """
        monkeypatch.setattr(http_client, "post_json", _raise_409("not_trial"))
        rc = main(["migrate-rollback"])
        captured = capsys.readouterr()
        assert rc == 0, f"Expected exit 0 for not_trial, got {rc}"
        assert "nothing to rollback" in captured.out.lower(), (
            f"Expected friendly message, got stdout: {captured.out!r}"
        )
        # Must not appear on stderr.
        assert "server returned HTTP" not in captured.err

    def test_409_other_error_exits_1_with_generic_message(self, monkeypatch, capsys):
        """409 with unrecognized error code → generic stderr + exit 1."""
        monkeypatch.setattr(http_client, "post_json", _raise_409("unknown_future_error"))
        rc = main(["migrate-rollback"])
        captured = capsys.readouterr()
        assert rc == 1
        assert "server returned HTTP" in captured.err or "409" in captured.err


# ---------------------------------------------------------------------------
# 5xx server errors
# ---------------------------------------------------------------------------


class TestMigrateRollbackServerError:
    def test_500_exits_1_with_message(self, monkeypatch, capsys):
        """500 server error → stderr + exit 1."""

        def _raise(*a, **kw):
            raise http_client.ServerHTTPError(
                status_code=500,
                url="http://127.0.0.1:8420/migration/rollback",
                body="Internal Server Error",
            )

        monkeypatch.setattr(http_client, "post_json", _raise)
        rc = main(["migrate-rollback"])
        captured = capsys.readouterr()
        assert rc == 1
        assert "500" in captured.err or "server returned" in captured.err.lower()
