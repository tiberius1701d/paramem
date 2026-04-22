"""Tests for paramem.cli.migrate (Slice 3b.1 renderer).

Mocks the HTTP client so no live server is required.

Covers:
- preview render (simulate notice → shape-change block → tier diff → unified diff → Proceed?)
- simulate notice y continues, N POSTs cancel + exits 1, EOF POSTs cancel + exits 1
- shape-change block rendered unconditionally
- Proceed? y prints "3b.2 not yet implemented" and exits 0
- Proceed? N POSTs cancel and exits 0
- 404 fallback message lists the four 3b.1 endpoints + names confirm/accept/rollback as 3b.2-pending
- migrate-cancel subcommand POSTs /migration/cancel
- --json mode bypasses prompts; simulate_mode_override at top level (Condition 8)
"""

from __future__ import annotations

import json
from unittest.mock import patch

from paramem.cli import http_client
from paramem.cli.main import main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_PREVIEW = {
    "state": "STAGING",
    "candidate_path": "/abs/server-new.yaml",
    "candidate_hash": "abc123",
    "staged_at": "2026-04-22T00:00:00+00:00",
    "simulate_mode_override": False,
    "unified_diff": (
        "--- server.yaml (live)\n+++ server.yaml (candidate)\n"
        "@@ -1 +1 @@\n-debug: false\n+debug: true"
    ),
    "tier_diff": [
        {
            "dotted_path": "debug",
            "old_value": False,
            "new_value": True,
            "tier": "pipeline_altering",
        }
    ],
    "shape_changes": [],
    "pre_flight_fail": None,
}

_CANCEL_RESPONSE = {"state": "LIVE", "cleared_path": "/abs/server-new.yaml"}


def _patched_post(responses: dict):
    """Return a post_json side-effect that dispatches by URL."""

    def _post(url, body=None, **kwargs):
        for pattern, resp in responses.items():
            if pattern in url:
                if isinstance(resp, Exception):
                    raise resp
                return resp
        raise AssertionError(f"Unexpected POST to {url!r}")

    return _post


# ---------------------------------------------------------------------------
# 404 fallback message
# ---------------------------------------------------------------------------


class TestMigrate404Fallback:
    def test_404_message_mentions_preview_and_version_alignment(self, monkeypatch, capsys):
        """ServerUnavailable → stderr mentions /migration/preview and version alignment.

        The 404 message is evergreen (no slice labels).  It must name the
        endpoint and instruct the operator to check version alignment.
        """
        monkeypatch.setattr(
            http_client,
            "post_json",
            lambda *a, **kw: (_ for _ in ()).throw(http_client.ServerUnavailable("404")),
        )
        rc = main(["migrate", "/abs/path.yaml"])
        captured = capsys.readouterr()
        assert rc == 1
        assert "/migration/preview" in captured.err
        assert "--version" in captured.err or "version" in captured.err.lower()

    def test_404_message_names_accept_and_rollback_subcommands(self, monkeypatch, capsys):
        """ServerUnavailable → stderr names migrate-accept / migrate-rollback.

        The evergreen 404 message refers operators to the standalone
        subcommands for non-interactive accept/rollback.
        """

        def _raise(*a, **kw):
            raise http_client.ServerUnavailable("404")

        monkeypatch.setattr(http_client, "post_json", _raise)
        main(["migrate", "/abs/path.yaml"])
        captured = capsys.readouterr()
        assert "migrate-accept" in captured.err
        assert "migrate-rollback" in captured.err


# ---------------------------------------------------------------------------
# --json mode
# ---------------------------------------------------------------------------


class TestMigrateJsonMode:
    def test_json_mode_emits_raw_json(self, monkeypatch, capsys):
        """--json emits the raw PreviewResponse as JSON."""
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: _BASE_PREVIEW)
        rc = main(["migrate", "/abs/path.yaml", "--json"])
        captured = capsys.readouterr()
        assert rc == 0
        parsed = json.loads(captured.out)
        assert parsed["state"] == "STAGING"

    def test_json_mode_simulate_mode_override_at_top_level(self, monkeypatch, capsys):
        """--json: simulate_mode_override is a top-level field (Condition 8)."""
        preview = {**_BASE_PREVIEW, "simulate_mode_override": True}
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: preview)
        rc = main(["migrate", "/abs/path.yaml", "--json"])
        captured = capsys.readouterr()
        assert rc == 0
        parsed = json.loads(captured.out)
        assert "simulate_mode_override" in parsed, "simulate_mode_override must be a top-level key"
        assert parsed["simulate_mode_override"] is True

    def test_json_mode_bypasses_prompts(self, monkeypatch):
        """--json never calls input() — would raise if it did."""
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: _BASE_PREVIEW)
        # input() is not patched; if called it would fail in non-interactive context
        with patch("builtins.input", side_effect=AssertionError("input() called in --json mode")):
            rc = main(["migrate", "/abs/path.yaml", "--json"])
        assert rc == 0


# ---------------------------------------------------------------------------
# Proceed? prompt — y path
# ---------------------------------------------------------------------------


class TestMigrateProceedy:
    def test_proceed_y_enters_long_poll_flow_and_defers_on_c(self, monkeypatch, capsys):
        """Proceed? y → enters long-poll flow; 'c' (defer) exits 0.

        With Slice 3b.3 shipped, 'y' on the Proceed? prompt enters the
        long-poll flow instead of printing a stub.  This test verifies:
        - rc == 0 (deferred)
        - cancel is NOT posted (operator deferred voluntarily)
        """
        _status_staging = {"state": "STAGING", "gates": None, "comparison_report": None}
        _status_trial_pass = {
            "state": "TRIAL",
            "gates": {"status": "pass", "completed_at": "2026-04-22T01:00:00+00:00"},
            "comparison_report": None,
        }
        get_responses = iter([_status_staging, _status_trial_pass])

        def _get(url, **kwargs):
            return next(get_responses)

        def _post(url, body=None, **kwargs):
            if "preview" in url:
                return _BASE_PREVIEW
            if "confirm" in url:
                return {"state": "TRIAL"}
            raise AssertionError(f"Unexpected POST to {url!r}")

        monkeypatch.setattr(http_client, "get_json", _get)
        monkeypatch.setattr(http_client, "post_json", _post)
        monkeypatch.setattr("time.sleep", lambda _: None)
        # Inputs: Proceed?=y, then 3-way prompt=c (defer)
        inputs = iter(["y", "c"])
        with patch("builtins.input", side_effect=lambda *a: next(inputs)):
            rc = main(["migrate", "/abs/path.yaml"])
        assert rc == 0

    def test_proceed_y_does_not_post_cancel(self, monkeypatch, capsys):
        """Proceed? y must NOT post cancel."""
        cancel_called = []

        def _post(url, body=None, **kwargs):
            if "cancel" in url:
                cancel_called.append(url)
            return _BASE_PREVIEW if "preview" in url else _CANCEL_RESPONSE

        monkeypatch.setattr(http_client, "post_json", _post)
        with patch("builtins.input", return_value="y"):
            main(["migrate", "/abs/path.yaml"])
        assert not cancel_called, "cancel must not be posted on y"


# ---------------------------------------------------------------------------
# Proceed? prompt — N path
# ---------------------------------------------------------------------------


class TestMigrateProceedN:
    def test_proceed_n_posts_cancel(self, monkeypatch, capsys):
        """Proceed? N → POSTs /migration/cancel."""
        cancel_called = []

        def _post(url, body=None, **kwargs):
            if "cancel" in url:
                cancel_called.append(url)
                return _CANCEL_RESPONSE
            return _BASE_PREVIEW

        monkeypatch.setattr(http_client, "post_json", _post)
        with patch("builtins.input", return_value="n"):
            rc = main(["migrate", "/abs/path.yaml"])
        assert rc == 1, f"Expected exit 1 on N, got {rc}"
        assert cancel_called, "cancel was not posted on N"

    def test_proceed_empty_posts_cancel(self, monkeypatch, capsys):
        """Proceed? (empty/default) → POSTs cancel and exits 1."""
        cancel_called = []

        def _post(url, body=None, **kwargs):
            if "cancel" in url:
                cancel_called.append(url)
                return _CANCEL_RESPONSE
            return _BASE_PREVIEW

        monkeypatch.setattr(http_client, "post_json", _post)
        with patch("builtins.input", return_value=""):
            rc = main(["migrate", "/abs/path.yaml"])
        assert rc == 1
        assert cancel_called


# ---------------------------------------------------------------------------
# EOF on Proceed? prompt (Condition 4)
# ---------------------------------------------------------------------------


class TestMigrateEOFCondition4:
    def test_eof_on_proceed_posts_cancel_and_exits_1(self, monkeypatch, capsys):
        """EOF on Proceed? → POSTs /migration/cancel + exits 1 (Condition 4)."""
        cancel_called = []

        def _post(url, body=None, **kwargs):
            if "cancel" in url:
                cancel_called.append(url)
                return _CANCEL_RESPONSE
            return _BASE_PREVIEW

        monkeypatch.setattr(http_client, "post_json", _post)
        with patch("builtins.input", side_effect=EOFError):
            rc = main(["migrate", "/abs/path.yaml"])
        assert rc == 1, f"Expected exit 1 on EOF, got {rc}"
        assert cancel_called, "cancel must be posted on EOF (Condition 4)"


# ---------------------------------------------------------------------------
# Simulate-mode notice
# ---------------------------------------------------------------------------


class TestSimulateNotice:
    def _simulate_preview(self):
        return {**_BASE_PREVIEW, "simulate_mode_override": True}

    def test_simulate_notice_printed(self, monkeypatch, capsys):
        """simulate_mode_override=True → simulate notice printed."""
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: self._simulate_preview())
        with patch("builtins.input", return_value="n"):
            main(["migrate", "/abs/path.yaml"])
        captured = capsys.readouterr()
        assert "simulate-mode" in captured.out.lower() or "NOTICE" in captured.out

    def test_simulate_notice_y_continues_to_proceed_and_defer(self, monkeypatch, capsys):
        """simulate notice y → continues to Proceed? prompt; defer exits 0.

        With Slice 3b.3 shipped, 'y' on the Proceed? prompt enters the
        long-poll flow.  This test uses three inputs:
          1. simulate notice → y
          2. Proceed? → y (enters long-poll flow)
          3. 3-way prompt → c (deferred)
        and verifies rc==0 and cancel not posted.
        """
        cancel_called = []

        _status_staging = {"state": "STAGING", "gates": None, "comparison_report": None}
        _status_trial_pass = {
            "state": "TRIAL",
            "gates": {"status": "pass", "completed_at": "2026-04-22T01:00:00+00:00"},
            "comparison_report": None,
        }
        get_responses = iter([_status_staging, _status_trial_pass])

        def _get(url, **kwargs):
            return next(get_responses)

        def _post(url, body=None, **kwargs):
            if "cancel" in url:
                cancel_called.append(url)
                return _CANCEL_RESPONSE
            if "confirm" in url:
                return {"state": "TRIAL"}
            return self._simulate_preview()

        monkeypatch.setattr(http_client, "get_json", _get)
        monkeypatch.setattr(http_client, "post_json", _post)
        monkeypatch.setattr("time.sleep", lambda _: None)
        # Inputs: simulate notice=y, Proceed?=y, 3-way prompt=c
        inputs = iter(["y", "y", "c"])
        with patch("builtins.input", side_effect=lambda *a: next(inputs)):
            rc = main(["migrate", "/abs/path.yaml"])
        assert rc == 0
        assert not cancel_called

    def test_simulate_notice_n_posts_cancel_and_exits_1(self, monkeypatch, capsys):
        """simulate notice N → POSTs cancel, exits 1."""
        cancel_called = []

        def _post(url, body=None, **kwargs):
            if "cancel" in url:
                cancel_called.append(url)
                return _CANCEL_RESPONSE
            return self._simulate_preview()

        monkeypatch.setattr(http_client, "post_json", _post)
        with patch("builtins.input", return_value="n"):
            rc = main(["migrate", "/abs/path.yaml"])
        assert rc == 1
        assert cancel_called

    def test_simulate_notice_eof_posts_cancel_and_exits_1(self, monkeypatch):
        """EOF on simulate notice → POSTs cancel + exits 1 (Condition 4)."""
        cancel_called = []

        def _post(url, body=None, **kwargs):
            if "cancel" in url:
                cancel_called.append(url)
                return _CANCEL_RESPONSE
            return self._simulate_preview()

        monkeypatch.setattr(http_client, "post_json", _post)
        with patch("builtins.input", side_effect=EOFError):
            rc = main(["migrate", "/abs/path.yaml"])
        assert rc == 1
        assert cancel_called, "cancel must be posted on simulate-notice EOF"


# ---------------------------------------------------------------------------
# Shape-change block
# ---------------------------------------------------------------------------


class TestShapeChangeBlock:
    def test_shape_change_block_rendered_unconditionally(self, monkeypatch, capsys):
        """Shape-change block is printed even without --verbose."""
        preview = {
            **_BASE_PREVIEW,
            "shape_changes": [
                {
                    "adapter": "episodic",
                    "field": "rank",
                    "old_value": 8,
                    "new_value": 16,
                    "consequence": "weights discarded",
                }
            ],
        }
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: preview)
        with patch("builtins.input", return_value="n"):
            main(["migrate", "/abs/path.yaml"])
        captured = capsys.readouterr()
        assert "SHAPE CHANGE" in captured.out
        assert "DESTRUCTIVE" in captured.out
        assert "episodic" in captured.out

    def test_no_shape_change_block_when_empty(self, monkeypatch, capsys):
        """No shape_changes → SHAPE CHANGE block not printed."""
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: _BASE_PREVIEW)
        with patch("builtins.input", return_value="y"):
            main(["migrate", "/abs/path.yaml"])
        captured = capsys.readouterr()
        assert "SHAPE CHANGE" not in captured.out


# ---------------------------------------------------------------------------
# migrate-cancel subcommand
# ---------------------------------------------------------------------------


class TestMigrateCancelSubcommand:
    def test_migrate_cancel_posts_to_cancel_endpoint(self, monkeypatch, capsys):
        """migrate-cancel POSTs to /migration/cancel."""
        cancel_called = []

        def _post(url, body=None, **kwargs):
            cancel_called.append(url)
            return _CANCEL_RESPONSE

        monkeypatch.setattr(http_client, "post_json", _post)
        rc = main(["migrate-cancel"])
        assert rc == 0
        assert any("cancel" in url for url in cancel_called), "cancel endpoint not called"

    def test_migrate_cancel_prints_response(self, monkeypatch, capsys):
        """migrate-cancel prints state and cleared_path."""
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: _CANCEL_RESPONSE)
        rc = main(["migrate-cancel"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "LIVE" in captured.out or "state" in captured.out.lower()

    def test_migrate_cancel_404(self, monkeypatch, capsys):
        """404 from cancel endpoint → rc=1, error message."""

        def _raise(*a, **kw):
            raise http_client.ServerUnavailable("404")

        monkeypatch.setattr(http_client, "post_json", _raise)
        rc = main(["migrate-cancel"])
        assert rc == 1

    def test_migrate_cancel_unreachable(self, monkeypatch, capsys):
        """ServerUnreachable → rc=2."""

        def _raise(*a, **kw):
            raise http_client.ServerUnreachable("refused")

        monkeypatch.setattr(http_client, "post_json", _raise)
        rc = main(["migrate-cancel"])
        assert rc == 2

    def test_migrate_cancel_json_mode(self, monkeypatch, capsys):
        """migrate-cancel --json emits raw JSON."""
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: _CANCEL_RESPONSE)
        rc = main(["migrate-cancel", "--json"])
        captured = capsys.readouterr()
        assert rc == 0
        parsed = json.loads(captured.out)
        assert parsed["state"] == "LIVE"

    def test_migrate_cancel_409_not_staging_exits_0_with_friendly_message(
        self, monkeypatch, capsys
    ):
        """409 not_staging → friendly message to stdout and exit 0 (Fix 5).

        When no candidate is staged the operator's intent is already
        satisfied — cancel is idempotent, so exit 0.
        """

        def _raise(*a, **kw):
            raise http_client.ServerHTTPError(
                status_code=409,
                url="http://localhost:8420/migration/cancel",
                body='{"detail":{"error":"not_staging","message":"No candidate staged."}}',
            )

        monkeypatch.setattr(http_client, "post_json", _raise)
        rc = main(["migrate-cancel"])
        captured = capsys.readouterr()
        assert rc == 0, f"Expected exit 0 for 409 not_staging, got {rc}"
        assert "nothing to cancel" in captured.out.lower(), (
            f"Expected friendly message, got stdout: {captured.out!r}"
        )
        # Must not appear on stderr (this is an informational message, not an error).
        assert "server returned HTTP" not in captured.err

    def test_migrate_cancel_409_other_error_exits_1(self, monkeypatch, capsys):
        """409 with a different error code (not not_staging) → generic message + exit 1."""

        def _raise(*a, **kw):
            raise http_client.ServerHTTPError(
                status_code=409,
                url="http://localhost:8420/migration/cancel",
                body='{"detail":{"error":"trial_active","message":"Trial is active."}}',
            )

        monkeypatch.setattr(http_client, "post_json", _raise)
        rc = main(["migrate-cancel"])
        captured = capsys.readouterr()
        assert rc == 1, f"Expected exit 1 for 409 trial_active, got {rc}"
        assert "server returned HTTP" in captured.err


# ---------------------------------------------------------------------------
# Tier diff rendering
# ---------------------------------------------------------------------------


class TestTierDiffRendering:
    def test_destructive_tier_shown_in_output(self, monkeypatch, capsys):
        """Destructive tier rows appear in output."""
        preview = {
            **_BASE_PREVIEW,
            "tier_diff": [
                {
                    "dotted_path": "model",
                    "old_value": "mistral",
                    "new_value": "gemma",
                    "tier": "destructive",
                }
            ],
        }
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: preview)
        with patch("builtins.input", return_value="y"):
            main(["migrate", "/abs/path.yaml"])
        captured = capsys.readouterr()
        assert "DESTRUCTIVE" in captured.out or "destructive" in captured.out.lower()
        assert "model" in captured.out

    def test_unified_diff_shown_under_header(self, monkeypatch, capsys):
        """Unified diff appears under 'Diff (server.yaml):' header."""
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: _BASE_PREVIEW)
        with patch("builtins.input", return_value="y"):
            main(["migrate", "/abs/path.yaml"])
        captured = capsys.readouterr()
        assert "Diff (server.yaml)" in captured.out
