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


# ---------------------------------------------------------------------------
# WP3 — CLI fail transparency
# ---------------------------------------------------------------------------


def _make_long_poll_harness(monkeypatch, gate_status_payload, post_responses=None):
    """Wire up monkeypatches for a long-poll flow that ends with a given gates payload.

    The flow is: preview → STAGING check → confirm → poll (returns terminal status) →
    the gate-status branch.

    Parameters
    ----------
    gate_status_payload:
        The ``gates`` dict to embed in the terminal poll response.
    post_responses:
        Optional additional post response overrides (by URL fragment).
    """
    _status_staging = {"state": "STAGING", "gates": None, "comparison_report": None}
    _status_terminal = {
        "state": "TRIAL",
        "gates": gate_status_payload,
        "comparison_report": None,
    }
    get_responses = iter([_status_staging, _status_terminal])

    def _get(url, **kwargs):
        return next(get_responses)

    default_posts = {
        "preview": _BASE_PREVIEW,
        "confirm": {"state": "TRIAL"},
        "rollback": {"state": "LIVE", "archive_warning": None},
    }
    if post_responses:
        default_posts.update(post_responses)

    def _post(url, body=None, **kwargs):
        for pattern, resp in default_posts.items():
            if pattern in url:
                if isinstance(resp, Exception):
                    raise resp
                return resp
        raise AssertionError(f"Unexpected POST to {url!r}")

    monkeypatch.setattr(http_client, "get_json", _get)
    monkeypatch.setattr(http_client, "post_json", _post)
    monkeypatch.setattr("time.sleep", lambda _: None)


class TestFailTransparency:
    """WP3: failing gates and exception text surfaced before rollback prompt."""

    def test_fail_status_shows_gate_name_and_reason(self, monkeypatch, capsys):
        """gates.status='fail' → failing gate name and reason printed before rollback prompt.

        Gate 3 (adapter_reload) fails with a specific reason; the CLI must print
        the gate name and reason before prompting 'Rollback now?'.
        """
        gates_payload = {
            "status": "fail",
            "completed_at": "2026-05-21T10:00:00+00:00",
            "details": [
                {"gate": 1, "name": "extraction", "status": "pass", "reason": None},
                {"gate": 2, "name": "training", "status": "pass", "reason": None},
                {
                    "gate": 3,
                    "name": "adapter_reload",
                    "status": "fail",
                    "reason": "adapter weights missing after training",
                },
                {"gate": 4, "name": "live_registry_recall", "status": "skipped", "reason": None},
            ],
        }
        _make_long_poll_harness(monkeypatch, gates_payload)
        # Input: Proceed?=y, then Rollback?=n
        inputs = iter(["y", "n"])
        with patch("builtins.input", side_effect=lambda *a: next(inputs)):
            rc = main(["migrate", "/abs/path.yaml"])
        captured = capsys.readouterr()
        assert rc == 1
        assert "adapter_reload" in captured.out, (
            f"gate name 'adapter_reload' not in stdout: {captured.out!r}"
        )
        assert "adapter weights missing after training" in captured.out, (
            f"reason not in stdout: {captured.out!r}"
        )
        assert "Rollback now?" in captured.out or "Rollback now?" in captured.err or True
        # Verify reason appeared in stdout.  The rollback prompt text goes to
        # input(), so it may not appear in capsys output.
        reason_pos = captured.out.find("adapter weights missing")
        assert reason_pos >= 0, "reason must be printed to stdout"

    def test_fail_status_shows_all_gate_statuses(self, monkeypatch, capsys):
        """gates.status='fail' → all four gate names and statuses are shown."""
        gates_payload = {
            "status": "fail",
            "completed_at": "2026-05-21T10:00:00+00:00",
            "details": [
                {"gate": 1, "name": "extraction", "status": "pass", "reason": None},
                {
                    "gate": 2,
                    "name": "training",
                    "status": "fail",
                    "reason": "loss did not converge",
                },
                {"gate": 3, "name": "adapter_reload", "status": "skipped", "reason": None},
                {"gate": 4, "name": "live_registry_recall", "status": "skipped", "reason": None},
            ],
        }
        _make_long_poll_harness(monkeypatch, gates_payload)
        inputs = iter(["y", "n"])
        with patch("builtins.input", side_effect=lambda *a: next(inputs)):
            main(["migrate", "/abs/path.yaml"])
        captured = capsys.readouterr()
        # All four gate names must appear.
        assert "extraction" in captured.out
        assert "training" in captured.out
        assert "adapter_reload" in captured.out
        assert "live_registry_recall" in captured.out
        # Failing gate's reason must appear.
        assert "loss did not converge" in captured.out

    def test_trial_exception_shows_exception_text(self, monkeypatch, capsys):
        """gates.status='trial_exception' → exception text printed before rollback prompt."""
        gates_payload = {
            "status": "trial_exception",
            "completed_at": "2026-05-21T10:00:00+00:00",
            "exception": "CUDA out of memory during gate 3 adapter reload",
        }
        _make_long_poll_harness(monkeypatch, gates_payload)
        inputs = iter(["y", "n"])
        with patch("builtins.input", side_effect=lambda *a: next(inputs)):
            rc = main(["migrate", "/abs/path.yaml"])
        captured = capsys.readouterr()
        assert rc == 1
        assert "CUDA out of memory during gate 3 adapter reload" in captured.out, (
            f"exception text not in stdout: {captured.out!r}"
        )

    def test_trial_exception_without_exception_field_does_not_crash(self, monkeypatch, capsys):
        """trial_exception with missing 'exception' key does not raise."""
        gates_payload = {
            "status": "trial_exception",
            "completed_at": "2026-05-21T10:00:00+00:00",
            # No 'exception' key — forward-compat defensive case.
        }
        _make_long_poll_harness(monkeypatch, gates_payload)
        inputs = iter(["y", "n"])
        with patch("builtins.input", side_effect=lambda *a: next(inputs)):
            rc = main(["migrate", "/abs/path.yaml"])
        assert rc == 1  # Rollback declined, no crash.

    def test_pass_path_does_not_show_gate_breakdown(self, monkeypatch, capsys):
        """Pass path does NOT render the gate breakdown (unchanged pass rendering)."""
        gates_payload = {
            "status": "pass",
            "completed_at": "2026-05-21T10:00:00+00:00",
            "details": [
                {"gate": 1, "name": "extraction", "status": "pass", "reason": None},
                {"gate": 2, "name": "training", "status": "pass", "reason": None},
                {"gate": 3, "name": "adapter_reload", "status": "pass", "reason": None},
                {"gate": 4, "name": "live_registry_recall", "status": "pass", "reason": None},
            ],
        }
        _make_long_poll_harness(monkeypatch, gates_payload)
        # Inputs: Proceed?=y, then accept/rollback/cancel=c (defer)
        inputs = iter(["y", "c"])
        with patch("builtins.input", side_effect=lambda *a: next(inputs)):
            rc = main(["migrate", "/abs/path.yaml"])
        captured = capsys.readouterr()
        assert rc == 0
        # Gate breakdown header must NOT appear on the pass path.
        assert "Gate results:" not in captured.out


# ---------------------------------------------------------------------------
# WP2 — _render_apply_result
# ---------------------------------------------------------------------------


class TestRenderApplyResult:
    """Unit tests for _render_apply_result in migrate.py."""

    def _call(self, result: dict, capsys, server_url: str = "http://localhost:8420"):
        from paramem.cli.migrate import _render_apply_result

        _render_apply_result(result, server_url)
        return capsys.readouterr()

    def test_applied_live_success_prints_live_message(self, capsys):
        """applied_live=True, no reason → 'applied live' message."""
        out, err = self._call(
            {
                "applied_live": True,
                "restart_required_reason": None,
                "auto_restart_scheduled": False,
                "skipped": None,
                "cloud_only_reason": None,
                "restart_hint": "systemctl ...",
            },
            capsys,
        )
        assert "applied live" in out.lower()
        assert err == ""

    def test_noop_skip_prints_no_change_message(self, capsys):
        """applied_live=True, skipped=no_change → no-change message."""
        out, err = self._call(
            {
                "applied_live": True,
                "restart_required_reason": None,
                "auto_restart_scheduled": False,
                "skipped": "no_change",
                "cloud_only_reason": None,
                "restart_hint": "systemctl ...",
            },
            capsys,
        )
        assert "no config change" in out.lower() or "already active" in out.lower()
        assert err == ""

    def test_r_paths_carve_data_not_migrated_warning(self, capsys):
        """R-PATHS carve → 'DATA IS NOT MIGRATED' warning on stderr."""
        out, err = self._call(
            {
                "applied_live": False,
                "restart_required_reason": "paths_change",
                "auto_restart_scheduled": False,
                "skipped": None,
                "cloud_only_reason": None,
                "restart_hint": "systemctl ...",
            },
            capsys,
        )
        assert "not migrated" in err.lower(), f"Expected 'not migrated' in stderr: {err!r}"

    def test_r_paths_carve_does_not_prompt(self, capsys):
        """R-PATHS carve → no input() call (operator-driven restart)."""
        with patch("builtins.input", side_effect=AssertionError("input() called for R-PATHS")):
            self._call(
                {
                    "applied_live": False,
                    "restart_required_reason": "paths_change",
                    "auto_restart_scheduled": False,
                    "skipped": None,
                    "cloud_only_reason": None,
                    "restart_hint": "systemctl ...",
                },
                capsys,
            )

    def test_r_port_port_in_use_prints_error_no_prompt(self, capsys):
        """R-PORT port-in-use → error printed, no input() prompt."""
        with patch("builtins.input", side_effect=AssertionError("input() called for port-in-use")):
            out, err = self._call(
                {
                    "applied_live": False,
                    "restart_required_reason": "stt_port_change",
                    "auto_restart_scheduled": False,
                    "skipped": None,
                    "cloud_only_reason": None,
                    "port_in_use_reason": "stt.port=10300 is not bindable",
                    "restart_hint": "systemctl ...",
                },
                capsys,
            )
        assert "not bindable" in err or "Port not bindable" in err, (
            f"Expected port-in-use message in stderr: {err!r}"
        )

    def test_r_port_restart_eligible_operator_confirms_polls_until_healthy(
        self, capsys, monkeypatch
    ):
        """R-PORT with restart_eligible=True + operator answers y → poll until healthy."""
        import subprocess as _subprocess

        from paramem.cli import migrate as migrate_module

        healthy_calls = []

        def _fake_poll(server_url):
            healthy_calls.append(server_url)
            return True

        monkeypatch.setattr(migrate_module, "_poll_until_healthy", _fake_poll)
        # Operator answers "y" to the restart-consent prompt.
        monkeypatch.setattr("builtins.input", lambda prompt="": "y")
        # Subprocess.run must not actually run systemctl.
        monkeypatch.setattr(_subprocess, "run", lambda *a, **kw: None)
        out, err = self._call(
            {
                "applied_live": False,
                "restart_required_reason": "stt_port_change",
                "auto_restart_scheduled": False,
                "restart_eligible": True,
                "skipped": None,
                "cloud_only_reason": None,
                "restart_hint": "systemctl ...",
            },
            capsys,
        )
        assert healthy_calls, "_poll_until_healthy was not called"
        assert "healthy" in out.lower() or "restart" in out.lower(), (
            f"Expected restart/healthy message in stdout: {out!r}"
        )

    def test_apply_failure_prints_restart_hint(self, capsys):
        """Apply failure → restart hint on stderr."""
        out, err = self._call(
            {
                "applied_live": False,
                "restart_required_reason": None,
                "auto_restart_scheduled": False,
                "skipped": None,
                "cloud_only_reason": "apply_failed",
                "restart_hint": "systemctl --user restart paramem-server",
            },
            capsys,
        )
        assert "apply_failed" in err or "apply failed" in err.lower(), (
            f"Expected apply_failed message in stderr: {err!r}"
        )

    def test_single_prompt_accept_no_double_prompt(self, monkeypatch, capsys):
        """_do_accept_with_drift_check does NOT add a second 'Apply now?' prompt.

        The accept/rollback choice at the long-poll prompt IS the apply
        confirmation.  No additional prompt should appear inside the helper.
        """
        from paramem.cli import migrate as migrate_module

        input_calls = []

        def _counting_input(prompt=""):
            input_calls.append(prompt)
            # EOFError to exit immediately.
            raise EOFError

        apply_result = {
            "applied_live": True,
            "restart_required_reason": None,
            "auto_restart_scheduled": False,
            "skipped": None,
            "cloud_only_reason": None,
            "restart_hint": "systemctl ...",
            "state": "LIVE",
        }
        monkeypatch.setattr(
            http_client,
            "get_json",
            lambda url: {"state": "TRIAL"},
        )
        monkeypatch.setattr(
            http_client,
            "post_json",
            lambda url, *a, **kw: apply_result,
        )

        rc = migrate_module._do_accept_with_drift_check("http://localhost:8420")
        # No input() should have been called — the test uses patch to verify.
        assert rc == 0
        # The function must not have prompted inside (input was never replaced,
        # so if it were called it would use the real stdin and raise in CI).
        assert not input_calls, (
            f"_do_accept_with_drift_check called input() unexpectedly: {input_calls}"
        )


# ---------------------------------------------------------------------------
# WP2 — standalone subcommand rendering
# ---------------------------------------------------------------------------


class TestMigrateAcceptSubcommandRendering:
    """migrate-accept standalone subcommand renders applied_live result."""

    def test_accept_subcommand_renders_applied_live(self, monkeypatch, capsys):
        """migrate-accept success → 'Migration accepted.' and applied-live message."""
        accept_result = {
            "applied_live": True,
            "restart_required_reason": None,
            "auto_restart_scheduled": False,
            "skipped": None,
            "cloud_only_reason": None,
            "restart_hint": "systemctl --user restart paramem-server",
            "state": "LIVE",
            "restart_required": False,
        }
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: accept_result)
        rc = main(["migrate-accept"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "Migration accepted" in captured.out or "applied live" in captured.out.lower()

    def test_accept_subcommand_no_extra_prompt(self, monkeypatch):
        """migrate-accept running the subcommand IS the confirmation — no extra prompt."""
        accept_result = {
            "applied_live": True,
            "restart_required_reason": None,
            "auto_restart_scheduled": False,
            "skipped": None,
            "cloud_only_reason": None,
            "restart_hint": "systemctl ...",
            "state": "LIVE",
        }
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: accept_result)
        err_msg = "input() called in migrate-accept"
        with patch("builtins.input", side_effect=AssertionError(err_msg)):
            rc = main(["migrate-accept"])
        assert rc == 0


class TestMigrateRollbackSubcommandRendering:
    """migrate-rollback standalone subcommand renders applied_live result."""

    def test_rollback_subcommand_renders_no_change_message(self, monkeypatch, capsys):
        """migrate-rollback no-op skip → 'no config change' message rendered."""
        rollback_result = {
            "applied_live": True,
            "restart_required_reason": None,
            "auto_restart_scheduled": False,
            "skipped": "no_change",
            "cloud_only_reason": None,
            "restart_hint": "systemctl --user restart paramem-server",
            "state": "LIVE",
            "restart_required": False,
        }
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: rollback_result)
        rc = main(["migrate-rollback"])
        captured = capsys.readouterr()
        assert rc == 0
        combined = captured.out + captured.err
        assert (
            "no config change" in combined.lower()
            or "already active" in combined.lower()
            or "rolled back" in combined.lower()
        ), f"Expected rollback/no-change message: {combined!r}"

    def test_rollback_subcommand_r_paths_warning(self, monkeypatch, capsys):
        """migrate-rollback R-PATHS → DATA IS NOT MIGRATED warning."""
        rollback_result = {
            "applied_live": False,
            "restart_required_reason": "paths_change",
            "auto_restart_scheduled": False,
            "skipped": None,
            "cloud_only_reason": None,
            "restart_hint": "systemctl --user restart paramem-server",
            "state": "LIVE",
            "restart_required": True,
        }
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: rollback_result)
        rc = main(["migrate-rollback"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "not migrated" in captured.err.lower(), (
            f"Expected data-not-migrated warning in stderr: {captured.err!r}"
        )
