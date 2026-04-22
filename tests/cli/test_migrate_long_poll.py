"""Tests for the CLI long-poll flow in paramem.cli.migrate (Slice 3b.3).

Covers:
- STAGING-drift check before confirm (server state lost).
- Ctrl+C exits 130 with correct stderr message.
- fail/trial_exception → rollback-only prompt.
- pass/no_new_sessions → 3-way prompt (accept/rollback/cancel).
- Deferred cancel exits 0.
- Case-insensitive accept/rollback matching (IMPROVEMENT 6).
- Drift check before accept/rollback POSTs.
- Comparison report rendering.
"""

from __future__ import annotations

import pytest

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
    "unified_diff": "--- a\n+++ b\n@@ -1 +1 @@\n-debug: false\n+debug: true",
    "tier_diff": [],
    "shape_changes": [],
    "pre_flight_fail": None,
}

_STATUS_STAGING = {
    "state": "STAGING",
    "gates": None,
    "comparison_report": None,
    "server_started_at": "2026-04-22T00:00:00+00:00",
}

_STATUS_TRIAL_PENDING = {
    "state": "TRIAL",
    "gates": {"status": "pending"},
    "comparison_report": None,
    "server_started_at": "2026-04-22T00:00:00+00:00",
}

_STATUS_TRIAL_PASS = {
    "state": "TRIAL",
    "gates": {"status": "pass", "completed_at": "2026-04-22T01:00:00+00:00"},
    "comparison_report": {
        "schema_version": 1,
        "gates_status": "pass",
        "rows": [
            {"metric": "Triples extracted — last session", "pre_trial": "—", "trial": "—"},
            {"metric": "Recall on prior-cycle keys", "pre_trial": "—", "trial": "—"},
            {"metric": "Routing-probe classification", "pre_trial": "—", "trial": "—"},
            {"metric": "New ERROR lines in trial log", "pre_trial": "—", "trial": "—"},
            {"metric": "Graph shape", "pre_trial": "—", "trial": "—"},
        ],
        "operator_line": "These are the raw numbers before vs. after.",
    },
    "server_started_at": "2026-04-22T00:00:00+00:00",
}

_STATUS_TRIAL_FAIL = {
    "state": "TRIAL",
    "gates": {"status": "fail", "completed_at": "2026-04-22T01:00:00+00:00"},
    "comparison_report": None,
    "server_started_at": "2026-04-22T00:00:00+00:00",
}

_STATUS_TRIAL_NO_NEW_SESSIONS = {
    "state": "TRIAL",
    "gates": {"status": "no_new_sessions", "completed_at": "2026-04-22T01:00:00+00:00"},
    "comparison_report": {
        "schema_version": 1,
        "gates_status": "no_new_sessions",
        "rows": [],
        "operator_line": "These are the raw numbers before vs. after.",
    },
    "server_started_at": "2026-04-22T00:00:00+00:00",
}

_ACCEPT_RESPONSE = {
    "state": "LIVE",
    "trial_adapter_archive_path": "/abs/trial_adapters/ts",
    "restart_required": True,
    "restart_hint": "systemctl --user restart paramem-server",
    "pre_migration_backup_retained": True,
}

_ROLLBACK_RESPONSE = {
    "state": "LIVE",
    "trial_adapter_archive_path": "/abs/trial_adapters/ts",
    "rollback_pre_mortem_backup_path": "/abs/backups/config/ts",
    "restart_required": True,
    "restart_hint": "systemctl --user restart paramem-server",
}

_CONFIRM_RESPONSE = {"state": "TRIAL"}


def _make_get_responses(*status_sequence):
    """Return a get_json side-effect that iterates through status responses."""
    responses = list(status_sequence)
    index = [0]

    def _get(url, **kwargs):
        if "/migration/status" in url:
            r = responses[min(index[0], len(responses) - 1)]
            index[0] += 1
            return r
        raise AssertionError(f"Unexpected GET to {url!r}")

    return _get


def _make_post_responses(**url_responses):
    """Return a post_json side-effect dispatching by URL pattern."""

    def _post(url, body=None, **kwargs):
        for pattern, resp in url_responses.items():
            if pattern in url:
                if isinstance(resp, Exception):
                    raise resp
                return resp
        raise AssertionError(f"Unexpected POST to {url!r}")

    return _post


# ---------------------------------------------------------------------------
# STAGING-drift check (step 1): server not STAGING → exit 1
# ---------------------------------------------------------------------------


class TestStagingDriftCheck:
    def test_drift_check_server_returns_live_exits_1(self, monkeypatch, capsys):
        """If status is LIVE before confirm, CLI prints drift message and exits 1."""
        # get_json returns LIVE (server restarted during preview)
        status_live = {"state": "LIVE", "gates": None, "comparison_report": None}
        post_responses = {"preview": _BASE_PREVIEW}

        monkeypatch.setattr(http_client, "get_json", _make_get_responses(status_live))
        monkeypatch.setattr(
            http_client,
            "post_json",
            _make_post_responses(**post_responses),
        )
        # Patch input to return "y" for the Proceed? prompt.
        monkeypatch.setattr("builtins.input", lambda _="": "y")
        monkeypatch.setattr("time.sleep", lambda _: None)

        rc = main(["migrate", "/abs/server-new.yaml"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "candidate not staged" in captured.err or "Rerun" in captured.err

    def test_drift_check_server_returns_trial_exits_1(self, monkeypatch, capsys):
        """If status is TRIAL before confirm (another confirm already ran), exit 1."""
        status_trial = {"state": "TRIAL", "gates": None, "comparison_report": None}
        monkeypatch.setattr(http_client, "get_json", _make_get_responses(status_trial))
        monkeypatch.setattr(
            http_client,
            "post_json",
            _make_post_responses(**{"preview": _BASE_PREVIEW}),
        )
        monkeypatch.setattr("builtins.input", lambda _="": "y")
        monkeypatch.setattr("time.sleep", lambda _: None)

        rc = main(["migrate", "/abs/server-new.yaml"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "candidate not staged" in captured.err or "Rerun" in captured.err


# ---------------------------------------------------------------------------
# Ctrl+C exits 130
# ---------------------------------------------------------------------------


class TestCtrlCExit130:
    def test_ctrl_c_during_poll_exits_130(self, monkeypatch, capsys):
        """Ctrl+C during the long-poll loop exits 130 with the interrupt message."""
        call_count = [0]

        def _get_interrupted(url, **kwargs):
            call_count[0] += 1
            if "/migration/status" in url:
                if call_count[0] == 1:
                    # First call: STAGING drift check OK.
                    return _STATUS_STAGING
                # Second call: long-poll — raise KeyboardInterrupt.
                raise KeyboardInterrupt()
            raise AssertionError(f"Unexpected GET {url!r}")

        monkeypatch.setattr(http_client, "get_json", _get_interrupted)
        monkeypatch.setattr(
            http_client,
            "post_json",
            _make_post_responses(**{"preview": _BASE_PREVIEW, "confirm": _CONFIRM_RESPONSE}),
        )
        monkeypatch.setattr("builtins.input", lambda _="": "y")
        monkeypatch.setattr("time.sleep", lambda _: None)

        rc = main(["migrate", "/abs/server-new.yaml"])
        assert rc == 130
        captured = capsys.readouterr()
        assert "poll interrupted" in captured.err


# ---------------------------------------------------------------------------
# gates=fail → rollback-only prompt
# ---------------------------------------------------------------------------


class TestFailGatesRollbackPrompt:
    def test_fail_gates_prompts_rollback(self, monkeypatch, capsys):
        """gates=fail → rollback-only prompt; y → POST /migration/rollback."""
        get_sequence = [_STATUS_STAGING, _STATUS_TRIAL_FAIL]

        monkeypatch.setattr(http_client, "get_json", _make_get_responses(*get_sequence))
        monkeypatch.setattr(
            http_client,
            "post_json",
            _make_post_responses(
                **{
                    "preview": _BASE_PREVIEW,
                    "confirm": _CONFIRM_RESPONSE,
                    "rollback": _ROLLBACK_RESPONSE,
                    "status": _STATUS_TRIAL_FAIL,
                }
            ),
        )
        inputs = iter(["y", "y"])
        monkeypatch.setattr("builtins.input", lambda _="": next(inputs))
        monkeypatch.setattr("time.sleep", lambda _: None)

        rc = main(["migrate", "/abs/server-new.yaml"])
        assert rc == 0

    def test_fail_gates_n_answer_exits_1(self, monkeypatch, capsys):
        """gates=fail → rollback prompt; N → exit 1."""
        get_sequence = [_STATUS_STAGING, _STATUS_TRIAL_FAIL]

        monkeypatch.setattr(http_client, "get_json", _make_get_responses(*get_sequence))
        monkeypatch.setattr(
            http_client,
            "post_json",
            _make_post_responses(**{"preview": _BASE_PREVIEW, "confirm": _CONFIRM_RESPONSE}),
        )
        inputs = iter(["y", "N"])
        monkeypatch.setattr("builtins.input", lambda _="": next(inputs))
        monkeypatch.setattr("time.sleep", lambda _: None)

        rc = main(["migrate", "/abs/server-new.yaml"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "migrate-rollback" in captured.err

    def test_trial_exception_gates_prompts_rollback_only(self, monkeypatch, capsys):
        """gates=trial_exception → rollback-only prompt (no accept offered)."""
        status_exception = {
            "state": "TRIAL",
            "gates": {
                "status": "trial_exception",
                "completed_at": "2026-04-22T01:00:00+00:00",
            },
            "comparison_report": None,
            "server_started_at": "",
        }
        get_sequence = [_STATUS_STAGING, status_exception]

        monkeypatch.setattr(http_client, "get_json", _make_get_responses(*get_sequence))
        monkeypatch.setattr(
            http_client,
            "post_json",
            _make_post_responses(**{"preview": _BASE_PREVIEW, "confirm": _CONFIRM_RESPONSE}),
        )
        inputs = iter(["y", "N"])
        monkeypatch.setattr("builtins.input", lambda _="": next(inputs))
        monkeypatch.setattr("time.sleep", lambda _: None)

        rc = main(["migrate", "/abs/server-new.yaml"])
        # N on rollback prompt → exit 1
        assert rc == 1


# ---------------------------------------------------------------------------
# gates=pass → 3-way prompt
# ---------------------------------------------------------------------------


class TestPassGates3WayPrompt:
    def test_pass_gates_accept_answer_calls_accept(self, monkeypatch, capsys):
        """gates=pass → 3-way prompt → 'accept' calls POST /migration/accept."""
        # 3 GET calls: drift check, poll pass, pre-accept drift check
        get_sequence = [
            _STATUS_STAGING,
            _STATUS_TRIAL_PASS,
            _STATUS_TRIAL_PASS,  # drift check before accept
        ]
        monkeypatch.setattr(http_client, "get_json", _make_get_responses(*get_sequence))
        monkeypatch.setattr(
            http_client,
            "post_json",
            _make_post_responses(
                **{
                    "preview": _BASE_PREVIEW,
                    "confirm": _CONFIRM_RESPONSE,
                    "accept": _ACCEPT_RESPONSE,
                }
            ),
        )
        inputs = iter(["y", "accept"])
        monkeypatch.setattr("builtins.input", lambda _="": next(inputs))
        monkeypatch.setattr("time.sleep", lambda _: None)

        rc = main(["migrate", "/abs/server-new.yaml"])
        assert rc == 0

    def test_pass_gates_rollback_answer_calls_rollback(self, monkeypatch, capsys):
        """gates=pass → 3-way prompt → 'rollback' calls POST /migration/rollback."""
        get_sequence = [_STATUS_STAGING, _STATUS_TRIAL_PASS, _STATUS_TRIAL_PASS]
        monkeypatch.setattr(http_client, "get_json", _make_get_responses(*get_sequence))
        monkeypatch.setattr(
            http_client,
            "post_json",
            _make_post_responses(
                **{
                    "preview": _BASE_PREVIEW,
                    "confirm": _CONFIRM_RESPONSE,
                    "rollback": _ROLLBACK_RESPONSE,
                }
            ),
        )
        inputs = iter(["y", "rollback"])
        monkeypatch.setattr("builtins.input", lambda _="": next(inputs))
        monkeypatch.setattr("time.sleep", lambda _: None)

        rc = main(["migrate", "/abs/server-new.yaml"])
        assert rc == 0

    def test_pass_gates_cancel_defers(self, monkeypatch, capsys):
        """gates=pass → 3-way prompt → 'c' defers; exits 0."""
        get_sequence = [_STATUS_STAGING, _STATUS_TRIAL_PASS]
        monkeypatch.setattr(http_client, "get_json", _make_get_responses(*get_sequence))
        monkeypatch.setattr(
            http_client,
            "post_json",
            _make_post_responses(**{"preview": _BASE_PREVIEW, "confirm": _CONFIRM_RESPONSE}),
        )
        inputs = iter(["y", "c"])
        monkeypatch.setattr("builtins.input", lambda _="": next(inputs))
        monkeypatch.setattr("time.sleep", lambda _: None)

        rc = main(["migrate", "/abs/server-new.yaml"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "deferred" in captured.out or "deferred" in captured.err

    def test_pass_gates_empty_answer_defers(self, monkeypatch, capsys):
        """Empty answer on 3-way prompt defers; exits 0."""
        get_sequence = [_STATUS_STAGING, _STATUS_TRIAL_PASS]
        monkeypatch.setattr(http_client, "get_json", _make_get_responses(*get_sequence))
        monkeypatch.setattr(
            http_client,
            "post_json",
            _make_post_responses(**{"preview": _BASE_PREVIEW, "confirm": _CONFIRM_RESPONSE}),
        )
        inputs = iter(["y", ""])
        monkeypatch.setattr("builtins.input", lambda _="": next(inputs))
        monkeypatch.setattr("time.sleep", lambda _: None)

        rc = main(["migrate", "/abs/server-new.yaml"])
        assert rc == 0

    def test_no_new_sessions_gates_accept_works(self, monkeypatch, capsys):
        """gates=no_new_sessions → accept is still offered (accept-eligible)."""
        get_sequence = [
            _STATUS_STAGING,
            _STATUS_TRIAL_NO_NEW_SESSIONS,
            _STATUS_TRIAL_NO_NEW_SESSIONS,  # drift check
        ]
        monkeypatch.setattr(http_client, "get_json", _make_get_responses(*get_sequence))
        monkeypatch.setattr(
            http_client,
            "post_json",
            _make_post_responses(
                **{
                    "preview": _BASE_PREVIEW,
                    "confirm": _CONFIRM_RESPONSE,
                    "accept": _ACCEPT_RESPONSE,
                }
            ),
        )
        inputs = iter(["y", "a"])
        monkeypatch.setattr("builtins.input", lambda _="": next(inputs))
        monkeypatch.setattr("time.sleep", lambda _: None)

        rc = main(["migrate", "/abs/server-new.yaml"])
        assert rc == 0


# ---------------------------------------------------------------------------
# Case-insensitive matching (IMPROVEMENT 6)
# ---------------------------------------------------------------------------


class TestCaseInsensitiveMatching:
    @pytest.mark.parametrize("answer", ["A", "Accept", "ACCEPT"])
    def test_accept_case_insensitive(self, answer, monkeypatch, capsys):
        """'A', 'Accept', 'ACCEPT' all route to accept (IMPROVEMENT 6)."""
        get_sequence = [_STATUS_STAGING, _STATUS_TRIAL_PASS, _STATUS_TRIAL_PASS]
        monkeypatch.setattr(http_client, "get_json", _make_get_responses(*get_sequence))
        monkeypatch.setattr(
            http_client,
            "post_json",
            _make_post_responses(
                **{
                    "preview": _BASE_PREVIEW,
                    "confirm": _CONFIRM_RESPONSE,
                    "accept": _ACCEPT_RESPONSE,
                }
            ),
        )
        inputs = iter(["y", answer])
        monkeypatch.setattr("builtins.input", lambda _="": next(inputs))
        monkeypatch.setattr("time.sleep", lambda _: None)

        rc = main(["migrate", "/abs/server-new.yaml"])
        assert rc == 0

    @pytest.mark.parametrize("answer", ["R", "Rollback", "ROLLBACK"])
    def test_rollback_case_insensitive(self, answer, monkeypatch, capsys):
        """'R', 'Rollback', 'ROLLBACK' all route to rollback (IMPROVEMENT 6)."""
        get_sequence = [_STATUS_STAGING, _STATUS_TRIAL_PASS, _STATUS_TRIAL_PASS]
        monkeypatch.setattr(http_client, "get_json", _make_get_responses(*get_sequence))
        monkeypatch.setattr(
            http_client,
            "post_json",
            _make_post_responses(
                **{
                    "preview": _BASE_PREVIEW,
                    "confirm": _CONFIRM_RESPONSE,
                    "rollback": _ROLLBACK_RESPONSE,
                }
            ),
        )
        inputs = iter(["y", answer])
        monkeypatch.setattr("builtins.input", lambda _="": next(inputs))
        monkeypatch.setattr("time.sleep", lambda _: None)

        rc = main(["migrate", "/abs/server-new.yaml"])
        assert rc == 0


# ---------------------------------------------------------------------------
# Drift check before accept/rollback
# ---------------------------------------------------------------------------


class TestDriftCheckBeforeAction:
    def test_drift_before_accept_exits_1_if_trial_lost(self, monkeypatch, capsys):
        """Drift check before accept: TRIAL lost → exit 1."""
        # After pass gates, status drifts to LIVE.
        get_sequence = [
            _STATUS_STAGING,
            _STATUS_TRIAL_PASS,
            {"state": "LIVE", "gates": None, "comparison_report": None},
        ]
        monkeypatch.setattr(http_client, "get_json", _make_get_responses(*get_sequence))
        monkeypatch.setattr(
            http_client,
            "post_json",
            _make_post_responses(**{"preview": _BASE_PREVIEW, "confirm": _CONFIRM_RESPONSE}),
        )
        inputs = iter(["y", "accept"])
        monkeypatch.setattr("builtins.input", lambda _="": next(inputs))
        monkeypatch.setattr("time.sleep", lambda _: None)

        rc = main(["migrate", "/abs/server-new.yaml"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "drifted" in captured.err

    def test_drift_before_rollback_exits_1_if_trial_lost(self, monkeypatch, capsys):
        """Drift check before rollback: TRIAL lost → exit 1."""
        get_sequence = [
            _STATUS_STAGING,
            _STATUS_TRIAL_FAIL,
            {"state": "LIVE", "gates": None, "comparison_report": None},
        ]
        monkeypatch.setattr(http_client, "get_json", _make_get_responses(*get_sequence))
        monkeypatch.setattr(
            http_client,
            "post_json",
            _make_post_responses(**{"preview": _BASE_PREVIEW, "confirm": _CONFIRM_RESPONSE}),
        )
        inputs = iter(["y", "y"])
        monkeypatch.setattr("builtins.input", lambda _="": next(inputs))
        monkeypatch.setattr("time.sleep", lambda _: None)

        rc = main(["migrate", "/abs/server-new.yaml"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "drifted" in captured.err


# ---------------------------------------------------------------------------
# Rollback with archive_warning (degraded success — body-content detection)
# ---------------------------------------------------------------------------


_ROLLBACK_207_BODY = {
    "state": "LIVE",
    "trial_adapter_archive_path": None,
    "rollback_pre_mortem_backup_path": "/abs/backups/config/rb-ts",
    "restart_required": True,
    "restart_hint": "systemctl --user restart paramem-server",
    "archive_warning": {
        "path": "/abs/state/trial_adapter",
        "message": "shutil.move failed: permission denied",
    },
}


class TestRollbackWithArchiveWarning:
    def test_rollback_archive_warning_body_exits_0(self, monkeypatch, capsys):
        """gates=fail + rollback returns archive_warning body → exit 0.

        HTTP 207 passes through post_json as a plain dict (207 < 400), so the
        CLI detects degraded rollback via the ``archive_warning`` key, not by
        catching a ServerHTTPError.  The long-poll rollback path must handle
        this correctly.
        """
        get_sequence = [_STATUS_STAGING, _STATUS_TRIAL_FAIL, _STATUS_TRIAL_FAIL]
        monkeypatch.setattr(http_client, "get_json", _make_get_responses(*get_sequence))
        monkeypatch.setattr(
            http_client,
            "post_json",
            _make_post_responses(
                **{
                    "preview": _BASE_PREVIEW,
                    "confirm": _CONFIRM_RESPONSE,
                    "rollback": _ROLLBACK_207_BODY,
                }
            ),
        )
        inputs = iter(["y", "y"])
        monkeypatch.setattr("builtins.input", lambda _="": next(inputs))
        monkeypatch.setattr("time.sleep", lambda _: None)

        rc = main(["migrate", "/abs/server-new.yaml"])
        assert rc == 0, f"Expected exit 0 for degraded rollback (archive_warning), got {rc}"

    def test_rollback_archive_warning_printed_to_stderr(self, monkeypatch, capsys):
        """Degraded rollback archive_warning message appears in stderr (operator action needed)."""
        get_sequence = [_STATUS_STAGING, _STATUS_TRIAL_FAIL, _STATUS_TRIAL_FAIL]
        monkeypatch.setattr(http_client, "get_json", _make_get_responses(*get_sequence))
        monkeypatch.setattr(
            http_client,
            "post_json",
            _make_post_responses(
                **{
                    "preview": _BASE_PREVIEW,
                    "confirm": _CONFIRM_RESPONSE,
                    "rollback": _ROLLBACK_207_BODY,
                }
            ),
        )
        inputs = iter(["y", "y"])
        monkeypatch.setattr("builtins.input", lambda _="": next(inputs))
        monkeypatch.setattr("time.sleep", lambda _: None)

        main(["migrate", "/abs/server-new.yaml"])
        captured = capsys.readouterr()
        # Warning goes to stderr (operator action needed signal alongside main stdout flow).
        has_warning = (
            "permission denied" in captured.err
            or "archive_warning" in captured.err
            or "rotation failed" in captured.err
        )
        assert has_warning, f"Expected archive_warning content on stderr, got: {captured.err!r}"
