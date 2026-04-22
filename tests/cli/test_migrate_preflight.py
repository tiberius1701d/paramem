"""Tests for migrate CLI pre-flight short-circuit (Slice 6b).

Tests cover:
9  — pre_flight_fail="disk_pressure" → spec message + paramem backup-prune + rc=1
10 — pre_flight_fail="something_new" → generic message + rc=1
11 — pre_flight_fail=None → reaches proceed prompt (operator N → cancel + rc=1)
"""

from __future__ import annotations

from unittest.mock import patch

from paramem.cli import http_client
from paramem.cli.main import main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _preview_response(pre_flight_fail=None, used_gb=None, cap_gb=None):
    """Build a minimal /migration/preview response dict."""
    resp = {
        "state": "LIVE" if pre_flight_fail else "STAGING",
        "candidate_path": "/tmp/cand.yaml",
        "candidate_hash": "abc123",
        "staged_at": "2026-04-22T00:00:00+00:00",
        "simulate_mode_override": False,
        "unified_diff": "--- a\n+++ b\n@@ -1 +1 @@ debug: false\n+debug: true",
        "tier_diff": [
            {"tier": "operational", "dotted_path": "debug", "old_value": False, "new_value": True}
        ],
        "shape_changes": [],
        "pre_flight_fail": pre_flight_fail,
    }
    if used_gb is not None:
        resp["pre_flight_disk_used_gb"] = used_gb
    if cap_gb is not None:
        resp["pre_flight_disk_cap_gb"] = cap_gb
    return resp


# ---------------------------------------------------------------------------
# Test 9 — disk_pressure pre-flight fail → spec message, backup-prune, rc=1
# ---------------------------------------------------------------------------


class TestMigrateCliPrintsPreFlightMessageAndExits1:
    def test_migrate_cli_prints_preflight_message_and_exits_1(self, monkeypatch, capsys) -> None:
        """pre_flight_fail='disk_pressure' → spec L582–586 message; rc=1; no cancel POST."""
        cancel_called = []

        def _fake_post(url, body=None, **kw):
            if "cancel" in url:
                cancel_called.append(url)
            return _preview_response(pre_flight_fail="disk_pressure", used_gb=18.5, cap_gb=20.0)

        monkeypatch.setattr(http_client, "post_json", _fake_post)

        rc = main(["migrate", "/tmp/cand.yaml"])
        captured = capsys.readouterr()

        assert rc == 1, f"Expected rc=1, got {rc}"
        # Spec L582–586 wording.
        assert "disk" in captured.err.lower(), f"Stderr must mention disk: {captured.err!r}"
        assert "backup-prune" in captured.err, f"Stderr must mention backup-prune: {captured.err!r}"
        assert "18.50" in captured.err or "18.5" in captured.err, (
            f"Stderr must include used_gb: {captured.err!r}"
        )
        assert "20.00" in captured.err or "20.0" in captured.err, (
            f"Stderr must include cap_gb: {captured.err!r}"
        )
        # No /migration/cancel must be POSTed (state is LIVE, nothing to cancel).
        assert not cancel_called, f"cancel should not have been called: {cancel_called}"


# ---------------------------------------------------------------------------
# Test 10 — unknown pre_flight code → generic message + rc=1
# ---------------------------------------------------------------------------


class TestMigrateCliUnknownPreFlightCodeExits1:
    def test_migrate_cli_unknown_preflight_code_exits_1(self, monkeypatch, capsys) -> None:
        """pre_flight_fail='something_new' → generic stderr + rc=1."""

        def _fake_post(url, body=None, **kw):
            return _preview_response(pre_flight_fail="something_new")

        monkeypatch.setattr(http_client, "post_json", _fake_post)

        rc = main(["migrate", "/tmp/cand.yaml"])
        captured = capsys.readouterr()

        assert rc == 1, f"Expected rc=1, got {rc}"
        assert "pre-flight check failed" in captured.err.lower(), (
            f"Stderr must say 'pre-flight check failed': {captured.err!r}"
        )
        assert "something_new" in captured.err, (
            f"Stderr must include the unknown code: {captured.err!r}"
        )


# ---------------------------------------------------------------------------
# Test 11 — pre_flight_fail=None → reaches proceed prompt; N → cancel + rc=1
# ---------------------------------------------------------------------------


class TestMigrateCliNoPreflightContinuesToPrompt:
    def test_migrate_cli_no_preflight_continues_to_prompt(self, monkeypatch, capsys) -> None:
        """pre_flight_fail=None → reach Proceed? prompt; N → /migration/cancel + rc=1."""
        cancel_called = []

        def _fake_post(url, body=None, **kw):
            if "cancel" in url:
                cancel_called.append(url)
                return {"state": "LIVE", "cleared_path": ""}
            if "preview" in url:
                return _preview_response(pre_flight_fail=None)
            return {}

        monkeypatch.setattr(http_client, "post_json", _fake_post)

        # Simulate operator pressing N at the prompt.
        with patch("builtins.input", return_value="N"):
            rc = main(["migrate", "/tmp/cand.yaml"])

        assert rc == 1, f"Expected rc=1 (N at prompt), got {rc}"
        assert any("cancel" in url for url in cancel_called), (
            "Expected /migration/cancel to be POSTed on N"
        )
