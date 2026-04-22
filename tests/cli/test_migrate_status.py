"""Tests for paramem.cli.migrate_status (Slice 3b.1).

Covers:
- Normal MigrationStatusResponse render.
- ServerUnreachable → fallback to state/trial.json.
- state/trial.json exists → print contents.
- state/trial.json does not exist → print "server offline; no trial marker on disk" + exit 0.
- --json mode.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from paramem.cli import http_client
from paramem.cli.main import main

_STATUS_RESPONSE = {
    "state": "LIVE",
    "candidate_path": None,
    "candidate_hash": None,
    "staged_at": None,
    "simulate_mode_override": False,
    "consolidating": False,
    "server_started_at": "2026-04-22T00:00:00+00:00",
}

_STATUS_STAGING = {
    "state": "STAGING",
    "candidate_path": "/abs/server-new.yaml",
    "candidate_hash": "abc123",
    "staged_at": "2026-04-22T01:00:00+00:00",
    "simulate_mode_override": False,
    "consolidating": False,
    "server_started_at": "2026-04-22T00:00:00+00:00",
}


# ---------------------------------------------------------------------------
# Normal render
# ---------------------------------------------------------------------------


class TestMigrateStatusNormalRender:
    def test_normal_render_key_value_lines(self, monkeypatch, capsys):
        """Normal response → key: value lines on stdout."""
        monkeypatch.setattr(http_client, "get_json", lambda *a, **kw: _STATUS_RESPONSE)
        rc = main(["migrate-status"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "state" in captured.out

    def test_normal_render_staging_state(self, monkeypatch, capsys):
        """STAGING response renders candidate_path."""
        monkeypatch.setattr(http_client, "get_json", lambda *a, **kw: _STATUS_STAGING)
        rc = main(["migrate-status"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "STAGING" in captured.out or "staging" in captured.out.lower()

    def test_json_mode_emits_raw_json(self, monkeypatch, capsys):
        """--json emits raw JSON."""
        monkeypatch.setattr(http_client, "get_json", lambda *a, **kw: _STATUS_RESPONSE)
        rc = main(["migrate-status", "--json"])
        captured = capsys.readouterr()
        assert rc == 0
        parsed = json.loads(captured.out)
        assert parsed["state"] == "LIVE"

    def test_404_exits_1_with_message(self, monkeypatch, capsys):
        """ServerUnavailable (404) → exit 1, message on stderr."""

        def _raise(*a, **kw):
            raise http_client.ServerUnavailable("404")

        monkeypatch.setattr(http_client, "get_json", _raise)
        rc = main(["migrate-status"])
        captured = capsys.readouterr()
        assert rc == 1
        assert "/migration/status" in captured.err or "status" in captured.err.lower()

    def test_http_error_exits_1(self, monkeypatch, capsys):
        """ServerHTTPError → exit 1."""

        def _raise(*a, **kw):
            raise http_client.ServerHTTPError(500, "/migration/status", "internal error")

        monkeypatch.setattr(http_client, "get_json", _raise)
        rc = main(["migrate-status"])
        assert rc == 1


# ---------------------------------------------------------------------------
# ServerUnreachable → fallback to state/trial.json
# ---------------------------------------------------------------------------


class TestMigrateStatusOfflineFallback:
    def _unreachable(*a, **kw):
        raise http_client.ServerUnreachable("connection refused")

    def test_offline_no_trial_json_prints_message_exits_0(self, monkeypatch, capsys, tmp_path):
        """Server offline, no trial.json → print message, exit 0 (not 2)."""
        monkeypatch.setattr(
            http_client,
            "get_json",
            lambda *a, **kw: (_ for _ in ()).throw(http_client.ServerUnreachable("refused")),
        )
        # Patch the trial.json path to a non-existent file in tmp_path
        with patch(
            "paramem.cli.migrate_status._trial_json_path",
            return_value=tmp_path / "nonexistent.json",
        ):
            rc = main(["migrate-status"])
        captured = capsys.readouterr()
        assert rc == 0, f"Expected exit 0 (not an error), got {rc}"
        assert "offline" in captured.out.lower() or "no trial marker" in captured.out.lower()

    def test_offline_with_trial_json_prints_contents(self, monkeypatch, capsys, tmp_path):
        """Server offline + trial.json exists → print contents, exit 0."""
        trial_data = {"state": "trial", "started_at": "2026-04-22T01:00:00+00:00"}
        trial_path = tmp_path / "trial.json"
        trial_path.write_text(json.dumps(trial_data), encoding="utf-8")

        monkeypatch.setattr(
            http_client,
            "get_json",
            lambda *a, **kw: (_ for _ in ()).throw(http_client.ServerUnreachable("refused")),
        )
        with patch("paramem.cli.migrate_status._trial_json_path", return_value=trial_path):
            rc = main(["migrate-status"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "offline" in captured.out.lower()
        assert "trial" in captured.out.lower()

    def test_offline_with_trial_json_json_mode(self, monkeypatch, capsys, tmp_path):
        """Server offline + trial.json + --json → emit JSON contents."""
        trial_data = {"state": "trial", "started_at": "2026-04-22T01:00:00+00:00"}
        trial_path = tmp_path / "trial.json"
        trial_path.write_text(json.dumps(trial_data), encoding="utf-8")

        monkeypatch.setattr(
            http_client,
            "get_json",
            lambda *a, **kw: (_ for _ in ()).throw(http_client.ServerUnreachable("refused")),
        )
        with patch("paramem.cli.migrate_status._trial_json_path", return_value=trial_path):
            rc = main(["migrate-status", "--json"])
        captured = capsys.readouterr()
        assert rc == 0
        # JSON should appear in stdout after the offline line
        lines = captured.out.strip().splitlines()
        # Find the JSON block (skip the "server offline" line)
        json_lines = [
            ln for ln in lines if ln.strip().startswith("{") or ln.strip().startswith('"')
        ]
        assert json_lines, f"No JSON found in output: {captured.out!r}"

    def test_offline_no_trial_json_exit_is_0_not_2(self, monkeypatch, tmp_path):
        """Offline + no trial.json → exit 0, NOT exit 2 (spec §L228)."""
        monkeypatch.setattr(
            http_client,
            "get_json",
            lambda *a, **kw: (_ for _ in ()).throw(http_client.ServerUnreachable("refused")),
        )
        with patch(
            "paramem.cli.migrate_status._trial_json_path", return_value=tmp_path / "none.json"
        ):
            rc = main(["migrate-status"])
        assert rc == 0, (
            f"Offline + no trial.json must exit 0 (absence of trial marker is not an "
            f"error per spec §L228); got {rc}"
        )
