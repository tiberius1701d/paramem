"""Tests for paramem.cli.migrate_status (Slice 3b.1 + 3b.2 path correction).

Covers:
- Normal MigrationStatusResponse render.
- ServerUnreachable → fallback to state/trial.json.
- state/trial.json exists → print contents.
- state/trial.json does not exist → print "server offline; no trial marker on disk" + exit 0.
- --json mode.
- Correction 4: _trial_json_path returns data/ha/state/trial.json (not cwd-relative state/).
- Marker-present: offline + real TrialMarker JSON at data/ha/state/trial.json → renders TRIAL.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from paramem.cli import http_client
from paramem.cli.main import main
from paramem.cli.migrate_status import _trial_json_path
from paramem.server.trial_state import TRIAL_MARKER_SCHEMA_VERSION, TrialMarker

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


# ---------------------------------------------------------------------------
# Correction 4: path constant is data/ha/state/trial.json (not state/trial.json)
# ---------------------------------------------------------------------------


class TestTrialJsonPathCorrection4:
    """Verify that _trial_json_path returns data/ha/state/trial.json.

    Correction 4 (Slice 3b.2): the marker lives at data/ha/state/trial.json
    inside the data/ha/ tree, NOT at a bare state/trial.json path relative to
    the project root.  This test asserts the path constant directly so a
    future regression is caught at the unit level.
    """

    def test_migrate_status_offline_reads_data_ha_state_trial_json(self):
        """_trial_json_path() returns Path('data/ha/state/trial.json').

        The path must match data/ha/state/trial.json exactly — a cwd-relative
        state/trial.json would fail to locate the marker when the server writes
        to data/ha/state/ (spec §L516–545, Correction 4).
        """
        path = _trial_json_path("http://localhost:8420")
        expected = Path("data") / "ha" / "state" / "trial.json"
        assert path == expected, (
            f"_trial_json_path returned {path!r}; expected {expected!r}. "
            "Correction 4: marker lives at data/ha/state/trial.json, not state/trial.json."
        )

    def test_migrate_status_offline_marker_present_renders_trial_state(
        self, monkeypatch, capsys, tmp_path
    ):
        """Offline fallback reads a real TrialMarker JSON and renders TRIAL state.

        Sets up data/ha/state/trial.json inside tmp_path with a valid
        TrialMarker dict (Slice 3b.2 schema), redirects _trial_json_path to
        that location, and confirms migrate-status prints the marker contents
        including the started_at timestamp.

        This verifies that the path resolution and the render loop both work
        end-to-end when a real marker is present.
        """
        marker = TrialMarker(
            schema_version=TRIAL_MARKER_SCHEMA_VERSION,
            started_at="2026-04-22T01:00:00+00:00",
            pre_trial_config_sha256="a" * 64,
            candidate_config_sha256="b" * 64,
            backup_paths={
                "config": str(tmp_path / "backups" / "config" / "slot1"),
                "graph": str(tmp_path / "backups" / "graph" / "slot1"),
                "registry": str(tmp_path / "backups" / "registry" / "slot1"),
            },
            trial_adapter_dir=str(tmp_path / "data" / "ha" / "state" / "trial_adapter"),
            trial_graph_dir=str(tmp_path / "data" / "ha" / "state" / "trial_graph"),
        )
        marker_path = tmp_path / "data" / "ha" / "state" / "trial.json"
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(json.dumps(marker.to_dict()), encoding="utf-8")

        monkeypatch.setattr(
            http_client,
            "get_json",
            lambda *a, **kw: (_ for _ in ()).throw(http_client.ServerUnreachable("refused")),
        )
        with patch("paramem.cli.migrate_status._trial_json_path", return_value=marker_path):
            rc = main(["migrate-status"])

        captured = capsys.readouterr()
        assert rc == 0
        # Offline fallback banner must appear.
        assert "offline" in captured.out.lower(), f"Expected 'offline' in: {captured.out!r}"
        # Key fields from the marker must be rendered.
        assert "2026-04-22T01:00:00" in captured.out, (
            f"started_at not found in output: {captured.out!r}"
        )
