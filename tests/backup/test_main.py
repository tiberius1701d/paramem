"""Tests for the scheduled-backup CLI entry point (``paramem.backup.__main__``).

Bundle backups require the running server (passphrase + adapter context), so
the runner delegates to the server via ``POST /backup/create`` when it is
reachable.  When the server is unreachable, the runner records a
degraded/skipped ``ScheduledBackupResult`` (success=False) via
``update_backup_state`` and exits 0 (best-effort, non-fatal).

These tests exercise that branching with no server, no GPU, and no real
config load — ``load_server_config`` is stubbed to a lightweight namespace.
"""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paramem.backup import __main__ as backup_main
from paramem.cli import http_client


def _fake_config(tmp_path: Path, *, schedule: str = "daily 04:00", port: int = 8420):
    """Minimal stand-in for ServerConfig with the attributes ``__main__`` reads."""
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    backups = types.SimpleNamespace(
        schedule=schedule,
        artifacts=["config", "graph", "registry"],
    )
    return types.SimpleNamespace(
        paths=types.SimpleNamespace(data=data),
        security=types.SimpleNamespace(backups=backups),
        server=types.SimpleNamespace(port=port),
    )


@pytest.fixture()
def config_file(tmp_path: Path, monkeypatch):
    """A real config path on disk + ``load_server_config`` stubbed to a fake config."""
    cfg_path = tmp_path / "server.yaml"
    cfg_path.write_text("model: mistral\n")
    fake = _fake_config(tmp_path)
    monkeypatch.setattr("paramem.server.config.load_server_config", lambda _p: fake)
    return cfg_path, fake


def test_delegates_to_server_when_reachable(config_file, monkeypatch):
    """Server up → POST /backup/create with the configured kinds + tier; no standalone run."""
    cfg_path, _ = config_file
    calls = {}

    def fake_post(url, body, *, timeout, token=None):
        calls["url"] = url
        calls["body"] = body
        calls["token"] = token
        return {
            "success": True,
            "tier": "daily",
            "written_slots": {"config": "/x", "graph": "/y", "registry": "/z"},
            "skipped_artifacts": [],
            "error": None,
        }

    monkeypatch.setattr("paramem.cli.http_client.post_json", fake_post)

    def _boom(*a, **k):
        raise AssertionError("run_scheduled_backup must not run when delegating to the server")

    monkeypatch.setattr("paramem.backup.runner.run_scheduled_backup", _boom)

    rc = backup_main.main(["--config", str(cfg_path), "--tier", "daily"])
    assert rc == 0
    assert calls["url"].endswith("/backup/create")
    assert calls["body"]["tier"] == "daily"
    assert calls["body"]["kinds"] == ["config", "graph", "registry"]


def test_falls_back_to_degraded_state_when_unreachable(config_file, monkeypatch):
    """Server down → degraded/skipped state recorded; run_scheduled_backup NOT called.

    Bundle backups require the server (passphrase, adapter context).  When
    the server is unreachable, the standalone path records a ScheduledBackupResult
    with success=False and exits 0 (best-effort, non-fatal).
    """
    cfg_path, _ = config_file
    from paramem.cli import http_client

    def fake_post(url, body, *, timeout, token=None):
        raise http_client.ServerUnreachable("connection refused")

    monkeypatch.setattr("paramem.cli.http_client.post_json", fake_post)

    def _boom(**kwargs):
        raise AssertionError("run_scheduled_backup must not run when server is unreachable")

    monkeypatch.setattr("paramem.backup.runner.run_scheduled_backup", _boom)

    state_written = {}

    def fake_update(state_dir, result):
        state_written["result"] = result

    monkeypatch.setattr("paramem.backup.state.update_backup_state", fake_update)

    rc = backup_main.main(["--config", str(cfg_path), "--tier", "daily"])
    assert rc == 0
    # Degraded state must be recorded.
    assert "result" in state_written, "update_backup_state was not called"
    result = state_written["result"]
    assert result.success is False
    assert result.error == "server unavailable — bundle backup requires the running server"
    assert result.written_slots == {}


def test_falls_back_to_degraded_state_when_endpoint_missing(config_file, monkeypatch):
    """Old server without /backup/create (404) → degraded state; no standalone backup."""
    cfg_path, _ = config_file
    from paramem.cli import http_client

    def fake_post(url, body, *, timeout, token=None):
        raise http_client.ServerUnavailable("404 from /backup/create")

    monkeypatch.setattr("paramem.cli.http_client.post_json", fake_post)

    def _boom(**kwargs):
        raise AssertionError("run_scheduled_backup must not run when server is unreachable")

    monkeypatch.setattr("paramem.backup.runner.run_scheduled_backup", _boom)

    state_written = {}

    def fake_update(state_dir, result):
        state_written["result"] = result

    monkeypatch.setattr("paramem.backup.state.update_backup_state", fake_update)

    rc = backup_main.main(["--config", str(cfg_path), "--tier", "daily"])
    assert rc == 0
    assert "result" in state_written, "update_backup_state was not called"
    assert state_written["result"].success is False
    assert "server unavailable" in state_written["result"].error


def test_schedule_off_is_noop(tmp_path, monkeypatch):
    """schedule=off → no server contact, no standalone run, exit 0."""
    cfg_path = tmp_path / "server.yaml"
    cfg_path.write_text("model: mistral\n")
    fake = _fake_config(tmp_path, schedule="off")
    monkeypatch.setattr("paramem.server.config.load_server_config", lambda _p: fake)

    def _boom_post(*a, **k):
        raise AssertionError("must not contact the server when schedule=off")

    monkeypatch.setattr("paramem.cli.http_client.post_json", _boom_post)

    rc = backup_main.main(["--config", str(cfg_path)])
    assert rc == 0


def test_delegated_failure_returns_1(config_file, monkeypatch):
    """Server returns success=False (e.g. disk pressure) → exit 1."""
    cfg_path, _ = config_file

    def fake_post(url, body, *, timeout, token=None):
        return {
            "success": False,
            "tier": "daily",
            "written_slots": {},
            "skipped_artifacts": [],
            "error": "disk_pressure: cap reached",
        }

    monkeypatch.setattr("paramem.cli.http_client.post_json", fake_post)

    rc = backup_main.main(["--config", str(cfg_path), "--tier", "daily"])
    assert rc == 1


def test_server_http_error_returns_1_without_fallback(config_file, monkeypatch):
    """A non-404 HTTP error (e.g. 500) → exit 1, no standalone fallback.

    A malformed/broken request must not be silently retried standalone —
    that would mask the failure and could produce a graphless backup.
    """
    cfg_path, _ = config_file
    from paramem.cli import http_client

    def fake_post(url, body, *, timeout, token=None):
        raise http_client.ServerHTTPError(500, f"{url}", "boom")

    monkeypatch.setattr("paramem.cli.http_client.post_json", fake_post)

    def _boom(*a, **k):
        raise AssertionError("must not fall back to standalone on a non-404 HTTP error")

    monkeypatch.setattr("paramem.backup.runner.run_scheduled_backup", _boom)

    rc = backup_main.main(["--config", str(cfg_path), "--tier", "daily"])
    assert rc == 1


def test_missing_config_returns_1(tmp_path):
    """Nonexistent config path → exit 1 before any work."""
    rc = backup_main.main(["--config", str(tmp_path / "nope.yaml")])
    assert rc == 1


# ---------------------------------------------------------------------------
# http_client.post_json — token / Authorization header
# ---------------------------------------------------------------------------


class TestPostJsonToken:
    """Unit tests for the token parameter added to http_client.post_json."""

    def _make_mock_response(self, status_code=200, json_body=None):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_body or {"ok": True}
        resp.text = ""
        return resp

    def test_token_sets_authorization_header(self):
        """post_json(token='abc') → Authorization: Bearer abc sent to httpx."""
        captured_headers = {}

        def fake_post(url, *, json=None, headers=None):
            captured_headers.update(headers or {})
            return self._make_mock_response()

        mock_client = MagicMock()
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post = fake_post

        with patch("paramem.cli.http_client.httpx.Client", return_value=mock_client):
            http_client.post_json("http://127.0.0.1:8420/test", token="my-token")

        assert captured_headers.get("Authorization") == "Bearer my-token"

    def test_no_token_sends_no_authorization_header(self):
        """post_json() with no token → no Authorization header."""
        captured_headers = {}

        def fake_post(url, *, json=None, headers=None):
            captured_headers.update(headers or {})
            return self._make_mock_response()

        mock_client = MagicMock()
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post = fake_post

        with patch("paramem.cli.http_client.httpx.Client", return_value=mock_client):
            http_client.post_json("http://127.0.0.1:8420/test")

        assert "Authorization" not in captured_headers

    def test_token_none_explicit_sends_no_header(self):
        """post_json(token=None) → no Authorization header (backward-compatible)."""
        captured_headers = {}

        def fake_post(url, *, json=None, headers=None):
            captured_headers.update(headers or {})
            return self._make_mock_response()

        mock_client = MagicMock()
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post = fake_post

        with patch("paramem.cli.http_client.httpx.Client", return_value=mock_client):
            http_client.post_json("http://127.0.0.1:8420/test", token=None)

        assert "Authorization" not in captured_headers


def test_token_passed_to_post_json_when_env_set(config_file, monkeypatch):
    """PARAMEM_API_TOKEN in env → token forwarded to post_json as Authorization bearer."""
    cfg_path, _ = config_file
    monkeypatch.setenv("PARAMEM_API_TOKEN", "test-secret-token")
    calls = {}

    def fake_post(url, body, *, timeout, token=None):
        calls["token"] = token
        return {
            "success": True,
            "tier": "daily",
            "written_slots": {},
            "skipped_artifacts": [],
            "error": None,
        }

    monkeypatch.setattr("paramem.cli.http_client.post_json", fake_post)
    monkeypatch.setattr("paramem.backup.runner.run_scheduled_backup", lambda **k: None)

    backup_main.main(["--config", str(cfg_path), "--tier", "daily"])
    assert calls.get("token") == "test-secret-token"


def test_token_is_none_when_env_unset(config_file, monkeypatch):
    """PARAMEM_API_TOKEN absent → token=None (unauthenticated, preserves auth-OFF behaviour)."""
    cfg_path, _ = config_file
    monkeypatch.delenv("PARAMEM_API_TOKEN", raising=False)
    calls = {}

    def fake_post(url, body, *, timeout, token=None):
        calls["token"] = token
        return {
            "success": True,
            "tier": "daily",
            "written_slots": {},
            "skipped_artifacts": [],
            "error": None,
        }

    monkeypatch.setattr("paramem.cli.http_client.post_json", fake_post)
    monkeypatch.setattr("paramem.backup.runner.run_scheduled_backup", lambda **k: None)

    backup_main.main(["--config", str(cfg_path), "--tier", "daily"])
    assert calls.get("token") is None
