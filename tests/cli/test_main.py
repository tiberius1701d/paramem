"""Tests for paramem.cli.main — argparse dispatch, 404 degradation, error handling.

Mocks httpx at the ServerUnavailable / ServerUnreachable boundary so no live
server is needed.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from paramem.cli import http_client
from paramem.cli.main import main


class TestMigrateUnreachable:
    def test_migrate_unreachable_prints_message_and_returns_2(self, monkeypatch, capsys):
        """ServerUnreachable from post_json → rc=2, error message on stderr."""

        def _raise(*_args, **_kwargs):
            raise http_client.ServerUnreachable("connection refused")

        monkeypatch.setattr(http_client, "post_json", _raise)

        rc = main(["migrate", "/abs/path.yaml"])
        captured = capsys.readouterr()

        assert rc == 2, f"Expected exit 2, got {rc}"
        assert "unreachable" in captured.err.lower(), (
            f"Expected 'unreachable' in stderr, got: {captured.err!r}"
        )
        assert "systemctl" in captured.err, (
            "Error message should mention systemctl for troubleshooting"
        )


class TestMigrate404PrintsSlice3Message:
    def test_migrate_404_prints_slice3_message_and_returns_1(self, monkeypatch, capsys):
        """ServerUnavailable from post_json → rc=1, message mentions Slice 3."""

        def _raise(*_args, **_kwargs):
            raise http_client.ServerUnavailable("404 from /migration/preview")

        monkeypatch.setattr(http_client, "post_json", _raise)

        rc = main(["migrate", "/abs/path.yaml"])
        captured = capsys.readouterr()

        assert rc == 1, f"Expected exit 1, got {rc}"
        assert "/migration/preview" in captured.err, "Stderr must mention the endpoint"
        assert "Slice 3" in captured.err, (
            "Stderr must mention Slice 3 so the operator knows when it ships"
        )


class TestMigrateRequiresAbsolutePath:
    def test_migrate_requires_absolute_path(self, capsys):
        """Relative path argument → rc != 0, stderr explains requirement."""
        rc = main(["migrate", "relative/path.yaml"])
        captured = capsys.readouterr()

        assert rc != 0, "Relative path must be rejected"
        assert "absolute" in captured.err.lower(), (
            f"Stderr must mention 'absolute', got: {captured.err!r}"
        )


class TestMigrateStatusSuccessPath:
    def test_migrate_status_success_path_renders_response(self, monkeypatch, capsys):
        """Successful GET /migration/status renders key: value lines."""
        fake_response = {"state": "idle", "trial_id": None, "started_at": None}

        monkeypatch.setattr(http_client, "get_json", lambda *a, **kw: fake_response)

        rc = main(["migrate-status"])
        captured = capsys.readouterr()

        assert rc == 0, f"Expected exit 0, got {rc}"
        assert "state" in captured.out, "Output must contain the 'state' key"
        assert "idle" in captured.out, "Output must contain the state value"


class TestMigrateStatusJsonFlag:
    def test_migrate_status_json_flag_emits_raw_json(self, monkeypatch, capsys):
        """With --json the response is dumped as valid JSON to stdout."""
        fake_response = {"state": "idle", "trial_id": "t1"}

        monkeypatch.setattr(http_client, "get_json", lambda *a, **kw: fake_response)

        rc = main(["migrate-status", "--json"])
        captured = capsys.readouterr()

        assert rc == 0, f"Expected exit 0, got {rc}"
        parsed = json.loads(captured.out)
        assert parsed == fake_response, "JSON output must round-trip the response"


class TestBackupList404PrintsSlice6Message:
    def test_backup_list_404_prints_slice6_message(self, monkeypatch, capsys):
        """ServerUnavailable from /backup/list → rc=1, message mentions Slice 6."""

        def _raise(*_args, **_kwargs):
            raise http_client.ServerUnavailable("404 from /backup/list")

        monkeypatch.setattr(http_client, "get_json", _raise)

        rc = main(["backup", "list"])
        captured = capsys.readouterr()

        assert rc == 1, f"Expected exit 1, got {rc}"
        assert "Slice 6" in captured.err, "Stderr must mention Slice 6 for backup list"


class TestParamemHelp:
    def test_paramem_help(self, capsys):
        """``main(["--help"])`` exits 0 (argparse SystemExit(0))."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0, "--help must exit 0"


class TestParamemVersion:
    def test_paramem_version_flag_prints_version_and_exits_0(self, capsys):
        """``main(["--version"])`` prints ``paramem <ver>`` and exits 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0, "--version must exit 0"
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert combined.strip().startswith("paramem "), (
            f"--version output must begin with 'paramem ', got: {combined!r}"
        )


class TestMigrateHTTPError:
    def test_migrate_http_500_prints_body_and_returns_1(self, monkeypatch, capsys):
        """ServerHTTPError from post_json → rc=1, body and status code on stderr."""

        def _raise(*_args, **_kwargs):
            raise http_client.ServerHTTPError(500, "http://x/migration/preview", "boom")

        monkeypatch.setattr(http_client, "post_json", _raise)

        rc = main(["migrate", "/abs/path.yaml"])
        captured = capsys.readouterr()

        assert rc == 1, f"Expected exit 1, got {rc}"
        assert "500" in captured.err, "Stderr must include the status code"
        assert "boom" in captured.err, "Stderr must include the response body"


class TestBackupListHTTPError:
    def test_backup_list_http_500_prints_body_and_returns_1(self, monkeypatch, capsys):
        """ServerHTTPError from get_json on backup list → rc=1 with diagnostic body."""

        def _raise(*_args, **_kwargs):
            raise http_client.ServerHTTPError(503, "http://x/backup/list", "maintenance")

        monkeypatch.setattr(http_client, "get_json", _raise)

        rc = main(["backup", "list"])
        captured = capsys.readouterr()

        assert rc == 1, f"Expected exit 1, got {rc}"
        assert "503" in captured.err
        assert "maintenance" in captured.err


class TestParamemMAlias:
    def test_paramem_m_alias_equivalent(self):
        """``python -m paramem --help`` exits 0, confirming __main__.py is wired."""
        result = subprocess.run(
            [sys.executable, "-m", "paramem", "--help"],
            capture_output=True,
        )
        assert result.returncode == 0, (
            f"python -m paramem --help exited {result.returncode}.\n"
            f"stdout: {result.stdout.decode()}\n"
            f"stderr: {result.stderr.decode()}"
        )
