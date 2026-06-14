"""Tests for paramem.cli.integrity (``paramem integrity``).

Covers:
- Happy path: ok=True → human-readable output, exit 0.
- Integrity failure: ok=False → human-readable output with failures, exit 1.
- --json mode: ok response emits raw JSON + exit 0.
- --json mode: not-ok response emits raw JSON + exit 1.
- ServerUnavailable (404) → exit 1, message on stderr.
- ServerUnreachable → exit 2, message on stderr.
- ServerHTTPError → exit 1, message on stderr.
- Zero checks (empty store) → summary line on stdout.
- main() dispatch: ``paramem integrity`` routes to integrity.run().
"""

from __future__ import annotations

import json

from paramem.cli import http_client
from paramem.cli.main import main

# ---------------------------------------------------------------------------
# Fake responses
# ---------------------------------------------------------------------------

_OK_RESPONSE = {
    "ok": True,
    "checks": [
        {
            "path": "/data/ha/adapters/episodic/indexed_key_registry.json",
            "category": "registry",
            "tier": "episodic",
            "status": "ok",
            "detail": "",
        },
        {
            "path": "/data/ha/adapters/episodic/indexed_key_registry.json",
            "category": "simhash",
            "tier": "episodic",
            "status": "ok",
            "detail": "",
        },
    ],
    "failures": [],
}

_FAIL_RESPONSE = {
    "ok": False,
    "checks": [
        {
            "path": "/data/ha/adapters/episodic/indexed_key_registry.json",
            "category": "registry",
            "tier": "episodic",
            "status": "parse_error",
            "detail": "Expecting property name: line 1 col 2 (char 1)",
        },
    ],
    "failures": [
        {
            "path": "/data/ha/adapters/episodic/indexed_key_registry.json",
            "category": "registry",
            "tier": "episodic",
            "status": "parse_error",
            "detail": "Expecting property name: line 1 col 2 (char 1)",
        },
    ],
}

_EMPTY_RESPONSE = {
    "ok": True,
    "checks": [],
    "failures": [],
}


# ---------------------------------------------------------------------------
# Human-readable output
# ---------------------------------------------------------------------------


class TestIntegrityHumanReadable:
    def test_ok_response_exit_0(self, monkeypatch, capsys):
        """ok=True → exit 0."""
        monkeypatch.setattr(http_client, "get_json", lambda *a, **kw: _OK_RESPONSE)
        rc = main(["integrity"])
        assert rc == 0

    def test_ok_response_prints_checks(self, monkeypatch, capsys):
        """ok=True → per-file status lines on stdout."""
        monkeypatch.setattr(http_client, "get_json", lambda *a, **kw: _OK_RESPONSE)
        main(["integrity"])
        captured = capsys.readouterr()
        assert "episodic" in captured.out
        # At least one [ok] line must appear
        assert "ok" in captured.out.lower()

    def test_ok_response_prints_summary_line(self, monkeypatch, capsys):
        """ok=True → summary line indicating success."""
        monkeypatch.setattr(http_client, "get_json", lambda *a, **kw: _OK_RESPONSE)
        main(["integrity"])
        captured = capsys.readouterr()
        assert "OK" in captured.out or "ok" in captured.out.lower()

    def test_fail_response_exit_1(self, monkeypatch, capsys):
        """ok=False → exit 1."""
        monkeypatch.setattr(http_client, "get_json", lambda *a, **kw: _FAIL_RESPONSE)
        rc = main(["integrity"])
        assert rc == 1

    def test_fail_response_prints_failures(self, monkeypatch, capsys):
        """ok=False → failure paths appear in output."""
        monkeypatch.setattr(http_client, "get_json", lambda *a, **kw: _FAIL_RESPONSE)
        main(["integrity"])
        captured = capsys.readouterr()
        assert "parse_error" in captured.out or "FAIL" in captured.out

    def test_empty_checks_ok_exit_0(self, monkeypatch, capsys):
        """Empty checks list with ok=True → exit 0."""
        monkeypatch.setattr(http_client, "get_json", lambda *a, **kw: _EMPTY_RESPONSE)
        rc = main(["integrity"])
        assert rc == 0

    def test_empty_checks_prints_summary(self, monkeypatch, capsys):
        """Empty checks list → summary line with 0 checks."""
        monkeypatch.setattr(http_client, "get_json", lambda *a, **kw: _EMPTY_RESPONSE)
        main(["integrity"])
        captured = capsys.readouterr()
        assert "0" in captured.out


# ---------------------------------------------------------------------------
# --json mode
# ---------------------------------------------------------------------------


class TestIntegrityJsonMode:
    def test_json_ok_emits_valid_json(self, monkeypatch, capsys):
        """--json + ok=True → valid JSON on stdout."""
        monkeypatch.setattr(http_client, "get_json", lambda *a, **kw: _OK_RESPONSE)
        rc = main(["integrity", "--json"])
        captured = capsys.readouterr()
        assert rc == 0
        parsed = json.loads(captured.out)
        assert parsed["ok"] is True

    def test_json_ok_exit_0(self, monkeypatch, capsys):
        """--json + ok=True → exit 0."""
        monkeypatch.setattr(http_client, "get_json", lambda *a, **kw: _OK_RESPONSE)
        rc = main(["integrity", "--json"])
        assert rc == 0

    def test_json_fail_emits_valid_json(self, monkeypatch, capsys):
        """--json + ok=False → valid JSON on stdout."""
        monkeypatch.setattr(http_client, "get_json", lambda *a, **kw: _FAIL_RESPONSE)
        rc = main(["integrity", "--json"])
        captured = capsys.readouterr()
        assert rc == 1
        parsed = json.loads(captured.out)
        assert parsed["ok"] is False

    def test_json_fail_exit_1(self, monkeypatch, capsys):
        """--json + ok=False → exit 1."""
        monkeypatch.setattr(http_client, "get_json", lambda *a, **kw: _FAIL_RESPONSE)
        rc = main(["integrity", "--json"])
        assert rc == 1

    def test_json_output_contains_checks_key(self, monkeypatch, capsys):
        """--json output has 'checks' and 'failures' keys."""
        monkeypatch.setattr(http_client, "get_json", lambda *a, **kw: _OK_RESPONSE)
        main(["integrity", "--json"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "checks" in parsed
        assert "failures" in parsed


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestIntegrityErrorHandling:
    def test_server_unavailable_404_exit_1(self, monkeypatch, capsys):
        """ServerUnavailable (404) → exit 1, message on stderr."""

        def _raise(*a, **kw):
            raise http_client.ServerUnavailable("404")

        monkeypatch.setattr(http_client, "get_json", _raise)
        rc = main(["integrity"])
        captured = capsys.readouterr()
        assert rc == 1
        assert "integrity" in captured.err.lower() or "404" in captured.err

    def test_server_unreachable_exit_2(self, monkeypatch, capsys):
        """ServerUnreachable → exit 2, message on stderr."""

        def _raise(*a, **kw):
            raise http_client.ServerUnreachable("connection refused")

        monkeypatch.setattr(http_client, "get_json", _raise)
        rc = main(["integrity"])
        captured = capsys.readouterr()
        assert rc == 2
        assert "unreachable" in captured.err.lower() or "running" in captured.err.lower()

    def test_http_error_exit_1(self, monkeypatch, capsys):
        """ServerHTTPError → exit 1, status code on stderr."""

        def _raise(*a, **kw):
            raise http_client.ServerHTTPError(500, "/integrity", "internal server error")

        monkeypatch.setattr(http_client, "get_json", _raise)
        rc = main(["integrity"])
        captured = capsys.readouterr()
        assert rc == 1
        assert "500" in captured.err or "HTTP" in captured.err

    def test_server_unavailable_message_mentions_version(self, monkeypatch, capsys):
        """404 error hint mentions version alignment."""

        def _raise(*a, **kw):
            raise http_client.ServerUnavailable("404")

        monkeypatch.setattr(http_client, "get_json", _raise)
        main(["integrity"])
        captured = capsys.readouterr()
        assert "version" in captured.err.lower() or "aligned" in captured.err.lower()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


class TestIntegrityDispatch:
    def test_main_dispatches_integrity_command(self, monkeypatch, capsys):
        """``paramem integrity`` dispatches to integrity.run()."""
        calls = []

        def _fake_get_json(url):
            calls.append(url)
            return _OK_RESPONSE

        monkeypatch.setattr(http_client, "get_json", _fake_get_json)
        rc = main(["integrity"])
        assert rc == 0
        assert len(calls) == 1
        assert calls[0].endswith("/integrity")

    def test_custom_server_url(self, monkeypatch, capsys):
        """``--server-url`` is forwarded to the GET /integrity request."""
        calls = []

        def _fake_get_json(url):
            calls.append(url)
            return _OK_RESPONSE

        monkeypatch.setattr(http_client, "get_json", _fake_get_json)
        main(["--server-url", "http://10.0.0.5:8420", "integrity"])
        assert calls[0] == "http://10.0.0.5:8420/integrity"
