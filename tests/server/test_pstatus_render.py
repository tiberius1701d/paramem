"""Tests for paramem-status.sh Attention block + Migrate footer rendering.

Strategy: spawn the bash script via subprocess.run with PARAMEM_SERVER_PORT
overridden to point at a temporary HTTP server serving a hand-crafted
/status JSON. This exercises the full bash parsing + rendering path without
touching a real ParaMem server.

The temporary HTTP server is a minimal Python http.server running in a
background thread on an ephemeral port.

All tests assert on rendered text (stdout) because the script exits 0 when
the server responds; the rendered content is what matters.
"""

from __future__ import annotations

import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Path to the script under test.
_SCRIPT = Path(__file__).parents[2] / "scripts" / "dev" / "paramem-status.sh"

# Minimal /status JSON that satisfies all parser fields with safe defaults.
_BASE_STATUS: dict = {
    "mode": "local",
    "cloud_only_reason": None,
    "model": "mistral",
    "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
    "model_device": "cuda",
    "episodic_rank": 8,
    "adapter_loaded": True,
    "adapter_config": {"episodic": 1},
    "active_adapter": "episodic",
    "keys_count": 10,
    "pending_sessions": 0,
    "consolidating": False,
    "last_consolidation": "2026-04-22T04:00:00+00:00",
    "last_consolidation_result": None,
    "refresh_cadence": "",
    "consolidation_period": "",
    "max_interim_count": 0,
    "mode_config": "train",
    "next_run_seconds": None,
    "next_interim_seconds": None,
    "scheduler_started": False,
    "orphaned_pending": 0,
    "oldest_pending_seconds": None,
    "speaker_profiles": 0,
    "pending_enrollments": 0,
    "adapter_specs": {},
    "speakers": [],
    "adapter_health": {},
    "speaker_embedding_backend": None,
    "speaker_embedding_model": None,
    "speaker_embedding_device": None,
    "stt_loaded": False,
    "stt_engine": None,
    "stt_model": None,
    "stt_device": None,
    "tts_loaded": False,
    "tts_engine": None,
    "tts_languages": [],
    "tts_device": None,
    "bg_trainer_active": False,
    "bg_trainer_adapter": None,
    "thermal_policy": {
        "mode": "always_off",
        "start": "00:00",
        "end": "00:00",
        "temp_limit": 0,
        "currently_throttling": False,
    },
    "config_drift": {
        "detected": False,
        "loaded_hash": "a1b2c3d4e5f6a7b8",
        "disk_hash": "a1b2c3d4e5f6a7b8",
        "last_checked_at": "2026-04-22T08:00:00+00:00",
    },
    "attention": {"items": []},
    "migration": {
        "state": "live",
        "config_rev": "a1b2c3d4",
        "trial_started_at": None,
        "gates": None,
        "comparison": None,
    },
    "server_started_at": "2026-04-22T08:00:00+00:00",
    "adapter_manifest": {},
    "encryption": "on",
}


# ---------------------------------------------------------------------------
# Temporary HTTP server
# ---------------------------------------------------------------------------


class _StatusHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that serves a fixed JSON at GET /status."""

    status_json: bytes = b"{}"

    def do_GET(self):  # noqa: N802
        body = self.__class__.status_json
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # silence request logs in test output
        pass


def _run_pstatus(status_dict: dict) -> subprocess.CompletedProcess:
    """Spin up a one-request HTTP server and run the pstatus script against it.

    Returns the CompletedProcess with stdout captured.
    """
    body = json.dumps(status_dict).encode()
    _StatusHandler.status_json = body

    server = HTTPServer(("127.0.0.1", 0), _StatusHandler)
    port = server.server_address[1]

    # The server must handle at least one request per pstatus call
    # (one GET /status).  We run it in a daemon thread so it dies with the
    # test process if something goes wrong.
    t = threading.Thread(target=lambda: server.handle_request(), daemon=True)
    t.start()

    # Run bash without a live PID check — pstatus checks `lsof -i :PORT -t`
    # which would return nothing for our test server.  Inject a fake lsof via
    # PATH substitution so the PID check sees a non-empty result and does not
    # take the "NOT RUNNING" early-exit branch.
    import os
    import stat
    import tempfile

    with tempfile.TemporaryDirectory() as fake_bin:
        fake_lsof = Path(fake_bin) / "lsof"
        fake_lsof.write_text("#!/bin/sh\necho $$\n")
        fake_lsof.chmod(fake_lsof.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)

        env = os.environ.copy()
        env["PATH"] = f"{fake_bin}:{env.get('PATH', '/usr/bin:/bin')}"
        env["PARAMEM_SERVER_PORT"] = str(port)

        result = subprocess.run(
            ["bash", str(_SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=15,
        )

    t.join(timeout=5)
    server.server_close()
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAttentionBlockOmitted:
    def test_block_omitted_when_no_items(self):
        """attention.items=[] → output does NOT contain 'ATTENTION'."""
        status = dict(_BASE_STATUS)
        status["attention"] = {"items": []}
        result = _run_pstatus(status)
        assert "ATTENTION" not in result.stdout

    def test_migrate_footer_rendered_even_without_attention(self):
        """Migrate footer is rendered even when Attention block is omitted."""
        status = dict(_BASE_STATUS)
        status["attention"] = {"items": []}
        result = _run_pstatus(status)
        assert "Migrate:" in result.stdout


class TestAttentionBlockRendered:
    def _make_item(
        self,
        kind: str = "migration_trial_pass",
        level: str = "action_required",
        summary: str = "TRIAL active (gates PASS)",
        action_hint: str | None = "paramem migrate-accept",
        age_seconds: int | None = 2520,
    ) -> dict:
        return {
            "kind": kind,
            "level": level,
            "summary": summary,
            "action_hint": action_hint,
            "age_seconds": age_seconds,
        }

    def test_yellow_banner_when_action_required(self):
        """items=[action_required] → output contains ATTENTION and ⚠."""
        status = dict(_BASE_STATUS)
        status["attention"] = {"items": [self._make_item(level="action_required")]}
        result = _run_pstatus(status)
        assert "ATTENTION" in result.stdout
        assert "⚠" in result.stdout

    def test_red_banner_when_any_failed(self):
        """items includes failed item → output contains ✗."""
        status = dict(_BASE_STATUS)
        status["attention"] = {
            "items": [
                self._make_item(level="action_required"),
                self._make_item(
                    kind="adapter_fingerprint_mismatch_primary",
                    level="failed",
                    summary="FINGERPRINT MISMATCH (episodic)",
                    action_hint=None,
                    age_seconds=None,
                ),
            ]
        }
        result = _run_pstatus(status)
        assert "✗" in result.stdout
        assert "ATTENTION" in result.stdout

    def test_action_hint_rendered_with_arrow(self):
        """item with action_hint → output contains '→ paramem migrate-accept'."""
        status = dict(_BASE_STATUS)
        status["attention"] = {"items": [self._make_item(action_hint="paramem migrate-accept")]}
        result = _run_pstatus(status)
        assert "→ paramem migrate-accept" in result.stdout

    def test_age_rendered_when_present(self):
        """item with age_seconds=2520 → output contains '(age 42m'."""
        status = dict(_BASE_STATUS)
        status["attention"] = {"items": [self._make_item(age_seconds=2520)]}
        result = _run_pstatus(status)
        assert "(age 42m" in result.stdout

    def test_no_age_tag_when_age_none(self):
        """item with age_seconds=None → output does NOT contain '(age '."""
        status = dict(_BASE_STATUS)
        status["attention"] = {"items": [self._make_item(age_seconds=None)]}
        result = _run_pstatus(status)
        # The summary line should not contain "(age "
        lines_with_summary = [line for line in result.stdout.splitlines() if "TRIAL active" in line]
        assert lines_with_summary
        for line in lines_with_summary:
            assert "(age " not in line

    def test_kind_to_label_map(self):
        """Different kinds render with their expected labels."""
        label_expectations = [
            ("migration_trial_pass", "Migration:"),
            ("consolidation_blocked", "Consol:"),
            ("sweeper_held", "Sweeper:"),
            ("config_drift", "Config:"),
            ("adapter_fingerprint_mismatch_primary", "Adapter:"),
        ]
        for kind, expected_label in label_expectations:
            status = dict(_BASE_STATUS)
            status["attention"] = {
                "items": [
                    self._make_item(kind=kind, level="info", action_hint=None, age_seconds=None)
                ]
            }
            result = _run_pstatus(status)
            assert expected_label in result.stdout, (
                f"Expected label '{expected_label}' for kind '{kind}' "
                f"not found in stdout:\n{result.stdout[:500]}"
            )


class TestMigrateFooter:
    def test_migrate_footer_rendered_in_live(self):
        """migration.state=live, config_rev=a1b2c3d4 → 'Migrate:' and 'config rev a1b2c3d4'."""
        status = dict(_BASE_STATUS)
        status["migration"] = {
            "state": "live",
            "config_rev": "a1b2c3d4",
            "trial_started_at": None,
            "gates": None,
            "comparison": None,
        }
        result = _run_pstatus(status)
        assert "Migrate:" in result.stdout
        assert "config rev" in result.stdout
        assert "a1b2c3d4" in result.stdout

    def test_migrate_footer_color_trial(self):
        """migration.state=trial → output contains 'TRIAL'."""
        status = dict(_BASE_STATUS)
        status["migration"]["state"] = "trial"
        result = _run_pstatus(status)
        assert "TRIAL" in result.stdout

    def test_migrate_footer_color_failed(self):
        """migration.state=failed → output contains 'FAILED'."""
        status = dict(_BASE_STATUS)
        status["migration"]["state"] = "failed"
        result = _run_pstatus(status)
        assert "FAILED" in result.stdout

    def test_migrate_footer_applied_date(self):
        """server_started_at on /status → applied YYYY-MM-DD in footer."""
        status = dict(_BASE_STATUS)
        status["server_started_at"] = "2026-04-18T06:00:00+00:00"
        result = _run_pstatus(status)
        assert "2026-04-18" in result.stdout

    def test_status_response_schema_unchanged_for_existing_consumers(self):
        """Pre-Slice-5a JSON (no attention/migration keys) → script does not crash."""
        status = {
            k: v
            for k, v in _BASE_STATUS.items()
            if k not in ("attention", "migration", "server_started_at")
        }
        result = _run_pstatus(status)
        # Script must exit 0 and render something without crashing.
        assert result.returncode == 0
        assert "ParaMem Server" in result.stdout


class TestSecurityFooter:
    """The pstatus renderer surfaces the startup security posture."""

    def test_security_on_rendered(self):
        """encryption='on' → 'Security: ON' with the age-daily annotation."""
        status = dict(_BASE_STATUS)
        status["encryption"] = "on"
        result = _run_pstatus(status)
        assert "Security:" in result.stdout
        assert "ON" in result.stdout
        assert "age daily" in result.stdout

    def test_security_off_rendered(self):
        """encryption='off' → 'Security: OFF' with the plaintext warning."""
        status = dict(_BASE_STATUS)
        status["encryption"] = "off"
        result = _run_pstatus(status)
        assert "Security:" in result.stdout
        assert "OFF" in result.stdout
        assert "plaintext" in result.stdout

    def test_security_field_absent_renders_placeholder(self):
        """Legacy /status JSON without encryption → dim placeholder, no crash."""
        status = {k: v for k, v in _BASE_STATUS.items() if k != "encryption"}
        result = _run_pstatus(status)
        assert result.returncode == 0
        assert "Security:" in result.stdout
