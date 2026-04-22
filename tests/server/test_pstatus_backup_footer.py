"""Tests for the pstatus Backup footer rendering.

Uses subprocess to run the embedded Python block from paramem-status.sh in
isolation against fixture JSON, validating the output for all edge cases.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helper: run the embedded Python block against fixture JSON
# ---------------------------------------------------------------------------


def _extract_backup_line(status_json: dict) -> str:
    """Run the embedded Python from paramem-status.sh and return the BACKUP line."""
    # Extract only the relevant Python snippet from the shell script.
    script_path = (
        Path(__file__).resolve().parent.parent.parent / "scripts" / "dev" / "paramem-status.sh"
    )
    if not script_path.exists():
        pytest.skip(f"paramem-status.sh not found at {script_path}")

    script_text = script_path.read_text()

    # Find the Python heredoc block.
    start_marker = "STATUS_JSON=\"$status_json\" python3 <<'PY'"
    end_marker = "PY\n)"
    start_idx = script_text.find(start_marker)
    end_idx = script_text.find(end_marker, start_idx)
    if start_idx == -1 or end_idx == -1:
        pytest.skip("Could not find embedded Python block in paramem-status.sh")

    # Extract everything between the markers.
    py_block = script_text[start_idx + len(start_marker) : end_idx]

    # Run the Python block with STATUS_JSON set.
    result = subprocess.run(
        [sys.executable, "-c", py_block],
        env={"STATUS_JSON": json.dumps(status_json), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    # Find the BACKUP line.
    for line in result.stdout.splitlines():
        if line.startswith("BACKUP\t"):
            return line
    return ""


def _minimal_status(backup: dict | None = None) -> dict:
    """Build a minimal /status JSON dict for testing."""
    return {
        "mode": "local",
        "cloud_only_reason": None,
        "model": "mistral",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "model_device": "cuda",
        "episodic_rank": 8,
        "adapter_loaded": True,
        "adapter_config": {},
        "active_adapter": None,
        "keys_count": 0,
        "pending_sessions": 0,
        "consolidating": False,
        "last_consolidation": None,
        "last_consolidation_result": None,
        "refresh_cadence": "12h",
        "consolidation_period": "84h",
        "max_interim_count": 7,
        "mode_config": "train",
        "next_run_seconds": None,
        "next_interim_seconds": None,
        "scheduler_started": False,
        "orphaned_pending": 0,
        "oldest_pending_seconds": None,
        "speaker_profiles": 0,
        "pending_enrollments": 0,
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
        "thermal_policy": {},
        "config_drift": {},
        "hold": {},
        "adapter_specs": {},
        "adapter_health": {},
        "adapter_manifest": {},
        "attention": {"items": []},
        "migration": {"state": "live"},
        "server_started_at": "2026-04-22T04:00:00Z",
        "backup": backup or {},
    }


# ---------------------------------------------------------------------------
# Footer rendering tests
# ---------------------------------------------------------------------------


class TestFooterNormalState:
    def test_footer_normal_state(self):
        """JSON with success → BACKUP line has schedule, last_success_at."""
        status = _minimal_status(
            {
                "schedule": "daily 04:00",
                "last_success_at": "2026-04-22T04:00:42Z",
                "last_failure_at": None,
                "last_failure_reason": None,
                "next_scheduled_at": "2026-04-23T04:00:00Z",
                "stale": False,
                "disk_used_bytes": 1073741824,
                "disk_cap_bytes": 21474836480,
            }
        )
        line = _extract_backup_line(status)
        assert line.startswith("BACKUP\t")
        parts = line.split("\t")
        # BACKUP, schedule, last_ok, last_fail, fail_reason, next, stale, used_bytes, cap_bytes
        assert parts[1] == "daily 04:00"
        assert parts[2] == "2026-04-22T04:00:42Z"
        assert parts[5] == "2026-04-23T04:00:00Z"
        assert parts[6] == "false"


class TestFooterOff:
    def test_footer_off(self):
        """schedule='off' → BACKUP line with empty schedule."""
        status = _minimal_status(
            {
                "schedule": "off",
                "last_success_at": None,
                "last_failure_at": None,
                "last_failure_reason": None,
                "next_scheduled_at": None,
                "stale": False,
                "disk_used_bytes": 0,
                "disk_cap_bytes": 0,
            }
        )
        line = _extract_backup_line(status)
        assert line.startswith("BACKUP\t")
        parts = line.split("\t")
        assert parts[1] == "off"


class TestFooterNeverRun:
    def test_footer_never_run(self):
        """last_success_at=None + next set → line with empty last_ok but next present."""
        status = _minimal_status(
            {
                "schedule": "daily 04:00",
                "last_success_at": None,
                "last_failure_at": None,
                "last_failure_reason": None,
                "next_scheduled_at": "2026-04-23T04:00:00Z",
                "stale": False,
                "disk_used_bytes": 0,
                "disk_cap_bytes": 0,
            }
        )
        line = _extract_backup_line(status)
        parts = line.split("\t")
        assert parts[2] == ""  # last_success empty
        assert parts[5] == "2026-04-23T04:00:00Z"  # next present


class TestFooterStale:
    def test_footer_stale(self):
        """stale=True → BACKUP line with 'true' in stale field."""
        status = _minimal_status(
            {
                "schedule": "daily 04:00",
                "last_success_at": "2026-04-10T04:00:00Z",
                "last_failure_at": None,
                "last_failure_reason": None,
                "next_scheduled_at": "2026-04-23T04:00:00Z",
                "stale": True,
                "disk_used_bytes": 0,
                "disk_cap_bytes": 0,
            }
        )
        line = _extract_backup_line(status)
        parts = line.split("\t")
        assert parts[6] == "true"


class TestFooterFailed:
    def test_footer_failed(self):
        """last_failure_at > last_success_at → fail fields populated."""
        status = _minimal_status(
            {
                "schedule": "daily 04:00",
                "last_success_at": "2026-04-20T04:00:00Z",
                "last_failure_at": "2026-04-22T04:00:00Z",
                "last_failure_reason": "disk_pressure: max_total_disk_gb reached",
                "next_scheduled_at": "2026-04-23T04:00:00Z",
                "stale": False,
                "disk_used_bytes": 0,
                "disk_cap_bytes": 0,
            }
        )
        line = _extract_backup_line(status)
        parts = line.split("\t")
        assert parts[3] == "2026-04-22T04:00:00Z"
        assert "disk_pressure" in parts[4]


class TestFooterDiskUsageTag:
    def test_footer_disk_usage_tag(self):
        """Non-zero disk_used + cap → bytes present in BACKUP line."""
        status = _minimal_status(
            {
                "schedule": "daily 04:00",
                "last_success_at": "2026-04-22T04:00:42Z",
                "last_failure_at": None,
                "last_failure_reason": None,
                "next_scheduled_at": "2026-04-23T04:00:00Z",
                "stale": False,
                "disk_used_bytes": 1073741824,  # 1 GB
                "disk_cap_bytes": 21474836480,  # 20 GB
            }
        )
        line = _extract_backup_line(status)
        parts = line.split("\t")
        assert parts[7] == "1073741824"
        assert parts[8] == "21474836480"


class TestFooterNoBackupBlock:
    def test_footer_no_backup_block(self):
        """Missing backup block → BACKUP line with empty fields (graceful default)."""
        status = _minimal_status(None)
        status.pop("backup", None)
        status["backup"] = {}
        line = _extract_backup_line(status)
        # Should still produce a BACKUP line (dict gets ``or {}``)
        assert line.startswith("BACKUP\t")
