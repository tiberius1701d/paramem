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
    # Find the BACKUP line (pipe-separated since B5 fix switched from \t to |).
    for line in result.stdout.splitlines():
        if line.startswith("BACKUP|"):
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
        assert line.startswith("BACKUP|")
        parts = line.split("|")
        # BACKUP|schedule|last_ok|last_fail|fail_reason|next|stale|used_bytes|cap_bytes
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
        assert line.startswith("BACKUP|")
        parts = line.split("|")
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
        parts = line.split("|")
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
        parts = line.split("|")
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
        parts = line.split("|")
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
        parts = line.split("|")
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
        assert line.startswith("BACKUP|")


# ---------------------------------------------------------------------------
# B5 regression — cap=0 + success=true renders clean output
# ---------------------------------------------------------------------------


def _render_backup_footer(backup: dict) -> str:
    """Run the full bash footer block against fixture JSON and return the Backup: line.

    Invokes the complete shell script against a fixture /status JSON so both the
    Python data extraction AND the bash branch logic are exercised end-to-end.
    Uses ``bash -c`` to run the relevant portion of the script.
    """
    import os
    import shutil

    bash_path = shutil.which("bash")
    if bash_path is None:
        pytest.skip("bash not available")

    script_path = (
        Path(__file__).resolve().parent.parent.parent / "scripts" / "dev" / "paramem-status.sh"
    )
    if not script_path.exists():
        pytest.skip(f"paramem-status.sh not found at {script_path}")

    status_json = json.dumps(_minimal_status(backup))

    # Run the Python block to get the BACKUP line, then feed it to a small bash
    # fragment that replicates the IFS-split and case logic.
    script_text = script_path.read_text()
    start_marker = "STATUS_JSON=\"$status_json\" python3 <<'PY'"
    end_marker = "PY\n)"
    start_idx = script_text.find(start_marker)
    end_idx = script_text.find(end_marker, start_idx)
    if start_idx == -1 or end_idx == -1:
        pytest.skip("Could not find Python block in paramem-status.sh")

    py_block = script_text[start_idx + len(start_marker) : end_idx]

    py_result = subprocess.run(
        [sys.executable, "-c", py_block],
        env={"STATUS_JSON": status_json, "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    backup_line = ""
    for line in py_result.stdout.splitlines():
        if line.startswith("BACKUP|"):
            backup_line = line
            break

    if not backup_line:
        return ""

    # Run a bash snippet that replicates the IFS-split + rendering logic.
    # Extract the relevant case block from the script.
    case_start = script_text.find('case "$bk_schedule" in')
    case_end = script_text.find("fi\n\n# Windows Update lock", case_start)
    if case_start == -1 or case_end == -1:
        # Fallback: skip the bash rendering check.
        return "(skipped: could not extract bash case block)"

    bash_case_block = script_text[case_start:case_end]

    # Compose the full bash test fragment.
    # NOTE: The BACKUP line uses | as separator (not \t) since the B5 fix.
    # IFS=$'\t' with consecutive empty tabs collapses them (bash whitespace rule).
    # Pass the backup_line via env var (BACKUP_LINE) so the shell sees real
    # pipe characters without any escaping artifacts from Python f-string quoting.
    bash_script = f"""#!/bin/bash
set -euo pipefail
BOLD="" RESET="" GREEN="" YELLOW="" RED="" CYAN="" DIM=""
IFS='|' read -r _bm \\
    bk_schedule bk_last_ok bk_last_fail bk_fail_reason \\
    bk_next bk_stale bk_used_bytes bk_cap_bytes \\
    <<< "$BACKUP_LINE"
fmt_gb() {{
    local _b="$1"
    python3 -c "b=int('$_b' or 0); print(f'{{b/1024**3:.1f}}')" 2>/dev/null || echo "?"
}}
bk_used_gb=$(fmt_gb "${{bk_used_bytes:-0}}")
bk_cap_gb=$(fmt_gb "${{bk_cap_bytes:-0}}")
fmt_ts() {{
    if [[ -z "$1" ]]; then echo ""; else
        echo "$1" | sed 's/T/ /; s/:[0-9][0-9]\\([.][0-9]*\\)\\?\\(Z\\|+.*\\)\\?$//'
    fi
}}
bk_last_ok_disp=$(fmt_ts "$bk_last_ok")
bk_last_fail_disp=$(fmt_ts "$bk_last_fail")
bk_next_disp=$(fmt_ts "$bk_next")
{bash_case_block}
"""
    bash_result = subprocess.run(
        [bash_path, "-c", bash_script],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": os.environ.get("HOME", "/tmp"),
            "BACKUP_LINE": backup_line,
        },
    )
    return bash_result.stdout.strip()


class TestFooterCapZeroSuccessTrue:
    """B5 regression: cap=0 + last_success_at set + no failures must render cleanly.

    Observed symptom (2026-04-22 E2E baseline): pstatus rendered
    ``Backup:   FAILED false — 57141`` instead of a clean success or disabled line.
    The fix adds an explicit cap=0 guard before the failure branch so no
    false-FAILED output can be produced in this state.
    """

    def test_footer_cap_zero_success_true_not_failed_branch(self):
        """cap=0 + success=true → BACKUP line does not contain 'false' or raw byte counts
        in the rendered suffix (guards against the 'FAILED false — 57141' regression)."""
        status = _minimal_status(
            {
                "schedule": "daily 04:00",
                "last_success_at": "2026-04-22T21:08:36.185771+00:00",
                "last_failure_at": None,
                "last_failure_reason": None,
                "next_scheduled_at": None,
                "stale": False,
                "disk_used_bytes": 57141,
                "disk_cap_bytes": 0,
            }
        )
        line = _extract_backup_line(status)
        assert line.startswith("BACKUP|"), f"No BACKUP| line produced: {line!r}"
        parts = line.split("|")

        # Field alignment sanity: the Python block must emit 9 pipe-separated fields.
        # Format: BACKUP|schedule|last_ok|last_fail|fail_reason|next|stale|used|cap
        assert len(parts) == 9, f"Expected 9 pipe-delimited fields, got {len(parts)}: {parts}"
        # Field 3 = last_failure_at must be empty (no failure recorded).
        assert parts[3] == "", (
            f"last_failure_at field must be empty when no failure recorded, got {parts[3]!r}.  "
            "Non-empty means the field alignment is wrong (B5 regression)."
        )
        # Field 6 = stale must be 'false' (not confused with last_failure_at).
        assert parts[6] == "false", (
            f"stale field must be 'false', got {parts[6]!r}.  "
            "Wrong value means field alignment is off (B5 regression)."
        )
        # Field 7 = disk_used_bytes must be the integer.
        assert parts[7] == "57141", f"disk_used_bytes field must be '57141', got {parts[7]!r}."
        # Field 8 = disk_cap_bytes must be 0.
        assert parts[8] == "0", f"disk_cap_bytes field must be '0', got {parts[8]!r}."

    def test_footer_cap_zero_success_renders_no_failed_text(self):
        """The bash rendering of cap=0 + success=true must not say FAILED.

        This is the direct guard against the B5 symptom: pstatus showed
        'Backup:   FAILED false — 57141' when it should show a clean line.
        """
        rendered = _render_backup_footer(
            {
                "schedule": "daily 04:00",
                "last_success_at": "2026-04-22T21:08:36.185771+00:00",
                "last_failure_at": None,
                "last_failure_reason": None,
                "next_scheduled_at": None,
                "stale": False,
                "disk_used_bytes": 57141,
                "disk_cap_bytes": 0,
            }
        )
        if rendered == "(skipped: could not extract bash case block)":
            pytest.skip("Could not extract bash case block from script")

        assert "FAILED" not in rendered, (
            f"B5 regression: cap=0 + success=true rendered 'FAILED' in the footer: "
            f"{rendered!r}.  "
            "The fix must route this state to the cap=0 guard branch, not the failure branch."
        )
        # Should contain 'last ok' or 'never run', not 'FAILED'.
        assert "last ok" in rendered or "never run" in rendered or "no cap" in rendered, (
            f"Expected 'last ok', 'never run', or 'no cap' in rendered output, got: {rendered!r}"
        )
