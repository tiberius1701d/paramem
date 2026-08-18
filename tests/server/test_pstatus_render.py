"""Tests for paramem-status.sh rendering: Attention block, Migrate footer,
Security footer, and ``fmt_result``'s per-outcome Consol-line rendering.

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
import re
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
    # take the "NOT RUNNING" early-exit branch. Also shadow systemctl,
    # nvidia-smi, and powershell.exe with fakes so the script's read-only
    # host queries never reach the real host during a test run.
    import os
    import stat
    import tempfile

    with tempfile.TemporaryDirectory() as fake_bin:
        fake_lsof = Path(fake_bin) / "lsof"
        fake_lsof.write_text("#!/bin/sh\necho $$\n")

        fake_systemctl = Path(fake_bin) / "systemctl"
        fake_systemctl.write_text(
            "#!/bin/sh\n"
            'case "$2" in\n'
            "  is-active)\n"
            "    echo active\n"
            "    ;;\n"
            "  show)\n"
            "    echo 'ActiveEnterTimestamp=Mon 2026-08-10 08:00:00 UTC'\n"
            "    ;;\n"
            "esac\n"
            "exit 0\n"
        )

        fake_nvidia_smi = Path(fake_bin) / "nvidia-smi"
        fake_nvidia_smi.write_text(
            "#!/bin/sh\n"
            'for arg in "$@"; do\n'
            '  case "$arg" in\n'
            "    --query-gpu=temperature.gpu*) echo 45 ;;\n"
            "    --query-gpu=power.draw*) echo 15.00 ;;\n"
            "    --query-gpu=memory.used*) echo 1024 ;;\n"
            "    --query-gpu=memory.total*) echo 8192 ;;\n"
            "  esac\n"
            "done\n"
            "exit 0\n"
        )

        fake_powershell = Path(fake_bin) / "powershell.exe"
        fake_powershell.write_text("#!/bin/sh\nexit 0\n")

        for fake_exe in (fake_lsof, fake_systemctl, fake_nvidia_smi, fake_powershell):
            fake_exe.chmod(fake_exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)

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
# ANSI-aware line helpers
#
# The script colors every rendered field via `echo -e "\033[...m...\033[0m"`;
# `echo -e` interprets those escapes unconditionally (no TTY check), so the
# captured subprocess stdout carries real ESC bytes. Substring assertions
# are unaffected (the color codes wrap the text, never split it), but exact
# line comparisons need the codes stripped first.
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _line_with_prefix(stdout: str, prefix: str) -> str:
    """Return the single ANSI-stripped stdout line starting with ``prefix``.

    Fails loudly (with full stdout) on zero or multiple matches rather than
    silently picking the first — a duplicate or missing line is itself a
    finding.
    """
    matches = [
        stripped
        for stripped in (_strip_ansi(line) for line in stdout.splitlines())
        if stripped.startswith(prefix)
    ]
    assert len(matches) == 1, (
        f"expected exactly one line starting with {prefix!r}, found {len(matches)}:\n{stdout}"
    )
    return matches[0]


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

    def test_no_arrow_line_when_action_hint_none(self):
        """item with action_hint=None → the arrow line (paramem-status.sh:829-831,
        rendered only when ahint is non-empty) is absent immediately after
        that item's own summary line — scoped to the item, not the whole
        output, since another item elsewhere could legitimately carry one."""
        status = dict(_BASE_STATUS)
        status["attention"] = {
            "items": [
                self._make_item(summary="NO_HINT_ITEM_SUMMARY", action_hint=None, age_seconds=None)
            ]
        }
        result = _run_pstatus(status)
        lines = result.stdout.splitlines()
        summary_indices = [i for i, line in enumerate(lines) if "NO_HINT_ITEM_SUMMARY" in line]
        assert summary_indices
        for i in summary_indices:
            assert i + 1 < len(lines)
            assert "→" not in lines[i + 1]

    def test_age_present_hint_none_no_arrow_shift(self):
        """Regression: item with action_hint=None AND age_seconds=2520 — the
        shape that broke under IFS=$'\\t' reads. Bash `read` collapses
        consecutive tab delimiters when a field is empty, so the empty
        action_hint shifted age_seconds into the hint variable and rendered
        a bogus '→ 2520' arrow line instead of the '(age 42m)' tag with no
        arrow (paramem-status.sh:492-500 print, :807-831 read/render — both
        now use '|' delimiters, mirroring the BACKUP line at :515-531)."""
        status = dict(_BASE_STATUS)
        status["attention"] = {
            "items": [
                self._make_item(
                    summary="AGE_NO_HINT_ITEM_SUMMARY", action_hint=None, age_seconds=2520
                )
            ]
        }
        result = _run_pstatus(status)
        lines = result.stdout.splitlines()
        summary_indices = [i for i, line in enumerate(lines) if "AGE_NO_HINT_ITEM_SUMMARY" in line]
        assert summary_indices
        for i in summary_indices:
            assert "(age 42m" in lines[i]
            assert i + 1 < len(lines)
            assert "→" not in lines[i + 1]
        assert "→ 2520" not in result.stdout

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
        """migration.config_rev renders verbatim and is NOT the config_drift
        fallback. ``_BASE_STATUS['config_drift']['loaded_hash']`` is
        ``a1b2c3d4e5f6a7b8``, whose ``[:8]`` fallback prefix (``a1b2c3d4``)
        is deliberately made to differ from ``config_rev`` here so this
        assertion cannot pass merely because the two values collide (as they
        did prior to this fixture change)."""
        status = dict(_BASE_STATUS)
        status["migration"] = {
            "state": "live",
            "config_rev": "ffeeddcc",
            "trial_started_at": None,
            "gates": None,
            "comparison": None,
        }
        result = _run_pstatus(status)
        assert "Migrate:" in result.stdout
        assert "config rev" in result.stdout
        assert "ffeeddcc" in result.stdout
        assert "a1b2c3d4" not in result.stdout

    def test_migrate_footer_config_rev_empty_falls_back_to_loaded_hash(self):
        """migration.config_rev='' (falsy) → the footer renders
        config_drift.loaded_hash[:8] instead (paramem-status.sh:504)."""
        status = dict(_BASE_STATUS)
        status["migration"] = {
            "state": "live",
            "config_rev": "",
            "trial_started_at": None,
            "gates": None,
            "comparison": None,
        }
        result = _run_pstatus(status)
        assert "a1b2c3d4" in result.stdout

    def test_migrate_footer_live_line_exact(self):
        """Exact rendered Migrate footer line for state=live, ANSI-stripped."""
        status = dict(_BASE_STATUS)
        status["migration"] = {
            "state": "live",
            "config_rev": "ffeeddcc",
            "trial_started_at": None,
            "gates": None,
            "comparison": None,
        }
        status["server_started_at"] = "2026-04-22T08:00:00+00:00"
        result = _run_pstatus(status)
        line = _line_with_prefix(result.stdout, "  Migrate:")
        assert line == "  Migrate:  LIVE (config rev ffeeddcc applied 2026-04-22)"

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
        """JSON from before the attention/migration keys existed → script does not crash."""
        status = {
            k: v
            for k, v in _BASE_STATUS.items()
            if k not in ("attention", "migration", "server_started_at")
        }
        result = _run_pstatus(status)
        # Script must exit 0 and render something without crashing.
        assert result.returncode == 0
        assert "ParaMem Server" in result.stdout


class TestMigrateFooterGateBranches:
    """state=trial dispatches on migration.gates.status
    (paramem-status.sh:1041-1061); each branch renders distinct text. Every
    fixture pins gates to the shape the script parses: ``{"status": ...}``.
    """

    def _run_trial(self, gate_status: str) -> subprocess.CompletedProcess:
        status = dict(_BASE_STATUS)
        status["migration"] = {
            "state": "trial",
            "config_rev": "abc",
            "trial_started_at": None,
            "gates": {"status": gate_status},
            "comparison": None,
        }
        return _run_pstatus(status)

    def test_reload_deferred_renders_swap_paused(self):
        result = self._run_trial("reload_deferred")
        assert "SWAP PAUSED" in result.stdout
        assert "new-base reload deferred — Phase B pending" in result.stdout

    def test_phase_a_failed_renders_swap_failed_capture(self):
        result = self._run_trial("phase_a_failed")
        assert "SWAP FAILED" in result.stdout
        assert "Phase A (capture)" in result.stdout

    def test_phase_b_failed_renders_swap_failed_relearn(self):
        result = self._run_trial("phase_b_failed")
        assert "SWAP FAILED" in result.stdout
        assert "Phase B (relearn)" in result.stdout

    def test_phase_b_model_mismatch_renders_swap_aborted(self):
        result = self._run_trial("phase_b_model_mismatch")
        assert "SWAP ABORTED" in result.stdout
        assert "loaded model ≠ target" in result.stdout

    def test_fail_renders_trial_failed(self):
        result = self._run_trial("fail")
        line = _line_with_prefix(result.stdout, "  Migrate:")
        assert line == "  Migrate:  TRIAL FAILED (config rev abc applied 2026-04-22)"

    def test_pass_renders_trial_gates_passed(self):
        result = self._run_trial("pass")
        line = _line_with_prefix(result.stdout, "  Migrate:")
        assert line == (
            "  Migrate:  TRIAL — gates passed, awaiting accept/rollback "
            "(config rev abc applied 2026-04-22)"
        )


class TestRehydrateLine:
    """Rehydrate line (paramem-status.sh:662-671) is rendered only when
    pending_rehydration is true; its progress text depends on whether
    completed/failed tiers are present in the last consolidation result's
    detail (migration_completed_tiers/migration_failed_tiers,
    paramem-status.sh:428-439)."""

    def test_pending_rehydration_not_started(self):
        status = dict(_BASE_STATUS)
        status["pending_rehydration"] = True
        status["effective_mode"] = "cloud-only"
        result = _run_pstatus(status)
        line = _line_with_prefix(result.stdout, "  Rehydrate:")
        assert line == (
            "  Rehydrate:REHYDRATING → effective_mode=cloud-only "
            "(not started — trigger via /consolidate)"
        )

    def test_pending_rehydration_completed_and_failed_tiers(self):
        status = dict(_BASE_STATUS)
        status["pending_rehydration"] = True
        status["effective_mode"] = "local"
        status["last_consolidation_result"] = {
            "detail": {
                "completed_tiers": ["episodic", "semantic"],
                "failed_tiers": ["procedural"],
            }
        }
        result = _run_pstatus(status)
        line = _line_with_prefix(result.stdout, "  Rehydrate:")
        assert line == (
            "  Rehydrate:REHYDRATING → effective_mode=local "
            "(completed:[episodic,semantic] failed:[procedural])"
        )

    def test_no_rehydration_line_when_not_pending(self):
        status = dict(_BASE_STATUS)
        status["pending_rehydration"] = False
        result = _run_pstatus(status)
        assert "Rehydrate:" not in result.stdout


class TestConsolResultRendering:
    """``fmt_result`` must read the REAL writer detail keys (RunRecord.detail
    as written by ``_finalize_interim`` -- every interim-shaped outcome,
    train or simulate venue alike -- and ``_finalize_full`` in
    ``paramem/server/app.py``), not stale/never-written key names.

    Each assertion below discriminates the pre-fix script: pre-fix, the
    ``simulated`` branch read ``detail['episodic_qa']`` (no longer written by
    any current writer — see b547c73/e4d5587 for its history) and always
    rendered ``0ep``; the ``trained`` branch read ``detail['jobs']`` (same —
    no longer written) and always rendered ``(?)``; ``full_trained`` fell
    through to the bare outcome string with zero detail.
    """

    def test_simulated_renders_episodic_and_procedural_rel_counts(self):
        status = dict(_BASE_STATUS)
        status["consolidating"] = False
        status["last_consolidation_result"] = {
            "op_type": "consolidation",
            "outcome": "simulated",
            "summary": "Simulate: 5 episodic, 2 procedural",
            "detail": {
                "sessions": 3,
                "skipped_oom": 0,
                "episodic_rels": 5,
                "procedural_rels": 2,
                "simulated": True,
            },
        }
        result = _run_pstatus(status)
        assert "simulated 3s" in result.stdout
        assert "5rel" in result.stdout
        assert "2pr" in result.stdout
        # Pre-fix key ("episodic_qa", no longer written) always rendered "0ep" here.
        assert "0ep" not in result.stdout

    def test_trained_renders_total_keys_sessions_and_adapter(self):
        status = dict(_BASE_STATUS)
        status["consolidating"] = False
        status["last_consolidation_result"] = {
            "op_type": "consolidation",
            "outcome": "trained",
            "summary": "Interim trained: adapter=episodic_interim_20260801T0000, 120 total keys",
            "detail": {
                "sessions": 4,
                "total_keys": 120,
                "adapter": "episodic_interim_20260801T0000",
            },
        }
        result = _run_pstatus(status)
        assert "trained 120 keys" in result.stdout
        assert "4s" in result.stdout
        assert "adapter=episodic_interim_20260801T0000" in result.stdout
        # Pre-fix key ("jobs", no longer written) always rendered "(?)" here.
        assert "(?)" not in result.stdout

    def test_trained_consol_line_exact(self):
        """Exact rendered Consol line for outcome=trained, ANSI-stripped."""
        status = dict(_BASE_STATUS)
        status["consolidating"] = False
        status["last_consolidation"] = "2026-04-22T04:00:00+00:00"
        status["last_consolidation_result"] = {
            "op_type": "consolidation",
            "outcome": "trained",
            "summary": "Interim trained: adapter=episodic_interim_20260801T0000, 120 total keys",
            "detail": {
                "sessions": 4,
                "total_keys": 120,
                "adapter": "episodic_interim_20260801T0000",
            },
        }
        result = _run_pstatus(status)
        line = _line_with_prefix(result.stdout, "  Consol:")
        assert line == (
            "  Consol:   last 2026-04-22T04:00:00+00:00 | "
            "trained 120 keys (4s, adapter=episodic_interim_20260801T0000)"
        )

    def test_full_trained_renders_tiers_and_total_keys(self):
        status = dict(_BASE_STATUS)
        status["consolidating"] = False
        status["last_consolidation_result"] = {
            "op_type": "consolidation",
            "outcome": "full_trained",
            "summary": "Full cycle full_trained: 400 total keys",
            "detail": {
                "tiers_rebuilt": ["semantic"],
                "total_keys": 400,
            },
        }
        result = _run_pstatus(status)
        assert "full_trained 400 keys (tiers=semantic)" in result.stdout

    def test_interim_discarded_renders_tier_and_adapter_counts(self):
        """``fmt_result`` must render the discard writer's real keys
        (``discarded_tiers``/``unloaded_adapters``/``removed_dirs`` —
        ``interim_discard`` in ``paramem/server/app.py``); pre-fix this
        outcome fell through to the bare ``interim_discarded`` string with
        zero detail."""
        status = dict(_BASE_STATUS)
        status["consolidating"] = False
        status["last_consolidation_result"] = {
            "op_type": "consolidation",
            "outcome": "interim_discarded",
            "summary": "Interim ring discarded: 2 tier(s), 3 active key(s)",
            "detail": {
                "discarded_tiers": [
                    "episodic_interim_20260801T0000",
                    "episodic_interim_20260802T0000",
                ],
                "unloaded_adapters": ["episodic_interim_20260801T0000"],
                "removed_dirs": ["interim_20260801T0000", "interim_20260802T0000"],
            },
        }
        result = _run_pstatus(status)
        assert "interim_discarded 2 tier(s)" in result.stdout
        assert (
            "tiers=episodic_interim_20260801T0000,episodic_interim_20260802T0000" in result.stdout
        )
        assert "adapters=1" in result.stdout
        assert "dirs=2" in result.stdout


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
        """Legacy /status JSON without encryption → dim '-' placeholder, no
        crash, and neither the ON nor OFF wording renders (paramem-status.sh's
        default `*)` case, :1078-1082)."""
        status = {k: v for k, v in _BASE_STATUS.items() if k != "encryption"}
        result = _run_pstatus(status)
        assert result.returncode == 0
        line = _line_with_prefix(result.stdout, "  Security:")
        assert line == "  Security: -"
        assert "age daily" not in result.stdout
        assert "plaintext" not in result.stdout
