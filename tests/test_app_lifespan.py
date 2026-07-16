"""Integration tests for app lifespan scheduling and debounce gates.

Tests cover:
1. ConsolidationScheduleConfig.training_idle_debounce_s field validation.
2. _dispatch_consolidation idle-debounce gate.

All GPU/model calls are mocked — no hardware required.
"""

from __future__ import annotations

import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# TestIdleDebounceConfig — ConsolidationScheduleConfig.training_idle_debounce_s
# ---------------------------------------------------------------------------


class TestIdleDebounceConfig:
    """ConsolidationScheduleConfig.training_idle_debounce_s field validation."""

    def test_debounce_default_30_seconds(self) -> None:
        """training_idle_debounce_s defaults to 30."""
        from paramem.server.config import ConsolidationScheduleConfig

        cfg = ConsolidationScheduleConfig()
        assert cfg.training_idle_debounce_s == 30

    def test_debounce_negative_rejected(self) -> None:
        """Negative training_idle_debounce_s raises ValueError."""
        import pytest

        from paramem.server.config import ConsolidationScheduleConfig

        with pytest.raises(ValueError, match="training_idle_debounce_s must be >= 0"):
            ConsolidationScheduleConfig(training_idle_debounce_s=-1)

    def test_debounce_zero_allowed(self) -> None:
        """training_idle_debounce_s=0 is valid (disables the gate)."""
        from paramem.server.config import ConsolidationScheduleConfig

        cfg = ConsolidationScheduleConfig(training_idle_debounce_s=0)
        assert cfg.training_idle_debounce_s == 0


# ---------------------------------------------------------------------------
# TestAbortQuiesceTimeoutConfig — ConsolidationScheduleConfig.abort_quiesce_timeout_s
# ---------------------------------------------------------------------------


class TestAbortQuiesceTimeoutConfig:
    """ConsolidationScheduleConfig.abort_quiesce_timeout_s field validation."""

    def test_default_30_seconds(self) -> None:
        """abort_quiesce_timeout_s defaults to 30.0."""
        from paramem.server.config import ConsolidationScheduleConfig

        cfg = ConsolidationScheduleConfig()
        assert cfg.abort_quiesce_timeout_s == 30.0

    def test_zero_rejected(self) -> None:
        """abort_quiesce_timeout_s=0.0 raises ValueError."""
        import pytest

        from paramem.server.config import ConsolidationScheduleConfig

        with pytest.raises(ValueError, match="abort_quiesce_timeout_s must be > 0"):
            ConsolidationScheduleConfig(abort_quiesce_timeout_s=0.0)

    def test_negative_rejected(self) -> None:
        """Negative abort_quiesce_timeout_s raises ValueError."""
        import pytest

        from paramem.server.config import ConsolidationScheduleConfig

        with pytest.raises(ValueError, match="abort_quiesce_timeout_s must be > 0"):
            ConsolidationScheduleConfig(abort_quiesce_timeout_s=-1.0)


# ---------------------------------------------------------------------------
# TestSchedulerIdleDebounce — _dispatch_consolidation gate
# ---------------------------------------------------------------------------


def _make_scheduler_state(last_chat_monotonic=None, debounce_s: int = 30) -> tuple:
    """Return (state_patch_dict, config_mock) for scheduler debounce tests.

    These tests focus on the idle-debounce gate only.  ``_is_full_cycle_due``
    is patched to ``False`` in each caller that needs to reach the pending-
    session path — callers add that patch to their ``with`` block.
    """
    cfg = MagicMock()
    cfg.consolidation.training_idle_debounce_s = debounce_s
    # Calendar-exact cadence — keeps the suspend/power-off catch-up gate
    # (systemd_timer.heartbeat_seconds) a no-op so these tests exercise only
    # the idle-debounce gate.
    cfg.consolidation.refresh_cadence = "12h"

    buf = MagicMock()
    buf.pending_facts.return_value = []

    state_patch = {
        "consolidating": False,
        "mode": "local",
        "background_trainer": None,
        "config": cfg,
        "session_buffer": buf,
        "speaker_store": None,
        "pending_rehydration": False,
        "store_load_degraded": False,
        "last_chat_monotonic": last_chat_monotonic,
    }
    return state_patch, cfg


class TestSchedulerIdleDebounce:
    """_dispatch_consolidation returns 'deferred_idle' within the window."""

    def test_scheduled_tick_returns_deferred_idle_within_debounce_window(self) -> None:
        """Tick arriving 5 s after /chat with debounce=30 returns 'deferred_idle'."""
        import paramem.server.app as app_module

        state_patch, _ = _make_scheduler_state(
            last_chat_monotonic=time.monotonic() - 5,
            debounce_s=30,
        )
        with (
            patch.dict(app_module._state, state_patch, clear=False),
            patch("paramem.server.app._retro_claim_orphan_sessions", return_value=0),
        ):
            result, _action = app_module._dispatch_consolidation(
                app_module.ConsolidationAction.AUTO, apply_schedule_gate=True
            )

        assert result == "deferred_idle", (
            f"Expected 'deferred_idle' within debounce window but got {result!r}"
        )

    def test_scheduled_tick_proceeds_after_debounce_elapsed(self) -> None:
        """Tick arriving 60 s after /chat with debounce=30 proceeds past the gate."""
        import paramem.server.app as app_module

        state_patch, _ = _make_scheduler_state(
            last_chat_monotonic=time.monotonic() - 60,
            debounce_s=30,
        )
        # The tick should reach the no-pending check and return noop_no_pending
        # (session_buffer.pending_facts() returns [] from the mock).
        # _is_full_cycle_due is patched False so this test exercises the debounce
        # gate in isolation without triggering the full-cycle event-loop path.
        with (
            patch.dict(app_module._state, state_patch, clear=False),
            patch("paramem.server.app._retro_claim_orphan_sessions", return_value=0),
            patch("paramem.server.app._is_full_cycle_due", return_value=False),
        ):
            result, _action = app_module._dispatch_consolidation(
                app_module.ConsolidationAction.AUTO, apply_schedule_gate=True
            )

        assert result != "deferred_idle", (
            f"Expected tick to proceed past debounce gate but got {result!r}"
        )

    def test_scheduled_tick_debounce_zero_disables_gate(self) -> None:
        """debounce_s=0 disables the gate even when chat fired right now."""
        import paramem.server.app as app_module

        state_patch, _ = _make_scheduler_state(
            last_chat_monotonic=time.monotonic(),
            debounce_s=0,
        )
        # _is_full_cycle_due is patched False so this test exercises the debounce
        # gate in isolation without triggering the full-cycle event-loop path.
        with (
            patch.dict(app_module._state, state_patch, clear=False),
            patch("paramem.server.app._retro_claim_orphan_sessions", return_value=0),
            patch("paramem.server.app._is_full_cycle_due", return_value=False),
        ):
            result, _action = app_module._dispatch_consolidation(
                app_module.ConsolidationAction.AUTO, apply_schedule_gate=True
            )

        assert result != "deferred_idle", f"debounce_s=0 must disable gate; got {result!r}"

    def test_scheduled_tick_no_chat_yet_proceeds(self) -> None:
        """last_chat_monotonic=None skips the gate (no /chat has fired yet)."""
        import paramem.server.app as app_module

        state_patch, _ = _make_scheduler_state(last_chat_monotonic=None, debounce_s=30)
        # _is_full_cycle_due is patched False so this test exercises the debounce
        # gate in isolation without triggering the full-cycle event-loop path.
        with (
            patch.dict(app_module._state, state_patch, clear=False),
            patch("paramem.server.app._retro_claim_orphan_sessions", return_value=0),
            patch("paramem.server.app._is_full_cycle_due", return_value=False),
        ):
            result, _action = app_module._dispatch_consolidation(
                app_module.ConsolidationAction.AUTO, apply_schedule_gate=True
            )

        assert result != "deferred_idle", f"No chat yet must not defer; got {result!r}"


# ---------------------------------------------------------------------------
# TestApplyConfigLiveSchedulerParticipation — _apply_config_live re-reads
# consolidation.refresh_cadence from config B and reconciles the systemd
# timer to it, so a cadence-only edit applies live and drift clears without
# a restart. All systemctl calls are mocked (_run_systemctl) and unit files
# are redirected into tmp_path — no test here can reach the live user
# systemd session.
# ---------------------------------------------------------------------------


def _make_apply_live_config(
    refresh_cadence: str = "12h",
    stt_port: int = 10300,
    tts_port: int = 10301,
    sessions_path: str = "/data/sessions",
    data_path: str = "/data",
):
    """Minimal mock ServerConfig for ``_apply_config_live`` scheduler tests.

    Mirrors ``_make_config`` in ``tests/server/test_gpu_acquire.py`` (the
    existing ``_apply_config_live`` test pattern), extended with
    ``consolidation.refresh_cadence`` so the reconcile call under test has a
    real schedule string to act on.
    """
    cfg = MagicMock()
    cfg.stt.port = stt_port
    cfg.tts.port = tts_port
    cfg.paths.sessions = sessions_path
    cfg.paths.data = data_path
    cfg.source_path = None
    cfg.consolidation.refresh_cadence = refresh_cadence
    return cfg


@contextmanager
def _null_gpu_lock_sync(timeout=-1):
    """No-op replacement for gpu_lock_sync — always succeeds immediately."""
    yield


def _mock_run_systemctl(*args, **kwargs):
    return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")


class TestApplyConfigLiveSchedulerParticipation:
    def _install_timer_paths(self, tmp_path, monkeypatch):
        from paramem.server import systemd_timer

        monkeypatch.setattr(systemd_timer, "UNIT_DIR", tmp_path)
        monkeypatch.setattr(systemd_timer, "TIMER_PATH", tmp_path / "paramem-consolidate.timer")
        monkeypatch.setattr(systemd_timer, "SERVICE_PATH", tmp_path / "paramem-consolidate.service")
        return systemd_timer

    def test_cadence_change_applies_live_without_restart(self, tmp_path, monkeypatch):
        """A cadence-only edit (config A '12h' -> config B '6h') is reconciled
        into the systemd timer by _apply_config_live, with no restart
        required — proving the scheduler is now a live-apply participant.
        """
        import paramem.server.app as app_module

        systemd_timer = self._install_timer_paths(tmp_path, monkeypatch)

        config_a = _make_apply_live_config(refresh_cadence="12h")
        config_b = _make_apply_live_config(refresh_cadence="6h")

        state_patch = {
            "mode": "cloud-only",
            "cloud_only_reason": "live_reload",
            "config": config_a,
            "config_path": "configs/server.yaml",
            "consolidating": False,
        }

        with (
            patch.dict(app_module._state, state_patch, clear=False),
            patch("paramem.server.gpu_lock.gpu_lock_sync", _null_gpu_lock_sync),
            patch(
                "paramem.server.drift.compute_config_hash",
                side_effect=["disk_hash_b", "mem_hash_a"],
            ),
            patch.object(Path, "exists", return_value=True),
            patch.object(app_module, "load_server_config", return_value=config_b),
            patch.object(
                systemd_timer, "_run_systemctl", side_effect=_mock_run_systemctl
            ) as mock_systemctl,
            patch.object(app_module, "_live_reload_base_model"),
            patch.object(app_module, "_set_voice_pipeline_profile"),
        ):
            app_module._state["mode"] = "local"
            result = app_module._apply_config_live()

        assert result["restart_required_reason"] is None, (
            f"Cadence-only change must not require a restart, got: {result}"
        )
        timer_text = (tmp_path / "paramem-consolidate.timer").read_text()
        assert "OnCalendar=*-*-* 00,06,12,18:00:00" in timer_text, (
            f"Timer unit was not reconciled to the '6h' cadence in config B: {timer_text!r}"
        )
        first_args = [c.args[0] for c in mock_systemctl.call_args_list]
        assert "enable" in first_args, (
            "systemd_timer.reconcile was not invoked from _apply_config_live "
            f"(no enable call seen): {mock_systemctl.call_args_list}"
        )

    def test_cadence_off_disarms_timer_via_apply_config_live(self, tmp_path, monkeypatch):
        """refresh_cadence='' in config B fully disarms the timer when applied
        live — the off path stays absolute after wiring in the reconcile call.
        """
        import paramem.server.app as app_module

        systemd_timer = self._install_timer_paths(tmp_path, monkeypatch)

        # Pre-install an active timer (as if a prior '12h' cadence was live).
        with patch.object(systemd_timer, "_run_systemctl", side_effect=_mock_run_systemctl):
            systemd_timer.reconcile("every 12h")
        assert (tmp_path / "paramem-consolidate.timer").exists()

        config_a = _make_apply_live_config(refresh_cadence="12h")
        config_b = _make_apply_live_config(refresh_cadence="")

        state_patch = {
            "mode": "cloud-only",
            "cloud_only_reason": "live_reload",
            "config": config_a,
            "config_path": "configs/server.yaml",
            "consolidating": False,
        }

        with (
            patch.dict(app_module._state, state_patch, clear=False),
            patch("paramem.server.gpu_lock.gpu_lock_sync", _null_gpu_lock_sync),
            patch(
                "paramem.server.drift.compute_config_hash",
                side_effect=["disk_hash_b", "mem_hash_a"],
            ),
            patch.object(Path, "exists", return_value=True),
            patch.object(app_module, "load_server_config", return_value=config_b),
            patch.object(systemd_timer, "_run_systemctl", side_effect=_mock_run_systemctl),
            patch.object(app_module, "_live_reload_base_model"),
            patch.object(app_module, "_set_voice_pipeline_profile"),
        ):
            app_module._state["mode"] = "local"
            app_module._apply_config_live()

        assert not (tmp_path / "paramem-consolidate.timer").exists(), (
            "refresh_cadence='' must disarm (remove) the timer unit via _apply_config_live"
        )
        assert not (tmp_path / "paramem-consolidate.service").exists()

    def test_no_config_b_skips_scheduler_reconcile(self, tmp_path, monkeypatch):
        """When config B fails to load, the scheduler reconcile is skipped
        (nothing to read) rather than acting on a stale/absent config."""
        import paramem.server.app as app_module

        systemd_timer = self._install_timer_paths(tmp_path, monkeypatch)

        config_a = _make_apply_live_config(refresh_cadence="12h")

        state_patch = {
            "mode": "cloud-only",
            "cloud_only_reason": "live_reload",
            "config": config_a,
            "config_path": "configs/server.yaml",
            "consolidating": False,
        }

        def _failing_load(path, **kw):
            raise ValueError("simulated parse failure")

        with (
            patch.dict(app_module._state, state_patch, clear=False),
            patch("paramem.server.gpu_lock.gpu_lock_sync", _null_gpu_lock_sync),
            patch(
                "paramem.server.drift.compute_config_hash",
                side_effect=["disk_hash_b", "mem_hash_a"],
            ),
            patch.object(Path, "exists", return_value=True),
            patch.object(app_module, "load_server_config", _failing_load),
            patch.object(
                systemd_timer, "_run_systemctl", side_effect=_mock_run_systemctl
            ) as mock_systemctl,
            patch.object(app_module, "_live_reload_base_model"),
            patch.object(app_module, "_set_voice_pipeline_profile"),
        ):
            app_module._state["mode"] = "local"
            app_module._apply_config_live()

        assert mock_systemctl.call_args_list == [], (
            "Scheduler reconcile must not run when config B failed to load: "
            f"{mock_systemctl.call_args_list}"
        )
        assert not (tmp_path / "paramem-consolidate.timer").exists()
