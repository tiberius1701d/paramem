"""Integration tests for app lifespan scheduling and debounce gates.

Tests cover:
1. ConsolidationScheduleConfig.training_idle_debounce_s field validation.
2. _dispatch_consolidation idle-debounce gate.
3. _apply_config_live reconciling both systemd timers via
   _reconcile_scheduling_timers.
4. _run_boot_completion_tasks — the boot-completion catch-up task (base-swap
   await, off-loop timer reconcile, backup-before-consolidation catch-up
   dispatch ordering) and _clear_state_task, the done-callback that clears a
   completed task's _state slot.

All GPU/model calls are mocked — no hardware required.
"""

from __future__ import annotations

import asyncio
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
    # Cadence off: no real schedule to be due against, so the durable-stamp
    # catch-up gate (schedule_grammar.scheduled_run_due) never reads/writes
    # anything below — these tests exercise only the idle-debounce gate,
    # which runs before it. ``config.paths`` is a bare MagicMock here (no
    # tmp_path backing), so any cadence that DID reach the durable-stamp
    # read would crash on ``Path(MagicMock())``.
    cfg.consolidation.refresh_cadence = ""

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
                app_module.ConsolidationAction.AUTO
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
                app_module.ConsolidationAction.AUTO
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
                app_module.ConsolidationAction.AUTO
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
                app_module.ConsolidationAction.AUTO
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
    backups_schedule: str = "",
):
    """Minimal mock ServerConfig for ``_apply_config_live`` scheduler tests.

    Mirrors ``_make_config`` in ``tests/server/test_gpu_acquire.py`` (the
    existing ``_apply_config_live`` test pattern), extended with
    ``consolidation.refresh_cadence`` and ``security.backups.schedule`` so
    the reconcile calls under test have real schedule strings to act on.
    ``backups_schedule`` defaults to off so tests that only care about the
    consolidation timer are not surprised by a second real timer write.
    """
    cfg = MagicMock()
    cfg.stt.port = stt_port
    cfg.tts.port = tts_port
    cfg.paths.sessions = sessions_path
    cfg.paths.data = data_path
    cfg.source_path = None
    cfg.consolidation.refresh_cadence = refresh_cadence
    cfg.security.backups.schedule = backups_schedule
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

    def _install_backup_timer_paths(self, tmp_path, monkeypatch):
        from paramem.backup import timer as backup_timer

        monkeypatch.setattr(backup_timer, "UNIT_DIR", tmp_path)
        monkeypatch.setattr(backup_timer, "TIMER_PATH", tmp_path / "paramem-backup.timer")
        monkeypatch.setattr(backup_timer, "SERVICE_PATH", tmp_path / "paramem-backup.service")
        return backup_timer

    def test_backup_schedule_change_applies_live_via_reconcile_scheduling_timers(
        self, tmp_path, monkeypatch
    ):
        """A backup-schedule-only edit (config A off -> config B 'daily 05:00')
        is reconciled into the paramem-backup systemd timer by
        _apply_config_live — proving _reconcile_scheduling_timers (not just
        the consolidation-only call it replaced) is wired into the live-apply
        path, closing the gap where a live security.backups.schedule edit
        never reached systemd.
        """
        import paramem.server.app as app_module

        systemd_timer = self._install_timer_paths(tmp_path, monkeypatch)
        self._install_backup_timer_paths(tmp_path, monkeypatch)

        config_a = _make_apply_live_config(refresh_cadence="12h", backups_schedule="off")
        config_b = _make_apply_live_config(refresh_cadence="12h", backups_schedule="daily 05:00")

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
            result = app_module._apply_config_live()

        assert result["restart_required_reason"] is None
        timer_text = (tmp_path / "paramem-backup.timer").read_text()
        assert "OnCalendar=*-*-* 05:00:00" in timer_text, (
            f"Backup timer unit was not reconciled to 'daily 05:00' in config B: {timer_text!r}"
        )

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


# ---------------------------------------------------------------------------
# TestClearStateTask — the shared asyncio.Task done-callback used by both
# base_swap_task and boot_completion_task.
# ---------------------------------------------------------------------------


class TestClearStateTask:
    def test_clears_matching_task(self):
        """The callback clears the slot when it still holds the completed task."""
        import paramem.server.app as app_module

        async def _noop():
            return None

        async def _run():
            task = asyncio.create_task(_noop())
            app_module._state["_test_task_slot"] = task
            await task
            app_module._clear_state_task("_test_task_slot", task)

        try:
            asyncio.run(_run())
            assert app_module._state.get("_test_task_slot") is None
        finally:
            app_module._state.pop("_test_task_slot", None)

    def test_does_not_clobber_a_newer_task(self):
        """A stale done-callback must not clear a slot a newer task already owns.

        Guards the race where a fresh launch replaces the slot before an
        older task's own done-callback fires.
        """
        import paramem.server.app as app_module

        async def _noop():
            return None

        async def _run():
            old_task = asyncio.create_task(_noop())
            await old_task
            new_task = asyncio.create_task(_noop())
            app_module._state["_test_task_slot"] = new_task
            app_module._clear_state_task("_test_task_slot", old_task)
            await new_task

        try:
            asyncio.run(_run())
            assert app_module._state.get("_test_task_slot") is not None
        finally:
            app_module._state.pop("_test_task_slot", None)


# ---------------------------------------------------------------------------
# TestBootCompletionTaskCatchUp — _run_boot_completion_tasks ordering and
# gating: base-swap await first, off-loop timer reconcile, backup-before-
# consolidation catch-up, cadence-off/schedule-off skips.
# ---------------------------------------------------------------------------


def _make_boot_config(tmp_path, *, refresh_cadence="", backup_schedule="off", artifacts=None):
    """Minimal mock ServerConfig for ``_run_boot_completion_tasks`` tests.

    ``paths.data`` is a real ``tmp_path`` so ``read_backup_state`` can do
    real (empty-dir-tolerant) file I/O without touching production paths.
    """
    cfg = MagicMock()
    cfg.consolidation.refresh_cadence = refresh_cadence
    cfg.security.backups.schedule = backup_schedule
    cfg.security.backups.artifacts = (
        artifacts if artifacts is not None else ["config", "graph", "registry"]
    )
    cfg.paths.data = tmp_path
    return cfg


class TestBootCompletionTaskCatchUp:
    def _run_boot_task(self, config, base_swap_task=None, **overrides):
        import paramem.server.app as app_module

        mocks = {
            "_reconcile_scheduling_timers": MagicMock(),
            "_create_backup": MagicMock(),
            "_dispatch_consolidation": MagicMock(
                return_value=("started_full", app_module.ConsolidationAction.FULL)
            ),
        }
        mocks.update(overrides)

        async def _go():
            with (
                patch.dict(
                    app_module._state,
                    {"config": config, "base_swap_task": base_swap_task},
                    clear=False,
                ),
                patch.object(
                    app_module,
                    "_reconcile_scheduling_timers",
                    mocks["_reconcile_scheduling_timers"],
                ),
                patch.object(app_module, "_create_backup", mocks["_create_backup"]),
                patch.object(
                    app_module, "_dispatch_consolidation", mocks["_dispatch_consolidation"]
                ),
            ):
                await app_module._run_boot_completion_tasks()

        asyncio.run(_go())
        return mocks

    def test_both_schedules_off_no_dispatch(self, tmp_path):
        """Both schedules off -> timers reconciled, but no backup and no
        consolidation dispatch."""
        config = _make_boot_config(tmp_path, refresh_cadence="", backup_schedule="off")
        mocks = self._run_boot_task(config)

        mocks["_reconcile_scheduling_timers"].assert_called_once_with(config)
        mocks["_create_backup"].assert_not_called()
        mocks["_dispatch_consolidation"].assert_not_called()

    def test_backup_no_stamp_runs_before_consolidation_dispatch(self, tmp_path):
        """No backup.json yet (NO_STAMP) -> _create_backup runs with tier
        'daily' strictly BEFORE the consolidation AUTO dispatch — the fold
        rewrites adapter dirs the snapshot bundle reads, so backup must not
        run after it.

        The consolidation cadence's own catch-up stamp is pre-seeded stale
        (DUE) so the due-peek in front of the AUTO dispatch lets this
        ordering test reach the dispatch at all — see
        ``test_consolidation_stale_stamp_dispatches`` /
        ``test_consolidation_no_stamp_does_not_dispatch`` for the peek itself.
        """
        import paramem.server.app as app_module
        from paramem.server.schedule_state import write_last_scheduled_run

        config = _make_boot_config(
            tmp_path,
            refresh_cadence="every 12h",
            backup_schedule="daily 04:00",
            artifacts=["snapshot_bundle"],
        )
        write_last_scheduled_run(tmp_path / "state", time.time() - 86400)

        call_order: list = []

        def _fake_backup(kinds, tier, label):
            call_order.append(("backup", list(kinds), tier, label))

        def _fake_dispatch(action):
            call_order.append(("consolidate", action))
            return "started_full", action

        self._run_boot_task(
            config,
            _create_backup=_fake_backup,
            _dispatch_consolidation=_fake_dispatch,
        )

        assert [c[0] for c in call_order] == ["backup", "consolidate"], call_order
        assert call_order[0] == ("backup", ["snapshot_bundle"], "daily", None)
        assert call_order[1][1] is app_module.ConsolidationAction.AUTO

    def test_backup_not_due_within_window_skips(self, tmp_path):
        """A recent backup.json completed_at inside the current mark's window
        -> NOT_DUE -> _create_backup is never invoked."""
        from datetime import datetime, timezone

        from paramem.backup.state import (
            BACKUP_STATE_SCHEMA_VERSION,
            BackupStateRecord,
            write_backup_state,
        )

        state_dir = tmp_path / "state"
        now_iso = datetime.now(timezone.utc).isoformat()
        write_backup_state(
            state_dir,
            BackupStateRecord(
                schema_version=BACKUP_STATE_SCHEMA_VERSION,
                last_run={"completed_at": now_iso, "success": True},
                last_success_at=now_iso,
                last_failure_at=None,
                last_failure_reason=None,
            ),
        )
        config = _make_boot_config(tmp_path, refresh_cadence="", backup_schedule="every 5h")
        mocks = self._run_boot_task(config)

        mocks["_create_backup"].assert_not_called()

    def test_consolidation_cadence_off_never_dispatches(self, tmp_path):
        """refresh_cadence='' -> AUTO is never dispatched, even though the
        arbitrator itself would otherwise fall through to the content gates
        on every boot."""
        config = _make_boot_config(tmp_path, refresh_cadence="", backup_schedule="off")
        mocks = self._run_boot_task(config)

        mocks["_dispatch_consolidation"].assert_not_called()

    def test_consolidation_stale_stamp_dispatches_auto(self, tmp_path):
        """A real refresh_cadence with a stale (DUE) catch-up stamp ->
        _dispatch_consolidation invoked with AUTO."""
        import paramem.server.app as app_module
        from paramem.server.schedule_state import write_last_scheduled_run

        config = _make_boot_config(tmp_path, refresh_cadence="every 12h", backup_schedule="off")
        write_last_scheduled_run(tmp_path / "state", time.time() - 86400)
        mocks = self._run_boot_task(config)

        mocks["_dispatch_consolidation"].assert_called_once_with(
            app_module.ConsolidationAction.AUTO
        )

    def test_consolidation_no_stamp_does_not_dispatch(self, tmp_path):
        """A real refresh_cadence with NO catch-up stamp on disk yet ->
        the due-peek reads NO_STAMP -> _dispatch_consolidation is NOT
        invoked. Seeding the stamp stays the arbitrator's own job on the
        next real tick (one seeding owner) — the boot task never seeds it.
        """
        config = _make_boot_config(tmp_path, refresh_cadence="every 12h", backup_schedule="off")
        mocks = self._run_boot_task(config)

        mocks["_dispatch_consolidation"].assert_not_called()

    def test_consolidation_fresh_stamp_does_not_dispatch(self, tmp_path):
        """A real refresh_cadence with a catch-up stamp already inside the
        current mark's window -> the due-peek reads NOT_DUE ->
        _dispatch_consolidation is NOT invoked."""
        from paramem.server.schedule_grammar import previous_mark
        from paramem.server.schedule_state import write_last_scheduled_run

        config = _make_boot_config(tmp_path, refresh_cadence="every 12h", backup_schedule="off")
        # Stamp the current mark itself (mirrors scheduled_run_stamp_value's
        # "stamp the mark, not raw now") so the peek reads NOT_DUE.
        mark = previous_mark("every 12h", time.time())
        write_last_scheduled_run(tmp_path / "state", mark)
        mocks = self._run_boot_task(config)

        mocks["_dispatch_consolidation"].assert_not_called()

    def test_backup_due_with_stale_stamp_runs(self, tmp_path):
        """A backup schedule with a stale (DUE, not just absent) completed_at
        -> _create_backup is invoked — the DUE branch, distinct from the
        NO_STAMP branch already covered by
        ``test_backup_no_stamp_runs_before_consolidation_dispatch``."""
        from datetime import datetime, timedelta, timezone

        from paramem.backup.state import (
            BACKUP_STATE_SCHEMA_VERSION,
            BackupStateRecord,
            write_backup_state,
        )

        state_dir = tmp_path / "state"
        stale = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
        write_backup_state(
            state_dir,
            BackupStateRecord(
                schema_version=BACKUP_STATE_SCHEMA_VERSION,
                last_run={"completed_at": stale, "success": True},
                last_success_at=stale,
                last_failure_at=None,
                last_failure_reason=None,
            ),
        )
        config = _make_boot_config(tmp_path, refresh_cadence="", backup_schedule="every 5h")
        mocks = self._run_boot_task(config)

        mocks["_create_backup"].assert_called_once()

    def test_corrupt_backup_state_is_treated_as_no_stamp_and_runs(self, tmp_path):
        """A corrupt backup.json (bad JSON) -> treated as NO_STAMP -> RUN,
        no traceback exit — the same policy as backup/__main__.py's runner
        gate (see tests/backup/test_state.py::TestLastAttemptEpoch)."""
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "backup.json").write_text("NOT JSON {{{{", encoding="utf-8")

        config = _make_boot_config(tmp_path, refresh_cadence="", backup_schedule="every 5h")
        mocks = self._run_boot_task(config)

        mocks["_create_backup"].assert_called_once()

    def test_backup_step_failure_does_not_block_consolidation_dispatch(self, tmp_path):
        """The backup step raising must not prevent the consolidation
        catch-up step from still running — the two are isolated."""
        import paramem.server.app as app_module
        from paramem.server.schedule_state import write_last_scheduled_run

        config = _make_boot_config(
            tmp_path, refresh_cadence="every 12h", backup_schedule="every 5h"
        )
        write_last_scheduled_run(tmp_path / "state", time.time() - 86400)

        def _boom(*a, **k):
            raise RuntimeError("backup step exploded")

        mocks = self._run_boot_task(config, _create_backup=_boom)

        mocks["_dispatch_consolidation"].assert_called_once_with(
            app_module.ConsolidationAction.AUTO
        )

    def test_base_swap_task_awaited_before_catch_up(self, tmp_path):
        """A pending base_swap_task is awaited to completion before the
        timer reconcile (or any catch-up work) runs."""
        import paramem.server.app as app_module

        config = _make_boot_config(tmp_path, refresh_cadence="", backup_schedule="off")
        call_order: list = []

        async def _base_swap():
            await asyncio.sleep(0)
            call_order.append("base_swap_done")

        def _fake_reconcile(cfg):
            call_order.append("reconcile")

        async def _go():
            task = asyncio.create_task(_base_swap())
            with (
                patch.dict(
                    app_module._state,
                    {"config": config, "base_swap_task": task},
                    clear=False,
                ),
                patch.object(app_module, "_reconcile_scheduling_timers", _fake_reconcile),
                patch.object(app_module, "_create_backup"),
                patch.object(app_module, "_dispatch_consolidation"),
            ):
                await app_module._run_boot_completion_tasks()

        asyncio.run(_go())
        assert call_order == ["base_swap_done", "reconcile"], call_order

    def test_base_swap_task_raises_does_not_block_remaining_steps(self, tmp_path):
        """A base_swap_task that raises is isolated in its own try/except —
        the backup, consolidation, and reconcile steps that follow it still
        run (docstring step 1: 'a failure in one never prevents the
        remaining, independent steps from running')."""
        import paramem.server.app as app_module
        from paramem.server.schedule_state import write_last_scheduled_run

        config = _make_boot_config(
            tmp_path, refresh_cadence="every 12h", backup_schedule="every 5h"
        )
        write_last_scheduled_run(tmp_path / "state", time.time() - 86400)

        async def _base_swap():
            raise RuntimeError("base swap exploded")

        async def _go():
            task = asyncio.create_task(_base_swap())
            with (
                patch.dict(
                    app_module._state,
                    {"config": config, "base_swap_task": task},
                    clear=False,
                ),
                patch.object(app_module, "_reconcile_scheduling_timers") as mock_reconcile,
                patch.object(app_module, "_create_backup") as mock_backup,
                patch.object(
                    app_module,
                    "_dispatch_consolidation",
                    MagicMock(return_value=("started_full", app_module.ConsolidationAction.FULL)),
                ) as mock_dispatch,
            ):
                await app_module._run_boot_completion_tasks()
            return mock_reconcile, mock_backup, mock_dispatch

        mock_reconcile, mock_backup, mock_dispatch = asyncio.run(_go())

        mock_backup.assert_called_once()
        mock_dispatch.assert_called_once_with(app_module.ConsolidationAction.AUTO)
        mock_reconcile.assert_called_once_with(config)

    def test_reconcile_runs_last_with_active_schedules(self, tmp_path):
        """With both a real backup schedule and a real consolidation cadence
        DUE, the timer reconcile still runs strictly LAST, after both
        catch-ups have completed (docstring step 4) — the ordering test
        above only exercises this with both schedules off, so it never
        observes reconcile relative to the catch-ups themselves."""
        import paramem.server.app as app_module
        from paramem.server.schedule_state import write_last_scheduled_run

        config = _make_boot_config(
            tmp_path, refresh_cadence="every 12h", backup_schedule="every 5h"
        )
        write_last_scheduled_run(tmp_path / "state", time.time() - 86400)

        call_order: list = []

        def _fake_backup(kinds, tier, label):
            call_order.append("backup")

        def _fake_dispatch(action):
            call_order.append("consolidate")
            return "started_full", action

        def _fake_reconcile(cfg):
            call_order.append("reconcile")

        async def _go():
            with (
                patch.dict(
                    app_module._state,
                    {"config": config, "base_swap_task": None},
                    clear=False,
                ),
                patch.object(app_module, "_reconcile_scheduling_timers", _fake_reconcile),
                patch.object(app_module, "_create_backup", _fake_backup),
                patch.object(app_module, "_dispatch_consolidation", _fake_dispatch),
            ):
                await app_module._run_boot_completion_tasks()

        asyncio.run(_go())

        assert call_order == ["backup", "consolidate", "reconcile"], call_order

    def test_config_none_returns_without_error(self):
        """_state['config'] is None -> returns quietly, no reconcile/backup/
        consolidate work attempted."""
        import paramem.server.app as app_module

        with (
            patch.dict(app_module._state, {"config": None, "base_swap_task": None}, clear=False),
            patch.object(app_module, "_reconcile_scheduling_timers") as mock_reconcile,
            patch.object(app_module, "_create_backup") as mock_backup,
            patch.object(app_module, "_dispatch_consolidation") as mock_dispatch,
        ):
            asyncio.run(app_module._run_boot_completion_tasks())

        mock_reconcile.assert_not_called()
        mock_backup.assert_not_called()
        mock_dispatch.assert_not_called()


# ---------------------------------------------------------------------------
# TestBootCompletionTaskLifespan — full-lifespan integration: the task is
# created pre-yield, stored in _state, and cancelled + cleared at shutdown.
# Mirrors the cloud_only=True lifespan-driving pattern in
# tests/server/test_gpu_release.py::test_lifespan_teardown_data_persisted_before_gpu_release
# (bypasses all CUDA/model-load paths; no GPU touched).
# ---------------------------------------------------------------------------


class TestBootCompletionTaskLifespan:
    def test_created_pre_yield_and_cancelled_cleared_at_shutdown(self, tmp_path):
        import paramem.server.app as app_module
        from paramem.server.config import PathsConfig, ServerConfig, STTConfig, TTSConfig

        config = ServerConfig(model_name="mistral")
        config.cloud_only = True
        config.stt = STTConfig(enabled=False)
        config.tts = TTSConfig(enabled=False)
        root = tmp_path / "data"
        config.paths = PathsConfig(data=root, sessions=root / "sessions", debug=root / "debug")

        saved_state = {
            key: app_module._state.get(key)
            for key in (
                "config",
                "cloud_only_startup",
                "defer_model",
                "boot_completion_task",
                "base_swap_task",
            )
        }
        app_module._state["config"] = config
        app_module._state["cloud_only_startup"] = True
        app_module._state["defer_model"] = False
        app_module._state["boot_completion_task"] = None
        app_module._state["base_swap_task"] = None

        task_holder: dict = {}

        async def _run():
            with (
                patch.object(app_module, "predict_base_bytes", return_value=None),
                patch.object(app_module, "_gpu_occupied", return_value=False),
                patch.object(app_module, "_build_config_derived_state"),
                patch.object(app_module, "_arm_active_store_migration", return_value=False),
                patch.object(app_module, "_release_base_model_in_process"),
                patch.object(app_module, "safe_empty_cache"),
                # The catch-up steps themselves are exercised in
                # TestBootCompletionTaskCatchUp — here we only assert the
                # task's lifecycle (created / stored / cancelled / cleared),
                # so keep them inert regardless of whether the task gets a
                # chance to actually run before cancellation.
                patch.object(app_module, "_reconcile_scheduling_timers"),
                patch.object(app_module, "_create_backup"),
                patch.object(app_module, "_dispatch_consolidation"),
                patch.dict(
                    app_module._state,
                    {"session_buffer": MagicMock(), "speaker_store": MagicMock()},
                    clear=False,
                ),
            ):
                async with app_module.lifespan(app_module.app):
                    task = app_module._state.get("boot_completion_task")
                    task_holder["task"] = task
                    assert task is not None, "boot_completion_task must be created pre-yield"
                    assert isinstance(task, asyncio.Task)
                    assert not task.done()
                # __aexit__ ran the shutdown block, which calls task.cancel().
                # Give the loop a couple of turns to actually process the
                # cancellation and run the done-callback.
                for _ in range(5):
                    await asyncio.sleep(0)

        try:
            asyncio.run(_run())
        finally:
            for key, val in saved_state.items():
                if val is None:
                    app_module._state.pop(key, None)
                else:
                    app_module._state[key] = val

        task = task_holder["task"]
        assert task.cancelled(), "boot_completion_task must be cancelled cleanly at shutdown"
        assert app_module._state.get("boot_completion_task") is None, (
            "boot_completion_task slot must be cleared after the task completes"
        )

    def _make_cloud_only_config(self, tmp_path):
        from paramem.server.config import PathsConfig, ServerConfig, STTConfig, TTSConfig

        config = ServerConfig(model_name="mistral")
        config.cloud_only = True
        config.stt = STTConfig(enabled=False)
        config.tts = TTSConfig(enabled=False)
        root = tmp_path / "data"
        config.paths = PathsConfig(data=root, sessions=root / "sessions", debug=root / "debug")
        return config

    def test_lifespan_start_resets_stale_task_slots(self, tmp_path):
        """A stale, non-None handle left in base_swap_task/boot_completion_task
        by a prior lifespan (whose shutdown .cancel()'d but never awaited them
        to completion) must be reset to None before this lifespan's body runs
        — otherwise the boot-completion task would await a task handle that
        may belong to an already-closed event loop.
        """
        import paramem.server.app as app_module

        config = self._make_cloud_only_config(tmp_path)

        saved_state = {
            key: app_module._state.get(key)
            for key in (
                "config",
                "cloud_only_startup",
                "defer_model",
                "boot_completion_task",
                "base_swap_task",
            )
        }
        app_module._state["config"] = config
        app_module._state["cloud_only_startup"] = True
        app_module._state["defer_model"] = False
        # Simulate the hazard directly: stale, non-None handles as a previous
        # lifespan's un-awaited .cancel() would leave behind.
        app_module._state["boot_completion_task"] = object()
        app_module._state["base_swap_task"] = object()

        observed: dict = {}

        async def _run():
            with (
                patch.object(app_module, "predict_base_bytes", return_value=None),
                patch.object(app_module, "_gpu_occupied", return_value=False),
                patch.object(app_module, "_build_config_derived_state"),
                patch.object(app_module, "_arm_active_store_migration", return_value=False),
                patch.object(app_module, "_release_base_model_in_process"),
                patch.object(app_module, "safe_empty_cache"),
                patch.object(app_module, "_reconcile_scheduling_timers"),
                patch.object(app_module, "_create_backup"),
                patch.object(app_module, "_dispatch_consolidation"),
                patch.dict(
                    app_module._state,
                    {"session_buffer": MagicMock(), "speaker_store": MagicMock()},
                    clear=False,
                ),
            ):
                async with app_module.lifespan(app_module.app):
                    # By the time the lifespan body runs, the stale handles
                    # must already be gone — well before boot_completion_task
                    # is (re)created and could try to await base_swap_task.
                    observed["base_swap_task"] = app_module._state.get("base_swap_task")
                for _ in range(5):
                    await asyncio.sleep(0)

        try:
            asyncio.run(_run())
        finally:
            for key, val in saved_state.items():
                if val is None:
                    app_module._state.pop(key, None)
                else:
                    app_module._state[key] = val

        # base_swap_task is reset to None (no resume marker in this test, so
        # nothing re-populates it) and was never the stale sentinel object.
        assert observed["base_swap_task"] is None

    def test_two_consecutive_lifespans_in_one_process_run_cleanly(self, tmp_path):
        """Two lifespans in sequence (two separate event loops, mirroring
        TestClient reuse or an in-process restart) — the second must create
        and cleanly cancel its own boot_completion_task without choking on
        anything left behind by the first.
        """
        import paramem.server.app as app_module

        saved_state = {
            key: app_module._state.get(key)
            for key in (
                "config",
                "cloud_only_startup",
                "defer_model",
                "boot_completion_task",
                "base_swap_task",
            )
        }

        def _run_one_lifespan() -> "asyncio.Task":
            config = self._make_cloud_only_config(tmp_path)
            app_module._state["config"] = config
            app_module._state["cloud_only_startup"] = True
            app_module._state["defer_model"] = False
            task_holder: dict = {}

            async def _run():
                with (
                    patch.object(app_module, "predict_base_bytes", return_value=None),
                    patch.object(app_module, "_gpu_occupied", return_value=False),
                    patch.object(app_module, "_build_config_derived_state"),
                    patch.object(app_module, "_arm_active_store_migration", return_value=False),
                    patch.object(app_module, "_release_base_model_in_process"),
                    patch.object(app_module, "safe_empty_cache"),
                    patch.object(app_module, "_reconcile_scheduling_timers"),
                    patch.object(app_module, "_create_backup"),
                    patch.object(app_module, "_dispatch_consolidation"),
                    patch.dict(
                        app_module._state,
                        {"session_buffer": MagicMock(), "speaker_store": MagicMock()},
                        clear=False,
                    ),
                ):
                    async with app_module.lifespan(app_module.app):
                        task_holder["task"] = app_module._state.get("boot_completion_task")
                        assert task_holder["task"] is not None
                    for _ in range(5):
                        await asyncio.sleep(0)

            asyncio.run(_run())
            return task_holder["task"]

        try:
            first_task = _run_one_lifespan()
            # asyncio.run() always spins up a fresh event loop — this mirrors
            # a real second lifespan in the same process.
            second_task = _run_one_lifespan()
        finally:
            for key, val in saved_state.items():
                if val is None:
                    app_module._state.pop(key, None)
                else:
                    app_module._state[key] = val

        assert first_task.cancelled()
        assert second_task.cancelled()
        assert app_module._state.get("boot_completion_task") is None


# ---------------------------------------------------------------------------
# Eager consolidation-loop creation must degrade the boot, not crash it —
# ConsolidationLoop.__init__ allocates GPU memory before the post-load VRAM
# gate runs, so a failure there (VramExhausted or otherwise) must not
# propagate out of the lifespan.
# ---------------------------------------------------------------------------


class TestEagerConsolidationLoopBootDegrade:
    def test_eager_loop_failure_does_not_crash_boot(self, tmp_path):
        """A boot-time failure inside ``_eager_create_consolidation_loop``
        (e.g. ``VramExhausted`` from ``ensure_adapters`` -> ``create_adapter``)
        must degrade, not crash: the lifespan completes normally, no
        ``consolidation_loop`` is installed, and the lazy get-or-create
        remains the fallback for every later caller that needs one."""
        import paramem.server.app as app_module
        from paramem.server.config import PathsConfig, ServerConfig, STTConfig, TTSConfig

        config = ServerConfig(model_name="mistral")
        config.cloud_only = True
        config.stt = STTConfig(enabled=False)
        config.tts = TTSConfig(enabled=False)
        root = tmp_path / "data"
        config.paths = PathsConfig(data=root, sessions=root / "sessions", debug=root / "debug")

        saved_state = {
            key: app_module._state.get(key)
            for key in (
                "config",
                "cloud_only_startup",
                "defer_model",
                "boot_completion_task",
                "base_swap_task",
                "consolidation_loop",
            )
        }
        app_module._state["config"] = config
        app_module._state["cloud_only_startup"] = True
        app_module._state["defer_model"] = False
        app_module._state["boot_completion_task"] = None
        app_module._state["base_swap_task"] = None
        app_module._state["consolidation_loop"] = None

        async def _run():
            with (
                patch.object(app_module, "predict_base_bytes", return_value=None),
                patch.object(app_module, "_gpu_occupied", return_value=False),
                patch.object(app_module, "_build_config_derived_state"),
                patch.object(app_module, "_arm_active_store_migration", return_value=False),
                patch.object(app_module, "_release_base_model_in_process"),
                patch.object(app_module, "safe_empty_cache"),
                patch.object(app_module, "_reconcile_scheduling_timers"),
                patch.object(app_module, "_create_backup"),
                patch.object(app_module, "_dispatch_consolidation"),
                patch.object(
                    app_module,
                    "_eager_create_consolidation_loop",
                    side_effect=RuntimeError("boom — simulated eager-loop failure"),
                ),
                patch.dict(
                    app_module._state,
                    {"session_buffer": MagicMock(), "speaker_store": MagicMock()},
                    clear=False,
                ),
            ):
                async with app_module.lifespan(app_module.app):
                    pass
                for _ in range(5):
                    await asyncio.sleep(0)

        try:
            asyncio.run(_run())
        finally:
            for key, val in saved_state.items():
                if val is None:
                    app_module._state.pop(key, None)
                else:
                    app_module._state[key] = val

        assert app_module._state.get("consolidation_loop") is None
