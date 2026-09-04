"""Integration tests for app lifespan scheduling and debounce gates.

Tests cover:
1. ConsolidationScheduleConfig.training_idle_debounce_s and
   .abort_quiesce_timeout_s field validation.
2. _apply_config_live reconciling both systemd timers via
   _reconcile_scheduling_timers.
3. _run_boot_completion_tasks — the boot-completion catch-up task (base-swap
   await, off-loop timer reconcile, backup-before-consolidation catch-up
   dispatch ordering) and _clear_state_task, the done-callback that clears a
   completed task's _state slot.

All GPU/model calls are mocked — no hardware required.
"""

from __future__ import annotations

import asyncio
import subprocess
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

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
# TestApplyConfigLiveSchedulerParticipation — _apply_config_live re-reads
# consolidation.refresh_cadence from config B and reconciles the systemd
# timer to it, so a cadence-only edit applies live and drift clears without
# a restart. All systemctl calls are mocked (paramem.utils.systemctl.run) and
# unit files are redirected into tmp_path — no test here can reach the live
# user systemd session.
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
    """``_apply_config_live`` reconciles both systemd timers against config B
    -- the on-disk config actually being applied -- via
    :func:`_reconcile_scheduling_timers`. That call happens BEFORE the
    R-PORT/R-PATHS carve classification and reads config B's schedule
    fields unconditionally: it never diffs against config A's former
    values, so a live apply reconciles the timer even when the schedule
    fields did not change (see ``_apply_config_live``'s own docstring, step
    3b). Both real timer reconciles (``systemd_timer.reconcile``,
    ``backup_timer.reconcile``) are replaced by recorders -- no live systemd
    session is touched."""

    def _make_schedule_config(
        self,
        *,
        refresh_cadence: str,
        full_window: str = "01:00-04:00",
        interim_resume: str = "immediate",
        max_interim_count: int = 7,
        backups_schedule: str = "",
    ):
        cfg = _make_apply_live_config(
            refresh_cadence=refresh_cadence, backups_schedule=backups_schedule
        )
        cfg.consolidation.full_window = full_window
        cfg.consolidation.interim_resume = interim_resume
        cfg.consolidation.max_interim_count = max_interim_count
        return cfg

    def _run_apply(self, config_a, config_b):
        from pathlib import Path

        import paramem.server.app as app_module
        from paramem.backup import timer as backup_timer
        from paramem.server import systemd_timer

        consolidation_calls: list[tuple[tuple, dict]] = []
        backup_calls: list[tuple[tuple, dict]] = []

        def _fake_systemd_reconcile(*args, **kwargs):
            consolidation_calls.append((args, kwargs))
            return "ok"

        def _fake_backup_reconcile(*args, **kwargs):
            backup_calls.append((args, kwargs))
            return "ok"

        state_patch = {
            "mode": "cloud-only",
            "cloud_only_reason": "live_reload",
            "config": config_a,
            "config_path": "configs/server.yaml",
            "consolidating": False,
            "config_drift": {},
        }

        with (
            patch.dict(app_module._state, state_patch, clear=False),
            patch("paramem.server.gpu_lock.gpu_lock_sync", _null_gpu_lock_sync),
            patch("paramem.server.drift.compute_config_hash", return_value="disk_hash_b"),
            patch.object(Path, "exists", return_value=True),
            patch.object(app_module, "load_server_config", return_value=config_b),
            patch.object(systemd_timer, "reconcile", _fake_systemd_reconcile),
            patch.object(backup_timer, "reconcile", _fake_backup_reconcile),
            patch.object(app_module, "_live_reload_base_model", return_value=None),
            patch.object(app_module, "_set_voice_pipeline_profile"),
        ):
            app_module._apply_config_live()

        return consolidation_calls, backup_calls

    def test_reconcile_reads_config_bs_new_cadence_and_extra_calendars(self) -> None:
        """The reconcile call's cadence argument and computed
        ``extra_calendars`` (a ring's own ``full_window`` start) both come
        from config B -- the incoming config -- never config A's former
        cadence."""
        config_a = self._make_schedule_config(refresh_cadence="6h")
        config_b = self._make_schedule_config(refresh_cadence="12h")

        consolidation_calls, _backup_calls = self._run_apply(config_a, config_b)

        assert len(consolidation_calls) == 1
        args, kwargs = consolidation_calls[0]
        assert args == ("12h",)
        assert kwargs["extra_calendars"] == ["*-*-* 01:00:00"]

    def test_reconcile_runs_even_when_the_schedule_fields_are_unchanged(self) -> None:
        """``_apply_config_live`` re-reads config B's PRESENT schedule
        fields unconditionally -- it never diffs against config A's former
        values -- so the reconcile still fires on an apply that changes
        nothing about the schedule."""
        config_a = self._make_schedule_config(refresh_cadence="12h")
        config_b = self._make_schedule_config(refresh_cadence="12h")

        consolidation_calls, _backup_calls = self._run_apply(config_a, config_b)

        assert len(consolidation_calls) == 1
        args, _kwargs = consolidation_calls[0]
        assert args == ("12h",)


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
# TestReconcileSchedulingTimers — _reconcile_scheduling_timers: the
# consolidation timer carries the cadence entry plus a window-start entry
# per configured window; the backup timer reconciles on its own schedule
# alone, with no extra_calendars.
# ---------------------------------------------------------------------------


class TestReconcileSchedulingTimers:
    def _reconcile(self, config):
        """Call ``_reconcile_scheduling_timers`` with both timer reconciles
        replaced by recorders. Returns (consolidation_calls, backup_calls),
        each a list of ``(args, kwargs)``."""
        import paramem.server.app as app_module
        from paramem.backup import timer as backup_timer
        from paramem.server import systemd_timer

        consolidation_calls: list[tuple[tuple, dict]] = []
        backup_calls: list[tuple[tuple, dict]] = []

        def _fake_systemd_reconcile(*args, **kwargs):
            consolidation_calls.append((args, kwargs))
            return "ok"

        def _fake_backup_reconcile(*args, **kwargs):
            backup_calls.append((args, kwargs))
            return "ok"

        with (
            patch.object(systemd_timer, "reconcile", _fake_systemd_reconcile),
            patch.object(backup_timer, "reconcile", _fake_backup_reconcile),
        ):
            app_module._reconcile_scheduling_timers(config)

        return consolidation_calls, backup_calls

    def test_default_fixture_config_adds_the_full_window_start_only(self) -> None:
        """Fixture defaults: refresh_cadence=12h, max_interim_count=7 (a
        ring), interim_resume=immediate (not a window) -> one extra entry,
        the full_window start."""
        from paramem.server.config import load_server_config

        config = load_server_config("tests/fixtures/server.yaml")

        consolidation_calls, backup_calls = self._reconcile(config)

        assert len(consolidation_calls) == 1
        args, kwargs = consolidation_calls[0]
        assert args == (config.consolidation.refresh_cadence,)
        assert kwargs["extra_calendars"] == ["*-*-* 01:00:00"]

        assert len(backup_calls) == 1
        _backup_args, backup_kwargs = backup_calls[0]
        assert "extra_calendars" not in backup_kwargs

    def test_no_ring_adds_no_window_start(self) -> None:
        """max_interim_count=0: no ring, so full_window is never read and no
        extra entry is added."""
        from paramem.server.config import load_server_config

        config = load_server_config("tests/fixtures/server.yaml")
        config.consolidation.max_interim_count = 0

        consolidation_calls, _backup_calls = self._reconcile(config)

        assert consolidation_calls[0][1]["extra_calendars"] == []

    def test_windowed_interim_resume_adds_a_second_entry_in_order(self) -> None:
        """A ring (full_window start) plus a windowed interim_resume (its
        own start) -> two entries, full_window first, interim_resume second."""
        from paramem.server.config import load_server_config

        config = load_server_config("tests/fixtures/server.yaml")
        config.consolidation.interim_resume = "22:00-23:00"

        consolidation_calls, _backup_calls = self._reconcile(config)

        assert consolidation_calls[0][1]["extra_calendars"] == [
            "*-*-* 01:00:00",
            "*-*-* 22:00:00",
        ]


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

    def test_backup_catch_up_runs_before_the_consolidation_dispatch(self, tmp_path):
        """The backup step (when due) completes BEFORE the consolidation
        catch-up dispatches — a fold rewrites the tier adapter directories
        the backup snapshot reads, so running the backup after a fold would
        capture the fold's own output as though it predated the fold."""
        import paramem.server.app as app_module

        config = _make_boot_config(tmp_path, refresh_cadence="", backup_schedule="every 5h")
        call_order: list = []

        def _fake_create_backup(*args, **kwargs):
            call_order.append("backup")

        def _fake_dispatch(*args, **kwargs):
            call_order.append("dispatch")
            return "started_full", app_module.ConsolidationAction.FULL

        async def _go():
            with (
                patch.dict(
                    app_module._state,
                    {"config": config, "base_swap_task": None},
                    clear=False,
                ),
                patch.object(app_module, "_reconcile_scheduling_timers"),
                patch.object(app_module, "_create_backup", _fake_create_backup),
                patch.object(app_module, "_dispatch_consolidation", _fake_dispatch),
            ):
                await app_module._run_boot_completion_tasks()

        asyncio.run(_go())
        assert call_order == ["backup", "dispatch"], call_order

    @pytest.mark.parametrize("refresh_cadence", ["", "12h"])
    def test_consolidation_dispatch_requests_auto_with_boot_reason_regardless_of_cadence(
        self, tmp_path, refresh_cadence
    ):
        """The boot task's own consolidation catch-up always requests
        ``AUTO`` with ``reason=BOOT`` — the decider owns dueness, so the
        request is unconditional whether the cadence is off (manual-only)
        or a real cadence."""
        import paramem.server.app as app_module
        from paramem.server.consolidation_choice import DispatchReason

        config = _make_boot_config(tmp_path, refresh_cadence=refresh_cadence, backup_schedule="off")
        mocks = self._run_boot_task(config)

        mocks["_dispatch_consolidation"].assert_called_once_with(
            app_module.ConsolidationAction.AUTO, reason=DispatchReason.BOOT
        )

    def test_timer_reconcile_runs_last(self, tmp_path):
        """The timer reconcile is dispatched after both the backup catch-up
        and the consolidation catch-up have already run/stamped — so a
        ``Persistent=true`` tick the reconcile's own ``enable`` might fire
        curls a server that already answers not-due, rather than racing the
        boot task's own catch-up work into a double run."""
        import paramem.server.app as app_module

        config = _make_boot_config(tmp_path, refresh_cadence="", backup_schedule="every 5h")
        call_order: list = []

        def _fake_create_backup(*args, **kwargs):
            call_order.append("backup")

        def _fake_dispatch(*args, **kwargs):
            call_order.append("dispatch")
            return "started_full", app_module.ConsolidationAction.FULL

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
                patch.object(app_module, "_create_backup", _fake_create_backup),
                patch.object(app_module, "_dispatch_consolidation", _fake_dispatch),
            ):
                await app_module._run_boot_completion_tasks()

        asyncio.run(_go())
        assert call_order == ["backup", "dispatch", "reconcile"], call_order

    def test_a_raising_backup_step_does_not_block_the_dispatch(self, tmp_path):
        """Each catch-up step is isolated in its own try/except — a raise in
        the backup step is logged and swallowed, and the consolidation
        catch-up still runs."""
        config = _make_boot_config(tmp_path, refresh_cadence="", backup_schedule="every 5h")
        mocks = self._run_boot_task(
            config, _create_backup=MagicMock(side_effect=RuntimeError("backup boom"))
        )

        mocks["_dispatch_consolidation"].assert_called_once()

    def test_a_raising_base_swap_task_does_not_block_remaining_steps(self, tmp_path):
        """A ``base_swap_task`` that raises is caught (logged) at step 1 —
        the backup, consolidation-dispatch, and timer-reconcile steps still
        run to completion afterward."""
        import paramem.server.app as app_module

        config = _make_boot_config(tmp_path, refresh_cadence="", backup_schedule="every 5h")

        async def _failing_base_swap():
            await asyncio.sleep(0)
            raise RuntimeError("base-swap boom")

        mocks = {
            "_reconcile_scheduling_timers": MagicMock(),
            "_create_backup": MagicMock(),
            "_dispatch_consolidation": MagicMock(
                return_value=("started_full", app_module.ConsolidationAction.FULL)
            ),
        }

        async def _go():
            task = asyncio.create_task(_failing_base_swap())
            with (
                patch.dict(
                    app_module._state,
                    {"config": config, "base_swap_task": task},
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
        mocks["_create_backup"].assert_called_once()
        mocks["_dispatch_consolidation"].assert_called_once()
        mocks["_reconcile_scheduling_timers"].assert_called_once()

    def test_config_none_returns_without_error(self):
        """``_state["config"] is None`` (never booted, or a config load
        failure) returns immediately — no backup, dispatch, or reconcile
        step runs, and nothing raises."""
        import paramem.server.app as app_module

        mocks = {
            "_reconcile_scheduling_timers": MagicMock(),
            "_create_backup": MagicMock(),
            "_dispatch_consolidation": MagicMock(),
        }

        async def _go():
            with (
                patch.dict(
                    app_module._state,
                    {"config": None, "base_swap_task": None},
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
        mocks["_create_backup"].assert_not_called()
        mocks["_dispatch_consolidation"].assert_not_called()
        mocks["_reconcile_scheduling_timers"].assert_not_called()


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
                patch.object(app_module, "_build_runtime_components"),
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
                patch.object(app_module, "_build_runtime_components"),
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
                    patch.object(app_module, "_build_runtime_components"),
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
# TestReclaimLoopPermanentDegradeGate — auto-reclaim must never arm after a
# permanent cloud-only degrade
# ---------------------------------------------------------------------------


class TestReclaimLoopPermanentDegradeGate:
    """The reclaim-task creation at lifespan boot must consult
    ``_PERMANENT_CLOUD_ONLY_REASONS`` at the point it actually creates the
    task — not rely on a snapshot taken before a mid-boot degrade can change
    ``cloud_only_reason``."""

    def _make_defer_model_config(self, tmp_path):
        from paramem.server.config import PathsConfig, ServerConfig, STTConfig, TTSConfig

        config = ServerConfig(model_name="mistral")
        config.cloud_only = False
        config.stt = STTConfig(enabled=False)
        config.tts = TTSConfig(enabled=False)
        root = tmp_path / "data"
        config.paths = PathsConfig(data=root, sessions=root / "sessions", debug=root / "debug")
        return config

    def _seeded_state_keys(self):
        return (
            "config",
            "cloud_only_startup",
            "defer_model",
            "cloud_only_reason",
            "mode",
            "boot_completion_task",
            "base_swap_task",
            "reclaim_task",
            "consolidation_loop",
        )

    def test_transient_defer_model_boot_still_arms_reclaim_task(self, tmp_path):
        """A --defer-model boot that hits no fault (``cloud_only_reason``
        stays ``'training'``, never permanent) must still arm the
        auto-reclaim loop — the permanence gate must not over-block a
        legitimately reclaimable cloud-only boot."""
        import paramem.server.app as app_module

        config = self._make_defer_model_config(tmp_path)

        saved_state = {key: app_module._state.get(key) for key in self._seeded_state_keys()}
        app_module._state["config"] = config
        app_module._state["cloud_only_startup"] = False
        app_module._state["defer_model"] = True
        app_module._state["cloud_only_reason"] = None
        app_module._state["mode"] = "local"
        app_module._state["boot_completion_task"] = None
        app_module._state["base_swap_task"] = None
        app_module._state["reclaim_task"] = None
        app_module._state["consolidation_loop"] = None

        task_holder: dict = {}

        async def _run():
            with (
                patch.object(app_module, "predict_base_bytes", return_value=None),
                patch.object(app_module, "_gpu_occupied", return_value=False),
                patch("paramem.server.app.torch.cuda.is_available", return_value=True),
                patch.object(app_module, "_build_runtime_components"),
                patch.object(app_module, "_arm_active_store_migration", return_value=False),
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
                    assert app_module._state.get("cloud_only_reason") == "training"
                    task = app_module._state.get("reclaim_task")
                    task_holder["task"] = task
                    assert task is not None, "reclaim_task must be armed for a transient reason"
                    assert isinstance(task, asyncio.Task)
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

        assert task_holder["task"].cancelled(), "reclaim_task must be cancelled at shutdown"

    def test_mid_boot_fatal_cuda_fault_blocks_the_reclaim_task(self, tmp_path):
        """A --defer-model boot starts with a transient reason ('training'),
        but a fatal CUDA fault surfacing from _build_runtime_components
        (still reached even when cloud_only=True from the start) degrades
        cloud_only_reason to the permanent 'cuda_fault_persistent' via
        _fail_fast_cuda -> _degrade_to_cloud_only. The reclaim-task arm
        check must read that live value at task-creation time, not a
        snapshot taken before the fault -- arming it here would reload the
        base model straight back into the same poisoned CUDA context."""
        from paramem.utils.vram_guard import is_fatal_cuda_fault

        fault = RuntimeError("CUDA error: an illegal memory access was encountered")
        assert is_fatal_cuda_fault(fault), (
            "precondition: the injected fault must actually classify as fatal"
        )

        import paramem.server.app as app_module

        config = self._make_defer_model_config(tmp_path)

        saved_state = {key: app_module._state.get(key) for key in self._seeded_state_keys()}
        app_module._state["config"] = config
        app_module._state["cloud_only_startup"] = False
        app_module._state["defer_model"] = True
        app_module._state["cloud_only_reason"] = None
        app_module._state["mode"] = "local"
        app_module._state["boot_completion_task"] = None
        app_module._state["base_swap_task"] = None
        app_module._state["reclaim_task"] = None
        app_module._state["consolidation_loop"] = None

        async def _run():
            with (
                patch.object(app_module, "predict_base_bytes", return_value=None),
                patch.object(app_module, "_gpu_occupied", return_value=False),
                patch("paramem.server.app.torch.cuda.is_available", return_value=True),
                patch.object(app_module, "_build_runtime_components", side_effect=fault),
                # _fail_fast_cuda os._exit(1)s on a FRESH crash-loop burst
                # (systemd-restart recovery) -- only a burst the crash-loop
                # guard reports exhausted degrades in-process instead, which
                # is the scenario this test drives.
                patch.object(app_module, "_cuda_crashloop_exhausted", return_value=True),
                patch.object(app_module, "_release_base_model_in_process"),
                patch.object(app_module, "_arm_active_store_migration", return_value=False),
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
                    assert app_module._state.get("cloud_only_reason") == "cuda_fault_persistent"
                    assert app_module._state.get("reclaim_task") is None, (
                        "reclaim_task must NOT be armed once a mid-boot fault has "
                        "landed cloud_only_reason on a permanent reason"
                    )

        try:
            asyncio.run(_run())
        finally:
            for key, val in saved_state.items():
                if val is None:
                    app_module._state.pop(key, None)
                else:
                    app_module._state[key] = val


# ---------------------------------------------------------------------------
# TestModeSlotHygiene — "mode" must reset to its pre-boot default at the
# start of every lifespan
# ---------------------------------------------------------------------------


class TestModeSlotHygiene:
    """A second lifespan in the same process (TestClient reuse, in-process
    restart) must never inherit a stale ``mode="cloud-only"`` left behind by
    a prior lifespan's degrade — the slot-hygiene block resets it to the
    pre-boot default ("local", the module-level ``_state`` initializer's own
    value) before the mode-write guard can see it.

    The mode-write guard (``if _state.get("mode") != "cloud-only":``) only
    WRITES when the current value differs from "cloud-only" — so a stale
    "cloud-only" left by a prior run is indistinguishable from THIS run's own
    in-progress degrade unless the hygiene block resets it first. The
    regression only surfaces when this run's OWN boot-computed mode is
    "local" (a healthy boot) while a prior run's stale value was
    "cloud-only": without the reset, the guard sees "cloud-only" already
    present and never overwrites it, permanently pinning a healthy boot into
    cloud-only reporting.
    """

    def test_second_lifespan_resets_stale_cloud_only_mode_for_healthy_local_boot(self, tmp_path):
        """Drives the REAL lifespan with a stale ``mode="cloud-only"``
        seeded up front (as a prior lifespan's degrade would leave it) and
        every condition that would otherwise compute ``cloud_only=True``
        turned off (no ``--cloud-only``, no ``--defer-model``, no GPU
        conflict) — the boot-computed value for THIS run is "local". The
        hygiene reset must let that value win.
        """
        import paramem.server.app as app_module
        from paramem.server.config import PathsConfig, ServerConfig, STTConfig, TTSConfig

        config = ServerConfig(model_name="mistral")
        config.cloud_only = False
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
                "cloud_only_reason",
                "mode",
                "boot_completion_task",
                "base_swap_task",
                "reclaim_task",
                "consolidation_loop",
            )
        }
        app_module._state["config"] = config
        app_module._state["cloud_only_startup"] = False
        app_module._state["defer_model"] = False
        app_module._state["boot_completion_task"] = None
        app_module._state["base_swap_task"] = None
        app_module._state["reclaim_task"] = None
        app_module._state["consolidation_loop"] = None
        # Simulate the residue of a PRIOR lifespan's degrade: a stale
        # "cloud-only" mode with no corresponding condition forcing THIS
        # run's own boot-computed cloud_only to True.
        app_module._state["mode"] = "cloud-only"
        app_module._state["cloud_only_reason"] = "cuda_fault_persistent"

        async def _run():
            with (
                patch.object(app_module, "predict_base_bytes", return_value=None),
                patch.object(app_module, "_gpu_occupied", return_value=False),
                patch("paramem.server.app.torch.cuda.is_available", return_value=True),
                patch.object(app_module, "_load_model_into_state"),
                patch.object(app_module, "_build_runtime_components"),
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
                    assert app_module._state.get("mode") == "local", (
                        "this run's own healthy boot must compute mode='local'; "
                        "a lingering 'cloud-only' proves the hygiene reset did "
                        "not run ahead of the mode-write guard"
                    )
                    assert app_module._state.get("cloud_only_reason") is None

        try:
            asyncio.run(_run())
        finally:
            for key, val in saved_state.items():
                if val is None:
                    app_module._state.pop(key, None)
                else:
                    app_module._state[key] = val


# ---------------------------------------------------------------------------
# TestBootDegradeIsNarrowedToVramAndFatalCuda — the boot call to
# _load_model_into_state degrades to cloud-only ONLY for VramExhausted
# (cold-tier creation is a VRAM allocation) and the fatal-CUDA path; every
# other exception -- in particular a config-vs-disk refusal, which is a
# RuntimeError raised from inside _load_model_into_state -- must propagate
# and abort the boot loudly rather than being swallowed into a silent
# cloud-only degrade.
# ---------------------------------------------------------------------------


class TestBootDegradeIsNarrowedToVramAndFatalCuda:
    def test_vram_exhausted_during_tier_creation_degrades_to_cloud_only(self, tmp_path):
        """VramExhausted from _load_model_into_state (cold-tier creation
        overflowing VRAM) must degrade the boot to cloud-only with
        cloud_only_reason='insufficient_vram' -- never propagate and kill
        the unit."""
        import pytest

        from paramem.server import app as app_module
        from paramem.server.config import PathsConfig, ServerConfig
        from paramem.utils.vram_guard import VramExhausted

        class _Sentinel(Exception):
            pass

        config = ServerConfig(model_name="mistral")
        config.cloud_only = False
        root = tmp_path / "data"
        config.paths = PathsConfig(data=root, sessions=root / "sessions", debug=root / "debug")

        saved_state = {
            key: app_module._state.get(key)
            for key in ("config", "cloud_only_startup", "defer_model")
        }
        app_module._state["config"] = config
        app_module._state["cloud_only_startup"] = False
        app_module._state["defer_model"] = False

        try:
            with (
                patch.object(app_module, "predict_base_bytes", return_value=None),
                patch.object(app_module, "_gpu_occupied", return_value=False),
                patch("paramem.server.app.torch.cuda.is_available", return_value=True),
                patch.object(
                    app_module,
                    "_load_model_into_state",
                    side_effect=VramExhausted("tier creation"),
                ),
                patch.object(app_module, "_release_base_model_in_process"),
                patch.object(
                    app_module,
                    "_build_runtime_components",
                    side_effect=_Sentinel("short-circuit after degrade"),
                ),
            ):

                async def _run():
                    async with app_module.lifespan(app_module.app):
                        pass

                with pytest.raises(_Sentinel):
                    asyncio.run(_run())

            # Assert while the degrade's writes are still live -- the
            # restore below intentionally resets cloud_only_reason/model so
            # this test cannot leak boot-degrade state into a later test.
            assert app_module._state.get("cloud_only_reason") == "insufficient_vram"
            assert app_module._state.get("model") is None
        finally:
            for key, val in saved_state.items():
                if val is None:
                    app_module._state.pop(key, None)
                else:
                    app_module._state[key] = val
            app_module._state.pop("cloud_only_reason", None)
            app_module._state.pop("model", None)
            app_module._state.pop("topology_assessment", None)
            app_module._state.pop("usable_ceiling_bytes", None)
            app_module._state.pop("device_total_memory_bytes", None)

    def test_a_config_vs_disk_refusal_is_not_swallowed_into_cloud_only(self, tmp_path):
        """A ``ConfigStoreMismatch`` raised from _load_model_into_state is
        neither VramExhausted nor a fatal-CUDA fault -- it must propagate out
        of lifespan and abort the boot, never be swallowed into a silent
        cloud-only degrade. ``ConfigStoreMismatch`` is a ``RuntimeError``
        subclass, so ``pytest.raises(RuntimeError, ...)`` still holds."""
        import pytest

        from paramem.server import app as app_module
        from paramem.server.config import PathsConfig, ServerConfig
        from paramem.server.config_store_validator import ConfigStoreMismatch

        config = ServerConfig(model_name="mistral")
        config.cloud_only = False
        root = tmp_path / "data"
        config.paths = PathsConfig(data=root, sessions=root / "sessions", debug=root / "debug")

        saved_state = {
            key: app_module._state.get(key)
            for key in ("config", "cloud_only_startup", "defer_model", "cloud_only_reason", "mode")
        }
        app_module._state["config"] = config
        app_module._state["cloud_only_startup"] = False
        app_module._state["defer_model"] = False

        refusal_message = (
            "adapters.episodic.enabled=false but 1 interim slot(s) still exist "
            "under adapter_dir/episodic"
        )

        try:
            with (
                patch.object(app_module, "predict_base_bytes", return_value=None),
                patch.object(app_module, "_gpu_occupied", return_value=False),
                patch("paramem.server.app.torch.cuda.is_available", return_value=True),
                patch.object(
                    app_module,
                    "_load_model_into_state",
                    side_effect=ConfigStoreMismatch(
                        refusal_message, check="interim_ring_without_episodic"
                    ),
                ),
            ):

                async def _run():
                    async with app_module.lifespan(app_module.app):
                        pass

                with pytest.raises(RuntimeError, match="interim slot"):
                    asyncio.run(_run())
        finally:
            for key, val in saved_state.items():
                if val is None:
                    app_module._state.pop(key, None)
                else:
                    app_module._state[key] = val


# ---------------------------------------------------------------------------
# TestShutdownGpuLockRelease — lifespan shutdown releases the base model
# under a bounded gpu_lock_sync wait
# ---------------------------------------------------------------------------


class TestShutdownGpuLockRelease:
    """Lifespan shutdown's base-model release must run under
    ``gpu_lock_sync`` with a bounded wait: locked when the lock is free,
    logged-and-unlocked when a holder outlasts the wait — never a bare
    unlocked release racing a legitimate in-fold lock holder, and never an
    indefinite block."""

    def _make_config(self, tmp_path):
        from paramem.server.config import PathsConfig, ServerConfig, STTConfig, TTSConfig

        config = ServerConfig(model_name="mistral")
        config.cloud_only = True
        config.stt = STTConfig(enabled=False)
        config.tts = TTSConfig(enabled=False)
        root = tmp_path / "data"
        config.paths = PathsConfig(data=root, sessions=root / "sessions", debug=root / "debug")
        return config

    def _seeded_keys(self):
        return (
            "config",
            "cloud_only_startup",
            "defer_model",
            "boot_completion_task",
            "base_swap_task",
            "reclaim_task",
            "consolidation_loop",
        )

    def test_shutdown_with_lock_free_releases_under_lock_no_error(self, tmp_path, caplog):
        """No contention: the release runs while the lock is held BY
        shutdown itself, and no lock-timeout ERROR is logged."""
        import logging

        import paramem.server.app as app_module
        from paramem.server.gpu_lock import gpu_lock_is_held

        caplog.set_level(logging.ERROR, logger="paramem.server.app")

        config = self._make_config(tmp_path)
        saved_state = {key: app_module._state.get(key) for key in self._seeded_keys()}
        app_module._state["config"] = config
        app_module._state["cloud_only_startup"] = True
        app_module._state["defer_model"] = False
        app_module._state["boot_completion_task"] = None
        app_module._state["base_swap_task"] = None
        app_module._state["reclaim_task"] = None
        app_module._state["consolidation_loop"] = None

        held_during_release: list[bool] = []

        def _record_lock_state():
            held_during_release.append(gpu_lock_is_held())

        async def _run():
            with (
                patch.object(app_module, "predict_base_bytes", return_value=None),
                patch.object(app_module, "_gpu_occupied", return_value=False),
                patch.object(app_module, "_build_runtime_components"),
                patch.object(app_module, "_arm_active_store_migration", return_value=False),
                patch.object(
                    app_module,
                    "_release_base_model_in_process",
                    side_effect=_record_lock_state,
                ),
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

        assert held_during_release == [True], (
            "_release_base_model_in_process must run while shutdown holds the GPU lock"
        )
        assert not any("could not acquire GPU lock" in r.message for r in caplog.records), (
            "no lock-timeout ERROR expected when the lock was free"
        )
        assert not gpu_lock_is_held(), "the lock must be released after shutdown completes"

    def test_shutdown_with_lock_held_logs_error_and_still_releases(self, tmp_path, caplog):
        """A holder thread outlasting the bounded wait: shutdown logs an
        ERROR naming the timeout, then still calls the release (unlocked) —
        shutdown must complete rather than block indefinitely."""
        import logging
        import threading

        import paramem.server.app as app_module
        from paramem.server.gpu_lock import _gpu_thread_lock

        caplog.set_level(logging.ERROR, logger="paramem.server.app")

        config = self._make_config(tmp_path)
        saved_state = {key: app_module._state.get(key) for key in self._seeded_keys()}
        app_module._state["config"] = config
        app_module._state["cloud_only_startup"] = True
        app_module._state["defer_model"] = False
        app_module._state["boot_completion_task"] = None
        app_module._state["base_swap_task"] = None
        app_module._state["reclaim_task"] = None
        app_module._state["consolidation_loop"] = None

        release_calls: list[bool] = []
        release_event = threading.Event()
        acquired_event = threading.Event()

        def _hold_lock():
            _gpu_thread_lock.acquire()
            acquired_event.set()
            release_event.wait(timeout=5.0)
            _gpu_thread_lock.release()

        holder = threading.Thread(target=_hold_lock, daemon=True)
        holder.start()
        try:
            # A failed wait here must still fall through to the finally
            # block so release_event is set and the daemon holder does not
            # keep the real lock past this test.
            assert acquired_event.wait(timeout=5.0), "holder thread failed to acquire the lock"

            async def _run():
                with (
                    patch.object(app_module, "predict_base_bytes", return_value=None),
                    patch.object(app_module, "_gpu_occupied", return_value=False),
                    patch.object(app_module, "_build_runtime_components"),
                    patch.object(app_module, "_arm_active_store_migration", return_value=False),
                    patch.object(
                        app_module,
                        "_release_base_model_in_process",
                        side_effect=lambda: release_calls.append(True),
                    ),
                    patch.object(app_module, "safe_empty_cache"),
                    patch.object(app_module, "_reconcile_scheduling_timers"),
                    patch.object(app_module, "_create_backup"),
                    patch.object(app_module, "_dispatch_consolidation"),
                    # Bounded — well under the holder's 5s self-release —
                    # so this test does not wait the full production timeout.
                    patch.object(app_module, "_SHUTDOWN_GPU_LOCK_TIMEOUT_S", 0.2),
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

            asyncio.run(_run())
        finally:
            release_event.set()
            holder.join(timeout=5.0)
            for key, val in saved_state.items():
                if val is None:
                    app_module._state.pop(key, None)
                else:
                    app_module._state[key] = val

        assert release_calls == [True], (
            "_release_base_model_in_process must still run after the lock-acquire timeout"
        )
        error_records = [
            r
            for r in caplog.records
            if r.levelno >= logging.ERROR and "could not acquire GPU lock" in r.message
        ]
        assert error_records, "expected an ERROR log naming the lock-acquire timeout"


# ---------------------------------------------------------------------------
# TestIdleFiringResumesPendingEventAtNoCadenceCost — the abort-then-resume
# arc at the dispatch boundary: a pending interim event, found by an idle
# firing while the server is idle, resumes without consuming a cadence
# mark. No training runs — the executor hop is a spy, matching
# tests/server/test_consolidate_dispatch.py's own arbitrator harness.
# ---------------------------------------------------------------------------


class TestIdleFiringResumesPendingEventAtNoCadenceCost:
    def test_idle_dispatch_resumes_a_pending_interim_event_without_a_cadence_write(
        self, tmp_path, monkeypatch
    ) -> None:
        """A pending interim event, found on disk by an IDLE-reasoned
        dispatch while the server has never used the model, is resumed
        through the real arbitrator (``started_resume``, one submission to
        the executor spy) and the schedule stamp file's cadence mark is
        byte-for-byte unchanged — the idle watch's own resume earns no
        cadence credit, only a cadence-firing resume does.
        """
        import time

        import paramem.server.app as app_module
        from paramem.server.consolidation_choice import DispatchReason
        from paramem.server.schedule_state import ScheduleMarks, read_marks, write_marks
        from tests.server._state_builders import _write_pending_ledger
        from tests.server.test_consolidate_dispatch import _ExecutorSpy, _make_arbitrator_state

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        _write_pending_ledger(tmp_path, event="interim")
        # Idle since boot: the clock has never been stamped.
        state["last_model_use_monotonic"] = None

        state_dir = tmp_path / "state"
        seeded = ScheduleMarks(
            last_cadence_mark_epoch=time.time() - 3600, last_full_start_epoch=None
        )
        write_marks(state_dir, seeded)
        before = read_marks(state_dir)

        spy = _ExecutorSpy()
        state["event_loop"] = spy.loop
        monkeypatch.setattr(app_module, "_state", state)
        monkeypatch.setattr(app_module, "_retro_claim_orphan_sessions", lambda: 0)

        status, _action, _choice = app_module._arbitrate_consolidation(
            app_module.ConsolidationAction.AUTO, reason=DispatchReason.IDLE
        )

        assert status == "started_resume"
        assert spy.call_count == 1

        after = read_marks(state_dir)
        assert after.last_cadence_mark_epoch == before.last_cadence_mark_epoch
