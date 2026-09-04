"""``_watch_for_idle`` / ``_arm_idle_watch`` — the in-process idle-firing
watch's own lifecycle: arming, the clock rule that decides when it fires,
and the one policy that reads every answer the arbitrator can hand it.

Every dispatch the watch itself makes is stood in for by a recorder that
returns a canned ``(status, action, choice)`` triple — the arbitrator's own
resolution of each status is covered by ``tests/server/test_consolidate_dispatch.py``
and ``tests/test_consolidation.py``; this module owns only what the watch
task does with the answer.

No real multi-second sleeps: every config here carries a small
``session.idle_timeout_minutes`` (a plain float — the field is read, never
validated, off a ``MagicMock``), and every timing assertion carries a
tolerance sized for scheduler jitter rather than for exactness.
"""

from __future__ import annotations

import asyncio
import time
import warnings
from contextlib import suppress
from unittest.mock import MagicMock, patch

import pytest

import paramem.server.app as app_module
from paramem.server.consolidation_choice import ConsolidationChoice, DispatchReason

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

#: A wait long enough that a watch armed under it never reaches its first
#: dispatch within one of these tests — used by tests that only need a
#: live, sleeping task to assert against (arming behaviour, not the clock
#: rule or the dispatch policy).
LONG_IDLE_TIMEOUT_S = 5.0

#: A short wait used by tests that measure the clock rule itself, or that
#: want the watch's first pass to reach a dispatch quickly.
SHORT_IDLE_TIMEOUT_S = 0.3


def _idle_config(idle_timeout_s: float) -> MagicMock:
    cfg = MagicMock()
    cfg.session.idle_timeout_minutes = idle_timeout_s / 60
    return cfg


def _sleeping_state(idle_timeout_s: float = LONG_IDLE_TIMEOUT_S) -> dict:
    """``_state`` overrides for a watch that must stay pending for the
    whole test: the clock has never been stamped and the timeout is long,
    so the watch's own first pass never leaves its initial sleep."""
    return {
        "config": _idle_config(idle_timeout_s),
        "background_trainer": None,
        "last_model_use_monotonic": None,
        "idle_watch_task": None,
    }


def _already_due_state(idle_timeout_s: float = SHORT_IDLE_TIMEOUT_S) -> dict:
    """``_state`` overrides for a watch whose first pass reaches a dispatch
    immediately — the clock was stamped well past the timeout already."""
    return {
        "config": _idle_config(idle_timeout_s),
        "background_trainer": None,
        "last_model_use_monotonic": time.monotonic() - (idle_timeout_s + 10),
        "idle_watch_task": None,
    }


async def _cancel_and_reap(task: "asyncio.Task | None") -> None:
    if task is None:
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


def _idle_owned_deferral(seconds: int = 5) -> ConsolidationChoice:
    return ConsolidationChoice(
        run=None,
        resume_pending=False,
        status="deferred_resume_waiting",
        consumes_cadence_mark=False,
        starts_full_fold=False,
        next_opportunity_seconds=seconds,
        next_opportunity_reason=DispatchReason.IDLE,
    )


def _timer_owned_deferral(seconds: int = 5) -> ConsolidationChoice:
    return ConsolidationChoice(
        run=None,
        resume_pending=False,
        status="deferred_resume_waiting",
        consumes_cadence_mark=False,
        starts_full_fold=False,
        next_opportunity_seconds=seconds,
        next_opportunity_reason=DispatchReason.TIMER,
    )


# ---------------------------------------------------------------------------
# Arming — where the watch is started, and the aliveness invariants
# _arm_idle_watch keeps.
# ---------------------------------------------------------------------------


class TestArming:
    def test_abort_site_leaves_a_live_task_in_the_slot(self) -> None:
        """``_abort_background_training_for_inference`` arms the watch as its
        last step; the slot holds a live (not-done) task right after."""

        async def _go() -> None:
            with patch.dict(app_module._state, _sleeping_state(), clear=False):
                app_module._abort_background_training_for_inference()
                task = app_module._state.get("idle_watch_task")
                assert task is not None
                await asyncio.sleep(0)
                assert not task.done()
                await _cancel_and_reap(task)

        asyncio.run(_go())

    def test_second_arming_while_the_task_runs_starts_no_second_task(self) -> None:
        """Arming twice while the first watch is still alive returns the
        same task object both times — no second task is created."""

        async def _go() -> None:
            with patch.dict(app_module._state, _sleeping_state(), clear=False):
                app_module._arm_idle_watch()
                first = app_module._state.get("idle_watch_task")
                app_module._arm_idle_watch()
                second = app_module._state.get("idle_watch_task")

                assert first is not None
                assert second is first
                await _cancel_and_reap(first)

        asyncio.run(_go())

    def test_arming_after_the_task_exits_starts_a_fresh_task(self) -> None:
        """Once the watch ends (the slot returns to ``None`` via the shared
        done-callback), arming again creates a new, different task."""

        def _recorder(action, *, reason, spec=None):
            return "noop_nothing_pending", action, None

        async def _go() -> None:
            with (
                patch.dict(app_module._state, _already_due_state(), clear=False),
                patch.object(app_module, "_arbitrate_consolidation", _recorder),
            ):
                app_module._arm_idle_watch()
                first = app_module._state.get("idle_watch_task")
                await asyncio.wait_for(first, timeout=2.0)
                assert app_module._state.get("idle_watch_task") is None

                app_module._arm_idle_watch()
                second = app_module._state.get("idle_watch_task")
                assert second is not None
                assert second is not first
                await _cancel_and_reap(second)

        asyncio.run(_go())

    def test_cancelled_at_shutdown_raises_no_pending_task_warning(self) -> None:
        """Shutdown's ``.cancel()`` (fire-and-forget, no await — see
        ``app.py``'s shutdown block) leaves no "Task was destroyed but it is
        pending" warning once the event loop finishes tearing down."""

        async def _go() -> None:
            with patch.dict(app_module._state, _sleeping_state(), clear=False):
                app_module._arm_idle_watch()
                task = app_module._state.get("idle_watch_task")
                await asyncio.sleep(0)
                assert task is not None and not task.done()
                # Mirrors the shutdown block exactly: cancel, do not await.
                task.cancel()

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            asyncio.run(_go())


# ---------------------------------------------------------------------------
# The clock rule: when the watch fires relative to the last model use and
# its own arming moment.
# ---------------------------------------------------------------------------


class TestClockRuleTiming:
    """Re-derives elapsed time explicitly (arm instant captured outside the
    coroutine boundary), so every assertion compares two ``time.monotonic()``
    readings taken in the same test — no reliance on an implicit baseline.
    """

    TIMEOUT_S = 0.4

    @staticmethod
    def _tolerance(expected: float) -> float:
        return max(0.15, 0.35 * expected)

    def _run(self, *, last_use_offset: "float | None") -> tuple[float, float]:
        """Arm with ``last_model_use_monotonic`` set to ``arm_time -
        last_use_offset`` (``None`` -> never stamped), and return
        ``(arm_time, dispatch_time)``.
        """
        call_times: list[float] = []
        event = asyncio.Event()
        timing: dict[str, float] = {}

        def _recorder(action, *, reason, spec=None):
            call_times.append(time.monotonic())
            event.set()
            return "noop_nothing_pending", action, None

        async def _go() -> None:
            arm_time = time.monotonic()
            last_use = None if last_use_offset is None else arm_time - last_use_offset
            state = {
                "config": _idle_config(self.TIMEOUT_S),
                "background_trainer": None,
                "last_model_use_monotonic": last_use,
                "idle_watch_task": None,
            }
            timing["arm"] = arm_time
            with (
                patch.dict(app_module._state, state, clear=False),
                patch.object(app_module, "_arbitrate_consolidation", _recorder),
            ):
                app_module._arm_idle_watch()
                task = app_module._state.get("idle_watch_task")
                await asyncio.wait_for(event.wait(), timeout=2.0)
                await asyncio.wait_for(task, timeout=2.0)

        asyncio.run(_go())
        assert len(call_times) == 1
        return timing["arm"], call_times[0]

    def test_fires_about_half_the_timeout_after_arming(self) -> None:
        arm_time, dispatch_time = self._run(last_use_offset=0.5 * self.TIMEOUT_S)
        expected = 0.5 * self.TIMEOUT_S
        elapsed = dispatch_time - arm_time
        assert abs(elapsed - expected) < self._tolerance(expected), (
            f"expected ~{expected:.3f}s, measured {elapsed:.3f}s"
        )

    def test_fires_about_a_tenth_of_the_timeout_after_arming(self) -> None:
        arm_time, dispatch_time = self._run(last_use_offset=0.9 * self.TIMEOUT_S)
        expected = 0.1 * self.TIMEOUT_S
        elapsed = dispatch_time - arm_time
        assert abs(elapsed - expected) < self._tolerance(expected), (
            f"expected ~{expected:.3f}s, measured {elapsed:.3f}s"
        )

    def test_fires_about_one_timeout_after_arming_when_the_clock_was_never_stamped(self) -> None:
        arm_time, dispatch_time = self._run(last_use_offset=None)
        expected = self.TIMEOUT_S
        elapsed = dispatch_time - arm_time
        assert abs(elapsed - expected) < self._tolerance(expected), (
            f"expected ~{expected:.3f}s, measured {elapsed:.3f}s"
        )

    def test_a_restamp_mid_wait_pushes_the_firing_out(self) -> None:
        """A model use landing partway through the wait re-anchors the
        firing to the new stamp, one full timeout later — later than an
        unrestamped arming would have fired, not sooner."""
        call_times: list[float] = []
        event = asyncio.Event()
        restamp_offset = 0.4 * self.TIMEOUT_S
        timing: dict[str, float] = {}

        def _recorder(action, *, reason, spec=None):
            call_times.append(time.monotonic())
            event.set()
            return "noop_nothing_pending", action, None

        async def _go() -> None:
            arm_time = time.monotonic()
            state = {
                "config": _idle_config(self.TIMEOUT_S),
                "background_trainer": None,
                "last_model_use_monotonic": None,
                "idle_watch_task": None,
            }
            timing["arm"] = arm_time
            with (
                patch.dict(app_module._state, state, clear=False),
                patch.object(app_module, "_arbitrate_consolidation", _recorder),
            ):
                app_module._arm_idle_watch()
                task = app_module._state.get("idle_watch_task")

                await asyncio.sleep(restamp_offset)
                restamp_time = time.monotonic()
                app_module._state["last_model_use_monotonic"] = restamp_time
                timing["restamp"] = restamp_time

                await asyncio.wait_for(event.wait(), timeout=2.0)
                await asyncio.wait_for(task, timeout=2.0)

        asyncio.run(_go())

        assert len(call_times) == 1
        dispatch_time = call_times[0]
        elapsed_from_restamp = dispatch_time - timing["restamp"]
        tol = max(0.15, 0.35 * self.TIMEOUT_S)
        assert abs(elapsed_from_restamp - self.TIMEOUT_S) < tol, (
            f"expected ~{self.TIMEOUT_S:.3f}s after the restamp, "
            f"measured {elapsed_from_restamp:.3f}s"
        )

        elapsed_from_arm = dispatch_time - timing["arm"]
        assert elapsed_from_arm > self.TIMEOUT_S + 0.05, (
            "a restamp mid-wait must push the firing out past what an "
            f"unrestamped arming would have fired at ({self.TIMEOUT_S:.3f}s); "
            f"measured {elapsed_from_arm:.3f}s"
        )


# ---------------------------------------------------------------------------
# The one policy: every answer the watch's dispatch can get back, and
# whether the task is still alive afterwards.
# ---------------------------------------------------------------------------

_ENDING_CASES = [
    pytest.param("started_resume", None, id="started_resume"),
    pytest.param("noop_nothing_pending", None, id="noop_nothing_pending"),
    pytest.param("deferred_event_unreadable", None, id="deferred_event_unreadable"),
    pytest.param("deferred_schedule_unreadable", None, id="deferred_schedule_unreadable"),
    pytest.param(
        "deferred_resume_waiting",
        _timer_owned_deferral(),
        id="deferred_resume_waiting_owned_by_timer",
    ),
]

_SLEEPING_CASES = [
    # Stands for the five busy guards, which all share this shape.
    pytest.param("deferred_bg_training", None, id="deferred_bg_training"),
    pytest.param("deferred_model_in_use", None, id="deferred_model_in_use"),
    pytest.param(
        "deferred_resume_waiting",
        _idle_owned_deferral(),
        id="deferred_resume_waiting_owned_by_idle",
    ),
]


class TestOnePolicyPerAnswer:
    async def _dispatch_once(self, status: str, choice: "ConsolidationChoice | None") -> bool:
        """Arm the watch, let it dispatch once against a canned answer,
        and return whether the task was done shortly afterwards."""
        call_count = 0
        event = asyncio.Event()

        def _recorder(action, *, reason, spec=None):
            nonlocal call_count
            call_count += 1
            event.set()
            return status, action, choice

        with (
            patch.dict(app_module._state, _already_due_state(), clear=False),
            patch.object(app_module, "_arbitrate_consolidation", _recorder),
        ):
            app_module._arm_idle_watch()
            task = app_module._state.get("idle_watch_task")
            await asyncio.wait_for(event.wait(), timeout=2.0)
            # Give the loop a couple of turns to act on the answer (return,
            # or fall through into the next sleep) before we look.
            for _ in range(3):
                await asyncio.sleep(0)
            still_alive = not task.done()
            await _cancel_and_reap(task)

        assert call_count == 1
        return still_alive

    @pytest.mark.parametrize("status,choice", _ENDING_CASES)
    def test_ends_the_watch(self, status, choice) -> None:
        still_alive = asyncio.run(self._dispatch_once(status, choice))
        assert not still_alive, f"{status} must end the watch"

    @pytest.mark.parametrize("status,choice", _SLEEPING_CASES)
    def test_leaves_the_watch_sleeping(self, status, choice) -> None:
        still_alive = asyncio.run(self._dispatch_once(status, choice))
        assert still_alive, f"{status} must leave the watch sleeping, waiting to ask again"


# ---------------------------------------------------------------------------
# Boot completion: arms the watch and dispatches BOOT with no dueness read
# of its own.
# ---------------------------------------------------------------------------


def _make_boot_config(tmp_path, *, refresh_cadence="", backup_schedule="off"):
    cfg = MagicMock()
    cfg.consolidation.refresh_cadence = refresh_cadence
    cfg.security.backups.schedule = backup_schedule
    cfg.security.backups.artifacts = ["config", "graph", "registry"]
    cfg.paths.data = tmp_path
    cfg.session.idle_timeout_minutes = 60.0
    return cfg


class TestBootCompletionArmsTheWatchWithoutItsOwnDuenessRead:
    def test_dispatches_boot_reason_and_reads_no_stamp_file_itself(self, tmp_path) -> None:
        import paramem.server.schedule_state as schedule_state_module

        read_marks_calls: list[tuple] = []

        def _fake_read_marks(*args, **kwargs):
            read_marks_calls.append((args, kwargs))
            from paramem.server.schedule_state import ScheduleMarks

            return ScheduleMarks(None, None)

        dispatch_calls: list[dict] = []

        def _fake_dispatch(action, *, reason, spec=None):
            dispatch_calls.append({"action": action, "reason": reason, "spec": spec})
            return "started_resume", action

        config = _make_boot_config(tmp_path)

        async def _go() -> None:
            with (
                patch.dict(
                    app_module._state,
                    {"config": config, "base_swap_task": None},
                    clear=False,
                ),
                patch.object(app_module, "_dispatch_consolidation", _fake_dispatch),
                patch.object(app_module, "_reconcile_scheduling_timers", MagicMock()),
                patch.object(app_module, "_create_backup", MagicMock()),
                patch.object(schedule_state_module, "read_marks", _fake_read_marks),
            ):
                await app_module._run_boot_completion_tasks()
                task = app_module._state.get("idle_watch_task")
                assert task is not None and not task.done()
                await _cancel_and_reap(task)

        asyncio.run(_go())

        assert len(dispatch_calls) == 1
        assert dispatch_calls[0]["action"] is app_module.ConsolidationAction.AUTO
        assert dispatch_calls[0]["reason"] is DispatchReason.BOOT
        assert read_marks_calls == [], (
            "the boot task must carry no dueness read of its own — "
            f"schedule_state.read_marks was called: {read_marks_calls}"
        )

    def test_arms_a_live_watch_even_when_the_dispatch_raises(self, tmp_path) -> None:
        def _raising_dispatch(action, *, reason, spec=None):
            raise RuntimeError("boom")

        config = _make_boot_config(tmp_path)

        async def _go() -> None:
            with (
                patch.dict(
                    app_module._state,
                    {"config": config, "base_swap_task": None},
                    clear=False,
                ),
                patch.object(app_module, "_dispatch_consolidation", _raising_dispatch),
                patch.object(app_module, "_reconcile_scheduling_timers", MagicMock()),
                patch.object(app_module, "_create_backup", MagicMock()),
            ):
                # _run_boot_completion_tasks isolates each step in its own
                # try/except — a raising dispatch must not stop the finally
                # from arming the watch, nor propagate out of this call.
                await app_module._run_boot_completion_tasks()
                task = app_module._state.get("idle_watch_task")
                assert task is not None and not task.done()
                await _cancel_and_reap(task)

        asyncio.run(_go())


# ---------------------------------------------------------------------------
# A window-start firing that finds the server busy defers owned by IDLE and
# arms the watch — the resume happens once the conversation ends, not at
# the next window opening.
# ---------------------------------------------------------------------------


class TestWindowStartFiringFindsServerBusy:
    def test_busy_server_defers_owned_by_idle_and_arms_the_watch(
        self, tmp_path, monkeypatch
    ) -> None:
        from datetime import datetime, timedelta

        from tests.server._state_builders import _write_pending_ledger
        from tests.server.test_consolidate_dispatch import _make_arbitrator_state

        now_dt = datetime.now()
        window = (
            f"{(now_dt - timedelta(minutes=30)).strftime('%H:%M')}-"
            f"{(now_dt + timedelta(minutes=30)).strftime('%H:%M')}"
        )

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        state["config"].consolidation.interim_resume = window
        _write_pending_ledger(tmp_path, event="interim")
        # Busy: model used well inside the debounce window's outer bound
        # (idle_timeout_minutes=10) but past training_idle_debounce_s (30s)
        # — the decider is reached, not the debounce.
        state["last_model_use_monotonic"] = time.monotonic() - 60

        monkeypatch.setattr(app_module, "_state", state)
        arm_mock = MagicMock()
        monkeypatch.setattr(app_module, "_arm_idle_watch", arm_mock)

        status, _action, choice = app_module._arbitrate_consolidation(
            app_module.ConsolidationAction.AUTO, reason=DispatchReason.TIMER
        )

        assert status == "deferred_resume_waiting"
        assert choice.next_opportunity_reason is DispatchReason.IDLE
        arm_mock.assert_called_once()


# ---------------------------------------------------------------------------
# Static pin: the idle clock's write sites. Every model-using door writes
# the same one stamp, at the moment it starts using the model — an AST scan
# in the style of tests/server/test_consolidation_envelope_structure.py.
# ---------------------------------------------------------------------------


class TestIdleClockStampSites:
    _EXPECTED_FUNCTIONS = frozenset(
        {
            "_run_chat_turn",
            "debug_probe",
            "debug_recall",
            "_arbitrate_consolidation",
        }
    )

    def _stamp_sites(self) -> list[tuple[int, str]]:
        import ast
        import inspect

        source = inspect.getsource(app_module)
        tree = ast.parse(source)
        hits: list[tuple[int, str]] = []

        class _Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.stack: list[str] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

            def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
                for target in node.targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "_state"
                        and self._subscript_key(target) == "last_model_use_monotonic"
                    ):
                        hits.append((node.lineno, self.stack[-1] if self.stack else "<module>"))
                self.generic_visit(node)

            @staticmethod
            def _subscript_key(node: ast.Subscript) -> "str | None":
                sl = node.slice
                if isinstance(sl, ast.Constant):
                    return sl.value
                return None

        _Visitor().visit(tree)
        return hits

    def test_stamp_sites_are_exactly_the_documented_four_functions(self) -> None:
        hits = self._stamp_sites()
        assert hits, "no last_model_use_monotonic stamp sites found at all"
        owning_functions = {name for _lineno, name in hits}
        assert owning_functions == self._EXPECTED_FUNCTIONS, (
            f"idle clock stamp sites moved: found {owning_functions}, "
            f"expected {self._EXPECTED_FUNCTIONS}"
        )

    def test_debug_probe_stamps_at_both_of_its_abort_call_sites(self) -> None:
        """/debug/probe has two branches (cloud-only, local) that each
        abort training ahead of using the model — both must stamp."""
        hits = self._stamp_sites()
        debug_probe_hits = [lineno for lineno, name in hits if name == "debug_probe"]
        assert len(debug_probe_hits) == 2, (
            f"expected exactly two stamp sites inside debug_probe; found {debug_probe_hits}"
        )
