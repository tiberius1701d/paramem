"""Tests for the server-classification and unknown-process-approval paths.

Updated to patch at the correct module locations after extraction:
  - gpu_guard._core for generic symbols (_get_gpu_pids, _wait_for_vram_clear,
    os.kill, time.sleep, sys.stdin, subprocess.run, os.getpid)
  - paramem.gpu_consumer for paramem-specific detection helpers
    (_systemd_main_pid, _listener_pid, _pid_cmdline, _is_paramem_server_cmdline,
    _identify_server_pids, _server_is_cloud_only, _send_release_signal)

Behavior covered:
  * Safety gate — no SIGTERM/SIGKILL without affirmative answer on interactive TTY.
  * SIGKILL escalation — process that ignores SIGTERM gets SIGKILL after 3 s.
  * Server-classification routing — cmdline-matched server routes to defer, not kill.
  * Mixed-case handoff — single _wait_for_vram_clear (no double call).
"""

from __future__ import annotations

import signal
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import gpu_guard._core as _core_module
import pytest
from gpu_guard._core import GPUAcquireError, _GPUGuard
from gpu_guard.inhibitor import NullInhibitor
from gpu_guard.notifier import NullNotifier

import paramem.gpu_consumer as _consumer_module
from paramem.gpu_consumer import (
    _identify_server_pids,
    _is_paramem_server_cmdline,
)

# ---------------------------------------------------------------------------
# _identify_server_pids — server classification with three detection sources
# ---------------------------------------------------------------------------


class TestIdentifyServerPids:
    def test_systemd_main_pid_is_primary_source(self):
        """When systemd reports MainPID, it is included regardless of port state."""
        with (
            patch.object(_consumer_module, "_systemd_main_pid", return_value=555),
            patch.object(_consumer_module, "_listener_pid", return_value=None),
            patch.object(_consumer_module, "_pid_cmdline", return_value=""),
        ):
            assert _identify_server_pids([555, 9999], port=8420) == [555]

    def test_port_listener_fallback_when_systemd_silent(self):
        """Systemd says 'not running' (None) — port-listener pid promotes."""
        with (
            patch.object(_consumer_module, "_systemd_main_pid", return_value=None),
            patch.object(_consumer_module, "_listener_pid", return_value=777),
            patch.object(_consumer_module, "_pid_cmdline", return_value=""),
        ):
            assert _identify_server_pids([777, 9999], port=8420) == [777]

    def test_cmdline_backstop_when_lsof_and_systemd_both_miss(self):
        """Both primary sources fail but cmdline matches → server routes correctly."""

        def fake_cmdline(pid: int) -> str:
            return {
                222: "/path/to/python -m paramem.server.app --config s.yaml",
                9999: "/path/to/other-tool --arg",
            }[pid]

        with (
            patch.object(_consumer_module, "_systemd_main_pid", return_value=None),
            patch.object(_consumer_module, "_listener_pid", return_value=None),
            patch.object(_consumer_module, "_pid_cmdline", side_effect=fake_cmdline),
        ):
            assert _identify_server_pids([222, 9999], port=8420) == [222]

    def test_returns_only_pids_actually_on_gpu(self):
        """A systemd MainPID not in external_pids is dropped."""
        with (
            patch.object(_consumer_module, "_systemd_main_pid", return_value=555),
            patch.object(_consumer_module, "_listener_pid", return_value=None),
            patch.object(_consumer_module, "_pid_cmdline", return_value=""),
        ):
            assert _identify_server_pids([9999], port=8420) == []

    def test_empty_when_all_sources_miss(self):
        with (
            patch.object(_consumer_module, "_systemd_main_pid", return_value=None),
            patch.object(_consumer_module, "_listener_pid", return_value=None),
            patch.object(_consumer_module, "_pid_cmdline", return_value="some other thing"),
        ):
            assert _identify_server_pids([9999], port=8420) == []


class TestCmdlineMarker:
    def test_matches_module_invocation(self):
        assert _is_paramem_server_cmdline("python -m paramem.server.app --config x")

    def test_matches_path_invocation(self):
        assert _is_paramem_server_cmdline("/env/bin/python paramem/server/app.py")

    def test_rejects_unrelated_process(self):
        assert not _is_paramem_server_cmdline("python some_experiment.py")
        assert not _is_paramem_server_cmdline("")


# ---------------------------------------------------------------------------
# _GPUGuard.__enter__ — safety gate + kill escalation + mixed-case handoff
# ---------------------------------------------------------------------------


def _null_consumer_that_finds(pids: list[int]):
    """Return a minimal consumer object that claims exactly ``pids``."""
    from gpu_guard.consumer import NullConsumer

    c = NullConsumer()
    c.find_pids = lambda candidates: [p for p in candidates if p in pids]  # type: ignore[method-assign]
    c.describe = lambda pid: f"test-consumer (PID {pid})"  # type: ignore[method-assign]
    return c


@contextmanager
def _mocked_guard(
    *,
    interactive: bool,
    isatty: bool,
    input_answer: str | None,
    server_pids: list[int] | None = None,
    sigterm_ignored: bool = False,
    server_cloud_only: bool = False,
):
    """Mock stack for a single unknown occupant (PID 9999)."""
    if server_pids is None:
        server_pids = []

    ps_result = MagicMock()
    ps_result.stdout = "9999 00:42 python runaway.py"
    ps_result.returncode = 0

    if input_answer is None:
        input_ctx = patch(
            "builtins.input",
            side_effect=AssertionError("input() must not be called on non-TTY path"),
        )
    else:
        input_ctx = patch("builtins.input", return_value=input_answer)

    def kill_side_effect(pid: int, sig: int) -> None:
        if sig == 0 and not sigterm_ignored:
            raise ProcessLookupError
        return None

    external = [9999] + list(server_pids)
    consumer = _null_consumer_that_finds(server_pids)
    consumer.is_idle = lambda: server_cloud_only  # type: ignore[method-assign]

    with (
        patch.object(_core_module, "_get_gpu_pids", return_value=external),
        patch.object(_core_module, "_wait_for_vram_clear", return_value=True),
        patch.object(_core_module, "read_all_live", return_value=[]),
        patch.object(_core_module, "write_holder"),
        patch.object(_core_module, "remove_holder"),
        patch("gpu_guard._core.subprocess.run", return_value=ps_result),
        patch("gpu_guard._core.time.sleep"),
        patch("gpu_guard._core.sys.stdin") as stdin_mock,
        patch("gpu_guard._core.os.kill", side_effect=kill_side_effect) as kill_mock,
        input_ctx,
    ):
        stdin_mock.isatty.return_value = isatty
        guard = _GPUGuard(
            priority=0,
            name="test",
            interactive=interactive,
            notifier=NullNotifier(),
            consumers=[consumer] if server_pids else [],
            inhibitor=NullInhibitor(),
        )
        yield guard, kill_mock


def _term_calls(kill_mock) -> list[int]:
    return [args[0] for args, _ in kill_mock.call_args_list if args[1] == signal.SIGTERM]


def _kill_calls(kill_mock) -> list[int]:
    return [args[0] for args, _ in kill_mock.call_args_list if args[1] == signal.SIGKILL]


class TestSafetyGate:
    def test_interactive_tty_yes_kills_and_enters(self):
        with _mocked_guard(interactive=True, isatty=True, input_answer="y") as (guard, kill_mock):
            with guard:
                pass
            assert _term_calls(kill_mock) == [9999]
            assert _kill_calls(kill_mock) == []

    def test_interactive_tty_empty_defaults_to_yes(self):
        with _mocked_guard(interactive=True, isatty=True, input_answer="") as (guard, kill_mock):
            with guard:
                pass
            assert _term_calls(kill_mock) == [9999]

    def test_interactive_tty_no_raises_and_no_kill(self):
        with _mocked_guard(interactive=True, isatty=True, input_answer="n") as (guard, kill_mock):
            with pytest.raises(GPUAcquireError, match="User declined"):
                guard.__enter__()
            assert _term_calls(kill_mock) == []
            assert _kill_calls(kill_mock) == []

    def test_interactive_no_tty_raises_and_no_kill(self):
        """Pins the original bug: interactive=True + no TTY must NOT kill."""
        with _mocked_guard(interactive=True, isatty=False, input_answer=None) as (
            guard,
            kill_mock,
        ):
            with pytest.raises(GPUAcquireError, match="no TTY"):
                guard.__enter__()
            assert _term_calls(kill_mock) == []

    def test_non_interactive_raises_and_no_kill(self):
        with _mocked_guard(interactive=False, isatty=True, input_answer=None) as (
            guard,
            kill_mock,
        ):
            with pytest.raises(GPUAcquireError, match="non-interactive"):
                guard.__enter__()
            assert _term_calls(kill_mock) == []

    def test_non_interactive_no_tty_raises_and_no_kill(self):
        with _mocked_guard(interactive=False, isatty=False, input_answer=None) as (
            guard,
            kill_mock,
        ):
            with pytest.raises(GPUAcquireError, match="non-interactive"):
                guard.__enter__()
            assert _term_calls(kill_mock) == []


class TestSigkillEscalation:
    def test_sigterm_ignored_triggers_sigkill(self):
        with _mocked_guard(
            interactive=True,
            isatty=True,
            input_answer="y",
            sigterm_ignored=True,
        ) as (guard, kill_mock):
            with guard:
                pass
            assert _term_calls(kill_mock) == [9999]
            assert _kill_calls(kill_mock) == [9999]

    def test_sigterm_honoured_skips_sigkill(self):
        with _mocked_guard(
            interactive=True,
            isatty=True,
            input_answer="y",
            sigterm_ignored=False,
        ) as (guard, kill_mock):
            with guard:
                pass
            assert _kill_calls(kill_mock) == []


class TestServerClassificationRouting:
    def test_recognized_server_routes_to_defer_not_kill(self):
        """Server-only scenario: consumer classifies GPU PID, no kill."""
        with (
            patch.object(_core_module, "_get_gpu_pids", return_value=[555]),
            patch.object(_core_module, "_wait_for_vram_clear", return_value=True),
            patch.object(_core_module, "read_all_live", return_value=[]),
            patch.object(_core_module, "write_holder"),
            patch.object(_core_module, "remove_holder"),
            patch("gpu_guard._core.sys.stdin") as stdin_mock,
            patch("gpu_guard._core.os.kill") as kill_mock,
        ):
            stdin_mock.isatty.return_value = False
            consumer = _null_consumer_that_finds([555])
            consumer.is_idle = lambda: True  # type: ignore[method-assign]
            guard = _GPUGuard(
                priority=0,
                name="test",
                interactive=False,
                notifier=NullNotifier(),
                consumers=[consumer],
                inhibitor=NullInhibitor(),
            )
            with guard:
                pass
            assert _term_calls(kill_mock) == []
            assert _kill_calls(kill_mock) == []

    def test_mixed_case_kills_unknown_and_defers_server(self):
        """Mixed: unknown PID 9999 + server PID 555. Kill 9999, defer server."""
        with _mocked_guard(
            interactive=True,
            isatty=True,
            input_answer="y",
            server_pids=[555],
            server_cloud_only=True,
        ) as (guard, kill_mock):
            with guard:
                pass
            assert _term_calls(kill_mock) == [9999]
            assert 555 not in _term_calls(kill_mock)
            assert 555 not in _kill_calls(kill_mock)
