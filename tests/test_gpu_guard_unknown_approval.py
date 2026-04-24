"""Tests for the server-classification and unknown-process-approval paths in
``experiments.utils.gpu_guard``.

Covers the cluster of bugs fixed together:

* **Safety gate** — no SIGTERM/SIGKILL to an unknown GPU occupant without an
  affirmative operator answer on an interactive TTY. The three non-TTY paths
  (``interactive=False``, or ``interactive=True + no TTY``) must raise
  ``GPUAcquireError`` before reaching the kill loop.
* **Server classification** — ``_identify_server_pids`` uses three sources
  (systemd MainPID, port listener, ``/proc/<pid>/cmdline``). When the primary
  sources miss, the cmdline backstop must still recognize the server so it
  routes to defer-to-cloud, not the kill path. This pins the regression where
  a pytest run was killed because lsof briefly misclassified ParaMem as
  unknown.
* **SIGKILL escalation** — a process that ignores SIGTERM gets SIGKILL after
  a 3 s grace, before the 30 s VRAM-wait times out.
"""

from __future__ import annotations

import signal
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from experiments.utils import gpu_guard
from experiments.utils.gpu_guard import (
    GPUAcquireError,
    _GPUGuard,
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
            patch.object(gpu_guard, "_systemd_main_pid", return_value=555),
            patch.object(gpu_guard, "_listener_pid", return_value=None),
            patch.object(gpu_guard, "_pid_cmdline", return_value=""),
        ):
            assert _identify_server_pids([555, 9999], port=8420) == [555]

    def test_port_listener_fallback_when_systemd_silent(self):
        """Systemd says 'not running' (None) — port-listener pid promotes."""
        with (
            patch.object(gpu_guard, "_systemd_main_pid", return_value=None),
            patch.object(gpu_guard, "_listener_pid", return_value=777),
            patch.object(gpu_guard, "_pid_cmdline", return_value=""),
        ):
            assert _identify_server_pids([777, 9999], port=8420) == [777]

    def test_cmdline_backstop_when_lsof_and_systemd_both_miss(self):
        """Root-cause regression: both primary sources fail, but cmdline
        matches ``paramem.server`` → server routes correctly (not unknown)."""

        def fake_cmdline(pid: int) -> str:
            return {
                222: "/path/to/python -m paramem.server.app --config s.yaml",
                9999: "/path/to/other-tool --arg",
            }[pid]

        with (
            patch.object(gpu_guard, "_systemd_main_pid", return_value=None),
            patch.object(gpu_guard, "_listener_pid", return_value=None),
            patch.object(gpu_guard, "_pid_cmdline", side_effect=fake_cmdline),
        ):
            assert _identify_server_pids([222, 9999], port=8420) == [222]

    def test_returns_only_pids_actually_on_gpu(self):
        """A systemd MainPID that isn't in ``external_pids`` is dropped."""
        with (
            patch.object(gpu_guard, "_systemd_main_pid", return_value=555),
            patch.object(gpu_guard, "_listener_pid", return_value=None),
            patch.object(gpu_guard, "_pid_cmdline", return_value=""),
        ):
            assert _identify_server_pids([9999], port=8420) == []

    def test_empty_when_all_sources_miss(self):
        with (
            patch.object(gpu_guard, "_systemd_main_pid", return_value=None),
            patch.object(gpu_guard, "_listener_pid", return_value=None),
            patch.object(gpu_guard, "_pid_cmdline", return_value="some other thing"),
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
    """Mock stack for a single unknown occupant (PID 9999).

    ``server_pids`` controls what ``_identify_server_pids`` returns; the default
    ``None`` is remapped to ``[]`` (unknown-only scenario). When a server pid
    is present the scenario is "mixed" — the test covers the handoff from the
    kill block to the defer-to-cloud block.

    ``sigterm_ignored=True`` simulates a process that refuses SIGTERM so the
    SIGKILL escalation fires.

    ``server_cloud_only=True`` shortcuts the server-release path (server is
    already in cloud-only) so the context manager can enter cleanly in the
    mixed case without having to simulate the SIGUSR1 round-trip.
    """
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

    # os.kill(pid, 0) behaviour: liveness probe. When sigterm_ignored is True
    # the process is still alive when probed; otherwise ProcessLookupError.
    def kill_side_effect(pid: int, sig: int):
        if sig == 0 and not sigterm_ignored:
            raise ProcessLookupError
        return None

    external = [9999] + list(server_pids)

    with (
        patch("experiments.utils.gpu_guard._SleepInhibitor"),
        patch("experiments.utils.gpu_guard.notify_ml"),
        patch("experiments.utils.gpu_guard.os.getpid", return_value=1000),
        patch("experiments.utils.gpu_guard._get_gpu_pids", return_value=external),
        patch("experiments.utils.gpu_guard._identify_server_pids", return_value=server_pids),
        patch("experiments.utils.gpu_guard.subprocess.run", return_value=ps_result),
        patch("experiments.utils.gpu_guard._wait_for_vram_clear", return_value=True),
        patch(
            "experiments.utils.gpu_guard._server_is_cloud_only",
            return_value=server_cloud_only,
        ),
        patch("experiments.utils.gpu_guard._send_release_signal"),
        patch("experiments.utils.gpu_guard.time.sleep"),
        patch("experiments.utils.gpu_guard.sys.stdin") as stdin_mock,
        patch("experiments.utils.gpu_guard.os.kill", side_effect=kill_side_effect) as kill_mock,
        input_ctx,
    ):
        stdin_mock.isatty.return_value = isatty
        guard = _GPUGuard(port=8420, interactive=interactive)
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
            assert _kill_calls(kill_mock) == []  # process died on SIGTERM

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
        with _mocked_guard(interactive=True, isatty=False, input_answer=None) as (guard, kill_mock):
            with pytest.raises(GPUAcquireError, match="no TTY"):
                guard.__enter__()
            assert _term_calls(kill_mock) == []

    def test_non_interactive_raises_and_no_kill(self):
        with _mocked_guard(interactive=False, isatty=True, input_answer=None) as (guard, kill_mock):
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
        """Regression: cmdline-matched server must defer (SIGUSR1), not kill.

        Scenario: GPU has only the ParaMem server (e.g. lsof missed it but
        cmdline catches it). No unknown processes. The guard must enter
        cleanly by deferring to cloud-only, never issuing SIGTERM.
        """
        with _mocked_guard(
            interactive=False,  # non-interactive: the bug kills without asking
            isatty=False,
            input_answer=None,
            server_pids=[555],  # server-only scenario (no unknowns)
            server_cloud_only=True,  # shortcut past SIGUSR1 round-trip
        ) as (guard, kill_mock):
            # Need to also suppress the unknown=[9999] default; patch gpu_pids
            # to only the server pid for this scenario.
            with patch("experiments.utils.gpu_guard._get_gpu_pids", return_value=[555]):
                with guard:
                    pass
            assert _term_calls(kill_mock) == []
            assert _kill_calls(kill_mock) == []

    def test_mixed_case_kills_unknown_and_defers_server(self):
        """Mixed: unknown PID 9999 + server PID 555. With approval we kill
        9999 and the server block takes over. Server already cloud-only so
        the test doesn't have to drive the SIGUSR1 wait."""
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
            # Server PID must never receive SIGTERM/SIGKILL.
            assert 555 not in _term_calls(kill_mock)
            assert 555 not in _kill_calls(kill_mock)
