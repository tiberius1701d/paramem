"""Tests for the PARAMEM_EXTRA_ARGS=--defer-model hold visibility layer.

Covers:
- ``_read_systemd_user_env`` parses ``systemctl --user show-environment`` output.
- ``_get_hold_state`` distinguishes inactive / legitimate / orphaned / unregistered
  holds and reports owner PID + liveness + age.
- ``_clear_hold_env`` invokes ``systemctl --user unset-environment`` with all
  three hold variables.

The auto-reclaim loop and the /gpu/force-local endpoint depend on these
helpers; they are the operator-visibility surface surfaced to /status and
``pstatus``.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from paramem.server.app import (
    _HOLD_ENV_VARS,
    _clear_hold_env,
    _get_hold_state,
    _pid_alive,
    _read_systemd_user_env,
    _unquote_systemd_value,
)
from paramem.utils.gpu_hold import format_cmd_hint


def _mk_completed(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


class TestUnquoteSystemdValue:
    def test_simple_value_passthrough(self):
        assert _unquote_systemd_value("simple_value") == "simple_value"

    def test_ansi_c_quoted_value_with_spaces_and_slashes(self):
        # systemd emits values with shell-special chars in $'...' form.
        raw = "$'python / paramem.server.app'"
        assert _unquote_systemd_value(raw) == "python / paramem.server.app"

    def test_ansi_c_quoted_value_with_backslash_escapes(self):
        # Tab (\t) and embedded quote (\') must unescape.
        raw = "$'a\\tb\\'c'"
        assert _unquote_systemd_value(raw) == "a\tb'c"

    def test_double_quoted_value(self):
        assert _unquote_systemd_value('"hello"') == "hello"


class TestReadSystemdUserEnv:
    def test_parses_key_value_lines(self):
        sample = (
            "PATH=/usr/bin\n"
            "PARAMEM_EXTRA_ARGS=--defer-model\n"
            "PARAMEM_HOLD_PID=12345\n"
            "PARAMEM_HOLD_STARTED_AT=1700000000\n"
            "PARAMEM_HOLD_CMD=$'python / paramem.server.app'\n"
        )
        with patch("paramem.server.app.subprocess.run", return_value=_mk_completed(sample)):
            env = _read_systemd_user_env()
        assert env["PARAMEM_EXTRA_ARGS"] == "--defer-model"
        assert env["PARAMEM_HOLD_PID"] == "12345"
        assert env["PARAMEM_HOLD_STARTED_AT"] == "1700000000"
        assert env["PATH"] == "/usr/bin"
        # ANSI-C quoting must be unwound so the hint is readable.
        assert env["PARAMEM_HOLD_CMD"] == "python / paramem.server.app"

    def test_handles_values_with_equals_sign(self):
        # Values themselves may contain "=" (config strings, URLs, etc.).
        sample = "FOO=a=b=c\nBAR=x\n"
        with patch("paramem.server.app.subprocess.run", return_value=_mk_completed(sample)):
            env = _read_systemd_user_env()
        assert env["FOO"] == "a=b=c"
        assert env["BAR"] == "x"

    def test_returns_empty_dict_on_timeout(self):
        with patch(
            "paramem.server.app.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="systemctl", timeout=5),
        ):
            assert _read_systemd_user_env() == {}

    def test_returns_empty_dict_when_systemctl_missing(self):
        with patch("paramem.server.app.subprocess.run", side_effect=FileNotFoundError):
            assert _read_systemd_user_env() == {}


class TestGetHoldState:
    def test_inactive_when_extra_args_missing(self):
        with patch("paramem.server.app._read_systemd_user_env", return_value={}):
            hold = _get_hold_state()
        assert hold == {
            "hold_active": False,
            "owner_pid": None,
            "owner_alive": None,
            "age_seconds": None,
            "owner_hint": None,
        }

    def test_inactive_when_extra_args_does_not_contain_defer_model(self):
        # Some other flag squatted on the env var — the hold semantics only
        # apply to --defer-model.
        env = {"PARAMEM_EXTRA_ARGS": "--other-flag"}
        with patch("paramem.server.app._read_systemd_user_env", return_value=env):
            hold = _get_hold_state()
        assert hold["hold_active"] is False

    def test_active_with_alive_holder(self):
        env = {
            "PARAMEM_EXTRA_ARGS": "--defer-model",
            "PARAMEM_HOLD_PID": "12345",
            "PARAMEM_HOLD_STARTED_AT": "1700000000",
            "PARAMEM_HOLD_CMD": "python / experiments.test8_large_scale",
        }
        with (
            patch("paramem.server.app._read_systemd_user_env", return_value=env),
            patch("paramem.server.app._pid_alive", return_value=True),
            patch("paramem.server.app.time.time", return_value=1700000042),
        ):
            hold = _get_hold_state()
        assert hold == {
            "hold_active": True,
            "owner_pid": 12345,
            "owner_alive": True,
            "age_seconds": 42,
            "owner_hint": "python / experiments.test8_large_scale",
        }

    def test_active_with_dead_holder(self):
        # SIGKILL case — registered PID but process is gone.  This is the
        # signal that tells auto-reclaim to warn + stop.
        env = {
            "PARAMEM_EXTRA_ARGS": "--defer-model",
            "PARAMEM_HOLD_PID": "99999",
            "PARAMEM_HOLD_STARTED_AT": "1700000000",
        }
        with (
            patch("paramem.server.app._read_systemd_user_env", return_value=env),
            patch("paramem.server.app._pid_alive", return_value=False),
            patch("paramem.server.app.time.time", return_value=1700000010),
        ):
            hold = _get_hold_state()
        assert hold["hold_active"] is True
        assert hold["owner_pid"] == 99999
        assert hold["owner_alive"] is False
        assert hold["age_seconds"] == 10

    def test_active_without_pid_registered(self):
        # Pre-stamp callers (legacy tresume) set PARAMEM_EXTRA_ARGS without
        # PARAMEM_HOLD_PID.  owner_alive stays None so the caller can
        # distinguish "no info" from "known dead".
        env = {"PARAMEM_EXTRA_ARGS": "--defer-model"}
        with patch("paramem.server.app._read_systemd_user_env", return_value=env):
            hold = _get_hold_state()
        assert hold["hold_active"] is True
        assert hold["owner_pid"] is None
        assert hold["owner_alive"] is None
        assert hold["age_seconds"] is None

    def test_non_integer_pid_is_ignored(self):
        env = {
            "PARAMEM_EXTRA_ARGS": "--defer-model",
            "PARAMEM_HOLD_PID": "not-a-pid",
        }
        with patch("paramem.server.app._read_systemd_user_env", return_value=env):
            hold = _get_hold_state()
        assert hold["owner_pid"] is None
        assert hold["owner_alive"] is None

    def test_negative_age_is_clamped_to_zero(self):
        # Clock skew between systemd set-env time and server read time
        # shouldn't render negative ages.
        env = {
            "PARAMEM_EXTRA_ARGS": "--defer-model",
            "PARAMEM_HOLD_STARTED_AT": "1700000100",
        }
        with (
            patch("paramem.server.app._read_systemd_user_env", return_value=env),
            patch("paramem.server.app.time.time", return_value=1700000000),
        ):
            hold = _get_hold_state()
        assert hold["age_seconds"] == 0


class TestPidAlive:
    def test_returns_true_for_own_pid(self):
        import os

        assert _pid_alive(os.getpid()) is True

    def test_returns_false_for_impossible_pid(self):
        # PID 0 is special on Linux — os.kill(0, 0) targets the process group
        # instead of existence-checking PID 0, so use a PID that can't exist.
        assert _pid_alive(2_147_483_646) is False


class TestFormatCmdHint:
    def test_python_dash_m(self):
        argv = [
            "/opt/conda/envs/ci/bin/python",
            "-m",
            "paramem.server.app",
            "--config",
            "configs/server.yaml",
        ]
        assert format_cmd_hint(argv) == "python / paramem.server.app"

    def test_python_script(self):
        argv = ["/usr/bin/python", "experiments/test8_large_scale.py", "--model", "mistral"]
        assert format_cmd_hint(argv) == "python / test8_large_scale.py"

    def test_interpreter_only_with_flag(self):
        # Bare "python --version" — no entry file, no -m → just the interpreter.
        assert format_cmd_hint(["python", "--version"]) == "python"

    def test_single_arg(self):
        assert format_cmd_hint(["bash"]) == "bash"

    def test_empty(self):
        assert format_cmd_hint([]) == ""


class TestClearHoldEnv:
    def test_invokes_unset_environment_with_all_hold_vars(self):
        with patch("paramem.server.app.subprocess.run", return_value=_mk_completed("")) as mock_run:
            ok = _clear_hold_env()
        assert ok is True
        assert mock_run.call_count == 1
        args = mock_run.call_args[0][0]
        assert args[:3] == ["systemctl", "--user", "unset-environment"]
        # All three hold variables must be listed — partial clears leave
        # orphan stamps behind.
        for var in _HOLD_ENV_VARS:
            assert var in args

    def test_returns_false_on_timeout(self):
        with patch(
            "paramem.server.app.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="systemctl", timeout=5),
        ):
            assert _clear_hold_env() is False
