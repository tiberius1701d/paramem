"""Unit tests for the paramem.utils.systemctl transport seam.

Captures the real ``run``/``spawn`` implementations at import time — before
the autouse host-isolation guard in ``tests/conftest.py`` monkeypatches the
module attributes for every test — and patches ``subprocess.run``/``Popen``
one layer below instead, so these tests exercise the actual seam body rather
than the guard's stub.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from paramem.utils import systemctl

_ORIGINAL_RUN = systemctl.run
_ORIGINAL_SPAWN = systemctl.spawn


class TestSystemctlRun:
    def test_prepends_systemctl_user(self):
        """The rendered argv is ['systemctl', '--user', *args]."""
        with patch.object(systemctl.subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )
            _ORIGINAL_RUN("restart", "paramem-server")
        called_args, _ = mock_run.call_args
        assert called_args[0] == ["systemctl", "--user", "restart", "paramem-server"]

    def test_check_false_capture_output_text(self):
        """check=False, capture_output=True, text=True on every call."""
        with patch.object(systemctl.subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )
            _ORIGINAL_RUN("show-environment")
        _, kwargs = mock_run.call_args
        assert kwargs["check"] is False
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True

    def test_timeout_passthrough(self):
        """A caller-supplied timeout reaches subprocess.run unchanged."""
        with patch.object(systemctl.subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )
            _ORIGINAL_RUN("show-environment", timeout=5)
        _, kwargs = mock_run.call_args
        assert kwargs["timeout"] == 5

    def test_timeout_none_by_default(self):
        """No timeout kwarg → None reaches subprocess.run (no bound)."""
        with patch.object(systemctl.subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )
            _ORIGINAL_RUN("daemon-reload")
        _, kwargs = mock_run.call_args
        assert kwargs["timeout"] is None

    def test_timeout_expired_propagates(self):
        """subprocess.TimeoutExpired is not swallowed by the seam."""
        with (
            patch.object(
                systemctl.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired(cmd="systemctl", timeout=5),
            ),
            pytest.raises(subprocess.TimeoutExpired),
        ):
            _ORIGINAL_RUN("show-environment", timeout=5)

    def test_returns_completed_process(self):
        """The CompletedProcess from subprocess.run is returned verbatim."""
        expected = subprocess.CompletedProcess(
            args=["systemctl", "--user", "show-environment"],
            returncode=0,
            stdout="FOO=bar\n",
            stderr="",
        )
        with patch.object(systemctl.subprocess, "run", return_value=expected):
            result = _ORIGINAL_RUN("show-environment")
        assert result is expected


class TestSystemctlSpawn:
    def test_prepends_systemctl_user_and_start_new_session(self):
        """spawn() renders the same argv prefix and detaches via start_new_session."""
        with patch.object(systemctl.subprocess, "Popen") as mock_popen:
            _ORIGINAL_SPAWN("restart", "paramem-server")
        called_args, called_kwargs = mock_popen.call_args
        assert called_args[0] == ["systemctl", "--user", "restart", "paramem-server"]
        assert called_kwargs["start_new_session"] is True

    def test_does_not_wait(self):
        """spawn() never blocks on the child — no .wait()/.communicate() call."""
        with patch.object(systemctl.subprocess, "Popen") as mock_popen:
            _ORIGINAL_SPAWN("restart", "paramem-server")
        mock_popen.return_value.wait.assert_not_called()
        mock_popen.return_value.communicate.assert_not_called()
