"""Tests for WSLPowerShellInhibitor (the original _SleepInhibitor).

Updated to patch at gpu_guard.inhibitor after extraction from paramem.
All PowerShell execution is mocked — no real child process is spawned.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# gpu_guard is provided by lab-tools (separate repo, not on PyPI).  CI does
# not install it; skip the whole module rather than erroring at collection.
pytest.importorskip("gpu_guard")

from gpu_guard.inhibitor import WSLPowerShellInhibitor, _is_wsl2  # noqa: E402

# Backward-compatible alias: existing callers use _SleepInhibitor.
_SleepInhibitor = WSLPowerShellInhibitor


# ---------------------------------------------------------------------------
# _is_wsl2 detection
# ---------------------------------------------------------------------------


class TestIsWsl2:
    def test_detects_microsoft_kernel(self, tmp_path):
        """Returns True when /proc/version contains 'microsoft'."""
        fake_proc = tmp_path / "version"
        fake_proc.write_text(
            "Linux version 5.15.90.1-microsoft-standard-WSL2 (gcc version 11.3.0)\n"
        )
        with patch("builtins.open", return_value=fake_proc.open()):
            assert _is_wsl2() is True

    def test_detects_wsl_kernel(self, tmp_path):
        """Returns True when /proc/version contains 'WSL' (case-insensitive)."""
        fake_proc = tmp_path / "version"
        fake_proc.write_text("Linux version 5.15.0-1234-WSL (Ubuntu)\n")
        with patch("builtins.open", return_value=fake_proc.open()):
            assert _is_wsl2() is True

    def test_returns_false_on_native_linux(self, tmp_path):
        """Returns False when /proc/version has no WSL markers."""
        fake_proc = tmp_path / "version"
        fake_proc.write_text("Linux version 6.1.0-20-amd64 (debian-kernel@lists.debian.org)\n")
        with patch("builtins.open", return_value=fake_proc.open()):
            assert _is_wsl2() is False

    def test_returns_false_on_oserror(self):
        """Returns False when /proc/version cannot be read."""
        with patch("builtins.open", side_effect=OSError("no such file")):
            assert _is_wsl2() is False


# ---------------------------------------------------------------------------
# _SleepInhibitor (WSLPowerShellInhibitor)
# ---------------------------------------------------------------------------


class TestSleepInhibitor:
    def _make_mock_proc(self) -> MagicMock:
        proc = MagicMock()
        proc.pid = 12345
        proc.kill = MagicMock()
        proc.wait = MagicMock(return_value=0)
        return proc

    def test_start_and_stop_on_wsl2(self):
        """start() spawns a process on WSL2; stop() kills and reaps it."""
        mock_proc = self._make_mock_proc()
        with (
            patch("gpu_guard.inhibitor._is_wsl2", return_value=True),
            patch("gpu_guard.inhibitor.subprocess.Popen", return_value=mock_proc) as mock_popen,
        ):
            inhibitor = _SleepInhibitor()
            inhibitor.start()

            mock_popen.assert_called_once()
            assert inhibitor._proc is mock_proc

            inhibitor.stop()

            mock_proc.kill.assert_called_once()
            mock_proc.wait.assert_called_once()
            assert inhibitor._proc is None

    def test_start_is_noop_on_non_wsl2(self):
        """start() does nothing on native Linux."""
        with (
            patch("gpu_guard.inhibitor._is_wsl2", return_value=False),
            patch("gpu_guard.inhibitor.subprocess.Popen") as mock_popen,
        ):
            inhibitor = _SleepInhibitor()
            inhibitor.start()

            mock_popen.assert_not_called()
            assert inhibitor._proc is None

    def test_stop_without_start_is_safe(self):
        """stop() can be called before start() without raising."""
        inhibitor = _SleepInhibitor()
        inhibitor.stop()

    def test_start_is_idempotent(self):
        """Calling start() twice does not spawn a second process."""
        mock_proc = self._make_mock_proc()
        with (
            patch("gpu_guard.inhibitor._is_wsl2", return_value=True),
            patch("gpu_guard.inhibitor.subprocess.Popen", return_value=mock_proc) as mock_popen,
        ):
            inhibitor = _SleepInhibitor()
            inhibitor.start()
            inhibitor.start()

            assert mock_popen.call_count == 1
            assert inhibitor._proc is mock_proc

            inhibitor.stop()

    def test_atexit_registered_on_start(self):
        """start() does not register per-instance atexit (module-level handles it)."""
        # The original test verified atexit.register was called once. The new
        # design moves atexit registration to the module level (registered at
        # import time via _module_atexit_stop). Per-instance registration was
        # moved off __init__ per the design-review fix. This test now verifies
        # no per-instance registration occurs.
        mock_proc = self._make_mock_proc()
        with (
            patch("gpu_guard.inhibitor._is_wsl2", return_value=True),
            patch("gpu_guard.inhibitor.subprocess.Popen", return_value=mock_proc),
            patch("gpu_guard.inhibitor.atexit.register") as mock_register,
        ):
            inhibitor = _SleepInhibitor()
            inhibitor.start()

            # No per-instance handler registered after the module-level fix.
            mock_register.assert_not_called()
            inhibitor.stop()

    def test_atexit_unregistered_on_stop(self):
        """stop() does not call atexit.unregister (no per-instance handler to remove)."""
        mock_proc = self._make_mock_proc()
        with (
            patch("gpu_guard.inhibitor._is_wsl2", return_value=True),
            patch("gpu_guard.inhibitor.subprocess.Popen", return_value=mock_proc),
            patch("gpu_guard.inhibitor.atexit.register"),
            patch("gpu_guard.inhibitor.atexit.unregister") as mock_unregister,
        ):
            inhibitor = _SleepInhibitor()
            inhibitor.start()
            inhibitor.stop()

            mock_unregister.assert_not_called()

    def test_atexit_not_registered_on_non_wsl2(self):
        """No atexit registration when not WSL2."""
        with (
            patch("gpu_guard.inhibitor._is_wsl2", return_value=False),
            patch("gpu_guard.inhibitor.atexit.register") as mock_register,
        ):
            inhibitor = _SleepInhibitor()
            inhibitor.start()

            mock_register.assert_not_called()

    def test_start_handles_missing_powershell(self):
        """start() logs a warning and does not raise when powershell.exe is absent."""
        with (
            patch("gpu_guard.inhibitor._is_wsl2", return_value=True),
            patch(
                "gpu_guard.inhibitor.subprocess.Popen",
                side_effect=FileNotFoundError("powershell.exe not found"),
            ),
        ):
            inhibitor = _SleepInhibitor()
            inhibitor.start()

            assert inhibitor._proc is None

    def test_stop_tolerates_dead_process(self):
        """stop() does not raise when the child process is already dead."""
        mock_proc = self._make_mock_proc()
        mock_proc.kill.side_effect = ProcessLookupError("no such process")
        with (
            patch("gpu_guard.inhibitor._is_wsl2", return_value=True),
            patch("gpu_guard.inhibitor.subprocess.Popen", return_value=mock_proc),
        ):
            inhibitor = _SleepInhibitor()
            inhibitor.start()
            inhibitor.stop()
