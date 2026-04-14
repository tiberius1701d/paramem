"""Tests for `paramem/utils/notify.py` — Windows toast from WSL2."""

from __future__ import annotations

from unittest.mock import patch

from paramem.utils import notify as notify_mod


def test_notify_noop_when_powershell_missing():
    """On non-WSL (no powershell.exe), notify silently no-ops."""
    with (
        patch.object(notify_mod, "_POWERSHELL", None),
        patch.object(notify_mod, "subprocess") as sp,
    ):
        notify_mod.notify("Title", "Message")
    sp.Popen.assert_not_called()


def test_notify_spawns_background_process_when_available():
    with (
        patch.object(notify_mod, "_POWERSHELL", "/mock/powershell.exe"),
        patch.object(notify_mod, "subprocess") as sp,
    ):
        notify_mod.notify("Title", "Message")
    sp.Popen.assert_called_once()
    args = sp.Popen.call_args[0][0]
    assert args[0] == "/mock/powershell.exe"
    # Non-blocking: no .wait() called.
    sp.Popen.return_value.wait.assert_not_called()


def test_notify_escapes_single_quotes():
    with (
        patch.object(notify_mod, "_POWERSHELL", "/mock/powershell.exe"),
        patch.object(notify_mod, "subprocess") as sp,
    ):
        notify_mod.notify("Alex's laptop", "it's running")
    cmd = sp.Popen.call_args[0][0][-1]
    assert "Alex''s laptop" in cmd
    assert "it''s running" in cmd


def test_notify_server_and_ml_wrappers():
    with patch.object(notify_mod, "notify") as mock_notify:
        notify_mod.notify_server("server up")
        notify_mod.notify_ml("training done")
    titles = [call.args[0] for call in mock_notify.call_args_list]
    assert titles == ["ParaMem Server", "ParaMem ML"]


def test_notify_handles_oserror_quietly():
    with (
        patch.object(notify_mod, "_POWERSHELL", "/mock/powershell.exe"),
        patch.object(notify_mod, "subprocess") as sp,
    ):
        sp.Popen.side_effect = OSError("spawn failed")
        # Must not raise.
        notify_mod.notify("title", "message")
