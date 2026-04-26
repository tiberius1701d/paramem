"""V2.5 behavior-preservation parity tests.

Seven tests that lock the V2.5 preservation contract: behaviors that must
hold after migrating from the hand-coded ``ParamemServerConsumer`` to the
config-driven ``ConfigConsumer``.

All tests load the ``[consumers.paramem-server]`` section from a copy of the
example TOML written to ``tmp_path``, so they neither require nor pollute
``~/.config/gpu-guard/config.toml``.

The SIGUSR1 path no longer exists in V2.5.  Test 2 (originally titled
``test_sigusr1_sent_on_release_when_not_cloud_only``) now asserts that the
HTTP POST to ``/gpu/release`` is sent via ``urllib.request`` (the primitive
used by the config-driven ``release_http`` releaser).
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path to the example config (source of truth for the TOML shape)
# ---------------------------------------------------------------------------

_EXAMPLE_TOML = (
    Path(__file__).parent.parent.parent / "lab-tools" / "gpu_guard" / "config.toml.example"
)


def _load_paramem_consumer(tmp_path: Path):
    """Write a copy of the example TOML to tmp_path and load the paramem-server consumer.

    Returns the ``ConfigConsumer`` for ``paramem-server``.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        The ``ConfigConsumer`` instance for the ``paramem-server`` section.
    """
    from gpu_guard.config import load_config

    dest = tmp_path / "config.toml"
    shutil.copy(_EXAMPLE_TOML, dest)
    consumers = load_config(path=dest)
    assert "paramem-server" in consumers, (
        f"paramem-server not found in config loaded from {dest}. Keys: {sorted(consumers)}"
    )
    return consumers["paramem-server"]


# ---------------------------------------------------------------------------
# Test 1: cloud-only short-circuit
# ---------------------------------------------------------------------------


class TestStatusCloudOnlyShortCircuitsRelease:
    """When the server is already cloud-only, release_server_gpu returns True
    without issuing any HTTP release call.

    In V2.5 the is_idle check is done by the ``ConfigConsumer`` using the
    ``idle_check_http`` primitive.  This test verifies that if the idle check
    returns True (cloud-only), the release pathway completes without calling
    the releaser.
    """

    def test_status_cloud_only_short_circuits_release(self, tmp_path: Path) -> None:
        """is_idle() returning True means no release call is made."""
        cs = _load_paramem_consumer(tmp_path)

        # Patch urlopen so the idle-check HTTP GET returns {"mode": "cloud-only"}.
        cloud_only_response = MagicMock()
        cloud_only_response.__enter__ = lambda s: s
        cloud_only_response.__exit__ = MagicMock(return_value=False)
        cloud_only_response.read.return_value = b'{"mode": "cloud-only"}'

        with (
            patch("urllib.request.urlopen", return_value=cloud_only_response),
            patch("urllib.request.Request", wraps=urllib.request.Request) as mock_request,
        ):
            # wait_for_idle on a cloud-only server should return True immediately
            result = cs.wait_for_idle(pid=12345, timeout=5)

        assert result is True, "wait_for_idle should return True when server is cloud-only"
        # Only the idle-check GET should be called, not a release POST.
        post_calls = [
            c
            for c in mock_request.call_args_list
            if len(c.args) >= 1 and "release" in str(c.args[0])
        ]
        assert post_calls == [], (
            f"No release POST should be issued when already cloud-only, "
            f"but these calls appeared: {post_calls}"
        )


# ---------------------------------------------------------------------------
# Test 2: HTTP POST sent on release (not SIGUSR1 — that path is gone in V2.5)
# ---------------------------------------------------------------------------


class TestHttpPostSentOnRelease:
    """When the server is not cloud-only, request_release() sends an HTTP POST
    to /gpu/release.

    In V2.5 there is no SIGUSR1 path.  The ``release_http`` primitive calls
    ``urllib.request.urlopen`` with a POST request to the configured URL.
    """

    def test_http_post_sent_on_release_when_not_cloud_only(self, tmp_path: Path) -> None:
        """request_release() calls urllib.request.urlopen with POST to /gpu/release."""
        cs = _load_paramem_consumer(tmp_path)

        # Stub urlopen to succeed silently.
        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = b""

        captured_requests: list[urllib.request.Request] = []

        def capturing_urlopen(req, timeout=None):
            captured_requests.append(req)
            return mock_response

        with patch("urllib.request.urlopen", side_effect=capturing_urlopen):
            cs.request_release(pid=12345)

        assert len(captured_requests) == 1, (
            f"Expected exactly one HTTP request, got {len(captured_requests)}"
        )
        req = captured_requests[0]
        assert req.get_method() == "POST", f"Expected POST, got {req.get_method()}"
        assert "/gpu/release" in req.full_url, f"/gpu/release not in URL: {req.full_url}"
        assert "8420" in req.full_url, f"Port 8420 not in URL: {req.full_url}"


# ---------------------------------------------------------------------------
# Test 3: detect union — systemd OR port OR cmdline
# ---------------------------------------------------------------------------


class TestDetectUnionSystemdOrPortOrCmdline:
    """find_pids on a ConfigConsumer returns the union of all detector results.

    The config has three detectors: systemd-unit, port-listener, cmdline-marker.
    Each should be able to independently classify a PID.  This test constructs
    a ``ConfigConsumer`` directly with three stub detector callables so the
    union composition is verified without depending on the DETECTORS registry
    binding order at config-load time.

    Covers the V1 contract: find_pids([systemd_pid, port_pid, cmdline_pid, 9999])
    matches the union of all three sources, and excludes unrelated PIDs.
    """

    def test_detect_union_systemd_or_port_or_cmdline(self, tmp_path: Path) -> None:
        """Union of three stub detectors: each contributes a disjoint PID."""
        from gpu_guard.config import ConfigConsumer

        systemd_pid = 111
        port_pid = 222
        cmdline_pid = 333
        unrelated_pid = 9999
        all_candidates = [systemd_pid, port_pid, cmdline_pid, unrelated_pid]

        # Build a ConfigConsumer with three stub detectors, one per detection source.
        # This mirrors the TOML-driven paramem-server consumer's three detect entries.
        def systemd_detector(candidates: list[int]) -> list[int]:
            return [p for p in candidates if p == systemd_pid]

        def port_detector(candidates: list[int]) -> list[int]:
            return [p for p in candidates if p == port_pid]

        def cmdline_detector(candidates: list[int]) -> list[int]:
            return [p for p in candidates if p == cmdline_pid]

        cs = ConfigConsumer(
            name="paramem-server",
            priority=5,
            non_evictable_without_confirm=False,
            detectors=[systemd_detector, port_detector, cmdline_detector],
            releaser=lambda pid: None,
            idle_check=lambda pid: True,
            describe_template="{name} (PID {pid}) is using the GPU.",
        )

        result = cs.find_pids(all_candidates)

        assert systemd_pid in result, f"systemd PID {systemd_pid} missing from {result}"
        assert port_pid in result, f"port PID {port_pid} missing from {result}"
        assert cmdline_pid in result, f"cmdline PID {cmdline_pid} missing from {result}"
        assert unrelated_pid not in result, (
            f"unrelated PID {unrelated_pid} should not be in {result}"
        )


# ---------------------------------------------------------------------------
# Test 4: PARAMEM_HOLD env vars stamped and cleared
# ---------------------------------------------------------------------------


class TestParamemHoldEnvVarsStampedAndCleared:
    """on_acquired stamps PARAMEM_HOLD_* env vars; on_released clears them.

    Equivalent of the existing test_gpu_guard_shim_on_released regression,
    now exercising ``ParamemEnvStampAdapter`` directly.
    """

    def test_paramem_hold_env_vars_stamped_and_cleared(self) -> None:
        """on_acquired issues set-environment; on_released issues unset-environment."""
        from paramem.gpu_consumer import ParamemEnvStampAdapter

        adapter = ParamemEnvStampAdapter()

        run_calls: list[list] = []

        def recording_run(args, **kwargs):
            run_calls.append(list(args))
            m = MagicMock()
            m.returncode = 0
            return m

        with patch("paramem.gpu_consumer.subprocess.run", side_effect=recording_run):
            import sys

            adapter.on_acquired(own_pid=12345, argv=sys.argv)
            adapter.on_released()

        # set-environment must appear (on_acquired).
        set_calls = [
            args
            for args in run_calls
            if len(args) >= 3 and args[1] == "--user" and args[2] == "set-environment"
        ]
        assert set_calls, f"Expected set-environment call, got: {run_calls}"
        assert any("PARAMEM_HOLD_PID=12345" in arg for arg in set_calls[0]), (
            f"PARAMEM_HOLD_PID=12345 not in set-environment args: {set_calls[0]}"
        )

        # unset-environment must appear (on_released).
        unset_calls = [
            args
            for args in run_calls
            if len(args) >= 3 and args[1] == "--user" and args[2] == "unset-environment"
        ]
        assert unset_calls, f"Expected unset-environment call, got: {run_calls}"
        assert "PARAMEM_HOLD_PID" in unset_calls[0], (
            f"PARAMEM_HOLD_PID missing from unset-environment args: {unset_calls[0]}"
        )


# ---------------------------------------------------------------------------
# Test 5: describe string matches V1 byte-for-byte
# ---------------------------------------------------------------------------


class TestDescribeStringMatchesV1:
    """The describe string seen by the operator in interactive conflicts is pinned.

    The TOML ``describe_template`` for ``paramem-server`` must produce exactly
    the V1 prompt string so the UX is unchanged after the V2.5 migration.
    """

    _EXPECTED = (
        "ParaMem server (PID 12345) is using the GPU.\n"
        "  It will switch to cloud-only mode during this workload."
    )

    def test_describe_string_matches_v1(self, tmp_path: Path) -> None:
        """describe(12345) byte-for-byte equals the V1 ParamemServerConsumer output."""
        cs = _load_paramem_consumer(tmp_path)
        got = cs.describe(pid=12345)
        assert got == self._EXPECTED, (
            f"describe() string mismatch.\nExpected: {self._EXPECTED!r}\nGot:      {got!r}"
        )


# ---------------------------------------------------------------------------
# Test 6: release_server_gpu raises GPUConfigMissing when config missing
# ---------------------------------------------------------------------------


class TestReleaseServerGpuRaisesOnMissingConfig:
    """release_server_gpu() raises GPUConfigMissing (not returns False) when
    no config registers paramem-server.

    Closes D4-sub at the paramem level: a misconfigured workstation must
    surface loudly, not masquerade as a release timeout.
    """

    def test_release_server_gpu_raises_on_missing_config(self, tmp_path: Path) -> None:
        """GPUConfigMissing raised when GPU_GUARD_CONFIG points to nonexistent path."""
        from gpu_guard import GPUConfigMissing
        from gpu_guard._core import _reset_autoload_for_tests, clear_default_consumers

        from experiments.utils.gpu_guard import release_server_gpu

        # Clear all consumers and prevent autoload from finding any config.
        clear_default_consumers()
        _reset_autoload_for_tests()
        nonexistent = str(tmp_path / "nonexistent.toml")
        try:
            with (
                patch.dict(os.environ, {"GPU_GUARD_NO_AUTOLOAD": "1"}),
                patch.dict(os.environ, {"GPU_GUARD_CONFIG": nonexistent}),
            ):
                with pytest.raises(GPUConfigMissing) as exc_info:
                    release_server_gpu()
            assert "paramem-server" in str(exc_info.value)
        finally:
            # Restore the env-stamp adapter so other tests are not disturbed.
            from gpu_guard import add_default_consumer

            from paramem.gpu_consumer import adapter

            add_default_consumer(adapter)
            _reset_autoload_for_tests()


# ---------------------------------------------------------------------------
# Test 7: wait_for_idle owns the poll loop
# ---------------------------------------------------------------------------


class TestWaitForIdleOwnsLoop:
    """wait_for_idle on the ConfigConsumer polls every 2 s until idle or timeout.

    Loads ``cs["paramem-server"]`` from the example TOML, mocks ``_idle_check``
    to return False four times then True, and verifies that ``time.sleep`` was
    called four times with argument ``2``.

    This closes D5-sub-A at the paramem level: the idle loop is owned by
    ConfigConsumer, not delegated to any outer caller.
    """

    def test_wait_for_idle_owns_loop_paramem_side(self, tmp_path: Path) -> None:
        """wait_for_idle calls time.sleep(2) for each non-idle probe, stops on idle."""
        cs = _load_paramem_consumer(tmp_path)

        idle_returns = [False, False, False, False, True]
        sleep_calls: list[float] = []

        def mock_idle_check(pid: int) -> bool:
            return idle_returns.pop(0)

        def mock_sleep(secs: float) -> None:
            sleep_calls.append(secs)

        # Patch the internal idle check and time.sleep inside gpu_guard.config.
        with (
            patch.object(cs, "_idle_check", side_effect=mock_idle_check),
            patch("gpu_guard.config.time.sleep", side_effect=mock_sleep),
        ):
            result = cs.wait_for_idle(pid=42, timeout=30)

        assert result is True, "wait_for_idle should return True when idle check eventually passes"
        assert len(sleep_calls) == 4, (
            f"Expected 4 sleep calls (one per False idle-check), "
            f"got {len(sleep_calls)}: {sleep_calls}"
        )
        for i, secs in enumerate(sleep_calls):
            assert secs == 2, f"sleep call {i} used {secs!r}s, expected 2s"
