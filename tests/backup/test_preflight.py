"""Unit tests for paramem.backup.preflight — compute_pre_flight_check.

Tests cover:
- Clean store → pass (fail_code=None)
- Over-cap → disk_pressure
- Cloud-only (loop=None) → graph contribution = 0
- Missing registry file → registry contribution = 0
- I/O errors swallowed (PermissionError) → valid result without crash
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from paramem.backup.preflight import PreFlightCheck, compute_pre_flight_check
from paramem.server.config import (
    PathsConfig,
    SecurityConfig,
    ServerBackupsConfig,
    ServerConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, max_total_disk_gb: float = 1.0) -> ServerConfig:
    """Build a minimal ServerConfig pointing at tmp_path."""
    config = ServerConfig.__new__(ServerConfig)
    config.paths = PathsConfig(
        data=tmp_path / "ha",
        sessions=tmp_path / "ha" / "sessions",
        debug=tmp_path / "ha" / "debug",
    )
    config.security = SecurityConfig(
        backups=ServerBackupsConfig(
            schedule="daily 04:00",
            artifacts=["config", "graph", "registry"],
            max_total_disk_gb=max_total_disk_gb,
        )
    )
    return config


def _make_loop(graph_bytes: bytes = b"{}") -> MagicMock:
    """Return a mock ConsolidationLoop with a merger that saves bytes."""
    loop = MagicMock()
    loop.merger = MagicMock()
    loop.merger.save_bytes.return_value = graph_bytes
    return loop


# ---------------------------------------------------------------------------
# Test 1 — empty backups dir → fail_code=None
# ---------------------------------------------------------------------------


class TestPreFlightPassCleanStore:
    def test_preflight_pass_clean_store(self, tmp_path: Path) -> None:
        """Empty backups dir → fail_code=None (no disk pressure)."""
        config = _make_config(tmp_path, max_total_disk_gb=1.0)
        backups_root = tmp_path / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        live_config = tmp_path / "server.yaml"
        live_config.write_bytes(b"model: mistral\n")

        result = compute_pre_flight_check(
            server_config=config,
            loop=_make_loop(b"{}"),
            backups_root=backups_root,
            live_config_path=live_config,
            registry_path=None,
        )

        assert isinstance(result, PreFlightCheck)
        assert result.fail_code is None
        assert result.disk_used_bytes == 0  # empty store
        assert result.disk_cap_bytes == 1 * 1024**3
        assert result.estimate_bytes > 0  # config + graph bytes


# ---------------------------------------------------------------------------
# Test 2 — over-cap → fail_code="disk_pressure"
# ---------------------------------------------------------------------------


class TestPreFlightFailsAtOverCap:
    def test_preflight_fails_at_over_cap(self, tmp_path: Path) -> None:
        """Pre-populate slots so used + estimate > cap → disk_pressure."""
        cap_gb = 0.0001  # 100 KB cap — very tight
        config = _make_config(tmp_path, max_total_disk_gb=cap_gb)
        backups_root = tmp_path / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        # Seed a slot to consume capacity.
        slot = backups_root / "config" / "20260421-040000"
        slot.mkdir(parents=True)
        big_file = slot / "config-20260421-040000.bin"
        big_file.write_bytes(b"x" * 200_000)  # 200 KB > 100 KB cap

        live_config = tmp_path / "server.yaml"
        live_config.write_bytes(b"model: mistral\n")

        result = compute_pre_flight_check(
            server_config=config,
            loop=None,
            backups_root=backups_root,
            live_config_path=live_config,
            registry_path=None,
        )

        assert result.fail_code == "disk_pressure"
        assert result.disk_used_bytes > 0
        assert result.disk_cap_bytes == int(cap_gb * 1024**3)


# ---------------------------------------------------------------------------
# Test 3 — loop=None → graph contribution = 0, config + registry still sum
# ---------------------------------------------------------------------------


class TestPreFlightCloudOnlyLoopNone:
    def test_preflight_cloud_only_loop_none(self, tmp_path: Path) -> None:
        """loop=None → graph contribution = 0; config + registry still summed."""
        config = _make_config(tmp_path, max_total_disk_gb=10.0)
        backups_root = tmp_path / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        live_config = tmp_path / "server.yaml"
        live_config.write_bytes(b"model: mistral\n")

        registry = tmp_path / "registry.json"
        registry.write_bytes(b'{"keys": []}')

        result = compute_pre_flight_check(
            server_config=config,
            loop=None,
            backups_root=backups_root,
            live_config_path=live_config,
            registry_path=registry,
        )

        # estimate = config_bytes + 0 (no graph) + registry_bytes
        assert result.estimate_bytes == len(b"model: mistral\n") + len(b'{"keys": []}')
        assert result.fail_code is None  # well under 10 GB cap


# ---------------------------------------------------------------------------
# Test 4 — registry file absent → registry contribution = 0
# ---------------------------------------------------------------------------


class TestPreFlightNoRegistryFile:
    def test_preflight_no_registry_file(self, tmp_path: Path) -> None:
        """registry_path absent → registry contribution = 0."""
        config = _make_config(tmp_path, max_total_disk_gb=10.0)
        backups_root = tmp_path / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        live_config = tmp_path / "server.yaml"
        live_config.write_bytes(b"model: mistral\n")

        missing_registry = tmp_path / "no_such_file.json"

        result = compute_pre_flight_check(
            server_config=config,
            loop=_make_loop(b"graph_bytes"),
            backups_root=backups_root,
            live_config_path=live_config,
            registry_path=missing_registry,
        )

        # estimate = config_bytes + graph_bytes + 0 (missing registry)
        expected = len(b"model: mistral\n") + len(b"graph_bytes")
        assert result.estimate_bytes == expected
        assert result.fail_code is None


# ---------------------------------------------------------------------------
# Test 5 — I/O errors swallowed → valid PreFlightCheck without crash
# ---------------------------------------------------------------------------


class TestPreFlightSwallowsReadErrors:
    def test_preflight_swallows_permission_error_on_config(self, tmp_path: Path) -> None:
        """PermissionError reading live config → counted as 0 bytes; no crash."""
        config = _make_config(tmp_path, max_total_disk_gb=10.0)
        backups_root = tmp_path / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        live_config = tmp_path / "server.yaml"
        live_config.write_bytes(b"model: mistral\n")

        # Simulate PermissionError on read_bytes for the config path.
        original_read_bytes = Path.read_bytes

        def _patched_read_bytes(self):
            if self == live_config:
                raise PermissionError("permission denied")
            return original_read_bytes(self)

        with patch.object(Path, "read_bytes", _patched_read_bytes):
            result = compute_pre_flight_check(
                server_config=config,
                loop=_make_loop(b"graph"),
                backups_root=backups_root,
                live_config_path=live_config,
                registry_path=None,
            )

        # Should not raise; config contribution = 0, graph still counted.
        assert isinstance(result, PreFlightCheck)
        assert result.fail_code is None
        # estimate = 0 (config error) + len(b"graph") = 5
        assert result.estimate_bytes == len(b"graph")

    def test_preflight_swallows_graph_save_error(self, tmp_path: Path) -> None:
        """loop.merger.save_bytes() raising → graph counted as 0 bytes; no crash."""
        config = _make_config(tmp_path, max_total_disk_gb=10.0)
        backups_root = tmp_path / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        live_config = tmp_path / "server.yaml"
        live_config.write_bytes(b"model: mistral\n")

        bad_loop = MagicMock()
        bad_loop.merger = MagicMock()
        bad_loop.merger.save_bytes.side_effect = RuntimeError("graph serialisation failed")

        result = compute_pre_flight_check(
            server_config=config,
            loop=bad_loop,
            backups_root=backups_root,
            live_config_path=live_config,
            registry_path=None,
        )

        assert isinstance(result, PreFlightCheck)
        assert result.fail_code is None
        assert result.estimate_bytes == len(b"model: mistral\n")  # only config


# ---------------------------------------------------------------------------
# Test 6 — MagicMock config → no-pressure result (no false positive)
# ---------------------------------------------------------------------------


class TestPreFlightMagicMockConfigNoFalsePositive:
    def test_preflight_mock_config_returns_no_pressure(self, tmp_path: Path) -> None:
        """MagicMock server_config → fail_code=None (no false positive).

        MagicMock attributes resolve to MagicMock objects, not real numerics.
        The guard inside compute_pre_flight_check must detect this and return
        a no-pressure result rather than letting int(MagicMock()) silently
        produce cap_bytes=1 and emit a false disk_pressure fail.
        """
        mock_config = MagicMock()

        backups_root = tmp_path / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)
        live_config = tmp_path / "server.yaml"
        live_config.write_bytes(b"model: mistral\n")

        result = compute_pre_flight_check(
            server_config=mock_config,
            loop=None,
            backups_root=backups_root,
            live_config_path=live_config,
            registry_path=None,
        )

        assert isinstance(result, PreFlightCheck)
        assert result.fail_code is None
        assert result.disk_used_bytes == 0
        assert result.disk_cap_bytes == 0
        assert result.estimate_bytes == 0
