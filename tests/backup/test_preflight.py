"""Unit tests for paramem.backup.preflight — compute_pre_flight_check.

Tests cover:
- Clean store → pass (fail_code=None)
- Over-cap → disk_pressure
- Cloud-only (loop=None) → graph contribution = 0
- Missing registry file → registry contribution = 0
- Unreadable component (config, graph, registry) → raises; no partial estimate
- Cap unavailable (MagicMock / None config) → raises PreFlightUnavailable
- Disk-usage scan failure → raises
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paramem.backup.preflight import (
    PreFlightCheck,
    PreFlightUnavailable,
    compute_pre_flight_check,
)
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


def _make_loop(graph_bytes: bytes, tmp_path: Path) -> MagicMock:
    """Return a mock ConsolidationLoop with output_dir set and episodic/graph.json on disk.

    Preflight reads the on-disk ``output_dir/episodic/graph.json`` via
    ``read_maybe_encrypted`` rather than calling ``loop.merger.save_bytes()``
    (which serialises an empty in-memory graph after the cycle's finally-block reset).
    The test fixture writes ``graph_bytes`` to the file so the preflight
    estimate equals ``len(graph_bytes)`` for the graph contribution.

    The assertion in each test must match ``len(graph_bytes)`` (the on-disk
    length returned by ``read_maybe_encrypted``), NOT ``loop.merger.save_bytes()``.
    """
    loop = MagicMock()
    graph_path = tmp_path / "episodic" / "graph.json"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_bytes(graph_bytes)
    loop.output_dir = str(tmp_path)
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
            loop=_make_loop(b"{}", tmp_path),
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
            loop=_make_loop(b"graph_bytes", tmp_path),
            backups_root=backups_root,
            live_config_path=live_config,
            registry_path=missing_registry,
        )

        # estimate = config_bytes + on-disk graph_bytes + 0 (missing registry).
        # _make_loop writes graph_bytes to output_dir/episodic/graph.json;
        # read_maybe_encrypted returns exactly those bytes.
        expected = len(b"model: mistral\n") + len(b"graph_bytes")
        assert result.estimate_bytes == expected
        assert result.fail_code is None


# ---------------------------------------------------------------------------
# Test 5 — unreadable components raise; no partial estimate
# ---------------------------------------------------------------------------


class TestPreFlightPropagatesReadErrors:
    def test_preflight_raises_when_config_unreadable(self, tmp_path: Path) -> None:
        """PermissionError reading live config propagates — no partial estimate."""
        config = _make_config(tmp_path, max_total_disk_gb=10.0)
        backups_root = tmp_path / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        live_config = tmp_path / "server.yaml"
        live_config.write_bytes(b"model: mistral\n")

        # Use a separate subdirectory for the loop so _patched_read_bytes can
        # distinguish the config path from the graph.json path.
        loop_dir = tmp_path / "loop_dir"
        loop = _make_loop(b"graph", loop_dir)

        # Simulate PermissionError on read_bytes ONLY for the config path.
        original_read_bytes = Path.read_bytes

        def _patched_read_bytes(self):
            if self == live_config:
                raise PermissionError("permission denied")
            return original_read_bytes(self)

        with (
            patch.object(Path, "read_bytes", _patched_read_bytes),
            pytest.raises(PermissionError),
        ):
            compute_pre_flight_check(
                server_config=config,
                loop=loop,
                backups_root=backups_root,
                live_config_path=live_config,
                registry_path=None,
            )

    def test_preflight_raises_when_graph_unreadable(self, tmp_path: Path) -> None:
        """read_maybe_encrypted on graph.json raising propagates — no partial estimate.

        Preflight reads graph bytes from output_dir/episodic/graph.json via
        read_maybe_encrypted instead of loop.merger.save_bytes(). When that read
        fails (e.g. PermissionError), the failure must propagate rather than be
        counted as a 0-byte contribution.
        """
        config = _make_config(tmp_path, max_total_disk_gb=10.0)
        backups_root = tmp_path / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        live_config = tmp_path / "server.yaml"
        live_config.write_bytes(b"model: mistral\n")

        # Write a valid graph.json so output_dir/episodic/graph.json exists,
        # then simulate a PermissionError on its read_bytes call.
        loop_dir = tmp_path / "loop_dir"
        loop = _make_loop(b'{"nodes":[],"links":[]}', loop_dir)
        graph_path = loop_dir / "episodic" / "graph.json"

        original_read_bytes = Path.read_bytes

        def _fail_graph_read(self):
            if self == graph_path:
                raise PermissionError("graph read denied")
            return original_read_bytes(self)

        with (
            patch.object(Path, "read_bytes", _fail_graph_read),
            pytest.raises(PermissionError),
        ):
            compute_pre_flight_check(
                server_config=config,
                loop=loop,
                backups_root=backups_root,
                live_config_path=live_config,
                registry_path=None,
            )

    def test_preflight_raises_when_registry_unreadable(self, tmp_path: Path) -> None:
        """PermissionError reading the registry file propagates — no partial estimate."""
        config = _make_config(tmp_path, max_total_disk_gb=10.0)
        backups_root = tmp_path / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        live_config = tmp_path / "server.yaml"
        live_config.write_bytes(b"model: mistral\n")

        registry = tmp_path / "registry.json"
        registry.write_bytes(b'{"keys": []}')

        original_read_bytes = Path.read_bytes

        def _fail_registry_read(self):
            if self == registry:
                raise PermissionError("registry read denied")
            return original_read_bytes(self)

        with (
            patch.object(Path, "read_bytes", _fail_registry_read),
            pytest.raises(PermissionError),
        ):
            compute_pre_flight_check(
                server_config=config,
                loop=None,
                backups_root=backups_root,
                live_config_path=live_config,
                registry_path=registry,
            )

    def test_preflight_raises_when_disk_usage_scan_fails(self, tmp_path: Path) -> None:
        """compute_disk_usage raising propagates — a failed scan is not a 0-byte usage."""
        config = _make_config(tmp_path, max_total_disk_gb=10.0)
        backups_root = tmp_path / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        live_config = tmp_path / "server.yaml"
        live_config.write_bytes(b"model: mistral\n")

        with (
            patch(
                "paramem.backup.retention.compute_disk_usage",
                side_effect=OSError("scan failed"),
            ),
            pytest.raises(OSError, match="scan failed"),
        ):
            compute_pre_flight_check(
                server_config=config,
                loop=None,
                backups_root=backups_root,
                live_config_path=live_config,
                registry_path=None,
            )

    def test_preflight_raises_when_encrypted_component_unreadable(self, tmp_path: Path) -> None:
        """An age-encrypted artifact with no daily identity loaded raises — every
        component present (config, graph, registry) but none is estimated."""
        config = _make_config(tmp_path, max_total_disk_gb=10.0)
        backups_root = tmp_path / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        live_config = tmp_path / "server.yaml"
        live_config.write_bytes(b"model: mistral\n")

        registry = tmp_path / "registry.json"
        registry.write_bytes(b'{"keys": []}')

        loop = _make_loop(b'{"nodes":[],"links":[]}', tmp_path)

        with (
            patch(
                "paramem.backup.encryption.read_maybe_encrypted",
                side_effect=RuntimeError("daily identity not loadable"),
            ),
            pytest.raises(RuntimeError, match="daily identity not loadable"),
        ):
            compute_pre_flight_check(
                server_config=config,
                loop=loop,
                backups_root=backups_root,
                live_config_path=live_config,
                registry_path=registry,
            )


# ---------------------------------------------------------------------------
# Test 6 — cap unavailable → PreFlightUnavailable
# ---------------------------------------------------------------------------


class TestPreFlightRejectsUnmeasurableCap:
    def test_preflight_raises_when_cap_unavailable(self, tmp_path: Path) -> None:
        """MagicMock server_config → raises PreFlightUnavailable (no false pass).

        MagicMock attributes resolve to MagicMock objects, not real numerics.
        The guard inside compute_pre_flight_check must detect this and raise
        rather than let int(MagicMock()) silently produce cap_bytes=1, or a
        zeroed result render as an honest, unmeasured pass.
        """
        mock_config = MagicMock()

        backups_root = tmp_path / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)
        live_config = tmp_path / "server.yaml"
        live_config.write_bytes(b"model: mistral\n")

        with pytest.raises(PreFlightUnavailable):
            compute_pre_flight_check(
                server_config=mock_config,
                loop=None,
                backups_root=backups_root,
                live_config_path=live_config,
                registry_path=None,
            )

    def test_preflight_raises_when_config_none(self, tmp_path: Path) -> None:
        """server_config=None → raises PreFlightUnavailable."""
        backups_root = tmp_path / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)
        live_config = tmp_path / "server.yaml"
        live_config.write_bytes(b"model: mistral\n")

        with pytest.raises(PreFlightUnavailable):
            compute_pre_flight_check(
                server_config=None,
                loop=None,
                backups_root=backups_root,
                live_config_path=live_config,
                registry_path=None,
            )
