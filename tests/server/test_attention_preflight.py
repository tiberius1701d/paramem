"""Unit tests for _collect_pre_flight_items populator (Slice 6b).

Tests cover:
- Clean store → []
- Over-cap store → one item, level=info
- Suppressed during STAGING
- Suppressed during TRIAL
- config=None → []
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from paramem.server.attention import _collect_pre_flight_items
from paramem.server.config import (
    PathsConfig,
    SecurityConfig,
    ServerBackupsConfig,
    ServerConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    tmp_path: Path,
    max_total_disk_gb: float = 20.0,
) -> ServerConfig:
    """Build a minimal ServerConfig pointing at tmp_path."""
    config = ServerConfig.__new__(ServerConfig)
    config.paths = PathsConfig(
        data=tmp_path / "ha",
        sessions=tmp_path / "ha" / "sessions",
        debug=tmp_path / "ha" / "debug",
    )
    config.paths.data.mkdir(parents=True, exist_ok=True)
    config.security = SecurityConfig(
        backups=ServerBackupsConfig(
            schedule="daily 04:00",
            artifacts=["config", "graph", "registry"],
            max_total_disk_gb=max_total_disk_gb,
        )
    )
    return config


def _live_state() -> dict:
    return {"migration": {"state": "LIVE", "recovery_required": []}}


def _staging_state() -> dict:
    return {"migration": {"state": "STAGING", "recovery_required": []}}


def _trial_state() -> dict:
    return {"migration": {"state": "TRIAL", "recovery_required": []}}


# ---------------------------------------------------------------------------
# Test 21 — clean store → []
# ---------------------------------------------------------------------------


class TestPreFlightItemsEmptyWhenUnderCap:
    def test_preflight_items_empty_when_under_cap(self, tmp_path: Path) -> None:
        """Clean store → _collect_pre_flight_items returns []."""
        config = _make_config(tmp_path, max_total_disk_gb=20.0)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        live_config = tmp_path / "server.yaml"
        live_config.write_bytes(b"model: mistral\n")
        state = _live_state()
        state["config_path"] = str(live_config)

        items = _collect_pre_flight_items(state, config)
        assert items == []


# ---------------------------------------------------------------------------
# Test 22 — over-cap store → one item, level=info
# ---------------------------------------------------------------------------


class TestPreFlightItemsEmitsOnOverCap:
    def test_preflight_items_emits_on_over_cap(self, tmp_path: Path) -> None:
        """Over-cap store → one item with kind=migration_pre_flight_fail."""
        cap_gb = 0.0001  # 100 KB cap
        config = _make_config(tmp_path, max_total_disk_gb=cap_gb)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        # Seed 200 KB > 100 KB cap.
        slot = backups_root / "config" / "20260421-040000"
        slot.mkdir(parents=True)
        (slot / "config.bin").write_bytes(b"x" * 200_000)

        live_config = tmp_path / "server.yaml"
        live_config.write_bytes(b"model: mistral\n")
        state = _live_state()
        state["config_path"] = str(live_config)

        items = _collect_pre_flight_items(state, config)
        assert len(items) == 1
        item = items[0]
        assert item.kind == "migration_pre_flight_fail"
        assert item.level == "info"
        assert "disk pressure" in item.summary.lower()
        assert "backup-prune" in item.action_hint.lower()
        # Used/cap GB must appear in summary.
        assert "GB" in item.summary


# ---------------------------------------------------------------------------
# Test 23 — suppressed during STAGING
# ---------------------------------------------------------------------------


class TestPreFlightItemsSuppressedDuringStaging:
    def test_preflight_items_suppressed_during_staging(self, tmp_path: Path) -> None:
        """Migration state=STAGING → _collect_pre_flight_items returns []."""
        cap_gb = 0.0001
        config = _make_config(tmp_path, max_total_disk_gb=cap_gb)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)
        slot = backups_root / "config" / "20260421-040000"
        slot.mkdir(parents=True)
        (slot / "config.bin").write_bytes(b"x" * 200_000)

        state = _staging_state()
        items = _collect_pre_flight_items(state, config)
        assert items == []


# ---------------------------------------------------------------------------
# Test 24 — suppressed during TRIAL
# ---------------------------------------------------------------------------


class TestPreFlightItemsSuppressedDuringTrial:
    def test_preflight_items_suppressed_during_trial(self, tmp_path: Path) -> None:
        """Migration state=TRIAL → _collect_pre_flight_items returns []."""
        cap_gb = 0.0001
        config = _make_config(tmp_path, max_total_disk_gb=cap_gb)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)
        slot = backups_root / "config" / "20260421-040000"
        slot.mkdir(parents=True)
        (slot / "config.bin").write_bytes(b"x" * 200_000)

        state = _trial_state()
        items = _collect_pre_flight_items(state, config)
        assert items == []


# ---------------------------------------------------------------------------
# Test 25 — config=None → []
# ---------------------------------------------------------------------------


class TestPreFlightItemsToleratesNoneConfig:
    def test_preflight_items_tolerates_none_config(self) -> None:
        """config=None → [] without crash."""
        items = _collect_pre_flight_items({}, None)
        assert items == []


# ---------------------------------------------------------------------------
# Test 26 — MagicMock config → [] (no false positive from _collect_pre_flight_items)
# ---------------------------------------------------------------------------


class TestPreFlightItemsToleratesMockConfig:
    def test_preflight_items_tolerates_mock_config(self, tmp_path: Path) -> None:
        """MagicMock config → [] — the guard inside compute_pre_flight_check prevents
        a false migration_pre_flight_fail item when tests seed a backup dir but supply
        a MagicMock config (e.g. via /status in unit tests).
        """
        mock_config = MagicMock()
        # Seed a large backup slot that would exceed any small cap.
        mock_backups_root = tmp_path / "backups"
        mock_backups_root.mkdir(parents=True, exist_ok=True)
        slot = mock_backups_root / "config" / "20260421-040000"
        slot.mkdir(parents=True)
        (slot / "config.bin").write_bytes(b"x" * 200_000)

        # Wire paths.data so _collect_pre_flight_items resolves backups_root.
        mock_config.paths.data = tmp_path

        state = {"migration": {"state": "LIVE", "recovery_required": []}}
        items = _collect_pre_flight_items(state, mock_config)
        assert items == []
