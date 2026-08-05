"""Unit tests for _collect_pre_flight_items populator.

Tests cover:
- Clean store → []
- Over-cap store → one item, level=action_required
- Suppressed during STAGING
- Suppressed during TRIAL
- config=None → []
- Unevaluable check (raise) → one migration_pre_flight_check_error item, logged
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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
# Test 22 — over-cap store → one item, level=action_required
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
        assert item.level == "action_required"
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
# Test 26 — MagicMock config → one migration_pre_flight_check_error item, logged
# ---------------------------------------------------------------------------


class TestPreFlightItemsSurfacesMockConfigAsActionRequired:
    def test_preflight_items_surfaces_mock_config_as_action_required(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """MagicMock config → one migration_pre_flight_check_error item, logged.

        A MagicMock config has no real max_total_disk_gb, so
        compute_pre_flight_check raises PreFlightUnavailable rather than
        rendering a fake pass; the populator surfaces that as a distinct
        attention item instead of swallowing it silently.
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
        with caplog.at_level(logging.ERROR, logger="paramem.server.attention"):
            items = _collect_pre_flight_items(state, mock_config)

        assert len(items) == 1
        item = items[0]
        assert item.kind == "migration_pre_flight_check_error"
        assert item.level == "action_required"
        assert any(r.exc_info is not None for r in caplog.records)


# ---------------------------------------------------------------------------
# Test 27 — compute_pre_flight_check raising → one item, logged, no crash
# ---------------------------------------------------------------------------


class TestPreFlightItemsSurfacesRaise:
    def test_preflight_items_surfaces_raise(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """compute_pre_flight_check raising → one migration_pre_flight_check_error
        item; the populator logs the failure and does not propagate it."""
        config = _make_config(tmp_path, max_total_disk_gb=20.0)
        state = _live_state()

        with (
            caplog.at_level(logging.ERROR, logger="paramem.server.attention"),
            patch(
                "paramem.backup.preflight.compute_pre_flight_check",
                side_effect=RuntimeError("scan failed"),
            ),
        ):
            items = _collect_pre_flight_items(state, config)

        assert len(items) == 1
        assert items[0].kind == "migration_pre_flight_check_error"
        assert items[0].level == "action_required"
        assert any(r.exc_info is not None for r in caplog.records)

    def test_preflight_items_staging_suppresses_raise(self, tmp_path: Path) -> None:
        """STAGING suppression wins even when the underlying check would raise —
        the check is never reached."""
        config = _make_config(tmp_path, max_total_disk_gb=20.0)
        state = _staging_state()

        with patch(
            "paramem.backup.preflight.compute_pre_flight_check",
            side_effect=RuntimeError("scan failed"),
        ):
            items = _collect_pre_flight_items(state, config)

        assert items == []

    def test_preflight_items_trial_suppresses_raise(self, tmp_path: Path) -> None:
        """TRIAL suppression wins even when the underlying check would raise —
        the check is never reached."""
        config = _make_config(tmp_path, max_total_disk_gb=20.0)
        state = _trial_state()

        with patch(
            "paramem.backup.preflight.compute_pre_flight_check",
            side_effect=RuntimeError("scan failed"),
        ):
            items = _collect_pre_flight_items(state, config)

        assert items == []
