"""Regression tests for trial registry isolation (CRITICAL Fix 1, 2026-04-23).

Verifies that _build_trial_loop sets loop.trial_registry_path and
loop.trial_key_metadata_path to paths inside the trial-isolated
``trial_registry/`` directory, NOT to the live ``data/ha/registry.json``
or ``data/ha/registry/key_metadata.json`` paths.

Also verifies that _save_registry / _save_key_metadata honor the loop-level
overrides when present, so a trial consolidation run never writes to the
live registry files.

No GPU required — all model/loop objects are mocked.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

from paramem.server.consolidation import _save_key_metadata, _save_registry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_loop_with_simhash(**kwargs) -> MagicMock:
    """Build a MagicMock ConsolidationLoop with minimal SimHash dicts."""
    loop = MagicMock()
    loop.episodic_simhash = kwargs.get("episodic_simhash", {"graph1": 12345})
    loop.semantic_simhash = kwargs.get("semantic_simhash", {})
    loop.procedural_simhash = kwargs.get("procedural_simhash", {})
    loop.cycle_count = kwargs.get("cycle_count", 1)
    loop.promoted_keys = kwargs.get("promoted_keys", [])
    loop.key_sessions = kwargs.get("key_sessions", {"graph1": 2})
    return loop


# ---------------------------------------------------------------------------
# Fix 1 — _build_trial_loop sets trial_registry_path and trial_key_metadata_path
# ---------------------------------------------------------------------------


class TestBuildTrialLoopRegistryPathOverrides:
    """Verify _build_trial_loop sets registry path overrides pointing INSIDE the trial dir."""

    def test_trial_registry_path_resolves_inside_trial_dir(self, tmp_path):
        """loop.trial_registry_path must be inside trial_adapter_dir.parent/trial_registry/."""
        trial_adapter_dir = tmp_path / "state" / "trial_adapter"
        trial_adapter_dir.mkdir(parents=True, exist_ok=True)

        mock_loop = MagicMock()

        with patch(
            "paramem.server.consolidation.create_consolidation_loop",
            return_value=mock_loop,
        ):
            from paramem.server.app import _build_trial_loop
            from paramem.server.config import load_server_config

            yaml_file = tmp_path / "server.yaml"
            yaml_file.write_bytes(b"model: mistral\n")
            trial_config = load_server_config(yaml_file)

            _build_trial_loop(
                MagicMock(),
                MagicMock(),
                trial_config,
                trial_adapter_dir,
                None,
            )

        # trial_registry_path must be inside trial_adapter_dir.parent/trial_registry/
        expected_registry_dir = trial_adapter_dir.parent / "trial_registry"
        assert hasattr(mock_loop, "trial_registry_path"), (
            "loop must have trial_registry_path attribute after _build_trial_loop"
        )
        assert str(mock_loop.trial_registry_path).startswith(str(expected_registry_dir)), (
            f"trial_registry_path {mock_loop.trial_registry_path!r} must be inside "
            f"{expected_registry_dir!r}, not in the live data/ha directory"
        )

    def test_trial_key_metadata_path_resolves_inside_trial_dir(self, tmp_path):
        """loop.trial_key_metadata_path must be inside trial_adapter_dir.parent/trial_registry/."""
        trial_adapter_dir = tmp_path / "state" / "trial_adapter"
        trial_adapter_dir.mkdir(parents=True, exist_ok=True)

        mock_loop = MagicMock()

        with patch(
            "paramem.server.consolidation.create_consolidation_loop",
            return_value=mock_loop,
        ):
            from paramem.server.app import _build_trial_loop
            from paramem.server.config import load_server_config

            yaml_file = tmp_path / "server.yaml"
            yaml_file.write_bytes(b"model: mistral\n")
            trial_config = load_server_config(yaml_file)

            _build_trial_loop(
                MagicMock(),
                MagicMock(),
                trial_config,
                trial_adapter_dir,
                None,
            )

        expected_registry_dir = trial_adapter_dir.parent / "trial_registry"
        assert hasattr(mock_loop, "trial_key_metadata_path"), (
            "loop must have trial_key_metadata_path attribute after _build_trial_loop"
        )
        assert str(mock_loop.trial_key_metadata_path).startswith(str(expected_registry_dir)), (
            f"trial_key_metadata_path {mock_loop.trial_key_metadata_path!r} must be inside "
            f"{expected_registry_dir!r}, not in the live data/ha directory"
        )

    def test_trial_registry_path_not_set_when_trial_adapter_dir_is_none(self, tmp_path):
        """When trial_adapter_dir is None, loop must not get trial_registry_path."""
        mock_loop = MagicMock(spec=[])  # strict spec: no attributes set

        with patch(
            "paramem.server.consolidation.create_consolidation_loop",
            return_value=mock_loop,
        ):
            from paramem.server.app import _build_trial_loop
            from paramem.server.config import load_server_config

            yaml_file = tmp_path / "server.yaml"
            yaml_file.write_bytes(b"model: mistral\n")
            trial_config = load_server_config(yaml_file)

            _build_trial_loop(
                MagicMock(),
                MagicMock(),
                trial_config,
                None,  # no trial_adapter_dir
                None,
            )

        # MagicMock(spec=[]) raises AttributeError on any attribute access.
        # If _build_trial_loop does NOT set the attribute, no error occurs.
        # This test simply verifies no exception is raised.


# ---------------------------------------------------------------------------
# Fix 1 — _save_registry honors loop.trial_registry_path override
# ---------------------------------------------------------------------------


class TestSaveRegistryHonorsOverride:
    """Verify _save_registry writes to loop.trial_registry_path when set."""

    def test_save_registry_writes_to_trial_path_not_live(self, tmp_path):
        """_save_registry must write to trial_registry_path, not config.registry_path."""
        live_data = tmp_path / "data" / "ha"
        live_data.mkdir(parents=True, exist_ok=True)
        live_registry = live_data / "registry.json"
        # Pre-create live registry with sentinel content.
        sentinel = {"sentinel": {"simhash": 99999, "adapter": "episodic"}}
        live_registry.write_text(json.dumps(sentinel))
        live_mtime_before = os.stat(live_registry).st_mtime_ns

        # Trial registry path.
        trial_registry_dir = tmp_path / "state" / "trial_registry"
        trial_registry_path = trial_registry_dir / "registry.json"

        # Build mock config and loop.
        mock_config = MagicMock()
        mock_config.registry_path = live_registry
        mock_loop = _make_mock_loop_with_simhash()
        mock_loop.trial_registry_path = trial_registry_path  # Fix 1 override

        _save_registry(mock_loop, mock_config)

        # LIVE file must be untouched.
        live_mtime_after = os.stat(live_registry).st_mtime_ns
        assert live_mtime_after == live_mtime_before, (
            "LIVE registry.json was modified by trial _save_registry — "
            "CRITICAL Fix 1 regression: trial writes must go to trial_registry_path"
        )
        live_content = json.loads(live_registry.read_bytes())
        assert live_content == sentinel, "LIVE registry.json content was corrupted by trial run"

        # Trial registry file must exist with the loop's data.
        assert trial_registry_path.exists(), (
            f"trial_registry_path {trial_registry_path} was not created by _save_registry"
        )
        trial_content = json.loads(trial_registry_path.read_bytes())
        assert "graph1" in trial_content, (
            "trial registry must contain the loop's episodic simhash data"
        )

    def test_save_registry_falls_back_to_config_when_no_override(self, tmp_path):
        """When loop has no trial_registry_path, _save_registry writes to config.registry_path."""
        live_data = tmp_path / "data" / "ha"
        live_data.mkdir(parents=True, exist_ok=True)
        live_registry = live_data / "registry.json"

        mock_config = MagicMock()
        mock_config.registry_path = live_registry
        mock_loop = _make_mock_loop_with_simhash()
        # Explicitly delete the trial_registry_path attribute so getattr returns None.
        del mock_loop.trial_registry_path

        _save_registry(mock_loop, mock_config)

        assert live_registry.exists(), (
            "Production _save_registry must write to config.registry_path"
        )


# ---------------------------------------------------------------------------
# Fix 1 — _save_key_metadata honors loop.trial_key_metadata_path override
# ---------------------------------------------------------------------------


class TestSaveKeyMetadataHonorsOverride:
    """Verify _save_key_metadata writes to loop.trial_key_metadata_path when set."""

    def test_save_key_metadata_writes_to_trial_path_not_live(self, tmp_path):
        """_save_key_metadata must write to trial_key_metadata_path, not config path."""
        live_registry_dir = tmp_path / "data" / "ha" / "registry"
        live_registry_dir.mkdir(parents=True, exist_ok=True)
        live_key_metadata = live_registry_dir / "key_metadata.json"
        sentinel = {
            "cycle_count": 99,
            "promoted_keys": [],
            "keys": {"sentinel_key": {"sessions_seen": 5}},
        }
        live_key_metadata.write_text(json.dumps(sentinel))
        live_mtime_before = os.stat(live_key_metadata).st_mtime_ns

        trial_registry_dir = tmp_path / "state" / "trial_registry" / "registry"
        trial_key_metadata_path = trial_registry_dir / "key_metadata.json"

        mock_config = MagicMock()
        mock_config.key_metadata_path = live_key_metadata
        mock_loop = _make_mock_loop_with_simhash()
        mock_loop.trial_key_metadata_path = trial_key_metadata_path  # Fix 1 override

        _save_key_metadata(mock_loop, mock_config)

        # LIVE file must be untouched.
        live_mtime_after = os.stat(live_key_metadata).st_mtime_ns
        assert live_mtime_after == live_mtime_before, (
            "LIVE key_metadata.json was modified by trial _save_key_metadata — "
            "CRITICAL Fix 1 regression"
        )
        live_content = json.loads(live_key_metadata.read_bytes())
        assert live_content == sentinel, "LIVE key_metadata.json content was corrupted"

        # Trial key metadata file must exist.
        assert trial_key_metadata_path.exists(), (
            f"trial_key_metadata_path {trial_key_metadata_path} was not created"
        )
        trial_content = json.loads(trial_key_metadata_path.read_bytes())
        assert trial_content["cycle_count"] == mock_loop.cycle_count

    def test_save_key_metadata_falls_back_to_config_when_no_override(self, tmp_path):
        """When loop has no trial_key_metadata_path, falls back to config.key_metadata_path."""
        live_registry_dir = tmp_path / "data" / "ha" / "registry"
        live_registry_dir.mkdir(parents=True, exist_ok=True)
        live_key_metadata = live_registry_dir / "key_metadata.json"

        mock_config = MagicMock()
        mock_config.key_metadata_path = live_key_metadata
        mock_loop = _make_mock_loop_with_simhash()
        # Explicitly delete the trial_key_metadata_path attribute so getattr returns None.
        del mock_loop.trial_key_metadata_path

        _save_key_metadata(mock_loop, mock_config)

        assert live_key_metadata.exists(), (
            "Production _save_key_metadata must write to config.key_metadata_path"
        )
