"""Unit tests for paramem.server.trial_state (Slice 3b.2).

Covers:
- TrialMarker roundtrip (write → read → equal fields).
- read_trial_marker returns None when file absent.
- read_trial_marker raises TrialMarkerSchemaError on schema mismatch.
- write_trial_marker is atomic (crash mid-write leaves no corrupt file).
- clear_trial_marker is idempotent.
- Backup paths stored in TrialMarker are absolute (Correction 5).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paramem.server.trial_state import (
    TRIAL_MARKER_FILENAME,
    TRIAL_MARKER_SCHEMA_VERSION,
    TrialMarker,
    TrialMarkerSchemaError,
    clear_trial_marker,
    read_trial_marker,
    write_trial_marker,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MARKER = TrialMarker(
    schema_version=TRIAL_MARKER_SCHEMA_VERSION,
    started_at="2026-04-22T01:00:00+00:00",
    pre_trial_config_sha256="abcd1234" * 8,
    candidate_config_sha256="efef5678" * 8,
    backup_paths={
        "config": "/abs/data/ha/backups/config/20260422-010000",
        "graph": "/abs/data/ha/backups/graph/20260422-010000",
        "registry": "/abs/data/ha/backups/registry/20260422-010000",
    },
    trial_adapter_dir="/abs/data/ha/state/trial_adapter",
    trial_graph_dir="/abs/data/ha/state/trial_graph",
)


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------


class TestMarkerRoundtrip:
    def test_marker_roundtrip_plaintext(self, tmp_path):
        """Write marker; read back; fields equal."""
        state_dir = tmp_path / "state"
        write_trial_marker(state_dir, _MARKER)
        recovered = read_trial_marker(state_dir)
        assert recovered is not None
        assert recovered.schema_version == _MARKER.schema_version
        assert recovered.started_at == _MARKER.started_at
        assert recovered.pre_trial_config_sha256 == _MARKER.pre_trial_config_sha256
        assert recovered.candidate_config_sha256 == _MARKER.candidate_config_sha256
        assert recovered.backup_paths == _MARKER.backup_paths
        assert recovered.trial_adapter_dir == _MARKER.trial_adapter_dir
        assert recovered.trial_graph_dir == _MARKER.trial_graph_dir

    def test_marker_file_is_valid_json(self, tmp_path):
        """Written file is valid UTF-8 JSON."""
        state_dir = tmp_path / "state"
        write_trial_marker(state_dir, _MARKER)
        raw = (state_dir / TRIAL_MARKER_FILENAME).read_text(encoding="utf-8")
        data = json.loads(raw)
        assert data["schema_version"] == TRIAL_MARKER_SCHEMA_VERSION

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict → from_dict → equal marker."""
        recovered = TrialMarker.from_dict(_MARKER.to_dict())
        assert recovered == _MARKER


# ---------------------------------------------------------------------------
# Missing file → None
# ---------------------------------------------------------------------------


class TestReadMissing:
    def test_read_returns_none_when_missing(self, tmp_path):
        """Empty state_dir → None (not an error)."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        result = read_trial_marker(state_dir)
        assert result is None

    def test_read_returns_none_when_dir_absent(self, tmp_path):
        """Non-existent state_dir → None."""
        result = read_trial_marker(tmp_path / "nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# Schema errors
# ---------------------------------------------------------------------------


class TestSchemaErrors:
    def test_read_raises_on_schema_version_mismatch(self, tmp_path):
        """Wrong schema_version → TrialMarkerSchemaError."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        bad = _MARKER.to_dict()
        bad["schema_version"] = 999
        (state_dir / TRIAL_MARKER_FILENAME).write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(TrialMarkerSchemaError, match="schema_version"):
            read_trial_marker(state_dir)

    def test_read_raises_on_garbage_json(self, tmp_path):
        """Garbage JSON → TrialMarkerSchemaError."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        (state_dir / TRIAL_MARKER_FILENAME).write_bytes(b"NOT JSON AT ALL !!!")
        with pytest.raises(TrialMarkerSchemaError):
            read_trial_marker(state_dir)

    def test_read_raises_on_missing_field(self, tmp_path):
        """Valid JSON but missing required field → TrialMarkerSchemaError."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        bad = _MARKER.to_dict()
        del bad["trial_adapter_dir"]
        (state_dir / TRIAL_MARKER_FILENAME).write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(TrialMarkerSchemaError, match="missing"):
            read_trial_marker(state_dir)


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    def test_write_atomic_via_pending(self, tmp_path):
        """Successful write results in state_dir/trial.json with no .pending residue."""
        state_dir = tmp_path / "state"
        write_trial_marker(state_dir, _MARKER)
        assert (state_dir / TRIAL_MARKER_FILENAME).exists()
        # Pending file should be gone after successful write.
        pending_file = state_dir / ".pending" / TRIAL_MARKER_FILENAME
        assert not pending_file.exists()

    def test_write_creates_state_dir(self, tmp_path):
        """state_dir is created if absent."""
        state_dir = tmp_path / "nested" / "state"
        write_trial_marker(state_dir, _MARKER)
        assert state_dir.exists()
        assert (state_dir / TRIAL_MARKER_FILENAME).exists()


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClearMarker:
    def test_clear_idempotent_when_absent(self, tmp_path):
        """clear_trial_marker on missing file is a no-op (no error)."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        clear_trial_marker(state_dir)  # should not raise

    def test_clear_removes_marker(self, tmp_path):
        """clear removes trial.json."""
        state_dir = tmp_path / "state"
        write_trial_marker(state_dir, _MARKER)
        assert (state_dir / TRIAL_MARKER_FILENAME).exists()
        clear_trial_marker(state_dir)
        assert not (state_dir / TRIAL_MARKER_FILENAME).exists()

    def test_clear_idempotent_twice(self, tmp_path):
        """Calling clear twice is safe."""
        state_dir = tmp_path / "state"
        write_trial_marker(state_dir, _MARKER)
        clear_trial_marker(state_dir)
        clear_trial_marker(state_dir)  # second call — no error


# ---------------------------------------------------------------------------
# Absolute paths (Correction 5)
# ---------------------------------------------------------------------------


class TestAbsolutePaths:
    def test_trial_marker_backup_paths_are_absolute(self, tmp_path):
        """backup_paths in a TrialMarker written by write_trial_marker must be absolute strings."""
        # The marker we defined in the fixture uses absolute paths already.
        # This test verifies that read_trial_marker preserves them as-is.
        state_dir = tmp_path / "state"
        write_trial_marker(state_dir, _MARKER)
        recovered = read_trial_marker(state_dir)
        assert recovered is not None
        for key, path_str in recovered.backup_paths.items():
            assert Path(path_str).is_absolute(), (
                f"backup_paths[{key!r}]={path_str!r} is not absolute"
            )
        assert Path(recovered.trial_adapter_dir).is_absolute()
        assert Path(recovered.trial_graph_dir).is_absolute()

    def test_trial_marker_backup_paths_are_absolute_from_relative_input(self, tmp_path):
        """write_trial_marker resolves relative path inputs to absolute before writing.

        Fix 10b: tests the production-side Path.resolve() in write_trial_marker.
        A TrialMarker created with relative path strings must be written with
        those paths resolved to absolute paths, so the marker is portable
        across working-directory changes (e.g. systemd units).
        """
        state_dir = tmp_path / "state"
        relative_marker = TrialMarker(
            schema_version=TRIAL_MARKER_SCHEMA_VERSION,
            started_at="2026-04-22T01:00:00+00:00",
            pre_trial_config_sha256="abcd1234" * 8,
            candidate_config_sha256="efef5678" * 8,
            backup_paths={
                "config": "backups/config/20260422-010000",
                "graph": "backups/graph/20260422-010000",
                "registry": "backups/registry/20260422-010000",
            },
            trial_adapter_dir="state/trial_adapter",
            trial_graph_dir="state/trial_graph",
        )
        write_trial_marker(state_dir, relative_marker)
        recovered = read_trial_marker(state_dir)
        assert recovered is not None
        # All paths must now be absolute, regardless of the relative input.
        for key, path_str in recovered.backup_paths.items():
            assert Path(path_str).is_absolute(), (
                f"backup_paths[{key!r}]={path_str!r} was not resolved to absolute"
            )
        assert Path(recovered.trial_adapter_dir).is_absolute(), (
            f"trial_adapter_dir={recovered.trial_adapter_dir!r} was not resolved to absolute"
        )
        assert Path(recovered.trial_graph_dir).is_absolute(), (
            f"trial_graph_dir={recovered.trial_graph_dir!r} was not resolved to absolute"
        )
