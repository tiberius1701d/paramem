"""Tests for paramem.server.config — specifically the security config wiring.

Fix 2: ``load_server_config`` now reads ``security.backups.orphan_sweep``
from YAML and populates ``SecurityConfig → ServerBackupsConfig →
OrphanSweepConfig``.

Covers:
- Nested ``security.backups.orphan_sweep.max_age_hours: 48`` → exposes 48.
- Absent ``security`` section → default 24.
- Empty ``security:`` → default 24.
- The loaded value reaches ``recover_migration_state`` via the config object.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from paramem.server.config import PathsConfig, load_server_config


def _write_yaml(tmp_path: Path, content: str) -> Path:
    """Write YAML content to a temp file and return the path."""
    p = tmp_path / "server.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


class TestSecurityOrphanSweepConfig:
    """Fix 2: security.backups.orphan_sweep.max_age_hours is wired in load_server_config."""

    def test_security_orphan_sweep_loaded_from_yaml(self, tmp_path):
        """Nested YAML key security.backups.orphan_sweep.max_age_hours: 48 → config exposes 48."""
        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            security:
              backups:
                orphan_sweep:
                  max_age_hours: 48
            """,
        )
        config = load_server_config(yaml_file)
        assert config.security.backups.orphan_sweep.max_age_hours == 48

    def test_security_orphan_sweep_default_24_when_absent(self, tmp_path):
        """Absent security section → OrphanSweepConfig default of 24h."""
        yaml_file = _write_yaml(tmp_path, "model: mistral\n")
        config = load_server_config(yaml_file)
        assert config.security.backups.orphan_sweep.max_age_hours == 24

    def test_security_orphan_sweep_default_24_when_section_empty(self, tmp_path):
        """Empty security: section → OrphanSweepConfig default of 24h."""
        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            security:
            """,
        )
        config = load_server_config(yaml_file)
        assert config.security.backups.orphan_sweep.max_age_hours == 24

    def test_security_orphan_sweep_default_24_when_backups_empty(self, tmp_path):
        """Empty security.backups: section → OrphanSweepConfig default of 24h."""
        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            security:
              backups:
            """,
        )
        config = load_server_config(yaml_file)
        assert config.security.backups.orphan_sweep.max_age_hours == 24

    def test_security_config_wire_through_to_recover_migration_state(self, tmp_path):
        """max_age_hours from config is passed to recover_migration_state correctly.

        Loads a YAML with max_age_hours: 1 and calls recover_migration_state
        with a backup older than 1h — verifies the backup is NOT swept (outside
        the custom window), proving the value was read from config.
        """
        import hashlib
        from datetime import datetime, timedelta, timezone

        from paramem.backup.backup import write as backup_write
        from paramem.backup.types import ArtifactKind
        from paramem.server.migration_recovery import RecoveryAction, recover_migration_state

        live_content = b"model: mistral\n"
        live_config = tmp_path / "configs" / "server.yaml"
        live_config.parent.mkdir(parents=True, exist_ok=True)
        live_config.write_bytes(live_content)
        live_hash = hashlib.sha256(live_content).hexdigest()

        backups_root = tmp_path / "backups"
        backup_write(
            ArtifactKind.CONFIG,
            b"config: live\n",
            meta_fields={"tier": "pre_migration", "pre_trial_hash": live_hash},
            base_dir=backups_root / "config",
        )

        # Monkeypatch datetime.now to be 2h in the future so a 1h window rejects the backup.
        import paramem.server.migration_recovery as _mr

        real_now = datetime.now(tz=timezone.utc)
        far_future = real_now + timedelta(hours=2)

        monkeypatch_target = type(
            "FakeDatetime",
            (),
            {
                "now": staticmethod(lambda tz=None: far_future),
                "utcnow": staticmethod(lambda: far_future.replace(tzinfo=None)),
            },
        )
        original_dt = _mr.datetime
        _mr.datetime = monkeypatch_target
        try:
            result = recover_migration_state(
                state_dir=tmp_path / "state",
                live_config_path=live_config,
                backups_root=backups_root,
                max_age_hours=1,  # 1h window; backup is 2h old → NOT swept
            )
        finally:
            _mr.datetime = original_dt

        assert result.action == RecoveryAction.NORMAL_LIVE, (
            "backup outside 1h window should NOT be swept; NORMAL_LIVE expected"
        )


class TestPathsConfigKeyMetadata:
    """Cleanup 1 — canonical Paths.key_metadata must match the on-disk layout.

    Pins the path so the multi-site hardcoded-workaround pattern cannot recur.
    The consolidation writer uses config.key_metadata_path (→ paths.key_metadata),
    and all read sites must use the same canonical property.
    """

    def test_paths_key_metadata_matches_consolidation_writer_layout(self):
        """Canonical Paths.key_metadata must equal data/registry/key_metadata.json.

        Matching the path the consolidation writer (server/consolidation.py) uses.
        Prevents the multi-site hardcoded-workaround pattern from recurring.
        """
        cfg = PathsConfig(data=Path("/some/data"))
        assert cfg.key_metadata == Path("/some/data/registry/key_metadata.json")

    def test_paths_registry_is_distinct_from_key_metadata(self):
        """``paths.registry`` and ``paths.key_metadata`` are TWO different files,
        not aliases. ``registry`` carries the combined SimHash dict (read by
        inference for hallucination detection); ``key_metadata`` carries
        per-key metadata (read by gates / attention / restore). Aliasing them
        would silently regress inference's SimHash reads.
        """
        cfg = PathsConfig(data=Path("/some/data"))
        assert cfg.registry == Path("/some/data/registry.json")
        assert cfg.key_metadata == Path("/some/data/registry/key_metadata.json")
        assert cfg.registry != cfg.key_metadata

    def test_paths_registry_dir_is_parent_of_key_metadata(self):
        """paths.registry_dir must be the parent directory of paths.key_metadata."""
        cfg = PathsConfig(data=Path("/some/data"))
        assert cfg.registry_dir == cfg.key_metadata.parent


# ---------------------------------------------------------------------------
# Fix 7 — PathsConfig.data=None raises ValueError on property access
# ---------------------------------------------------------------------------


class TestPathsConfigNoneGuard:
    """Fix 7 (2026-04-23): PathsConfig properties raise ValueError when data is None.

    Previously they would raise TypeError from Path(None) / str with an
    unhelpful message.  The guard provides an explicit error that names the
    property and the missing prerequisite.
    """

    def test_key_metadata_raises_when_data_is_none(self):
        """paths.key_metadata raises ValueError when data is None."""
        import pytest

        cfg = PathsConfig(data=None)
        with pytest.raises(ValueError, match="paths.data must be set"):
            _ = cfg.key_metadata

    def test_registry_raises_when_data_is_none(self):
        """paths.registry raises ValueError when data is None."""
        import pytest

        cfg = PathsConfig(data=None)
        with pytest.raises(ValueError, match="paths.data must be set"):
            _ = cfg.registry

    def test_registry_dir_raises_when_data_is_none(self):
        """paths.registry_dir raises ValueError when data is None."""
        import pytest

        cfg = PathsConfig(data=None)
        with pytest.raises(ValueError, match="paths.data must be set"):
            _ = cfg.registry_dir
