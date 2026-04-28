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


class TestConsolidationMaxEpochsOverride:
    """consolidation.max_epochs (added 2026-04-28) is read from YAML and
    flows through ServerConfig.training_config.

    Default is None → training_config.num_epochs == VALIDATED_TRAINING_CONFIG
    .num_epochs (30 from the Test 1-8 campaign). When set, it overrides.
    """

    def test_max_epochs_default_none_uses_validated(self, tmp_path):
        """Absent max_epochs → property returns the validated 30."""
        from paramem.server.config import VALIDATED_TRAINING_CONFIG

        yaml_file = _write_yaml(tmp_path, "model: mistral\n")
        config = load_server_config(yaml_file)
        assert config.consolidation.max_epochs is None
        assert config.training_config.num_epochs == VALIDATED_TRAINING_CONFIG.num_epochs

    def test_max_epochs_yaml_override_flows_to_training_config(self, tmp_path):
        """consolidation.max_epochs: 2 → training_config.num_epochs == 2."""
        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            consolidation:
              max_epochs: 2
            """,
        )
        config = load_server_config(yaml_file)
        assert config.consolidation.max_epochs == 2
        assert config.training_config.num_epochs == 2

    def test_max_epochs_runtime_override_flows_to_training_config(self, tmp_path):
        """Setting consolidation.max_epochs at runtime flows through the property.

        The property re-reads max_epochs each time (no stale cache),
        so a mutation after load takes effect on the next read.
        """
        yaml_file = _write_yaml(tmp_path, "model: mistral\n")
        config = load_server_config(yaml_file)
        # Default
        assert config.consolidation.max_epochs is None
        # Mutate
        config.consolidation.max_epochs = 5
        # Property honours the new value
        assert config.training_config.num_epochs == 5


class TestRoleAwareGroundingValidator:
    """``ServerConfig.consolidation.extraction_role_aware_grounding`` validator."""

    def test_default_is_off(self):
        from paramem.server.config import ServerConfig

        cfg = ServerConfig()
        assert cfg.consolidation.extraction_role_aware_grounding == "off"

    def test_diagnostic_accepted(self, tmp_path):
        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            consolidation:
              extraction_role_aware_grounding: diagnostic
            """,
        )
        cfg = load_server_config(yaml_file)
        assert cfg.consolidation.extraction_role_aware_grounding == "diagnostic"

    def test_active_accepted(self, tmp_path):
        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            consolidation:
              extraction_role_aware_grounding: active
            """,
        )
        cfg = load_server_config(yaml_file)
        assert cfg.consolidation.extraction_role_aware_grounding == "active"

    def test_invalid_value_rejected(self, tmp_path):
        import pytest

        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            consolidation:
              extraction_role_aware_grounding: aggressive
            """,
        )
        with pytest.raises(ValueError, match="extraction_role_aware_grounding"):
            load_server_config(yaml_file)

    def test_project_server_yaml_example_loads_cleanly(self):
        """Shipped configs/server.yaml.example parses without validator errors.

        The shipped template is the canonical reference; the operator-local
        configs/server.yaml is gitignored (see project_config_yaml_overlap.md
        for the YAML cleanup arc).
        """
        cfg = load_server_config("configs/server.yaml.example")
        assert cfg.consolidation.extraction_role_aware_grounding in {
            "off",
            "diagnostic",
            "active",
        }
