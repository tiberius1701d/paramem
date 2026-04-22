"""Tests for Slice 6a ServerBackupsConfig additions in paramem.server.config.

Covers:
- RetentionTierConfig, RetentionConfig defaults and YAML loading.
- schedule / artifacts / max_total_disk_gb fields.
- Invalid keep strings and invalid artifact names raise ValueError.
- Existing orphan_sweep tests still pass alongside new fields.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from paramem.server.config import load_server_config


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "server.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


class TestSecurityRetentionConfig:
    def test_security_retention_defaults_when_absent(self, tmp_path):
        """No retention key → all defaults (daily=7, manual=unlimited+5GB, etc.)."""
        yaml_file = _write_yaml(tmp_path, "model: mistral\n")
        config = load_server_config(yaml_file)
        ret = config.security.backups.retention
        assert ret.daily.keep == 7
        assert ret.weekly.keep == 4
        assert ret.monthly.keep == 12
        assert ret.yearly.keep == 3
        assert ret.pre_migration.keep == 10
        assert ret.trial_adapter.keep == 5
        assert ret.manual.keep == "unlimited"
        assert ret.manual.max_disk_gb == 5.0

    def test_security_retention_loaded_from_yaml(self, tmp_path):
        """Full retention block → all 7 tiers with correct keep + max_disk_gb."""
        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            security:
              backups:
                retention:
                  daily:         { keep: 14 }
                  weekly:        { keep: 8 }
                  monthly:       { keep: 6 }
                  yearly:        { keep: 2 }
                  pre_migration: { keep: 3 }
                  trial_adapter: { keep: 2 }
                  manual:        { keep: "unlimited", max_disk_gb: 10 }
            """,
        )
        config = load_server_config(yaml_file)
        ret = config.security.backups.retention
        assert ret.daily.keep == 14
        assert ret.weekly.keep == 8
        assert ret.monthly.keep == 6
        assert ret.yearly.keep == 2
        assert ret.pre_migration.keep == 3
        assert ret.trial_adapter.keep == 2
        assert ret.manual.keep == "unlimited"
        assert ret.manual.max_disk_gb == 10.0

    def test_security_retention_partial_override(self, tmp_path):
        """Only daily.keep: 14 overridden → other tiers stay at defaults."""
        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            security:
              backups:
                retention:
                  daily: { keep: 14 }
            """,
        )
        config = load_server_config(yaml_file)
        ret = config.security.backups.retention
        assert ret.daily.keep == 14
        # Other tiers unchanged.
        assert ret.weekly.keep == 4
        assert ret.manual.keep == "unlimited"

    def test_security_retention_keep_unlimited_string(self, tmp_path):
        """keep: "unlimited" parsed verbatim (string, not int)."""
        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            security:
              backups:
                retention:
                  daily: { keep: "unlimited" }
            """,
        )
        config = load_server_config(yaml_file)
        assert config.security.backups.retention.daily.keep == "unlimited"

    def test_security_retention_keep_int_coercion(self, tmp_path):
        """keep: 5 → int 5 (not string '5')."""
        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            security:
              backups:
                retention:
                  daily: { keep: 5 }
            """,
        )
        config = load_server_config(yaml_file)
        assert config.security.backups.retention.daily.keep == 5
        assert isinstance(config.security.backups.retention.daily.keep, int)

    def test_security_retention_keep_invalid_raises(self, tmp_path):
        """keep: 'abc' → ValueError mentioning the field path."""
        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            security:
              backups:
                retention:
                  daily: { keep: "abc" }
            """,
        )
        with pytest.raises(ValueError, match="daily"):
            load_server_config(yaml_file)


class TestSecurityScheduleConfig:
    def test_security_schedule_default_is_daily_0400(self, tmp_path):
        """Absent schedule → 'daily 04:00'."""
        yaml_file = _write_yaml(tmp_path, "model: mistral\n")
        config = load_server_config(yaml_file)
        assert config.security.backups.schedule == "daily 04:00"

    def test_security_schedule_off(self, tmp_path):
        """schedule: 'off' → 'off'."""
        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            security:
              backups:
                schedule: "off"
            """,
        )
        config = load_server_config(yaml_file)
        assert config.security.backups.schedule == "off"

    def test_security_schedule_every_12h(self, tmp_path):
        """schedule: 'every 12h' → 'every 12h'."""
        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            security:
              backups:
                schedule: "every 12h"
            """,
        )
        config = load_server_config(yaml_file)
        assert config.security.backups.schedule == "every 12h"


class TestSecurityArtifactsConfig:
    def test_security_artifacts_default(self, tmp_path):
        """Absent artifacts → ['config', 'graph', 'registry']."""
        yaml_file = _write_yaml(tmp_path, "model: mistral\n")
        config = load_server_config(yaml_file)
        assert config.security.backups.artifacts == ["config", "graph", "registry"]

    def test_security_artifacts_subset(self, tmp_path):
        """artifacts: [config] → ['config']."""
        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            security:
              backups:
                artifacts: [config]
            """,
        )
        config = load_server_config(yaml_file)
        assert config.security.backups.artifacts == ["config"]

    def test_security_artifacts_invalid_raises(self, tmp_path):
        """artifacts: [foo] → ValueError."""
        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            security:
              backups:
                artifacts: [foo]
            """,
        )
        with pytest.raises(ValueError, match="foo"):
            load_server_config(yaml_file)


class TestSecurityMaxTotalDiskConfig:
    def test_security_max_total_disk_default(self, tmp_path):
        """Absent max_total_disk_gb → 20.0."""
        yaml_file = _write_yaml(tmp_path, "model: mistral\n")
        config = load_server_config(yaml_file)
        assert config.security.backups.max_total_disk_gb == 20.0

    def test_security_max_total_disk_from_yaml(self, tmp_path):
        """max_total_disk_gb: 50 → 50.0."""
        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            security:
              backups:
                max_total_disk_gb: 50
            """,
        )
        config = load_server_config(yaml_file)
        assert config.security.backups.max_total_disk_gb == 50.0


class TestSecurityOrphanSweepStillWorks:
    """Existing Slice 3b.2 test — orphan_sweep coexists with new fields."""

    def test_security_orphan_sweep_still_works(self, tmp_path):
        """orphan_sweep.max_age_hours: 48 still loads correctly alongside 6a fields."""
        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            security:
              backups:
                orphan_sweep:
                  max_age_hours: 48
                retention:
                  daily: { keep: 14 }
                schedule: "off"
                max_total_disk_gb: 30
            """,
        )
        config = load_server_config(yaml_file)
        assert config.security.backups.orphan_sweep.max_age_hours == 48
        assert config.security.backups.retention.daily.keep == 14
        assert config.security.backups.schedule == "off"
        assert config.security.backups.max_total_disk_gb == 30.0
