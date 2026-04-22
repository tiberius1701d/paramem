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

from paramem.server.config import load_server_config


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
        from paramem.backup.encryption import SecurityBackupsConfig
        from paramem.backup.types import ArtifactKind
        from paramem.server.migration_recovery import RecoveryAction, recover_migration_state

        live_content = b"model: mistral\n"
        live_config = tmp_path / "configs" / "server.yaml"
        live_config.parent.mkdir(parents=True, exist_ok=True)
        live_config.write_bytes(live_content)
        live_hash = hashlib.sha256(live_content).hexdigest()

        backups_root = tmp_path / "backups"
        sec = SecurityBackupsConfig()
        backup_write(
            ArtifactKind.CONFIG,
            b"config: live\n",
            meta_fields={"tier": "pre_migration", "pre_trial_hash": live_hash},
            base_dir=backups_root / "config",
            security_config=sec,
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
